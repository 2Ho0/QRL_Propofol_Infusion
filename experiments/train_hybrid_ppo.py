"""
Hybrid PPO Training: Offline Pre-training → Online Fine-tuning
================================================================

Hybrid training pipeline for PPO agents:
1. Stage 1: Offline behavioral cloning on VitalDB data
2. Stage 2: Online PPO fine-tuning on simulator
3. Stage 3: Evaluation on both datasets

Note: PPO is inherently on-policy, so offline pre-training uses
behavioral cloning (supervised learning) to warm-start the policy.

Usage:
------
# Full hybrid PPO training (quantum)
python experiments/train_hybrid_ppo.py --algorithm quantum_ppo --n_cases 100 --offline_epochs 50 --online_episodes 500

# Classical PPO
python experiments/train_hybrid_ppo.py --algorithm classical_ppo --n_cases 100 --offline_epochs 50 --online_episodes 500

# With LSTM encoder
python experiments/train_hybrid_ppo.py --algorithm quantum_ppo --encoder lstm --n_cases 100 --offline_epochs 50 --online_episodes 500

# Skip offline (resume from checkpoint)
python experiments/train_hybrid_ppo.py --skip_offline --resume logs/hybrid_ppo_20231221/checkpoints/offline_best.pt --online_episodes 500
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml
from datetime import datetime
from typing import Dict, Tuple
import matplotlib.pyplot as plt

from agents.quantum_ppo_agent import QuantumPPOAgent
from agents.classical_ppo_agent import ClassicalPPOAgent
from environment.propofol_env import PropofolEnv
from environment.patient_simulator import create_patient_population
from data.vitaldb_loader import VitalDBLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hybrid PPO Training")
    
    # Stage control
    parser.add_argument('--skip_offline', action='store_true',
                        help='Skip offline pre-training')
    parser.add_argument('--skip_online', action='store_true',
                        help='Skip online fine-tuning')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    # Algorithm
    parser.add_argument('--algorithm', type=str, default='quantum_ppo',
                       choices=['quantum_ppo', 'classical_ppo'],
                       help='PPO algorithm type')
    parser.add_argument('--encoder', type=str, default='none',
                       choices=['none', 'lstm', 'transformer'],
                       help='Temporal encoder type')
    
    # Data
    parser.add_argument('--n_cases', type=int, default=20,
                        help='Number of VitalDB cases')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Pre-saved VitalDB data path')
    
    # Stage 1: Offline (Behavioral Cloning)
    parser.add_argument('--offline_epochs', type=int, default=50,
                        help='BC epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr_offline', type=float, default=1e-3,
                        help='Learning rate for offline training')
    
    # Stage 2: Online (PPO)
    parser.add_argument('--online_episodes', type=int, default=500,
                        help='Online episodes')
    parser.add_argument('--warmup_episodes', type=int, default=50,
                        help='Warmup episodes')
    parser.add_argument('--update_interval', type=int, default=2048,
                        help='Steps between PPO updates')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='PPO epochs per update')
    
    # Agent
    parser.add_argument('--state_dim', type=int, default=8)
    parser.add_argument('--action_dim', type=int, default=1)
    parser.add_argument('--n_qubits', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=4)
    
    # PPO hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--lr_actor', type=float, default=3e-4)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    
    # Misc
    parser.add_argument('--config', type=str, default='config/hyperparameters.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--device', type=str, default='auto')
    
    return parser.parse_args()


def setup_directories(log_dir: str, algorithm: str) -> Dict[str, Path]:
    """Setup directory structure."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path(log_dir) / f'hybrid_{algorithm}_{timestamp}'
    
    dirs = {
        'base': base_dir,
        'checkpoints': base_dir / 'checkpoints',
        'figures': base_dir / 'figures',
        'stage1_offline': base_dir / 'stage1_offline',
        'stage2_online': base_dir / 'stage2_online',
        'stage3_test': base_dir / 'stage3_test'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def load_vitaldb_data(n_cases: int, data_path: str = None):
    """Load and split VitalDB data."""
    print("\n" + "="*70)
    print("LOADING VITALDB DATA")
    print("="*70)
    
    if data_path and Path(data_path).exists():
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded from {data_path}")
    else:
        loader = VitalDBLoader(bis_range=(30, 70))
        data = loader.prepare_training_data(n_cases=n_cases)
        print(f"✓ Loaded {n_cases} cases")
    
    # Split 80/10/10
    n_total = len(data['states'])
    train_end = int(n_total * 0.8)
    val_end = int(n_total * 0.9)
    
    train_data = {
        'states': data['states'][:train_end],
        'actions': data['actions'][:train_end]
    }
    
    val_data = {
        'states': data['states'][train_end:val_end],
        'actions': data['actions'][train_end:val_end]
    }
    
    test_data = {
        'states': data['states'][val_end:],
        'actions': data['actions'][val_end:]
    }
    
    print(f"Train: {len(train_data['states']):,} transitions")
    print(f"Val: {len(val_data['states']):,} transitions")
    print(f"Test: {len(test_data['states']):,} transitions")
    
    return train_data, val_data, test_data


def stage1_offline_behavioral_cloning(
    agent, train_data, val_data, args, dirs, device
):
    """
    Stage 1: Offline Behavioral Cloning.
    
    Train policy to imitate expert actions using supervised learning.
    This provides a warm start for online PPO training.
    """
    print("\n" + "="*70)
    print("STAGE 1: OFFLINE BEHAVIORAL CLONING")
    print("="*70)
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_data['states']),
        torch.FloatTensor(train_data['actions'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_data['states']),
        torch.FloatTensor(val_data['actions'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # BC optimizer
    bc_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=args.lr_offline)
    criterion = nn.MSELoss()
    
    # Training log
    bc_log = {
        'epochs': [],
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    print(f"Training for {args.offline_epochs} epochs...")
    for epoch in tqdm(range(args.offline_epochs), desc="BC Training"):
        # Train
        agent.train()
        train_losses = []
        
        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            
            # Forward pass
            if agent.encoder is not None:
                # Expand dims for temporal encoder
                states_seq = states.unsqueeze(1)  # [B, 1, state_dim]
                encoded = agent.encoder(states_seq)
                encoded = encoded[:, -1, :]  # Take last timestep
                pred_actions = agent.actor.get_mean(encoded)
            else:
                pred_actions = agent.actor.get_mean(states)
            
            # BC loss
            loss = criterion(pred_actions, actions)
            
            # Backward
            bc_optimizer.zero_grad()
            loss.backward()
            bc_optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        agent.eval()
        val_losses = []
        
        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                
                if agent.encoder is not None:
                    states_seq = states.unsqueeze(1)
                    encoded = agent.encoder(states_seq)
                    encoded = encoded[:, -1, :]
                    pred_actions = agent.actor.get_mean(encoded)
                else:
                    pred_actions = agent.actor.get_mean(states)
                
                loss = criterion(pred_actions, actions)
                val_losses.append(loss.item())
        
        # Log
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        bc_log['epochs'].append(epoch)
        bc_log['train_loss'].append(train_loss)
        bc_log['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.offline_epochs} - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'agent_state_dict': agent.state_dict(),
                'val_loss': val_loss
            }, dirs['checkpoints'] / 'offline_best.pt')
    
    # Save final
    torch.save({
        'epoch': args.offline_epochs,
        'agent_state_dict': agent.state_dict(),
        'bc_log': bc_log
    }, dirs['checkpoints'] / 'offline_final.pt')
    
    # Save log
    with open(dirs['stage1_offline'] / 'bc_log.pkl', 'wb') as f:
        pickle.dump(bc_log, f)
    
    # Plot
    plot_bc_curves(bc_log, dirs['stage1_offline'])
    
    print(f"\n✓ Offline BC complete. Best val loss: {best_val_loss:.6f}")
    return bc_log


def stage2_online_ppo(agent, args, dirs, device):
    """
    Stage 2: Online PPO Fine-tuning.
    
    Continue training with PPO on simulator using the BC warm-started policy.
    """
    print("\n" + "="*70)
    print("STAGE 2: ONLINE PPO FINE-TUNING")
    print("="*70)
    
    # Create environment
    env = PropofolEnv(config_path=args.config, seed=args.seed)
    
    # Training log
    ppo_log = {
        'episodes': [],
        'rewards': [],
        'mdapes': [],
        'time_in_target': [],
        'policy_losses': [],
        'value_losses': []
    }
    
    best_mdape = float('inf')
    
    print(f"Training for {args.online_episodes} episodes...")
    for episode in tqdm(range(args.online_episodes), desc="PPO Training"):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_steps = 0
        
        while not done and not truncated:
            # Select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, log_prob, value = agent.select_action(state_tensor)
                action = action.cpu().numpy()
                log_prob = log_prob.cpu().item()
                value = value.cpu().item()
            
            # Step
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store
            agent.buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value
            )
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Update
            if agent.buffer.is_full():
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    _, _, last_value = agent.select_action(state_tensor)
                    last_value = last_value.cpu().item()
                
                agent.buffer.compute_advantages(last_value)
                policy_loss, value_loss, entropy = agent.update()
                
                ppo_log['policy_losses'].append(policy_loss)
                ppo_log['value_losses'].append(value_loss)
                
                agent.buffer.clear()
        
        # Log episode
        metrics = env.get_episode_metrics()
        ppo_log['episodes'].append(episode)
        ppo_log['rewards'].append(episode_reward)
        ppo_log['mdapes'].append(metrics.get('mdape', 0))
        ppo_log['time_in_target'].append(metrics.get('time_in_target', 0))
        
        # Evaluate
        if (episode + 1) % args.eval_interval == 0:
            eval_mdape = evaluate_agent(agent, env, 10, device)
            
            print(f"\nEpisode {episode+1}/{args.online_episodes}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  MDAPE: {metrics.get('mdape', 0):.2f}%")
            print(f"  Eval MDAPE: {eval_mdape:.2f}%")
            
            if eval_mdape < best_mdape:
                best_mdape = eval_mdape
                torch.save({
                    'episode': episode,
                    'agent_state_dict': agent.state_dict(),
                    'mdape': eval_mdape
                }, dirs['checkpoints'] / 'online_best.pt')
                print(f"  ✓ New best model (MDAPE: {eval_mdape:.2f}%)")
    
    # Save final
    torch.save({
        'episode': args.online_episodes,
        'agent_state_dict': agent.state_dict(),
        'ppo_log': ppo_log
    }, dirs['checkpoints'] / 'online_final.pt')
    
    with open(dirs['stage2_online'] / 'ppo_log.pkl', 'wb') as f:
        pickle.dump(ppo_log, f)
    
    plot_ppo_curves(ppo_log, dirs['stage2_online'])
    
    print(f"\n✓ Online PPO complete. Best MDAPE: {best_mdape:.2f}%")
    return ppo_log


def evaluate_agent(agent, env, n_episodes, device):
    """Evaluate agent."""
    agent.eval()
    mdapes = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        
        while not done and not truncated:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, _, _ = agent.select_action(state_tensor, deterministic=True)
                action = action.cpu().numpy()
            
            state, _, done, truncated, _ = env.step(action)
        
        metrics = env.get_episode_metrics()
        mdapes.append(metrics.get('mdape', 0))
    
    agent.train()
    return np.mean(mdapes)


def plot_bc_curves(log, save_dir):
    """Plot BC training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(log['epochs'], log['train_loss'], label='Train')
    plt.plot(log['epochs'], log['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Behavioral Cloning Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'bc_loss.png', dpi=300)
    plt.close()


def plot_ppo_curves(log, save_dir):
    """Plot PPO training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(log['episodes'], log['rewards'])
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(log['episodes'], log['mdapes'])
    axes[0, 1].set_title('MDAPE')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(log['episodes'], log['time_in_target'])
    axes[1, 0].set_title('Time in Target')
    axes[1, 0].grid(True)
    
    if log['policy_losses']:
        axes[1, 1].plot(log['policy_losses'])
        axes[1, 1].set_title('Policy Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'ppo_curves.png', dpi=300)
    plt.close()


def main():
    args = parse_args()
    
    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config.update({
        'device': str(device),
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'clip_epsilon': args.clip_epsilon,
        'entropy_coef': args.entropy_coef,
        'vf_coef': args.vf_coef,
        'lr_actor': args.lr_actor,
        'lr_critic': args.lr_critic,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'update_interval': args.update_interval
    })
    
    # Directories
    dirs = setup_directories(args.log_dir, args.algorithm)
    
    # Save args
    with open(dirs['base'] / 'args.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    print(f"\n{'='*70}")
    print(f"HYBRID {args.algorithm.upper()} TRAINING")
    print(f"{'='*70}")
    print(f"Log directory: {dirs['base']}")
    
    # Create agent
    if args.algorithm == 'quantum_ppo':
        agent = QuantumPPOAgent(
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            config=config,
            encoder_type=args.encoder,
            seed=args.seed
        )
    else:
        agent = ClassicalPPOAgent(
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            config=config,
            encoder_type=args.encoder,
            seed=args.seed
        )
    
    agent.to(device)
    
    # Stage 1: Offline BC
    if not args.skip_offline:
        train_data, val_data, test_data = load_vitaldb_data(args.n_cases, args.data_path)
        stage1_offline_behavioral_cloning(agent, train_data, val_data, args, dirs, device)
    elif args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        print(f"✓ Resumed from {args.resume}")
    
    # Stage 2: Online PPO
    if not args.skip_online:
        stage2_online_ppo(agent, args, dirs, device)
    
    print(f"\n{'='*70}")
    print("HYBRID TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Logs saved to: {dirs['base']}")


if __name__ == "__main__":
    main()
