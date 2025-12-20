"""
Offline Training with VitalDB Dataset
======================================

Train Quantum DDPG agent using offline data from VitalDB.

This implements offline RL (batch RL) where the agent learns
from pre-collected data without interacting with the environment.

Usage:
    python experiments/train_offline.py --data_path ./data/offline_dataset/vitaldb_offline_data.pkl --n_episodes 1000
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from datetime import datetime

from agents.quantum_agent import QuantumDDPGAgent
from data.vitaldb_loader import VitalDBDataset
# from utils.metrics import calculate_mdape, calculate_mdpe, calculate_wobble
# from utils.visualization import plot_training_curves


def train_offline(
    agent: QuantumDDPGAgent,
    dataloader: DataLoader,
    n_epochs: int = 100,
    log_dir: str = './logs',
    save_freq: int = 10
):
    """
    Train agent using offline data.
    
    Args:
        agent: Quantum DDPG agent
        dataloader: DataLoader for offline dataset
        n_epochs: Number of training epochs
        log_dir: Directory for logs and checkpoints
        save_freq: Checkpoint saving frequency
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = log_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Starting Offline Training")
    print("=" * 70)
    print(f"  Log directory: {log_dir}")
    print(f"  Training epochs: {n_epochs}")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Dataset size: {len(dataloader.dataset):,}")
    print("=" * 70)
    
    # Training metrics
    epoch_losses = []
    epoch_rewards = []
    
    best_loss = float('inf')
    
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_reward = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for batch in pbar:
            states, actions, rewards, next_states, dones = batch
            
            # Convert to numpy for agent
            states_np = states.numpy()
            actions_np = actions.numpy()
            rewards_np = rewards.numpy()
            next_states_np = next_states.numpy()
            dones_np = dones.numpy()
            
            # Train on each transition in batch
            batch_loss = []
            for i in range(len(states_np)):
                metrics = agent.train_step(
                    states_np[i],
                    actions_np[i],
                    rewards_np[i],
                    next_states_np[i],
                    dones_np[i].item()
                )
                
                if metrics:
                    batch_loss.append(metrics.get('critic_loss', 0))
            
            # Log batch statistics
            if batch_loss:
                avg_loss = np.mean(batch_loss)
                epoch_loss.append(avg_loss)
                epoch_reward.append(rewards_np.mean())
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'reward': f'{rewards_np.mean():.3f}'
                })
        
        # Epoch summary
        avg_epoch_loss = np.mean(epoch_loss) if epoch_loss else 0
        avg_epoch_reward = np.mean(epoch_reward) if epoch_reward else 0
        
        epoch_losses.append(avg_epoch_loss)
        epoch_rewards.append(avg_epoch_reward)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {avg_epoch_loss:.4f}")
        print(f"  Reward: {avg_epoch_reward:.3f}")
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{epoch+1}.pt"
            agent.save(str(checkpoint_path))
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = checkpoint_dir / "best_loss.pt"
            agent.save(str(best_path))
            print(f"  New best model saved (loss: {best_loss:.4f})")
    
    # Save final model
    final_path = checkpoint_dir / "final.pt"
    agent.save(str(final_path))
    
    # Save training history
    history = {
        'epoch_losses': epoch_losses,
        'epoch_rewards': epoch_rewards,
    }
    
    history_path = log_dir / "training_history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"  Final loss: {epoch_losses[-1]:.4f}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Final model: {final_path}")
    print(f"  Best model: {best_path}")
    print("=" * 70)
    
    return history


def evaluate_agent(
    agent: QuantumDDPGAgent,
    data: dict,
    n_episodes: int = 10
):
    """
    Evaluate trained agent on test data.
    
    Args:
        agent: Trained agent
        data: Test data dictionary
        n_episodes: Number of episodes to evaluate
    """
    print("\n" + "=" * 70)
    print("Evaluating Agent")
    print("=" * 70)
    
    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    
    # Sample episodes
    episode_length = 200
    n_samples = len(states)
    
    episode_rewards = []
    bis_errors = []
    
    for ep in range(n_episodes):
        # Sample random starting point
        start_idx = np.random.randint(0, max(1, n_samples - episode_length))
        
        ep_states = states[start_idx:start_idx + episode_length]
        ep_reward = 0
        ep_bis_errors = []
        
        for state in ep_states:
            # Get agent action
            action = agent.select_action(state, deterministic=True)
            
            # BIS error
            bis_error = abs(state[0])  # First element is BIS error
            ep_bis_errors.append(bis_error)
            
            # Reward
            reward = 1.0 / (bis_error + 1.0)
            ep_reward += reward
        
        episode_rewards.append(ep_reward)
        bis_errors.extend(ep_bis_errors)
    
    # Statistics
    avg_reward = np.mean(episode_rewards)
    avg_bis_error = np.mean(bis_errors)
    
    print(f"  Episodes: {n_episodes}")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average BIS error: {avg_bis_error:.2f}")
    print(f"  Reward std: {np.std(episode_rewards):.2f}")
    print(f"  BIS error std: {np.std(bis_errors):.2f}")
    
    # Clinical metrics (if we have ground truth BIS)
    bis_values = 50 - np.array(bis_errors)  # Convert error to BIS
    target = 50
    
    mdape = np.mean(np.abs((bis_values - target) / target)) * 100
    mdpe = np.mean((bis_values - target) / target) * 100
    
    print(f"\n  Clinical Metrics:")
    print(f"    MDAPE: {mdape:.2f}%")
    print(f"    MDPE:  {mdpe:.2f}%")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Train Quantum DDPG with offline VitalDB data')
    parser.add_argument('--data_path', type=str,
                        default='./data/offline_dataset/vitaldb_offline_data.pkl',
                        help='Path to offline dataset')
    parser.add_argument('--config', type=str,
                        default='./config/hyperparameters.yaml',
                        help='Path to config file')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (auto-generated if not specified)')
    parser.add_argument('--encoder_type', type=str, default='none',
                        choices=['none', 'lstm', 'transformer'],
                        help='Encoder type')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate, do not train')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to load for evaluation')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create log directory
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = f"./logs/offline_{timestamp}"
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = log_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    print("=" * 70)
    print("Quantum DDPG Offline Training")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Config: {args.config}")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Encoder: {args.encoder_type}")
    print(f"  Log directory: {args.log_dir}")
    print(f"  Seed: {args.seed}")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading dataset...")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  Dataset size: {len(data['states']):,} transitions")
    print(f"  State dim: {data['states'].shape[1]}")
    print(f"  Action dim: {data['actions'].shape[1]}")
    
    # Create PyTorch dataset and dataloader
    dataset = VitalDBDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Initialize agent
    print("\nInitializing agent...")
    agent = QuantumDDPGAgent(
        state_dim=8,
        action_dim=1,
        config=config,
        encoder_type=args.encoder_type,
        seed=args.seed
    )
    
    print(f"  Quantum info: {agent.get_quantum_info()}")
    print(f"  Encoder info: {agent.get_encoder_info()}")
    
    # Load checkpoint if specified
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)
    
    # Training or evaluation
    if args.eval_only:
        if not args.checkpoint:
            print("Error: --checkpoint required for evaluation")
            return
        evaluate_agent(agent, data, n_episodes=20)
    else:
        # Train
        history = train_offline(
            agent=agent,
            dataloader=dataloader,
            n_epochs=args.n_epochs,
            log_dir=args.log_dir,
            save_freq=10
        )
        
        # Evaluate
        evaluate_agent(agent, data, n_episodes=10)


if __name__ == "__main__":
    main()
