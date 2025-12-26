"""
Train Quantum/Classical PPO Agent for Propofol Control
=======================================================

This script trains a Proximal Policy Optimization (PPO) agent for
propofol infusion control using online reinforcement learning.

PPO is an on-policy algorithm that collects trajectories, computes
advantages using GAE, and updates the policy with clipped objective.

Usage:
------
# Train Quantum PPO
python experiments/train_ppo.py --algorithm quantum_ppo --episodes 1000

# Train Classical PPO
python experiments/train_ppo.py --algorithm classical_ppo --episodes 1000

# With LSTM encoder
python experiments/train_ppo.py --algorithm quantum_ppo --encoder lstm --episodes 1000

# Resume training
python experiments/train_ppo.py --resume logs/ppo_20231221_120000/checkpoints/checkpoint_500.pt
"""

import sys
import argparse
import yaml
import pickle
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.quantum_ppo_agent import QuantumPPOAgent
from agents.classical_ppo_agent import ClassicalPPOAgent
from environment.propofol_env import PropofolEnv, make_env
from environment.patient_simulator import create_patient_population
from utils.logger import ExperimentLogger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent for propofol control")
    
    # Training
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--steps_per_episode", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for PPO updates")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of PPO epochs per update")
    parser.add_argument("--update_interval", type=int, default=2048, help="Steps between PPO updates")
    
    # Algorithm
    parser.add_argument("--algorithm", type=str, default="quantum_ppo", 
                       choices=["quantum_ppo", "classical_ppo"],
                       help="PPO algorithm type")
    parser.add_argument("--encoder", type=str, default="none",
                       choices=["none", "lstm", "transformer"],
                       help="Temporal encoder type")
    
    # Agent hyperparameters
    parser.add_argument("--state_dim", type=int, default=8, help="State dimension")
    parser.add_argument("--action_dim", type=int, default=1, help="Action dimension")
    parser.add_argument("--n_qubits", type=int, default=2, help="Number of qubits (quantum only)")
    parser.add_argument("--n_layers", type=int, default=4, help="VQC layers (quantum only)")
    
    # PPO hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--lr_actor", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="Critic learning rate")
    
    # Environment
    parser.add_argument("--config", type=str, default="config/hyperparameters.yaml",
                       help="Path to config file")
    parser.add_argument("--reward_type", type=str, default="paper",
                       choices=["paper", "shaped"],
                       help="Reward function type")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--save_interval", type=int, default=50, help="Checkpoint save interval")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation interval")
    parser.add_argument("--n_eval_episodes", type=int, default=10, help="Episodes for evaluation")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use")
    
    return parser.parse_args()


def train_ppo(args):
    """
    Train PPO agent with online learning.
    
    PPO Algorithm:
    --------------
    1. Collect trajectories using current policy
    2. Compute advantages using GAE
    3. Update policy with clipped objective
    4. Update value function
    5. Repeat
    
    Args:
        args: Command-line arguments
    """
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_default_dtype(torch.float32)
    
    # Device configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Update config with command-line args
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
    
    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.log_dir) / f"{args.algorithm}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    checkpoint_dir = log_dir / "checkpoints"
    figure_dir = log_dir / "figures"
    checkpoint_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    
    # Save config and args
    with open(log_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    with open(log_dir / "args.txt", 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    print(f"\n{'='*70}")
    print(f"TRAINING {args.algorithm.upper()}")
    print(f"{'='*70}")
    print(f"Log directory: {log_dir}")
    print(f"Episodes: {args.episodes}")
    print(f"Encoder: {args.encoder}")
    print(f"Update interval: {args.update_interval} steps")
    print(f"{'='*70}\n")
    
    # Create environment
    env = PropofolEnv(
        config_path=args.config,
        seed=args.seed,
        reward_type=args.reward_type
    )
    
    # Create agent
    if args.algorithm == "quantum_ppo":
        agent = QuantumPPOAgent(
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            config=config,
            encoder_type=args.encoder,
            seed=args.seed
        )
    else:  # classical_ppo
        agent = ClassicalPPOAgent(
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            config=config,
            encoder_type=args.encoder,
            seed=args.seed
        )
    
    # Move to device
    agent.to(device)
    
    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        start_episode = checkpoint.get('episode', 0)
        print(f"Resumed from episode {start_episode}")
    
    # Training logs
    training_log = {
        'episodes': [],
        'rewards': [],
        'mdapes': [],
        'time_in_target': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': []
    }
    
    # Training loop
    global_step = 0
    best_mdape = float('inf')
    
    print("\nStarting training...")
    for episode in tqdm(range(start_episode, args.episodes), desc="Training"):
        state, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_steps = 0
        
        # Collect trajectory
        while not done and not truncated and episode_steps < args.steps_per_episode:
            # Select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, log_prob, value = agent.select_action(state_tensor)
                action = action.cpu().numpy()
                log_prob = log_prob.cpu().item()
                value = value.cpu().item()
            
            # Environment step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition
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
            global_step += 1
            
            # Update policy at intervals
            if agent.buffer.is_full():
                # Compute last value for bootstrapping
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    _, _, last_value = agent.select_action(state_tensor)
                    last_value = last_value.cpu().item()
                
                # Compute advantages
                agent.buffer.compute_advantages(last_value)
                
                # Update policy
                policy_loss, value_loss, entropy = agent.update()
                
                # Log
                training_log['policy_losses'].append(policy_loss)
                training_log['value_losses'].append(value_loss)
                training_log['entropies'].append(entropy)
                
                # Clear buffer
                agent.buffer.clear()
        
        # Episode metrics
        metrics = env.get_episode_metrics()
        training_log['episodes'].append(episode)
        training_log['rewards'].append(episode_reward)
        training_log['mdapes'].append(metrics.get('mdape', 0))
        training_log['time_in_target'].append(metrics.get('time_in_target', 0))
        
        # Evaluation
        if (episode + 1) % args.eval_interval == 0:
            eval_mdape = evaluate_agent(agent, env, args.n_eval_episodes, device)
            
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  MDAPE: {metrics.get('mdape', 0):.2f}%")
            print(f"  Eval MDAPE: {eval_mdape:.2f}%")
            print(f"  Time in Target: {metrics.get('time_in_target', 0):.1f}%")
            if training_log['policy_losses']:
                print(f"  Policy Loss: {training_log['policy_losses'][-1]:.4f}")
                print(f"  Value Loss: {training_log['value_losses'][-1]:.4f}")
            
            # Save best model
            if eval_mdape < best_mdape:
                best_mdape = eval_mdape
                torch.save({
                    'episode': episode,
                    'agent_state_dict': agent.state_dict(),
                    'mdape': eval_mdape,
                    'config': config
                }, checkpoint_dir / "best.pt")
                print(f"  ✓ New best model saved (MDAPE: {eval_mdape:.2f}%)")
        
        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.state_dict(),
                'training_log': training_log,
                'config': config
            }, checkpoint_dir / f"checkpoint_{episode+1}.pt")
    
    # Save final model
    torch.save({
        'episode': args.episodes,
        'agent_state_dict': agent.state_dict(),
        'training_log': training_log,
        'config': config
    }, checkpoint_dir / "final.pt")
    
    # Save training log
    with open(log_dir / "training_log.pkl", 'wb') as f:
        pickle.dump(training_log, f)
    
    # Plot training curves
    plot_training_curves(training_log, figure_dir)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best MDAPE: {best_mdape:.2f}%")
    print(f"Final MDAPE: {training_log['mdapes'][-1]:.2f}%")
    print(f"Logs saved to: {log_dir}")
    print(f"{'='*70}\n")


def evaluate_agent(agent, env, n_episodes, device):
    """Evaluate agent performance."""
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


def plot_training_curves(log, save_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(log['episodes'], log['rewards'])
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].grid(True)
    
    # MDAPE
    axes[0, 1].plot(log['episodes'], log['mdapes'])
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('MDAPE (%)')
    axes[0, 1].set_title('Performance Error')
    axes[0, 1].grid(True)
    
    # Time in target
    axes[0, 2].plot(log['episodes'], log['time_in_target'])
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Time in Target (%)')
    axes[0, 2].set_title('Time in Target Range')
    axes[0, 2].grid(True)
    
    # Policy loss
    if log['policy_losses']:
        axes[1, 0].plot(log['policy_losses'])
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Policy Loss')
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].grid(True)
    
    # Value loss
    if log['value_losses']:
        axes[1, 1].plot(log['value_losses'])
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Value Loss')
        axes[1, 1].set_title('Value Function Loss')
        axes[1, 1].grid(True)
    
    # Entropy
    if log['entropies']:
        axes[1, 2].plot(log['entropies'])
        axes[1, 2].set_xlabel('Update')
        axes[1, 2].set_ylabel('Entropy')
        axes[1, 2].set_title('Policy Entropy')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    train_ppo(args)
