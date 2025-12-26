"""
Train Dual Drug Control Agent
=============================

Train Quantum/Classical RL agents for simultaneous Propofol + Remifentanil control.

Usage:
    python train_dual_drug.py --agent ddpg --mode hybrid
    python train_dual_drug.py --agent ppo --mode online --episodes 500
    python train_dual_drug.py --agent ddpg --encoder lstm --quantum_layers 4
"""

import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environment import DualDrugEnv, create_patient_parameters
from agents import QuantumDDPGAgent, QuantumPPOAgent
from utils.logging import Logger
from utils.replay_buffer import ReplayBuffer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train dual drug control agent")
    
    # Agent configuration
    parser.add_argument("--agent", type=str, default="ddpg", choices=["ddpg", "ppo"],
                        help="Agent type")
    parser.add_argument("--mode", type=str, default="online", choices=["online", "hybrid"],
                        help="Training mode (online only or hybrid offline+online)")
    
    # Environment
    parser.add_argument("--target_bis", type=float, default=50.0,
                        help="Target BIS level")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max steps per episode (minutes)")
    
    # Training
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--buffer_size", type=int, default=100000,
                        help="Replay buffer size")
    
    # Quantum architecture
    parser.add_argument("--quantum_layers", type=int, default=4,
                        help="Number of variational layers in VQC")
    parser.add_argument("--encoder", type=str, default=None,
                        choices=[None, "lstm", "transformer", "hybrid"],
                        help="Temporal encoder type")
    parser.add_argument("--encoded_dim", type=int, default=32,
                        help="Encoded state dimension")
    
    # Learning rates
    parser.add_argument("--lr_actor", type=float, default=1e-4,
                        help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-3,
                        help="Critic learning rate")
    
    # Other hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft update parameter")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for logs")
    parser.add_argument("--save_freq", type=int, default=100,
                        help="Save checkpoint every N episodes")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def create_agent(args, state_dim: int, action_dim: int, device):
    """Create RL agent based on arguments."""
    if args.agent == "ddpg":
        agent = QuantumDDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],
            quantum_layers=args.quantum_layers,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            tau=args.tau,
            encoder_type=args.encoder,
            encoded_dim=args.encoded_dim if args.encoder else None,
            device=device
        )
    elif args.agent == "ppo":
        agent = QuantumPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],
            quantum_layers=args.quantum_layers,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            encoder_type=args.encoder,
            encoded_dim=args.encoded_dim if args.encoder else None,
            device=device
        )
    else:
        raise ValueError(f"Unknown agent: {args.agent}")
    
    return agent


def train_online(agent, env, args, logger):
    """Train agent with online learning in simulator."""
    print("\n" + "=" * 70)
    print(f"Online Training: {args.episodes} episodes")
    print("=" * 70)
    
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_size,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    best_reward = -float('inf')
    
    for episode in range(args.episodes):
        # Reset environment with random patient
        patient_params = create_patient_parameters(
            age=np.random.uniform(20, 80),
            weight=np.random.uniform(50, 120),
            height=np.random.uniform(150, 200),
            gender=np.random.randint(0, 2)
        )
        env.patient = patient_params
        
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Episode rollout
        while True:
            # Select action
            action = agent.select_action(state, add_noise=True)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Update agent
            if len(replay_buffer) >= args.batch_size:
                batch = replay_buffer.sample(args.batch_size)
                metrics = agent.update(batch)
                
                # Log metrics
                if metrics and episode_steps % 10 == 0:
                    logger.log_metrics(metrics, episode * args.max_steps + episode_steps)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        # Log episode
        logger.log_episode(episode, episode_reward, episode_steps, info)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(logger.checkpoint_dir / "best_model.pt")
        
        # Periodic checkpoint
        if (episode + 1) % args.save_freq == 0:
            agent.save(logger.checkpoint_dir / f"episode_{episode+1}.pt")
            print(f"\nEpisode {episode+1}/{args.episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Steps: {episode_steps} | "
                  f"BIS: {info['bis']:.1f}")
    
    return agent


def evaluate_agent(agent, env, num_episodes: int = 10):
    """Evaluate trained agent."""
    print("\n" + "=" * 70)
    print(f"Evaluation: {num_episodes} episodes")
    print("=" * 70)
    
    total_rewards = []
    bis_errors = []
    drug_consumptions = []
    
    for episode in range(num_episodes):
        # Random patient
        patient_params = create_patient_parameters(
            age=np.random.uniform(20, 80),
            weight=np.random.uniform(50, 120),
            height=np.random.uniform(150, 200),
            gender=np.random.randint(0, 2)
        )
        env.patient = patient_params
        
        state, _ = env.reset()
        episode_reward = 0
        episode_bis_errors = []
        total_ppf = 0
        total_rftn = 0
        
        while True:
            action = agent.select_action(state, add_noise=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_bis_errors.append(abs(env.target_bis - info['bis']))
            total_ppf += action[0]
            total_rftn += action[1]
            
            state = next_state
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        bis_errors.append(np.mean(episode_bis_errors))
        drug_consumptions.append((total_ppf, total_rftn))
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Mean BIS Error={np.mean(episode_bis_errors):.2f}, "
              f"PPF={total_ppf:.1f}, RFTN={total_rftn:.1f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Evaluation Summary:")
    print(f"  Mean Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Mean BIS Error: {np.mean(bis_errors):.2f} ± {np.std(bis_errors):.2f}")
    print(f"  Mean Propofol: {np.mean([d[0] for d in drug_consumptions]):.2f}")
    print(f"  Mean Remifentanil: {np.mean([d[1] for d in drug_consumptions]):.2f}")
    print("=" * 70)


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = DualDrugEnv(
        target_bis=args.target_bis,
        max_steps=args.max_steps,
        seed=args.seed
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"\nEnvironment: Dual Drug (Propofol + Remifentanil)")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Target BIS: {args.target_bis}")
    print(f"  Max steps: {args.max_steps}")
    
    # Create agent
    agent = create_agent(args, state_dim, action_dim, device)
    print(f"\nAgent: {args.agent.upper()}")
    print(f"  Encoder: {args.encoder if args.encoder else 'None'}")
    print(f"  Quantum layers: {args.quantum_layers}")
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"dual_drug_{args.agent}_{timestamp}"
    log_dir = Path(args.log_dir) / exp_name
    logger = Logger(log_dir, args)
    
    # Train
    if args.mode == "online":
        agent = train_online(agent, env, args, logger)
    elif args.mode == "hybrid":
        print("Hybrid mode not yet implemented for dual drug control")
        print("Using online training only")
        agent = train_online(agent, env, args, logger)
    
    # Evaluate
    evaluate_agent(agent, env, num_episodes=20)
    
    # Save final model
    agent.save(logger.checkpoint_dir / "final_model.pt")
    print(f"\nTraining complete! Logs saved to: {log_dir}")


if __name__ == "__main__":
    main()
