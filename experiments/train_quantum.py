"""
Quantum RL Training Script for Propofol Infusion Control
==========================================================

This script implements the complete training loop for the
hybrid Quantum-Classical RL agents (DDPG and PPO) for propofol anesthesia control.

Following the CBIM paper methodology:
1. Train on simulated patients using the Schnider PK/PD model
2. Support dual drug interaction model (propofol + remifentanil) per (32)
3. Evaluate using clinical metrics (MDPE, MDAPE, Wobble) per (50)-(52)
4. Test generalization across patient populations

Algorithm Options:
- DDPG (Deep Deterministic Policy Gradient) with VQC actor
- PPO (Proximal Policy Optimization) with VQC actor per (41)-(49)

Encoder Options:
- None: Direct state input
- LSTM: Temporal feature extraction (Fig.4)
- Transformer: Attention-based temporal features (Fig.4)

Usage:
    python train_quantum.py --config config/hyperparameters.yaml
    python train_quantum.py --algorithm ppo --encoder lstm --episodes 1000
    python train_quantum.py --algorithm ddpg --encoder transformer --seed 42
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.propofol_env import PropofolEnv
from src.environment.patient_simulator import PatientParameters, create_patient_population
from src.agents.quantum_agent import QuantumDDPGAgent, EncoderType
from src.agents.quantum_ppo_agent import QuantumPPOAgent
from src.utils.metrics import (
    calculate_all_metrics,
    PerformanceMetrics,
    MetricsTracker,
    format_metrics_report
)
from src.utils.visualization import (
    plot_episode,
    plot_training_curves,
    create_summary_figure
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Quantum RL Agent for Propofol Infusion Control"
    )
    
    parser.add_argument(
        '--config', type=str,
        default='config/hyperparameters.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--algorithm', type=str, default=None,
        choices=['ddpg', 'ppo'],
        help='RL algorithm: ddpg or ppo (overrides config)'
    )
    parser.add_argument(
        '--encoder', type=str, default=None,
        choices=['none', 'lstm', 'transformer', 'hybrid'],
        help='Temporal encoder type (overrides config)'
    )
    parser.add_argument(
        '--episodes', type=int, default=None,
        help='Number of training episodes (overrides config)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs',
        help='Directory for logs and checkpoints'
    )
    parser.add_argument(
        '--eval_interval', type=int, default=50,
        help='Evaluate every N episodes'
    )
    parser.add_argument(
        '--save_interval', type=int, default=100,
        help='Save checkpoint every N episodes'
    )
    parser.add_argument(
        '--no_tensorboard', action='store_true',
        help='Disable TensorBoard logging'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--use_original_reward', action='store_true',
        help='Use original reward R = 1/(|g - BIS| + α) per (40)'
    )
    parser.add_argument(
        '--remifentanil', action='store_true',
        help='Enable remifentanil external input'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}


def setup_directories(log_dir: str, experiment_name: str) -> Dict[str, Path]:
    """Set up directories for logging and saving."""
    base_dir = Path(log_dir) / experiment_name
    
    dirs = {
        'base': base_dir,
        'checkpoints': base_dir / 'checkpoints',
        'figures': base_dir / 'figures',
        'tensorboard': base_dir / 'tensorboard'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def create_agent(
    algorithm: str,
    state_dim: int,
    action_dim: int,
    config: Dict,
    encoder_type: str,
    seed: int
) -> Union[QuantumDDPGAgent, QuantumPPOAgent]:
    """
    Create the appropriate RL agent based on algorithm choice.
    
    Args:
        algorithm: 'ddpg' or 'ppo'
        state_dim: Dimension of state space (8 for extended state)
        action_dim: Dimension of action space (1 for propofol dose)
        config: Configuration dictionary
        encoder_type: Type of temporal encoder ('none', 'lstm', 'transformer', 'hybrid')
        seed: Random seed
    
    Returns:
        Quantum RL agent (DDPG or PPO)
    """
    encoder_config = config.get('encoder', {})
    sequence_length = encoder_config.get('sequence_length', 10)
    
    if algorithm == 'ddpg':
        # Create Quantum DDPG Agent
        agent = QuantumDDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config,
            seed=seed,
            encoder_type=encoder_type,
            sequence_length=sequence_length
        )
        print(f"\n✓ Created Quantum DDPG Agent")
        
    elif algorithm == 'ppo':
        # Create Quantum PPO Agent per formulations (41)-(49)
        ppo_config = config.get('algorithm', {}).get('ppo', {})
        agent = QuantumPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config,
            seed=seed,
            encoder_type=encoder_type,
            sequence_length=sequence_length,
            # PPO-specific parameters per (41)-(49)
            n_steps=ppo_config.get('n_steps', 2048),
            n_epochs=ppo_config.get('n_epochs', 10),
            batch_size=ppo_config.get('minibatch_size', 64),
            gamma=config.get('training', {}).get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_epsilon=ppo_config.get('clip_epsilon', 0.2),
            value_coef=ppo_config.get('value_coef', 0.5),
            entropy_coef=ppo_config.get('entropy_coef', 0.01)
        )
        print(f"\n✓ Created Quantum PPO Agent (formulations 41-49)")
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Print encoder info
    if hasattr(agent, 'get_encoder_info'):
        enc_info = agent.get_encoder_info()
        print(f"  Encoder: {enc_info['encoder_type']} (output dim: {enc_info['encoded_dim']})")
    
    # Print quantum info
    q_info = agent.get_quantum_info()
    print(f"  VQC: {q_info['n_qubits']} qubits, {q_info['n_layers']} layers, {q_info['n_params']} parameters")
    
    return agent


def train_episode_ddpg(
    env: PropofolEnv,
    agent: QuantumDDPGAgent,
    train: bool = True
) -> Dict:
    """
    Run a single training episode for DDPG.
    
    Args:
        env: Propofol environment
        agent: Quantum DDPG agent
        train: Whether to update the agent
    
    Returns:
        Dictionary of episode statistics
    """
    state, info = env.reset()
    agent.reset_noise()
    
    episode_reward = 0.0
    episode_steps = 0
    actor_losses = []
    critic_losses = []
    q_values = []
    
    done = False
    while not done:
        # Select action
        action = agent.select_action(state, deterministic=not train)
        
        # Normalize action for environment (expects [0, 1])
        action_normalized = np.clip(action / agent.action_scale, 0, 1)
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action_normalized)
        done = terminated or truncated
        
        # Train step
        if train:
            metrics = agent.train_step(state, action, reward, next_state, done)
            
            if 'actor_loss' in metrics:
                actor_losses.append(metrics['actor_loss'])
            if 'critic_loss' in metrics:
                critic_losses.append(metrics['critic_loss'])
            if 'q_value_mean' in metrics:
                q_values.append(metrics['q_value_mean'])
        
        episode_reward += reward
        episode_steps += 1
        state = next_state
    
    # Decay exploration noise
    if train:
        agent.decay_noise()
    
    # Get episode metrics per (50)-(52)
    episode_metrics = env.get_episode_metrics()
    
    return {
        'reward': episode_reward,
        'steps': episode_steps,
        'mdpe': episode_metrics.get('mdpe', 0),
        'mdape': episode_metrics.get('mdape', 0),
        'wobble': episode_metrics.get('wobble', 0),
        'time_in_target': episode_metrics.get('time_in_target', 0),
        'mean_dose': episode_metrics.get('mean_dose', 0),
        'actor_loss': np.mean(actor_losses) if actor_losses else 0,
        'critic_loss': np.mean(critic_losses) if critic_losses else 0,
        'q_value': np.mean(q_values) if q_values else 0,
        'history': env.episode_history
    }


def train_episode_ppo(
    env: PropofolEnv,
    agent: QuantumPPOAgent,
    n_steps: int = 2048
) -> Dict:
    """
    Collect rollout for PPO training per formulations (41)-(49).
    
    PPO requires collecting a batch of experiences before updating.
    
    Args:
        env: Propofol environment
        agent: Quantum PPO agent
        n_steps: Number of steps to collect before update
    
    Returns:
        Dictionary of episode statistics
    """
    state, info = env.reset()
    
    episode_reward = 0.0
    episode_steps = 0
    all_rewards = []
    episodes_completed = 0
    
    # Collect rollout for n_steps
    for step in range(n_steps):
        # Select action
        action, log_prob, value = agent.select_action_with_info(state)
        
        # Normalize action for environment
        action_normalized = np.clip(action / agent.action_scale, 0, 1)
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action_normalized)
        done = terminated or truncated
        
        # Store transition in rollout buffer
        agent.store_transition(state, action, reward, done, log_prob, value)
        
        episode_reward += reward
        episode_steps += 1
        state = next_state
        
        if done:
            all_rewards.append(episode_reward)
            episodes_completed += 1
            episode_reward = 0.0
            state, info = env.reset()
    
    # Compute returns and advantages using GAE per (46)
    # Then update policy per (41)-(45)
    with torch.no_grad():
        _, _, last_value = agent.select_action_with_info(state)
    
    metrics = agent.update(last_value)
    
    # Get final episode metrics per (50)-(52)
    episode_metrics = env.get_episode_metrics()
    
    return {
        'reward': np.mean(all_rewards) if all_rewards else episode_reward,
        'steps': n_steps,
        'episodes_in_rollout': episodes_completed,
        'mdpe': episode_metrics.get('mdpe', 0),
        'mdape': episode_metrics.get('mdape', 0),
        'wobble': episode_metrics.get('wobble', 0),
        'time_in_target': episode_metrics.get('time_in_target', 0),
        'mean_dose': episode_metrics.get('mean_dose', 0),
        'actor_loss': metrics.get('policy_loss', 0),
        'critic_loss': metrics.get('value_loss', 0),
        'entropy': metrics.get('entropy', 0),
        'kl_divergence': metrics.get('kl_divergence', 0),
        'history': env.episode_history
    }


def train_episode(
    env: PropofolEnv,
    agent: Union[QuantumDDPGAgent, QuantumPPOAgent],
    algorithm: str,
    train: bool = True,
    n_steps: int = 2048
) -> Dict:
    """
    Unified training episode function.
    
    Args:
        env: Propofol environment
        agent: Quantum RL agent
        algorithm: 'ddpg' or 'ppo'
        train: Whether to update the agent
        n_steps: Steps per rollout (PPO only)
    
    Returns:
        Dictionary of episode statistics
    """
    if algorithm == 'ddpg':
        return train_episode_ddpg(env, agent, train)
    elif algorithm == 'ppo':
        return train_episode_ppo(env, agent, n_steps)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate_agent(
    env: PropofolEnv,
    agent: Union[QuantumDDPGAgent, QuantumPPOAgent],
    n_episodes: int = 10,
    patients: Optional[List[PatientParameters]] = None
) -> Dict:
    """
    Evaluate agent performance using clinical metrics (50)-(52).
    
    Args:
        env: Propofol environment
        agent: Quantum RL agent
        n_episodes: Number of evaluation episodes
        patients: Optional list of patients to evaluate on
    
    Returns:
        Dictionary of evaluation statistics including MDPE, MDAPE, Wobble
    """
    agent.set_eval_mode()
    
    eval_results = {
        'rewards': [],
        'mdpe': [],         # (50) Median Performance Error
        'mdape': [],        # (51) Median Absolute Performance Error
        'wobble': [],       # (52) Intra-individual variability
        'time_in_target': [],
        'mean_dose': []
    }
    
    for ep in range(n_episodes):
        # Use different patient if provided
        options = None
        if patients is not None and ep < len(patients):
            options = {'patient': patients[ep]}
        
        state, info = env.reset(options=options)
        
        episode_reward = 0.0
        done = False
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            action_normalized = np.clip(action / agent.action_scale, 0, 1)
            next_state, reward, terminated, truncated, info = env.step(action_normalized)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        # Get clinical performance metrics per (50)-(52)
        metrics = env.get_episode_metrics()
        
        eval_results['rewards'].append(episode_reward)
        eval_results['mdpe'].append(metrics.get('mdpe', 0))
        eval_results['mdape'].append(metrics.get('mdape', 0))
        eval_results['wobble'].append(metrics.get('wobble', 0))
        eval_results['time_in_target'].append(metrics.get('time_in_target', 0))
        eval_results['mean_dose'].append(metrics.get('mean_dose', 0))
    
    agent.set_train_mode()
    
    # Compute summary statistics
    summary = {
        'mean_reward': np.mean(eval_results['rewards']),
        'std_reward': np.std(eval_results['rewards']),
        'mean_mdpe': np.mean(eval_results['mdpe']),      # (50)
        'mean_mdape': np.mean(eval_results['mdape']),    # (51)
        'mean_wobble': np.mean(eval_results['wobble']),  # (52)
        'mean_time_in_target': np.mean(eval_results['time_in_target']),
        'mean_dose': np.mean(eval_results['mean_dose'])
    }
    
    return summary


def train(config: Dict, args) -> None:
    """
    Main training loop for Quantum RL agents.
    
    Supports both DDPG and PPO algorithms with optional temporal encoders.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Extract training config
    training_config = config.get('training', {})
    algorithm_config = config.get('algorithm', {})
    
    # Determine algorithm (command line overrides config)
    algorithm = args.algorithm or algorithm_config.get('type', 'ddpg')
    
    # Determine encoder type
    encoder_type = args.encoder or config.get('encoder', {}).get('type', 'none')
    
    n_episodes = args.episodes or training_config.get('total_episodes', 1000)
    seed = args.seed or config.get('seed', 42)
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"qrl_propofol_{algorithm}_{encoder_type}_{timestamp}"
    
    # Setup directories
    dirs = setup_directories(args.log_dir, experiment_name)
    
    # Apply command line overrides to config
    if args.use_original_reward:
        config.setdefault('environment', {})['use_original_reward'] = True
    if args.remifentanil:
        config.setdefault('environment', {}).setdefault('remifentanil', {})['enabled'] = True
    
    # Save config
    with open(dirs['base'] / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create environment with extended state per (36)-(39)
    env = PropofolEnv(config=config, seed=seed)
    
    # Create agent (DDPG or PPO)
    agent = create_agent(
        algorithm=algorithm,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=config,
        encoder_type=encoder_type,
        seed=seed
    )
    
    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume:
        agent.load(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Setup TensorBoard
    writer = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=str(dirs['tensorboard']))
    
    # Create evaluation patients
    eval_patients = create_patient_population(10, seed=seed + 1000)
    
    # Metrics tracker
    tracker = MetricsTracker(window_size=50)
    
    # Training history
    history = {
        'rewards': [],
        'mdpe': [],
        'mdape': [],
        'time_in_target': [],
        'actor_loss': [],
        'critic_loss': []
    }
    
    # Print training configuration
    print("\n" + "="*60)
    print("QUANTUM RL PROPOFOL INFUSION CONTROL")
    print("="*60)
    print(f"\nAlgorithm: {algorithm.upper()}")
    print(f"Encoder: {encoder_type}")
    print(f"Training Configuration:")
    print(f"  Episodes: {n_episodes}")
    print(f"  Seed: {seed}")
    print(f"  Log Directory: {dirs['base']}")
    
    if algorithm == 'ppo':
        ppo_config = algorithm_config.get('ppo', {})
        print(f"\nPPO Configuration (Formulations 41-49):")
        print(f"  GAE Lambda (λ): {ppo_config.get('gae_lambda', 0.95)} - per (46)")
        print(f"  Clip Epsilon (ε): {ppo_config.get('clip_epsilon', 0.2)} - per (42)")
        print(f"  Value Coefficient: {ppo_config.get('value_coef', 0.5)} - per (43)")
        print(f"  Entropy Coefficient: {ppo_config.get('entropy_coef', 0.01)} - per (45)")
    
    print("="*60 + "\n")
    
    # Training loop
    best_mdape = float('inf')
    best_reward = float('-inf')
    
    # PPO-specific parameters
    ppo_n_steps = algorithm_config.get('ppo', {}).get('n_steps', 2048) if algorithm == 'ppo' else None
    
    pbar = tqdm(range(start_episode, n_episodes), desc=f"Training {algorithm.upper()}")
    
    for episode in pbar:
        # Train episode
        result = train_episode(
            env, agent, algorithm,
            train=True,
            n_steps=ppo_n_steps
        )
        
        # Record history
        history['rewards'].append(result['reward'])
        history['mdpe'].append(result['mdpe'])
        history['mdape'].append(result['mdape'])
        history['time_in_target'].append(result['time_in_target'])
        history['actor_loss'].append(result['actor_loss'])
        history['critic_loss'].append(result['critic_loss'])
        
        # Add to tracker
        tracker.add_episode(
            PerformanceMetrics(
                mdpe=result['mdpe'],
                mdape=result['mdape'],
                wobble=result['wobble'],
                divergence=0,
                time_in_target=result['time_in_target'],
                mean_dose=result['mean_dose'],
                total_dose=0,
                settling_time=None
            ),
            result['reward']
        )
        
        # Update progress bar
        avg_reward = np.mean(history['rewards'][-50:])
        avg_mdape = np.mean(history['mdape'][-50:])
        pbar.set_postfix({
            'Reward': f'{avg_reward:.1f}',
            'MDAPE': f'{avg_mdape:.1f}%',
            'TiT': f'{result["time_in_target"]:.0f}%'
        })
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('Training/Reward', result['reward'], episode)
            writer.add_scalar('Training/MDPE', result['mdpe'], episode)
            writer.add_scalar('Training/MDAPE', result['mdape'], episode)
            writer.add_scalar('Training/TimeInTarget', result['time_in_target'], episode)
            writer.add_scalar('Training/MeanDose', result['mean_dose'], episode)
            writer.add_scalar('Loss/Actor', result['actor_loss'], episode)
            writer.add_scalar('Loss/Critic', result['critic_loss'], episode)
            
            if algorithm == 'ppo':
                writer.add_scalar('PPO/Entropy', result.get('entropy', 0), episode)
                writer.add_scalar('PPO/KL_Divergence', result.get('kl_divergence', 0), episode)
            else:
                writer.add_scalar('DDPG/QValue', result.get('q_value', 0), episode)
        
        # Periodic evaluation
        if (episode + 1) % args.eval_interval == 0:
            eval_summary = evaluate_agent(env, agent, n_episodes=5, patients=eval_patients[:5])
            
            print(f"\n[Episode {episode + 1}] Evaluation:")
            print(f"  Mean Reward: {eval_summary['mean_reward']:.2f} ± {eval_summary['std_reward']:.2f}")
            print(f"  Mean MDPE: {eval_summary['mean_mdpe']:.2f}% (50)")
            print(f"  Mean MDAPE: {eval_summary['mean_mdape']:.2f}% (51)")
            print(f"  Mean Wobble: {eval_summary['mean_wobble']:.2f}% (52)")
            print(f"  Mean Time in Target: {eval_summary['mean_time_in_target']:.1f}%")
            
            if writer is not None:
                writer.add_scalar('Eval/MeanReward', eval_summary['mean_reward'], episode)
                writer.add_scalar('Eval/MeanMDPE', eval_summary['mean_mdpe'], episode)
                writer.add_scalar('Eval/MeanMDAPE', eval_summary['mean_mdape'], episode)
                writer.add_scalar('Eval/MeanWobble', eval_summary['mean_wobble'], episode)
                writer.add_scalar('Eval/MeanTimeInTarget', eval_summary['mean_time_in_target'], episode)
            
            # Save best model
            if eval_summary['mean_mdape'] < best_mdape:
                best_mdape = eval_summary['mean_mdape']
                agent.save(str(dirs['checkpoints'] / 'best_mdape.pt'))
                print(f"  → New best MDAPE: {best_mdape:.2f}%")
            
            if eval_summary['mean_reward'] > best_reward:
                best_reward = eval_summary['mean_reward']
                agent.save(str(dirs['checkpoints'] / 'best_reward.pt'))
        
        # Periodic saving
        if (episode + 1) % args.save_interval == 0:
            agent.save(str(dirs['checkpoints'] / f'checkpoint_{episode + 1}.pt'))
            
            # Save training curves
            plot_training_curves(
                rewards=history['rewards'],
                mdape=history['mdape'],
                time_in_target=history['time_in_target'],
                actor_losses=history['actor_loss'],
                critic_losses=history['critic_loss'],
                title=f'{algorithm.upper()} Training Progress (Episode {episode + 1})',
                save_path=str(dirs['figures'] / f'training_curves_{episode + 1}.png'),
                show=False
            )
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print(f"Algorithm: {algorithm.upper()} | Encoder: {encoder_type}")
    print("="*60)
    
    final_eval = evaluate_agent(env, agent, n_episodes=10, patients=eval_patients)
    
    print(f"\nFinal Performance (10 patients):")
    print(f"  Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"  Mean MDPE: {final_eval['mean_mdpe']:.2f}% (Formulation 50)")
    print(f"  Mean MDAPE: {final_eval['mean_mdape']:.2f}% (Formulation 51)")
    print(f"  Mean Wobble: {final_eval['mean_wobble']:.2f}% (Formulation 52)")
    print(f"  Mean Time in Target: {final_eval['mean_time_in_target']:.1f}%")
    print(f"  Mean Dose: {final_eval['mean_dose']:.1f} μg/kg/min")
    
    # Save final model
    agent.save(str(dirs['checkpoints'] / 'final.pt'))
    
    # Save final training curves
    plot_training_curves(
        rewards=history['rewards'],
        mdape=history['mdape'],
        time_in_target=history['time_in_target'],
        actor_losses=history['actor_loss'],
        critic_losses=history['critic_loss'],
        title=f'{algorithm.upper()} Final Training Progress',
        save_path=str(dirs['figures'] / 'training_curves_final.png'),
        show=False
    )
    
    # Generate a sample episode visualization
    print("\nGenerating sample episode visualization...")
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, deterministic=True)
        action_normalized = np.clip(action / agent.action_scale, 0, 1)
        state, _, terminated, truncated, _ = env.step(action_normalized)
        done = terminated or truncated
    
    episode_data = {
        'time': np.array(env.episode_history['time']),
        'bis': np.array(env.episode_history['bis']),
        'dose': np.array(env.episode_history['dose']),
        'ce': np.array(env.episode_history['ce'])
    }
    metrics = env.get_episode_metrics()
    
    create_summary_figure(
        episode_data=episode_data,
        metrics=metrics,
        title=f'{algorithm.upper()} Sample Episode - Trained Agent',
        save_path=str(dirs['figures'] / 'sample_episode.png'),
        show=False
    )
    
    # Cleanup
    if writer is not None:
        writer.close()
    env.close()
    
    print(f"\n✓ Training complete! Results saved to: {dirs['base']}")
    print("="*60)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run training
    train(config, args)


if __name__ == "__main__":
    main()
