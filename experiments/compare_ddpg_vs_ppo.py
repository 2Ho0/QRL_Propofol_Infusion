"""
Compare DDPG vs PPO Algorithms for Propofol Control
====================================================

This script trains and compares DDPG and PPO agents (both quantum and classical)
for propofol infusion control, evaluating which algorithm performs better.

DDPG (Deep Deterministic Policy Gradient):
- Off-policy algorithm
- Deterministic policy
- Replay buffer
- Continuous action space

PPO (Proximal Policy Optimization):
- On-policy algorithm
- Stochastic policy
- Trajectory collection
- Clipped objective for stable updates

Comparison Metrics:
-------------------
- MDPE (Median Performance Error)
- MDAPE (Median Absolute Performance Error)
- Wobble (Variability)
- Time in Target (45-55 BIS)
- Sample Efficiency (reward vs timesteps)
- Training Stability

Usage:
------
# Compare Quantum DDPG vs Quantum PPO
python experiments/compare_ddpg_vs_ppo.py --quantum --online_episodes 500

# Compare Classical DDPG vs Classical PPO
python experiments/compare_ddpg_vs_ppo.py --classical --online_episodes 500

# Compare all four (Quantum DDPG, Classical DDPG, Quantum PPO, Classical PPO)
python experiments/compare_ddpg_vs_ppo.py --all --online_episodes 500

# With LSTM encoder
python experiments/compare_ddpg_vs_ppo.py --all --encoder lstm --online_episodes 500
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
import yaml
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from agents.quantum_agent import QuantumDDPGAgent
from agents.classical_agent import ClassicalDDPGAgent
from agents.quantum_ppo_agent import QuantumPPOAgent
from agents.classical_ppo_agent import ClassicalPPOAgent
from environment.propofol_env import PropofolEnv
from environment.patient_simulator import create_patient_population


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare DDPG vs PPO")
    
    # Comparison scope
    parser.add_argument('--quantum', action='store_true',
                       help='Compare Quantum DDPG vs Quantum PPO')
    parser.add_argument('--classical', action='store_true',
                       help='Compare Classical DDPG vs Classical PPO')
    parser.add_argument('--all', action='store_true',
                       help='Compare all four combinations')
    
    # Training
    parser.add_argument('--online_episodes', type=int, default=500,
                       help='Training episodes')
    parser.add_argument('--n_test_episodes', type=int, default=50,
                       help='Test episodes')
    
    # Agent configuration
    parser.add_argument('--state_dim', type=int, default=8)
    parser.add_argument('--action_dim', type=int, default=1)
    parser.add_argument('--n_qubits', type=int, default=2)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--encoder', type=str, default='none',
                       choices=['none', 'lstm', 'transformer'])
    
    # Misc
    parser.add_argument('--config', type=str, default='config/hyperparameters.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level for statistical tests')
    
    return parser.parse_args()


def train_ddpg_agent(agent, env, n_episodes, device, algorithm_name):
    """
    Train DDPG agent.
    
    DDPG is off-policy, so we collect experiences in replay buffer
    and update after each step.
    """
    print(f"\nTraining {algorithm_name}...")
    
    training_log = {
        'episodes': [],
        'rewards': [],
        'mdapes': [],
        'time_in_target': []
    }
    
    for episode in tqdm(range(n_episodes), desc=f"{algorithm_name} Training"):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        # Enable exploration after warmup
        add_noise = episode >= 50
        
        while not done and not truncated:
            # Select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = agent.select_action(state_tensor, add_noise=add_noise)
                action = action.cpu().numpy()
            
            # Step
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Train
            if len(agent.replay_buffer) > agent.batch_size:
                agent.train_step()
            
            state = next_state
            episode_reward += reward
        
        # Decay noise
        if hasattr(agent, 'decay_noise'):
            agent.decay_noise()
        
        # Log
        metrics = env.get_episode_metrics()
        training_log['episodes'].append(episode)
        training_log['rewards'].append(episode_reward)
        training_log['mdapes'].append(metrics.get('mdape', 0))
        training_log['time_in_target'].append(metrics.get('time_in_target', 0))
    
    return training_log


def train_ppo_agent(agent, env, n_episodes, device, algorithm_name):
    """
    Train PPO agent.
    
    PPO is on-policy, so we collect trajectories and update periodically.
    """
    print(f"\nTraining {algorithm_name}...")
    
    training_log = {
        'episodes': [],
        'rewards': [],
        'mdapes': [],
        'time_in_target': []
    }
    
    for episode in tqdm(range(n_episodes), desc=f"{algorithm_name} Training"):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
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
            
            # Update when buffer is full
            if agent.buffer.is_full():
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    _, _, last_value = agent.select_action(state_tensor)
                    last_value = last_value.cpu().item()
                
                agent.buffer.compute_advantages(last_value)
                agent.update()
                agent.buffer.clear()
        
        # Log
        metrics = env.get_episode_metrics()
        training_log['episodes'].append(episode)
        training_log['rewards'].append(episode_reward)
        training_log['mdapes'].append(metrics.get('mdape', 0))
        training_log['time_in_target'].append(metrics.get('time_in_target', 0))
    
    return training_log


def evaluate_agent_on_simulator(agent, n_episodes, seed, device):
    """Evaluate agent on diverse patient population."""
    agent.eval()
    
    # Create diverse patients
    patients = create_patient_population(
        n_patients=n_episodes,
        age_range=(18, 80),
        weight_range=(50, 120),
        height_range=(150, 195),
        gender_dist={'M': 0.5, 'F': 0.5},
        seed=seed
    )
    
    results = {
        'mdape': [],
        'mdpe': [],
        'wobble': [],
        'time_in_target': [],
        'rewards': [],
        'induction_time': [],
        'propofol_usage': []
    }
    
    for patient in tqdm(patients, desc="Evaluating"):
        env = PropofolEnv(patient=patient, seed=seed)
        state, _ = env.reset()
        done = False
        truncated = False
        
        while not done and not truncated:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # Get action (different for DDPG vs PPO)
                if hasattr(agent, 'buffer'):  # PPO
                    action, _, _ = agent.select_action(state_tensor, deterministic=True)
                else:  # DDPG
                    action = agent.select_action(state_tensor, add_noise=False)
                
                action = action.cpu().numpy()
            
            state, _, done, truncated, _ = env.step(action)
        
        # Get metrics
        metrics = env.get_episode_metrics()
        results['mdape'].append(metrics.get('mdape', 0))
        results['mdpe'].append(metrics.get('mdpe', 0))
        results['wobble'].append(metrics.get('wobble', 0))
        results['time_in_target'].append(metrics.get('time_in_target', 0))
        results['rewards'].append(metrics.get('total_reward', 0))
        results['induction_time'].append(metrics.get('induction_time', 0) or 0)
        results['propofol_usage'].append(metrics.get('mean_dose_ppf', 0))
    
    agent.train()
    
    # Compute statistics
    summary = {
        'mdape_mean': np.mean(results['mdape']),
        'mdape_std': np.std(results['mdape']),
        'mdape_list': results['mdape'],
        'mdpe_mean': np.mean(results['mdpe']),
        'wobble_mean': np.mean(results['wobble']),
        'time_in_target_mean': np.mean(results['time_in_target']),
        'reward_mean': np.mean(results['rewards']),
        'induction_time_mean': np.mean([t for t in results['induction_time'] if t > 0]),
        'propofol_usage_mean': np.mean(results['propofol_usage'])
    }
    
    return summary


def statistical_comparison(results_a, results_b, name_a, name_b, alpha=0.05):
    """Perform statistical comparison between two algorithms."""
    mdapes_a = results_a['mdape_list']
    mdapes_b = results_b['mdape_list']
    
    # T-test
    t_stat, t_pval = stats.ttest_ind(mdapes_a, mdapes_b)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(mdapes_a, mdapes_b, alternative='two-sided')
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(mdapes_a) - np.mean(mdapes_b)
    pooled_std = np.sqrt((np.var(mdapes_a) + np.var(mdapes_b)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    # Determine winner
    if t_pval < alpha:
        winner = name_a if mean_diff < 0 else name_b
        significant = True
    else:
        winner = "Tie"
        significant = False
    
    return {
        't_statistic': t_stat,
        't_pvalue': t_pval,
        'u_statistic': u_stat,
        'u_pvalue': u_pval,
        'cohens_d': cohens_d,
        'mean_difference': mean_diff,
        'winner': winner,
        'significant': significant
    }


def plot_comparison(ddpg_results, ppo_results, ddpg_name, ppo_name, save_path):
    """Plot comparison between DDPG and PPO."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # MDAPE box plot
    axes[0, 0].boxplot([ddpg_results['mdape_list'], ppo_results['mdape_list']],
                        labels=[ddpg_name, ppo_name])
    axes[0, 0].set_ylabel('MDAPE (%)')
    axes[0, 0].set_title('Performance Error Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reward comparison
    data = [[ddpg_results['reward_mean']], [ppo_results['reward_mean']]]
    axes[0, 1].bar([ddpg_name, ppo_name],
                    [ddpg_results['reward_mean'], ppo_results['reward_mean']],
                    color=['lightblue', 'lightcoral'])
    axes[0, 1].set_ylabel('Mean Reward')
    axes[0, 1].set_title('Average Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Time in target
    axes[0, 2].bar([ddpg_name, ppo_name],
                    [ddpg_results['time_in_target_mean'], ppo_results['time_in_target_mean']],
                    color=['lightblue', 'lightcoral'])
    axes[0, 2].set_ylabel('Time in Target (%)')
    axes[0, 2].set_title('Time in Target Range (45-55 BIS)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # MDAPE histogram
    axes[1, 0].hist(ddpg_results['mdape_list'], bins=20, alpha=0.6,
                     label=ddpg_name, color='blue')
    axes[1, 0].hist(ppo_results['mdape_list'], bins=20, alpha=0.6,
                     label=ppo_name, color='red')
    axes[1, 0].set_xlabel('MDAPE (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('MDAPE Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Propofol usage
    axes[1, 1].bar([ddpg_name, ppo_name],
                    [ddpg_results['propofol_usage_mean'], ppo_results['propofol_usage_mean']],
                    color=['lightblue', 'lightcoral'])
    axes[1, 1].set_ylabel('Propofol (μg/kg/min)')
    axes[1, 1].set_title('Mean Propofol Usage')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Induction time
    axes[1, 2].bar([ddpg_name, ppo_name],
                    [ddpg_results['induction_time_mean'], ppo_results['induction_time_mean']],
                    color=['lightblue', 'lightcoral'])
    axes[1, 2].set_ylabel('Time (minutes)')
    axes[1, 2].set_title('Mean Induction Time')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    
    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['device'] = str(device)
    
    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.log_dir) / f'comparison_ddpg_vs_ppo_{timestamp}'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("DDPG vs PPO COMPARISON")
    print(f"{'='*70}")
    print(f"Log directory: {log_dir}")
    print(f"Training episodes: {args.online_episodes}")
    print(f"Test episodes: {args.n_test_episodes}")
    print(f"{'='*70}\n")
    
    # Determine which comparisons to run
    comparisons = []
    if args.all:
        comparisons = [
            ('quantum_ddpg', 'quantum_ppo'),
            ('classical_ddpg', 'classical_ppo'),
            ('quantum_ddpg', 'classical_ddpg'),
            ('quantum_ppo', 'classical_ppo')
        ]
    elif args.quantum:
        comparisons = [('quantum_ddpg', 'quantum_ppo')]
    elif args.classical:
        comparisons = [('classical_ddpg', 'classical_ppo')]
    else:
        # Default: compare quantum DDPG vs PPO
        comparisons = [('quantum_ddpg', 'quantum_ppo')]
    
    # Run comparisons
    all_results = {}
    
    for alg_a, alg_b in comparisons:
        print(f"\n{'='*70}")
        print(f"COMPARING: {alg_a.upper()} vs {alg_b.upper()}")
        print(f"{'='*70}")
        
        # Create environments
        env_a = PropofolEnv(config_path=args.config, seed=args.seed)
        env_b = PropofolEnv(config_path=args.config, seed=args.seed + 1000)
        
        # Create agents
        agents = {}
        for alg in [alg_a, alg_b]:
            if alg == 'quantum_ddpg':
                agents[alg] = QuantumDDPGAgent(
                    state_dim=args.state_dim,
                    action_dim=args.action_dim,
                    n_qubits=args.n_qubits,
                    n_layers=args.n_layers,
                    config=config,
                    encoder_type=args.encoder,
                    seed=args.seed
                )
            elif alg == 'classical_ddpg':
                agents[alg] = ClassicalDDPGAgent(
                    state_dim=args.state_dim,
                    action_dim=args.action_dim,
                    config=config,
                    encoder_type=args.encoder,
                    seed=args.seed
                )
            elif alg == 'quantum_ppo':
                agents[alg] = QuantumPPOAgent(
                    state_dim=args.state_dim,
                    action_dim=args.action_dim,
                    n_qubits=args.n_qubits,
                    n_layers=args.n_layers,
                    config=config,
                    encoder_type=args.encoder,
                    seed=args.seed
                )
            elif alg == 'classical_ppo':
                agents[alg] = ClassicalPPOAgent(
                    state_dim=args.state_dim,
                    action_dim=args.action_dim,
                    config=config,
                    encoder_type=args.encoder,
                    seed=args.seed
                )
            
            agents[alg].to(device)
        
        # Train agents
        if 'ddpg' in alg_a:
            log_a = train_ddpg_agent(agents[alg_a], env_a, args.online_episodes, device, alg_a)
        else:
            log_a = train_ppo_agent(agents[alg_a], env_a, args.online_episodes, device, alg_a)
        
        if 'ddpg' in alg_b:
            log_b = train_ddpg_agent(agents[alg_b], env_b, args.online_episodes, device, alg_b)
        else:
            log_b = train_ppo_agent(agents[alg_b], env_b, args.online_episodes, device, alg_b)
        
        # Evaluate
        print(f"\nEvaluating {alg_a}...")
        results_a = evaluate_agent_on_simulator(
            agents[alg_a], args.n_test_episodes, args.seed + 2000, device
        )
        
        print(f"Evaluating {alg_b}...")
        results_b = evaluate_agent_on_simulator(
            agents[alg_b], args.n_test_episodes, args.seed + 3000, device
        )
        
        # Statistical comparison
        stats_results = statistical_comparison(results_a, results_b, alg_a, alg_b, args.alpha)
        
        # Save results
        comparison_name = f"{alg_a}_vs_{alg_b}"
        all_results[comparison_name] = {
            f'{alg_a}_results': results_a,
            f'{alg_b}_results': results_b,
            f'{alg_a}_training': log_a,
            f'{alg_b}_training': log_b,
            'statistics': stats_results
        }
        
        # Plot
        plot_comparison(
            results_a, results_b, alg_a.upper(), alg_b.upper(),
            log_dir / f'{comparison_name}.png'
        )
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"RESULTS: {alg_a.upper()} vs {alg_b.upper()}")
        print(f"{'='*70}")
        print(f"\n{alg_a.upper()}:")
        print(f"  MDAPE: {results_a['mdape_mean']:.2f} ± {results_a['mdape_std']:.2f}%")
        print(f"  MDPE: {results_a['mdpe_mean']:.2f}%")
        print(f"  Wobble: {results_a['wobble_mean']:.2f}%")
        print(f"  Time in Target: {results_a['time_in_target_mean']:.1f}%")
        
        print(f"\n{alg_b.upper()}:")
        print(f"  MDAPE: {results_b['mdape_mean']:.2f} ± {results_b['mdape_std']:.2f}%")
        print(f"  MDPE: {results_b['mdpe_mean']:.2f}%")
        print(f"  Wobble: {results_b['wobble_mean']:.2f}%")
        print(f"  Time in Target: {results_b['time_in_target_mean']:.1f}%")
        
        print(f"\nStatistical Test:")
        print(f"  Winner: {stats_results['winner']}")
        print(f"  p-value: {stats_results['t_pvalue']:.4f}")
        print(f"  Significant: {stats_results['significant']} (α={args.alpha})")
        print(f"  Cohen's d: {stats_results['cohens_d']:.3f}")
        print(f"{'='*70}\n")
    
    # Save all results
    with open(log_dir / 'all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Generate summary report
    with open(log_dir / 'summary_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("DDPG vs PPO COMPARISON SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        for comparison_name, data in all_results.items():
            f.write(f"\n{comparison_name.upper()}\n")
            f.write("-"*70 + "\n")
            
            alg_a, alg_b = comparison_name.split('_vs_')
            results_a = data[f'{alg_a}_results']
            results_b = data[f'{alg_b}_results']
            stats_res = data['statistics']
            
            f.write(f"\n{alg_a.upper()}:\n")
            f.write(f"  MDAPE: {results_a['mdape_mean']:.2f}%\n")
            f.write(f"  Time in Target: {results_a['time_in_target_mean']:.1f}%\n")
            
            f.write(f"\n{alg_b.upper()}:\n")
            f.write(f"  MDAPE: {results_b['mdape_mean']:.2f}%\n")
            f.write(f"  Time in Target: {results_b['time_in_target_mean']:.1f}%\n")
            
            f.write(f"\nWinner: {stats_res['winner']}\n")
            f.write(f"p-value: {stats_res['t_pvalue']:.4f}\n")
            f.write(f"Significant: {stats_res['significant']}\n\n")
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {log_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
