"""
Quantum vs Classical RL Comparison for Dual Drug Control
=========================================================

Compare Quantum and Classical agents on dual drug environment
(Propofol + Remifentanil).

Key differences from single drug comparison:
- Action space: [propofol_rate, remifentanil_rate] (2D)
- State space: Extended to include remifentanil Ce (10D)
- Agents: action_dim=2 instead of 1

Usage:
    # Full comparison
    python experiments/compare_quantum_vs_classical_dualdrug.py \
        --online_episodes 500 --n_test_episodes 50
    
    # Quick test
    python experiments/compare_quantum_vs_classical_dualdrug.py \
        --online_episodes 50 --n_test_episodes 10
    
    # With LSTM encoder
    python experiments/compare_quantum_vs_classical_dualdrug.py \
        --online_episodes 500 --encoder lstm
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import argparse
import pickle
import numpy as np
import torch
import yaml
from datetime import datetime
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from agents.quantum_agent import QuantumDDPGAgent
from agents.classical_agent import ClassicalDDPGAgent
from environment.dual_drug_env import DualDrugEnv
from environment.patient_simulator import create_patient_population


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare Quantum vs Classical RL for Dual Drug Control'
    )
    
    # Training
    parser.add_argument('--online_episodes', type=int, default=500,
                       help='Number of online training episodes')
    parser.add_argument('--warmup_episodes', type=int, default=50,
                       help='Warmup episodes before exploration')
    
    # Agent (dual drug specific)
    parser.add_argument('--state_dim', type=int, default=10,
                       help='State dimension for dual drug (extended)')
    parser.add_argument('--action_dim', type=int, default=2,
                       help='Action dimension (propofol + remifentanil)')
    parser.add_argument('--encoder', type=str, default='none',
                       choices=['none', 'lstm', 'transformer'],
                       help='Temporal encoder type')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Directories
    parser.add_argument('--config', type=str, 
                       default='config/hyperparameters.yaml',
                       help='Path to config file')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for logs')
    
    # Evaluation
    parser.add_argument('--n_test_episodes', type=int, default=50,
                       help='Number of test episodes per agent')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level for statistical tests')
    
    return parser.parse_args()


def evaluate_agent_on_simulator_dualdrug(
    agent, 
    n_episodes: int, 
    seed: int,
    device: torch.device
) -> Dict:
    """
    Evaluate agent on dual drug simulator.
    
    Key differences:
    - DualDrugEnv instead of PropofolEnv
    - Action is 2D: [propofol_rate, remifentanil_rate]
    - State is 10D (includes remifentanil Ce)
    """
    patients = create_patient_population(n_patients=n_episodes, seed=seed)
    
    mdapes = []
    rewards_total = []
    propofol_usage = []
    remifentanil_usage = []
    time_in_target = []
    
    for i, patient in enumerate(tqdm(patients, desc="Evaluating on Dual Drug Simulator")):
        # Create dual drug environment
        env = DualDrugEnv(patient=patient, seed=seed + i)
        
        state, _ = env.reset()
        episode_reward = 0
        episode_ppf = []
        episode_rftn = []
        states_list = []
        bis_history = []
        
        for step in range(200):
            state_tensor = torch.FloatTensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if agent.encoder is not None:
                    if len(states_list) == 0:
                        states_seq = state_tensor.unsqueeze(0)
                    else:
                        recent_states = states_list[-19:] if len(states_list) >= 19 else states_list
                        states_array = np.array(recent_states + [state], dtype=np.float32)
                        states_seq = torch.FloatTensor(states_array).unsqueeze(0).to(device)
                    
                    encoded = agent.encoder(states_seq)
                    if isinstance(encoded, tuple):
                        encoded = encoded[0]
                    action = agent.actor(encoded[:, -1, :])
                else:
                    action = agent.actor(state_tensor)
            
            # Action is 2D: [propofol, remifentanil]
            action_np = action.cpu().numpy().flatten()
            
            # Ensure action is 2D
            if action_np.shape[0] != 2:
                print(f"Warning: Expected action_dim=2, got {action_np.shape[0]}. Using first 2 dimensions.")
                action_np = action_np[:2]
            
            next_state, reward, done, truncated, info = env.step(action_np)
            
            episode_reward += reward
            episode_ppf.append(action_np[0])
            episode_rftn.append(action_np[1])
            states_list.append(state)
            
            # Track BIS for time in target
            bis = info.get('bis', 50 - state[0])  # state[0] is BIS_error
            bis_history.append(bis)
            
            state = next_state
            
            if done or truncated:
                break
        
        # Get metrics
        metrics = env.get_episode_metrics()
        mdapes.append(metrics.get('mdape', 0))
        rewards_total.append(episode_reward)
        propofol_usage.append(np.mean(episode_ppf))
        remifentanil_usage.append(np.mean(episode_rftn))
        
        # Calculate time in target (BIS 45-55)
        bis_array = np.array(bis_history)
        tit = np.sum((bis_array >= 45) & (bis_array <= 55)) / len(bis_array) * 100
        time_in_target.append(tit)
    
    return {
        'mdape_mean': np.mean(mdapes),
        'mdape_std': np.std(mdapes),
        'mdape_list': mdapes,
        'reward_mean': np.mean(rewards_total),
        'reward_std': np.std(rewards_total),
        'reward_list': rewards_total,
        'propofol_usage_mean': np.mean(propofol_usage),
        'propofol_usage_std': np.std(propofol_usage),
        'remifentanil_usage_mean': np.mean(remifentanil_usage),
        'remifentanil_usage_std': np.std(remifentanil_usage),
        'time_in_target_mean': np.mean(time_in_target),
        'time_in_target_std': np.std(time_in_target)
    }


def statistical_comparison(quantum_results: Dict, classical_results: Dict, 
                          alpha: float = 0.05) -> Dict:
    """Perform statistical significance tests."""
    
    # Two-sample t-test
    t_stat, t_pval = stats.ttest_ind(
        quantum_results['mdape_list'],
        classical_results['mdape_list']
    )
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(
        quantum_results['mdape_list'],
        classical_results['mdape_list'],
        alternative='two-sided'
    )
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.var(quantum_results['mdape_list']) + np.var(classical_results['mdape_list'])) / 2
    )
    cohens_d = (quantum_results['mdape_mean'] - classical_results['mdape_mean']) / pooled_std
    
    # Mean difference and CI
    mean_diff = quantum_results['mdape_mean'] - classical_results['mdape_mean']
    se_diff = np.sqrt(
        quantum_results['mdape_std']**2 / len(quantum_results['mdape_list']) +
        classical_results['mdape_std']**2 / len(classical_results['mdape_list'])
    )
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff
    
    # Improvement percentage
    improvement_pct = (classical_results['mdape_mean'] - quantum_results['mdape_mean']) / \
                     classical_results['mdape_mean'] * 100
    
    return {
        't_statistic': t_stat,
        't_pvalue': t_pval,
        't_significant': t_pval < alpha,
        'u_statistic': u_stat,
        'u_pvalue': u_pval,
        'u_significant': u_pval < alpha,
        'cohens_d': cohens_d,
        'mean_difference': mean_diff,
        'ci_95': (ci_lower, ci_upper),
        'winner': 'Quantum' if mean_diff < 0 else 'Classical',
        'improvement_pct': improvement_pct
    }


def plot_dualdrug_comparison(
    quantum_results: Dict, 
    classical_results: Dict, 
    title: str, 
    save_path: Path
):
    """
    Generate dual drug comparison plots.
    
    6 subplots:
    1. MDAPE box plot
    2. Reward box plot
    3. Propofol usage
    4. MDAPE histogram
    5. Cumulative distribution
    6. Remifentanil usage
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. MDAPE Box Plot
    ax = axes[0, 0]
    data = [quantum_results['mdape_list'], classical_results['mdape_list']]
    bp = ax.boxplot(data, labels=['Quantum', 'Classical'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('MDAPE (%)', fontsize=12)
    ax.set_title('MDAPE Distribution', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean markers
    ax.plot(1, quantum_results['mdape_mean'], 'D', color='blue', markersize=8, label='Mean')
    ax.plot(2, classical_results['mdape_mean'], 'D', color='red', markersize=8)
    ax.legend()
    
    # 2. Reward Box Plot
    ax = axes[0, 1]
    data = [quantum_results['reward_list'], classical_results['reward_list']]
    bp = ax.boxplot(data, labels=['Quantum', 'Classical'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Reward Distribution', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean markers
    ax.plot(1, quantum_results['reward_mean'], 'D', color='blue', markersize=8, label='Mean')
    ax.plot(2, classical_results['reward_mean'], 'D', color='red', markersize=8)
    ax.legend()
    
    # 3. Propofol Usage
    ax = axes[0, 2]
    ppf_means = [
        quantum_results['propofol_usage_mean'],
        classical_results['propofol_usage_mean']
    ]
    ppf_stds = [
        quantum_results['propofol_usage_std'],
        classical_results['propofol_usage_std']
    ]
    x_pos = [1, 2]
    ax.bar(x_pos, ppf_means, yerr=ppf_stds, capsize=5, 
           color=['lightblue', 'lightcoral'], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Quantum', 'Classical'])
    ax.set_ylabel('Mean Propofol Rate (mg/kg/h)', fontsize=12)
    ax.set_title('Propofol Usage', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. MDAPE Histogram
    ax = axes[1, 0]
    ax.hist(quantum_results['mdape_list'], bins=20, alpha=0.6, label='Quantum', color='blue')
    ax.hist(classical_results['mdape_list'], bins=20, alpha=0.6, label='Classical', color='red')
    ax.axvline(quantum_results['mdape_mean'], color='blue', linestyle='--', linewidth=2)
    ax.axvline(classical_results['mdape_mean'], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('MDAPE (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('MDAPE Histogram', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 5. Cumulative Distribution
    ax = axes[1, 1]
    quantum_sorted = np.sort(quantum_results['mdape_list'])
    classical_sorted = np.sort(classical_results['mdape_list'])
    quantum_cdf = np.arange(1, len(quantum_sorted) + 1) / len(quantum_sorted)
    classical_cdf = np.arange(1, len(classical_sorted) + 1) / len(classical_sorted)
    
    ax.plot(quantum_sorted, quantum_cdf, label='Quantum', color='blue', linewidth=2)
    ax.plot(classical_sorted, classical_cdf, label='Classical', color='red', linewidth=2)
    ax.set_xlabel('MDAPE (%)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 6. Remifentanil Usage
    ax = axes[1, 2]
    rftn_means = [
        quantum_results['remifentanil_usage_mean'],
        classical_results['remifentanil_usage_mean']
    ]
    rftn_stds = [
        quantum_results['remifentanil_usage_std'],
        classical_results['remifentanil_usage_std']
    ]
    ax.bar(x_pos, rftn_means, yerr=rftn_stds, capsize=5,
           color=['lightblue', 'lightcoral'], alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Quantum', 'Classical'])
    ax.set_ylabel('Mean Remifentanil Rate (μg/kg/min)', fontsize=12)
    ax.set_title('Remifentanil Usage', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {save_path}")
    plt.close()


def main():
    args = parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    torch.set_default_dtype(torch.float32)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"DEVICE CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    print(f"{'='*70}\n")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['device'] = str(device)
    
    # Setup directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_dir = Path(args.log_dir) / f'comparison_dualdrug_{timestamp}'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("QUANTUM vs CLASSICAL COMPARISON (DUAL DRUG)")
    print("="*70)
    print(f"Configuration:")
    print(f"  State dim: {args.state_dim}")
    print(f"  Action dim: {args.action_dim} (propofol + remifentanil)")
    print(f"  Online episodes: {args.online_episodes}")
    print(f"  Test episodes: {args.n_test_episodes}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Log dir: {comparison_dir}")
    print("="*70)
    
    # ========================================
    # Train Quantum Agent (Dual Drug)
    # ========================================
    print("\n" + "="*70)
    print("TRAINING QUANTUM AGENT (DUAL DRUG)")
    print("="*70)
    
    quantum_agent = QuantumDDPGAgent(
        state_dim=args.state_dim,   # 10D state
        action_dim=args.action_dim,  # 2D action
        config=config,
        encoder_type=args.encoder,
        seed=args.seed
    )
    
    # Move to device
    quantum_agent.actor = quantum_agent.actor.to(device)
    quantum_agent.actor_target = quantum_agent.actor_target.to(device)
    quantum_agent.critic = quantum_agent.critic.to(device)
    quantum_agent.critic_target = quantum_agent.critic_target.to(device)
    if quantum_agent.encoder is not None:
        quantum_agent.encoder = quantum_agent.encoder.to(device)
        quantum_agent.encoder_target = quantum_agent.encoder_target.to(device)
    
    quantum_dirs = {
        'base': comparison_dir / 'quantum',
        'checkpoints': comparison_dir / 'quantum' / 'checkpoints',
        'stage2': comparison_dir / 'quantum' / 'stage2_online',
    }
    for d in quantum_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # Online training on dual drug simulator
    print(f"\nOnline training on dual drug simulator...")
    print(f"  Episodes: {args.online_episodes}")
    print(f"  Warmup: {args.warmup_episodes}")
    
    env = DualDrugEnv(seed=args.seed)
    
    for episode in tqdm(range(args.online_episodes), desc="Quantum Online Training"):
        state, _ = env.reset()
        done = False
        states_history = []
        
        add_noise = episode >= args.warmup_episodes
        
        while not done:
            # Select action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if quantum_agent.encoder is not None and len(states_history) > 0:
                    recent_states = states_history[-19:] if len(states_history) >= 19 else states_history
                    states_array = np.array(recent_states + [state], dtype=np.float32)
                    states_seq = torch.FloatTensor(states_array).unsqueeze(0).to(device)
                    encoded = quantum_agent.encoder(states_seq)
                    if isinstance(encoded, tuple):
                        encoded = encoded[0]
                    action = quantum_agent.actor(encoded[:, -1, :])
                else:
                    action = quantum_agent.actor(state_tensor)
            
            action_np = action.cpu().numpy().flatten()
            
            # Add exploration noise
            if add_noise:
                noise = quantum_agent.noise()
                action_np = action_np + noise[:args.action_dim]
                action_np = np.clip(action_np, 0, 1)
            
            # Step
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            # Train
            quantum_agent.train_step(state, action_np, reward, next_state, done)
            
            states_history.append(state)
            state = next_state
        
        quantum_agent.decay_noise()
    
    # Save
    quantum_agent.save(str(quantum_dirs['checkpoints'] / 'final.pt'))
    print(f"✓ Quantum agent trained and saved")
    
    # ========================================
    # Train Classical Agent (Dual Drug)
    # ========================================
    print("\n" + "="*70)
    print("TRAINING CLASSICAL AGENT (DUAL DRUG)")
    print("="*70)
    
    classical_agent = ClassicalDDPGAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        config=config,
        encoder_type=args.encoder,
        seed=args.seed + 1000
    )
    
    # Move to device
    classical_agent.actor = classical_agent.actor.to(device)
    classical_agent.actor_target = classical_agent.actor_target.to(device)
    classical_agent.critic = classical_agent.critic.to(device)
    classical_agent.critic_target = classical_agent.critic_target.to(device)
    if classical_agent.encoder is not None:
        classical_agent.encoder = classical_agent.encoder.to(device)
        classical_agent.encoder_target = classical_agent.encoder_target.to(device)
    
    classical_dirs = {
        'base': comparison_dir / 'classical',
        'checkpoints': comparison_dir / 'classical' / 'checkpoints',
        'stage2': comparison_dir / 'classical' / 'stage2_online',
    }
    for d in classical_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # Online training
    print(f"\nOnline training on dual drug simulator...")
    
    env = DualDrugEnv(seed=args.seed + 1000)
    
    for episode in tqdm(range(args.online_episodes), desc="Classical Online Training"):
        state, _ = env.reset()
        done = False
        states_history = []
        
        add_noise = episode >= args.warmup_episodes
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if classical_agent.encoder is not None and len(states_history) > 0:
                    recent_states = states_history[-19:] if len(states_history) >= 19 else states_history
                    states_array = np.array(recent_states + [state], dtype=np.float32)
                    states_seq = torch.FloatTensor(states_array).unsqueeze(0).to(device)
                    encoded = classical_agent.encoder(states_seq)
                    if isinstance(encoded, tuple):
                        encoded = encoded[0]
                    action = classical_agent.actor(encoded[:, -1, :])
                else:
                    action = classical_agent.actor(state_tensor)
            
            action_np = action.cpu().numpy().flatten()
            
            if add_noise:
                noise = classical_agent.noise()
                action_np = action_np + noise[:args.action_dim]
                action_np = np.clip(action_np, 0, 1)
            
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            classical_agent.train_step(state, action_np, reward, next_state, done)
            
            states_history.append(state)
            state = next_state
        
        classical_agent.decay_noise()
    
    classical_agent.save(str(classical_dirs['checkpoints'] / 'final.pt'))
    print(f"✓ Classical agent trained and saved")
    
    # ========================================
    # Evaluation
    # ========================================
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    # Evaluate on dual drug simulator
    print(f"\nEvaluating Quantum agent...")
    quantum_results = evaluate_agent_on_simulator_dualdrug(
        quantum_agent, args.n_test_episodes, args.seed + 2000, device
    )
    
    print(f"\nEvaluating Classical agent...")
    classical_results = evaluate_agent_on_simulator_dualdrug(
        classical_agent, args.n_test_episodes, args.seed + 3000, device
    )
    
    # Statistical comparison
    print(f"\nPerforming statistical tests...")
    stats_results = statistical_comparison(quantum_results, classical_results, args.alpha)
    
    # Generate plots
    print(f"\nGenerating comparison plots...")
    plot_dualdrug_comparison(
        quantum_results, classical_results,
        'Dual Drug Control: Quantum vs Classical',
        comparison_dir / 'dualdrug_comparison.png'
    )
    
    # Save results
    results = {
        'quantum': quantum_results,
        'classical': classical_results,
        'statistics': stats_results,
        'config': vars(args)
    }
    
    with open(comparison_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Generate summary report
    report_path = comparison_dir / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DUAL DRUG CONTROL: QUANTUM vs CLASSICAL RL COMPARISON\n")
        f.write("="*70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"State dimension: {args.state_dim}\n")
        f.write(f"Action dimension: {args.action_dim} (propofol + remifentanil)\n")
        f.write(f"Online episodes: {args.online_episodes}\n")
        f.write(f"Test episodes: {args.n_test_episodes}\n")
        f.write(f"Encoder: {args.encoder}\n")
        f.write(f"Seed: {args.seed}\n\n")
        
        f.write("RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"MDAPE:\n")
        f.write(f"  Quantum:   {quantum_results['mdape_mean']:.2f} ± {quantum_results['mdape_std']:.2f}%\n")
        f.write(f"  Classical: {classical_results['mdape_mean']:.2f} ± {classical_results['mdape_std']:.2f}%\n\n")
        
        f.write(f"Reward:\n")
        f.write(f"  Quantum:   {quantum_results['reward_mean']:.2f} ± {quantum_results['reward_std']:.2f}\n")
        f.write(f"  Classical: {classical_results['reward_mean']:.2f} ± {classical_results['reward_std']:.2f}\n\n")
        
        f.write(f"Time in Target (BIS 45-55):\n")
        f.write(f"  Quantum:   {quantum_results['time_in_target_mean']:.1f}%\n")
        f.write(f"  Classical: {classical_results['time_in_target_mean']:.1f}%\n\n")
        
        f.write(f"Drug Usage:\n")
        f.write(f"  Propofol:\n")
        f.write(f"    Quantum:   {quantum_results['propofol_usage_mean']:.2f} ± {quantum_results['propofol_usage_std']:.2f} mg/kg/h\n")
        f.write(f"    Classical: {classical_results['propofol_usage_mean']:.2f} ± {classical_results['propofol_usage_std']:.2f} mg/kg/h\n")
        f.write(f"  Remifentanil:\n")
        f.write(f"    Quantum:   {quantum_results['remifentanil_usage_mean']:.2f} ± {quantum_results['remifentanil_usage_std']:.2f} μg/kg/min\n")
        f.write(f"    Classical: {classical_results['remifentanil_usage_mean']:.2f} ± {classical_results['remifentanil_usage_std']:.2f} μg/kg/min\n\n")
        
        f.write("STATISTICAL TESTS\n")
        f.write("-"*70 + "\n")
        f.write(f"Winner: {stats_results['winner']}\n")
        f.write(f"Improvement: {stats_results['improvement_pct']:.2f}%\n")
        f.write(f"Mean difference: {stats_results['mean_difference']:.2f}%\n")
        f.write(f"95% CI: [{stats_results['ci_95'][0]:.2f}, {stats_results['ci_95'][1]:.2f}]\n\n")
        
        f.write(f"T-test:\n")
        f.write(f"  t-statistic: {stats_results['t_statistic']:.4f}\n")
        f.write(f"  p-value: {stats_results['t_pvalue']:.4f}\n")
        f.write(f"  Significant: {'YES' if stats_results['t_significant'] else 'NO'} (α={args.alpha})\n\n")
        
        f.write(f"Mann-Whitney U test:\n")
        f.write(f"  U-statistic: {stats_results['u_statistic']:.4f}\n")
        f.write(f"  p-value: {stats_results['u_pvalue']:.4f}\n")
        f.write(f"  Significant: {'YES' if stats_results['u_significant'] else 'NO'} (α={args.alpha})\n\n")
        
        f.write(f"Effect size (Cohen's d): {stats_results['cohens_d']:.4f}\n")
        f.write("  (0.2=small, 0.5=medium, 0.8=large)\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"✓ Summary report saved to: {report_path}")
    
    # Print summary to console
    print("\n" + "="*70)
    print("RESULTS SUMMARY (DUAL DRUG)")
    print("="*70)
    print(f"\nMDAPE:")
    print(f"  Quantum:   {quantum_results['mdape_mean']:.2f} ± {quantum_results['mdape_std']:.2f}%")
    print(f"  Classical: {classical_results['mdape_mean']:.2f} ± {classical_results['mdape_std']:.2f}%")
    print(f"  Winner: {stats_results['winner']} ({stats_results['improvement_pct']:.2f}% improvement)")
    print(f"  Significant: {'YES' if stats_results['t_significant'] else 'NO'} (p={stats_results['t_pvalue']:.4f})")
    
    print(f"\nTime in Target (BIS 45-55):")
    print(f"  Quantum:   {quantum_results['time_in_target_mean']:.1f}%")
    print(f"  Classical: {classical_results['time_in_target_mean']:.1f}%")
    
    print(f"\nPropofol usage:")
    print(f"  Quantum:   {quantum_results['propofol_usage_mean']:.2f} ± {quantum_results['propofol_usage_std']:.2f} mg/kg/h")
    print(f"  Classical: {classical_results['propofol_usage_mean']:.2f} ± {classical_results['propofol_usage_std']:.2f} mg/kg/h")
    
    print(f"\nRemifentanil usage:")
    print(f"  Quantum:   {quantum_results['remifentanil_usage_mean']:.2f} ± {quantum_results['remifentanil_usage_std']:.2f} μg/kg/min")
    print(f"  Classical: {classical_results['remifentanil_usage_mean']:.2f} ± {classical_results['remifentanil_usage_std']:.2f} μg/kg/min")
    
    print(f"\n✓ All results saved to: {comparison_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
