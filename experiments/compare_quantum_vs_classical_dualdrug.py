"""
Quantum vs Classical RL Comparison for Dual Drug Control
=========================================================

Compare Quantum and Classical agents on dual drug environment
(Propofol + Remifentanil).

Key differences from single drug comparison:
- Action space: [propofol_rate, remifentanil_rate] (2D)
- State space: Extended to include remifentanil Ce + demographics (13D)
- Agents: action_dim=2, n_qubits=3 (increased for 13D state)

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
import csv
import pandas as pd

# Use relative imports (we're in experiments/ directory)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.quantum_agent import QuantumDDPGAgent
from src.agents.classical_agent import ClassicalDDPGAgent
from src.environment.dual_drug_env import DualDrugEnv
from src.environment.patient_simulator import create_patient_population
from src.data.vitaldb_loader import VitalDBLoader
from train_hybrid import stage1_offline_pretraining, stage2_online_finetuning


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare Quantum vs Classical RL for Dual Drug Control'
    )
    
    # VitalDB data
    parser.add_argument('--n_cases', type=int, default=1000,
                       help='Number of VitalDB dual drug cases to load')
    parser.add_argument('--sampling_interval', type=int, default=1,
                       help='Sampling interval for VitalDB data (1=all data, 5=every 5 seconds)')
    parser.add_argument('--offline_epochs', type=int, default=100,
                       help='Number of offline pre-training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for offline training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of DataLoader workers for parallel loading')
    parser.add_argument('--bc_weight', type=float, default=1.0,
                       help='Behavioral cloning weight')
    
    # CQL (Conservative Q-Learning) parameters
    parser.add_argument('--use_cql', action='store_true',
                       help='Use CQL instead of standard off-policy RL')
    parser.add_argument('--cql_alpha', type=float, default=1.0,
                       help='CQL penalty weight')
    parser.add_argument('--cql_temp', type=float, default=1.0,
                       help='Temperature for CQL logsumexp')
    parser.add_argument('--cql_num_random', type=int, default=5,
                       help='Number of random actions for CQL penalty')
    parser.add_argument('--cql_warmup_epochs', type=int, default=50,
                       help='Number of epochs to use CQL penalty (after this, only BC+RL)')
    
    # Training
    parser.add_argument('--online_episodes', type=int, default=200,
                       help='Number of online training episodes')
    parser.add_argument('--warmup_episodes', type=int, default=50,
                       help='Warmup episodes before exploration')
    
    # Agent (dual drug specific)
    parser.add_argument('--state_dim', type=int, default=13,
                       help='State dimension for dual drug (extended with demographics: age, sex, BMI)')
    parser.add_argument('--action_dim', type=int, default=2,
                       help='Action dimension (propofol + remifentanil)')
    parser.add_argument('--n_qubits', type=int, default=3,
                       help='Number of qubits for quantum circuit (3 qubits for 13D state)')
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
    - State is 13D (includes remifentanil Ce + patient demographics)
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
                    # Handle both 2D and 3D encoder outputs
                    if encoded.dim() == 3:
                        encoded = encoded[:, -1, :]
                    action = agent.actor(encoded)
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
        
        # Get metrics if available
        if hasattr(env, 'get_episode_metrics'):
            metrics = env.get_episode_metrics()
            mdapes.append(metrics.get('mdape', 0))
        else:
            # Calculate MDAPE manually from BIS error if metrics not available
            if bis_history:
                bis_array = np.array(bis_history)
                target_bis = 50
                mdape = np.mean(np.abs(bis_array - target_bis) / target_bis) * 100
                mdapes.append(mdape)
            else:
                mdapes.append(0)
        
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


def save_results_to_csv(
    quantum_results: Dict,
    classical_results: Dict,
    stats_results: Dict,
    save_dir: Path
) -> None:
    """
    Save comparison results to CSV files for later analysis.
    
    Args:
        quantum_results: Quantum agent evaluation results
        classical_results: Classical agent evaluation results
        stats_results: Statistical comparison results
        save_dir: Directory to save CSV files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary statistics CSV
    summary_data = {
        'Agent': ['Quantum', 'Classical'],
        'MDAPE_mean': [quantum_results['mdape_mean'], classical_results['mdape_mean']],
        'MDAPE_std': [quantum_results['mdape_std'], classical_results['mdape_std']],
        'Reward_mean': [quantum_results['reward_mean'], classical_results['reward_mean']],
        'Reward_std': [quantum_results['reward_std'], classical_results['reward_std']],
        'Propofol_mean': [quantum_results['propofol_usage_mean'], classical_results['propofol_usage_mean']],
        'Propofol_std': [quantum_results['propofol_usage_std'], classical_results['propofol_usage_std']],
        'Remifentanil_mean': [quantum_results['remifentanil_usage_mean'], classical_results['remifentanil_usage_mean']],
        'Remifentanil_std': [quantum_results['remifentanil_usage_std'], classical_results['remifentanil_usage_std']],
        'TimeInTarget_mean': [quantum_results['time_in_target_mean'], classical_results['time_in_target_mean']],
        'TimeInTarget_std': [quantum_results['time_in_target_std'], classical_results['time_in_target_std']],
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(save_dir / 'summary_statistics.csv', index=False)
    print(f"✓ Saved summary statistics to {save_dir / 'summary_statistics.csv'}")
    
    # 2. Episode-by-episode results CSV
    n_episodes = len(quantum_results['mdape_list'])
    episode_data = {
        'Episode': list(range(1, n_episodes + 1)),
        'Quantum_MDAPE': quantum_results['mdape_list'],
        'Classical_MDAPE': classical_results['mdape_list'],
        'Quantum_Reward': quantum_results['reward_list'],
        'Classical_Reward': classical_results['reward_list'],
    }
    df_episodes = pd.DataFrame(episode_data)
    df_episodes.to_csv(save_dir / 'episode_results.csv', index=False)
    print(f"✓ Saved episode results to {save_dir / 'episode_results.csv'}")
    
    # 3. Statistical test results CSV
    stats_data = {
        'Test': ['t-test', 'Mann-Whitney U', 'Cohen\'s d', 'Mean Difference', 'Winner'],
        'Statistic': [
            stats_results['t_statistic'],
            stats_results['u_statistic'],
            stats_results['cohens_d'],
            stats_results['mean_difference'],
            0
        ],
        'P-value': [
            stats_results['t_pvalue'],
            stats_results['u_pvalue'],
            np.nan,
            np.nan,
            np.nan
        ],
        'Significant': [
            stats_results['t_significant'],
            stats_results['u_significant'],
            np.nan,
            np.nan,
            np.nan
        ],
        'Notes': [
            f"Quantum vs Classical MDAPE",
            f"Non-parametric test",
            f"Effect size: {stats_results['cohens_d']:.3f}",
            f"Difference: {stats_results['mean_difference']:.2f}% (95% CI: [{stats_results['ci_95'][0]:.2f}, {stats_results['ci_95'][1]:.2f}])",
            f"{stats_results['winner']} wins by {abs(stats_results['improvement_pct']):.1f}%"
        ]
    }
    df_stats = pd.DataFrame(stats_data)
    df_stats.to_csv(save_dir / 'statistical_tests.csv', index=False)
    print(f"✓ Saved statistical tests to {save_dir / 'statistical_tests.csv'}")
    
    # 4. Metadata CSV (experiment configuration)
    metadata = {
        'Parameter': ['Timestamp', 'Episodes', 'Mean_Improvement_%', 'Winner', 'Significant'],
        'Value': [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            n_episodes,
            f"{stats_results['improvement_pct']:.2f}",
            stats_results['winner'],
            'Yes' if stats_results['t_significant'] else 'No'
        ]
    }
    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(save_dir / 'experiment_metadata.csv', index=False)
    print(f"✓ Saved metadata to {save_dir / 'experiment_metadata.csv'}")


def statistical_comparison(quantum_results: Dict, classical_results: Dict, 
                          alpha: float = 0.05) -> Dict:
    """Perform statistical significance tests."""
    
    # Check if we have enough samples for statistical tests
    n_samples = len(quantum_results['mdape_list'])
    
    if n_samples < 2:
        print(f"⚠️  Warning: Only {n_samples} sample(s). Statistical tests require at least 2 samples.")
        print(f"   Skipping statistical tests and using simple comparison.")
        
        mean_diff = quantum_results['mdape_mean'] - classical_results['mdape_mean']
        improvement_pct = (classical_results['mdape_mean'] - quantum_results['mdape_mean']) / \
                         (classical_results['mdape_mean'] + 1e-8) * 100
        
        return {
            't_statistic': np.nan,
            't_pvalue': np.nan,
            't_significant': False,
            'u_statistic': np.nan,
            'u_pvalue': np.nan,
            'u_significant': False,
            'cohens_d': np.nan,
            'mean_difference': mean_diff,
            'ci_95': (np.nan, np.nan),
            'winner': 'Quantum' if mean_diff < 0 else 'Classical',
            'improvement_pct': improvement_pct,
            'note': f'Insufficient samples (n={n_samples}) for statistical tests'
        }
    
    # Two-sample t-test
    try:
        t_stat, t_pval = stats.ttest_ind(
            quantum_results['mdape_list'],
            classical_results['mdape_list']
        )
    except Exception as e:
        print(f"⚠️  Warning: t-test failed ({e}). Using NaN.")
        t_stat, t_pval = np.nan, np.nan
    
    # Mann-Whitney U test (non-parametric)
    try:
        u_stat, u_pval = stats.mannwhitneyu(
            quantum_results['mdape_list'],
            classical_results['mdape_list'],
            alternative='two-sided'
        )
    except Exception as e:
        print(f"⚠️  Warning: Mann-Whitney U test failed ({e}). Using NaN.")
        u_stat, u_pval = np.nan, np.nan
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.var(quantum_results['mdape_list']) + np.var(classical_results['mdape_list'])) / 2
    )
    # Add epsilon to prevent divide by zero
    epsilon = 1e-8
    cohens_d = (quantum_results['mdape_mean'] - classical_results['mdape_mean']) / (pooled_std + epsilon)
    
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


def evaluate_on_vitaldb_test_set(
    agent,
    test_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device
) -> Dict:
    """
    Evaluate agent on VitalDB test set.
    
    Args:
        agent: Quantum or Classical agent
        test_data: (states, actions, next_states) from VitalDB
        device: torch device
    
    Returns:
        Dictionary with evaluation metrics
    """
    states, actions, next_states = test_data
    
    # Predict actions
    predicted_actions = []
    
    for state in tqdm(states, desc="VitalDB Test Set Evaluation"):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if hasattr(agent, 'encoder') and agent.encoder is not None:
                # Create single-step sequence for encoder
                state_seq = state_tensor.unsqueeze(0)  # [1, 1, state_dim]
                
                encoded = agent.encoder(state_seq)
                if isinstance(encoded, tuple):
                    encoded = encoded[0]  # Take output, ignore hidden state
                
                # Handle both 2D and 3D encoder outputs
                if encoded.dim() == 3:
                    encoded = encoded[:, -1, :]  # Get last timestep [batch, encoded_dim]
                
                action = agent.actor(encoded)
            else:
                action = agent.actor(state_tensor)
        
        predicted_actions.append(action.cpu().numpy().flatten())
    
    predicted_actions = np.array(predicted_actions)
    
    # Ensure 2D actions
    if predicted_actions.shape[1] != 2:
        print(f"Warning: Expected 2D actions, got shape {predicted_actions.shape}")
        # Pad or truncate to 2D
        if predicted_actions.shape[1] < 2:
            predicted_actions = np.pad(predicted_actions, ((0, 0), (0, 2 - predicted_actions.shape[1])))
        else:
            predicted_actions = predicted_actions[:, :2]
    
    # Compute MDAPE for each drug
    ppf_error = np.abs(predicted_actions[:, 0] - actions[:, 0]) / (np.abs(actions[:, 0]) + 1e-6)
    rftn_error = np.abs(predicted_actions[:, 1] - actions[:, 1]) / (np.abs(actions[:, 1]) + 1e-6)
    
    mdape_ppf = np.median(ppf_error) * 100
    mdape_rftn = np.median(rftn_error) * 100
    mdape_avg = (mdape_ppf + mdape_rftn) / 2
    
    return {
        'mdape_mean': mdape_avg,
        'mdape_std': np.std([mdape_ppf, mdape_rftn]),
        'mdape_propofol': mdape_ppf,
        'mdape_remifentanil': mdape_rftn,
        'mdape_list': [mdape_ppf, mdape_rftn]
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
    
    # Check if we have enough data for boxplot (need at least 2 points)
    use_boxplot = len(quantum_results['mdape_list']) >= 2
    
    # 1. MDAPE Box Plot (or Bar Plot for small datasets)
    ax = axes[0, 0]
    if use_boxplot:
        data = [quantum_results['mdape_list'], classical_results['mdape_list']]
        bp = ax.boxplot(data, tick_labels=['Quantum', 'Classical'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.plot(1, quantum_results['mdape_mean'], 'D', color='blue', markersize=8, label='Mean')
        ax.plot(2, classical_results['mdape_mean'], 'D', color='red', markersize=8)
        ax.legend()
    else:
        # Use bar plot for small datasets
        x_pos = [1, 2]
        means = [quantum_results['mdape_mean'], classical_results['mdape_mean']]
        stds = [quantum_results['mdape_std'], classical_results['mdape_std']]
        ax.bar(x_pos, means, yerr=stds, capsize=5, color=['lightblue', 'lightcoral'], alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Quantum', 'Classical'])
    ax.set_ylabel('MDAPE (%)', fontsize=12)
    ax.set_title('MDAPE Distribution', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Reward Box Plot (or Bar Plot)
    ax = axes[0, 1]
    if use_boxplot:
        data = [quantum_results['reward_list'], classical_results['reward_list']]
        bp = ax.boxplot(data, tick_labels=['Quantum', 'Classical'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.plot(1, quantum_results['reward_mean'], 'D', color='blue', markersize=8, label='Mean')
        ax.plot(2, classical_results['reward_mean'], 'D', color='red', markersize=8)
        ax.legend()
    else:
        x_pos = [1, 2]
        means = [quantum_results['reward_mean'], classical_results['reward_mean']]
        stds = [quantum_results['reward_std'], classical_results['reward_std']]
        ax.bar(x_pos, means, yerr=stds, capsize=5, color=['lightblue', 'lightcoral'], alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Quantum', 'Classical'])
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Reward Distribution', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
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
    if use_boxplot:
        # Only create histogram if we have enough data
        n_bins = min(20, len(quantum_results['mdape_list']))
        ax.hist(quantum_results['mdape_list'], bins=n_bins, alpha=0.6, label='Quantum', color='blue')
        ax.hist(classical_results['mdape_list'], bins=n_bins, alpha=0.6, label='Classical', color='red')
        ax.axvline(quantum_results['mdape_mean'], color='blue', linestyle='--', linewidth=2)
        ax.axvline(classical_results['mdape_mean'], color='red', linestyle='--', linewidth=2)
        ax.legend()
    else:
        # For single data point, show as scatter
        ax.scatter([quantum_results['mdape_mean']], [1], s=200, alpha=0.6, label='Quantum', color='blue')
        ax.scatter([classical_results['mdape_mean']], [1], s=200, alpha=0.6, label='Classical', color='red')
        ax.set_ylim(0, 2)
        ax.legend()
    ax.set_xlabel('MDAPE (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('MDAPE Histogram', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 5. Cumulative Distribution
    ax = axes[1, 1]
    if use_boxplot:
        quantum_sorted = np.sort(quantum_results['mdape_list'])
        classical_sorted = np.sort(classical_results['mdape_list'])
        quantum_cdf = np.arange(1, len(quantum_sorted) + 1) / len(quantum_sorted)
        classical_cdf = np.arange(1, len(classical_sorted) + 1) / len(classical_sorted)
        
        ax.plot(quantum_sorted, quantum_cdf, label='Quantum', color='blue', linewidth=2)
        ax.plot(classical_sorted, classical_cdf, label='Classical', color='red', linewidth=2)
        ax.legend()
    else:
        # For single data point, show as step function
        ax.scatter([quantum_results['mdape_mean']], [1.0], s=200, label='Quantum', color='blue', marker='o')
        ax.scatter([classical_results['mdape_mean']], [1.0], s=200, label='Classical', color='red', marker='s')
        ax.set_ylim(0, 1.2)
        ax.legend()
    ax.set_xlabel('MDAPE (%)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
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
    
    # Override config for dual drug control
    config['state_dim'] = args.state_dim
    config['action_dim'] = args.action_dim
    config['quantum']['n_qubits'] = args.n_qubits
    print(f"\n🔧 Config overrides for dual drug:")
    print(f"  state_dim: {config['state_dim']}")
    print(f"  action_dim: {config['action_dim']}")
    print(f"  n_qubits: {config['quantum']['n_qubits']}")
    
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
    print(f"  Quantum qubits: {args.n_qubits}")
    print(f"  VitalDB cases: {args.n_cases}")
    print(f"  Offline epochs: {args.offline_epochs}")
    print(f"  Online episodes: {args.online_episodes}")
    print(f"  Test episodes: {args.n_test_episodes}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Log dir: {comparison_dir}")
    print("="*70 + "\n")
    
    # ========================================
    # Load VitalDB Dual Drug Data
    # ========================================
    print("\n" + "="*70)
    print("LOADING VITALDB DUAL DRUG DATA")
    print("="*70)
    
    loader = VitalDBLoader(
        cache_dir='data/vitaldb_cache',
        use_cache=True
    )
    
    print(f"\nLoading {args.n_cases} dual drug cases from VitalDB...")
    print(f"Sampling interval: {args.sampling_interval}s (1=all data, higher=faster)")
    
    # Try loading with more relaxed criteria if n_cases is small
    min_duration = 300 if args.n_cases <= 10 else 1800  # 5 min for testing, 30 min for training
    print(f"Using min_duration: {min_duration}s ({min_duration/60:.1f} min)")
    
    states, actions, next_states, rewards, dones = loader.prepare_dual_drug_training_data(
        n_cases=args.n_cases,
        min_duration=min_duration,
        sampling_interval=args.sampling_interval,
        save_path=comparison_dir / 'vitaldb_dual_drug.pkl'
    )
    
    # Debug: Print raw data info
    print(f"\n🔍 DEBUG - Raw data info:")
    print(f"  states type: {type(states)}, shape: {states.shape if hasattr(states, 'shape') else 'N/A'}")
    print(f"  actions type: {type(actions)}, shape: {actions.shape if hasattr(actions, 'shape') else 'N/A'}")
    print(f"  next_states type: {type(next_states)}, shape: {next_states.shape if hasattr(next_states, 'shape') else 'N/A'}")
    print(f"  rewards type: {type(rewards)}, shape: {rewards.shape if hasattr(rewards, 'shape') else 'N/A'}")
    print(f"  dones type: {type(dones)}, shape: {dones.shape if hasattr(dones, 'shape') else 'N/A'}")
    
    # Check if data is empty
    if len(states) == 0:
        raise ValueError(f"No data loaded! VitalDB loader returned empty arrays. "
                        f"Check if cases with dual drug data exist and meet the criteria.")
    
    # Validate dimensions with better error messages
    if len(states.shape) != 2:
        raise ValueError(f"Expected states to be 2D array, got shape {states.shape}. "
                        f"This might indicate data loading failed.")
    
    if states.shape[1] != args.state_dim:
        raise ValueError(f"State dimension mismatch! Expected {args.state_dim}, got {states.shape[1]}. "
                        f"Make sure VitalDB loader's _extract_dual_drug_state() returns 13D state.")
    
    if actions.shape[1] != args.action_dim:
        raise ValueError(f"Action dimension mismatch! Expected {args.action_dim}, got {actions.shape[1]}. "
                        f"Dual drug should have 2D actions [propofol, remifentanil].")
    
    if next_states.shape[1] != args.state_dim:
        raise ValueError(f"Next state dimension mismatch! Expected {args.state_dim}, got {next_states.shape[1]}.")
    
    print(f"\n✓ Loaded dual drug data:")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Next states shape: {next_states.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Rewards range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    print(f"  Rewards mean: {rewards.mean():.3f} ± {rewards.std():.3f}")
    print(f"  Dones shape: {dones.shape}")
    print(f"  Episodes completed: {dones.sum()}")
    
    # Split data (80/10/10)
    n_total = len(states)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_data = (states[train_indices], actions[train_indices], next_states[train_indices])
    val_data = (states[val_indices], actions[val_indices], next_states[val_indices])
    test_data = (states[test_indices], actions[test_indices], next_states[test_indices])
    
    # Convert to dict format for stage1_offline_pretraining
    # Use REAL rewards and dones from VitalDB data
    train_data_dict = {
        'states': states[train_indices], 
        'actions': actions[train_indices], 
        'next_states': next_states[train_indices],
        'rewards': rewards[train_indices],
        'dones': dones[train_indices]
    }
    val_data_dict = {
        'states': states[val_indices], 
        'actions': actions[val_indices], 
        'next_states': next_states[val_indices],
        'rewards': rewards[val_indices],
        'dones': dones[val_indices]
    }
    
    # Debug: Print shapes
    print(f"\nTraining data shapes:")
    print(f"  States: {train_data_dict['states'].shape}")
    print(f"  Actions: {train_data_dict['actions'].shape}")
    print(f"  Next states: {train_data_dict['next_states'].shape}")
    print(f"  Rewards: {train_data_dict['rewards'].shape}")
    print(f"  Dones: {train_data_dict['dones'].shape}")
    
    print(f"\nData split:")
    print(f"  Train: {len(train_indices):,} samples")
    print(f"  Val: {len(val_indices):,} samples")
    print(f"  Test: {len(test_indices):,} samples")
    print("="*70)
    
    # ========================================
    # Train Quantum Agent (2-Stage)
    # ========================================
    print("\n" + "="*70)
    print("TRAINING QUANTUM AGENT (2-STAGE: OFFLINE + ONLINE)")
    print("="*70)
    
    quantum_agent = QuantumDDPGAgent(
        state_dim=args.state_dim,   # 10D state
        action_dim=args.action_dim,  # 2D action
        config=config,
        encoder_type=args.encoder,
        seed=args.seed
    )
    
    # Move to device FIRST
    quantum_agent.actor = quantum_agent.actor.to(device)
    quantum_agent.actor_target = quantum_agent.actor_target.to(device)
    quantum_agent.critic = quantum_agent.critic.to(device)
    quantum_agent.critic_target = quantum_agent.critic_target.to(device)
    if quantum_agent.encoder is not None:
        quantum_agent.encoder = quantum_agent.encoder.to(device)
        quantum_agent.encoder_target = quantum_agent.encoder_target.to(device)
    
    # DEBUG: Check actor output dimension (after moving to device)
    test_state = torch.randn(1, args.state_dim).to(device)
    with torch.no_grad():
        # Use encoder if available
        if quantum_agent.encoder is not None:
            # For LSTM/Transformer, we need a sequence
            test_state_seq = test_state.unsqueeze(0)  # [1, seq_len=1, state_dim]
            test_encoded = quantum_agent.encoder(test_state_seq)
            if isinstance(test_encoded, tuple):  # LSTM returns (output, hidden)
                test_encoded = test_encoded[0]
            # Get last timestep output
            test_encoded = test_encoded[:, -1, :] if test_encoded.dim() == 3 else test_encoded
            test_action = quantum_agent.actor(test_encoded)
        else:
            test_action = quantum_agent.actor(test_state)
    print(f"\n🔍 DEBUG - Quantum Actor Check:")
    print(f"  Input state shape: {test_state.shape}")
    if quantum_agent.encoder is not None:
        print(f"  Encoded state shape: {test_encoded.shape}")
        print(f"  Expected encoded shape: torch.Size([1, {quantum_agent.encoded_dim}])")
    print(f"  Output action shape: {test_action.shape}")
    print(f"  Expected action shape: torch.Size([1, {args.action_dim}])")
    
    if test_action.shape[1] != args.action_dim:
        raise ValueError(
            f"Actor output dimension mismatch! "
            f"Expected {args.action_dim}D actions, got {test_action.shape[1]}D. "
            f"Check agent initialization and config."
        )
    
    quantum_dirs = {
        'base': comparison_dir / 'quantum',
        'checkpoints': comparison_dir / 'quantum' / 'checkpoints',
        'stage1': comparison_dir / 'quantum' / 'stage1_offline',
        'stage2': comparison_dir / 'quantum' / 'stage2_online',
    }
    for d in quantum_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Offline Pre-training on VitalDB
    print("\n" + "-"*70)
    print("STAGE 1: OFFLINE PRE-TRAINING (VitalDB Dual Drug)")
    print("-"*70)
    
    quantum_agent = stage1_offline_pretraining(
        agent=quantum_agent,
        train_data=train_data_dict,
        val_data=val_data_dict,
        n_epochs=args.offline_epochs,
        batch_size=args.batch_size,
        bc_weight=args.bc_weight,
        use_cql=args.use_cql,
        cql_alpha=args.cql_alpha,
        cql_temp=args.cql_temp,
        cql_num_random=args.cql_num_random,
        cql_warmup_epochs=args.cql_warmup_epochs,
        num_workers=args.num_workers,
        dirs={'stage1': quantum_dirs['stage1']},
        device=device
    )
    
    print(f"✓ Stage 1 complete: Quantum agent pre-trained on VitalDB dual drug data")
    
    # Stage 2: Online Fine-tuning on Simulator
    print("\n" + "-"*70)
    print("STAGE 2: ONLINE FINE-TUNING (Dual Drug Simulator)")
    print("-"*70)
    
    quantum_agent = stage2_online_finetuning(
        agent=quantum_agent,
        n_episodes=args.online_episodes,
        warmup_episodes=args.warmup_episodes,
        dirs={'stage2': quantum_dirs['stage2']},
        eval_interval=50,
        seed=args.seed,
        device=device
    )
    
    print(f"✓ Stage 2 complete: Quantum agent fine-tuned on simulator")
    
    # Save
    quantum_agent.save(str(quantum_dirs['checkpoints'] / 'final.pt'))
    print(f"✓ Quantum agent trained and saved")
    
    # ========================================
    # Train Classical Agent (2-Stage)
    # ========================================
    print("\n" + "="*70)
    print("TRAINING CLASSICAL AGENT (2-STAGE: OFFLINE + ONLINE)")
    print("="*70)
    
    classical_agent = ClassicalDDPGAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        config=config,
        encoder_type=args.encoder,
        seed=args.seed + 1000
    )
    
    # Move to device FIRST
    classical_agent.actor = classical_agent.actor.to(device)
    classical_agent.actor_target = classical_agent.actor_target.to(device)
    classical_agent.critic = classical_agent.critic.to(device)
    classical_agent.critic_target = classical_agent.critic_target.to(device)
    if classical_agent.encoder is not None:
        classical_agent.encoder = classical_agent.encoder.to(device)
        classical_agent.encoder_target = classical_agent.encoder_target.to(device)
    
    # DEBUG: Check actor output dimension (after moving to device)
    test_state = torch.randn(1, args.state_dim).to(device)
    with torch.no_grad():
        # Use encoder if available
        if classical_agent.encoder is not None:
            # For LSTM/Transformer, we need a sequence
            test_state_seq = test_state.unsqueeze(0)  # [1, seq_len=1, state_dim]
            test_encoded = classical_agent.encoder(test_state_seq)
            if isinstance(test_encoded, tuple):  # LSTM returns (output, hidden)
                test_encoded = test_encoded[0]
            # Get last timestep output
            test_encoded = test_encoded[:, -1, :] if test_encoded.dim() == 3 else test_encoded
            test_action = classical_agent.actor(test_encoded)
        else:
            test_action = classical_agent.actor(test_state)
    print(f"\n🔍 DEBUG - Classical Actor Check:")
    print(f"  Input state shape: {test_state.shape}")
    if classical_agent.encoder is not None:
        print(f"  Encoded state shape: {test_encoded.shape}")
        print(f"  Expected encoded shape: torch.Size([1, {classical_agent.encoded_dim}])")
    print(f"  Output action shape: {test_action.shape}")
    print(f"  Expected action shape: torch.Size([1, {args.action_dim}])")
    
    if test_action.shape[1] != args.action_dim:
        raise ValueError(
            f"Actor output dimension mismatch! "
            f"Expected {args.action_dim}D actions, got {test_action.shape[1]}D. "
            f"Check agent initialization and config."
        )
    
    classical_dirs = {
        'base': comparison_dir / 'classical',
        'checkpoints': comparison_dir / 'classical' / 'checkpoints',
        'stage1': comparison_dir / 'classical' / 'stage1_offline',
        'stage2': comparison_dir / 'classical' / 'stage2_online',
    }
    for d in classical_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Offline Pre-training
    print("\n" + "-"*70)
    print("STAGE 1: OFFLINE PRE-TRAINING (VitalDB Dual Drug)")
    print("-"*70)
    
    classical_agent = stage1_offline_pretraining(
        agent=classical_agent,
        train_data=train_data_dict,
        val_data=val_data_dict,
        n_epochs=args.offline_epochs,
        batch_size=args.batch_size,
        bc_weight=args.bc_weight,
        use_cql=args.use_cql,
        cql_alpha=args.cql_alpha,
        cql_temp=args.cql_temp,
        cql_num_random=args.cql_num_random,
        cql_warmup_epochs=args.cql_warmup_epochs,
        num_workers=args.num_workers,
        dirs={'stage1': classical_dirs['stage1']},
        device=device
    )
    
    print(f"✓ Stage 1 complete: Classical agent pre-trained on VitalDB dual drug data")
    
    # Stage 2: Online Fine-tuning
    print("\n" + "-"*70)
    print("STAGE 2: ONLINE FINE-TUNING (Dual Drug Simulator)")
    print("-"*70)
    
    classical_agent = stage2_online_finetuning(
        agent=classical_agent,
        n_episodes=args.online_episodes,
        warmup_episodes=args.warmup_episodes,
        dirs={'stage2': classical_dirs['stage2']},
        eval_interval=50,
        seed=args.seed + 1000,
        device=device
    )
    
    print(f"✓ Stage 2 complete: Classical agent fine-tuned on simulator")
    
    # ========================================
    # Evaluation
    # ========================================
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    # 1. Evaluate on VitalDB test set
    print("\n" + "-"*70)
    print("EVALUATING ON VITALDB TEST SET")
    print("-"*70)
    
    print(f"\nEvaluating Quantum agent on VitalDB test set...")
    quantum_vitaldb_results = evaluate_on_vitaldb_test_set(
        quantum_agent, test_data, device
    )
    
    print(f"\nEvaluating Classical agent on VitalDB test set...")
    classical_vitaldb_results = evaluate_on_vitaldb_test_set(
        classical_agent, test_data, device
    )
    
    print(f"\n✓ VitalDB Test Set Results:")
    print(f"  Quantum  - MDAPE: {quantum_vitaldb_results['mdape_mean']:.2f}% "
          f"(PPF: {quantum_vitaldb_results['mdape_propofol']:.2f}%, "
          f"RFTN: {quantum_vitaldb_results['mdape_remifentanil']:.2f}%)")
    print(f"  Classical - MDAPE: {classical_vitaldb_results['mdape_mean']:.2f}% "
          f"(PPF: {classical_vitaldb_results['mdape_propofol']:.2f}%, "
          f"RFTN: {classical_vitaldb_results['mdape_remifentanil']:.2f}%)")
    
    # 2. Evaluate on dual drug simulator
    print("\n" + "-"*70)
    print("EVALUATING ON DUAL DRUG SIMULATOR")
    print("-"*70)
    
    print(f"\nEvaluating Quantum agent on simulator...")
    quantum_sim_results = evaluate_agent_on_simulator_dualdrug(
        quantum_agent, args.n_test_episodes, args.seed + 2000, device
    )
    
    print(f"\nEvaluating Classical agent on simulator...")
    classical_sim_results = evaluate_agent_on_simulator_dualdrug(
        classical_agent, args.n_test_episodes, args.seed + 3000, device
    )
    
    print(f"\n✓ Simulator Results:")
    print(f"  Quantum  - MDAPE: {quantum_sim_results['mdape_mean']:.2f}%, "
          f"Reward: {quantum_sim_results['reward_mean']:.2f}")
    print(f"  Classical - MDAPE: {classical_sim_results['mdape_mean']:.2f}%, "
          f"Reward: {classical_sim_results['reward_mean']:.2f}")
    
    # Statistical comparison (using simulator results)
    print(f"\n" + "-"*70)
    print("STATISTICAL COMPARISON")
    print("-"*70)
    stats_results = statistical_comparison(quantum_sim_results, classical_sim_results, args.alpha)
    
    print(f"\nStatistical Test Results:")
    print(f"  Winner: {stats_results['winner']}")
    print(f"  Improvement: {stats_results['improvement_pct']:.2f}%")
    print(f"  t-test p-value: {stats_results['t_pvalue']:.4f} "
          f"({'significant' if stats_results['t_significant'] else 'not significant'} at α={args.alpha})")
    print(f"  Mann-Whitney U p-value: {stats_results['u_pvalue']:.4f} "
          f"({'significant' if stats_results['u_significant'] else 'not significant'})")
    print(f"  Cohen's d: {stats_results['cohens_d']:.3f}")
    
    # Generate plots
    print(f"\n" + "-"*70)
    print("GENERATING PLOTS")
    print("-"*70)
    plot_dualdrug_comparison(
        quantum_sim_results, classical_sim_results,
        'Dual Drug Control: Quantum vs Classical (Simulator)',
        comparison_dir / 'dualdrug_comparison_simulator.png'
    )
    
    # Save results
    results = {
        'quantum_vitaldb': quantum_vitaldb_results,
        'classical_vitaldb': classical_vitaldb_results,
        'quantum_simulator': quantum_sim_results,
        'classical_simulator': classical_sim_results,
        'statistics': stats_results,
        'config': vars(args)
    }
    
    with open(comparison_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✓ Results saved to: {comparison_dir / 'results.pkl'}")
    
    # Save results to CSV files
    print(f"\n" + "-"*70)
    print("SAVING RESULTS TO CSV")
    print("-"*70)
    
    save_results_to_csv(
        quantum_results=quantum_sim_results,
        classical_results=classical_sim_results,
        stats_results=stats_results,
        save_dir=comparison_dir / 'csv_results'
    )
    
    # Generate summary report
    print(f"\n" + "-"*70)
    print("GENERATING SUMMARY REPORT")
    print("-"*70)
    
    report_path = comparison_dir / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DUAL DRUG CONTROL: QUANTUM vs CLASSICAL RL COMPARISON\n")
        f.write("="*70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*70 + "\n")
        f.write(f"State dimension: {args.state_dim}\n")
        f.write(f"Action dimension: {args.action_dim} (propofol + remifentanil)\n")
        f.write(f"VitalDB cases: {args.n_cases}\n")
        f.write(f"Offline epochs: {args.offline_epochs}\n")
        f.write(f"Online episodes: {args.online_episodes}\n")
        f.write(f"Test episodes: {args.n_test_episodes}\n")
        f.write(f"Encoder: {args.encoder}\n")
        f.write(f"Seed: {args.seed}\n\n")
        
        f.write("VITALDB TEST SET RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"Quantum Agent:\n")
        f.write(f"  Overall MDAPE: {quantum_vitaldb_results['mdape_mean']:.2f}% ± {quantum_vitaldb_results['mdape_std']:.2f}%\n")
        f.write(f"  Propofol MDAPE: {quantum_vitaldb_results['mdape_propofol']:.2f}%\n")
        f.write(f"  Remifentanil MDAPE: {quantum_vitaldb_results['mdape_remifentanil']:.2f}%\n\n")
        
        f.write(f"Classical Agent:\n")
        f.write(f"  Overall MDAPE: {classical_vitaldb_results['mdape_mean']:.2f}% ± {classical_vitaldb_results['mdape_std']:.2f}%\n")
        f.write(f"  Propofol MDAPE: {classical_vitaldb_results['mdape_propofol']:.2f}%\n")
        f.write(f"  Remifentanil MDAPE: {classical_vitaldb_results['mdape_remifentanil']:.2f}%\n\n")
        
        f.write("SIMULATOR TEST RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"MDAPE:\n")
        f.write(f"  Quantum:   {quantum_sim_results['mdape_mean']:.2f} ± {quantum_sim_results['mdape_std']:.2f}%\n")
        f.write(f"  Classical: {classical_sim_results['mdape_mean']:.2f} ± {classical_sim_results['mdape_std']:.2f}%\n\n")
        
        f.write(f"Reward:\n")
        f.write(f"  Quantum:   {quantum_sim_results['reward_mean']:.2f} ± {quantum_sim_results['reward_std']:.2f}\n")
        f.write(f"  Classical: {classical_sim_results['reward_mean']:.2f} ± {classical_sim_results['reward_std']:.2f}\n\n")
        
        f.write(f"Time in Target (BIS 45-55):\n")
        f.write(f"  Quantum:   {quantum_sim_results['time_in_target_mean']:.1f}%\n")
        f.write(f"  Classical: {classical_sim_results['time_in_target_mean']:.1f}%\n\n")
        
        f.write(f"Drug Usage:\n")
        f.write(f"  Propofol:\n")
        f.write(f"    Quantum:   {quantum_sim_results['propofol_usage_mean']:.2f} ± {quantum_sim_results['propofol_usage_std']:.2f} mg/kg/h\n")
        f.write(f"    Classical: {classical_sim_results['propofol_usage_mean']:.2f} ± {classical_sim_results['propofol_usage_std']:.2f} mg/kg/h\n")
        f.write(f"  Remifentanil:\n")
        f.write(f"    Quantum:   {quantum_sim_results['remifentanil_usage_mean']:.2f} ± {quantum_sim_results['remifentanil_usage_std']:.2f} μg/kg/min\n")
        f.write(f"    Classical: {classical_sim_results['remifentanil_usage_mean']:.2f} ± {classical_sim_results['remifentanil_usage_std']:.2f} μg/kg/min\n\n")
        
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
    
    print(f"\nVitalDB Test Set:")
    print(f"  Quantum MDAPE:   {quantum_vitaldb_results['mdape_mean']:.2f}%")
    print(f"  Classical MDAPE: {classical_vitaldb_results['mdape_mean']:.2f}%")
    
    print(f"\nSimulator Test Set:")
    print(f"  Quantum MDAPE:   {quantum_sim_results['mdape_mean']:.2f} ± {quantum_sim_results['mdape_std']:.2f}%")
    print(f"  Classical MDAPE: {classical_sim_results['mdape_mean']:.2f} ± {classical_sim_results['mdape_std']:.2f}%")
    print(f"  Winner: {stats_results['winner']} ({stats_results['improvement_pct']:.2f}% improvement)")
    print(f"  Significant: {'YES' if stats_results['t_significant'] else 'NO'} (p={stats_results['t_pvalue']:.4f})")







    print(f"  Quantum:   {quantum_sim_results['propofol_usage_mean']:.2f} ± {quantum_sim_results['propofol_usage_std']:.2f} mg/kg/h")    
    print(f"\nPropofol usage:")        
    print(f"  Classical: {classical_sim_results['time_in_target_mean']:.1f}%")    
    print(f"  Quantum:   {quantum_sim_results['time_in_target_mean']:.1f}%")    
    print(f"\nTime in Target (BIS 45-55):")        
    print(f"  Classical: {classical_sim_results['propofol_usage_mean']:.2f} ± {classical_sim_results['propofol_usage_std']:.2f} mg/kg/h")
    
    print(f"\nRemifentanil usage:")
    print(f"  Quantum:   {quantum_sim_results['remifentanil_usage_mean']:.2f} ± {quantum_sim_results['remifentanil_usage_std']:.2f} μg/kg/min")
    print(f"  Classical: {classical_sim_results['remifentanil_usage_mean']:.2f} ± {classical_sim_results['remifentanil_usage_std']:.2f} μg/kg/min")
    
    print(f"\n✓ All results saved to: {comparison_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
