"""
Quantum vs Classical RL Comparison for Dual Drug Control
=========================================================

Compare Quantum and Classical agents on dual drug environment
(Propofol + Remifentanil).

Key differences from single drug comparison:
- Action space: [propofol_rate, remifentanil_rate] (2D)
- State space: Extended to include remifentanil Ce + demographics (13D)
- Agents: action_dim=2, n_qubits=3 (increased for 13D state)

Training Order:
1. Classical Agent (faster, baseline)
2. Quantum Agent (slower, experimental)

Usage:
    # Full comparison
    python experiments/compare_quantum_vs_classical.py \
        --n_cases 100 --offline_epochs 50 --online_episodes 500
    
    # Quick test
    python experiments/compare_quantum_vs_classical.py \
        --n_cases 20 --offline_epochs 5 --online_episodes 50
    
    # With encoder
    python experiments/compare_quantum_vs_classical.py \
        --n_cases 100 --encoder transformer
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import yaml
from datetime import datetime
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

from agents.quantum_agent import QuantumDDPGAgent
from agents.classical_agent import ClassicalDDPGAgent
from environment.dual_drug_env import DualDrugEnv
from environment.patient_simulator import create_patient_population
from data.vitaldb_loader import VitalDBLoader
from data.vitaldb_loader_remi import prepare_training_data_with_remi
from train_hybrid import stage1_offline_pretraining, stage2_online_finetuning
from evaluation.clinical_metrics import evaluate_episode, ClinicalMetrics
from visualization.clinical_plots import plot_comparison, plot_radar_chart, plot_poincare


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare Quantum vs Classical RL for Dual Drug Control'
    )
    
    # VitalDB data
    parser.add_argument('--n_cases', type=int, default=6000,
                       help='Number of VitalDB cases to load (more data = better BC)')
    parser.add_argument('--sampling_interval', type=int, default=60,
                       help='Sampling interval for VitalDB data (1=all data, higher=faster)')
    parser.add_argument('--offline_epochs', type=int, default=10,
                       help='Number of offline pre-training epochs (more epochs for better BC)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for offline training (smaller for better BC convergence)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of DataLoader workers')
    parser.add_argument('--bc_weight', type=float, default=0.9,
                       help='Behavioral cloning weight (0.9 = 90% BC + 10% RL for hybrid training)')
    
    # CQL parameters
    parser.add_argument('--use_cql', type=lambda x: (str(x).lower() == 'true'), default=False,
                       help='Use CQL instead of standard off-policy RL')
    parser.add_argument('--cql_alpha', type=float, default=1.0,
                       help='CQL penalty weight')
    parser.add_argument('--cql_temp', type=float, default=1.0,
                       help='Temperature for CQL logsumexp')
    parser.add_argument('--cql_num_random', type=int, default=5,
                       help='Number of random actions for CQL')
    parser.add_argument('--cql_warmup_epochs', type=int, default=10,
                       help='Epochs to use CQL')
    parser.add_argument('--bc_warmup_epochs', type=int, default=20,
                       help='Initial epochs with BC-only training (BC weight=1.0) for stability')
    
    # Online training
    parser.add_argument('--online_episodes', type=int, default=300,
                       help='Number of online training episodes')
    parser.add_argument('--warmup_episodes', type=int, default=30,
                       help='Warmup episodes with forced high-dose exploration (induction learning)')
    
    # Agent configuration
    parser.add_argument('--state_dim', type=int, default=13,
                       help='State dimension for dual drug (extended with demographics: age, sex, BMI)')
    parser.add_argument('--action_dim', type=int, default=2,
                       help='Action dimension (propofol + remifentanil)')
    parser.add_argument('--n_qubits', type=int, default=2,
                       help='Number of qubits for quantum circuit (3 qubits for 13D state)')
    parser.add_argument('--encoder', type=str, default='transformer',
                       choices=['none', 'lstm', 'transformer'],
                       help='Temporal encoder type')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Reward function
    parser.add_argument('--reward_type', type=str, default='potential',
                       choices=['simple', 'paper', 'hybrid', 'potential'],
                       help='Reward function type')
    
    # Action Smoothing
    parser.add_argument('--smoothing_beta', type=float, default=0.8,
                       help='Exponential moving average factor for action smoothing (0.0=no smoothing, 0.8=strong smoothing)')
    
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


def evaluate_agent_on_simulator(
    agent, 
    n_episodes: int, 
    seed: int,
    device: torch.device,
    reward_type: str = 'potential',
    save_trajectories: bool = False,
    smoothing_beta: float = 0.0
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
    wobbles = []
    divergences = []
    global_scores = []
    rewards_total = []
    propofol_usage = []
    remifentanil_usage = []
    time_in_target = []
    
    # Store episode trajectories for detailed analysis
    episode_trajectories = []
    
    for i, patient in enumerate(tqdm(patients, desc="Evaluating on Dual Drug Simulator")):
        # Create dual drug environment with specified reward type
        env = DualDrugEnv(patient=patient, seed=seed + i, reward_type=reward_type)
        
        state, _ = env.reset()
        episode_reward = 0
        episode_ppf = []
        episode_rftn = []
        states_list = []
        bis_history = []
        actions_history = []
        rewards_history = []
        time_steps = []
        
        # Action smoothing initialization
        prev_action = None
        
        for step in range(200):
            # CRITICAL: Use agent.select_action(deterministic=True) for consistent evaluation
            # This ensures no exploration noise is added and matches training evaluation
            action_physical = agent.select_action(state, deterministic=True)
            
            # Ensure action is 2D for dual drug
            if agent.action_dim == 2 and action_physical.shape[0] != 2:
                print(f"Warning: Expected action_dim=2, got {action_physical.shape[0]}. Using first 2 dimensions.")
                action_physical = action_physical[:2]
            
            # Apply EMA Smoothing
            if smoothing_beta > 0.0:
                if prev_action is None:
                    prev_action = action_physical
                else:
                    action_physical = smoothing_beta * prev_action + (1 - smoothing_beta) * action_physical
                    prev_action = action_physical
            
            
            next_state, reward, done, truncated, info = env.step(action_physical)
            
            episode_reward += reward
            # Store actual drug infusion rates (already in physical units)
            ppf_denorm = action_physical[0]  # Propofol (mg/kg/h)
            rftn_denorm = action_physical[1]  # Remifentanil (μg/kg/min)
            episode_ppf.append(ppf_denorm)
            episode_rftn.append(rftn_denorm)
            states_list.append(state)
            
            # Track BIS for time in target
            bis = info.get('bis', 50 - state[0])  # state[0] is BIS_error
            bis_history.append(bis)
            actions_history.append(ppf_denorm)  # Store denormalized propofol for plot
            rewards_history.append(reward)
            time_steps.append(step)
            
            state = next_state
            
            if done or truncated:
                break
        
        # Calculate metrics using ClinicalMetrics
        bis_arr = np.array(bis_history)
        ppf_arr = np.array(episode_ppf)
        
        clinical_metrics = evaluate_episode(
            bis_history=bis_arr,
            drug_history=ppf_arr,
            reward_history=np.array(rewards_history),
            target_bis=50.0,
            threshold=5.0,
            patient_weight=patient.weight
        )
        
        mdapes.append(clinical_metrics.mdape)
        wobbles.append(clinical_metrics.wobble)
        divergences.append(clinical_metrics.divergence)
        global_scores.append(clinical_metrics.global_score)
        
        time_in_target.append(clinical_metrics.time_in_target)
        rewards_total.append(episode_reward)
        propofol_usage.append(np.mean(episode_ppf))
        remifentanil_usage.append(np.mean(episode_rftn))
        
        # Save episode trajectory
        if save_trajectories:
            episode_trajectories.append({
                'episode': i,
                'time': np.array(time_steps),
                'bis': np.array(bis_history),
                'action': np.array(actions_history),
                'reward': np.array(rewards_history),
                'mdape': mdapes[-1] if mdapes else 0,
                'wobble': wobbles[-1] if wobbles else 0,
                'divergence': divergences[-1] if divergences else 0,
                'global_score': global_scores[-1] if global_scores else 0,
                'total_reward': episode_reward,
                'time_in_target': clinical_metrics.time_in_target,
                'propofol': np.array(episode_ppf),
                'remifentanil': np.array(episode_rftn)
            })
    
    result = {
        'mdape_mean': np.mean(mdapes),
        'mdape_std': np.std(mdapes),
        'mdape_list': mdapes,
        
        'wobble_mean': np.mean(wobbles),
        'wobble_std': np.std(wobbles),
        'wobble_list': wobbles,
        
        'divergence_mean': np.mean(divergences),
        'divergence_std': np.std(divergences),
        'divergence_list': divergences,
        
        'global_score_mean': np.mean(global_scores),
        'global_score_std': np.std(global_scores),
        'global_score_list': global_scores,
        
        'reward_mean': np.mean(rewards_total),
        'reward_std': np.std(rewards_total),
        'reward_list': rewards_total,
        
        'propofol_usage_mean': np.mean(propofol_usage),
        'propofol_usage_std': np.std(propofol_usage),
        'propofol_usage_list': propofol_usage,
        
        'remifentanil_usage_mean': np.mean(remifentanil_usage),
        'remifentanil_usage_std': np.std(remifentanil_usage),
        'remifentanil_usage_list': remifentanil_usage,
        
        'time_in_target_mean': np.mean(time_in_target),
        'time_in_target_std': np.std(time_in_target),
        'time_in_target_list': time_in_target
    }
    
    if save_trajectories:
        result['trajectories'] = episode_trajectories
    
    return result


def save_trajectories_to_csv(trajectories: List[Dict], save_dir: Path, agent_name: str):
    """Save episode trajectories to CSV files."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary file
    summary_data = []
    for traj in trajectories:
        summary_data.append({
            'episode': traj['episode'],
            'mdape': traj.get('mdape', 0),
            'wobble': traj.get('wobble', 0),
            'divergence': traj.get('divergence', 0),
            'global_score': traj.get('global_score', 0),
            'total_reward': traj['total_reward'],
            'time_in_target': traj['time_in_target'],
            'duration': len(traj['time'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = save_dir / f'{agent_name}_trajectory_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved trajectory summary: {summary_path}")
    
    # Save detailed time-series for each episode
    for traj in trajectories:
        episode_data = pd.DataFrame({
            'time': traj['time'],
            'bis': traj['bis'],
            'propofol': traj['propofol'] if 'propofol' in traj else traj['action'],
            'remifentanil': traj['remifentanil'] if 'remifentanil' in traj else np.zeros_like(traj['time']),
            'reward': traj['reward']
        })
        episode_path = save_dir / f"{agent_name}_episode_{traj['episode']:03d}.csv"
        episode_data.to_csv(episode_path, index=False)
    
    print(f"✓ Saved {len(trajectories)} detailed episode trajectories to {save_dir}")


def plot_bis_trajectories(trajectories: List[Dict], save_path: Path, agent_name: str, n_episodes: int = 5):
    """Plot BIS trajectories over time for selected episodes."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Select episodes to plot (best, median, worst by MDAPE)
    sorted_traj = sorted(trajectories, key=lambda x: x['mdape'])
    n_episodes = min(n_episodes, len(trajectories))
    
    # Select evenly spaced episodes
    indices = np.linspace(0, len(sorted_traj)-1, n_episodes, dtype=int)
    selected_traj = [sorted_traj[i] for i in indices]
    
    # Plot 1: BIS over time
    ax = axes[0]
    for traj in selected_traj:
        label = f"Ep {traj['episode']} (MDAPE={traj['mdape']:.1f}%)"
        ax.plot(traj['time'], traj['bis'], label=label, alpha=0.7, linewidth=2)
    
    ax.axhline(y=50, color='green', linestyle='--', linewidth=2, label='Target BIS=50', alpha=0.7)
    ax.axhline(y=45, color='orange', linestyle=':', linewidth=1.5, label='Target Range', alpha=0.5)
    ax.axhline(y=55, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.fill_between([0, max([t['time'].max() for t in selected_traj])], 45, 55, 
                     color='green', alpha=0.1, label='Target Zone')
    
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('BIS Value', fontsize=12)
    ax.set_title(f'{agent_name} - BIS Control Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(20, 80)
    
    # Plot 2: Actions over time
    ax = axes[1]
    for traj in selected_traj:
        label = f"Ep {traj['episode']}"
        ax.plot(traj['time'], traj['action'], label=label, alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Propofol Infusion Rate (mg/kg/h)', fontsize=12)
    ax.set_title(f'{agent_name} - Control Actions Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved BIS trajectory plot: {save_path}")
    plt.close()


def save_results_to_csv(
    quantum_results: Dict,
    classical_results: Dict,
    stats_results: Dict,
    save_dir: Path
) -> None:
    """Save comparison results to CSV files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary statistics
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
    
    # 2. Episode results
    n_episodes = len(quantum_results['mdape_list'])
    episode_data = {
        'Episode': list(range(1, n_episodes + 1)),
        'Quantum_MDAPE': quantum_results['mdape_list'],
        'Classical_MDAPE': classical_results['mdape_list'],
        'Quantum_Reward': quantum_results['reward_list'],
        'Classical_Reward': classical_results['reward_list'],
        'Quantum_Propofol': quantum_results['propofol_usage_list'],
        'Classical_Propofol': classical_results['propofol_usage_list'],
        'Quantum_Remifentanil': quantum_results['remifentanil_usage_list'],
        'Classical_Remifentanil': classical_results['remifentanil_usage_list'],
        'Quantum_TimeInTarget': quantum_results['time_in_target_list'],
        'Classical_TimeInTarget': classical_results['time_in_target_list'],
    }
    df_episodes = pd.DataFrame(episode_data)
    df_episodes.to_csv(save_dir / 'episode_results.csv', index=False)
    print(f"✓ Saved episode results to {save_dir / 'episode_results.csv'}")
    
    # 3. Statistical tests
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
            f"Difference: {stats_results['mean_difference']:.2f}%",
            f"{stats_results['winner']} wins by {abs(stats_results['improvement_pct']):.1f}%"
        ]
    }
    df_stats = pd.DataFrame(stats_data)
    df_stats.to_csv(save_dir / 'statistical_tests.csv', index=False)
    print(f"✓ Saved statistical tests to {save_dir / 'statistical_tests.csv'}")


def statistical_comparison(quantum_results: Dict, classical_results: Dict, 
                          alpha: float = 0.05) -> Dict:
    """Perform statistical significance tests."""
    n_samples = len(quantum_results['mdape_list'])
    
    if n_samples < 2:
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
        }
    
    # T-test
    t_stat, t_pval = stats.ttest_ind(
        quantum_results['mdape_list'],
        classical_results['mdape_list']
    )
    
    # Mann-Whitney U test
    u_stat, u_pval = stats.mannwhitneyu(
        quantum_results['mdape_list'],
        classical_results['mdape_list'],
        alternative='two-sided'
    )
    
    # Cohen's d
    pooled_std = np.sqrt(
        (np.var(quantum_results['mdape_list']) + np.var(classical_results['mdape_list'])) / 2
    )
    cohens_d = (quantum_results['mdape_mean'] - classical_results['mdape_mean']) / (pooled_std + 1e-8)
    
    # Mean difference and CI
    mean_diff = quantum_results['mdape_mean'] - classical_results['mdape_mean']
    se_diff = np.sqrt(
        quantum_results['mdape_std']**2 / len(quantum_results['mdape_list']) +
        classical_results['mdape_std']**2 / len(classical_results['mdape_list'])
    )
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff
    
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
    """Evaluate agent on VitalDB test set."""
    states, actions, next_states = test_data
    
    predicted_actions = []
    
    # Set to eval mode for deterministic behavior
    agent.actor.eval()
    if hasattr(agent, 'encoder') and agent.encoder is not None:
        agent.encoder.eval()
    
    for state in tqdm(states, desc="VitalDB Test Set Evaluation"):
        # CRITICAL: Use agent.select_action(deterministic=True) for consistent evaluation
        action_physical = agent.select_action(state, deterministic=True)
        
        # Normalize action back to [0,1] for comparison with VitalDB normalized actions
        if hasattr(action_physical, '__len__') and len(action_physical) == 2:
            # Dual drug: denormalize back to [0,1] (must match VitalDB PPF_MAX/RFTN_MAX)
            action_normalized = np.array([
                action_physical[0] / 12.0,  # Propofol [0-12] → [0-1] (matches vitaldb_loader PPF_MAX)
                action_physical[1] / 2.0    # Remifentanil [0-2.0] → [0-1] (matches vitaldb_loader RFTN_MAX)
            ])
        else:
            # Single drug
            action_normalized = action_physical / 12.0
        
        predicted_actions.append(action_normalized)
    
    predicted_actions = np.array(predicted_actions)
    
    # Ensure shapes match for comparison
    if predicted_actions.shape[1] != actions.shape[1]:
        # Pad or trim if needed
        if predicted_actions.shape[1] < actions.shape[1]:
            padding = np.zeros((predicted_actions.shape[0], actions.shape[1] - predicted_actions.shape[1]))
            predicted_actions = np.concatenate([predicted_actions, padding], axis=1)
        else:
            predicted_actions = predicted_actions[:, :actions.shape[1]]
    
    # MDAPE
    error = np.abs(predicted_actions - actions) / (np.abs(actions) + 1e-6)
    mdape = np.median(error) * 100
    
    return {
        'mdape_mean': mdape,
        'mdape_std': np.std(error) * 100,
    }


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
    print(f"{'='*70}\n")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['device'] = str(device)
    
    # CLI --n_qubits가 config를 동적으로 override:
    # - quantum.n_qubits: VQC 큐빗 수 (= action_dim, 큐빗 1개가 약물 1개에 대응)
    # - networks.encoder.output_dim: QuantumPolicy 내부 Classical Encoder의 출력 차원
    #   (반드시 n_qubits와 동일해야 VQC 입력 차원이 맞음)
    config.setdefault('quantum', {})['n_qubits'] = args.n_qubits
    config.setdefault('networks', {}).setdefault('encoder', {})['output_dim'] = args.n_qubits
    
    print(f"  n_qubits CLI override → quantum.n_qubits={args.n_qubits}, "
          f"networks.encoder.output_dim={args.n_qubits}")

    
    # Setup directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_dir = Path(args.log_dir) / f'comparison_{timestamp}'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("QUANTUM vs CLASSICAL COMPARISON (DUAL DRUG CONTROL)")
    print("="*70)
    print(f"Configuration:")
    print(f"  State dim: {args.state_dim} (includes remifentanil Ce + demographics)")
    print(f"  Action dim: {args.action_dim} (propofol + remifentanil)")
    print(f"  VitalDB cases: {args.n_cases}")
    print(f"  Offline epochs: {args.offline_epochs}")
    print(f"  Online episodes: {args.online_episodes}")
    print(f"  Test episodes: {args.n_test_episodes}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Training order: Classical → Quantum")
    print(f"  Log dir: {comparison_dir}")
    print("="*70 + "\n")
    
    # Load VitalDB data
    print("\n" + "="*70)
    print("LOADING VITALDB DATA")
    print("="*70)
    
    loader = VitalDBLoader(cache_dir='data/vitaldb_cache', use_cache=True)
    
    print(f"\nLoading {args.n_cases} dual drug cases from VitalDB...")
    print("  Mode: Dual drug control (Propofol + Remifentanil)")
    print(f"  Sampling interval: {args.sampling_interval}s ({'all data' if args.sampling_interval == 1 else f'{100 * (1 - 1/args.sampling_interval):.0f}% reduction'})")
    
    data = prepare_training_data_with_remi(
            loader=loader,
            n_cases=args.n_cases,
            sampling_interval=args.sampling_interval,  # 샘플링 간격 적용
            add_induction=True,  # Induction data 추가
            n_induction_samples=2000  # 2000개 샘플
        )
    states = data['states']
    actions = data['actions']
    next_states = data['next_states']
    rewards = data['rewards']
    dones = data['dones']
    
    # Update state_dim based on actual data shape
    actual_state_dim = states.shape[1]
    if actual_state_dim != args.state_dim:
        print(f"\n⚠️  Updating state_dim: {args.state_dim} → {actual_state_dim} (based on data)")
        args.state_dim = actual_state_dim
    
    print(f"\n✓ Loaded data:")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  State features: {args.state_dim}D (BIS error, Ce propofol, Ce remifentanil, demographics)")
    print(f"  Action features: {args.action_dim}D (propofol + remifentanil)")
    print(f"  Agent controls: Both propofol and remifentanil infusion rates")
    
    # Split data
    n_total = len(states)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
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
    test_data = (states[test_indices], actions[test_indices], next_states[test_indices])
    
    print(f"\nData split:")
    print(f"  Train: {len(train_indices):,} samples")
    print(f"  Val: {len(val_indices):,} samples")
    print(f"  Test: {len(test_indices):,} samples")
    print("="*70)
    
    # ========================================
    # Train Classical Agent FIRST (faster)
    # ========================================
    print("\n" + "="*70)
    print("TRAINING CLASSICAL AGENT (BASELINE)")
    print("="*70)
    
    classical_agent = ClassicalDDPGAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        config=config,
        encoder_type=args.encoder,
        seed=args.seed
    )
    
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
        'stage1': comparison_dir / 'classical' / 'stage1_offline',
        'stage2': comparison_dir / 'classical' / 'stage2_online',
    }
    for d in classical_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Offline
    print("\n" + "-"*70)
    print("STAGE 1: OFFLINE PRE-TRAINING (VitalDB)")
    print("-"*70)
    
    classical_agent = stage1_offline_pretraining(
        agent=classical_agent,
        train_data=train_data_dict,
        val_data=val_data_dict,
        n_epochs=args.offline_epochs,
        batch_size=args.batch_size,
        bc_weight=args.bc_weight,
        use_cql=False,
        cql_alpha=args.cql_alpha,
        cql_temp=args.cql_temp,
        cql_num_random=args.cql_num_random,
        cql_warmup_epochs=args.cql_warmup_epochs,
        bc_warmup_epochs=args.bc_warmup_epochs,
        num_workers=args.num_workers,
        dirs={'stage1': classical_dirs['stage1']},
        device=device
    )
    
    # Debugging: Check training data and agent initialization
    print("\n" + "="*70)
    print("DEBUGGING: ACTION DISTRIBUTION & AGENT STATE")
    print("="*70)
    
    # 1. VitalDB data distribution
    print(f"\nVitalDB Training Data:")
    print(f"  Actions mean: {train_data_dict['actions'].mean():.4f}")
    print(f"  Actions std: {train_data_dict['actions'].std():.4f}")
    print(f"  Actions min/max: [{train_data_dict['actions'].min():.4f}, {train_data_dict['actions'].max():.4f}]")
    print(f"  Actions median: {np.median(train_data_dict['actions']):.4f}")
    print(f"  Actions at percentiles [25%, 50%, 75%]: {np.percentile(train_data_dict['actions'], [25, 50, 75])}")
    
    # Check data distribution
    action_bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15]
    hist, _ = np.histogram(train_data_dict['actions'], bins=action_bins)
    print(f"\n  Action Distribution:")
    for i in range(len(hist)):
        bin_start = action_bins[i]
        bin_end = action_bins[i+1]
        percentage = hist[i] / len(train_data_dict['actions']) * 100
        print(f"    [{bin_start:.2f}, {bin_end:.2f}): {hist[i]:6d} samples ({percentage:5.1f}%)")
    
    # 2. Agent's initial output test
    test_state = torch.randn(1, args.state_dim).to(device)
    with torch.no_grad():
        if classical_agent.encoder is not None:
            test_seq = test_state.unsqueeze(0)
            encoded = classical_agent.encoder(test_seq)
            if isinstance(encoded, tuple):
                encoded = encoded[0]
            if encoded.dim() == 3:
                encoded = encoded[:, -1, :]
            test_action = classical_agent.actor(encoded)
        else:
            test_action = classical_agent.actor(test_state)
    
    print(f"\nAgent Output After Training:")
    print(f"  Raw actor output: {test_action.cpu().numpy()[0]}")
    print(f"  Expected range: [0, 1] (before scaling)")
    print(f"  After scaling (×{classical_agent.action_scale}): {test_action.cpu().numpy()[0] * classical_agent.action_scale}")
    
    # 3. Actor architecture check
    print(f"\nActor Architecture:")
    if hasattr(classical_agent.actor, 'network'):
        for name, module in classical_agent.actor.network.named_children():
            if isinstance(module, nn.Linear):
                print(f"  {name}: {module}")
                if hasattr(module, 'bias') and module.bias is not None:
                    print(f"    Bias mean: {module.bias.mean().item():.4f}, std: {module.bias.std().item():.4f}")
    
    # 4. Sample predictions on training data (compare normalized values)
    sample_indices = np.random.choice(len(train_data_dict['states']), min(100, len(train_data_dict['states'])), replace=False)
    sample_states = train_data_dict['states'][sample_indices]
    sample_actions_true = train_data_dict['actions'][sample_indices]
    
    sample_actions_pred = []
    for state in sample_states:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if classical_agent.encoder is not None:
                state_seq = state_tensor.unsqueeze(0)
                encoded = classical_agent.encoder(state_seq)
                if isinstance(encoded, tuple):
                    encoded = encoded[0]
                if encoded.dim() == 3:
                    encoded = encoded[:, -1, :]
                action = classical_agent.actor(encoded)
            else:
                action = classical_agent.actor(state_tensor)
        sample_actions_pred.append(action.cpu().numpy().flatten())  # Keep both actions
    sample_actions_pred = np.array(sample_actions_pred)
    
    print(f"\nPredictions on Training Sample (n={len(sample_indices)}) - Normalized:")
    print(f"  Predicted mean: {sample_actions_pred.mean():.4f} (normalized)")
    print(f"  True mean: {sample_actions_true.mean():.4f} (normalized)")
    print(f"  Predicted std: {sample_actions_pred.std():.4f}")
    print(f"  True std: {sample_actions_true.std():.4f}")
    print(f"  MAE (normalized): {np.mean(np.abs(sample_actions_pred - sample_actions_true)):.4f}")
    print(f"  Propofol MAE (mg/kg/h): {np.mean(np.abs(sample_actions_pred[:, 0] - sample_actions_true[:, 0])) * classical_agent.action_scale:.2f}")
    print(f"  Remifentanil MAE (μg/kg/min): {np.mean(np.abs(sample_actions_pred[:, 1] - sample_actions_true[:, 1])) * 0.5:.3f}")
    
    print("="*70 + "\n")
    
    # Stage 2: Online
    print("\n" + "-"*70)
    print("STAGE 2: ONLINE FINE-TUNING (Simulator)")
    print("-"*70)
    
    classical_agent = stage2_online_finetuning(
        agent=classical_agent,
        n_episodes=args.online_episodes,
        warmup_episodes=args.warmup_episodes,
        dirs={'stage2': classical_dirs['stage2']},
        eval_interval=50,
        seed=args.seed,
        env_class=DualDrugEnv,
        device=device,
        n_eval_patients=100  # Use 100 patients from large pool
    )
    
    print(f"✓ Classical agent trained")
    
    # Load best checkpoint for fair evaluation
    print("\n" + "-"*70)
    print("LOADING BEST CLASSICAL CHECKPOINT FOR EVALUATION")
    print("-"*70)
    best_classical_path = classical_dirs['stage2'] / 'best_mdape.pt'
    if best_classical_path.exists():
        classical_agent.load(str(best_classical_path))
        print(f"✓ Loaded best MDAPE checkpoint: {best_classical_path}")
    else:
        print(f"⚠️  Best MDAPE checkpoint not found, using current model")
    
    # ========================================
    # Evaluate Classical Agent (Interim Results)
    # ========================================
    print("\n" + "="*70)
    print("CLASSICAL AGENT - INTERIM EVALUATION")
    print("="*70)
    
    # Load best checkpoint for fair evaluation
    print("\n" + "-"*70)
    print("LOADING BEST CLASSICAL CHECKPOINT FOR EVALUATION")
    print("-"*70)
    best_classical_path = classical_dirs['stage2'] / 'best_mdape.pt'
    if best_classical_path.exists():
        classical_agent.load(str(best_classical_path))
        print(f"✓ Loaded best MDAPE checkpoint: {best_classical_path}")
    else:
        print(f"⚠️  Best MDAPE checkpoint not found, using current model")
    
    # VitalDB test set
    print("\n" + "-"*70)
    print("EVALUATING CLASSICAL AGENT ON VITALDB TEST SET")
    print("-"*70)
    
    classical_vitaldb_results = evaluate_on_vitaldb_test_set(classical_agent, test_data, device)
    
    print(f"\n✓ Classical VitalDB Results:")
    print(f"  MDAPE: {classical_vitaldb_results['mdape_mean']:.2f}% ± {classical_vitaldb_results['mdape_std']:.2f}%")
    
    # Simulator
    print("\n" + "-"*70)
    print("EVALUATING CLASSICAL AGENT ON SIMULATOR")
    print("-"*70)
    
    classical_sim_results = evaluate_agent_on_simulator(
        classical_agent, args.n_test_episodes, args.seed + 2000, device, args.reward_type,
        save_trajectories=True, smoothing_beta=args.smoothing_beta
    )
    
    print(f"\n✓ Classical Simulator Results:")
    print(f"  MDAPE: {classical_sim_results['mdape_mean']:.2f}% ± {classical_sim_results['mdape_std']:.2f}%")
    print(f"  Reward: {classical_sim_results['reward_mean']:.2f} ± {classical_sim_results['reward_std']:.2f}")
    print(f"  Time in Target: {classical_sim_results['time_in_target_mean']:.2f}% ± {classical_sim_results['time_in_target_std']:.2f}%")
    print(f"  Propofol Usage: {classical_sim_results['propofol_usage_mean']:.2f} ± {classical_sim_results['propofol_usage_std']:.2f} mg/kg/h")
    print(f"  Remifentanil Usage: {classical_sim_results['remifentanil_usage_mean']:.2f} ± {classical_sim_results['remifentanil_usage_std']:.2f} μg/kg/min")
    
    # Save classical trajectories
    print("\n" + "-"*70)
    print("SAVING CLASSICAL AGENT TRAJECTORIES")
    print("-"*70)
    
    trajectory_dir = comparison_dir / 'trajectories'
    save_trajectories_to_csv(classical_sim_results['trajectories'], trajectory_dir / 'classical', 'classical')
    
    plot_bis_trajectories(
        classical_sim_results['trajectories'],
        comparison_dir / 'classical_bis_trajectories.png',
        'Classical Agent',
        n_episodes=5
    )
    
    print("\n" + "="*70)
    print("✓ CLASSICAL AGENT EVALUATION COMPLETE")
    print("="*70)
    
    # ========================================
    # Train Quantum Agent SECOND
    # ========================================
    print("\n" + "="*70)
    print("TRAINING QUANTUM AGENT (EXPERIMENTAL)")
    print("="*70)
    
    quantum_agent = QuantumDDPGAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        config=config,
        encoder_type=args.encoder,
        seed=args.seed + 1000
    )
    
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
        'stage1': comparison_dir / 'quantum' / 'stage1_offline',
        'stage2': comparison_dir / 'quantum' / 'stage2_online',
    }
    for d in quantum_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # Stage 1: Offline
    print("\n" + "-"*70)
    print("STAGE 1: OFFLINE PRE-TRAINING (VitalDB)")
    print("-"*70)
    
    quantum_agent = stage1_offline_pretraining(
        agent=quantum_agent,
        train_data=train_data_dict,
        val_data=val_data_dict,
        n_epochs=args.offline_epochs,
        batch_size=args.batch_size,
        bc_weight=args.bc_weight,
        use_cql=False,
        cql_alpha=args.cql_alpha,
        cql_temp=args.cql_temp,
        cql_num_random=args.cql_num_random,
        cql_warmup_epochs=args.cql_warmup_epochs,
        bc_warmup_epochs=args.bc_warmup_epochs,
        num_workers=args.num_workers,
        dirs={'stage1': quantum_dirs['stage1']},
        device=device
    )
    
    # Stage 2: Online
    print("\n" + "-"*70)
    print("STAGE 2: ONLINE FINE-TUNING (Simulator)")
    print("-"*70)
    
    quantum_agent = stage2_online_finetuning(
        agent=quantum_agent,
        n_episodes=args.online_episodes,
        warmup_episodes=args.warmup_episodes,
        dirs={'stage2': quantum_dirs['stage2']},
        eval_interval=50,
        seed=args.seed + 1000,
        env_class=DualDrugEnv,
        device=device,
        n_eval_patients=100  # Use 100 patients from large pool
    )
    
    print(f"✓ Quantum agent trained")
    
    # Load best checkpoint for fair evaluation
    print("\n" + "-"*70)
    print("LOADING BEST QUANTUM CHECKPOINT FOR EVALUATION")
    print("-"*70)
    best_quantum_path = quantum_dirs['stage2'] / 'best_mdape.pt'
    if best_quantum_path.exists():
        quantum_agent.load(str(best_quantum_path))
        print(f"✓ Loaded best MDAPE checkpoint: {best_quantum_path}")
    else:
        print(f"⚠️  Best MDAPE checkpoint not found, using current model")
    
    # ========================================
    # Final Evaluation (Both Agents)
    # ========================================
    print("\n" + "="*70)
    print("FINAL EVALUATION - COMPARING BOTH AGENTS")
    print("="*70)
    
    # VitalDB test set
    print("\n" + "-"*70)
    print("EVALUATING ON VITALDB TEST SET")
    print("-"*70)
    
    quantum_vitaldb_results = evaluate_on_vitaldb_test_set(quantum_agent, test_data, device)
    
    print(f"\n✓ VitalDB Results:")
    print(f"  Classical - MDAPE: {classical_vitaldb_results['mdape_mean']:.2f}%")
    print(f"  Quantum   - MDAPE: {quantum_vitaldb_results['mdape_mean']:.2f}%")
    
    # Simulator
    print("\n" + "-"*70)
    print("EVALUATING ON SIMULATOR")
    print("-"*70)
    
    quantum_sim_results = evaluate_agent_on_simulator(
        quantum_agent, args.n_test_episodes, args.seed + 3000, device, args.reward_type,
        save_trajectories=True, smoothing_beta=args.smoothing_beta
    )
    
    print(f"\n✓ Simulator Results:")
    print(f"  Classical - MDAPE: {classical_sim_results['mdape_mean']:.2f}%, Reward: {classical_sim_results['reward_mean']:.2f}")
    print(f"  Quantum   - MDAPE: {quantum_sim_results['mdape_mean']:.2f}%, Reward: {quantum_sim_results['reward_mean']:.2f}")
    
    # Save quantum trajectories and plot BIS time-series
    print("\n" + "-"*70)
    print("SAVING QUANTUM AGENT TRAJECTORIES")
    print("-"*70)
    
    save_trajectories_to_csv(quantum_sim_results['trajectories'], trajectory_dir / 'quantum', 'quantum')
    
    plot_bis_trajectories(
        quantum_sim_results['trajectories'],
        comparison_dir / 'quantum_bis_trajectories.png',
        'Quantum Agent',
        n_episodes=5
    )
    
    # Statistical comparison
    print(f"\n" + "-"*70)
    print("STATISTICAL COMPARISON")
    print("-"*70)
    stats_results = statistical_comparison(quantum_sim_results, classical_sim_results, args.alpha)
    
    print(f"\nStatistical Test Results:")
    print(f"  Winner: {stats_results['winner']}")
    print(f"  Improvement: {stats_results['improvement_pct']:.2f}%")
    print(f"  t-test p-value: {stats_results['t_pvalue']:.4f}")
    print(f"  Cohen's d: {stats_results['cohens_d']:.3f}")
    
    # Generate plots
    print(f"\n" + "-"*70)
    print("GENERATING PLOTS")
    print("-"*70)
    
    results_map = {
        'Classical': classical_sim_results,
        'Quantum': quantum_sim_results
    }
    
    # 1. Box Plot (MDAPE)
    plot_comparison(
        results_map, 
        metric='mdape', 
        plot_type='box',
        title='MDAPE Distribution (Lower is Better)',
        save_path=comparison_dir / 'comparison_mdape_box.png'
    )
    
    # 2. Box Plot (Wobble)
    plot_comparison(
        results_map, 
        metric='wobble', 
        plot_type='box',
        title='Wobble Distribution (Lower is Better)',
        save_path=comparison_dir / 'comparison_wobble_box.png'
    )
    
    # 3. Box Plot (Global Score)
    plot_comparison(
        results_map, 
        metric='global_score', 
        plot_type='box',
        title='Global Score (Lower is Better)',
        save_path=comparison_dir / 'comparison_global_score_box.png'
    )
    
    # 4. Radar Chart
    radar_metrics = ['mdape', 'wobble', 'global_score', 'propofol_usage', 'time_in_target']
    plot_radar_chart(
        results_map,
        metrics=radar_metrics,
        title='Multi-Metric Comparison',
        save_path=comparison_dir / 'comparison_radar.png'
    )
    
    # 5. Poincaré Plots
    if 'trajectories' in classical_sim_results and classical_sim_results['trajectories']:
        plot_poincare(
            classical_sim_results['trajectories'][0]['bis'],
            title='Classical Agent Stability (Poincaré)',
            save_path=comparison_dir / 'poincare_classical.png'
        )
            
    if 'trajectories' in quantum_sim_results and quantum_sim_results['trajectories']:
        plot_poincare(
            quantum_sim_results['trajectories'][0]['bis'],
            title='Quantum Agent Stability (Poincaré)',
            save_path=comparison_dir / 'poincare_quantum.png'
        )
    
    # Save results
    results = {
        'classical_vitaldb': classical_vitaldb_results,
        'quantum_vitaldb': quantum_vitaldb_results,
        'classical_simulator': classical_sim_results,
        'quantum_simulator': quantum_sim_results,
        'statistics': stats_results,
        'config': vars(args)
    }
    
    with open(comparison_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    save_results_to_csv(
        quantum_results=quantum_sim_results,
        classical_results=classical_sim_results,
        stats_results=stats_results,
        save_dir=comparison_dir / 'csv_results'
    )
    
    # Summary report
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
        f.write(f"Training order: Classical → Quantum\n\n")
        
        f.write("VITALDB TEST SET RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"Classical: {classical_vitaldb_results['mdape_mean']:.2f}%\n")
        f.write(f"Quantum:   {quantum_vitaldb_results['mdape_mean']:.2f}%\n\n")
        
        f.write("SIMULATOR TEST RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"MDAPE:\n")
        f.write(f"  Classical: {classical_sim_results['mdape_mean']:.2f}% ± {classical_sim_results['mdape_std']:.2f}%\n")
        f.write(f"  Quantum:   {quantum_sim_results['mdape_mean']:.2f}% ± {quantum_sim_results['mdape_std']:.2f}%\n\n")
        
        f.write(f"Wobble:\n")
        f.write(f"  Classical: {classical_sim_results['wobble_mean']:.2f} ± {classical_sim_results['wobble_std']:.2f}\n")
        f.write(f"  Quantum:   {quantum_sim_results['wobble_mean']:.2f} ± {quantum_sim_results['wobble_std']:.2f}\n\n")
        
        f.write(f"Divergence:\n")
        f.write(f"  Classical: {classical_sim_results['divergence_mean']:.4f}\n")
        f.write(f"  Quantum:   {quantum_sim_results['divergence_mean']:.4f}\n\n")
        
        f.write(f"Global Score:\n")
        f.write(f"  Classical: {classical_sim_results['global_score_mean']:.2f}\n")
        f.write(f"  Quantum:   {quantum_sim_results['global_score_mean']:.2f}\n\n")
        
        f.write(f"Reward:\n")
        f.write(f"  Classical: {classical_sim_results['reward_mean']:.2f}\n")
        f.write(f"  Quantum:   {quantum_sim_results['reward_mean']:.2f}\n\n")
        
        f.write(f"Time in Target (BIS 45-55):\n")
        f.write(f"  Classical: {classical_sim_results['time_in_target_mean']:.1f}%\n")
        f.write(f"  Quantum:   {quantum_sim_results['time_in_target_mean']:.1f}%\n\n")
        
        f.write(f"Propofol Usage:\n")
        f.write(f"  Classical: {classical_sim_results['propofol_usage_mean']:.2f} mg/kg/h\n")
        f.write(f"  Quantum:   {quantum_sim_results['propofol_usage_mean']:.2f} mg/kg/h\n\n")
        
        f.write(f"Remifentanil Usage:\n")
        f.write(f"  Classical: {classical_sim_results['remifentanil_usage_mean']:.2f} μg/kg/min\n")
        f.write(f"  Quantum:   {quantum_sim_results['remifentanil_usage_mean']:.2f} μg/kg/min\n\n")
        
        f.write("STATISTICAL TESTS\n")
        f.write("-"*70 + "\n")
        f.write(f"Winner: {stats_results['winner']}\n")
        f.write(f"Improvement: {stats_results['improvement_pct']:.2f}%\n")
        f.write(f"t-test p-value: {stats_results['t_pvalue']:.4f}\n")
        f.write(f"Cohen's d: {stats_results['cohens_d']:.4f}\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Summary report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\nTraining Order: Classical (faster) → Quantum (experimental)")
    print(f"\nSimulator Test Set:")
    print(f"  Classical MDAPE: {classical_sim_results['mdape_mean']:.2f}%")
    print(f"  Quantum MDAPE:   {quantum_sim_results['mdape_mean']:.2f}%")
    print(f"  Winner: {stats_results['winner']} ({stats_results['improvement_pct']:.2f}% improvement)")
    print(f"\n✓ All results saved to: {comparison_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
