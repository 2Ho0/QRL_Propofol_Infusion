"""
Clinical Visualization Tools
=============================

Publication-quality plots for clinical anesthesia control results.

Functions:
- plot_episode_trajectory: Single episode BIS/drug trajectory
- plot_training_curves: Training loss/reward curves
- plot_comparison: Multi-agent comparison plots
- plot_patient_variability: Patient population statistics
- plot_clinical_metrics: Safety/efficacy metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


# Set publication-quality defaults
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def plot_episode_trajectory(
    bis_history: np.ndarray,
    drug_rate_history: np.ndarray,
    reward_history: np.ndarray,
    target_bis: float = 50.0,
    save_path: Optional[Path] = None,
    title: str = "Episode Trajectory",
    show_zones: bool = True
):
    """
    Plot single episode trajectory showing BIS, drug infusion, and reward.
    
    Args:
        bis_history: Array of BIS values over time (T,)
        drug_rate_history: Array of drug infusion rates (T,) in mg/kg/h
        reward_history: Array of rewards (T,)
        target_bis: Target BIS value (default 50)
        save_path: Path to save figure (None = display only)
        title: Figure title
        show_zones: Whether to show clinical zones (too deep, target, too light)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    time_steps = np.arange(len(bis_history))
    
    # ========================================
    # BIS Trajectory
    # ========================================
    ax = axes[0]
    
    if show_zones:
        # Clinical zones
        ax.axhspan(0, 40, alpha=0.15, color='red', label='Too Deep (BIS < 40)')
        ax.axhspan(40, 60, alpha=0.15, color='green', label='Target Zone (40-60)')
        ax.axhspan(60, 100, alpha=0.2, color='yellow', label='Too Light (BIS > 60)')
    
    # BIS trajectory
    ax.plot(time_steps, bis_history, 'b-', linewidth=2, label='BIS')
    ax.axhline(target_bis, color='red', linestyle='--', linewidth=1.5, label=f'Target ({target_bis})')
    
    # Calculate MDAPE
    mdape = np.mean(np.abs(bis_history - target_bis) / target_bis) * 100
    ax.text(0.02, 0.98, f'MDAPE: {mdape:.2f}%', 
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylabel('BIS Index', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend(loc='upper right', ncol=2)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # ========================================
    # Drug Infusion Rate
    # ========================================
    ax = axes[1]
    ax.fill_between(time_steps, 0, drug_rate_history, alpha=0.3, color='purple')
    ax.plot(time_steps, drug_rate_history, 'purple', linewidth=2, label='Propofol Rate')
    
    # Statistics
    mean_rate = np.mean(drug_rate_history)
    max_rate = np.max(drug_rate_history)
    ax.axhline(mean_rate, color='orange', linestyle='--', linewidth=1.5, 
              label=f'Mean: {mean_rate:.2f} mg/kg/h')
    
    ax.text(0.02, 0.98, f'Max Rate: {max_rate:.2f} mg/kg/h\nMean Rate: {mean_rate:.2f} mg/kg/h',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylabel('Propofol Rate\n(mg/kg/h)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(drug_rate_history.max() * 1.1, 0.1)])
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    # ========================================
    # Reward
    # ========================================
    ax = axes[2]
    colors = ['green' if r >= 0 else 'red' for r in reward_history]
    ax.bar(time_steps, reward_history, color=colors, alpha=0.6, width=1.0)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    
    # Cumulative reward
    cumulative_reward = np.cumsum(reward_history)
    ax_twin = ax.twinx()
    ax_twin.plot(time_steps, cumulative_reward, 'b-', linewidth=2, label='Cumulative')
    ax_twin.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold', color='blue')
    ax_twin.tick_params(axis='y', labelcolor='blue')
    ax_twin.legend(loc='upper left')
    
    total_reward = cumulative_reward[-1]
    ax.text(0.98, 0.02, f'Total Reward: {total_reward:.2f}',
           transform=ax.transAxes, fontsize=11, horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylabel('Instant Reward', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved trajectory plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    train_history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    title: str = "Training Curves",
    smoothing_window: int = 10
):
    """
    Plot training loss and reward curves.
    
    Args:
        train_history: Dictionary with keys like 'actor_loss', 'critic_loss', 'reward'
        save_path: Path to save figure
        title: Figure title
        smoothing_window: Window size for moving average smoothing
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    def smooth(data, window):
        """Apply moving average smoothing."""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # ========================================
    # Actor Loss
    # ========================================
    ax = axes[0, 0]
    if 'actor_loss' in train_history:
        data = train_history['actor_loss']
        ax.plot(data, alpha=0.3, color='blue', label='Raw')
        if len(data) >= smoothing_window:
            smoothed = smooth(data, smoothing_window)
            ax.plot(range(smoothing_window-1, len(data)), smoothed, 
                   color='blue', linewidth=2, label='Smoothed')
        ax.set_ylabel('Actor Loss', fontsize=12)
        ax.set_title('Actor Loss', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # ========================================
    # Critic Loss
    # ========================================
    ax = axes[0, 1]
    if 'critic_loss' in train_history:
        data = train_history['critic_loss']
        ax.plot(data, alpha=0.3, color='red', label='Raw')
        if len(data) >= smoothing_window:
            smoothed = smooth(data, smoothing_window)
            ax.plot(range(smoothing_window-1, len(data)), smoothed,
                   color='red', linewidth=2, label='Smoothed')
        ax.set_ylabel('Critic Loss', fontsize=12)
        ax.set_title('Critic Loss', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # ========================================
    # Episode Reward
    # ========================================
    ax = axes[1, 0]
    if 'reward' in train_history:
        data = train_history['reward']
        ax.plot(data, alpha=0.3, color='green', label='Raw')
        if len(data) >= smoothing_window:
            smoothed = smooth(data, smoothing_window)
            ax.plot(range(smoothing_window-1, len(data)), smoothed,
                   color='green', linewidth=2, label='Smoothed')
        ax.set_ylabel('Episode Reward', fontsize=12)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_title('Episode Reward', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # ========================================
    # MDAPE
    # ========================================
    ax = axes[1, 1]
    if 'mdape' in train_history:
        data = train_history['mdape']
        ax.plot(data, alpha=0.3, color='purple', label='Raw')
        if len(data) >= smoothing_window:
            smoothed = smooth(data, smoothing_window)
            ax.plot(range(smoothing_window-1, len(data)), smoothed,
                   color='purple', linewidth=2, label='Smoothed')
        ax.set_ylabel('MDAPE (%)', fontsize=12)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_title('Mean Absolute Percentage Error', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'mdape',
    save_path: Optional[Path] = None,
    title: str = "Agent Comparison"
):
    """
    Plot comparison between multiple agents.
    
    Args:
        results: Dictionary mapping agent_name -> {'metric_mean': value, 'metric_std': value}
        metric: Metric name ('mdape', 'reward', etc.)
        save_path: Path to save figure
        title: Figure title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    agent_names = list(results.keys())
    means = [results[name][f'{metric}_mean'] for name in agent_names]
    stds = [results[name][f'{metric}_std'] for name in agent_names]
    
    # Bar plot with error bars
    x_pos = np.arange(len(agent_names))
    colors = plt.cm.Set3(np.linspace(0, 1, len(agent_names)))
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
               f'{mean:.2f}±{std:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agent_names, fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight best performer
    best_idx = np.argmin(means) if metric == 'mdape' else np.argmax(means)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_patient_variability(
    patient_results: List[Dict],
    save_path: Optional[Path] = None,
    title: str = "Patient Population Variability"
):
    """
    Plot patient-to-patient variability in outcomes.
    
    Args:
        patient_results: List of dicts with keys 'mdape', 'total_reward', 'patient_id'
        save_path: Path to save figure
        title: Figure title
    """
    df = pd.DataFrame(patient_results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # ========================================
    # MDAPE Distribution
    # ========================================
    ax = axes[0]
    ax.hist(df['mdape'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(df['mdape'].mean(), color='red', linestyle='--', linewidth=2, 
              label=f"Mean: {df['mdape'].mean():.2f}%")
    ax.axvline(df['mdape'].median(), color='orange', linestyle='--', linewidth=2,
              label=f"Median: {df['mdape'].median():.2f}%")
    ax.set_xlabel('MDAPE (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax.set_title('MDAPE Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # ========================================
    # Reward Distribution
    # ========================================
    ax = axes[1]
    ax.hist(df['total_reward'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.axvline(df['total_reward'].mean(), color='red', linestyle='--', linewidth=2,
              label=f"Mean: {df['total_reward'].mean():.2f}")
    ax.axvline(df['total_reward'].median(), color='orange', linestyle='--', linewidth=2,
              label=f"Median: {df['total_reward'].median():.2f}")
    ax.set_xlabel('Total Reward', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax.set_title('Reward Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved patient variability plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_clinical_metrics(
    metrics: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None,
    save_path: Optional[Path] = None,
    title: str = "Clinical Performance Metrics"
):
    """
    Plot clinical safety and efficacy metrics.
    
    Args:
        metrics: Dictionary of metric_name -> value
        thresholds: Optional dictionary of metric_name -> acceptable_threshold
        save_path: Path to save figure
        title: Figure title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = list(metrics.keys())
    values = list(metrics.values())
    
    # Color code by threshold if provided
    if thresholds:
        colors = []
        for name, value in zip(metric_names, values):
            if name in thresholds:
                if value <= thresholds[name]:
                    colors.append('green')
                else:
                    colors.append('red')
            else:
                colors.append('skyblue')
    else:
        colors = 'skyblue'
    
    # Horizontal bar plot
    y_pos = np.arange(len(metric_names))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f' {value:.2f}',
               va='center', fontsize=11, fontweight='bold')
    
    # Add threshold lines if provided
    if thresholds:
        for i, name in enumerate(metric_names):
            if name in thresholds:
                ax.axvline(thresholds[name], color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_names, fontsize=11)
    ax.set_xlabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved clinical metrics plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_learning_progress(
    checkpoints: List[Dict],
    save_path: Optional[Path] = None,
    title: str = "Learning Progress"
):
    """
    Plot learning progress across training checkpoints.
    
    Args:
        checkpoints: List of dicts with keys 'epoch', 'train_loss', 'val_loss', 'val_mdape'
        save_path: Path to save figure
        title: Figure title
    """
    df = pd.DataFrame(checkpoints)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # ========================================
    # Loss Curves
    # ========================================
    ax = axes[0]
    ax.plot(df['epoch'], df['train_loss'], 'b-o', linewidth=2, markersize=6, label='Train Loss')
    ax.plot(df['epoch'], df['val_loss'], 'r-s', linewidth=2, markersize=6, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # ========================================
    # MDAPE
    # ========================================
    ax = axes[1]
    ax.plot(df['epoch'], df['val_mdape'], 'g-^', linewidth=2, markersize=6, label='Val MDAPE')
    
    # Mark best epoch
    best_epoch = df.loc[df['val_mdape'].idxmin(), 'epoch']
    best_mdape = df['val_mdape'].min()
    ax.axvline(best_epoch, color='orange', linestyle='--', linewidth=1.5, 
              label=f'Best @ Epoch {best_epoch:.0f}')
    ax.axhline(best_mdape, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('MDAPE (%)', fontsize=12, fontweight='bold')
    ax.set_title('Validation MDAPE', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved learning progress plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ========================================
# Usage Examples
# ========================================

if __name__ == "__main__":
    """
    Example usage of visualization functions.
    """
    
    # Example 1: Episode Trajectory
    print("Example 1: Episode Trajectory")
    np.random.seed(42)
    T = 200
    bis = 50 + 10 * np.sin(np.linspace(0, 4*np.pi, T)) + np.random.randn(T) * 2
    drug = 5 + 2 * np.cos(np.linspace(0, 4*np.pi, T)) + np.abs(np.random.randn(T))
    reward = -(np.abs(bis - 50) / 50) - 0.01 * drug
    
    plot_episode_trajectory(
        bis_history=bis,
        drug_rate_history=drug,
        reward_history=reward,
        title="Example Episode",
        save_path=Path("example_trajectory.png")
    )
    
    # Example 2: Training Curves
    print("\nExample 2: Training Curves")
    n_epochs = 100
    train_history = {
        'actor_loss': np.random.exponential(1, n_epochs) + 0.1,
        'critic_loss': np.random.exponential(2, n_epochs) + 0.5,
        'reward': np.cumsum(np.random.randn(n_epochs) * 0.1) + 10,
        'mdape': 30 * np.exp(-np.linspace(0, 3, n_epochs)) + 5 + np.random.randn(n_epochs)
    }
    
    plot_training_curves(
        train_history=train_history,
        title="Training Progress",
        save_path=Path("example_training.png")
    )
    
    # Example 3: Agent Comparison
    print("\nExample 3: Agent Comparison")
    results = {
        'Quantum DDPG': {'mdape_mean': 12.5, 'mdape_std': 2.3},
        'Classical DDPG': {'mdape_mean': 15.8, 'mdape_std': 3.1},
        'Random': {'mdape_mean': 45.2, 'mdape_std': 8.7}
    }
    
    plot_comparison(
        results=results,
        metric='mdape',
        title="MDAPE Comparison",
        save_path=Path("example_comparison.png")
    )
    
    print("\n✓ All example plots generated!")
