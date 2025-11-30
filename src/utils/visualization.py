"""
Visualization Utilities for Propofol Infusion Control
======================================================

This module provides plotting functions for:
1. Episode visualization (BIS, dose, concentrations over time)
2. Training curves (rewards, losses, metrics)
3. Comparison plots (quantum vs classical, different patients)
4. Quantum circuit visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_episode(
    time: np.ndarray,
    bis: np.ndarray,
    dose: np.ndarray,
    ce: Optional[np.ndarray] = None,
    bis_target: float = 50.0,
    bis_range: Tuple[float, float] = (40.0, 60.0),
    title: str = "Episode Visualization",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a complete episode with BIS, dose, and optionally Ce.
    
    Args:
        time: Time array in seconds
        bis: BIS values
        dose: Propofol infusion rate (μg/kg/min)
        ce: Effect-site concentration (optional)
        bis_target: Target BIS value
        bis_range: Safe BIS range (low, high)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the figure
    
    Returns:
        Matplotlib figure
    """
    # Convert time to minutes for display
    time_min = time / 60.0
    
    # Create figure
    if ce is not None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Plot BIS
    ax1 = axes[0]
    ax1.plot(time_min, bis, 'b-', linewidth=1.5, label='BIS')
    ax1.axhline(y=bis_target, color='g', linestyle='--', 
                linewidth=1.5, label=f'Target ({bis_target})')
    ax1.axhspan(bis_range[0], bis_range[1], alpha=0.2, color='green',
                label=f'Target Range ({bis_range[0]}-{bis_range[1]})')
    ax1.axhline(y=40, color='orange', linestyle=':', alpha=0.7)
    ax1.axhline(y=60, color='orange', linestyle=':', alpha=0.7)
    ax1.set_ylabel('BIS', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper right')
    ax1.set_title('Bispectral Index (BIS)')
    ax1.grid(True, alpha=0.3)
    
    # Plot Dose
    ax2 = axes[1]
    ax2.fill_between(time_min, 0, dose, alpha=0.3, color='red')
    ax2.plot(time_min, dose, 'r-', linewidth=1.5, label='Propofol Dose')
    ax2.set_ylabel('Dose (μg/kg/min)', fontsize=12)
    ax2.set_ylim(0, max(dose.max() * 1.1, 200))
    ax2.legend(loc='upper right')
    ax2.set_title('Propofol Infusion Rate')
    ax2.grid(True, alpha=0.3)
    
    # Plot Ce if provided
    if ce is not None:
        ax3 = axes[2]
        ax3.plot(time_min, ce, 'm-', linewidth=1.5, label='Effect-site Concentration')
        ax3.axhline(y=3.4, color='gray', linestyle='--', 
                    linewidth=1, label='EC50 (3.4 μg/ml)')
        ax3.set_ylabel('Ce (μg/ml)', fontsize=12)
        ax3.set_xlabel('Time (min)', fontsize=12)
        ax3.legend(loc='upper right')
        ax3.set_title('Effect-site Concentration')
        ax3.grid(True, alpha=0.3)
    else:
        ax2.set_xlabel('Time (min)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_training_curves(
    rewards: List[float],
    mdape: Optional[List[float]] = None,
    time_in_target: Optional[List[float]] = None,
    actor_losses: Optional[List[float]] = None,
    critic_losses: Optional[List[float]] = None,
    window: int = 50,
    title: str = "Training Progress",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training curves with moving averages.
    
    Args:
        rewards: Episode rewards
        mdape: MDAPE for each episode
        time_in_target: Time in target for each episode
        actor_losses: Actor loss values
        critic_losses: Critic loss values
        window: Window size for moving average
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
    
    Returns:
        Matplotlib figure
    """
    n_plots = 1 + (mdape is not None) + (time_in_target is not None) + \
              (actor_losses is not None or critic_losses is not None)
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    episodes = np.arange(len(rewards))
    ax_idx = 0
    
    def moving_average(values, w):
        if len(values) < w:
            return values
        return np.convolve(values, np.ones(w) / w, mode='valid')
    
    # Rewards
    ax = axes[ax_idx]
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
    if len(rewards) >= window:
        ma = moving_average(rewards, window)
        ax.plot(episodes[window-1:], ma, color='blue', linewidth=2, 
                label=f'MA({window})')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Training Reward')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax_idx += 1
    
    # MDAPE
    if mdape is not None:
        ax = axes[ax_idx]
        ax.plot(episodes, mdape, alpha=0.3, color='orange', label='Raw')
        if len(mdape) >= window:
            ma = moving_average(mdape, window)
            ax.plot(episodes[window-1:], ma, color='orange', linewidth=2,
                    label=f'MA({window})')
        ax.axhline(y=20, color='green', linestyle='--', label='Clinical Threshold (20%)')
        ax.set_ylabel('MDAPE (%)')
        ax.set_title('Median Absolute Performance Error')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax_idx += 1
    
    # Time in Target
    if time_in_target is not None:
        ax = axes[ax_idx]
        ax.plot(episodes, time_in_target, alpha=0.3, color='green', label='Raw')
        if len(time_in_target) >= window:
            ma = moving_average(time_in_target, window)
            ax.plot(episodes[window-1:], ma, color='green', linewidth=2,
                    label=f'MA({window})')
        ax.axhline(y=80, color='blue', linestyle='--', label='Target (80%)')
        ax.set_ylabel('Time in Target (%)')
        ax.set_title('Time in Target Range')
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax_idx += 1
    
    # Losses
    if actor_losses is not None or critic_losses is not None:
        ax = axes[ax_idx]
        if critic_losses is not None:
            ax.plot(critic_losses, alpha=0.5, color='red', label='Critic Loss')
        if actor_losses is not None:
            ax.plot(actor_losses, alpha=0.5, color='purple', label='Actor Loss')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    axes[-1].set_xlabel('Episode')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_comparison(
    results: Dict[str, Dict],
    metric: str = 'reward',
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare multiple methods/experiments.
    
    Args:
        results: Dictionary mapping method name to results dict
        metric: Which metric to compare
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for idx, (name, data) in enumerate(results.items()):
        values = data.get(metric, [])
        if len(values) > 0:
            episodes = np.arange(len(values))
            ax.plot(episodes, values, color=colors[idx], alpha=0.3)
            
            # Moving average
            window = min(50, len(values) // 5)
            if window > 1:
                ma = np.convolve(values, np.ones(window) / window, mode='valid')
                ax.plot(episodes[window-1:], ma, color=colors[idx], 
                        linewidth=2, label=name)
            else:
                ax.plot(episodes, values, color=colors[idx],
                        linewidth=2, label=name)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_patient_variability(
    patient_results: List[Dict],
    metric: str = 'mdape',
    title: str = "Patient Variability Analysis",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot performance across different patients.
    
    Args:
        patient_results: List of result dicts for each patient
        metric: Which metric to analyze
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    values = [r.get(metric, 0) for r in patient_results]
    patient_ids = list(range(len(patient_results)))
    
    # Bar plot
    ax1 = axes[0]
    bars = ax1.bar(patient_ids, values, color='steelblue', alpha=0.7)
    ax1.axhline(y=np.mean(values), color='red', linestyle='--', 
                label=f'Mean: {np.mean(values):.2f}')
    ax1.set_xlabel('Patient ID')
    ax1.set_ylabel(metric.upper())
    ax1.set_title(f'{metric.upper()} by Patient')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot(values, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_ylabel(metric.upper())
    ax2.set_title(f'{metric.upper()} Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics annotation
    stats_text = f"Mean: {np.mean(values):.2f}\n" \
                 f"Std: {np.std(values):.2f}\n" \
                 f"Min: {np.min(values):.2f}\n" \
                 f"Max: {np.max(values):.2f}"
    ax2.text(1.15, np.median(values), stats_text, 
             transform=ax2.get_yaxis_transform(),
             fontsize=10, verticalalignment='center')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_reward_components(
    bis_errors: np.ndarray,
    dose_penalties: np.ndarray,
    safety_penalties: np.ndarray,
    total_rewards: np.ndarray,
    time: np.ndarray,
    title: str = "Reward Components",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize reward components over an episode.
    
    Args:
        bis_errors: BIS error component
        dose_penalties: Dose penalty component
        safety_penalties: Safety penalty component
        total_rewards: Total reward
        time: Time array
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
    
    Returns:
        Matplotlib figure
    """
    time_min = time / 60.0
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Stacked area for components
    ax1 = axes[0]
    ax1.fill_between(time_min, 0, bis_errors, alpha=0.5, 
                      label='BIS Error', color='blue')
    ax1.fill_between(time_min, bis_errors, bis_errors + dose_penalties, 
                      alpha=0.5, label='Dose Penalty', color='orange')
    ax1.fill_between(time_min, bis_errors + dose_penalties,
                      bis_errors + dose_penalties + safety_penalties,
                      alpha=0.5, label='Safety Penalty', color='red')
    ax1.set_ylabel('Reward Component')
    ax1.set_title('Reward Component Breakdown')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Total reward
    ax2 = axes[1]
    ax2.plot(time_min, total_rewards, 'g-', linewidth=1.5)
    ax2.fill_between(time_min, total_rewards, 0, alpha=0.3, color='green')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Cumulative Reward')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def create_summary_figure(
    episode_data: Dict,
    metrics: Dict,
    title: str = "Episode Summary",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a comprehensive summary figure for an episode.
    
    Args:
        episode_data: Dict with 'time', 'bis', 'dose', 'ce' arrays
        metrics: Dict with performance metrics
        title: Figure title
        save_path: Path to save figure
        show: Whether to display
    
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    time_min = episode_data['time'] / 60.0
    bis = episode_data['bis']
    dose = episode_data['dose']
    ce = episode_data.get('ce')
    
    # Main BIS plot
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(time_min, bis, 'b-', linewidth=1.5)
    ax1.axhline(y=50, color='g', linestyle='--', linewidth=1.5)
    ax1.axhspan(40, 60, alpha=0.2, color='green')
    ax1.set_ylabel('BIS')
    ax1.set_title('Bispectral Index')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Dose plot
    ax2 = fig.add_subplot(gs[1, :2], sharex=ax1)
    ax2.fill_between(time_min, 0, dose, alpha=0.3, color='red')
    ax2.plot(time_min, dose, 'r-', linewidth=1.5)
    ax2.set_ylabel('Dose (μg/kg/min)')
    ax2.set_title('Propofol Infusion Rate')
    ax2.grid(True, alpha=0.3)
    
    # Ce plot (if available)
    if ce is not None:
        ax3 = fig.add_subplot(gs[2, :2], sharex=ax1)
        ax3.plot(time_min, ce, 'm-', linewidth=1.5)
        ax3.axhline(y=3.4, color='gray', linestyle='--')
        ax3.set_ylabel('Ce (μg/ml)')
        ax3.set_xlabel('Time (min)')
        ax3.set_title('Effect-site Concentration')
        ax3.grid(True, alpha=0.3)
    else:
        ax2.set_xlabel('Time (min)')
    
    # Metrics panel
    ax4 = fig.add_subplot(gs[:2, 2])
    ax4.axis('off')
    
    metrics_text = [
        f"MDPE: {metrics.get('mdpe', 0):+.2f}%",
        f"MDAPE: {metrics.get('mdape', 0):.2f}%",
        f"Wobble: {metrics.get('wobble', 0):.2f}%",
        f"Time in Target: {metrics.get('time_in_target', 0):.1f}%",
        f"Mean Dose: {metrics.get('mean_dose', 0):.1f} μg/kg/min",
        f"Total Dose: {metrics.get('total_dose', 0):.2f} mg/kg"
    ]
    
    text_y = 0.9
    ax4.text(0.1, text_y + 0.05, "Performance Metrics", fontsize=12, 
             fontweight='bold', transform=ax4.transAxes)
    for i, text in enumerate(metrics_text):
        ax4.text(0.15, text_y - i * 0.1, text, fontsize=11, 
                 transform=ax4.transAxes)
    
    # BIS distribution
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.hist(bis, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
    ax5.axvline(x=50, color='g', linestyle='--', linewidth=2)
    ax5.axvline(x=40, color='orange', linestyle=':', linewidth=1.5)
    ax5.axvline(x=60, color='orange', linestyle=':', linewidth=1.5)
    ax5.set_xlabel('BIS')
    ax5.set_ylabel('Count')
    ax5.set_title('BIS Distribution')
    ax5.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Test visualization functions
    print("Testing Visualization Utilities...")
    
    # Generate test data
    np.random.seed(42)
    n_steps = 720
    
    time = np.arange(n_steps) * 5.0
    bis = 97 - 47 * (1 - np.exp(-time / 300)) + np.random.randn(n_steps) * 3
    bis = np.clip(bis, 0, 100)
    dose = 150 * (1 - np.exp(-time / 100)) + np.random.randn(n_steps) * 10
    dose = np.clip(dose, 0, 200)
    ce = 4 * (1 - np.exp(-time / 200)) + np.random.randn(n_steps) * 0.3
    ce = np.clip(ce, 0, 10)
    
    # Test episode plot
    print("Creating episode plot...")
    plot_episode(time, bis, dose, ce, save_path=None, show=False)
    
    # Test training curves
    print("Creating training curves...")
    rewards = [np.random.randn() * 10 + i * 0.01 for i in range(200)]
    mdape = [30 - i * 0.1 + np.random.randn() * 3 for i in range(200)]
    plot_training_curves(rewards, mdape=mdape, save_path=None, show=False)
    
    # Test summary figure
    print("Creating summary figure...")
    episode_data = {'time': time, 'bis': bis, 'dose': dose, 'ce': ce}
    metrics = {
        'mdpe': -5.2,
        'mdape': 12.3,
        'wobble': 8.1,
        'time_in_target': 72.5,
        'mean_dose': 95.3,
        'total_dose': 5.72
    }
    create_summary_figure(episode_data, metrics, save_path=None, show=False)
    
    print("All visualization tests passed!")
