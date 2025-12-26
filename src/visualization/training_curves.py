"""
Training Curves Visualization
==============================

This module provides utilities for plotting and analyzing training curves
from reinforcement learning experiments.

Features:
---------
- Offline training curves (BC loss, RL loss)
- Online training curves (reward, MDAPE, time in target)
- PPO-specific curves (policy loss, value loss, entropy)
- Multi-experiment comparison
- Smoothing and confidence intervals
- TensorBoard export (optional)

Usage:
------
# Single experiment
from visualization.training_curves import plot_training_curves
plot_training_curves('logs/experiment_20231221_120000')

# Compare multiple experiments
from visualization.training_curves import compare_experiments
compare_experiments([
    'logs/quantum_ddpg_20231221',
    'logs/classical_ddpg_20231221',
    'logs/quantum_ppo_20231221'
], save_path='comparison.png')

# Load from log file
from visualization.training_curves import load_training_log
log = load_training_log('logs/experiment/training_log.pkl')
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy.signal import savgol_filter


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_training_log(log_path: Union[str, Path]) -> Dict:
    """
    Load training log from pickle file.
    
    Args:
        log_path: Path to training_log.pkl file
    
    Returns:
        Dictionary containing training logs
    """
    log_path = Path(log_path)
    
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    with open(log_path, 'rb') as f:
        log = pickle.load(f)
    
    return log


def smooth_curve(
    data: np.ndarray,
    window_size: int = 11,
    polyorder: int = 3
) -> np.ndarray:
    """
    Smooth curve using Savitzky-Golay filter.
    
    Args:
        data: Data to smooth
        window_size: Window size (must be odd)
        polyorder: Polynomial order
    
    Returns:
        Smoothed data
    """
    if len(data) < window_size:
        return data
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    try:
        smoothed = savgol_filter(data, window_size, polyorder)
        return smoothed
    except:
        return data


def compute_confidence_interval(
    data: np.ndarray,
    window_size: int = 10,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling confidence interval.
    
    Args:
        data: Data array
        window_size: Window for rolling statistics
        confidence: Confidence level (0-1)
    
    Returns:
        (lower_bound, upper_bound) arrays
    """
    n = len(data)
    lower = np.zeros(n)
    upper = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size)
        window = data[start:end]
        
        mean = np.mean(window)
        std = np.std(window)
        margin = 1.96 * std  # 95% confidence
        
        lower[i] = mean - margin
        upper[i] = mean + margin
    
    return lower, upper


def plot_offline_training(
    log: Dict,
    save_path: Optional[Path] = None,
    smooth: bool = True
):
    """
    Plot offline training curves (BC loss, RL loss).
    
    Args:
        log: Training log dictionary
        save_path: Path to save figure
        smooth: Apply smoothing
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # BC loss (if available)
    if 'bc_loss' in log:
        bc_loss = np.array(log['bc_loss'])
        epochs = np.arange(len(bc_loss))
        
        axes[0, 0].plot(epochs, bc_loss, alpha=0.3, color='blue', label='Raw')
        if smooth:
            bc_smooth = smooth_curve(bc_loss)
            axes[0, 0].plot(epochs, bc_smooth, color='blue', linewidth=2, label='Smoothed')
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('BC Loss')
        axes[0, 0].set_title('Behavioral Cloning Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # RL loss (if available)
    if 'rl_loss' in log:
        rl_loss = np.array(log['rl_loss'])
        epochs = np.arange(len(rl_loss))
        
        axes[0, 1].plot(epochs, rl_loss, alpha=0.3, color='red', label='Raw')
        if smooth:
            rl_smooth = smooth_curve(rl_loss)
            axes[0, 1].plot(epochs, rl_smooth, color='red', linewidth=2, label='Smoothed')
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RL Loss')
        axes[0, 1].set_title('Reinforcement Learning Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Validation loss
    if 'val_loss' in log:
        val_loss = np.array(log['val_loss'])
        epochs = np.arange(len(val_loss))
        
        axes[1, 0].plot(epochs, val_loss, color='green', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].set_title('Validation Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Combined loss
    if 'combined_loss' in log:
        combined = np.array(log['combined_loss'])
        epochs = np.arange(len(combined))
        
        axes[1, 1].plot(epochs, combined, alpha=0.3, color='purple', label='Raw')
        if smooth:
            combined_smooth = smooth_curve(combined)
            axes[1, 1].plot(epochs, combined_smooth, color='purple', linewidth=2, label='Smoothed')
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Combined Loss')
        axes[1, 1].set_title('Combined Training Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_online_training(
    log: Dict,
    save_path: Optional[Path] = None,
    smooth: bool = True
):
    """
    Plot online training curves (reward, MDAPE, time in target).
    
    Args:
        log: Training log dictionary
        save_path: Path to save figure
        smooth: Apply smoothing
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    episodes = np.array(log.get('episodes', range(len(log.get('rewards', [])))))
    
    # Episode rewards
    if 'rewards' in log:
        rewards = np.array(log['rewards'])
        
        axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
        if smooth:
            rewards_smooth = smooth_curve(rewards)
            axes[0, 0].plot(episodes, rewards_smooth, color='blue', linewidth=2, label='Smoothed')
        
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # MDAPE
    if 'mdapes' in log:
        mdapes = np.array(log['mdapes'])
        
        axes[0, 1].plot(episodes, mdapes, alpha=0.3, color='red', label='Raw')
        if smooth:
            mdapes_smooth = smooth_curve(mdapes)
            axes[0, 1].plot(episodes, mdapes_smooth, color='red', linewidth=2, label='Smoothed')
        
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('MDAPE (%)')
        axes[0, 1].set_title('Performance Error (MDAPE)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Time in target
    if 'time_in_target' in log:
        time_in_target = np.array(log['time_in_target'])
        
        axes[0, 2].plot(episodes, time_in_target, alpha=0.3, color='green', label='Raw')
        if smooth:
            time_smooth = smooth_curve(time_in_target)
            axes[0, 2].plot(episodes, time_smooth, color='green', linewidth=2, label='Smoothed')
        
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Time in Target (%)')
        axes[0, 2].set_title('Time in Target Range (45-55 BIS)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Policy loss (PPO)
    if 'policy_losses' in log:
        policy_losses = np.array(log['policy_losses'])
        updates = np.arange(len(policy_losses))
        
        axes[1, 0].plot(updates, policy_losses, alpha=0.5, color='purple')
        if smooth:
            policy_smooth = smooth_curve(policy_losses)
            axes[1, 0].plot(updates, policy_smooth, color='purple', linewidth=2)
        
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Policy Loss')
        axes[1, 0].set_title('Policy Loss (PPO)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Value loss (PPO)
    if 'value_losses' in log:
        value_losses = np.array(log['value_losses'])
        updates = np.arange(len(value_losses))
        
        axes[1, 1].plot(updates, value_losses, alpha=0.5, color='orange')
        if smooth:
            value_smooth = smooth_curve(value_losses)
            axes[1, 1].plot(updates, value_smooth, color='orange', linewidth=2)
        
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Value Loss')
        axes[1, 1].set_title('Value Function Loss (PPO)')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Entropy (PPO)
    if 'entropies' in log:
        entropies = np.array(log['entropies'])
        updates = np.arange(len(entropies))
        
        axes[1, 2].plot(updates, entropies, alpha=0.5, color='brown')
        if smooth:
            entropy_smooth = smooth_curve(entropies)
            axes[1, 2].plot(updates, entropy_smooth, color='brown', linewidth=2)
        
        axes[1, 2].set_xlabel('Update')
        axes[1, 2].set_ylabel('Entropy')
        axes[1, 2].set_title('Policy Entropy (PPO)')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    log_dir: Union[str, Path],
    save_dir: Optional[Union[str, Path]] = None
):
    """
    Plot all training curves from a log directory.
    
    Args:
        log_dir: Directory containing training logs
        save_dir: Directory to save figures (defaults to log_dir/figures)
    """
    log_dir = Path(log_dir)
    
    if save_dir is None:
        save_dir = log_dir / 'figures'
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PLOTTING TRAINING CURVES")
    print(f"{'='*70}")
    print(f"Log directory: {log_dir}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*70}\n")
    
    # Look for training log file
    log_file = log_dir / 'training_log.pkl'
    
    if not log_file.exists():
        # Try stage-specific logs
        stage1_log = log_dir / 'stage1_offline' / 'bc_log.pkl'
        stage2_log = log_dir / 'stage2_online' / 'ppo_log.pkl'
        
        if stage1_log.exists():
            print("Plotting Stage 1 (Offline)...")
            log = load_training_log(stage1_log)
            plot_offline_training(log, save_dir / 'stage1_offline.png')
        
        if stage2_log.exists():
            print("Plotting Stage 2 (Online)...")
            log = load_training_log(stage2_log)
            plot_online_training(log, save_dir / 'stage2_online.png')
        
        if not stage1_log.exists() and not stage2_log.exists():
            print(f"Warning: No training log found in {log_dir}")
            return
    
    else:
        # Single training log
        print("Loading training log...")
        log = load_training_log(log_file)
        
        # Determine training type
        has_offline = any(key in log for key in ['bc_loss', 'rl_loss'])
        has_online = any(key in log for key in ['rewards', 'mdapes'])
        
        if has_offline:
            print("Plotting offline training curves...")
            plot_offline_training(log, save_dir / 'offline_training.png')
        
        if has_online:
            print("Plotting online training curves...")
            plot_online_training(log, save_dir / 'online_training.png')
    
    print(f"\n✓ Figures saved to: {save_dir}")
    print(f"{'='*70}\n")


def compare_experiments(
    log_dirs: List[Union[str, Path]],
    labels: Optional[List[str]] = None,
    metric: str = 'mdape',
    save_path: Optional[Union[str, Path]] = None
):
    """
    Compare multiple experiments on a single plot.
    
    Args:
        log_dirs: List of log directories
        labels: Labels for each experiment
        metric: Metric to compare ('mdape', 'reward', 'time_in_target')
        save_path: Path to save comparison plot
    """
    if labels is None:
        labels = [f"Exp {i+1}" for i in range(len(log_dirs))]
    
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_dirs)))
    
    for log_dir, label, color in zip(log_dirs, labels, colors):
        log_dir = Path(log_dir)
        log_file = log_dir / 'training_log.pkl'
        
        if not log_file.exists():
            print(f"Warning: Log file not found for {label}")
            continue
        
        log = load_training_log(log_file)
        
        if metric not in log:
            print(f"Warning: Metric '{metric}' not found in {label}")
            continue
        
        data = np.array(log[metric])
        episodes = np.array(log.get('episodes', range(len(data))))
        
        # Plot raw and smoothed
        plt.plot(episodes, data, alpha=0.2, color=color)
        data_smooth = smooth_curve(data)
        plt.plot(episodes, data_smooth, color=color, linewidth=2, label=label)
    
    plt.xlabel('Episode')
    plt.ylabel(metric.upper())
    plt.title(f'Comparison: {metric.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def export_to_tensorboard(
    log_dir: Union[str, Path],
    tensorboard_dir: Union[str, Path]
):
    """
    Export training logs to TensorBoard format.
    
    Args:
        log_dir: Directory containing training logs
        tensorboard_dir: Directory for TensorBoard logs
    
    Note:
        Requires: pip install tensorboard
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("Error: TensorBoard not installed. Install with: pip install tensorboard")
        return
    
    log_dir = Path(log_dir)
    tensorboard_dir = Path(tensorboard_dir)
    
    log_file = log_dir / 'training_log.pkl'
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return
    
    log = load_training_log(log_file)
    writer = SummaryWriter(tensorboard_dir)
    
    # Export metrics
    episodes = log.get('episodes', range(len(log.get('rewards', []))))
    
    for episode in episodes:
        idx = int(episode)
        
        if 'rewards' in log and idx < len(log['rewards']):
            writer.add_scalar('Train/Reward', log['rewards'][idx], episode)
        
        if 'mdapes' in log and idx < len(log['mdapes']):
            writer.add_scalar('Train/MDAPE', log['mdapes'][idx], episode)
        
        if 'time_in_target' in log and idx < len(log['time_in_target']):
            writer.add_scalar('Train/TimeInTarget', log['time_in_target'][idx], episode)
    
    writer.close()
    print(f"✓ Exported to TensorBoard: {tensorboard_dir}")
    print(f"  Run: tensorboard --logdir {tensorboard_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument('log_dir', type=str, help='Log directory')
    parser.add_argument('--save_dir', type=str, default=None, help='Save directory')
    parser.add_argument('--tensorboard', action='store_true', help='Export to TensorBoard')
    
    args = parser.parse_args()
    
    plot_training_curves(args.log_dir, args.save_dir)
    
    if args.tensorboard:
        tb_dir = Path(args.log_dir) / 'tensorboard'
        export_to_tensorboard(args.log_dir, tb_dir)
