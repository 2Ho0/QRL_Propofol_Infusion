"""
Clinical Visualization Tools
=============================

Publication-quality plots for clinical anesthesia control results.

Functions:
- plot_episode_trajectory: Single episode BIS/drug trajectory
- plot_training_curves: Training loss/reward curves
- plot_comparison: Multi-agent comparison plots (Bar/Box/Violin)
- plot_patient_variability: Patient population statistics
- plot_clinical_metrics: Safety/efficacy metrics
- plot_radar_chart: Multi-metric radar comparison
- plot_poincare: Stability visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from math import pi


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
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    time_steps = np.arange(len(bis_history))
    
    # ========================================
    # BIS Trajectory
    # ========================================
    ax = axes[0]
    
    if show_zones:
        # Clinical zones
        ax.axhspan(0, 40, alpha=0.15, color='red', label='Too Deep (< 40)')
        ax.axhspan(40, 60, alpha=0.15, color='green', label='Target (40-60)')
        ax.axhspan(60, 100, alpha=0.2, color='yellow', label='Too Light (> 60)')
    
    # BIS trajectory
    ax.plot(time_steps, bis_history, 'b-', linewidth=2, label='BIS')
    ax.axhline(target_bis, color='red', linestyle='--', linewidth=1.5, label=f'Target ({target_bis})')
    
    # Calculate MDAPE
    mdape = np.mean(np.abs(bis_history - target_bis) / target_bis) * 100
    ax.text(0.02, 0.98, f'MDAPE: {mdape:.2f}%', 
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_ylabel('BIS', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend(loc='upper right', ncol=2)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # ========================================
    # Drug Infusion Rate
    # ========================================
    ax = axes[1]
    
    # Handle multi-drug (dual drug)
    if drug_rate_history.ndim > 1 and drug_rate_history.shape[1] == 2:
        # Dual drug
        ppf = drug_rate_history[:, 0]
        remi = drug_rate_history[:, 1]
        
        ax.plot(time_steps, ppf, 'purple', linewidth=2, label='Propofol (mg/kg/h)')
        ax.fill_between(time_steps, 0, ppf, alpha=0.2, color='purple')
        
        ax2 = ax.twinx()
        ax2.plot(time_steps, remi, 'orange', linewidth=2, label='Remifentanil (ug/kg/min)')
        ax2.fill_between(time_steps, 0, remi, alpha=0.2, color='orange')
        ax2.set_ylabel('Remifentanil (ug/kg/min)', fontsize=12, fontweight='bold', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Combined legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right')
        
    else:
        # Single drug (Propofol)
        if drug_rate_history.ndim > 1:
            drug_rate_history = drug_rate_history.flatten()
            
        ax.fill_between(time_steps, 0, drug_rate_history, alpha=0.3, color='purple')
        ax.plot(time_steps, drug_rate_history, 'purple', linewidth=2, label='Propofol Rate')
        ax.legend(loc='upper right')

    ax.set_ylabel('Propofol (mg/kg/h)', fontsize=12, fontweight='bold', color='purple')
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
    
    ax.set_ylabel('Instant Reward', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Step (min)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved trajectory plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(
    results: Dict[str, Dict[str, Union[float, List[float]]]],
    metric: str = 'mdape',
    plot_type: str = 'box',  # 'bar', 'box', 'violin'
    save_path: Optional[Path] = None,
    title: str = "Agent Comparison"
):
    """
    Plot comparison between multiple agents.
    
    Args:
        results: Dictionary mapping agent_name -> {'metric_mean': val, 'metric_list': [vals...]}
        metric: Metric name ('mdape', 'reward', 'wobble', etc.)
        plot_type: 'bar', 'box', or 'violin'
        save_path: Path to save figure
        title: Figure title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    agent_names = list(results.keys())
    
    # Check if we have list data for box/violin plots
    has_lists = all(f'{metric}_list' in results[name] for name in agent_names)
    
    if not has_lists or plot_type == 'bar':
        # Fallback to bar plot if no list data
        means = [results[name].get(f'{metric}_mean', 0) for name in agent_names]
        stds = [results[name].get(f'{metric}_std', 0) for name in agent_names]
        
        x_pos = np.arange(len(agent_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(agent_names)))
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{mean:.2f}±{std:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    elif plot_type == 'box':
        data = [results[name][f'{metric}_list'] for name in agent_names]
        
        # Create boxplot
        bplot = ax.boxplot(data, patch_artist=True, labels=agent_names, 
                          medianprops=dict(color="black", linewidth=1.5))
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(agent_names)))
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        # Add swarmplot overlay for data points
        for i, d in enumerate(data):
            y = d
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.plot(x, y, 'k.', alpha=0.3)
            
    elif plot_type == 'violin':
        data = [results[name][f'{metric}_list'] for name in agent_names]
        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(agent_names)))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
            
        # Set labels
        ax.set_xticks(np.arange(1, len(agent_names) + 1))
        ax.set_xticklabels(agent_names)

    if plot_type == 'bar':
        ax.set_xticks(range(len(agent_names)))
        ax.set_xticklabels(agent_names, fontsize=12, fontweight='bold')
    
    ax.set_ylabel(metric.upper().replace('_', ' '), fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot ({plot_type}) to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_radar_chart(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    save_path: Optional[Path] = None,
    title: str = "Multi-Metric Comparison"
):
    """
    Plot radar chart for multi-objective comparison.
    
    Args:
        results: Dict mapping agent_name -> {metric: value}
        metrics: List of metrics to include in radar (must be normalized ideally)
        save_path: Path to save
    """
    # Number of variables
    N = len(metrics)
    
    # What will be the angle of each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], [m.upper().replace('_', '\n') for m in metrics], size=10)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    
    # Normalize data for radar chart (0-1 scaling relative to max in set)
    # This is critical because metrics have different scales
    # We invert metrics where lower is better (MDAPE, Wobble)
    
    # Define which metrics are "lower is better"
    lower_is_better = ['mdape', 'mdpe', 'wobble', 'induction_time', 'drug_consumption']
    
    # Gather raw values
    raw_values = {m: [] for m in metrics}
    for agent in results:
        for m in metrics:
            val = results[agent].get(f'{m}_mean', results[agent].get(m, 0))
            raw_values[m].append(abs(val))
            
    # Find max for normalization
    max_values = {m: max(raw_values[m]) if raw_values[m] else 1.0 for m in metrics}
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for i, (agent_name, agent_data) in enumerate(results.items()):
        values = []
        for m in metrics:
            val = abs(agent_data.get(f'{m}_mean', agent_data.get(m, 0)))
            norm_val = val / max_values[m] if max_values[m] > 0 else 0
            
            # Invert if lower is better
            if any(lb in m.lower() for lb in lower_is_better):
                values.append(1.0 - norm_val + 0.1) # Add epsilon to avoid minimal
            else:
                values.append(norm_val)
                
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=agent_name, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)
        
    plt.title(title, size=15, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved radar chart to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_poincare(
    bis_history: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Poincaré Plot (BIS Stability)"
):
    """
    Plot Poincaré plot (BIS[t] vs BIS[t+1]) to visualize stability/chaos.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    x = bis_history[:-1]
    y = bis_history[1:]
    
    ax.scatter(x, y, alpha=0.5, s=10, c='blue')
    
    # Plot identity line
    min_val, max_val = min(min(x), min(y)), max(max(x), max(y))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Identity')
    
    # Calculate SD1, SD2 (Variability metrics)
    sd1 = np.std(np.subtract(x, y) / np.sqrt(2))
    sd2 = np.std(np.add(x, y) / np.sqrt(2))
    
    ax.text(0.05, 0.95, f'SD1 (Short-term): {sd1:.2f}\nSD2 (Long-term): {sd2:.2f}',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('BIS(t)', fontsize=12, fontweight='bold')
    ax.set_ylabel('BIS(t+1)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Poincaré plot to {save_path}")
    else:
        plt.show()
    
    plt.close()
