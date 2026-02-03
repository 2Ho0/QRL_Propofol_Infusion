"""
Evaluate best models on training environment and visualize BIS trajectories
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml

from agents.quantum_agent import QuantumDDPGAgent
from agents.classical_agent import ClassicalDDPGAgent
from environment.dual_drug_env import DualDrugEnv

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


def evaluate_single_episode(agent, env, device, max_steps=360):
    """Evaluate agent for one episode and return detailed trajectory"""
    state, _ = env.reset()
    
    trajectory = {
        'time': [],
        'bis': [],
        'bis_target': [],
        'propofol_action': [],
        'remifentanil_action': [],
        'propofol_ce': [],
        'remifentanil_ce': [],
        'reward': [],
        'bis_error': []
    }
    
    states_list = []
    
    for step in range(max_steps):
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
                if encoded.dim() == 3:
                    encoded = encoded[:, -1, :]
                action = agent.actor(encoded)
            else:
                action = agent.actor(state_tensor)
        
        action_np = action.cpu().numpy().flatten()
        if action_np.shape[0] != 2:
            action_np = action_np[:2]
        
        # Scale to physical units
        action_physical = np.array([
            action_np[0] * 30.0,  # Propofol [0,30 mg/kg/h]
            action_np[1] * 1.0    # Remifentanil [0,1.0 μg/kg/min]
        ])
        
        next_state, reward, done, truncated, info = env.step(action_physical)
        
        # Extract info
        bis = info.get('bis', 50 - state[0])
        propofol_ce = info.get('propofol_ce', state[1] if len(state) > 1 else 0)
        remi_ce = info.get('remifentanil_ce', state[2] if len(state) > 2 else 0)
        
        # Store trajectory
        trajectory['time'].append(step)
        trajectory['bis'].append(bis)
        trajectory['bis_target'].append(50)
        trajectory['propofol_action'].append(action_physical[0])
        trajectory['remifentanil_action'].append(action_physical[1])
        trajectory['propofol_ce'].append(propofol_ce)
        trajectory['remifentanil_ce'].append(remi_ce)
        trajectory['reward'].append(reward)
        trajectory['bis_error'].append(abs(bis - 50))
        
        states_list.append(state)
        state = next_state
        
        if done or truncated:
            break
    
    # Calculate metrics
    bis_array = np.array(trajectory['bis'])
    trajectory['mdape'] = np.mean(np.abs(bis_array - 50) / 50) * 100
    trajectory['time_in_target'] = np.mean((bis_array >= 45) & (bis_array <= 55)) * 100
    trajectory['total_reward'] = sum(trajectory['reward'])
    trajectory['avg_propofol'] = np.mean(trajectory['propofol_action'])
    trajectory['avg_remifentanil'] = np.mean(trajectory['remifentanil_action'])
    
    return trajectory


def plot_comparison(classical_traj, quantum_traj, save_dir):
    """Create comprehensive comparison plots"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Convert to arrays
    c_time = np.array(classical_traj['time'])
    c_bis = np.array(classical_traj['bis'])
    q_time = np.array(quantum_traj['time'])
    q_bis = np.array(quantum_traj['bis'])
    
    # 1. BIS Trajectory Comparison
    ax = axes[0, 0]
    ax.plot(c_time, c_bis, 'b-', label='Classical', linewidth=2, alpha=0.8)
    ax.plot(q_time, q_bis, 'r-', label='Quantum', linewidth=2, alpha=0.8)
    ax.axhline(y=50, color='green', linestyle='--', linewidth=2, label='Target BIS=50')
    ax.fill_between(c_time, 45, 55, color='green', alpha=0.1, label='Target Zone')
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('BIS Value', fontsize=12)
    ax.set_title('BIS Control Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(20, 80)
    
    # 2. BIS Error
    ax = axes[0, 1]
    ax.plot(c_time, classical_traj['bis_error'], 'b-', label='Classical', linewidth=2, alpha=0.8)
    ax.plot(q_time, quantum_traj['bis_error'], 'r-', label='Quantum', linewidth=2, alpha=0.8)
    ax.axhline(y=5, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='±5 threshold')
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('|BIS - 50|', fontsize=12)
    ax.set_title('BIS Absolute Error Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # 3. Propofol Actions
    ax = axes[1, 0]
    ax.plot(c_time, classical_traj['propofol_action'], 'b-', label='Classical', linewidth=2, alpha=0.8)
    ax.plot(q_time, quantum_traj['propofol_action'], 'r-', label='Quantum', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Propofol Rate (mg/kg/h)', fontsize=12)
    ax.set_title('Propofol Infusion Rate', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # 4. Remifentanil Actions
    ax = axes[1, 1]
    ax.plot(c_time, classical_traj['remifentanil_action'], 'b-', label='Classical', linewidth=2, alpha=0.8)
    ax.plot(q_time, quantum_traj['remifentanil_action'], 'r-', label='Quantum', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Remifentanil Rate (μg/kg/min)', fontsize=12)
    ax.set_title('Remifentanil Infusion Rate', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # 5. Propofol Ce
    ax = axes[2, 0]
    ax.plot(c_time, classical_traj['propofol_ce'], 'b-', label='Classical', linewidth=2, alpha=0.8)
    ax.plot(q_time, quantum_traj['propofol_ce'], 'r-', label='Quantum', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Propofol Ce (μg/ml)', fontsize=12)
    ax.set_title('Propofol Effect-Site Concentration', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # 6. Cumulative Reward
    ax = axes[2, 1]
    c_cumulative_reward = np.cumsum(classical_traj['reward'])
    q_cumulative_reward = np.cumsum(quantum_traj['reward'])
    ax.plot(c_time, c_cumulative_reward, 'b-', label='Classical', linewidth=2, alpha=0.8)
    ax.plot(q_time, q_cumulative_reward, 'r-', label='Quantum', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title('Cumulative Reward Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'best_models_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {save_dir / 'best_models_comparison.png'}")
    plt.close()


def create_summary_table(classical_traj, quantum_traj, save_dir):
    """Create summary statistics table"""
    summary = {
        'Metric': [
            'MDAPE (%)',
            'Time in Target (%)',
            'Total Reward',
            'Avg Propofol (mg/kg/h)',
            'Avg Remifentanil (μg/kg/min)',
            'Final BIS',
            'Max BIS Error',
            'Mean BIS Error'
        ],
        'Classical': [
            f"{classical_traj['mdape']:.2f}",
            f"{classical_traj['time_in_target']:.2f}",
            f"{classical_traj['total_reward']:.2f}",
            f"{classical_traj['avg_propofol']:.2f}",
            f"{classical_traj['avg_remifentanil']:.3f}",
            f"{classical_traj['bis'][-1]:.2f}",
            f"{max(classical_traj['bis_error']):.2f}",
            f"{np.mean(classical_traj['bis_error']):.2f}"
        ],
        'Quantum': [
            f"{quantum_traj['mdape']:.2f}",
            f"{quantum_traj['time_in_target']:.2f}",
            f"{quantum_traj['total_reward']:.2f}",
            f"{quantum_traj['avg_propofol']:.2f}",
            f"{quantum_traj['avg_remifentanil']:.3f}",
            f"{quantum_traj['bis'][-1]:.2f}",
            f"{max(quantum_traj['bis_error']):.2f}",
            f"{np.mean(quantum_traj['bis_error']):.2f}"
        ]
    }
    
    df = pd.DataFrame(summary)
    
    # Save to CSV
    csv_path = save_dir / 'summary_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved summary metrics: {csv_path}")
    
    # Print to console
    print("\n" + "="*70)
    print("SUMMARY METRICS (Training Environment)")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    return df


def save_trajectories_csv(classical_traj, quantum_traj, save_dir):
    """Save detailed trajectories to CSV"""
    save_dir = Path(save_dir)
    
    # Classical
    c_df = pd.DataFrame({
        'time': classical_traj['time'],
        'bis': classical_traj['bis'],
        'propofol_action': classical_traj['propofol_action'],
        'remifentanil_action': classical_traj['remifentanil_action'],
        'propofol_ce': classical_traj['propofol_ce'],
        'reward': classical_traj['reward'],
        'bis_error': classical_traj['bis_error']
    })
    c_df.to_csv(save_dir / 'classical_trajectory.csv', index=False)
    
    # Quantum
    q_df = pd.DataFrame({
        'time': quantum_traj['time'],
        'bis': quantum_traj['bis'],
        'propofol_action': quantum_traj['propofol_action'],
        'remifentanil_action': quantum_traj['remifentanil_action'],
        'propofol_ce': quantum_traj['propofol_ce'],
        'reward': quantum_traj['reward'],
        'bis_error': quantum_traj['bis_error']
    })
    q_df.to_csv(save_dir / 'quantum_trajectory.csv', index=False)
    
    print(f"✓ Saved trajectories: {save_dir}")


def main():
    # Use CPU to avoid GPU memory issues
    device = torch.device('cpu')
    print(f"Using device: {device} (to avoid GPU memory conflicts)")
    
    # Paths
    comparison_dir = Path('/home/mwilliam/QRL_Propofol_Infusion/logs/comparison_20260115_214922')
    classical_best = comparison_dir / 'classical' / 'stage2_online' / 'best_mdape.pt'
    classical_reward = comparison_dir / 'classical' / 'stage2_online' / 'best_reward.pt'
    quantum_best = comparison_dir / 'quantum' / 'stage2_online' / 'best_mdape.pt'
    
    output_dir = comparison_dir / 'best_models_train_env_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open('/home/mwilliam/QRL_Propofol_Infusion/config/hyperparameters.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    state_dim = 13
    action_dim = 2
    
    print("\n" + "="*70)
    print("EVALUATING BEST MODELS ON TRAINING ENVIRONMENT")
    print("="*70)
    
    # Training environment (same seed as used in training)
    training_seed = 1042  # Default seed from compare script
    # Create a patient for consistency (seed matches training)
    from environment.patient_simulator import create_patient_population
    train_patient = create_patient_population(1, seed=training_seed)[0]
    env = DualDrugEnv(patient=train_patient, seed=training_seed)
    
    print(f"\nEnvironment: DualDrugEnv (seed={training_seed})")
    print(f"Patient: Same as used during training (1 patient repeated)")
    
    # ===== Classical Agent =====
    print("\n" + "-"*70)
    print("CLASSICAL AGENT")
    print("-"*70)
    
    # Check encoder type from checkpoint
    classical_checkpoint = torch.load(str(classical_best), map_location='cpu')
    
    # Detect encoder type from state dict keys
    encoder_keys = [k for k in classical_checkpoint.get('encoder_state_dict', {}).keys()]
    if any('transformer' in k for k in encoder_keys):
        encoder_type_classical = 'transformer'
    elif any('lstm' in k for k in encoder_keys):
        encoder_type_classical = 'lstm'
    else:
        encoder_type_classical = None
    
    print(f"Detected encoder type: {encoder_type_classical}")
    if encoder_keys:
        print(f"Sample encoder keys: {encoder_keys[:3]}")
    
    classical_agent = ClassicalDDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        encoder_type=encoder_type_classical,
        seed=42
    )
    classical_agent.actor = classical_agent.actor.to(device)
    classical_agent.critic = classical_agent.critic.to(device)
    if classical_agent.encoder is not None:
        classical_agent.encoder = classical_agent.encoder.to(device)
    
    if classical_best.exists():
        # Load to CPU explicitly
        checkpoint = torch.load(str(classical_best), map_location=device)
        classical_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        classical_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        if classical_agent.encoder is not None and 'encoder_state_dict' in checkpoint:
            classical_agent.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(f"✓ Loaded: {classical_best}")
    else:
        print(f"❌ Not found: {classical_best}")
        return
    
    print("Evaluating...")
    classical_traj = evaluate_single_episode(classical_agent, env, device)
    
    print(f"  MDAPE: {classical_traj['mdape']:.2f}%")
    print(f"  Reward: {classical_traj['total_reward']:.2f}")
    print(f"  Time in Target: {classical_traj['time_in_target']:.2f}%")
    
    # ===== Quantum Agent =====
    print("\n" + "-"*70)
    print("QUANTUM AGENT")
    print("-"*70)
    
    # Check encoder type from checkpoint
    quantum_checkpoint = torch.load(str(quantum_best), map_location='cpu')
    
    # Detect encoder type from state dict keys
    encoder_keys = [k for k in quantum_checkpoint.get('encoder_state_dict', {}).keys()]
    if any('transformer' in k for k in encoder_keys):
        encoder_type_quantum = 'transformer'
    elif any('lstm' in k for k in encoder_keys):
        encoder_type_quantum = 'lstm'
    else:
        encoder_type_quantum = None
    
    print(f"Detected encoder type: {encoder_type_quantum}")
    
    quantum_agent = QuantumDDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        encoder_type=encoder_type_quantum,
        seed=1042
    )
    quantum_agent.actor = quantum_agent.actor.to(device)
    quantum_agent.critic = quantum_agent.critic.to(device)
    if quantum_agent.encoder is not None:
        quantum_agent.encoder = quantum_agent.encoder.to(device)
    
    if quantum_best.exists():
        # Load to CPU explicitly
        checkpoint = torch.load(str(quantum_best), map_location=device)
        quantum_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        quantum_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        if quantum_agent.encoder is not None and 'encoder_state_dict' in checkpoint:
            quantum_agent.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print(f"✓ Loaded: {quantum_best}")
    else:
        print(f"❌ Not found: {quantum_best}")
        return
    
    print("Evaluating...")
    quantum_traj = evaluate_single_episode(quantum_agent, env, device)
    
    print(f"  MDAPE: {quantum_traj['mdape']:.2f}%")
    print(f"  Reward: {quantum_traj['total_reward']:.2f}")
    print(f"  Time in Target: {quantum_traj['time_in_target']:.2f}%")
    
    # ===== Create Visualizations =====
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_comparison(classical_traj, quantum_traj, output_dir)
    create_summary_table(classical_traj, quantum_traj, output_dir)
    save_trajectories_csv(classical_traj, quantum_traj, output_dir)
    
    print(f"\n✓ All results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
