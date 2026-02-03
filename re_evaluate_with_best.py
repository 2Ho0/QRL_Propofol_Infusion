"""
Re-evaluate trained models using best checkpoints instead of final checkpoints
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import torch
import numpy as np
import pickle
from tqdm import tqdm

from agents.quantum_agent import QuantumDDPGAgent
from agents.classical_agent import ClassicalDDPGAgent
from environment.dual_drug_env import DualDrugEnv
from environment.patient_simulator import create_patient_population

def evaluate_agent(agent, n_episodes=100, seed=42, device='cuda'):
    """Evaluate agent on simulator"""
    patients = create_patient_population(n_patients=n_episodes, seed=seed)
    
    mdapes = []
    rewards_total = []
    propofol_usage = []
    remifentanil_usage = []
    time_in_target = []
    
    for i, patient in enumerate(tqdm(patients, desc="Evaluating")):
        env = DualDrugEnv(patient=patient, seed=seed + i, reward_type='potential')
        
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
                    if encoded.dim() == 3:
                        encoded = encoded[:, -1, :]
                    action = agent.actor(encoded)
                else:
                    action = agent.actor(state_tensor)
            
            action_np = action.cpu().numpy().flatten()
            if action_np.shape[0] != 2:
                action_np = action_np[:2]
            
            action_physical = np.array([
                action_np[0] * 30.0,  # Propofol
                action_np[1] * 1.0    # Remifentanil
            ])
            
            next_state, reward, done, truncated, info = env.step(action_physical)
            
            episode_reward += reward
            episode_ppf.append(action_physical[0])
            episode_rftn.append(action_physical[1])
            states_list.append(state)
            
            bis = info.get('bis', 50 - state[0])
            bis_history.append(bis)
            
            state = next_state
            
            if done or truncated:
                break
        
        # Calculate metrics
        if hasattr(env, 'get_episode_metrics'):
            metrics = env.get_episode_metrics()
            mdapes.append(metrics.get('mdape', 0))
        else:
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
        
        # Time in target
        if bis_history:
            bis_array = np.array(bis_history)
            time_in_target.append(np.mean((bis_array >= 45) & (bis_array <= 55)) * 100)
        else:
            time_in_target.append(0)
    
    return {
        'mdape_mean': np.mean(mdapes),
        'mdape_std': np.std(mdapes),
        'reward_mean': np.mean(rewards_total),
        'reward_std': np.std(rewards_total),
        'propofol_usage_mean': np.mean(propofol_usage),
        'remifentanil_usage_mean': np.mean(remifentanil_usage),
        'time_in_target_mean': np.mean(time_in_target),
        'time_in_target_std': np.std(time_in_target)
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    comparison_dir = Path('/home/mwilliam/QRL_Propofol_Infusion/logs/comparison_20260115_214922')
    
    state_dim = 13
    action_dim = 2
    
    # Load config
    import yaml
    with open('/home/mwilliam/QRL_Propofol_Infusion/config/hyperparameters.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*70)
    print("RE-EVALUATING WITH BEST CHECKPOINTS")
    print("="*70)
    
    # Classical Agent - Test different checkpoints
    print("\n" + "="*70)
    print("CLASSICAL AGENT")
    print("="*70)
    
    classical_agent = ClassicalDDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        encoder_type=None,
        seed=1042
    )
    classical_agent.actor = classical_agent.actor.to(device)
    classical_agent.critic = classical_agent.critic.to(device)
    if classical_agent.encoder is not None:
        classical_agent.encoder = classical_agent.encoder.to(device)
    
    classical_checkpoints = {
        'best_mdape': comparison_dir / 'classical' / 'stage2_online' / 'best_mdape.pt',
        'best_reward': comparison_dir / 'classical' / 'stage2_online' / 'best_reward.pt',
        'final': comparison_dir / 'classical' / 'stage2_online' / 'final.pt'
    }
    
    classical_results = {}
    for name, path in classical_checkpoints.items():
        if path.exists():
            print(f"\n--- Evaluating {name} ---")
            classical_agent.load(str(path))
            results = evaluate_agent(classical_agent, n_episodes=100, seed=1042, device=device)
            classical_results[name] = results
            print(f"MDAPE: {results['mdape_mean']:.2f}% ± {results['mdape_std']:.2f}%")
            print(f"Reward: {results['reward_mean']:.2f} ± {results['reward_std']:.2f}")
            print(f"Time in Target: {results['time_in_target_mean']:.2f}%")
        else:
            print(f"\n⚠️  {name} not found: {path}")
    
    # Quantum Agent - Test different checkpoints
    print("\n" + "="*70)
    print("QUANTUM AGENT")
    print("="*70)
    
    quantum_agent = QuantumDDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        encoder_type=None,
        seed=42
    )
    quantum_agent.actor = quantum_agent.actor.to(device)
    quantum_agent.critic = quantum_agent.critic.to(device)
    if quantum_agent.encoder is not None:
        quantum_agent.encoder = quantum_agent.encoder.to(device)
    
    quantum_checkpoints = {
        'best_mdape': comparison_dir / 'quantum' / 'stage2_online' / 'best_mdape.pt',
        'best_reward': comparison_dir / 'quantum' / 'stage2_online' / 'best_reward.pt',
        'final': comparison_dir / 'quantum' / 'stage2_online' / 'final.pt'
    }
    
    quantum_results = {}
    for name, path in quantum_checkpoints.items():
        if path.exists():
            print(f"\n--- Evaluating {name} ---")
            quantum_agent.load(str(path))
            results = evaluate_agent(quantum_agent, n_episodes=100, seed=42, device=device)
            quantum_results[name] = results
            print(f"MDAPE: {results['mdape_mean']:.2f}% ± {results['mdape_std']:.2f}%")
            print(f"Reward: {results['reward_mean']:.2f} ± {results['reward_std']:.2f}")
            print(f"Time in Target: {results['time_in_target_mean']:.2f}%")
        else:
            print(f"\n⚠️  {name} not found: {path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - BEST CHECKPOINT COMPARISON")
    print("="*70)
    
    for checkpoint_name in ['best_mdape', 'best_reward', 'final']:
        print(f"\n📊 {checkpoint_name.upper()}:")
        if checkpoint_name in classical_results:
            c_res = classical_results[checkpoint_name]
            print(f"  Classical - MDAPE: {c_res['mdape_mean']:.2f}%, Reward: {c_res['reward_mean']:.2f}, Time: {c_res['time_in_target_mean']:.1f}%")
        if checkpoint_name in quantum_results:
            q_res = quantum_results[checkpoint_name]
            print(f"  Quantum   - MDAPE: {q_res['mdape_mean']:.2f}%, Reward: {q_res['reward_mean']:.2f}, Time: {q_res['time_in_target_mean']:.1f}%")
    
    # Save results
    with open(comparison_dir / 'checkpoint_comparison_results.pkl', 'wb') as f:
        pickle.dump({
            'classical': classical_results,
            'quantum': quantum_results
        }, f)
    
    print(f"\n✓ Results saved to: {comparison_dir / 'checkpoint_comparison_results.pkl'}")
    print("="*70)


if __name__ == "__main__":
    main()
