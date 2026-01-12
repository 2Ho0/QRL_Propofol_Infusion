"""
Hybrid Training: Offline Pre-training → Online Fine-tuning
===========================================================

Complete hybrid training pipeline:
1. Stage 1: Offline pre-training on VitalDB real patient data (80%)
2. Stage 2: Online fine-tuning on simulator (new synthetic data)
3. Stage 3: Testing on both VitalDB test set (10%) and simulator test set

Data Split Strategy:
--------------------
VitalDB Dataset (100 cases)
├─ Train (80 cases): Offline pre-training
├─ Val (10 cases): Hyperparameter tuning
└─ Test (10 cases): Final evaluation

Simulator (Unlimited)
├─ Train: Online fine-tuning episodes
└─ Test: 20 diverse synthetic patients

Usage:
    # Full hybrid training
    python experiments/train_hybrid.py --n_cases 100 --offline_epochs 50 --online_episodes 500
    
    # Skip offline and resume from checkpoint
    python experiments/train_hybrid.py --skip_offline --resume logs/stage1_best.pt --online_episodes 500
    
    # Offline only (no online fine-tuning)
    python experiments/train_hybrid.py --n_cases 100 --offline_epochs 50 --skip_online
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from datetime import datetime
from typing import Dict, Tuple

from agents.quantum_agent import QuantumDDPGAgent
from environment.propofol_env import PropofolEnv
from environment.patient_simulator import create_patient_population
from data.vitaldb_loader import VitalDBLoader, VitalDBDataset
from models.networks import soft_update


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum RL Training: Offline → Online"
    )
    
    # Stage control
    parser.add_argument('--skip_offline', action='store_true',
                        help='Skip offline pre-training stage')
    parser.add_argument('--skip_online', action='store_true',
                        help='Skip online fine-tuning stage')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (for skipping offline)')
    
    # Data parameters
    parser.add_argument('--n_cases', type=int, default=20,
                        help='Total number of VitalDB cases to load')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to pre-saved VitalDB data (skip loading)')
    
    # Stage 1: Offline parameters
    parser.add_argument('--offline_epochs', type=int, default=50,
                        help='Number of epochs for offline pre-training')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for offline training')
    parser.add_argument('--bc_weight', type=float, default=0.8,
                        help='Weight for behavioral cloning loss (0-1)')
    
    # Stage 2: Online parameters
    parser.add_argument('--online_episodes', type=int, default=500,
                        help='Number of episodes for online fine-tuning')
    parser.add_argument('--warmup_episodes', type=int, default=50,
                        help='Episodes without exploration before enabling it')
    
    # General parameters
    parser.add_argument('--config', type=str, 
                        default='config/hyperparameters.yaml',
                        help='Path to configuration file')
    parser.add_argument('--encoder', type=str, default='none',
                        choices=['none', 'lstm', 'transformer'],
                        help='Temporal encoder type')
    parser.add_argument('--dual_drug', action='store_true',
                        help='Use dual drug control (propofol + remifentanil)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--eval_interval', type=int, default=50,
                        help='Evaluate every N episodes/epochs')
    
    # Reward function
    parser.add_argument('--reward_type', type=str, default='potential',
                       choices=['simple', 'paper', 'hybrid', 'potential'],
                       help='Reward function type: simple (original), paper (equation 1-3), hybrid (dense+sparse), potential (potential-based shaping)')
    
    
    return parser.parse_args()


def setup_directories(log_dir: str) -> Dict[str, Path]:
    """Set up directories for hybrid training."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path(log_dir) / f'hybrid_{timestamp}'
    
    dirs = {
        'base': base_dir,
        'checkpoints': base_dir / 'checkpoints',
        'figures': base_dir / 'figures',
        'stage1': base_dir / 'stage1_offline',
        'stage2': base_dir / 'stage2_online',
        'stage3': base_dir / 'stage3_test'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def split_vitaldb_data(
    n_total_cases: int = 100,
    data_path: str = None,
    dual_drug: bool = False
) -> Tuple[Dict, Dict, Dict]:
    """
    Split VitalDB data into train/val/test sets (80/10/10).
    
    Args:
        n_total_cases: Total number of cases to load
        data_path: Path to pre-saved data (optional)
        dual_drug: If True, load dual drug data (propofol + remifentanil)
    
    Returns:
        train_data, val_data, test_data dictionaries
    """
    print("\n" + "="*70)
    print("DATA LOADING & SPLITTING")
    print("="*70)
    
    if data_path and Path(data_path).exists():
        print(f"Loading pre-saved data from {data_path}...")
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        
        all_states = all_data['states']
        all_actions = all_data['actions']
        all_rewards = all_data['rewards']
        all_next_states = all_data['next_states']
        
        print(f"✓ Loaded {len(all_states):,} transitions")
    else:
        drug_type = "dual drug (propofol + remifentanil)" if dual_drug else "single drug (propofol)"
        print(f"Loading VitalDB data ({n_total_cases} cases, {drug_type})...")
        loader = VitalDBLoader(bis_range=(30, 70))
        
        # Load data (returns dict with states, actions, rewards, next_states, dones)
        if dual_drug:
            data = loader.prepare_training_data_dualdrug(n_cases=n_total_cases)
        else:
            data = loader.prepare_training_data(n_cases=n_total_cases)
        
        all_states = data['states']
        all_actions = data['actions']
        all_rewards = data['rewards']
        all_next_states = data['next_states']
        
        state_dim = all_states.shape[1] if len(all_states.shape) > 1 else 1
        action_dim = all_actions.shape[1] if len(all_actions.shape) > 1 else 1
        print(f"✓ Loaded {len(all_states):,} transitions (state_dim={state_dim}, action_dim={action_dim})")
    
    # Split by percentages (preserving temporal structure within cases)
    n_total = len(all_states)
    train_end = int(n_total * 0.8)
    val_end = int(n_total * 0.9)
    
    # Train set (80%)
    train_data = {
        'states': all_states[:train_end],
        'actions': all_actions[:train_end],
        'rewards': all_rewards[:train_end],
        'next_states': all_next_states[:train_end],
        'dones': np.zeros(train_end, dtype=bool)
    }
    
    # Validation set (10%)
    val_data = {
        'states': all_states[train_end:val_end],
        'actions': all_actions[train_end:val_end],
        'rewards': all_rewards[train_end:val_end],
        'next_states': all_next_states[train_end:val_end],
        'dones': np.zeros(val_end - train_end, dtype=bool)
    }
    
    # Test set (10%)
    test_data = {
        'states': all_states[val_end:],
        'actions': all_actions[val_end:],
        'rewards': all_rewards[val_end:],
        'next_states': all_next_states[val_end:],
        'dones': np.zeros(n_total - val_end, dtype=bool)
    }
    
    print(f"\n✓ Data split complete:")
    print(f"  Train: {len(train_data['states']):,} transitions (80%)")
    print(f"  Val:   {len(val_data['states']):,} transitions (10%)")
    print(f"  Test:  {len(test_data['states']):,} transitions (10%)")
    print(f"  BIS range: [{50 - all_states[:, 0].max():.0f}, {50 - all_states[:, 0].min():.0f}]")
    
    return train_data, val_data, test_data


def stage1_offline_pretraining(
    agent: QuantumDDPGAgent,
    train_data: Dict,
    val_data: Dict,
    n_epochs: int,
    batch_size: int,
    bc_weight: float,
    dirs: Dict[str, Path],
    eval_interval: int = 10,
    use_cql: bool = False,
    cql_alpha: float = 1.0,
    cql_temp: float = 1.0,
    cql_num_random: int = 5,
    cql_warmup_epochs: int = 50,
    bc_warmup_epochs: int = 20,
    device: torch.device = None,
    num_workers: int = 4,
    use_amp: bool = False,
) -> QuantumDDPGAgent:
    """
    Stage 1: Offline pre-training on VitalDB train set.
    
    Uses Behavioral Cloning + Off-policy RL (or CQL) on real patient data.
    
    Args:
        agent: Quantum DDPG agent
        train_data: Training data dictionary
        val_data: Validation data dictionary
        n_epochs: Number of training epochs
        batch_size: Batch size
        bc_weight: Weight for BC loss (0-1)
        dirs: Directory paths
        eval_interval: Evaluation interval
        use_cql: Use Conservative Q-Learning instead of standard RL
        cql_alpha: CQL penalty weight
        cql_temp: Temperature for CQL logsumexp
        cql_num_random: Number of random actions for CQL
        cql_warmup_epochs: Number of epochs to use CQL (후반에는 CQL 비활성화)
        device: Torch device
        num_workers: Number of data loader workers
        use_amp: Use Automatic Mixed Precision for faster training
    
    Returns:
        Trained agent
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("STAGE 1: OFFLINE PRE-TRAINING (VitalDB Real Data)")
    print("="*70)
    print(f"Training on {len(train_data['states']):,} transitions")
    print(f"Quantum Actor learning from expert demonstrations")
    print(f"Device: {device}")
    
    # Create PyTorch datasets
    train_dataset = VitalDBDataset(train_data)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
        )
    
    val_dataset = VitalDBDataset(val_data)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
        )
    
    # Training metrics
    best_val_loss = float('inf')
    best_val_mdape = float('inf')
    patience_counter = 0
    patience_limit = 25  # More patience for BC convergence
    train_history = {'train_loss': [], 'val_loss': [], 'bc_loss': [], 'rl_loss': []}
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  BC weight: {bc_weight:.2f}")
    print(f"  RL weight: {1 - bc_weight:.2f}")
    print(f"  Method: {'CQL (Conservative Q-Learning)' if use_cql else 'Standard Off-policy RL'}")
    if use_cql:
        print(f"  CQL alpha: {cql_alpha:.2f}")
        print(f"  CQL temp: {cql_temp:.2f}")
        print(f"  CQL random actions: {cql_num_random}")
        print(f"  CQL warmup epochs: {cql_warmup_epochs} (후반 {n_epochs - cql_warmup_epochs} epochs는 CQL 비활성화)")
    print(f"  Mixed Precision: {'Enabled' if use_amp and torch.cuda.is_available() else 'Disabled'}")
    print(f"  Num workers: {num_workers}")
    print()

    from torch.amp import autocast, GradScaler
    scaler = GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
    
    import time
    epoch_times = []
    
    for epoch in tqdm(range(n_epochs), desc="Offline Pre-training"):
        epoch_start = time.time()
        epoch_train_bc = []
        epoch_train_rl = []
        epoch_train_total = []
        
        # CQL warmup: 초반에만 CQL 사용
        use_cql_this_epoch = use_cql and epoch < cql_warmup_epochs
        
        # BC warmup: Start with pure BC, gradually introduce RL
        # First bc_warmup_epochs: Pure BC (weight=1.0) for stable initialization
        # After warmup: Use specified bc_weight for hybrid training
        if epoch < bc_warmup_epochs:
            current_bc_weight = 1.0  # Pure BC during warmup
        else:
            current_bc_weight = bc_weight  # Hybrid BC+RL after warmup
        
        # Training loop
        for batch in train_loader:
            states, actions, rewards, next_states, dones = batch
            
            # Ensure all tensors are float32 and move to correct device
            states = states.float().to(device)
            actions = actions.float().to(device)
            rewards = rewards.float().to(device)
            next_states = next_states.float().to(device)
            dones = dones.float().to(device)
            
            # ========== Mixed Precision Training ==========
            if scaler is not None:
                # Forward pass with autocast
                with autocast('cuda'):
                    # ========== Behavioral Cloning Loss ==========
                    # Use encoder if available
                    if agent.encoder is not None:
                        states_seq = states.unsqueeze(1)
                        encoded_states = agent.encoder(states_seq)
                        if isinstance(encoded_states, tuple):
                            encoded_states = encoded_states[0]
                        encoded_states = encoded_states[:, -1, :] if encoded_states.dim() == 3 else encoded_states
                        predicted_actions = agent.actor(encoded_states)
                        
                        next_states_seq = next_states.unsqueeze(1)
                        encoded_next = agent.encoder_target(next_states_seq)
                        if isinstance(encoded_next, tuple):
                            encoded_next = encoded_next[0]
                        encoded_next = encoded_next[:, -1, :] if encoded_next.dim() == 3 else encoded_next
                        next_actions = agent.actor_target(encoded_next)
                    else:
                        predicted_actions = agent.actor(states)
                        next_actions = agent.actor_target(next_states)
                    
                    bc_loss = torch.nn.functional.mse_loss(predicted_actions, actions)
                    
                    # ========== Off-policy RL Loss ==========
                    # Only compute critic loss if BC weight < 1.0 (pure BC doesn't need critic)
                    if current_bc_weight < 1.0:
                        with torch.no_grad():
                            if agent.encoder is not None:
                                q1_next, q2_next = agent.critic_target(encoded_next, next_actions)
                            else:
                                q1_next, q2_next = agent.critic_target(next_states, next_actions)
                            target_q = rewards.unsqueeze(1) + agent.gamma * torch.min(q1_next, q2_next) * (1 - dones.unsqueeze(1))
                        
                        if agent.encoder is not None:
                            q1, q2 = agent.critic(encoded_states, actions)
                        else:
                            q1, q2 = agent.critic(states, actions)
                        critic_loss = (
                            torch.nn.functional.mse_loss(q1, target_q) +
                            torch.nn.functional.mse_loss(q2, target_q)
                        ) / 2
                    else:
                        # Pure BC: no critic loss
                        critic_loss = torch.tensor(0.0, device=device)
                        q1 = torch.tensor(0.0, device=device)
                        q2 = torch.tensor(0.0, device=device)
                    
                    # ========== CQL Penalty (if enabled and in warmup period) ==========
                    if use_cql_this_epoch:
                        # Sample random actions
                        batch_size_cur = states.shape[0]
                        random_actions = torch.rand(batch_size_cur, cql_num_random, actions.shape[1], device=device)
                        random_actions = random_actions * 2 - 1  # Scale to [-1, 1]
                        
                        # Repeat states for random actions
                        if agent.encoder is not None:
                            encoded_states_repeated = encoded_states.unsqueeze(1).repeat(1, cql_num_random, 1)
                            encoded_states_flat = encoded_states_repeated.reshape(-1, encoded_states.shape[1])
                        else:
                            states_repeated = states.unsqueeze(1).repeat(1, cql_num_random, 1)
                            encoded_states_flat = states_repeated.reshape(-1, states.shape[1])
                        
                        random_actions_flat = random_actions.reshape(-1, actions.shape[1])
                        
                        # Q-values for random actions
                        q1_random, q2_random = agent.critic(encoded_states_flat, random_actions_flat)
                        q1_random = q1_random.reshape(batch_size_cur, cql_num_random)
                        q2_random = q2_random.reshape(batch_size_cur, cql_num_random)
                        
                        # CQL penalty: logsumexp of random Q-values minus data Q-values
                        cql_penalty = (
                            torch.logsumexp(q1_random / cql_temp, dim=1).mean() +
                            torch.logsumexp(q2_random / cql_temp, dim=1).mean() -
                            q1.mean() - q2.mean()
                        )
                        
                        critic_loss = critic_loss + cql_alpha * cql_penalty
                
                # Backward with gradient scaling
                # Only update critic if BC weight < 1.0
                if current_bc_weight < 1.0:
                    agent.critic_optimizer.zero_grad()
                    scaler.scale(critic_loss).backward()
                    scaler.unscale_(agent.critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
                    scaler.step(agent.critic_optimizer)
                
                # Update Actor with AMP
                agent.actor_optimizer.zero_grad()
                with autocast('cuda'):
                    if agent.encoder is not None:
                        # Recompute encoded states for actor update
                        states_seq = states.unsqueeze(1)
                        encoded_states_actor = agent.encoder(states_seq)
                        if isinstance(encoded_states_actor, tuple):
                            encoded_states_actor = encoded_states_actor[0]
                        encoded_states_actor = encoded_states_actor[:, -1, :] if encoded_states_actor.dim() == 3 else encoded_states_actor
                        pred_actions_for_bc = agent.actor(encoded_states_actor)
                    else:
                        pred_actions_for_bc = agent.actor(states)
                    
                    # Pure BC: only BC loss
                    if current_bc_weight >= 1.0:
                        bc_loss_final = torch.nn.functional.mse_loss(pred_actions_for_bc, actions)
                        combined_actor_loss = bc_loss_final
                    else:
                        # Hybrid BC + RL
                        pred_actions_for_q = agent.actor(encoded_states_actor if agent.encoder is not None else states)
                        q1_pred, _ = agent.critic(encoded_states_actor if agent.encoder is not None else states, pred_actions_for_q)
                        bc_loss_final = torch.nn.functional.mse_loss(pred_actions_for_bc, actions)
                        actor_rl_loss = -q1_pred.mean()
                        
                        # Scale RL loss to match BC loss magnitude
                        # BC loss ~ 0.0001, RL loss ~ -1.0, so scale RL by 0.0001 to balance
                        rl_scale = 0.0001
                        combined_actor_loss = current_bc_weight * bc_loss_final + (1 - current_bc_weight) * rl_scale * actor_rl_loss
                
                scaler.scale(combined_actor_loss).backward()
                scaler.unscale_(agent.actor_optimizer)
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
                scaler.step(agent.actor_optimizer)
                
                scaler.update()
                
            else:
                # Standard training without AMP
                # ========== Behavioral Cloning Loss ==========
                # Use encoder if available
                if agent.encoder is not None:
                    # For LSTM/Transformer encoder
                    states_seq = states.unsqueeze(1)  # [batch, 1, state_dim]
                    encoded_states = agent.encoder(states_seq)
                    if isinstance(encoded_states, tuple):  # LSTM returns (output, hidden)
                        encoded_states = encoded_states[0]
                    # Get last timestep
                    encoded_states = encoded_states[:, -1, :] if encoded_states.dim() == 3 else encoded_states
                    predicted_actions = agent.actor(encoded_states).float()
                    
                    # For target network
                    next_states_seq = next_states.unsqueeze(1)
                    encoded_next = agent.encoder_target(next_states_seq)
                    if isinstance(encoded_next, tuple):
                        encoded_next = encoded_next[0]
                    encoded_next = encoded_next[:, -1, :] if encoded_next.dim() == 3 else encoded_next
                    next_actions = agent.actor_target(encoded_next).float()
                else:
                    # No encoder, use raw states
                    predicted_actions = agent.actor(states).float()
                    next_actions = agent.actor_target(next_states).float()
                
                bc_loss = torch.nn.functional.mse_loss(predicted_actions, actions)
                
                # ========== Off-policy RL Loss ==========
                # Only compute critic loss if BC weight < 1.0
                if current_bc_weight < 1.0:
                    with torch.no_grad():
                        # next_actions already computed above (with or without encoder)
                        if agent.encoder is not None:
                            q1_next, q2_next = agent.critic_target(encoded_next, next_actions)
                        else:
                            q1_next, q2_next = agent.critic_target(next_states, next_actions)
                        q1_next = q1_next.float()
                        q2_next = q2_next.float()
                        target_q = (rewards.unsqueeze(1) + agent.gamma * torch.min(q1_next, q2_next) * (1 - dones.unsqueeze(1))).float()
                    
                    if agent.encoder is not None:
                        q1, q2 = agent.critic(encoded_states, actions)
                    else:
                        q1, q2 = agent.critic(states, actions)
                    q1 = q1.float()
                    q2 = q2.float()
                    critic_loss = (
                        torch.nn.functional.mse_loss(q1, target_q) +
                        torch.nn.functional.mse_loss(q2, target_q)
                    ) / 2
                else:
                    # Pure BC: no critic loss
                    critic_loss = torch.tensor(0.0, device=device)
                    q1 = torch.tensor(0.0, device=device)
                    q2 = torch.tensor(0.0, device=device)
                
                # ========== CQL Penalty (if enabled and in warmup period) ==========
                if use_cql_this_epoch:
                    # Sample random actions
                    batch_size_cur = states.shape[0]
                    random_actions = torch.rand(batch_size_cur, cql_num_random, actions.shape[1], device=device)
                    random_actions = random_actions * 2 - 1  # Scale to [-1, 1]
                    
                    # Repeat states for random actions
                    if agent.encoder is not None:
                        # Use encoded states
                        encoded_states_repeated = encoded_states.unsqueeze(1).repeat(1, cql_num_random, 1)
                        encoded_states_flat = encoded_states_repeated.reshape(-1, encoded_states.shape[1])
                    else:
                        states_repeated = states.unsqueeze(1).repeat(1, cql_num_random, 1)
                        encoded_states_flat = states_repeated.reshape(-1, states.shape[1])
                    
                    random_actions_flat = random_actions.reshape(-1, actions.shape[1])
                    
                    # Q-values for random actions
                    q1_random, q2_random = agent.critic(encoded_states_flat, random_actions_flat)
                    q1_random = q1_random.reshape(batch_size_cur, cql_num_random).float()
                    q2_random = q2_random.reshape(batch_size_cur, cql_num_random).float()
                    
                    # CQL penalty: logsumexp of random Q-values minus data Q-values
                    cql_penalty = (
                        torch.logsumexp(q1_random / cql_temp, dim=1).mean() +
                        torch.logsumexp(q2_random / cql_temp, dim=1).mean() -
                        q1.mean() - q2.mean()
                    )
                    
                    critic_loss = critic_loss + cql_alpha * cql_penalty
            
                # ========== Combined Update ==========
                # Update Critic first (only if BC weight < 1.0)
                if current_bc_weight < 1.0:
                    agent.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
                    agent.critic_optimizer.step()
                
                # Update Actor (BC only or BC + policy gradient)
                agent.actor_optimizer.zero_grad()
                
                # Pure BC: only BC loss
                if current_bc_weight >= 1.0:
                    if agent.encoder is not None:
                        states_seq = states.unsqueeze(1)
                        encoded_states_actor = agent.encoder(states_seq)
                        if isinstance(encoded_states_actor, tuple):
                            encoded_states_actor = encoded_states_actor[0]
                        if encoded_states_actor.dim() == 3:
                            encoded_states_actor = encoded_states_actor[:, -1, :]
                        pred_actions = agent.actor(encoded_states_actor)
                    else:
                        pred_actions = agent.actor(states)
                    
                    actor_loss = torch.nn.functional.mse_loss(pred_actions, actions)
                else:
                    # Hybrid BC + RL: recompute actor loss after critic update
                    if agent.encoder is not None:
                        states_seq = states.unsqueeze(1)
                        encoded_states = agent.encoder(states_seq)
                        if isinstance(encoded_states, tuple):
                            encoded_states = encoded_states[0]
                        # Handle both 2D and 3D encoder outputs
                        if encoded_states.dim() == 3:
                            encoded_states = encoded_states[:, -1, :]
                        pred_actions_for_bc = agent.actor(encoded_states).float()
                        pred_actions_for_q = agent.actor(encoded_states).float()
                        q1_pred, _ = agent.critic(encoded_states, pred_actions_for_q)
                    else:
                        pred_actions_for_bc = agent.actor(states).float()
                        pred_actions_for_q = agent.actor(states).float()
                        q1_pred, _ = agent.critic(states, pred_actions_for_q)
                    
                    bc_loss_final = torch.nn.functional.mse_loss(pred_actions_for_bc, actions)
                    q1_pred = q1_pred.float()
                    actor_rl_loss = -q1_pred.mean()
                    
                    # Scale RL loss to match BC loss magnitude
                    # BC loss ~ 0.0001, RL loss ~ -1.0, so scale RL by 0.0001 to balance
                    rl_scale = 0.0001
                    
                    # Combined loss - ensure float32
                    bc_weight_tensor = torch.tensor(current_bc_weight, dtype=torch.float32, device=device)
                    actor_loss = (bc_weight_tensor * bc_loss_final + (1 - bc_weight_tensor) * rl_scale * actor_rl_loss).float()
                
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
                agent.actor_optimizer.step()
            
            # Soft update target networks (공통)
            soft_update(agent.actor_target, agent.actor, agent.tau)
            if current_bc_weight < 1.0:
                soft_update(agent.critic_target, agent.critic, agent.tau)
            
            # Track metrics
            epoch_train_bc.append(bc_loss.item())
            
            # Only track RL losses if BC weight < 1.0
            if current_bc_weight < 1.0:
                # Scale RL loss for display to show actual contribution to total loss
                rl_scale = 0.0001
                if scaler is not None:
                    epoch_train_rl.append(actor_rl_loss.item() * rl_scale)
                    epoch_train_total.append(combined_actor_loss.item() + critic_loss.item())
                else:
                    epoch_train_rl.append(actor_rl_loss.item() * rl_scale)
                    epoch_train_total.append(actor_loss.item() + critic_loss.item())
            else:
                # Pure BC: RL loss is 0
                epoch_train_rl.append(0.0)
                if scaler is not None:
                    epoch_train_total.append(combined_actor_loss.item())
                else:
                    epoch_train_total.append(actor_loss.item())
        
        # Validation loop
        epoch_val_loss = []
        epoch_val_mdape = []
        with torch.no_grad():
            for batch in val_loader:
                states, actions, _, _, _ = batch
                states = states.float().to(device)
                actions = actions.float().to(device)
                
                if agent.encoder is not None:
                    states_seq = states.unsqueeze(1)
                    encoded_states = agent.encoder(states_seq)
                    if isinstance(encoded_states, tuple):
                        encoded_states = encoded_states[0]
                    # Handle both 2D and 3D encoder outputs
                    if encoded_states.dim() == 3:
                        encoded_states = encoded_states[:, -1, :]
                    predicted_actions = agent.actor(encoded_states)
                else:
                    predicted_actions = agent.actor(states)
                
                # MSE Loss
                val_loss = torch.nn.functional.mse_loss(predicted_actions, actions)
                epoch_val_loss.append(val_loss.item())
                
                # MDAPE (Median Absolute Percentage Error)
                pred_np = predicted_actions.detach().cpu().numpy()
                act_np = actions.detach().cpu().numpy()
                ape = np.abs(pred_np - act_np) / (np.abs(act_np) + 1e-8) * 100
                mdape = np.median(ape)
                epoch_val_mdape.append(mdape)
        
        # Track epoch time
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = n_epochs - epoch - 1
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_minutes = eta_seconds / 60
        
        # Record metrics
        avg_train_bc = np.mean(epoch_train_bc)
        avg_train_rl = np.mean(epoch_train_rl)
        avg_train_total = np.mean(epoch_train_total)
        avg_val_loss = np.mean(epoch_val_loss)
        avg_val_mdape = np.mean(epoch_val_mdape)
        
        train_history['train_loss'].append(avg_train_total)
        train_history['val_loss'].append(avg_val_loss)
        train_history['bc_loss'].append(avg_train_bc)
        train_history['rl_loss'].append(avg_train_rl)
        train_history['val_mdape'] = train_history.get('val_mdape', [])
        train_history['val_mdape'].append(avg_val_mdape)
        
        # 매 epoch마다 간단한 로그 (한 줄)
        cql_status = "CQL" if use_cql_this_epoch else "STD"
        bc_status = f"BC={current_bc_weight:.1f}"  # Show current BC weight
        print(f"Epoch {epoch+1:3d}/{n_epochs} [{cql_status}|{bc_status}] | Train: {avg_train_total:7.4f} | Val: {avg_val_loss:7.4f} | MDAPE: {avg_val_mdape:6.2f}% | BC: {avg_train_bc:7.4f} | RL: {avg_train_rl:7.4f} | Time: {epoch_time:5.1f}s | ETA: {eta_minutes:5.1f}m")
        
        # Periodic detailed logging
        if epoch % eval_interval == 0 or epoch == n_epochs - 1:
            print(f"\n  {'='*60}")
            print(f"  Epoch {epoch+1}/{n_epochs} Detailed Stats:")
            print(f"  {'='*60}")
            print(f"  Training Losses:")
            print(f"    - BC Loss (Behavioral Cloning): {avg_train_bc:.4f}")
            print(f"    - RL Loss (Policy Gradient):    {avg_train_rl:.4f}")
            print(f"    - Total Train Loss:             {avg_train_total:.4f}")
            print(f"  Validation Metrics:")
            print(f"    - Val Loss (MSE):               {avg_val_loss:.4f}")
            print(f"    - Val MDAPE:                    {avg_val_mdape:.2f}%")
            print(f"  Train-Val Gap:                    {avg_train_total - avg_val_loss:+.4f}")
            print(f"  Time: {epoch_time:.1f}s (avg: {avg_epoch_time:.1f}s) | ETA: {eta_minutes:.1f}min")
            print(f"  {'='*60}")
        
        # Save best model based on validation MDAPE (primary metric)
        if avg_val_mdape < best_val_mdape:
            best_val_mdape = avg_val_mdape
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience
            best_path = dirs['stage1'] / 'best_val.pt'
            agent.save(str(best_path))
            if epoch % eval_interval == 0:
                print(f"    → New best model saved (MDAPE: {best_val_mdape:.2f}%, val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= patience_limit:
            print(f"\n⚠️  Early stopping triggered!")
            print(f"    No improvement in validation MDAPE for {patience_limit} epochs")
            print(f"    Best val MDAPE: {best_val_mdape:.2f}%")
            print(f"    Stopping at epoch {epoch+1}/{n_epochs}")
            break
        
        # Save periodic checkpoints
        if (epoch + 1) % 20 == 0:
            checkpoint_path = dirs['stage1'] / f'checkpoint_epoch_{epoch+1}.pt'
            agent.save(str(checkpoint_path))
    
    # Save final model
    final_path = dirs['stage1'] / 'final.pt'
    agent.save(str(final_path))
    
    # Save training history (pickle)
    with open(dirs['stage1'] / 'training_history.pkl', 'wb') as f:
        pickle.dump(train_history, f)
    
    # Save training history to CSV
    # Use actual number of completed epochs (in case of early stopping)
    actual_epochs = len(train_history['train_loss'])
    loss_df = pd.DataFrame({
        'epoch': range(1, actual_epochs + 1),
        'train_loss': train_history['train_loss'],
        'val_loss': train_history['val_loss'],
        'bc_loss': train_history['bc_loss'],
        'rl_loss': train_history['rl_loss'],
        'val_mdape': train_history.get('val_mdape', [0] * actual_epochs)
    })
    loss_csv_path = dirs['stage1'] / 'loss_history.csv'
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"✓ Loss history saved to CSV: {loss_csv_path}")
    
    print(f"\n✓ Stage 1 Complete: Offline Pre-training")
    print(f"  Completed epochs: {actual_epochs}/{n_epochs}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best validation MDAPE: {best_val_mdape:.2f}%")
    if actual_epochs < n_epochs:
        print(f"  Early stopping triggered at epoch {actual_epochs}")
    if train_history.get('val_mdape'):
        print(f"  Final validation MDAPE: {train_history['val_mdape'][-1]:.2f}%")
    print(f"  Saved models:")
    print(f"    - Best: {dirs['stage1'] / 'best_val.pt'}")
    print(f"    - Final: {final_path}")
    
    return agent


def stage2_online_finetuning(
    agent: QuantumDDPGAgent,
    n_episodes: int,
    warmup_episodes: int,
    dirs: Dict[str, Path],
    eval_interval: int,
    seed: int,
    env_class=None,
    device: torch.device = None
) -> QuantumDDPGAgent:
    """
    Stage 2: Online fine-tuning on simulator (NEW synthetic data).
    
    The agent explores beyond expert data and optimizes for reward.
    
    Args:
        agent: Agent to fine-tune
        n_episodes: Number of online episodes
        warmup_episodes: Episodes without exploration noise
        dirs: Dictionary with 'stage2' key for saving
        eval_interval: Evaluation interval
        seed: Random seed
        env_class: Environment class to use (PropofolEnv or DualDrugEnv)
        device: Torch device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import environment classes
    from environment.propofol_env import PropofolEnv
    from environment.dual_drug_env import DualDrugEnv
    
    # Determine environment based on agent's action dimension
    if env_class is None:
        if agent.action_dim == 1:
            env_class = PropofolEnv
        elif agent.action_dim == 2:
            env_class = DualDrugEnv
        else:
            raise ValueError(f"Unsupported action_dim: {agent.action_dim}")
    
    print("\n" + "="*70)
    print("STAGE 2: ONLINE FINE-TUNING (Simulator - NEW Data!)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Environment: {env_class.__name__}")
    print(f"Fine-tuning for {n_episodes} episodes on simulator")
    print(f"Warmup (no exploration): {warmup_episodes} episodes")
    
    # Create environment
    env = env_class(seed=seed)
    
    # Check if state padding is needed (e.g., for remifentanil history)
    env_state_dim = env.observation_space.shape[0]
    agent_state_dim = agent.state_dim
    needs_padding = agent_state_dim > env_state_dim
    
    if needs_padding:
        padding_dim = agent_state_dim - env_state_dim
        print(f"\n⚠️  State dimension mismatch detected:")
        print(f"  Environment: {env_state_dim}D")
        print(f"  Agent: {agent_state_dim}D")
        print(f"  Padding with {padding_dim} zeros (remifentanil features)")
        
        # Use FIXED remifentanil values for padding (not random!)
        # Random noise causes agent confusion - use consistent values from offline training
        remi_ce_fixed = 0.1  # Median from VitalDB training data
        remi_rate_fixed = 0.2  # Median from VitalDB training data
    
    def pad_state(state: np.ndarray) -> np.ndarray:
        """Pad state with remifentanil features if needed."""
        if needs_padding:
            # Use FIXED padding (not random!) to match offline training distribution
            # This maintains consistency with what the agent learned offline
            padding = np.array([remi_ce_fixed, remi_rate_fixed])
            return np.concatenate([state, padding[:padding_dim]])
        return state
    
    # Create evaluation patients
    eval_patients = create_patient_population(5, seed=seed + 1000)
    
    # Training metrics
    episode_rewards = []
    episode_mdape = []
    episode_time_in_target = []
    
    best_reward = -float('inf')
    best_mdape = float('inf')
    
    # Create file for logging episode rewards
    rewards_log_path = dirs['stage2'] / 'episode_rewards.txt'
    rewards_log_file = open(rewards_log_path, 'w')
    
    print(f"\nStarting online fine-tuning...")
    print(f"Logging rewards to: {rewards_log_path}\n")
    
    # Training metrics tracking
    episode_steps = []
    episode_actor_losses = []
    episode_critic_losses = []
    
    for episode in tqdm(range(n_episodes), desc="Online Fine-tuning"):
        state, _ = env.reset()
        state = pad_state(state)  # Pad if needed
        agent.reset_noise()
        
        episode_reward = 0.0
        episode_step = 0
        done = False
        
        # Enable exploration after warmup
        add_noise = episode >= warmup_episodes
        
        while not done:
            episode_step += 1
            # Select action (Quantum Actor)
            action = agent.select_action(state, add_noise=add_noise)
            
            # Step environment (Simulator)
            action_normalized = np.clip(action / agent.action_scale, 0, 1)
            next_state, reward, terminated, truncated, _ = env.step(action_normalized)
            next_state = pad_state(next_state)  # Pad if needed
            done = terminated or truncated
            
            # Online RL update
            agent.train_step(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
        
        # Decay exploration noise
        if add_noise:
            agent.decay_noise()
        
        # Record metrics
        episode_steps.append(episode_step)
        if hasattr(env, 'get_episode_metrics'):
            metrics = env.get_episode_metrics()
            episode_mdape.append(metrics.get('mdape', 0))
            episode_time_in_target.append(metrics.get('time_in_target', 0))
        else:
            # For environments without get_episode_metrics, use placeholder values
            episode_mdape.append(0)
            episode_time_in_target.append(0)
        
        episode_rewards.append(episode_reward)
        
        # 매 episode 간단한 로그 (10 episode마다)
        if (episode + 1) % 10 == 0 or episode == 0:
            noise_status = "Explore" if add_noise else "Warmup"
            recent_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
            print(f"Ep {episode+1:4d}/{n_episodes} [{noise_status:7s}] | Reward: {episode_reward:7.2f} (avg: {recent_reward:7.2f}) | Steps: {episode_step:3d}")
        
        # Log reward to file
        rewards_log_file.write(f"{episode_reward}\n")
        rewards_log_file.flush()
        
        # Periodic evaluation
        if (episode + 1) % eval_interval == 0 or episode == n_episodes - 1:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_mdape = np.mean(episode_mdape[-eval_interval:])
            avg_tit = np.mean(episode_time_in_target[-eval_interval:])
            
            # Evaluate on test patients (simple evaluation)
            eval_rewards = []
            eval_mdapes = []
            for patient in eval_patients:
                state, _ = env.reset(options={'patient': patient})
                state = pad_state(state)  # Pad if needed
                episode_reward = 0
                done = False
                
                while not done:
                    action = agent.select_action(state, deterministic=True)
                    action_normalized = np.clip(action / agent.action_scale, 0, 1)
                    next_state, reward, terminated, truncated, _ = env.step(action_normalized)
                    next_state = pad_state(next_state)  # Pad if needed
                    done = terminated or truncated
                    episode_reward += reward
                    state = next_state
                
                if hasattr(env, 'get_episode_metrics'):
                    metrics = env.get_episode_metrics()
                    eval_mdapes.append(metrics['mdape'])
                eval_rewards.append(episode_reward)
            
            print(f"\n  {'='*60}")
            print(f"  Episode {episode+1}/{n_episodes} Evaluation:")
            print(f"  {'='*60}")
            print(f"  Training Performance (last {eval_interval} episodes):")
            print(f"    - Avg Reward:        {avg_reward:7.2f}")
            print(f"    - Min/Max Reward:    {np.min(episode_rewards[-eval_interval:]):7.2f} / {np.max(episode_rewards[-eval_interval:]):7.2f}")
            if hasattr(env, 'get_episode_metrics'):
                print(f"    - Avg MDAPE:         {avg_mdape:7.2f}%")
                print(f"    - Avg Time in Target: {avg_tit:6.1f}%")
            print(f"    - Avg Steps:         {np.mean(episode_steps[-eval_interval:]):.1f}")
            if eval_mdapes:
                print(f"  Test Performance (5 patients):")
                print(f"    - MDAPE:             {np.mean(eval_mdapes):7.2f}% ± {np.std(eval_mdapes):.2f}%")
                print(f"    - Rewards:           {np.mean(eval_rewards):7.2f} ± {np.std(eval_rewards):.2f}")
            print(f"  {'='*60}")
            
            # Save best models
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(str(dirs['stage2'] / 'best_reward.pt'))
                print(f"    → New best reward: {best_reward:.2f}")
            
            if hasattr(env, 'get_episode_metrics') and avg_mdape < best_mdape:
                best_mdape = avg_mdape
                agent.save(str(dirs['stage2'] / 'best_mdape.pt'))
                print(f"    → New best MDAPE: {best_mdape:.2f}%")
    
    # Save final model
    final_path = dirs['stage2'] / 'final.pt'
    agent.save(str(final_path))
    
    # Close rewards log file
    rewards_log_file.close()
    
    # Save training history (pickle)
    history = {
        'rewards': episode_rewards,
        'mdape': episode_mdape,
        'time_in_target': episode_time_in_target
    }
    with open(dirs['stage2'] / 'training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    # Save training history to CSV
    episode_df = pd.DataFrame({
        'episode': range(1, n_episodes + 1),
        'reward': episode_rewards,
        'mdape': episode_mdape,
        'time_in_target': episode_time_in_target,
        'steps': episode_steps,
        'phase': ['warmup' if i < warmup_episodes else 'exploration' for i in range(n_episodes)]
    })
    episode_csv_path = dirs['stage2'] / 'episode_history.csv'
    episode_df.to_csv(episode_csv_path, index=False)
    print(f"✓ Episode history saved to CSV: {episode_csv_path}")
    
    print(f"\n✓ Stage 2 Complete: Online Fine-tuning")
    print(f"  Best reward: {best_reward:.2f}")
    if hasattr(env, 'get_episode_metrics'):
        print(f"  Best MDAPE: {best_mdape:.2f}%")
    print(f"  Best MDAPE: {best_mdape:.2f}%")
    print(f"  Saved models:")
    print(f"    - Best reward: {dirs['stage2'] / 'best_reward.pt'}")
    print(f"    - Best MDAPE: {dirs['stage2'] / 'best_mdape.pt'}")
    print(f"    - Final: {final_path}")
    
    env.close()
    return agent


def stage3_testing(
    agent_pretrained: QuantumDDPGAgent,
    agent_finetuned: QuantumDDPGAgent,
    vitaldb_test_data: Dict,
    dirs: Dict[str, Path],
    seed: int,
    device: torch.device = None
):
    """
    Stage 3: Testing on BOTH VitalDB test set AND simulator test set.
    
    Compares pre-trained (offline only) vs fine-tuned (offline + online).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("STAGE 3: TESTING & COMPARISON")
    print("="*70)
    print(f"Device: {device}")
    
    results = {
        'pretrained': {},
        'finetuned': {}
    }
    
    # ========================================
    # Test 1: VitalDB Test Set (Real Data)
    # ========================================
    print("\n[Test 1] VitalDB Test Set (Real Patient Data - 10%)")
    print("-" * 70)
    
    test_states = vitaldb_test_data['states']
    
    def evaluate_on_vitaldb(agent, name):
        predicted_actions = []
        ground_truth_actions = []
        
        # Evaluate action prediction accuracy
        for i in range(0, min(1000, len(test_states)), 5):  # Sample every 5th
            state = test_states[i]
            action = vitaldb_test_data['actions'][i]
            
            # Get predicted action from agent
            pred_action = agent.select_action(state, deterministic=True)
            
            predicted_actions.append(pred_action[0] if len(pred_action) > 0 else pred_action)
            ground_truth_actions.append(action[0] if len(action) > 0 else action)
        
        predicted_actions = np.array(predicted_actions)
        ground_truth_actions = np.array(ground_truth_actions)
        
        # Calculate MDAPE (Median Absolute Percentage Error)
        ape = np.abs(predicted_actions - ground_truth_actions) / (np.abs(ground_truth_actions) + 1e-8) * 100
        mdape = np.median(ape)
        mae = np.mean(np.abs(predicted_actions - ground_truth_actions))
        
        print(f"  {name}:")
        print(f"    Action MDAPE: {mdape:.2f}%")
        print(f"    Action MAE: {mae:.4f}")
        print(f"    Pred range: [{predicted_actions.min():.4f}, {predicted_actions.max():.4f}]")
        print(f"    True range: [{ground_truth_actions.min():.4f}, {ground_truth_actions.max():.4f}]")
        
        return mdape
    
    pretrained_error = evaluate_on_vitaldb(agent_pretrained, "Pre-trained (Offline only)")
    finetuned_error = evaluate_on_vitaldb(agent_finetuned, "Fine-tuned (Offline + Online)")
    
    results['vitaldb'] = {
        'pretrained_mdape': pretrained_error,
        'finetuned_mdape': finetuned_error,
        'improvement': pretrained_error - finetuned_error
    }
    
    print(f"\n  MDAPE Improvement: {results['vitaldb']['improvement']:.2f}% (lower is better)")
    
    # ========================================
    # Test 2: Simulator Test Set (Synthetic)
    # ========================================
    print("\n[Test 2] Simulator Test Set (20 Diverse Synthetic Patients)")
    print("-" * 70)
    
    from environment.patient_simulator import create_patient_population
    test_patients = create_patient_population(20, seed=seed + 2000)
    env = PropofolEnv(seed=seed)
    
    def evaluate_on_simulator(agent, name):
        rewards = []
        mdapes = []
        times_in_target = []
        
        for patient in test_patients:
            state, _ = env.reset(options={'patient': patient})
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state, deterministic=True)
                action_normalized = np.clip(action / agent.action_scale, 0, 1)
                next_state, reward, terminated, truncated, _ = env.step(action_normalized)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
            
            metrics = env.get_episode_metrics()
            rewards.append(episode_reward)
            mdapes.append(metrics['mdape'])
            times_in_target.append(metrics['time_in_target'])
        
        print(f"  {name}:")
        print(f"    Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"    Mean MDAPE: {np.mean(mdapes):.2f}% ± {np.std(mdapes):.2f}%")
        print(f"    Mean Time in Target: {np.mean(times_in_target):.1f}%")
        
        return {
            'reward': np.mean(rewards),
            'mdape': np.mean(mdapes),
            'time_in_target': np.mean(times_in_target)
        }
    
    pretrained_sim = evaluate_on_simulator(agent_pretrained, "Pre-trained (Offline only)")
    finetuned_sim = evaluate_on_simulator(agent_finetuned, "Fine-tuned (Offline + Online)")
    
    results['simulator'] = {
        'pretrained': pretrained_sim,
        'finetuned': finetuned_sim,
        'reward_improvement': finetuned_sim['reward'] - pretrained_sim['reward'],
        'mdape_improvement': pretrained_sim['mdape'] - finetuned_sim['mdape']
    }
    
    print(f"\n  Improvements from fine-tuning:")
    print(f"    Reward: +{results['simulator']['reward_improvement']:.2f}")
    print(f"    MDAPE: -{results['simulator']['mdape_improvement']:.2f}% (lower is better)")
    
    env.close()
    
    # ========================================
    # Save Results
    # ========================================
    with open(dirs['stage3'] / 'test_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Stage 3 Complete: Testing")
    print(f"  Results saved: {dirs['stage3'] / 'test_results.pkl'}")
    
    return results


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set default dtype to float32 to avoid Double/Float mismatch
    torch.set_default_dtype(torch.float32)
    
    # Set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"DEVICE CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*70}\n")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add device to config
    config['device'] = str(device)
    
    # Setup directories
    dirs = setup_directories(args.log_dir)
    
    # Save config
    with open(dirs['base'] / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Save args
    with open(dirs['base'] / 'args.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    print("\n" + "="*70)
    print("HYBRID QUANTUM RL TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  VitalDB cases: {args.n_cases}")
    print(f"  Offline epochs: {args.offline_epochs}")
    print(f"  Online episodes: {args.online_episodes}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Seed: {args.seed}")
    print(f"  Log directory: {dirs['base']}")
    print("="*70)
    
    # ========================================
    # Data Loading & Splitting
    # ========================================
    train_data, val_data, test_data = split_vitaldb_data(
        n_total_cases=args.n_cases,
        data_path=args.data_path,
        dual_drug=args.dual_drug
    )
    
    # Determine state and action dimensions from data
    state_dim = train_data['states'].shape[1] if len(train_data['states'].shape) > 1 else 1
    action_dim = train_data['actions'].shape[1] if len(train_data['actions'].shape) > 1 else 1
    
    print(f"\n✓ Data dimensions:")
    print(f"  State dim: {state_dim} ({'dual drug' if args.dual_drug else 'single drug'})")
    print(f"  Action dim: {action_dim}")
    
    # ========================================
    # Stage 1: Offline Pre-training
    # ========================================
    if args.skip_offline:
        if not args.resume:
            print("\nError: --resume required when --skip_offline is used")
            return
        
        print(f"\nSkipping offline training, loading from: {args.resume}")
        agent = QuantumDDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config,
            encoder_type=args.encoder,
            seed=args.seed
        )
        agent.load(args.resume)
    else:
        # Create agent
        agent = QuantumDDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config,
            encoder_type=args.encoder,
            seed=args.seed
        )
        
        # Move agent to device first
        agent.actor = agent.actor.to(device)
        agent.actor_target = agent.actor_target.to(device)
        agent.critic = agent.critic.to(device)
        agent.critic_target = agent.critic_target.to(device)
        
        # Convert all networks and parameters to float32
        agent.actor.float()
        agent.actor_target.float()
        agent.critic.float()
        agent.critic_target.float()
        
        # Convert all parameters to float32 (including VQC parameters)
        for param in agent.actor.parameters():
            param.data = param.data.float()
        for param in agent.actor_target.parameters():
            param.data = param.data.float()
        for param in agent.critic.parameters():
            param.data = param.data.float()
        for param in agent.critic_target.parameters():
            param.data = param.data.float()
        
        # Print quantum info
        q_info = agent.get_quantum_info()
        print(f"\n✓ Quantum Actor initialized:")
        print(f"  Qubits: {q_info['n_qubits']}")
        print(f"  Layers: {q_info['n_layers']}")
        print(f"  Parameters: {q_info['n_params']}")
        print(f"  Device: {device}")
        
        # Train offline
        agent = stage1_offline_pretraining(
            agent=agent,
            train_data=train_data,
            val_data=val_data,
            n_epochs=args.offline_epochs,
            batch_size=args.batch_size,
            bc_weight=float(args.bc_weight),
            dirs=dirs,
            eval_interval=args.eval_interval,
            device=device
        )
    
    # Save pre-trained agent for comparison
    agent_pretrained = QuantumDDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        encoder_type=args.encoder,
        seed=args.seed
    )
    if args.skip_offline:
        agent_pretrained.load(args.resume)
    else:
        agent_pretrained.load(str(dirs['stage1'] / 'best_val.pt'))
    
    # ========================================
    # Stage 2: Online Fine-tuning
    # ========================================
    if args.skip_online:
        print("\nSkipping online fine-tuning")
        agent_finetuned = agent_pretrained
    else:
        agent_finetuned = stage2_online_finetuning(
            agent=agent,
            n_episodes=args.online_episodes,
            warmup_episodes=args.warmup_episodes,
            dirs=dirs,
            eval_interval=args.eval_interval,
            seed=args.seed,
            device=device
        )
        
        # Load best fine-tuned model for testing
        best_finetuned = QuantumDDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config,
            encoder_type=args.encoder,
            seed=args.seed
        )
        best_finetuned.load(str(dirs['stage2'] / 'best_mdape.pt'))
        agent_finetuned = best_finetuned
    
    # ========================================
    # Stage 3: Testing
    # ========================================
    results = stage3_testing(
        agent_pretrained=agent_pretrained,
        agent_finetuned=agent_finetuned,
        vitaldb_test_data=test_data,
        dirs=dirs,
        seed=args.seed,
        device=device
    )
    
    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "="*70)
    print("✓ HYBRID TRAINING COMPLETE!")
    print("="*70)
    print("\nFinal Results:")
    print(f"  VitalDB Test Set:")
    print(f"    Improvement: {results['vitaldb']['improvement']:.2f} BIS error reduction")
    print(f"  Simulator Test Set:")
    print(f"    Reward improvement: +{results['simulator']['reward_improvement']:.2f}")
    print(f"    MDAPE improvement: -{results['simulator']['mdape_improvement']:.2f}%")
    
    print(f"\nSaved models:")
    print(f"  Stage 1 (Offline):")
    print(f"    - {dirs['stage1'] / 'best_val.pt'}")
    print(f"  Stage 2 (Online):")
    print(f"    - {dirs['stage2'] / 'best_mdape.pt'}")
    print(f"    - {dirs['stage2'] / 'best_reward.pt'}")
    print("="*70)


if __name__ == "__main__":
    main()
