"""
Classical RL Training Script
=============================

Train Classical DDPG agent (pure MLP, no quantum) for baseline comparison.

This script mirrors train_hybrid.py but uses ClassicalDDPGAgent instead of
QuantumDDPGAgent to provide a fair comparison baseline.

Usage:
    # Full training
    python experiments/train_classical.py --n_cases 100 --offline_epochs 50 --online_episodes 500
    
    # Quick test
    python experiments/train_classical.py --n_cases 20 --offline_epochs 5 --online_episodes 50
    
    # With LSTM encoder
    python experiments/train_classical.py --n_cases 100 --encoder lstm
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from datetime import datetime
from typing import Dict, Tuple

from agents.classical_agent import ClassicalDDPGAgent
from environment.propofol_env import PropofolEnv
from environment.patient_simulator import create_patient_population
from data.vitaldb_loader import VitalDBLoader, VitalDBDataset
from models.networks import soft_update

# Import same functions from train_hybrid for consistency
import train_hybrid


def main():
    # Reuse argument parser from train_hybrid
    args = train_hybrid.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set default dtype to float32
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
    
    # Setup directories (with 'classical' prefix)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = Path(args.log_dir) / f'classical_{timestamp}'
    
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
    
    # Save config
    with open(dirs['base'] / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Save args
    with open(dirs['base'] / 'args.txt', 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    print("="*70)
    print("CLASSICAL DDPG TRAINING")
    print("="*70)
    print(f"Configuration:")
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
    train_data, val_data, test_data = train_hybrid.split_vitaldb_data(
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
        agent = ClassicalDDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config,
            encoder_type=args.encoder,
            seed=args.seed
        )
        agent.load(args.resume)
        
        # Move to device
        agent.actor = agent.actor.to(device)
        agent.actor_target = agent.actor_target.to(device)
        agent.critic = agent.critic.to(device)
        agent.critic_target = agent.critic_target.to(device)
        if agent.encoder is not None:
            agent.encoder = agent.encoder.to(device)
            agent.encoder_target = agent.encoder_target.to(device)
    else:
        # Create Classical agent
        agent = ClassicalDDPGAgent(
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
        
        if agent.encoder is not None:
            agent.encoder = agent.encoder.to(device)
            agent.encoder_target = agent.encoder_target.to(device)
        
        # Convert to float32
        agent.actor.float()
        agent.actor_target.float()
        agent.critic.float()
        agent.critic_target.float()
        
        # Print agent info
        agent_info = agent.get_info()
        print(f"\n✓ Classical DDPG Agent initialized:")
        print(f"  Type: {agent_info['type']}")
        print(f"  State dim: {agent_info['state_dim']}")
        print(f"  Action dim: {agent_info['action_dim']}")
        print(f"  Encoder: {agent_info['encoder_type']}")
        print(f"  Device: {device}")
        
        # Train offline (reuse stage1 function from train_hybrid with Classical agent)
        agent = train_hybrid.stage1_offline_pretraining(
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
    agent_pretrained = ClassicalDDPGAgent(
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
        agent_finetuned = train_hybrid.stage2_online_finetuning(
            agent=agent,
            n_episodes=args.online_episodes,
            warmup_episodes=args.warmup_episodes,
            dirs=dirs,
            eval_interval=args.eval_interval,
            seed=args.seed,
            device=device
        )
        
        # Load best fine-tuned model
        best_finetuned = ClassicalDDPGAgent(
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
    results = train_hybrid.stage3_testing(
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
    print("✓ CLASSICAL TRAINING COMPLETE!")
    print("="*70)
    print("\nFinal Results:")
    
    # VitalDB Test Results
    print(f"  VitalDB Test:")
    if 'vitaldb' in results:
        vitaldb_results = results['vitaldb']
        # Handle different possible key names
        pretrained_key = 'pretrained_mdape' if 'pretrained_mdape' in vitaldb_results else 'mdape_pretrained'
        finetuned_key = 'finetuned_mdape' if 'finetuned_mdape' in vitaldb_results else 'mdape_finetuned'
        
        if pretrained_key in vitaldb_results:
            print(f"    Pre-trained MDAPE: {vitaldb_results[pretrained_key]:.2f}%")
        if finetuned_key in vitaldb_results:
            print(f"    Fine-tuned MDAPE:  {vitaldb_results[finetuned_key]:.2f}%")
    
    # Simulator Test Results
    print(f"  Simulator Test:")
    if 'simulator' in results:
        simulator_results = results['simulator']
        # Handle different possible key names
        pretrained_key = 'pretrained_mdape' if 'pretrained_mdape' in simulator_results else 'mdape_pretrained'
        finetuned_key = 'finetuned_mdape' if 'finetuned_mdape' in simulator_results else 'mdape_finetuned'
        
        if pretrained_key in simulator_results:
            print(f"    Pre-trained MDAPE: {simulator_results[pretrained_key]:.2f}%")
        if finetuned_key in simulator_results:
            print(f"    Fine-tuned MDAPE:  {simulator_results[finetuned_key]:.2f}%")
    
    print(f"\n  Log directory: {dirs['base']}")
    print("="*70)


if __name__ == "__main__":
    main()
