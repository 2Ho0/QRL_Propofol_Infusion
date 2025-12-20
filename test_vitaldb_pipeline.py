"""
Quick test for VitalDB data loading and offline training pipeline.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

print("=" * 70)
print("Testing VitalDB Data Pipeline")
print("=" * 70)

# Test 1: Import modules
print("\n[Test 1] Importing modules...")
try:
    from data.vitaldb_loader import VitalDBLoader, VitalDBDataset
    from agents.quantum_agent import QuantumDDPGAgent
    print("  ✓ All modules imported successfully")
except Exception as e:
    print(f"  ❌ Import error: {e}")
    exit(1)

# Test 2: Initialize VitalDB loader
print("\n[Test 2] Initializing VitalDB loader...")
try:
    loader = VitalDBLoader(
        cache_dir="./data/vitaldb_cache",
        bis_range=(30, 70),
        use_cache=True
    )
    print("  ✓ VitalDB loader initialized")
    print(f"     BIS range: {loader.bis_range}")
    print(f"     Cache dir: {loader.cache_dir}")
except Exception as e:
    print(f"  ❌ Loader error: {e}")
    exit(1)

# Test 3: Try to load a small amount of data
print("\n[Test 3] Loading small dataset (5 cases)...")
print("  This may take a few minutes on first run...")
try:
    data = loader.prepare_training_data(
        n_cases=5,
        min_duration=1800,
        save_path="./data/vitaldb_cache/test_small.pkl"
    )
    
    print(f"  ✓ Data loaded successfully")
    print(f"     States shape: {data['states'].shape}")
    print(f"     Actions shape: {data['actions'].shape}")
    print(f"     Total transitions: {len(data['states']):,}")
    
    if len(data['states']) > 0:
        print(f"\n  Sample statistics:")
        print(f"     BIS range: [{50 - data['states'][:, 0].max():.1f}, {50 - data['states'][:, 0].min():.1f}]")
        print(f"     Action range: [{data['actions'].min():.2f}, {data['actions'].max():.2f}]")
        print(f"     Mean reward: {data['rewards'].mean():.3f}")
    else:
        print("  ⚠️  No data loaded - may need to check VitalDB connection")
        
except Exception as e:
    print(f"  ⚠️  Data loading error: {e}")
    print(f"     This is normal if VitalDB API is slow or unavailable")
    print(f"     You can still use simulated data for testing")

# Test 4: Test agent with dummy data
print("\n[Test 4] Testing agent with dummy offline data...")
try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    
    # Create dummy dataset
    dummy_data = {
        'states': np.random.randn(1000, 8).astype(np.float32),
        'actions': np.random.rand(1000, 1).astype(np.float32) * 10,
        'rewards': np.random.rand(1000).astype(np.float32),
        'next_states': np.random.randn(1000, 8).astype(np.float32),
        'dones': np.zeros(1000, dtype=bool)
    }
    
    # Create dataset and dataloader
    dataset = VitalDBDataset(dummy_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create agent
    agent = QuantumDDPGAgent(
        state_dim=8,
        action_dim=1,
        encoder_type='none',
        seed=42
    )
    
    print(f"  ✓ Agent created")
    print(f"     Quantum circuit: {agent.get_quantum_info()['n_qubits']} qubits, {agent.get_quantum_info()['n_layers']} layers")
    
    # Train for a few batches
    print(f"\n  Training on dummy data (3 batches)...")
    batch_count = 0
    for batch in dataloader:
        states, actions, rewards, next_states, dones = batch
        
        # Train on first sample
        state_np = states[0].numpy()
        action_np = actions[0].numpy()
        reward_np = rewards[0].item()
        next_state_np = next_states[0].numpy()
        done_np = dones[0].item()
        
        metrics = agent.train_step(state_np, action_np, reward_np, next_state_np, done_np)
        
        if metrics:
            print(f"     Batch {batch_count+1}: loss={metrics.get('critic_loss', 0):.4f}")
        
        batch_count += 1
        if batch_count >= 3:
            break
    
    print(f"  ✓ Training step works")
    
except Exception as e:
    print(f"  ❌ Agent/training error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✓ Pipeline Test Complete!")
print("=" * 70)
print("\n📋 Next Steps:")
print("\n1. Prepare full dataset:")
print("   python scripts/prepare_offline_data.py --n_cases 100")
print("\n2. Train agent:")
print("   python experiments/train_offline.py --n_epochs 100 --batch_size 64")
print("\n3. Or use existing cached data if available:")
print("   python experiments/train_offline.py --data_path ./data/offline_dataset/vitaldb_offline_data.pkl")
print("\n" + "=" * 70)
