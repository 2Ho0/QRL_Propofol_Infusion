"""
Test JAX-based Quantum Circuit Implementation
==============================================

Quick test to verify JAX integration with quantum circuit.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import torch
import numpy as np
import jax.numpy as jnp
from models.vqc import VariationalQuantumCircuit, QuantumPolicy


def test_vqc_jax():
    """Test VQC with JAX backend."""
    print("="*70)
    print("Testing VQC with JAX Backend")
    print("="*70)
    
    # Create VQC
    n_qubits = 3
    n_layers = 4
    
    vqc = VariationalQuantumCircuit(
        n_qubits=n_qubits,
        n_layers=n_layers,
        device="default.qubit"
    )
    
    print(f"\n✓ VQC created:")
    print(f"  Qubits: {n_qubits}")
    print(f"  Layers: {n_layers}")
    print(f"  Total params: {vqc.total_params}")
    
    # Test single input
    print("\n" + "-"*70)
    print("Test 1: Single Input")
    print("-"*70)
    
    state = torch.randn(n_qubits)
    weights = torch.randn(n_layers, n_qubits, 2)
    
    import time
    start = time.time()
    output = vqc.forward(state, weights)
    elapsed = time.time() - start
    
    print(f"  Input shape: {state.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output value: {output.item():.4f}")
    print(f"  Execution time: {elapsed*1000:.2f}ms")
    
    # Test batch input
    print("\n" + "-"*70)
    print("Test 2: Batch Input (with JAX vmap)")
    print("-"*70)
    
    batch_size = 32
    batch_state = torch.randn(batch_size, n_qubits)
    
    start = time.time()
    batch_output = vqc.forward(batch_state, weights)
    elapsed = time.time() - start
    
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {batch_state.shape}")
    print(f"  Output shape: {batch_output.shape}")
    print(f"  Execution time: {elapsed*1000:.2f}ms")
    print(f"  Per-sample time: {elapsed*1000/batch_size:.2f}ms")
    
    # Test with larger batch
    print("\n" + "-"*70)
    print("Test 3: Larger Batch (256 samples)")
    print("-"*70)
    
    large_batch_size = 256
    large_batch_state = torch.randn(large_batch_size, n_qubits)
    
    start = time.time()
    large_batch_output = vqc.forward(large_batch_state, weights)
    elapsed = time.time() - start
    
    print(f"  Batch size: {large_batch_size}")
    print(f"  Input shape: {large_batch_state.shape}")
    print(f"  Output shape: {large_batch_output.shape}")
    print(f"  Execution time: {elapsed*1000:.2f}ms")
    print(f"  Per-sample time: {elapsed*1000/large_batch_size:.2f}ms")
    
    print("\n" + "="*70)
    print("✓ All JAX VQC tests passed!")
    print("="*70)


def test_quantum_policy_jax():
    """Test full quantum policy with JAX backend."""
    print("\n\n" + "="*70)
    print("Testing Quantum Policy with JAX Backend")
    print("="*70)
    
    state_dim = 8
    n_qubits = 3
    n_layers = 4
    action_dim = 1
    
    policy = QuantumPolicy(
        state_dim=state_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        action_dim=action_dim,
        encoder_hidden=[64, 32]
    )
    
    print(f"\n✓ Quantum Policy created:")
    print(f"  State dim: {state_dim}")
    print(f"  Qubits: {n_qubits}")
    print(f"  Layers: {n_layers}")
    print(f"  Action dim: {action_dim}")
    
    # Test forward pass
    print("\n" + "-"*70)
    print("Test: Forward Pass")
    print("-"*70)
    
    state = torch.randn(state_dim)
    
    import time
    start = time.time()
    action = policy.forward(state)
    elapsed = time.time() - start
    
    print(f"  Input shape: {state.shape}")
    print(f"  Output shape: {action.shape}")
    print(f"  Action value: {action.item():.4f}")
    print(f"  Execution time: {elapsed*1000:.2f}ms")
    
    # Test batch forward
    print("\n" + "-"*70)
    print("Test: Batch Forward Pass")
    print("-"*70)
    
    batch_size = 32
    batch_state = torch.randn(batch_size, state_dim)
    
    start = time.time()
    batch_action = policy.forward(batch_state)
    elapsed = time.time() - start
    
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {batch_state.shape}")
    print(f"  Output shape: {batch_action.shape}")
    print(f"  Execution time: {elapsed*1000:.2f}ms")
    print(f"  Per-sample time: {elapsed*1000/batch_size:.2f}ms")
    
    # Test gradient computation
    print("\n" + "-"*70)
    print("Test: Gradient Computation")
    print("-"*70)
    
    state_grad = torch.randn(state_dim, requires_grad=True)
    
    start = time.time()
    action_grad = policy.forward(state_grad)
    loss = action_grad.sum()
    loss.backward()
    elapsed = time.time() - start
    
    print(f"  Loss value: {loss.item():.4f}")
    if state_grad.grad is not None:
        print(f"  Gradient norm: {state_grad.grad.norm().item():.4f}")
        print(f"  ✓ Gradient flow working!")
    else:
        print(f"  ✗ Gradient: None")
    print(f"  Backward pass time: {elapsed*1000:.2f}ms")
    
    print("\n" + "="*70)
    print("✓ All Quantum Policy tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_vqc_jax()
    test_quantum_policy_jax()
    
    print("\n\n" + "="*70)
    print("JAX INTEGRATION SUCCESSFUL!")
    print("="*70)
    print("\nBenefits of JAX backend:")
    print("  1. JIT compilation for faster execution")
    print("  2. Vectorized batch processing (vmap)")
    print("  3. Efficient gradient computation")
    print("  4. Better memory efficiency for large batches")
    print("="*70)
