"""
Variational Quantum Circuit (VQC) for Policy Network
======================================================

This module implements a 2-qubit Variational Quantum Circuit using PennyLane
with JAX backend for the quantum policy network in the Quantum RL Propofol control system.

Architecture:
-------------
1. State Encoding: Angle embedding of state features onto qubits
2. Variational Layers: Parameterized rotation gates (RY, RZ) with entanglement (CNOT)
3. Measurement: Expectation values of Pauli-Z operators

The circuit maps classical state features to a quantum-enhanced action
through variational quantum optimization.

For 2 Qubits:
- Input: 2 state features (BIS_error_normalized, Ce_normalized)
- Encoding: Angle embedding RX(feature * π)
- Layers: RY-RZ rotations + CNOT entanglement
- Output: Expectation value mapped to action [0, 1]

JAX Backend:
-------------
- Uses PennyLane with JAX interface for faster quantum gradient computation
- JIT compilation for circuit execution speedup
- Compatible with PyTorch models through conversion
"""

import numpy as np
import pennylane as qml
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
import os

# Force JAX to use GPU
os.environ['JAX_PLATFORMS'] = 'cuda'
# Enable XLA optimizations
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'


class JAXCircuitFunction(torch.autograd.Function):
    """
    Custom PyTorch autograd function to bridge JAX quantum circuit with PyTorch.
    
    This enables gradient flow from JAX backend to PyTorch models.
    Optimized for GPU execution with batched operations.
    """
    
    @staticmethod
    def forward(ctx, circuit_fn, inputs, weights):
        """
        Forward pass: Execute JAX circuit on GPU.
        
        Args:
            circuit_fn: JAX quantum circuit function
            inputs: Input features (PyTorch tensor)
            weights: Variational parameters (PyTorch tensor)
        
        Returns:
            Circuit output as PyTorch tensor
        """
        # Convert to JAX arrays and place on GPU
        with jax.default_device(jax.devices('gpu')[0]):
            inputs_jax = jnp.asarray(inputs.detach().cpu().numpy())
            weights_jax = jnp.asarray(weights.detach().cpu().numpy())
            
            # Execute circuit on GPU
            output_jax = circuit_fn(inputs_jax, weights_jax)
            
            # Convert back to numpy on host (make writable copy)
            output_np = np.asarray(output_jax).copy()
        
        # Store for backward
        ctx.circuit_fn = circuit_fn
        ctx.save_for_backward(inputs, weights)
        ctx.inputs_jax = inputs_jax
        ctx.weights_jax = weights_jax
        
        # Convert to PyTorch
        output = torch.from_numpy(output_np).float()
        if inputs.is_cuda:
            output = output.to(inputs.device)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradients using JAX on GPU.
        
        Args:
            grad_output: Gradient from downstream
        
        Returns:
            Gradients w.r.t inputs and weights
        """
        inputs, weights = ctx.saved_tensors
        circuit_fn = ctx.circuit_fn
        inputs_jax = ctx.inputs_jax
        weights_jax = ctx.weights_jax
        
        # Execute gradient computation on GPU
        with jax.default_device(jax.devices('gpu')[0]):
            # Define function to differentiate
            def loss_fn(inputs_jax, weights_jax):
                return circuit_fn(inputs_jax, weights_jax)
            
            # Compute gradients using JAX
            grad_fn = jax.grad(loss_fn, argnums=(0, 1))
            grads_jax = grad_fn(inputs_jax, weights_jax)
            
            # Convert gradients back to numpy (make writable copies)
            grad_inputs_jax, grad_weights_jax = grads_jax
            grad_inputs_np = np.asarray(grad_inputs_jax).copy()
            grad_weights_np = np.asarray(grad_weights_jax).copy()
        
        # Convert to PyTorch
        grad_inputs = torch.from_numpy(grad_inputs_np).float()
        grad_weights = torch.from_numpy(grad_weights_np).float()
        
        if inputs.is_cuda:
            grad_inputs = grad_inputs.to(inputs.device)
            grad_weights = grad_weights.to(inputs.device)
        
        # Scale by upstream gradient
        grad_output_np = grad_output.detach().cpu().numpy()
        if grad_output.numel() == 1:
            grad_inputs = grad_inputs * grad_output.item()
            grad_weights = grad_weights * grad_output.item()
        else:
            # Handle batch case
            grad_inputs = grad_inputs * grad_output.unsqueeze(-1)
            grad_weights = grad_weights * grad_output.mean()
        
        return None, grad_inputs, grad_weights


class JAXCircuitBatchFunction(torch.autograd.Function):
    """
    Custom PyTorch autograd function for batched JAX quantum circuits.
    Optimized for GPU execution with vectorized operations.
    """
    
    @staticmethod
    def forward(ctx, circuit_batch_fn, inputs, weights):
        """Forward pass for batch on GPU."""
        # Execute on GPU
        with jax.default_device(jax.devices('gpu')[0]):
            inputs_jax = jnp.asarray(inputs.detach().cpu().numpy())
            weights_jax = jnp.asarray(weights.detach().cpu().numpy())
            
            # Execute batched circuit on GPU
            outputs_jax = circuit_batch_fn(inputs_jax, weights_jax)
            
            # Convert to numpy (make writable copy)
            outputs_np = np.asarray(outputs_jax).copy()
        
        ctx.circuit_batch_fn = circuit_batch_fn
        ctx.save_for_backward(inputs, weights)
        ctx.inputs_jax = inputs_jax
        ctx.weights_jax = weights_jax
        
        # Convert to PyTorch
        outputs = torch.from_numpy(outputs_np).float()
        if inputs.is_cuda:
            outputs = outputs.to(inputs.device)
        
        return outputs
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for batch on GPU."""
        inputs, weights = ctx.saved_tensors
        circuit_batch_fn = ctx.circuit_batch_fn
        inputs_jax = ctx.inputs_jax
        weights_jax = ctx.weights_jax
        
        # Execute gradient computation on GPU
        with jax.default_device(jax.devices('gpu')[0]):
            # Define vectorized gradient function
            def single_circuit(input_single):
                return circuit_batch_fn(input_single.reshape(1, -1), weights_jax)[0]
            
            # Compute gradients for each sample using vmap
            grad_fn = jax.grad(single_circuit)
            grad_inputs_jax = vmap(grad_fn)(inputs_jax)
            
            # Gradient w.r.t weights (averaged over batch)
            def loss_fn(weights_jax):
                outputs = circuit_batch_fn(inputs_jax, weights_jax)
                return jnp.mean(outputs)
            
            grad_weights_jax = jax.grad(loss_fn)(weights_jax)
            
            # Convert to numpy (make writable copies)
            grad_inputs_np = np.asarray(grad_inputs_jax).copy()
            grad_weights_np = np.asarray(grad_weights_jax).copy()
        
        # Convert to PyTorch
        grad_inputs = torch.from_numpy(grad_inputs_np).float()
        grad_weights = torch.from_numpy(grad_weights_np).float()
        
        if inputs.is_cuda:
            grad_inputs = grad_inputs.to(inputs.device)
            grad_weights = grad_weights.to(inputs.device)
        
        # Scale by upstream gradient
        grad_output_np = grad_output.detach().cpu().numpy()
        grad_inputs = grad_inputs * grad_output.unsqueeze(-1)
        grad_weights = grad_weights * grad_output.sum()
        
        return None, grad_inputs, grad_weights
        
        # Gradient w.r.t weights (averaged over batch)
        def loss_fn(weights_jax):
            outputs = circuit_batch_fn(inputs_jax, weights_jax)
            return jnp.mean(outputs)
        
        grad_weights_jax = jax.grad(loss_fn)(weights_jax)
        
        # Convert to PyTorch
        grad_inputs = torch.from_numpy(np.array(grad_inputs_jax)).float()
        grad_weights = torch.from_numpy(np.array(grad_weights_jax)).float()
        
        if inputs.is_cuda:
            grad_inputs = grad_inputs.to(inputs.device)
            grad_weights = grad_weights.to(inputs.device)
        
        # Scale by upstream gradient
        grad_inputs = grad_inputs * grad_output.unsqueeze(-1)
        grad_weights = grad_weights * grad_output.sum()
        
        return None, grad_inputs, grad_weights


class VariationalQuantumCircuit:
    """
    Variational Quantum Circuit for continuous action output.
    
    This class implements a parameterized quantum circuit that takes
    encoded state features and outputs an expectation value that
    can be mapped to a continuous action.
    
    Architecture for 2 qubits:
    |0⟩ ─ RX(θ_in[0]) ─ RY(θ[0]) ─ RZ(θ[1]) ─●─ RY(θ[4]) ─ RZ(θ[5]) ─ M
                                              │
    |0⟩ ─ RX(θ_in[1]) ─ RY(θ[2]) ─ RZ(θ[3]) ─⊕─ RY(θ[6]) ─ RZ(θ[7]) ─ M
    
    Attributes:
        n_qubits: Number of qubits (fixed at 2)
        n_layers: Number of variational layers
        dev: PennyLane quantum device
        weight_shapes: Shape of variational parameters
    """
    
    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 4,
        device: str = "default.qubit",
        shots: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the VQC.
        
        Args:
            n_qubits: Number of qubits (2 for this implementation)
            n_layers: Number of variational layers
            device: PennyLane device name
            shots: Number of shots for measurement (None for analytic)
            seed: Random seed for device
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        
        # Create PennyLane device
        if shots is None:
            self.dev = qml.device(device, wires=n_qubits)
        else:
            self.dev = qml.device(device, wires=n_qubits, shots=shots)
        
        # Calculate parameter shapes
        # Each layer has: 2 RY + 2 RZ per qubit = 4 params per layer
        self.n_params_per_layer = 2 * n_qubits  # RY and RZ for each qubit
        self.total_params = self.n_params_per_layer * n_layers
        
        # Weight shapes for QNode
        self.weight_shapes = {"weights": (n_layers, n_qubits, 2)}  # 2 for RY, RZ
        
        # Create the quantum circuit as a QNode
        self._create_qnode()
    
    def _create_qnode(self):
        """Create the PennyLane QNode for the VQC with JAX interface."""
        
        @qml.qnode(self.dev, interface="jax", diff_method="backprop")
        def circuit(inputs, weights):
            """
            Variational quantum circuit with JAX.
            
            Args:
                inputs: Input features to encode (shape: [n_qubits])
                weights: Variational parameters (shape: [n_layers, n_qubits, 2])
            
            Returns:
                Expectation value of Pauli-Z on first qubit
            """
            # Input encoding using angle embedding
            # Scale inputs to [0, π] range for rotation gates
            for i in range(self.n_qubits):
                # Encode feature as rotation angle
                qml.RX(inputs[i] * jnp.pi, wires=i)
            
            # Variational layers
            for layer in range(self.n_layers):
                # Single-qubit rotations
                for qubit in range(self.n_qubits):
                    qml.RY(weights[layer, qubit, 0], wires=qubit)
                    qml.RZ(weights[layer, qubit, 1], wires=qubit)
                
                # Entangling layer (CNOT cascade)
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                
                # Circular entanglement for 2+ qubits
                if self.n_qubits > 1:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Measurement: expectation value of Z on first qubit
            return qml.expval(qml.PauliZ(0))
        
        # JIT compile the circuit for faster execution
        self.circuit = jit(circuit)
        
        # Create vectorized version for batch processing
        self.circuit_batch = jit(vmap(circuit, in_axes=(0, None)))
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the VQC with JAX backend and PyTorch gradient support.
        
        Args:
            inputs: Input features (shape: [batch_size, n_qubits] or [n_qubits])
            weights: Variational parameters (shape: [n_layers, n_qubits, 2])
        
        Returns:
            Expectation values (shape: [batch_size] or scalar)
        """
        # Handle batched inputs
        if inputs.dim() == 1:
            # Single input - use custom autograd function
            return JAXCircuitFunction.apply(self.circuit, inputs, weights)
        else:
            # Batch processing with vectorized circuit
            return JAXCircuitBatchFunction.apply(self.circuit_batch, inputs, weights)
    
    def get_initial_weights(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Get randomly initialized weights.
        
        Args:
            seed: Random seed for initialization
        
        Returns:
            Initialized weight tensor
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Initialize weights uniformly in [0, 2π]
        weights = torch.rand(self.n_layers, self.n_qubits, 2) * 2 * np.pi
        weights.requires_grad = True
        return weights
    
    def draw(self, inputs: Optional[np.ndarray] = None) -> str:
        """
        Draw the quantum circuit.
        
        ArConvert to JAX arrays
        inputs_jax = jnp.array(inputs)
        weights_jax = jnp.array(weights)
        
        # Use qml.draw with JAX arrays
        drawer = qml.draw(self.circuit)
        return drawer(inputs_jax, weights_jax
        Returns:
            String representation of the circuit
        """
        if inputs is None:
            inputs = np.zeros(self.n_qubits)
        
        weights = self.get_initial_weights().detach().numpy()
        
        # Use qml.draw
        drawer = qml.draw(self.circuit)
        return drawer(torch.tensor(inputs), torch.tensor(weights))


class QuantumPolicy(nn.Module):
    """
    Hybrid Quantum-Classical Policy Network.
    
    This network combines a classical state encoder with a VQC
    to produce continuous actions for propofol infusion control.
    
    Architecture:
    1. Classical Encoder: Maps full state to 2 features for qubits
    2. VQC: Variational quantum circuit for policy
    3. Output Scaling: Maps VQC output [-1, 1] to action [0, 1]
    
    Attributes:
        encoder: Classical neural network encoder
        vqc: Variational quantum circuit
        weights: Variational parameters (learned)
        action_scale: Maximum action value
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        n_qubits: int = 2,
        n_layers: int = 4,
        encoder_hidden: List[int] = [64, 32],
        action_scale: float = 1.0,
        action_dim: int = 1,
        device_name: str = "default.qubit",
        seed: Optional[int] = None
    ):
        """
        Initialize the quantum policy.
        
        Args:
            state_dim: Dimension of input state
            n_qubits: Number of qubits in VQC
            n_layers: Number of VQC layers
            encoder_hidden: Hidden layer sizes for encoder
            action_scale: Scale for output action
            action_dim: Dimension of action output
            device_name: PennyLane device name
            seed: Random seed
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.action_scale = action_scale
        self.action_dim = action_dim
        
        # Classical encoder: state_dim -> n_qubits features
        encoder_layers = []
        prev_dim = state_dim
        for hidden_dim in encoder_hidden:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, n_qubits))
        encoder_layers.append(nn.Tanh())  # Output in [-1, 1] for encoding
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # VQC
        self.vqc = VariationalQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device=device_name
        )
        
        # Variational parameters
        if seed is not None:
            torch.manual_seed(seed)
        self.weights = nn.Parameter(
            torch.rand(n_layers, n_qubits, 2) * 2 * np.pi
        )
        
        # Output scaling layer
        output_layer = nn.Linear(16, self.action_dim)
        
        # Initialize bias to match VitalDB data distribution
        # VitalDB normalized action mean: ~0.025 (5 μg/kg/min out of 200 max)
        # sigmoid(x) = 0.025 => x = log(0.025/0.975) ≈ -3.66
        # Start conservative to match expert data distribution
        nn.init.constant_(output_layer.bias, -3.5)  # sigmoid(-3.5) ≈ 0.029
        
        self.output_scale = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            output_layer,
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute action from state.
        
        Args:
            state: State tensor (shape: [batch_size, state_dim] or [state_dim])
        
        Returns:
            Action tensor (shape: [batch_size, 1] or [1])
        """
        # Ensure state is 2D
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = state.shape[0]
        
        # Encode state to qubit features
        encoded = self.encoder(state)  # [batch_size, n_qubits]
        
        # Pass through VQC
        # VQC expects inputs in range suitable for angle encoding
        vqc_outputs = []
        for i in range(batch_size):
            output = self.vqc.forward(encoded[i], self.weights)
            vqc_outputs.append(output)
        
        vqc_output = torch.stack(vqc_outputs)  # [batch_size]
        
        # Scale VQC output [-1, 1] to action through output layer
        # First normalize to [0, 1]
        normalized = (vqc_output + 1) / 2  # [batch_size]
        
        # Convert to float32 to match network dtype
        normalized = normalized.float()
        
        # Apply output scaling network
        action = self.output_scale(normalized.unsqueeze(-1)) * self.action_scale  # [batch_size, action_dim]
        
        if squeeze_output:
            action = action.squeeze(0)
        
        return action
    
    def get_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Get action from state for inference.
        
        Args:
            state: State array
            deterministic: Whether to use deterministic policy
        
        Returns:
            Action array
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action = self.forward(state_tensor)
            return action.numpy()
    
    def get_quantum_circuit_info(self) -> Dict:
        """
        Get information about the quantum circuit.
        
        Returns:
            Dictionary with circuit information
        """
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'n_params': self.vqc.total_params,
            'weight_shape': tuple(self.weights.shape),
            'circuit_diagram': self.vqc.draw()
        }


class QuantumPolicySimple(nn.Module):
    """
    Simplified Quantum Policy without classical encoder.
    
    This version directly encodes the first 2 state features
    into the quantum circuit without preprocessing.
    Useful for testing and simpler experiments.
    """
    
    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 4,
        action_scale: float = 1.0,
        feature_indices: List[int] = [0, 1],
        device_name: str = "default.qubit",
        seed: Optional[int] = None
    ):
        """
        Initialize simplified quantum policy.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of VQC layers
            action_scale: Scale for output action
            feature_indices: Which state features to encode
            device_name: PennyLane device
            seed: Random seed
        """
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.action_scale = action_scale
        self.feature_indices = feature_indices
        
        # VQC
        self.vqc = VariationalQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device=device_name
        )
        
        # Variational parameters
        if seed is not None:
            torch.manual_seed(seed)
        self.weights = nn.Parameter(
            torch.rand(n_layers, n_qubits, 2) * 2 * np.pi
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor
        
        Returns:
            Action tensor
        """
        # Ensure 2D
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = state.shape[0]
        
        # Extract features for encoding
        features = state[:, self.feature_indices]  # [batch_size, n_qubits]
        
        # Normalize features to [-1, 1] for encoding
        features = torch.tanh(features)
        
        # Pass through VQC
        vqc_outputs = []
        for i in range(batch_size):
            output = self.vqc.forward(features[i], self.weights)
            vqc_outputs.append(output)
        
        vqc_output = torch.stack(vqc_outputs)  # [batch_size]
        
        # Map [-1, 1] to [0, action_scale]
        action = ((vqc_output + 1) / 2) * self.action_scale
        action = action.unsqueeze(-1)  # [batch_size, 1]
        
        if squeeze_output:
            action = action.squeeze(0)
        
        return action


if __name__ == "__main__":
    # Test the VQC and policies
    print("Testing Variational Quantum Circuit...")
    
    # Create VQC
    vqc = VariationalQuantumCircuit(n_qubits=2, n_layers=4)
    print(f"VQC created with {vqc.total_params} parameters")
    
    # Test forward pass
    inputs = torch.tensor([0.5, -0.3])
    weights = vqc.get_initial_weights()
    output = vqc.forward(inputs, weights)
    print(f"VQC output: {output.item():.4f}")
    
    # Draw circuit
    print("\nQuantum Circuit:")
    print(vqc.draw())
    
    # Test QuantumPolicy
    print("\n" + "="*50)
    print("Testing QuantumPolicy...")
    
    policy = QuantumPolicy(
        state_dim=4,
        n_qubits=2,
        n_layers=4,
        encoder_hidden=[64, 32],
        seed=42
    )
    
    # Test forward pass
    state = torch.randn(4)
    action = policy.forward(state)
    print(f"State: {state.numpy()}")
    print(f"Action: {action.item():.4f}")
    
    # Test batched forward
    states = torch.randn(8, 4)
    actions = policy.forward(states)
    print(f"Batch actions shape: {actions.shape}")
    
    # Get circuit info
    info = policy.get_quantum_circuit_info()
    print(f"\nCircuit Info:")
    print(f"  Qubits: {info['n_qubits']}")
    print(f"  Layers: {info['n_layers']}")
    print(f"  Parameters: {info['n_params']}")
    
    print("\nTest complete!")
