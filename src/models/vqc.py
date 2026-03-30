"""
Variational Quantum Circuit (VQC) for Policy Network
======================================================

This module implements a 2-qubit Variational Quantum Circuit using PennyLane
with JAX backend for the quantum policy network in the Quantum RL Propofol control system.

Architecture:
-------------
1. Classical Encoder: State(13D) → n_qubits features  (state 압축, n_qubits = action_dim)
2. Variational Layers: Parameterized rotation gates (RY, RZ) with entanglement (CNOT)
3. Measurement: Expectation values of Pauli-Z operators → n_qubits measurements
4. Output Scaling: n_qubits measurements → action_dim via Linear + Sigmoid

Design Rationale (n_qubits == action_dim):
------------------------------------------
Dual drug control의 action_dim=2 (Propofol, Remifentanil)에 맞춰 n_qubits=2로 설정.
각 큐빗이 하나의 약물 출력에 대응하는 자연스러운 1:1 매핑 구조.
  - Qubit 0 → 측정값 → Propofol 제어량
  - Qubit 1 → 측정값 → Remifentanil 제어량

For 2 Qubits:
- Input: 13D state → Classical Encoder → 2 features (n_qubits=2)
- Encoding: Angle embedding RX(feature * π)
- Layers: RY-RZ rotations + CNOT entanglement (circular)
- Output: [expval(Z_0), expval(Z_1)] → Linear → [propofol_action, remifentanil_action]

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
            # For multi-output circuits, compute gradients for each output separately
            # and combine with chain rule using grad_output
            def scalar_loss_fn(inputs_jax, weights_jax, output_idx):
                """Extract single output for gradient computation."""
                outputs = circuit_fn(inputs_jax, weights_jax)
                # Convert list to array if needed
                if isinstance(outputs, list):
                    outputs = jnp.array(outputs)
                return outputs[output_idx]
            
            # Determine number of outputs
            test_output = circuit_fn(inputs_jax, weights_jax)
            if isinstance(test_output, list):
                n_outputs = len(test_output)
            else:
                n_outputs = test_output.shape[0] if test_output.ndim > 0 else 1
            
            # Accumulate gradients weighted by grad_output
            grad_inputs_jax = jnp.zeros_like(inputs_jax)
            grad_weights_jax = jnp.zeros_like(weights_jax)
            
            grad_output_np = grad_output.detach().cpu().numpy()
            if grad_output_np.ndim == 0:
                grad_output_weights = jnp.array([grad_output_np] * n_outputs)
            else:
                grad_output_weights = jnp.array(grad_output_np)
            
            for i in range(n_outputs):
                # Compute gradient for this output
                grad_fn = jax.grad(lambda inp, w: scalar_loss_fn(inp, w, i), argnums=(0, 1))
                g_inp, g_w = grad_fn(inputs_jax, weights_jax)
                
                # Weight by upstream gradient
                grad_inputs_jax = grad_inputs_jax + g_inp * grad_output_weights[i]
                grad_weights_jax = grad_weights_jax + g_w * grad_output_weights[i]
            
            # Convert gradients back to numpy (make writable copies)
            grad_inputs_np = np.asarray(grad_inputs_jax).copy()
            grad_weights_np = np.asarray(grad_weights_jax).copy()
        
        # Convert to PyTorch
        grad_inputs = torch.from_numpy(grad_inputs_np).float()
        grad_weights = torch.from_numpy(grad_weights_np).float()
        
        if inputs.is_cuda:
            grad_inputs = grad_inputs.to(inputs.device)
            grad_weights = grad_weights.to(inputs.device)
        
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
        
        batch_size = inputs.shape[0]
        
        # Execute gradient computation on GPU
        with jax.default_device(jax.devices('gpu')[0]):
            # For multi-output circuits, we need to handle gradients for each output
            def single_sample_loss(input_single, output_grad_single):
                """Compute loss for single sample with gradient weighting."""
                outputs = circuit_batch_fn(input_single.reshape(1, -1), weights_jax)[0]
                # Convert list to array if needed
                if isinstance(outputs, list):
                    outputs = jnp.array(outputs)
                # Weighted sum by upstream gradient
                return jnp.sum(outputs * output_grad_single)
            
            # Convert grad_output to JAX
            grad_output_jax = jnp.array(grad_output.detach().cpu().numpy())
            
            # Compute gradients for each sample using vmap
            grad_fn = jax.grad(single_sample_loss, argnums=0)
            grad_inputs_jax = vmap(grad_fn)(inputs_jax, grad_output_jax)
            
            # Gradient w.r.t weights (sum over batch)
            def batch_loss_fn(weights_jax):
                outputs = circuit_batch_fn(inputs_jax, weights_jax)
                # Convert list outputs to array
                if isinstance(outputs, list):
                    outputs = jnp.stack(outputs)
                elif outputs.ndim == 1:
                    # Single output per sample, expand dims
                    outputs = outputs[:, None]
                # Weight by upstream gradient and sum
                return jnp.sum(outputs * grad_output_jax)
            
            grad_weights_jax = jax.grad(batch_loss_fn)(weights_jax)
            
            # Convert to numpy (make writable copies)
            grad_inputs_np = np.asarray(grad_inputs_jax).copy()
            grad_weights_np = np.asarray(grad_weights_jax).copy()
        
        # Convert to PyTorch
        grad_inputs = torch.from_numpy(grad_inputs_np).float()
        grad_weights = torch.from_numpy(grad_weights_np).float()
        
        if inputs.is_cuda:
            grad_inputs = grad_inputs.to(inputs.device)
            grad_weights = grad_weights.to(inputs.device)
        
        return None, grad_inputs, grad_weights


class VariationalQuantumCircuit:
    """
    Variational Quantum Circuit for continuous action output.
    
    This class implements a parameterized quantum circuit that takes
    encoded state features and outputs expectation values that
    can be mapped to continuous actions.
    
    Architecture for 2 qubits (n_qubits == action_dim):
    |0⟩ ─ RX(θ_in[0]) ─ RY(θ[0]) ─ RZ(θ[1]) ─●─ RY(θ[4]) ─ RZ(θ[5]) ─ M → propofol
                                              │
    |0⟩ ─ RX(θ_in[1]) ─ RY(θ[2]) ─ RZ(θ[3]) ─⊕─ RY(θ[6]) ─ RZ(θ[7]) ─ M → remifentanil
    
    Attributes:
        n_qubits: Number of qubits (= action_dim, 1 qubit per drug)
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
        """
        Create the PennyLane QNode with Data Re-Uploading strategy.
        
        Unlike standard VQC (encode once → variational layers),
        Data Re-Uploading encodes the input AT EVERY LAYER.
        This allows the VQC to process different projections of the
        Transformer embedding at each depth, dramatically increasing
        expressibility without adding more qubits.
        
        Reference: Perez-Salinas et al., "Data re-uploading for a universal
        quantum classifier", Quantum 4, 226 (2020)
        """
        
        @qml.qnode(self.dev, interface="jax", diff_method="backprop")
        def circuit(layer_inputs, weights):
            """
            Data Re-Uploading VQC: encodes embedding at EVERY layer.
            
            Args:
                layer_inputs: Per-layer projected features (shape: [n_layers, n_qubits])
                              layer_inputs[l] = projection_l(transformer_embedding)
                              Each layer receives a DIFFERENT linear projection
                              of the same Transformer embedding.
                weights: Variational parameters (shape: [n_layers, n_qubits, 2])
            
            Returns:
                Expectation values of Pauli-Z on all qubits
            """
            for layer in range(self.n_layers):
                # === Data Re-Uploading ===
                # Each layer encodes its own specific view of the embedding
                # via RX(projection_l[i] * π) angle encoding
                for i in range(self.n_qubits):
                    qml.RX(layer_inputs[layer, i] * jnp.pi, wires=i)
                
                # Variational rotations (learned parameters)
                for qubit in range(self.n_qubits):
                    qml.RY(weights[layer, qubit, 0], wires=qubit)
                    qml.RZ(weights[layer, qubit, 1], wires=qubit)
                
                # Entangling layer (circular CNOT)
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                if self.n_qubits > 1:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Measurement: Pauli-Z expectation on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        # JIT compile for speed
        self.circuit = jit(circuit)
        # Vectorized version: vmap over batch dim (in_axes=(0, None) → batch layer_inputs, shared weights)
        self.circuit_batch = jit(vmap(circuit, in_axes=(0, None)))
    
    def forward(
        self,
        layer_inputs: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the Data Re-Uploading VQC.
        
        Args:
            layer_inputs: Per-layer projected features
                          shape: [n_layers, n_qubits]       for single sample
                                 [batch, n_layers, n_qubits] for batch
            weights: Variational parameters (shape: [n_layers, n_qubits, 2])
        
        Returns:
            PauliZ expectation values
            shape: [n_qubits] for single, [batch, n_qubits] for batch
        """
        if layer_inputs.dim() == 2:  # [n_layers, n_qubits] - single sample
            return JAXCircuitFunction.apply(self.circuit, layer_inputs, weights)
        else:  # [batch, n_layers, n_qubits]
            return JAXCircuitBatchFunction.apply(self.circuit_batch, layer_inputs, weights)
    
    def get_initial_weights(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Get randomly initialized variational weights.
        
        Returns:
            Initialized weight tensor [n_layers, n_qubits, 2]
        """
        if seed is not None:
            torch.manual_seed(seed)
        weights = torch.rand(self.n_layers, self.n_qubits, 2) * 2 * np.pi
        weights.requires_grad = True
        return weights
    
    def draw(self) -> str:
        """
        Draw the quantum circuit (Data Re-Uploading structure).
        
        Returns:
            String representation of the circuit
        """
        # layer_inputs: [n_layers, n_qubits] for drawing
        layer_inputs = np.zeros((self.n_layers, self.n_qubits))
        weights = self.get_initial_weights().detach().numpy()
        
        layer_inputs_jax = jnp.array(layer_inputs, dtype=jnp.float32)
        weights_jax = jnp.array(weights, dtype=jnp.float32)
        
        drawer = qml.draw(self.circuit)
        return drawer(layer_inputs_jax, weights_jax)


class QuantumPolicy(nn.Module):
    """
    Hybrid Quantum-Classical Policy Network with Data Re-Uploading.
    
    Architecture:
    ─────────────────────────────────────────────────────────────
    State (Transformer embedding, state_dim D)
      │
      ├─ layer_proj_0(state_dim → n_qubits) ─▶ RX encoding ─▶ VQC Layer 0
      ├─ layer_proj_1(state_dim → n_qubits) ─▶ RX encoding ─▶ VQC Layer 1
      ├─ layer_proj_2(state_dim → n_qubits) ─▶ RX encoding ─▶ VQC Layer 2
      └─ layer_proj_3(state_dim → n_qubits) ─▶ RX encoding ─▶ VQC Layer 3
                                                                     │
                                                    PauliZ 측정 [n_qubits 개]
                                                                     │
                                              Linear(n_qubits → action_dim) + Sigmoid
                                                                     │
                                                        Action [propofol, remifentanil]
    ─────────────────────────────────────────────────────────────
    
    Design Rationale (Data Re-Uploading):
        기존 단일 encoder MLP(state_dim → 2D 병목)를 제거하고,
        VQC의 각 레이어마다 독립적인 linear projection을 사용.
        → 레이어 l은 projection_l(embedding)로 임베딩의 다른 측면을 학습
        → 2 큐빗으로도 n_layers 번 다른 관점에서 임베딩을 처리 가능
        → Universal Approximator 성질 유지 (Perez-Salinas et al., 2020)
    
    Attributes:
        layer_projections: n_layers개의 독립 projection (state_dim → n_qubits)
        vqc: Variational quantum circuit (Data Re-Uploading 적용)
        weights: Variational parameters (learned)
        output_scale: Final action mapping (n_qubits → action_dim)
    """
    
    def __init__(
        self,
        state_dim: int = 32,      # Transformer encoder output dim
        n_qubits: int = 2,
        n_layers: int = 4,
        action_scale: float = 1.0,
        action_dim: int = 2,
        device_name: str = "default.qubit",
        seed: Optional[int] = None,
        # Backward-compatible: encoder_hidden is accepted but ignored
        encoder_hidden: Optional[List[int]] = None
    ):
        """
        Initialize the Data Re-Uploading Quantum Policy.
        
        Args:
            state_dim: Transformer embedding dimension (input to QuantumPolicy)
            n_qubits: Number of qubits (= action_dim for 1:1 drug mapping)
            n_layers: Number of VQC layers (= number of re-uploading steps)
            action_scale: Scale for output action
            action_dim: Dimension of action output
            device_name: PennyLane device name
            seed: Random seed
            encoder_hidden: Deprecated, kept for backward compatibility
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.action_scale = action_scale
        self.action_dim = action_dim
        
        # ── Per-Layer Projection (Data Re-Uploading의 핵심) ──────────────────
        # 기존 단일 bottleneck encoder(state_dim → 2D)를 n_layers개의 독립
        # projection으로 대체. 각 레이어가 임베딩의 다른 측면을 학습.
        # 구조: Linear(state_dim → state_dim) + GELU + Linear(state_dim → n_qubits) + Tanh
        # Tanh: RX 각도 인코딩을 위해 출력을 [-1, 1]로 제한
        self.layer_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, state_dim),
                nn.GELU(),
                nn.Linear(state_dim, n_qubits),
                nn.Tanh()   # 출력 ∈ [-1, 1] → RX(x * π) 각도 인코딩
            )
            for _ in range(n_layers)
        ])
        
        # ── Variational Quantum Circuit ───────────────────────────────────────
        self.vqc = VariationalQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device=device_name
        )
        
        # ── Variational Parameters (훈련 가능한 VQC 가중치) ──────────────────
        if seed is not None:
            torch.manual_seed(seed)
        self.weights = nn.Parameter(
            torch.rand(n_layers, n_qubits, 2) * 2 * np.pi
        )
        
        # ── Output Scaling ────────────────────────────────────────────────────
        # PauliZ 측정값 [n_qubits] → action [action_dim]
        # Qubit 0 → Propofol, Qubit 1 → Remifentanil
        output_layer = nn.Linear(n_qubits, self.action_dim)
        nn.init.constant_(output_layer.bias, 0.0)  # sigmoid(0) = 0.5 시작점
        self.output_scale = nn.Sequential(
            output_layer,
            nn.Sigmoid()  # Output ∈ [0, 1]
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Data Re-Uploading.
        
        Args:
            state: Transformer embedding [batch_size, state_dim] or [state_dim]
        
        Returns:
            Action tensor [batch_size, action_dim] or [action_dim]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = state.shape[0]
        
        # 각 레이어별 projection 계산: [batch, n_layers, n_qubits]
        # layer_proj_l(state): 레이어 l이 바라보는 임베딩의 특정 2D 방향
        layer_features = torch.stack(
            [proj(state) for proj in self.layer_projections],
            dim=1
        )  # [batch, n_layers, n_qubits]
        
        # Data Re-Uploading VQC 실행
        vqc_outputs = []
        for i in range(batch_size):
            # layer_features[i]: [n_layers, n_qubits]
            output = self.vqc.forward(layer_features[i], self.weights)
            vqc_outputs.append(output)
        
        vqc_output = torch.stack(vqc_outputs)  # [batch, n_qubits]
        
        # PauliZ 측정값 [-1, 1] → [0, 1] 정규화
        normalized = ((vqc_output + 1) / 2).float()  # [batch, n_qubits]
        
        # 최종 액션 출력: [batch, n_qubits] → [batch, action_dim]
        action = self.output_scale(normalized) * self.action_scale
        
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
            state: State array (Transformer embedding)
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
        Get information about the Data Re-Uploading quantum circuit.
        
        Returns:
            Dictionary with circuit and parameter information
        """
        n_proj_params = sum(
            sum(p.numel() for p in proj.parameters())
            for proj in self.layer_projections
        )
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'n_vqc_params': self.vqc.total_params,
            'n_projection_params': n_proj_params,
            'n_total_params': n_proj_params + self.vqc.total_params,
            'weight_shape': tuple(self.weights.shape),
            'architecture': 'Data Re-Uploading + Per-Layer Projection',
            'circuit_diagram': self.vqc.draw()
        }


class QuantumPolicySimple(nn.Module):
    """
    Simplified Quantum Policy with Data Re-Uploading (no classical encoder).
    
    Useful for testing and ablation studies.
    선택된 n_qubits개의 state feature를 각 레이어마다 직접 re-upload.
    """
    
    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 4,
        action_scale: float = 1.0,
        action_dim: int = 1,
        feature_indices: List[int] = [0, 1],
        device_name: str = "default.qubit",
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.action_scale = action_scale
        self.action_dim = action_dim
        self.feature_indices = feature_indices
        
        self.vqc = VariationalQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device=device_name
        )
        
        if seed is not None:
            torch.manual_seed(seed)
        self.weights = nn.Parameter(
            torch.rand(n_layers, n_qubits, 2) * 2 * np.pi
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Data Re-Uploading (same features repeated each layer).
        
        Args:
            state: State tensor
        
        Returns:
            Action tensor
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = state.shape[0]
        
        # 선택된 feature 추출 및 정규화
        features = torch.tanh(state[:, self.feature_indices])  # [batch, n_qubits]
        
        # Data Re-Uploading: 동일한 features를 n_layers번 반복 encode
        # layer_inputs: [batch, n_layers, n_qubits]
        layer_inputs = features.unsqueeze(1).expand(-1, self.n_layers, -1)
        
        vqc_outputs = []
        for i in range(batch_size):
            output = self.vqc.forward(layer_inputs[i], self.weights)
            vqc_outputs.append(output)
        
        vqc_output = torch.stack(vqc_outputs)  # [batch, n_qubits]
        
        if self.action_dim == 1:
            action = ((vqc_output.mean(dim=1, keepdim=True) + 1) / 2) * self.action_scale
        else:
            action = ((vqc_output[:, :self.action_dim] + 1) / 2) * self.action_scale
        
        if squeeze_output:
            action = action.squeeze(0)
        
        return action


if __name__ == "__main__":
    print("Testing Data Re-Uploading VQC...")
    
    # ── VQC 단독 테스트 ──────────────────────────────────────────────
    vqc = VariationalQuantumCircuit(n_qubits=2, n_layers=4)
    print(f"VQC created with {vqc.total_params} variational parameters")
    
    # layer_inputs: [n_layers, n_qubits]
    layer_inputs = torch.rand(4, 2) * 2 - 1  # uniform in [-1, 1]
    weights = vqc.get_initial_weights()
    output = vqc.forward(layer_inputs, weights)
    print(f"VQC output (single): {output}")
    
    # 배치 테스트: [batch, n_layers, n_qubits]
    batch_layer_inputs = torch.rand(8, 4, 2) * 2 - 1
    batch_output = vqc.forward(batch_layer_inputs, weights)
    print(f"VQC output (batch): {batch_output.shape}")
    
    print("\nQuantum Circuit (Data Re-Uploading):")
    print(vqc.draw())
    
    # ── QuantumPolicy 테스트 ─────────────────────────────────────────
    print("\n" + "="*50)
    print("Testing QuantumPolicy (Data Re-Uploading + Per-Layer Projection)...")
    
    # state_dim=32: Transformer encoder 출력 차원 (예시)
    policy = QuantumPolicy(
        state_dim=32,
        n_qubits=2,
        n_layers=4,
        action_dim=2,
        seed=42
    )
    
    # 단일 샘플 테스트
    state = torch.randn(32)
    action = policy.forward(state)
    print(f"Action (single): {action.detach().numpy()}")
    
    # 배치 테스트
    states = torch.randn(8, 32)
    actions = policy.forward(states)
    print(f"Actions (batch) shape: {actions.shape}")
    
    # 회로 정보
    info = policy.get_quantum_circuit_info()
    print(f"\nCircuit Info:")
    print(f"  Architecture: {info['architecture']}")
    print(f"  Qubits: {info['n_qubits']}")
    print(f"  Layers: {info['n_layers']}")
    print(f"  VQC params: {info['n_vqc_params']}")
    print(f"  Projection params: {info['n_projection_params']}")
    print(f"  Total params: {info['n_total_params']}")
    
    print("\nTest complete!")
