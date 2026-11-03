"""
Variational Quantum Circuit (VQC) for Policy Network
======================================================

This module implements a 3-qubit Variational Quantum Circuit using PennyLane
with JAX backend for the quantum policy network in the Quantum RL Propofol control system.

Architecture:
-------------
1. Classical Encoder: State(13D) → n_qubits features  (state 압축, n_qubits >= action_dim)
2. Variational Layers: Parameterized rotation gates (RY, RZ) with entanglement (CNOT)
3. Measurement: Expectation values of Pauli-Z operators → n_qubits measurements
4. Output Scaling: n_qubits measurements → action_dim via Linear + Sigmoid

Design Rationale (n_qubits > action_dim):
------------------------------------------
Dual drug control의 action_dim=2 (Propofol, Remifentanil)에 대해 n_qubits=3으로 설정.
3번째 큐빗은 약물 상호작용(drug interaction) 채널로 활용되어,
entanglement를 통해 두 약물 간 시너지/길항 관계를 더 풍부하게 학습.
  - Qubit 0 → entangled → Propofol 제어
  - Qubit 1 → entangled → Remifentanil 제어
  - Qubit 2 → 약물 상호작용 보조 (auxiliary interaction qubit)
  - Output Layer: Linear(3 → 2) + Sigmoid → [propofol, remifentanil]

For 3 Qubits:
- Input: 13D state → Classical Encoder → 3 features (n_qubits=3)
- Encoding: Angle embedding RX(feature * π)
- Layers: RY-RZ rotations + CNOT entanglement (circular: 0→1, 1→2, 2→0)
- Output: [expval(Z_0), expval(Z_1), expval(Z_2)] → Linear(3→2) → [propofol_action, remifentanil_action]

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

# Let JAX automatically decide between CPU and GPU
# (Windows JAX often lacks native GPU support without WSL2)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jax")


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
        # Get available device (GPU if available, else CPU)
        target_device = jax.devices()[0]
        
        # Convert to JAX arrays and place on device
        with jax.default_device(target_device):
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
        
        # Get available device
        target_device = jax.devices()[0]
        
        # Execute gradient computation on device
        with jax.default_device(target_device):
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


def _batch_first(outputs):
    """Stack a vmapped QNode's outputs as [batch, n_observables].

    A QNode with several expectation values returns one [batch] array per
    observable. Stacking on axis 1 is correct for every batch size; the
    previous shape-based transpose failed when batch == n_observables.
    """
    if isinstance(outputs, (list, tuple)):
        return jnp.stack(outputs, axis=1)
    return jnp.asarray(outputs)


class JAXCircuitBatchFunction(torch.autograd.Function):
    """
    Custom PyTorch autograd function for batched JAX quantum circuits.
    Optimized for GPU execution with vectorized operations.
    """
    
    @staticmethod
    def forward(ctx, circuit_batch_fn, inputs, weights):
        """Forward pass for a batch of inputs.

        Uses whichever backend JAX actually has. Requesting 'gpu'
        unconditionally, as this did previously, raises on a CPU-only JAX build
        and made the entire batched path unreachable - which is why the policy
        fell back to evaluating the circuit one sample at a time.
        """
        with jax.default_device(jax.devices()[0]):
            inputs_jax = jnp.asarray(inputs.detach().cpu().numpy())
            weights_jax = jnp.asarray(weights.detach().cpu().numpy())
            
            # Execute batched circuit on GPU
            outputs_jax = _batch_first(circuit_batch_fn(inputs_jax, weights_jax))

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
        
        # Execute gradient computation on whichever backend JAX has.
        with jax.default_device(jax.devices()[0]):
            def batch_outputs(inp, w):
                return _batch_first(circuit_batch_fn(inp, w))

            # For multi-output circuits, we need to handle gradients for each output
            def single_sample_loss(input_single, output_grad_single):
                """Compute loss for single sample with gradient weighting."""
                outputs = batch_outputs(input_single[None, ...], weights_jax)[0]
                # Weighted sum by upstream gradient
                return jnp.sum(outputs * output_grad_single)
            
            # Convert grad_output to JAX
            grad_output_jax = jnp.array(grad_output.detach().cpu().numpy())
            
            # Compute gradients for each sample using vmap
            grad_fn = jax.grad(single_sample_loss, argnums=0)
            grad_inputs_jax = vmap(grad_fn)(inputs_jax, grad_output_jax)
            
            # Gradient w.r.t weights (sum over batch)
            def batch_loss_fn(weights_jax):
                outputs = batch_outputs(inputs_jax, weights_jax)
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


# Trainable angles per qubit and layer for each ansatz family (R1-3).
# PennyLane versions of the Qiskit families, applied after the per-layer RX
# re-uploading: RealAmplitudes = RY + reverse-linear CNOTs, EfficientSU2 =
# RY, RZ + reverse-linear CNOTs, TwoLocal = RY + CZ on every pair.
ANSATZ_PARAMS_PER_QUBIT = {
    "strongly_entangling": 3,
    "real_amplitudes": 1,
    "efficient_su2": 2,
    "two_local": 1,
}


class VariationalQuantumCircuit:
    """
    Variational Quantum Circuit for continuous action output.
    
    This class implements a parameterized quantum circuit that takes
    encoded state features and outputs expectation values that
    can be mapped to continuous actions.
    
    Architecture for 3 qubits (n_qubits > action_dim, extra qubit for drug interaction):
    |0⟩ ─ RX(θ_in[0]) ─ RY(θ[0]) ─ RZ(θ[1]) ─●─── RY(θ[6])  ─ RZ(θ[7])  ──●─ M ┐
                                               │                              │    │
    |0⟩ ─ RX(θ_in[1]) ─ RY(θ[2]) ─ RZ(θ[3]) ─⊕─●─ RY(θ[8])  ─ RZ(θ[9])  ─── M ─ Linear(3→2) → [ppf, rftn]
                                                  │                            │
    |0⟩ ─ RX(θ_in[2]) ─ RY(θ[4]) ─ RZ(θ[5]) ────⊕─ RY(θ[10]) ─ RZ(θ[11]) ─⊕─ M ┘
    
    Attributes:
        n_qubits: Number of qubits (3 for this implementation, with auxiliary interaction qubit)
        n_layers: Number of variational layers
        dev: PennyLane quantum device
        weight_shapes: Shape of variational parameters
    """
    
    def __init__(
        self,
        n_qubits: int = 3,
        n_layers: int = 4,
        device: str = "default.qubit",
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        entanglement: bool = True,
        noise_probability: float = 0.0,
        ansatz: str = "strongly_entangling",
    ):
        """
        Initialize the VQC.

        Args:
            n_qubits: Number of qubits (3 for this implementation)
            n_layers: Number of variational layers
            device: PennyLane device name
            shots: Number of shots for measurement (None for analytic)
            seed: Random seed for device
            ansatz: Trainable block family, a key of ANSATZ_PARAMS_PER_QUBIT
        """
        if ansatz not in ANSATZ_PARAMS_PER_QUBIT:
            raise ValueError(f"unknown ansatz {ansatz!r}; "
                             f"choose from {sorted(ANSATZ_PARAMS_PER_QUBIT)}")
        self.ansatz = ansatz
        self.params_per_qubit = ANSATZ_PARAMS_PER_QUBIT[ansatz]
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.entanglement = bool(entanglement)
        self.noise_probability = float(noise_probability)
        if not 0.0 <= self.noise_probability <= 1.0:
            raise ValueError("noise_probability must be in [0, 1]")
        if self.noise_probability > 0.0 and device == "default.qubit":
            device = "default.mixed"
        
        # Create PennyLane device
        device_kwargs = {"wires": n_qubits}
        if shots is not None:
            device_kwargs["shots"] = shots
        if seed is not None and device != "default.mixed":
            device_kwargs["seed"] = seed
        self.dev = qml.device(device, **device_kwargs)
        
        # Calculate parameter shapes (3 per qubit and layer for the original
        # StronglyEntanglingLayers block)
        self.n_params_per_layer = self.params_per_qubit * n_qubits
        self.total_params = self.n_params_per_layer * n_layers

        # Weight shapes for QNode
        self.weight_shapes = {"weights": (n_layers, n_qubits, self.params_per_qubit)}
        
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
        
        diff_method = "backprop" if self.shots is None else "parameter-shift"

        @qml.qnode(self.dev, interface="jax", diff_method=diff_method)
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
                
                self._variational_block(weights, layer)
                if self.noise_probability > 0.0:
                    for i in range(self.n_qubits):
                        qml.DepolarizingChannel(
                            self.noise_probability, wires=i)
            
            # Measurement: Pauli-X, Pauli-Y, Pauli-Z expectation on all qubits
            measurements = []
            for i in range(self.n_qubits):
                measurements.append(qml.expval(qml.PauliX(i)))
                measurements.append(qml.expval(qml.PauliY(i)))
                measurements.append(qml.expval(qml.PauliZ(i)))
            return measurements
        
        # Keep the un-jitted QNode for resource inspection (qml.specs).
        self.qnode = circuit
        # JIT compile for speed
        self.circuit = jit(circuit)
        # Vectorized version: vmap over batch dim (in_axes=(0, None) → batch layer_inputs, shared weights)
        self.circuit_batch = jit(vmap(circuit, in_axes=(0, None)))

    def _variational_block(self, weights, layer: int) -> None:
        """Apply the trainable block of one layer for the chosen ansatz."""
        wires = range(self.n_qubits)
        w = weights[layer]
        if self.ansatz == "strongly_entangling":
            if self.entanglement:
                # weights[layer:layer+1] has shape (1, n_qubits, 3)
                qml.StronglyEntanglingLayers(weights[layer:layer + 1], wires=wires)
            else:
                # Same three trainable angles per qubit/layer, but no
                # two-qubit gates.  This is the prespecified entanglement
                # ablation and therefore preserves the parameter count.
                for i in wires:
                    qml.Rot(*w[i], wires=i)
            return
        for i in wires:
            qml.RY(w[i, 0], wires=i)
            if self.ansatz == "efficient_su2":
                qml.RZ(w[i, 1], wires=i)
        if self.ansatz == "two_local":
            for i in wires:
                for j in range(i + 1, self.n_qubits):
                    qml.CZ(wires=[i, j])
        else:  # real_amplitudes, efficient_su2: reverse-linear CNOT chain
            for i in reversed(range(self.n_qubits - 1)):
                qml.CNOT(wires=[i, i + 1])

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
            Initialized weight tensor [n_layers, n_qubits, params_per_qubit]
        """
        if seed is not None:
            torch.manual_seed(seed)
        weights = torch.rand(
            self.n_layers, self.n_qubits, self.params_per_qubit) * 2 * np.pi
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
    
    Design Rationale (Data Re-Uploading with n_qubits > action_dim):
        기존 단일 encoder MLP(state_dim → 2D 병목)를 제거하고,
        VQC의 각 레이어마다 독립적인 linear projection을 사용.
        → 레이어 l은 projection_l(embedding)로 임베딩의 다른 측면을 학습
        → 3 큐빗으로 n_layers 번 다른 관점에서 임베딩을 처리
        → 3번째 큐빗이 약물 상호작용 정보를 entanglement로 전달
        → Linear(3→2) 출력 레이어가 3개 측정값을 2개 액션으로 매핑
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
        n_qubits: int = 3,
        n_layers: int = 4,
        action_scale: float = 1.0,
        action_dim: int = 2,
        device_name: str = "default.qubit",
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        use_classical_skip: bool = True,
        entanglement: bool = True,
        noise_probability: float = 0.0,
        action_prior: Optional[List[float]] = None,
        residual_init_scale: float = 0.02,
        # Backward-compatible: encoder_hidden is accepted but ignored
        encoder_hidden: Optional[List[int]] = None,
        ansatz: str = "strongly_entangling",
    ):
        """
        Initialize the Data Re-Uploading Quantum Policy.
        
        Args:
            state_dim: Transformer embedding dimension (input to QuantumPolicy)
            n_qubits: Number of qubits (3 for richer entanglement, > action_dim)
            n_layers: Number of VQC layers (= number of re-uploading steps)
            action_scale: Scale for output action
            action_dim: Dimension of action output (2 for dual drug)
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
            device=device_name,
            shots=shots,
            seed=seed,
            entanglement=entanglement,
            noise_probability=noise_probability,
            ansatz=ansatz,
        )

        # ── Learnable Data Scaling ───────────────────────────────────────────
        # RX 각도 인코딩 전에 데이터를 스케일링하여 블로흐 구면 상의 매핑을 최적화
        self.input_scaling = nn.Parameter(torch.ones(n_layers, n_qubits))
        
        # ── Variational Parameters (훈련 가능한 VQC 가중치) ──────────────────
        if seed is not None:
            torch.manual_seed(seed)
        self.weights = nn.Parameter(
            torch.rand(n_layers, n_qubits, self.vqc.params_per_qubit) * 2 * np.pi
        )

        # ── Output Scaling (Residual Architecture) ─────────────────────────
        self.vqc_output_dim = 3 * n_qubits
        
        # 잔차 연결 (Classical Skip Connection)
        # Quantum-NoSkip ablation: when disabled, the action comes from the
        # circuit readout alone, which isolates the PQC's contribution.
        self.use_classical_skip = use_classical_skip
        if use_classical_skip:
            self.classical_skip = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )
        else:
            self.classical_skip = None
        
        # Quantum Residual mapping
        self.quantum_residual = nn.Linear(self.vqc_output_dim, action_dim)
        
        self.sigmoid = nn.Sigmoid()

        # Optional clinically informed initialization.  The prior is expressed
        # in normalized action space and affects initialization only; every
        # parameter remains trainable.  It prevents the remifentanil output
        # from starting near 50% of its ceiling (1.0 ug/kg/min).
        if action_prior is not None:
            prior = torch.as_tensor(action_prior, dtype=torch.float32)
            if prior.numel() != action_dim or torch.any((prior <= 0) | (prior >= 1)):
                raise ValueError("action_prior must contain one value in (0, 1) per action")
            prior_logits = torch.logit(prior)
            nn.init.normal_(self.quantum_residual.weight, mean=0.0,
                            std=residual_init_scale)
            if self.classical_skip is None:
                with torch.no_grad():
                    self.quantum_residual.bias.copy_(prior_logits)
            else:
                nn.init.zeros_(self.quantum_residual.bias)
                nn.init.normal_(self.classical_skip[-1].weight, mean=0.0,
                                std=residual_init_scale)
                with torch.no_grad():
                    self.classical_skip[-1].bias.copy_(prior_logits)
    
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
        
        # Learnable Data Scaling 적용
        layer_features = layer_features * self.input_scaling.unsqueeze(0)
        
        # Data Re-Uploading VQC 실행
        # Run the vmapped circuit once for the complete batch.  The previous
        # Python loop bypassed ``circuit_batch`` and dominated BC runtime.
        vqc_output = self.vqc.forward(
            layer_features, self.weights
        )  # [batch, 3 * n_qubits]

        # Pauli X, Y, Z are [-1, 1], we can feed them directly into Linear
        combined = self.quantum_residual(vqc_output)  # [batch, action_dim]

        # 잔차 연결 (Skip Connection), disabled for the NoSkip ablation
        if self.classical_skip is not None:
            combined = combined + self.classical_skip(state)
        action = self.sigmoid(combined) * self.action_scale
        
        if squeeze_output:
            action = action.squeeze(0)
        
        return action

    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """Return an action for a NumPy state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            return self.forward(state_tensor).numpy()

    def get_quantum_circuit_info(self) -> Dict:
        """Return executable-circuit and parameter metadata."""
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


class QuantumDisabledPolicy(nn.Module):
    """Parameter-matched classical control for the hybrid quantum policy.

    It preserves the per-layer projections, input scaling, 36-parameter
    nonlinear core (for the default 3-qubit/4-layer setting), readout and skip
    branch.  Only the quantum circuit is replaced by deterministic sine,
    cosine and tanh features.  Consequently a result against this arm tests
    the PQC rather than actor size, clinical initialization or skip capacity.
    """

    def __init__(
        self,
        state_dim: int = 32,
        n_qubits: int = 3,
        n_layers: int = 4,
        action_scale: float = 1.0,
        action_dim: int = 2,
        seed: Optional[int] = None,
        use_classical_skip: bool = True,
        action_prior: Optional[List[float]] = None,
        residual_init_scale: float = 0.02,
        **_: object,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.action_scale = action_scale
        self.action_dim = action_dim
        self.layer_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, state_dim), nn.GELU(),
                nn.Linear(state_dim, n_qubits), nn.Tanh())
            for _ in range(n_layers)
        ])
        self.input_scaling = nn.Parameter(torch.ones(n_layers, n_qubits))
        if seed is not None:
            torch.manual_seed(seed)
        self.surrogate_weights = nn.Parameter(
            torch.rand(n_layers, n_qubits, 3) * 2 * np.pi)
        self.use_classical_skip = use_classical_skip
        self.classical_skip = (nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
        ) if use_classical_skip else None)
        self.quantum_residual = nn.Linear(3 * n_qubits, action_dim)
        self.sigmoid = nn.Sigmoid()

        if action_prior is not None:
            prior = torch.as_tensor(action_prior, dtype=torch.float32)
            if prior.numel() != action_dim or torch.any((prior <= 0) | (prior >= 1)):
                raise ValueError(
                    "action_prior must contain one value in (0, 1) per action")
            prior_logits = torch.logit(prior)
            nn.init.normal_(self.quantum_residual.weight, mean=0.0,
                            std=residual_init_scale)
            if self.classical_skip is None:
                with torch.no_grad():
                    self.quantum_residual.bias.copy_(prior_logits)
            else:
                nn.init.zeros_(self.quantum_residual.bias)
                nn.init.normal_(self.classical_skip[-1].weight, mean=0.0,
                                std=residual_init_scale)
                with torch.no_grad():
                    self.classical_skip[-1].bias.copy_(prior_logits)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        squeeze_output = state.dim() == 1
        if squeeze_output:
            state = state.unsqueeze(0)
        features = torch.stack([proj(state) for proj in self.layer_projections], dim=1)
        features = features * self.input_scaling.unsqueeze(0)
        weights = self.surrogate_weights.unsqueeze(0)
        classical_features = torch.cat([
            (torch.sin(np.pi * features) * weights[..., 0]).mean(dim=1),
            (torch.cos(np.pi * features) * weights[..., 1]).mean(dim=1),
            (torch.tanh(features) * weights[..., 2]).mean(dim=1),
        ], dim=-1)
        combined = self.quantum_residual(classical_features)
        if self.classical_skip is not None:
            combined = combined + self.classical_skip(state)
        action = self.sigmoid(combined) * self.action_scale
        return action.squeeze(0) if squeeze_output else action
    
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
        """Return metadata for the parameter-matched classical surrogate."""
        n_proj_params = sum(
            sum(p.numel() for p in proj.parameters())
            for proj in self.layer_projections
        )
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'n_vqc_params': 0,
            'n_surrogate_params': self.surrogate_weights.numel(),
            'n_projection_params': n_proj_params,
            'n_total_params': sum(p.numel() for p in self.parameters()),
            'weight_shape': tuple(self.surrogate_weights.shape),
            'architecture': 'Classical trigonometric surrogate + Per-Layer Projection',
            'circuit_diagram': None,
        }


class FourierSurrogatePolicy(nn.Module):
    """Classical Fourier-feature control with the circuit's frequency set.

    Reviewer 2 (comment 5) noted that each of the n_layers * n_qubits projected
    angles enters the re-uploading circuit once, so the circuit output is a
    trigonometric polynomial with frequencies in {-1, 0, 1} per angle, and
    asked for a classical surrogate with a matched frequency set and parameter
    budget. This policy keeps the per-layer projections, input scaling and
    skip branch, takes D fixed frequency vectors from {-1, 0, 1}^(L*n)
    (nonzero, distinct up to sign, drawn once with a fixed seed so every
    training seed uses the same set), and maps cos/sin of their phases to the
    action logits with a linear readout. D is set so that the readout
    (4D + 2 parameters for two actions) matches the circuit's trainable angles
    plus its 9 -> 2 readout (36 + 20 = 56).
    """

    FREQUENCY_SEED = 20261019

    def __init__(
        self,
        state_dim: int = 32,
        n_qubits: int = 3,
        n_layers: int = 4,
        action_scale: float = 1.0,
        action_dim: int = 2,
        seed: Optional[int] = None,
        use_classical_skip: bool = True,
        action_prior: Optional[List[float]] = None,
        residual_init_scale: float = 0.02,
        n_frequencies: Optional[int] = None,
        **_: object,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.action_scale = action_scale
        self.action_dim = action_dim
        self.layer_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, state_dim), nn.GELU(),
                nn.Linear(state_dim, n_qubits), nn.Tanh())
            for _ in range(n_layers)
        ])
        self.input_scaling = nn.Parameter(torch.ones(n_layers, n_qubits))

        n_angles = n_layers * n_qubits
        if n_frequencies is None:
            circuit_budget = n_angles * 3 + (3 * n_qubits + 1) * action_dim
            n_frequencies = max(1, round((circuit_budget - action_dim) / (2 * action_dim)))
        rng = np.random.default_rng(self.FREQUENCY_SEED)
        chosen: List[np.ndarray] = []
        seen = set()
        while len(chosen) < n_frequencies:
            omega = rng.integers(-1, 2, size=n_angles)
            key, mirror = tuple(omega), tuple(-omega)
            if not omega.any() or key in seen or mirror in seen:
                continue
            seen.add(key)
            chosen.append(omega)
        self.register_buffer(
            "frequencies", torch.tensor(np.stack(chosen), dtype=torch.float32))

        if seed is not None:
            torch.manual_seed(seed)
        self.use_classical_skip = use_classical_skip
        self.classical_skip = (nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
        ) if use_classical_skip else None)
        # Named like the hybrid policy's readout so action-prior handling and
        # optimizer grouping treat it the same way.
        self.quantum_residual = nn.Linear(2 * n_frequencies, action_dim)
        self.sigmoid = nn.Sigmoid()

        if action_prior is not None:
            prior = torch.as_tensor(action_prior, dtype=torch.float32)
            if prior.numel() != action_dim or torch.any((prior <= 0) | (prior >= 1)):
                raise ValueError(
                    "action_prior must contain one value in (0, 1) per action")
            prior_logits = torch.logit(prior)
            nn.init.normal_(self.quantum_residual.weight, mean=0.0,
                            std=residual_init_scale)
            if self.classical_skip is None:
                with torch.no_grad():
                    self.quantum_residual.bias.copy_(prior_logits)
            else:
                nn.init.zeros_(self.quantum_residual.bias)
                nn.init.normal_(self.classical_skip[-1].weight, mean=0.0,
                                std=residual_init_scale)
                with torch.no_grad():
                    self.classical_skip[-1].bias.copy_(prior_logits)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        squeeze_output = state.dim() == 1
        if squeeze_output:
            state = state.unsqueeze(0)
        features = torch.stack([proj(state) for proj in self.layer_projections], dim=1)
        angles = np.pi * (features * self.input_scaling.unsqueeze(0)).flatten(1)
        phases = angles @ self.frequencies.T
        combined = self.quantum_residual(
            torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1))
        if self.classical_skip is not None:
            combined = combined + self.classical_skip(state)
        action = self.sigmoid(combined) * self.action_scale
        return action.squeeze(0) if squeeze_output else action

    def get_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            return self.forward(torch.FloatTensor(state)).numpy()


class QuantumPolicySimple(nn.Module):
    """
    Simplified Quantum Policy with Data Re-Uploading (no classical encoder).
    
    Useful for testing and ablation studies.
    선택된 n_qubits개의 state feature를 각 레이어마다 직접 re-upload.
    """
    
    def __init__(
        self,
        n_qubits: int = 3,
        n_layers: int = 4,
        action_scale: float = 1.0,
        action_dim: int = 1,
        feature_indices: List[int] = [0, 1, 2],
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
        
        self.input_scaling = nn.Parameter(torch.ones(n_layers, n_qubits))
        
        if seed is not None:
            torch.manual_seed(seed)
        self.weights = nn.Parameter(
            torch.rand(n_layers, n_qubits, 3) * 2 * np.pi
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
        
        # Learnable Data Scaling 적용
        layer_inputs = layer_inputs * self.input_scaling.unsqueeze(0)
        
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
    vqc = VariationalQuantumCircuit(n_qubits=3, n_layers=4)
    print(f"VQC created with {vqc.total_params} variational parameters")
    
    # layer_inputs: [n_layers, n_qubits]
    layer_inputs = torch.rand(4, 3) * 2 - 1  # uniform in [-1, 1]
    weights = vqc.get_initial_weights()
    output = vqc.forward(layer_inputs, weights)
    print(f"VQC output (single): {output}")
    
    # 배치 테스트: [batch, n_layers, n_qubits]
    batch_layer_inputs = torch.rand(8, 4, 3) * 2 - 1
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
        n_qubits=3,
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
