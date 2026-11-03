"""Hybrid quantum twin critic (Reviewer 1, comment 1).

Reviewer 1 argued that a quantum circuit belongs in the critic rather than in
the actor. Each Q head here is the classical critic used by every other arm
plus the linear readout of the same data re-uploading circuit as the hybrid
actor, applied to per-layer projections of the (state, action) pair:

    Q(s, a) = Q_classical(s, a) + w^T VQC(proj(s, a)) + b.

The readout starts small, so training begins from the classical critic and
any difference between arms comes from the quantum branch.
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.networks import CriticNetwork
from models.vqc import VariationalQuantumCircuit


class HybridQuantumCritic(nn.Module):
    """One Q head: classical critic plus a VQC residual."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Optional[List[int]] = None,
        n_qubits: int = 3,
        n_layers: int = 4,
        seed: Optional[int] = None,
        residual_init_scale: float = 0.02,
    ):
        super().__init__()
        input_dim = state_dim + action_dim
        self.classical = CriticNetwork(
            state_dim=state_dim, action_dim=action_dim,
            hidden_dims=hidden_dims or [256, 256])
        self.layer_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim), nn.GELU(),
                nn.Linear(input_dim, n_qubits), nn.Tanh())
            for _ in range(n_layers)
        ])
        self.input_scaling = nn.Parameter(torch.ones(n_layers, n_qubits))
        self.vqc = VariationalQuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
        if seed is not None:
            torch.manual_seed(seed)
        self.weights = nn.Parameter(torch.rand(n_layers, n_qubits, 3) * 2 * np.pi)
        self.quantum_readout = nn.Linear(3 * n_qubits, 1)
        nn.init.normal_(self.quantum_readout.weight, mean=0.0, std=residual_init_scale)
        nn.init.zeros_(self.quantum_readout.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        features = torch.stack([proj(x) for proj in self.layer_projections], dim=1)
        features = features * self.input_scaling.unsqueeze(0)
        quantum = self.quantum_readout(self.vqc.forward(features, self.weights))
        return self.classical(state, action) + quantum


class HybridQuantumTwinCritic(nn.Module):
    """Drop-in replacement for TwinCriticNetwork with hybrid heads."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Optional[List[int]] = None,
        n_qubits: int = 3,
        n_layers: int = 4,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.critic1 = HybridQuantumCritic(
            state_dim, action_dim, hidden_dims, n_qubits, n_layers,
            seed=None if seed is None else seed)
        self.critic2 = HybridQuantumCritic(
            state_dim, action_dim, hidden_dims, n_qubits, n_layers,
            seed=None if seed is None else seed + 1)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.critic1(state, action), self.critic2(state, action)

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.critic1(state, action)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
