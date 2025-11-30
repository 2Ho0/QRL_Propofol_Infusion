"""
Classical Neural Networks for Hybrid Quantum-Classical RL
============================================================

This module implements the classical neural network components
for the hybrid Quantum RL system:

1. CriticNetwork: Value/Q-function network for actor-critic methods
2. StateEncoder: Preprocesses state for quantum circuit encoding
3. ActionValueNetwork: Q(s,a) network for DDPG-style algorithms

These networks work alongside the VQC policy to form a complete
actor-critic architecture for propofol infusion control.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class StateEncoder(nn.Module):
    """
    Classical State Encoder for Quantum Policy.
    
    This network preprocesses the raw state observation into
    a compact representation suitable for quantum encoding.
    It maps the state_dim dimensional input to n_qubits outputs
    in the range [-1, 1] for angle encoding.
    
    Architecture:
        Input(state_dim) -> FC(hidden_dims) -> Output(n_qubits) -> Tanh
    
    Attributes:
        input_dim: Dimension of input state
        output_dim: Dimension of output (n_qubits)
        layers: Sequential network layers
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 2,
        hidden_dims: List[int] = [64, 32],
        activation: str = "relu"
    ):
        """
        Initialize the state encoder.
        
        Args:
            input_dim: Dimension of input state
            output_dim: Dimension of output (should match n_qubits)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "elu":
            act_fn = nn.ELU
        else:
            act_fn = nn.ReLU
        
        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        # Output layer with tanh to bound to [-1, 1]
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor (shape: [batch_size, input_dim] or [input_dim])
        
        Returns:
            Encoded features (shape: [batch_size, output_dim] or [output_dim])
        """
        return self.layers(state)


class CriticNetwork(nn.Module):
    """
    Critic Network for State-Value or Action-Value Estimation.
    
    This network estimates V(s) or Q(s, a) depending on configuration.
    For DDPG, it estimates Q(s, a) by concatenating state and action.
    For actor-critic variants, it can estimate V(s) from state alone.
    
    Architecture:
        Input(state_dim + action_dim) -> FC(hidden_dims) -> Output(1)
    
    Attributes:
        state_dim: Dimension of state input
        action_dim: Dimension of action input (0 for V(s))
        hidden_dims: Hidden layer dimensions
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 1,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu"
    ):
        """
        Initialize the critic network.
        
        Args:
            state_dim: Dimension of state
            action_dim: Dimension of action (0 for state-value V(s))
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim
        
        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "elu":
            act_fn = nn.ELU
        else:
            act_fn = nn.ReLU
        
        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor (shape: [batch_size, state_dim])
            action: Action tensor (shape: [batch_size, action_dim]), optional
        
        Returns:
            Value estimate (shape: [batch_size, 1])
        """
        if action is not None:
            # Q(s, a) - concatenate state and action
            x = torch.cat([state, action], dim=-1)
        else:
            # V(s) - state only
            x = state
        
        return self.layers(x)


class TwinCriticNetwork(nn.Module):
    """
    Twin Critic Networks for TD3-style algorithms.
    
    Implements two independent Q-networks to reduce overestimation
    bias by taking the minimum of the two Q-value estimates.
    
    Q1(s, a), Q2(s, a) -> min(Q1, Q2) for target computation
    
    Attributes:
        critic1: First Q-network
        critic2: Second Q-network
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 1,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu"
    ):
        """
        Initialize twin critic networks.
        
        Args:
            state_dim: Dimension of state
            action_dim: Dimension of action
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        super().__init__()
        
        self.critic1 = CriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation
        )
        
        self.critic2 = CriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation
        )
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both critics.
        
        Args:
            state: State tensor
            action: Action tensor
        
        Returns:
            Tuple of (Q1, Q2) values
        """
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q1 value only (for policy optimization)."""
        return self.critic1(state, action)
    
    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get minimum of Q1 and Q2 (for target computation)."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class ActorNetwork(nn.Module):
    """
    Classical Actor Network (Policy).
    
    A standard MLP policy for continuous action spaces.
    This serves as a baseline to compare with the quantum policy.
    
    Architecture:
        Input(state_dim) -> FC(hidden_dims) -> Output(action_dim) -> Sigmoid
    
    Output is scaled to [0, action_scale] for propofol dose.
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 1,
        hidden_dims: List[int] = [256, 256],
        action_scale: float = 1.0,
        activation: str = "relu"
    ):
        """
        Initialize the actor network.
        
        Args:
            state_dim: Dimension of state
            action_dim: Dimension of action
            hidden_dims: Hidden layer dimensions
            action_scale: Maximum action value
            activation: Activation function
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        
        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "elu":
            act_fn = nn.ELU
        else:
            act_fn = nn.ReLU
        
        # Build layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        # Output layer with sigmoid for [0, 1] action
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor
        
        Returns:
            Action tensor scaled to [0, action_scale]
        """
        action = self.layers(state)
        return action * self.action_scale
    
    def get_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Get action for inference.
        
        Args:
            state: State array
            deterministic: Whether to use deterministic action
        
        Returns:
            Action array
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            action = self.forward(state_tensor)
            return action.squeeze(0).numpy()


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck Noise Process for Exploration.
    
    Generates temporally correlated noise for smooth exploration
    in continuous action spaces. Commonly used with DDPG.
    
    dX_t = θ(μ - X_t)dt + σdW_t
    
    Attributes:
        mu: Mean (usually 0)
        theta: Rate of mean reversion
        sigma: Volatility
        state: Current noise state
    """
    
    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        seed: Optional[int] = None
    ):
        """
        Initialize OU noise.
        
        Args:
            action_dim: Dimension of action space
            mu: Mean of noise
            theta: Mean reversion rate
            sigma: Noise volatility
            seed: Random seed
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(self):
        """Reset noise state to initial conditions."""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """
        Sample noise.
        
        Returns:
            Noise array
        """
        dx = (self.theta * (self.mu - self.state) + 
              self.sigma * self.rng.standard_normal(self.action_dim))
        self.state += dx
        return self.state.copy()
    
    def __call__(self) -> np.ndarray:
        """Sample noise (callable interface)."""
        return self.sample()


class GaussianNoise:
    """
    Simple Gaussian Noise for Exploration.
    
    Generates i.i.d. Gaussian noise, simpler but less temporally
    smooth than Ornstein-Uhlenbeck noise.
    """
    
    def __init__(
        self,
        action_dim: int,
        sigma: float = 0.1,
        sigma_min: float = 0.01,
        decay: float = 0.995,
        seed: Optional[int] = None
    ):
        """
        Initialize Gaussian noise.
        
        Args:
            action_dim: Dimension of action space
            sigma: Initial standard deviation
            sigma_min: Minimum standard deviation
            decay: Decay rate per episode
            seed: Random seed
        """
        self.action_dim = action_dim
        self.sigma = sigma
        self.sigma_init = sigma
        self.sigma_min = sigma_min
        self.decay = decay
        
        self.rng = np.random.default_rng(seed)
    
    def sample(self) -> np.ndarray:
        """
        Sample noise.
        
        Returns:
            Noise array
        """
        return self.rng.normal(0, self.sigma, self.action_dim)
    
    def decay_sigma(self):
        """Decay sigma after each episode."""
        self.sigma = max(self.sigma_min, self.sigma * self.decay)
    
    def reset(self):
        """Reset sigma to initial value."""
        self.sigma = self.sigma_init
    
    def __call__(self) -> np.ndarray:
        """Sample noise (callable interface)."""
        return self.sample()


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """
    Soft update of target network parameters.
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    
    Args:
        target: Target network
        source: Source network
        tau: Soft update coefficient (0 < tau << 1)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update(target: nn.Module, source: nn.Module):
    """
    Hard update of target network parameters.
    
    θ_target = θ_source
    
    Args:
        target: Target network
        source: Source network
    """
    target.load_state_dict(source.state_dict())


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the networks
    print("Testing Classical Networks...")
    
    # Test StateEncoder
    encoder = StateEncoder(input_dim=4, output_dim=2, hidden_dims=[64, 32])
    state = torch.randn(8, 4)
    encoded = encoder(state)
    print(f"StateEncoder: {state.shape} -> {encoded.shape}")
    print(f"  Encoded range: [{encoded.min():.2f}, {encoded.max():.2f}]")
    print(f"  Parameters: {count_parameters(encoder)}")
    
    # Test CriticNetwork
    critic = CriticNetwork(state_dim=4, action_dim=1, hidden_dims=[256, 256])
    action = torch.rand(8, 1)
    q_value = critic(state, action)
    print(f"\nCriticNetwork: state{state.shape}, action{action.shape} -> {q_value.shape}")
    print(f"  Parameters: {count_parameters(critic)}")
    
    # Test TwinCriticNetwork
    twin_critic = TwinCriticNetwork(state_dim=4, action_dim=1, hidden_dims=[256, 256])
    q1, q2 = twin_critic(state, action)
    print(f"\nTwinCriticNetwork: Q1{q1.shape}, Q2{q2.shape}")
    print(f"  Parameters: {count_parameters(twin_critic)}")
    
    # Test ActorNetwork
    actor = ActorNetwork(state_dim=4, action_dim=1, hidden_dims=[256, 256])
    action_out = actor(state)
    print(f"\nActorNetwork: {state.shape} -> {action_out.shape}")
    print(f"  Action range: [{action_out.min():.2f}, {action_out.max():.2f}]")
    print(f"  Parameters: {count_parameters(actor)}")
    
    # Test noise
    ou_noise = OrnsteinUhlenbeckNoise(action_dim=1, seed=42)
    gaussian_noise = GaussianNoise(action_dim=1, seed=42)
    
    print(f"\nOU Noise samples: {[ou_noise().item() for _ in range(5)]}")
    print(f"Gaussian Noise samples: {[gaussian_noise().item() for _ in range(5)]}")
    
    print("\nTest complete!")
