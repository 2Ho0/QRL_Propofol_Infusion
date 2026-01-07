"""
Classical Neural Networks for Hybrid Quantum-Classical RL
============================================================

This module implements the classical neural network components
for the hybrid Quantum RL system following CBIM paper Fig.4:

1. LSTMEncoder: LSTM-based temporal encoder for time-series state
2. TransformerEncoder: Transformer-based alternative encoder
3. StateEncoder: Simple MLP encoder for quantum circuit input
4. CriticNetwork: Value/Q-function network for actor-critic methods
5. BISPredictor: Auxiliary network for BIS prediction (Formulation 48)
6. ActionValueNetwork: Q(s,a) network for DDPG-style algorithms

Architecture follows CBIM paper Fig.4:
- Time Series Input → LSTM/Transformer → Encoded Features
- Patient Demographics → MLP → Encoded Demographics
- Concatenated Features → Policy/Critic Networks

Formulation (48): BIS Prediction Loss
    G^Pred(θ) = (1/(T-1)) Σ |BIS(t+1) - BIS_pred(t)|²
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    
    Adds sinusoidal position information to input embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Tensor of shape [batch, seq_len, d_model]
        
        Returns:
            Position-encoded tensor of shape [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMEncoder(nn.Module):
    """
    LSTM-based Temporal Encoder for Time-Series State (CBIM Paper Fig.4).
    
    Processes time-series observations [s_{t-W}, ..., s_t] to extract
    temporal patterns for policy learning.
    
    Architecture:
        Input(seq_len, input_dim) -> LSTM(hidden_dim) -> FC -> Output(output_dim)
    
    Attributes:
        input_dim: Dimension of each timestep's observation
        hidden_dim: LSTM hidden state dimension
        output_dim: Output feature dimension
        num_layers: Number of LSTM layers
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM encoder.
        
        Args:
            input_dim: Dimension of input at each timestep
            hidden_dim: LSTM hidden dimension
            output_dim: Output feature dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, input_dim] or [batch, input_dim]
            hidden: Optional initial hidden state (h_0, c_0)
        
        Returns:
            Tuple of (encoded_features, (h_n, c_n))
            encoded_features: [batch, output_dim]
        """
        # Handle single timestep input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)  # [batch, seq_len, hidden_dim]
        
        # Take last timestep output
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # Output projection
        encoded = self.output_proj(last_output)  # [batch, output_dim]
        
        return encoded, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state for LSTM.
        
        Args:
            batch_size: Batch size
            device: Torch device
        
        Returns:
            Tuple of (h_0, c_0) hidden and cell states
        """
        num_directions = 2 if self.bidirectional else 1
        h_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)


class TransformerEncoder(nn.Module):
    """
    Transformer-based Temporal Encoder for Time-Series State.
    
    Alternative to LSTM for processing temporal observations.
    Uses self-attention to capture long-range dependencies.
    
    Architecture:
        Input -> Positional Encoding -> Transformer Encoder -> Mean Pool -> Output
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 64,
        output_dim: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        """
        Initialize Transformer encoder.
        
        Args:
            input_dim: Dimension of input at each timestep
            d_model: Transformer model dimension
            output_dim: Output feature dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
        
        # CLS token for sequence representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, seq_len, input_dim] or [batch, input_dim]
            mask: Optional attention mask
        
        Returns:
            encoded_features: [batch, output_dim]
        """
        # Handle single timestep input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, seq_len+1, d_model]
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x, mask=mask)  # [batch, seq_len+1, d_model]
        
        # Use CLS token output as sequence representation
        cls_output = x[:, 0, :]  # [batch, d_model]
        
        # Output projection
        encoded = self.output_proj(cls_output)  # [batch, output_dim]
        
        return encoded


class DemographicsEncoder(nn.Module):
    """
    Encoder for patient demographic features (age, weight, height).
    
    Following CBIM paper Fig.4, this encodes covariate information
    that is concatenated with temporal features.
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # age, weight, height
        hidden_dims: List[int] = [32, 16],
        output_dim: int = 16,
        activation: str = "relu"
    ):
        """
        Initialize demographics encoder.
        
        Args:
            input_dim: Number of demographic features
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function
        """
        super().__init__()
        
        # Activation function
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh
        
        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, demographics: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            demographics: [batch, input_dim] - normalized demographic features
        
        Returns:
            encoded: [batch, output_dim]
        """
        return self.layers(demographics)


class BISPredictor(nn.Module):
    """
    BIS Prediction Network for Auxiliary Task - Formulation (48).
    
    G^Pred(θ) = (1/(T-1)) Σ |BIS(t+1) - BIS_pred(t)|²
    
    This network predicts the BIS value τ seconds into the future
    based on current state and action, providing explainability.
    
    Architecture:
        [encoded_state, action] -> MLP -> BIS_predicted
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 1,
        hidden_dims: List[int] = [64, 32],
        prediction_horizon: int = 1  # steps ahead to predict
    ):
        """
        Initialize BIS predictor.
        
        Args:
            state_dim: Dimension of encoded state
            action_dim: Dimension of action
            hidden_dims: Hidden layer dimensions
            prediction_horizon: Number of steps ahead to predict
        """
        super().__init__()
        
        self.prediction_horizon = prediction_horizon
        
        input_dim = state_dim + action_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output single BIS value (0-100 range)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output in [0, 1], scale to [0, 100]
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, encoded_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            encoded_state: [batch, state_dim] - encoded temporal state
            action: [batch, action_dim] - action to take
        
        Returns:
            bis_predicted: [batch, 1] - predicted BIS value (normalized)
        """
        x = torch.cat([encoded_state, action], dim=-1)
        return self.layers(x) * 100  # Scale to BIS range [0, 100]
    
    def compute_loss(
        self, 
        encoded_states: torch.Tensor, 
        actions: torch.Tensor,
        next_bis: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute BIS prediction loss - Formulation (48).
        
        Args:
            encoded_states: [batch, state_dim]
            actions: [batch, action_dim]
            next_bis: [batch, 1] - actual next BIS values
        
        Returns:
            Prediction loss (MSE)
        """
        predicted_bis = self.forward(encoded_states, actions)
        return F.mse_loss(predicted_bis, next_bis)


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
        state = state.float()
        if action is not None:
            # Q(s, a) - concatenate state and action
            action = action.float()
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
        output_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize bias to encourage higher initial actions
        # sigmoid(0.5) ≈ 0.62, which prevents network from getting stuck at low outputs
        # This is critical for drug infusion where we need meaningful initial doses
        nn.init.constant_(output_layer.bias, 0.5)
        
        layers.append(output_layer)
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
    
    # Test LSTMEncoder
    lstm_encoder = LSTMEncoder(input_dim=8, hidden_dim=64, output_dim=32)
    seq_input = torch.randn(8, 12, 8)  # batch=8, seq_len=12, features=8
    lstm_out, hidden = lstm_encoder(seq_input)
    print(f"\nLSTMEncoder: {seq_input.shape} -> {lstm_out.shape}")
    print(f"  Parameters: {count_parameters(lstm_encoder)}")
    
    # Test TransformerEncoder
    transformer_encoder = TransformerEncoder(input_dim=8, d_model=64, output_dim=32)
    transformer_out = transformer_encoder(seq_input)
    print(f"\nTransformerEncoder: {seq_input.shape} -> {transformer_out.shape}")
    print(f"  Parameters: {count_parameters(transformer_encoder)}")
    
    # Test DemographicsEncoder
    demo_encoder = DemographicsEncoder(input_dim=3, output_dim=16)
    demographics = torch.randn(8, 3)  # age, weight, height
    demo_out = demo_encoder(demographics)
    print(f"\nDemographicsEncoder: {demographics.shape} -> {demo_out.shape}")
    print(f"  Parameters: {count_parameters(demo_encoder)}")
    
    # Test BISPredictor
    bis_predictor = BISPredictor(state_dim=32, action_dim=1)
    action = torch.rand(8, 1)
    bis_pred = bis_predictor(lstm_out, action)
    print(f"\nBISPredictor: state{lstm_out.shape}, action{action.shape} -> {bis_pred.shape}")
    print(f"  BIS range: [{bis_pred.min():.1f}, {bis_pred.max():.1f}]")
    print(f"  Parameters: {count_parameters(bis_predictor)}")
    
    # Test CriticNetwork
    critic = CriticNetwork(state_dim=4, action_dim=1, hidden_dims=[256, 256])
    q_value = critic(state, torch.rand(8, 1))
    print(f"\nCriticNetwork: state{state.shape}, action -> {q_value.shape}")
    print(f"  Parameters: {count_parameters(critic)}")
    
    # Test TwinCriticNetwork
    twin_critic = TwinCriticNetwork(state_dim=4, action_dim=1, hidden_dims=[256, 256])
    q1, q2 = twin_critic(state, torch.rand(8, 1))
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
