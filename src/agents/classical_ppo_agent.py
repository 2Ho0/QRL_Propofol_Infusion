"""
Classical PPO Agent (Baseline)
==============================

Pure classical Proximal Policy Optimization agent for comparison with quantum agents.
Uses standard MLP networks for both actor and critic.

Architecture:
- Actor: MLP → Gaussian policy (μ, σ)
- Critic: MLP → V(s)
- Optional: LSTM/Transformer encoder for temporal features
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.networks import (
    LSTMEncoder,
    TransformerEncoder
)


def create_mlp(input_dim: int, output_dim: int, hidden_dims: List[int], activation=nn.ReLU):
    """Create MLP network."""
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    
    return nn.Sequential(*layers)


class ClassicalActorCritic(nn.Module):
    """Classical Actor-Critic network with optional temporal encoder."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        encoder_type: Optional[str] = None,
        encoded_dim: int = 32,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.encoder_type = encoder_type
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Temporal encoder (optional)
        if encoder_type == "lstm":
            self.encoder = LSTMEncoder(
                input_dim=state_dim,
                hidden_dim=encoded_dim,
                num_layers=2
            )
            actor_input_dim = encoded_dim
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                input_dim=state_dim,
                d_model=encoded_dim,
                nhead=4,
                num_layers=2
            )
            actor_input_dim = encoded_dim
        elif encoder_type == "hybrid":
            # Use LSTM as fallback for hybrid
            self.encoder = LSTMEncoder(
                input_dim=state_dim,
                hidden_dim=encoded_dim,
                num_layers=2
            )
            actor_input_dim = encoded_dim
        else:
            self.encoder = None
            actor_input_dim = state_dim
        
        # Actor network (policy)
        self.actor = create_mlp(
            input_dim=actor_input_dim,
            output_dim=action_dim * 2,  # mean and log_std
            hidden_dims=hidden_dims,
            activation=nn.ReLU
        )
        
        # Critic network (value function)
        self.critic = create_mlp(
            input_dim=actor_input_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=nn.ReLU
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch, state_dim]
        
        Returns:
            action_mean: Mean of action distribution [batch, action_dim]
            action_std: Std of action distribution [batch, action_dim]
            value: State value [batch, 1]
        """
        # Encode state if encoder is available
        if self.encoder is not None:
            # Add sequence dimension for encoders
            if len(state.shape) == 2:
                state = state.unsqueeze(1)  # [batch, 1, state_dim]
            encoded = self.encoder(state)
            
            # Handle LSTM output (returns tuple)
            if isinstance(encoded, tuple):
                encoded = encoded[0]  # Take hidden state
            
            # Take last timestep if sequence
            if len(encoded.shape) == 3:
                encoded = encoded[:, -1, :]
        else:
            encoded = state
        
        # Actor: compute mean and log_std
        actor_out = self.actor(encoded)
        mean, log_std = torch.chunk(actor_out, 2, dim=-1)
        
        # Clamp log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # Critic: compute value
        value = self.critic(encoded)
        
        return mean, std, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy."""
        mean, std, value = self.forward(state)
        
        if deterministic:
            return mean, value
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            return action, value
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        """Evaluate actions for PPO update."""
        mean, std, value = self.forward(state)
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return value, log_prob, entropy


class ClassicalPPOAgent:
    """Classical PPO agent with optional temporal encoder."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        encoder_type: Optional[str] = None,
        encoded_dim: int = 32,
        device: str = "cpu"
    ):
        """
        Initialize Classical PPO agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            encoder_type: Encoder type (lstm/transformer/hybrid/None)
            encoded_dim: Encoded state dimension
            device: Device (cpu/cuda)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        
        # Network
        self.network = ClassicalActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            encoder_type=encoder_type,
            encoded_dim=encoded_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Storage for trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, value = self.network.get_action(state_tensor, deterministic)
        
        action_np = action.cpu().numpy().flatten()
        
        # Store for training (if not deterministic)
        if not deterministic:
            self.states.append(state)
            self.actions.append(action_np)
            self.values.append(value.item())
        
        return action_np
    
    def store_transition(self, reward: float, done: bool, log_prob: Optional[float] = None):
        """Store transition in trajectory buffer."""
        self.rewards.append(reward)
        self.dones.append(done)
        if log_prob is not None:
            self.log_probs.append(log_prob)
    
    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            next_value: Value of the next state (0 if terminal)
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        values = self.values + [next_value]
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + np.array(self.values)
        
        return advantages, returns
    
    def update(self, next_state: np.ndarray, done: bool, epochs: int = 10) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            next_state: Next state for GAE computation
            done: Whether episode is done
            epochs: Number of optimization epochs
        
        Returns:
            Dictionary of training metrics
        """
        # Compute next value
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, next_value = self.network.get_action(next_state_tensor, deterministic=True)
            next_value = next_value.item() if not done else 0.0
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(np.array(self.log_probs)).to(self.device).unsqueeze(1)
        advantages_tensor = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(epochs):
            # Evaluate actions
            values, log_probs, entropy = self.network.evaluate_actions(states_tensor, actions_tensor)
            
            # Policy loss (PPO clipped objective)
            ratio = torch.exp(log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = ((values - returns_tensor) ** 2).mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Track losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        # Clear trajectory buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        return {
            "policy_loss": total_policy_loss / epochs,
            "value_loss": total_value_loss / epochs,
            "entropy": total_entropy / epochs
        }
    
    def save(self, path: str):
        """Save agent."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


# Test
if __name__ == "__main__":
    print("Testing Classical PPO Agent...")
    
    # Create agent
    agent = ClassicalPPOAgent(
        state_dim=10,
        action_dim=2,
        hidden_dims=[256, 256],
        encoder_type="lstm",
        encoded_dim=32
    )
    
    # Test forward pass
    state = np.random.randn(10)
    action = agent.select_action(state)
    
    print(f"State shape: {state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
    
    print("✓ Classical PPO Agent test complete!")
