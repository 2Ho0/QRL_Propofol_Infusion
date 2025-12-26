"""
Classical DDPG Agent for Baseline Comparison
=============================================

This module implements a standard Classical DDPG agent using only
classical neural networks (no quantum components) as a baseline
to compare with the Quantum DDPG agent.

Architecture:
-------------
- Actor: MLP (256-128-64 hidden layers)
- Critic: Twin MLP Q-networks (TD3 style)
- Encoder: Optional LSTM/Transformer (same as Quantum version)
- Exploration: Ornstein-Uhlenbeck or Gaussian noise

Key Difference from Quantum DDPG:
- Actor is pure MLP (no VQC)
- Everything else identical for fair comparison
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.networks import (
    TwinCriticNetwork,
    OrnsteinUhlenbeckNoise,
    GaussianNoise,
    soft_update,
    hard_update,
    LSTMEncoder,
    TransformerEncoder
)


class MLPActor(nn.Module):
    """
    Classical MLP Actor (Policy Network).
    
    Pure classical neural network for action selection.
    No quantum components - direct baseline for comparison.
    
    Architecture:
        State [input_dim] → FC[256] → ReLU → FC[128] → ReLU → FC[64] → ReLU → FC[output_dim] → Sigmoid
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = 'relu'
    ):
        """
        Initialize MLP Actor.
        
        Args:
            input_dim: Input state dimension
            output_dim: Output action dimension
            hidden_dims: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            
            prev_dim = hidden_dim
        
        # Output layer with Sigmoid for [0, 1] action
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute action.
        
        Args:
            state: State tensor [batch_size, input_dim]
        
        Returns:
            action: Action tensor [batch_size, output_dim] in [0, 1]
        """
        return self.network(state)


class ClassicalDDPGAgent:
    """
    Classical DDPG Agent (Pure MLP, No Quantum).
    
    Standard Deep Deterministic Policy Gradient with:
    - MLP Actor (no VQC)
    - Twin Critic Networks (TD3 style)
    - Optional temporal encoder (LSTM/Transformer)
    - Experience replay
    - Target networks with soft updates
    - Exploration noise (OU or Gaussian)
    
    This serves as a baseline to compare with Quantum DDPG.
    All components except the actor are identical to Quantum version.
    """
    
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 1,
        config: Optional[Dict] = None,
        encoder_type: str = 'none',
        seed: Optional[int] = None
    ):
        """
        Initialize Classical DDPG Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
            encoder_type: Temporal encoder type ('none', 'lstm', 'transformer')
            seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = seed
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Load configuration
        if config is None:
            config = self._default_config()
        self.config = config
        
        # Extract config sections
        network_config = config.get('networks', {})
        training_config = config.get('training', {})
        encoder_config = config.get('encoder', {})
        
        # Set encoder type
        self.encoder_type = encoder_type
        self.encoder = None
        self.encoder_target = None
        
        # Build encoder (optional)
        if encoder_type != 'none':
            self.encoded_dim = encoder_config.get('output_dim', 32)
            if encoder_type == 'lstm':
                self.encoder = LSTMEncoder(
                    input_dim=state_dim,
                    hidden_dim=encoder_config.get('hidden_dim', 64),
                    output_dim=self.encoded_dim,
                    num_layers=encoder_config.get('num_layers', 2),
                    bidirectional=encoder_config.get('bidirectional', True),
                    dropout=encoder_config.get('dropout', 0.1)
                )
                self.encoder_target = LSTMEncoder(
                    input_dim=state_dim,
                    hidden_dim=encoder_config.get('hidden_dim', 64),
                    output_dim=self.encoded_dim,
                    num_layers=encoder_config.get('num_layers', 2),
                    bidirectional=encoder_config.get('bidirectional', True),
                    dropout=encoder_config.get('dropout', 0.1)
                )
            elif encoder_type == 'transformer':
                self.encoder = TransformerEncoder(
                    input_dim=state_dim,
                    d_model=encoder_config.get('d_model', 64),
                    output_dim=self.encoded_dim,
                    nhead=encoder_config.get('n_heads', 4),
                    num_layers=encoder_config.get('num_layers', 2),
                    dropout=encoder_config.get('dropout', 0.1)
                )
                self.encoder_target = TransformerEncoder(
                    input_dim=state_dim,
                    d_model=encoder_config.get('d_model', 64),
                    output_dim=self.encoded_dim,
                    nhead=encoder_config.get('n_heads', 4),
                    num_layers=encoder_config.get('num_layers', 2),
                    dropout=encoder_config.get('dropout', 0.1)
                )
            
            if self.encoder_target is not None:
                hard_update(self.encoder_target, self.encoder)
                for param in self.encoder_target.parameters():
                    param.requires_grad = False
        else:
            self.encoded_dim = state_dim
        
        # Build networks
        actor_config = network_config.get('actor', {})
        self.actor = MLPActor(
            input_dim=self.encoded_dim,
            output_dim=action_dim,
            hidden_dims=actor_config.get('hidden_dims', [256, 128, 64]),
            activation=actor_config.get('activation', 'relu')
        )
        
        self.actor_target = MLPActor(
            input_dim=self.encoded_dim,
            output_dim=action_dim,
            hidden_dims=actor_config.get('hidden_dims', [256, 128, 64]),
            activation=actor_config.get('activation', 'relu')
        )
        hard_update(self.actor_target, self.actor)
        
        for param in self.actor_target.parameters():
            param.requires_grad = False
        
        # Twin Critic Networks
        critic_config = network_config.get('critic', {})
        self.critic = TwinCriticNetwork(
            state_dim=self.encoded_dim,
            action_dim=action_dim,
            hidden_dims=critic_config.get('hidden_dims', [256, 256])
        )
        
        self.critic_target = TwinCriticNetwork(
            state_dim=self.encoded_dim,
            action_dim=action_dim,
            hidden_dims=critic_config.get('hidden_dims', [256, 256])
        )
        hard_update(self.critic_target, self.critic)
        
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=training_config.get('actor_lr', 0.0001)
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=training_config.get('critic_lr', 0.001)
        )
        
        if self.encoder is not None:
            self.encoder_optimizer = optim.Adam(
                self.encoder.parameters(),
                lr=training_config.get('encoder_lr', 0.001)
            )
        else:
            self.encoder_optimizer = None
        
        # Training parameters
        self.gamma = training_config.get('gamma', 0.99)
        self.tau = training_config.get('tau', 0.005)
        self.batch_size = training_config.get('batch_size', 64)
        self.action_scale = config.get('action_scale', 200.0)
        
        # Exploration noise
        noise_type = training_config.get('noise_type', 'ou')
        if noise_type == 'ou':
            self.noise = OrnsteinUhlenbeckNoise(
                action_dim=action_dim,
                mu=0.0,
                theta=training_config.get('noise_theta', 0.15),
                sigma=training_config.get('noise_sigma', 0.2)
            )
        else:
            self.noise = GaussianNoise(
                action_dim=action_dim,
                mu=0.0,
                sigma=training_config.get('noise_sigma', 0.2),
                sigma_min=training_config.get('noise_sigma_min', 0.05),
                decay_rate=training_config.get('noise_decay', 0.9999)
            )
        
        # Replay buffer (imported from quantum_agent if needed)
        from agents.quantum_agent import ReplayBuffer
        self.replay_buffer = ReplayBuffer(
            capacity=training_config.get('buffer_size', 100000),
            seed=seed
        )
        
        # Training state
        self.total_steps = 0
        self.update_count = 0
        self.warmup_steps = training_config.get('warmup_steps', 1000)
        self.update_every = training_config.get('update_every', 1)
    
    def _default_config(self) -> Dict:
        """
        Return default configuration for Classical DDPG.
        
        Returns:
            Dictionary containing default hyperparameters for networks, encoder, and training
        """
        return {
            'networks': {
                'actor': {
                    'hidden_dims': [256, 128, 64],
                    'activation': 'relu'
                },
                'critic': {
                    'hidden_dims': [256, 256]
                }
            },
            'encoder': {
                'hidden_dim': 64,
                'num_layers': 2,
                'output_dim': 32,
                'bidirectional': True,
                'dropout': 0.1
            },
            'training': {
                'gamma': 0.99,
                'tau': 0.005,
                'batch_size': 64,
                'actor_lr': 0.0001,
                'critic_lr': 0.001,
                'encoder_lr': 0.001,
                'buffer_size': 100000,
                'warmup_steps': 1000,
                'update_every': 1,
                'noise_type': 'ou',
                'noise_sigma': 0.2,
                'noise_theta': 0.15
            },
            'action_scale': 200.0
        }
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        add_noise: bool = True,
        state_sequence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Select action given state.
        
        Args:
            state: Current state observation
            deterministic: If True, return deterministic action without noise
            add_noise: If True, add exploration noise
            state_sequence: Optional sequence of states for temporal encoder
        
        Returns:
            Action array scaled to action_scale range
        """
        device = next(self.actor.parameters()).device
        
        with torch.no_grad():
            # Encode state if using encoder
            if self.encoder is not None and state_sequence is not None:
                state_seq = torch.FloatTensor(state_sequence).to(device)
                if state_seq.dim() == 2:
                    state_seq = state_seq.unsqueeze(0)
                encoded_state = self.encoder(state_seq)
            else:
                encoded_state = torch.FloatTensor(state).to(device)
                if encoded_state.dim() == 1:
                    encoded_state = encoded_state.unsqueeze(0)
            
            # Get action from actor
            action = self.actor(encoded_state).cpu().squeeze(0).numpy()
        
        # Add exploration noise
        if add_noise and not deterministic:
            noise = self.noise()
            action = action + noise
            action = np.clip(action, 0, 1)
        
        # Scale to action range
        action_scaled = action * self.action_scale
        
        return action_scaled
    
    def train_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        
        Returns:
            Dictionary of training metrics (critic_loss, actor_loss, q_value_mean) if update performed, empty dict otherwise
        """
        # Store transition
        action_normalized = action / self.action_scale
        self.replay_buffer.push(state, action_normalized, reward, next_state, done)
        self.total_steps += 1
        
        # Update if enough samples
        metrics = {}
        if self.total_steps >= self.warmup_steps and self.total_steps % self.update_every == 0:
            if self.replay_buffer.is_ready(self.batch_size):
                metrics = self.update()
        
        return metrics
    
    def update(self) -> Dict[str, float]:
        """
        Perform DDPG update step with Twin Critic networks.
        
        Updates critic networks using TD3-style double Q-learning,
        then updates actor network using policy gradient.
        
        Returns:
            Dictionary containing:
                - critic_loss: Combined loss of twin critics
                - actor_loss: Actor network loss
                - q_value_mean: Mean Q-value for monitoring
        """
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        device = next(self.actor.parameters()).device
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        
        # Encode states if using encoder
        if self.encoder is not None:
            encoded_states = self.encoder(states.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                encoded_next_states = self.encoder_target(next_states.unsqueeze(1)).squeeze(1)
        else:
            encoded_states = states
            encoded_next_states = next_states
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(encoded_next_states)
            q1_next, q2_next = self.critic_target(encoded_next_states, next_actions)
            target_q = rewards + self.gamma * torch.min(q1_next, q2_next) * (1 - dones)
        
        q1, q2 = self.critic(encoded_states, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        self.actor_optimizer.zero_grad()
        
        if self.encoder is not None:
            with torch.no_grad():
                encoded_states_for_actor = self.encoder(states.unsqueeze(1)).squeeze(1)
        else:
            encoded_states_for_actor = states
        
        actor_actions = self.actor(encoded_states_for_actor)
        q1_actor, _ = self.critic(encoded_states_for_actor, actor_actions)
        actor_loss = -q1_actor.mean()
        
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        
        if self.encoder is not None:
            soft_update(self.encoder_target, self.encoder, self.tau)
        
        self.update_count += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value_mean': q1.mean().item()
        }
    
    def reset_noise(self) -> None:
        """Reset exploration noise to initial state."""
        self.noise.reset()
    
    def decay_noise(self) -> None:
        """Decay exploration noise sigma for annealing."""
        if hasattr(self.noise, 'decay_sigma'):
            self.noise.decay_sigma()
    
    def save(self, path: str) -> None:
        """
        Save agent checkpoint to disk.
        
        Args:
            path: File path to save checkpoint
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'config': self.config
        }
        
        if self.encoder is not None:
            checkpoint['encoder_state_dict'] = self.encoder.state_dict()
            checkpoint['encoder_target_state_dict'] = self.encoder_target.state_dict()
            checkpoint['encoder_optimizer_state_dict'] = self.encoder_optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path: str) -> None:
        """
        Load agent checkpoint from disk.
        
        Args:
            path: File path to load checkpoint from
        """
        checkpoint = torch.load(path)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.encoder is not None and 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.encoder_target.load_state_dict(checkpoint['encoder_target_state_dict'])
            self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        
        self.total_steps = checkpoint.get('total_steps', 0)
        self.update_count = checkpoint.get('update_count', 0)
    
    def get_info(self) -> Dict:
        """Get agent information."""
        return {
            'type': 'Classical DDPG',
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'encoder_type': self.encoder_type,
            'encoded_dim': self.encoded_dim,
            'total_steps': self.total_steps,
            'update_count': self.update_count
        }
