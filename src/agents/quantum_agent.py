"""
Quantum DDPG Agent for Propofol Infusion Control
==================================================

This module implements a hybrid Quantum-Classical Deep Deterministic
Policy Gradient (DDPG) agent for closed-loop propofol infusion control.

Architecture:
-------------
- Actor (Policy): Quantum Policy with VQC (2 qubits, N layers)
- Critic (Value): Classical Twin Q-Networks for stability

The agent follows the DDPG algorithm with:
1. Experience Replay Buffer
2. Target Networks with soft updates
3. Ornstein-Uhlenbeck exploration noise
4. Parameter-shift rule for quantum gradients (via PennyLane)

Following the CBIM paper's RL formulation:
- State: [BIS_error_normalized, Ce_normalized, dBIS/dt, prev_dose]
- Action: Continuous propofol infusion rate [0, max_dose]
- Reward: Negative squared BIS error + safety penalties
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import yaml
from pathlib import Path

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.vqc import QuantumPolicy, QuantumPolicySimple
from models.networks import (
    CriticNetwork, 
    TwinCriticNetwork,
    StateEncoder,
    OrnsteinUhlenbeckNoise,
    GaussianNoise,
    soft_update,
    hard_update
)


@dataclass
class Transition:
    """
    A single transition in the environment.
    
    Attributes:
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Next state
        done: Whether episode terminated
    """
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Experience Replay Buffer for off-policy RL.
    
    Stores transitions and provides uniform random sampling
    for training the agent.
    
    Attributes:
        capacity: Maximum buffer size
        buffer: Deque storing transitions
    """
    
    def __init__(self, capacity: int = 100000, seed: Optional[int] = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for sampling
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.buffer.append(Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        ))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of tensors (states, actions, rewards, next_states, dones)
        """
        transitions = self.rng.sample(list(self.buffer), batch_size)
        
        states = torch.FloatTensor(np.array([t.state for t in transitions]))
        actions = torch.FloatTensor(np.array([t.action for t in transitions]))
        rewards = torch.FloatTensor(np.array([t.reward for t in transitions])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions]))
        dones = torch.FloatTensor(np.array([t.done for t in transitions])).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size


class QuantumDDPGAgent:
    """
    Hybrid Quantum-Classical DDPG Agent.
    
    This agent uses a Variational Quantum Circuit (VQC) as the policy
    network (actor) and classical neural networks as the value network
    (critic). It follows the DDPG algorithm for continuous action spaces.
    
    Key Components:
    - Quantum Actor: VQC-based policy for action selection
    - Classical Critic: Twin Q-networks for value estimation
    - Target Networks: Slowly updated copies for stable learning
    - Replay Buffer: Experience storage for off-policy learning
    - Exploration Noise: OU or Gaussian noise for exploration
    
    Attributes:
        config: Agent configuration
        actor: Quantum policy network
        actor_target: Target quantum policy
        critic: Twin critic networks
        critic_target: Target twin critics
        actor_optimizer: Optimizer for quantum policy
        critic_optimizer: Optimizer for critics
        replay_buffer: Experience replay buffer
        noise: Exploration noise generator
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 1,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        device: str = "cpu",
        seed: Optional[int] = None
    ):
        """
        Initialize the Quantum DDPG agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
            config_path: Path to YAML configuration
            device: Device for computation ('cpu' or 'cuda')
            seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.seed = seed
        
        # Load configuration
        self.config = self._load_config(config, config_path)
        
        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Extract config sections
        quantum_config = self.config.get('quantum', {})
        training_config = self.config.get('training', {})
        network_config = self.config.get('networks', {})
        
        # Hyperparameters
        self.gamma = training_config.get('gamma', 0.99)
        self.tau = training_config.get('tau', 0.005)
        self.batch_size = training_config.get('batch_size', 64)
        self.warmup_steps = training_config.get('warmup_steps', 1000)
        self.update_every = training_config.get('update_every', 1)
        self.policy_delay = training_config.get('policy_delay', 2)
        self.max_grad_norm = training_config.get('max_grad_norm', 1.0)
        
        # Action scaling
        self.action_scale = quantum_config.get('action_scale', 200.0)
        
        # Build networks
        self._build_networks(quantum_config, network_config)
        
        # Build optimizers
        self._build_optimizers(training_config)
        
        # Build replay buffer
        buffer_size = training_config.get('buffer_size', 100000)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        
        # Build exploration noise
        self._build_noise(training_config)
        
        # Training state
        self.total_steps = 0
        self.update_count = 0
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'q_values': []
        }
    
    def _load_config(
        self, 
        config: Optional[Dict], 
        config_path: Optional[str]
    ) -> Dict:
        """Load configuration from dictionary or file."""
        if config is not None:
            return config
        
        if config_path is not None:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'quantum': {
                'n_qubits': 2,
                'n_layers': 4,
                'encoding': 'angle',
                'device': 'default.qubit',
                'action_scale': 200.0
            },
            'networks': {
                'critic': {'hidden_dims': [256, 256]},
                'encoder': {'hidden_dims': [64, 32]}
            },
            'training': {
                'gamma': 0.99,
                'tau': 0.005,
                'batch_size': 64,
                'actor_lr': 0.0001,
                'critic_lr': 0.001,
                'buffer_size': 100000,
                'warmup_steps': 1000,
                'noise_type': 'ou',
                'noise_sigma': 0.2,
                'noise_theta': 0.15
            }
        }
    
    def _build_networks(self, quantum_config: Dict, network_config: Dict):
        """Build actor and critic networks."""
        # Quantum Actor (Policy)
        n_qubits = quantum_config.get('n_qubits', 2)
        n_layers = quantum_config.get('n_layers', 4)
        encoder_config = network_config.get('encoder', {})
        
        self.actor = QuantumPolicy(
            state_dim=self.state_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoder_hidden=encoder_config.get('hidden_dims', [64, 32]),
            action_scale=1.0,  # Normalized to [0, 1]
            device_name=quantum_config.get('device', 'default.qubit'),
            seed=self.seed
        )
        
        # Target Actor (copy of actor)
        self.actor_target = QuantumPolicy(
            state_dim=self.state_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoder_hidden=encoder_config.get('hidden_dims', [64, 32]),
            action_scale=1.0,
            device_name=quantum_config.get('device', 'default.qubit'),
            seed=self.seed
        )
        hard_update(self.actor_target, self.actor)
        
        # Freeze target actor
        for param in self.actor_target.parameters():
            param.requires_grad = False
        
        # Twin Critic Networks
        critic_config = network_config.get('critic', {})
        self.critic = TwinCriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=critic_config.get('hidden_dims', [256, 256])
        )
        
        # Target Critic
        self.critic_target = TwinCriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=critic_config.get('hidden_dims', [256, 256])
        )
        hard_update(self.critic_target, self.critic)
        
        # Freeze target critic
        for param in self.critic_target.parameters():
            param.requires_grad = False
    
    def _build_optimizers(self, training_config: Dict):
        """Build optimizers for actor and critic."""
        actor_lr = training_config.get('actor_lr', 0.0001)
        critic_lr = training_config.get('critic_lr', 0.001)
        
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=critic_lr
        )
    
    def _build_noise(self, training_config: Dict):
        """Build exploration noise generator."""
        noise_type = training_config.get('noise_type', 'ou')
        
        if noise_type == 'ou':
            self.noise = OrnsteinUhlenbeckNoise(
                action_dim=self.action_dim,
                mu=0.0,
                theta=training_config.get('noise_theta', 0.15),
                sigma=training_config.get('noise_sigma', 0.2),
                seed=self.seed
            )
        else:
            self.noise = GaussianNoise(
                action_dim=self.action_dim,
                sigma=training_config.get('noise_sigma', 0.2),
                sigma_min=training_config.get('noise_min', 0.01),
                decay=training_config.get('noise_decay', 0.995),
                seed=self.seed
            )
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Select action given state.
        
        Args:
            state: Current state
            deterministic: If True, return action without noise
            add_noise: Whether to add exploration noise
        
        Returns:
            Action array (scaled to [0, action_scale])
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            # Get action from quantum policy (in [0, 1])
            action = self.actor(state_tensor).squeeze(0).numpy()
        
        # Add exploration noise
        if add_noise and not deterministic:
            noise = self.noise()
            action = action + noise
            action = np.clip(action, 0, 1)
        
        # Scale to actual action range
        action_scaled = action * self.action_scale
        
        return action_scaled
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action taken (scaled)
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        # Normalize action back to [0, 1] for storage
        action_normalized = action / self.action_scale
        
        self.replay_buffer.push(
            state=state,
            action=action_normalized,
            reward=reward,
            next_state=next_state,
            done=done
        )
        
        self.total_steps += 1
    
    def update(self) -> Dict[str, float]:
        """
        Perform one update step.
        
        Returns:
            Dictionary of training metrics
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        states = states.float().to(self.device)
        actions = actions.float().to(self.device)
        rewards = rewards.float().to(self.device)
        next_states = next_states.float().to(self.device)
        dones = dones.float().to(self.device)
        
        # Update critic
        critic_loss, q_values = self._update_critic(
            states, actions, rewards, next_states, dones
        )
        
        # Update actor (with policy delay for TD3-style)
        actor_loss = None
        if self.update_count % self.policy_delay == 0:
            actor_loss = self._update_actor(states)
            
            # Soft update target networks
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
        
        self.update_count += 1
        
        # Record stats
        metrics = {
            'critic_loss': critic_loss,
            'q_value_mean': q_values.mean().item()
        }
        
        if actor_loss is not None:
            metrics['actor_loss'] = actor_loss
        
        return metrics
    
    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """
        Update critic networks.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
        
        Returns:
            Tuple of (critic_loss, q_values)
        """
        with torch.no_grad():
            # 텐서 dtype을 float32로 통일
            next_states = next_states.float()
            
            # next_actions 생성 (actor_target 사용)
            next_actions = self.actor_target(next_states)
            next_actions = next_actions.float()
            
            # Compute target Q-values (using minimum of twin critics)
            target_q = self.critic_target.q_min(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # states와 actions도 float32로 변환
        states = states.float()
        actions = actions.float()
        
        # Current Q-values
        q1, q2 = self.critic(states, actions)
        
        # Critic loss (MSE for both critics)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), 
                self.max_grad_norm
            )
        
        self.critic_optimizer.step()
        
        return critic_loss.item(), q1
    
    def _update_actor(self, states: torch.Tensor) -> float:
        """
        Update actor (quantum policy).
        
        Args:
            states: Batch of states
        
        Returns:
            Actor loss value
        """
        # Compute actions from current policy
        states = states.float()
        
        actions = self.actor(states)
        actions = actions.float()
        
        # Actor loss: maximize Q-value (minimize negative Q)
        actor_loss = -self.critic.q1(states, actions).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.max_grad_norm
            )
        
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def train_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """
        Perform a complete training step.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        
        Returns:
            Training metrics dictionary
        """
        # Store transition
        self.store_transition(state, action, reward, next_state, done)
        
        # Update if enough samples and past warmup
        metrics = {}
        if (self.total_steps >= self.warmup_steps and 
            self.total_steps % self.update_every == 0):
            metrics = self.update()
        
        return metrics
    
    def reset_noise(self):
        """Reset exploration noise at start of episode."""
        self.noise.reset()
    
    def decay_noise(self):
        """Decay exploration noise (for Gaussian noise)."""
        if hasattr(self.noise, 'decay_sigma'):
            self.noise.decay_sigma()
    
    def save(self, path: str):
        """
        Save agent state.
        
        Args:
            path: Path to save checkpoint
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
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """
        Load agent state.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.update_count = checkpoint['update_count']
    
    def get_quantum_info(self) -> Dict:
        """Get information about the quantum circuit."""
        return self.actor.get_quantum_circuit_info()
    
    def set_eval_mode(self):
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
    
    def set_train_mode(self):
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()


if __name__ == "__main__":
    # Test the agent
    print("Testing QuantumDDPGAgent...")
    
    # Create agent
    agent = QuantumDDPGAgent(
        state_dim=4,
        action_dim=1,
        seed=42
    )
    
    print(f"Agent created")
    print(f"Quantum circuit info: {agent.get_quantum_info()}")
    
    # Test action selection
    state = np.random.randn(4)
    action = agent.select_action(state)
    print(f"\nState: {state}")
    print(f"Action: {action}")
    
    # Test training loop (dummy data)
    print("\nTesting training loop...")
    for step in range(100):
        state = np.random.randn(4)
        action = agent.select_action(state)
        next_state = np.random.randn(4)
        reward = -np.random.rand()
        done = step == 99
        
        metrics = agent.train_step(state, action, reward, next_state, done)
        
        if step % 20 == 0 and metrics:
            print(f"Step {step}: {metrics}")
    
    # Test save/load
    agent.save("/tmp/test_agent.pt")
    agent.load("/tmp/test_agent.pt")
    print("\nSave/Load test passed!")
    
    print("\nTest complete!")
