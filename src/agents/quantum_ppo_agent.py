"""
Quantum PPO Agent for Propofol Infusion Control
==================================================

This module implements a hybrid Quantum-Classical Proximal Policy
Optimization (PPO) agent following the CBIM paper formulations (41)-(49).

Architecture:
-------------
- Actor (Policy): Quantum Policy with VQC (2 qubits, N layers) + Gaussian noise
- Critic (Value): Classical V(s) network
- Temporal Encoder: LSTM or Transformer for time-series state

CBIM Paper Formulations:
------------------------
- Formulation (33): P(τ|π_θ) = ρ(s₀) ∏ P(s_{t+1}|s_t,a_t) π_θ(a_t|s_t)
- Formulation (34): J(π_θ) = E[R(τ)]
- Formulation (35): V(s_t,g_t) = E[R_t] = E[Σ γ^t r_t | s_t, g_t]
- Formulation (41): G_RL(θ) = Σ[c₁·G^PPO + c₂·G^Val] + c₃·‖θ‖₂
- Formulation (42): L^PPO(θ) = min(ζ·Â_t, clip(ζ, 1-ε, 1+ε)·Â_t)
- Formulation (43): L^Val(θ) = ‖R_t - γV(s_t)‖
- Formulation (44): H(a_t) = ∫ a_t log₂(a_t) da_t (entropy)
- Formulation (45): ζ = π(a_t|s_t) / π_old(a_t|s_t) (policy ratio)
- Formulation (46): Â_t = Σ (γλ)^(k-t) (r_t + γV(s_{t+1}) - V(s_t)) (GAE)
- Formulation (47): a_t ~ N(μ_π(s_t), σ²_π(s_t)) (Gaussian policy)
- Formulation (48): G^Pred(θ) = (1/(T-1)) Σ |BIS(t+1) - BIS_pred(t)|²
- Formulation (49): G^Train(θ;B) = G^RL(θ) + G^Pred(θ)

References:
-----------
- Schulman et al., "Proximal Policy Optimization Algorithms"
- CBIM (Closed-loop BIS-guided Infusion Model) paper
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
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
    LSTMEncoder,
    TransformerEncoder,
    DemographicsEncoder,
    BISPredictor,
    StateEncoder,
    soft_update,
    hard_update
)


@dataclass
class PPOTransition:
    """
    A single transition for PPO training.
    
    Attributes:
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Next state
        done: Whether episode terminated
        log_prob: Log probability of action under policy
        value: Value estimate V(s)
    """
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class RolloutBuffer:
    """
    Rollout Buffer for On-Policy PPO Training.
    
    Stores trajectories and computes advantages using GAE.
    
    Implements Formulation (46): Generalized Advantage Estimation
    Â_t = Σ_{k=t}^{T} (γλ)^{k-t} δ_k
    where δ_k = r_k + γV(s_{k+1}) - V(s_k)
    """
    
    def __init__(
        self,
        buffer_size: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Maximum number of transitions per rollout
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.reset()
    
    def reset(self):
        """Clear the buffer."""
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.advantages: List[float] = []
        self.returns: List[float] = []
        
        self.ptr = 0
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.ptr += 1
    
    def compute_advantages(self, last_value: float = 0.0):
        """
        Compute GAE advantages - Formulation (46).
        
        Â_t = Σ_{k=t}^{T} (γλ)^{k-t} (r_k + γV(s_{k+1}) - V(s_k))
        
        Args:
            last_value: Value estimate for the last state (bootstrap)
        """
        advantages = []
        returns = []
        
        # Add last value for bootstrapping
        values = self.values + [last_value]
        
        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            if self.dones[t]:
                delta = self.rewards[t] - values[t]
                gae = delta
            else:
                delta = self.rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batches(self, batch_size: int, shuffle: bool = True) -> List[Dict[str, torch.Tensor]]:
        """
        Get mini-batches for training.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
        
        Returns:
            List of batches, each containing tensors
        """
        n_samples = len(self.states)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            
            batch = {
                'states': torch.FloatTensor(np.array([self.states[i] for i in batch_indices])),
                'actions': torch.FloatTensor(np.array([self.actions[i] for i in batch_indices])),
                'old_log_probs': torch.FloatTensor([self.log_probs[i] for i in batch_indices]),
                'advantages': torch.FloatTensor([self.advantages[i] for i in batch_indices]),
                'returns': torch.FloatTensor([self.returns[i] for i in batch_indices])
            }
            
            # Normalize advantages
            batch['advantages'] = (batch['advantages'] - batch['advantages'].mean()) / (batch['advantages'].std() + 1e-8)
            
            batches.append(batch)
        
        return batches
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.ptr
    
    def is_full(self) -> bool:
        """Check if buffer has enough samples."""
        return self.ptr >= self.buffer_size


class GaussianQuantumPolicy(nn.Module):
    """
    Gaussian Policy with Quantum Actor - Formulation (47).
    
    a_t ~ N(μ_π(s_t), σ²_π(s_t))
    
    The mean μ is produced by the quantum circuit, while σ is learned.
    
    Architecture:
        State -> [Encoder (optional)] -> VQC -> μ
        State -> MLP -> log(σ)
    """
    
    def __init__(
        self,
        state_dim: int = 8,
        n_qubits: int = 2,
        n_layers: int = 4,
        encoder_hidden: List[int] = [64, 32],
        action_scale: float = 1.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        device_name: str = "default.qubit",
        seed: Optional[int] = None
    ):
        """
        Initialize Gaussian quantum policy.
        
        Args:
            state_dim: Dimension of input state
            n_qubits: Number of qubits in VQC
            n_layers: Number of VQC layers
            encoder_hidden: Hidden layer sizes for encoder
            action_scale: Scale for output action
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
            device_name: PennyLane device name
            seed: Random seed
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_scale = action_scale
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Quantum policy for mean
        self.quantum_policy = QuantumPolicy(
            state_dim=state_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoder_hidden=encoder_hidden,
            action_scale=1.0,  # Output in [0, 1]
            device_name=device_name,
            seed=seed
        )
        
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(1))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor
        
        Returns:
            Tuple of (mean, std)
        """
        # Get mean from quantum policy
        mean = self.quantum_policy(state)  # [batch, 1] in [0, 1]
        
        # Clamp log_std
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mean)
        
        return mean, std
    
    def get_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Sample action from policy - Formulation (47).
        
        Args:
            state: State array
            deterministic: If True, return mean without sampling
        
        Returns:
            Tuple of (action, log_prob, value_placeholder)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            mean, std = self.forward(state_tensor)
            
            if deterministic:
                action = mean
                log_prob = 0.0
            else:
                # Sample from Gaussian
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1).item()
            
            # Clip action to [0, 1]
            action = torch.clamp(action, 0, 1)
            
            return action.squeeze().numpy() * self.action_scale, log_prob, 0.0
    
    def evaluate_actions(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given state-action pairs.
        
        Args:
            states: Batch of states
            actions: Batch of actions (normalized)
        
        Returns:
            Tuple of (log_probs, entropy)
        """
        mean, std = self.forward(states)
        
        # Normalize actions to [0, 1] for comparison
        actions_normalized = actions / self.action_scale
        
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions_normalized).sum(-1)
        
        # Formulation (44): Entropy H(a) = ∫ a log(a) da
        entropy = dist.entropy().sum(-1)
        
        return log_probs, entropy


class QuantumPPOAgent:
    """
    Hybrid Quantum-Classical PPO Agent.
    
    This agent uses a Variational Quantum Circuit (VQC) as the policy
    network (actor) and classical neural networks for value estimation.
    Follows PPO algorithm with GAE for advantage estimation.
    
    Key Components:
    - Quantum Actor: VQC-based Gaussian policy
    - Classical Critic: Value network V(s)
    - Temporal Encoder: LSTM or Transformer (optional)
    - BIS Predictor: Auxiliary task for explainability
    
    Training follows Formulations (41)-(49):
    - Formulation (41): G^Train = G^RL + G^Pred
    - Formulation (42): PPO clipped objective
    - Formulation (43): Value function loss
    - Formulation (46): GAE for advantages
    
    Attributes:
        config: Agent configuration
        actor: Gaussian quantum policy
        critic: Value network
        encoder: Temporal encoder (LSTM/Transformer)
        bis_predictor: BIS prediction network
        optimizer: Combined optimizer
        rollout_buffer: On-policy rollout storage
    """
    
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 1,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
        encoder_type: str = "none"  # "none", "lstm", "transformer"
    ):
        """
        Initialize the Quantum PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
            config_path: Path to YAML configuration
            device: Device for computation
            seed: Random seed
            encoder_type: Type of temporal encoder
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.seed = seed
        self.encoder_type = encoder_type
        
        # Load configuration
        self.config = self._load_config(config, config_path)
        
        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Extract config sections
        quantum_config = self.config.get('quantum', {})
        training_config = self.config.get('training', {})
        ppo_config = self.config.get('ppo', {})
        network_config = self.config.get('networks', {})
        
        # PPO Hyperparameters
        self.gamma = ppo_config.get('gamma', 0.99)
        self.gae_lambda = ppo_config.get('gae_lambda', 0.95)
        self.clip_epsilon = ppo_config.get('clip_epsilon', 0.2)
        self.c1 = ppo_config.get('c1', 1.0)      # Policy loss coefficient
        self.c2 = ppo_config.get('c2', 0.5)      # Value loss coefficient
        self.c3 = ppo_config.get('c3', 0.01)     # Entropy coefficient
        self.c_pred = ppo_config.get('c_pred', 0.1)  # BIS prediction loss coefficient
        self.max_grad_norm = ppo_config.get('max_grad_norm', 0.5)
        self.n_epochs = ppo_config.get('n_epochs', 10)
        self.batch_size = ppo_config.get('batch_size', 64)
        self.rollout_steps = ppo_config.get('rollout_steps', 2048)
        
        # Action scaling
        self.action_scale = quantum_config.get('action_scale', 200.0)
        
        # Build networks
        self._build_networks(quantum_config, network_config)
        
        # Build optimizer
        self._build_optimizer(training_config)
        
        # Build rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.rollout_steps,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        # Training state
        self.total_steps = 0
        self.update_count = 0
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'approx_kl': []
        }
    
    def _load_config(self, config: Optional[Dict], config_path: Optional[str]) -> Dict:
        """Load configuration from dictionary or file."""
        if config is not None:
            return config
        
        if config_path is not None:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default PPO configuration
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
            'ppo': {
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'c1': 1.0,
                'c2': 0.5,
                'c3': 0.01,
                'c_pred': 0.1,
                'n_epochs': 10,
                'batch_size': 64,
                'rollout_steps': 2048,
                'max_grad_norm': 0.5
            },
            'training': {
                'lr': 0.0003
            }
        }
    
    def _build_networks(self, quantum_config: Dict, network_config: Dict):
        """Build actor, critic, encoder, and predictor networks."""
        # Encoder input dimension depends on encoder type
        if self.encoder_type != "none":
            encoder_config = network_config.get('encoder', {})
            encoder_output_dim = encoder_config.get('output_dim', 32)
            
            if self.encoder_type == "lstm":
                self.encoder = LSTMEncoder(
                    input_dim=self.state_dim,
                    hidden_dim=encoder_config.get('hidden_dim', 64),
                    output_dim=encoder_output_dim,
                    num_layers=encoder_config.get('num_layers', 2)
                )
            else:  # transformer
                self.encoder = TransformerEncoder(
                    input_dim=self.state_dim,
                    d_model=encoder_config.get('d_model', 64),
                    output_dim=encoder_output_dim,
                    nhead=encoder_config.get('nhead', 4),
                    num_layers=encoder_config.get('num_layers', 2)
                )
            
            policy_input_dim = encoder_output_dim
        else:
            self.encoder = None
            policy_input_dim = self.state_dim
        
        # Gaussian Quantum Actor
        n_qubits = quantum_config.get('n_qubits', 2)
        n_layers = quantum_config.get('n_layers', 4)
        encoder_hidden = network_config.get('encoder', {}).get('hidden_dims', [64, 32])
        
        self.actor = GaussianQuantumPolicy(
            state_dim=policy_input_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            encoder_hidden=encoder_hidden,
            action_scale=self.action_scale,
            device_name=quantum_config.get('device', 'default.qubit'),
            seed=self.seed
        )
        
        # Value Network (Critic)
        critic_config = network_config.get('critic', {})
        self.critic = CriticNetwork(
            state_dim=policy_input_dim,
            action_dim=0,  # V(s) not Q(s,a)
            hidden_dims=critic_config.get('hidden_dims', [256, 256])
        )
        
        # BIS Predictor (Formulation 48)
        predictor_config = network_config.get('predictor', {})
        self.bis_predictor = BISPredictor(
            state_dim=policy_input_dim,
            action_dim=self.action_dim,
            hidden_dims=predictor_config.get('hidden_dims', [64, 32])
        )
    
    def _build_optimizer(self, training_config: Dict):
        """Build optimizer for all networks."""
        lr = training_config.get('lr', 0.0003)
        
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.encoder is not None:
            params += list(self.encoder.parameters())
        params += list(self.bis_predictor.parameters())
        
        self.optimizer = optim.Adam(params, lr=lr)
    
    def _encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state through temporal encoder if available."""
        if self.encoder is not None:
            if isinstance(self.encoder, LSTMEncoder):
                encoded, _ = self.encoder(state)
            else:
                encoded = self.encoder(state)
            return encoded
        return state
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action given state.
        
        Args:
            state: Current state
            deterministic: If True, return mean action
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            # Encode state
            encoded_state = self._encode_state(state_tensor)
            
            # Get value
            value = self.critic(encoded_state).item()
            
            # Get action
            action, log_prob, _ = self.actor.get_action(encoded_state.numpy(), deterministic)
        
        return action, log_prob, value
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Store transition in rollout buffer."""
        # Normalize action for storage
        action_normalized = action / self.action_scale
        
        self.rollout_buffer.push(
            state=state,
            action=action_normalized,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value
        )
        
        self.total_steps += 1
    
    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Perform PPO update - Formulations (41)-(49).
        
        Args:
            last_value: Bootstrap value for last state
        
        Returns:
            Dictionary of training metrics
        """
        # Compute advantages using GAE - Formulation (46)
        self.rollout_buffer.compute_advantages(last_value)
        
        # Training metrics
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0
        
        # Multiple epochs of updates
        for epoch in range(self.n_epochs):
            batches = self.rollout_buffer.get_batches(self.batch_size)
            
            for batch in batches:
                states = batch['states']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']
                
                # Encode states
                encoded_states = self._encode_state(states)
                
                # Get current policy log probs and entropy
                # Formulation (45): ζ = π(a|s) / π_old(a|s)
                new_log_probs, entropy = self.actor.evaluate_actions(
                    encoded_states, 
                    actions * self.action_scale
                )
                
                # Policy ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Formulation (42): Clipped PPO objective
                # L^PPO = min(ζ·Â, clip(ζ, 1-ε, 1+ε)·Â)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Formulation (43): Value loss
                # L^Val = ‖R_t - V(s_t)‖²
                values = self.critic(encoded_states).squeeze()
                critic_loss = F.mse_loss(values, returns)
                
                # Entropy bonus - Formulation (44)
                entropy_loss = -entropy.mean()
                
                # BIS prediction loss - Formulation (48)
                # (In practice, this needs next_bis which we don't have in this batch)
                # We'll skip it here and add it in the training loop if needed
                
                # Formulation (41): Combined loss
                # G^RL = c₁·G^PPO + c₂·G^Val - c₃·H(a)
                loss = (
                    self.c1 * actor_loss +
                    self.c2 * critic_loss +
                    self.c3 * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # Compute approximate KL divergence for logging
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                
                # Accumulate metrics
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                n_updates += 1
        
        # Clear buffer for next rollout
        self.rollout_buffer.reset()
        self.update_count += 1
        
        metrics = {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'approx_kl': total_approx_kl / n_updates
        }
        
        return metrics
    
    def train_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ) -> Dict[str, float]:
        """
        Perform a single training step (store transition, update if buffer full).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            log_prob: Log probability of action
            value: Value estimate
        
        Returns:
            Training metrics dictionary (empty if no update)
        """
        self.store_transition(state, action, reward, done, log_prob, value)
        
        metrics = {}
        if self.rollout_buffer.is_full():
            # Get bootstrap value
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                encoded_next = self._encode_state(next_state_tensor)
                last_value = self.critic(encoded_next).item() if not done else 0.0
            
            metrics = self.update(last_value)
        
        return metrics
    
    def save(self, path: str):
        """Save agent state."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'bis_predictor_state_dict': self.bis_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'config': self.config
        }
        
        if self.encoder is not None:
            checkpoint['encoder_state_dict'] = self.encoder.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.bis_predictor.load_state_dict(checkpoint['bis_predictor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.update_count = checkpoint['update_count']
        
        if self.encoder is not None and 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    def get_quantum_info(self) -> Dict:
        """Get information about the quantum circuit."""
        return self.actor.quantum_policy.get_quantum_circuit_info()
    
    def set_eval_mode(self):
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        if self.encoder is not None:
            self.encoder.eval()
    
    def set_train_mode(self):
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()
        if self.encoder is not None:
            self.encoder.train()


if __name__ == "__main__":
    # Test the PPO agent
    print("Testing QuantumPPOAgent...")
    
    # Create agent
    agent = QuantumPPOAgent(
        state_dim=8,
        action_dim=1,
        seed=42,
        encoder_type="none"
    )
    
    print(f"Agent created")
    print(f"Quantum circuit info: {agent.get_quantum_info()}")
    
    # Test action selection
    state = np.random.randn(8)
    action, log_prob, value = agent.select_action(state)
    print(f"\nState shape: {state.shape}")
    print(f"Action: {action}, Log prob: {log_prob:.4f}, Value: {value:.4f}")
    
    # Test rollout collection
    print("\nTesting rollout collection...")
    for step in range(100):
        state = np.random.randn(8)
        action, log_prob, value = agent.select_action(state)
        next_state = np.random.randn(8)
        reward = -np.random.rand()
        done = step == 99
        
        metrics = agent.train_step(
            state, np.array([action]), reward, next_state, done, log_prob, value
        )
        
        if metrics:
            print(f"Update {agent.update_count}: {metrics}")
    
    # Test save/load
    agent.save("/tmp/test_ppo_agent.pt")
    agent.load("/tmp/test_ppo_agent.pt")
    print("\nSave/Load test passed!")
    
    # Test with LSTM encoder
    print("\n" + "="*50)
    print("Testing with LSTM encoder...")
    agent_lstm = QuantumPPOAgent(
        state_dim=8,
        action_dim=1,
        seed=42,
        encoder_type="lstm"
    )
    
    state = np.random.randn(8)
    action, log_prob, value = agent_lstm.select_action(state)
    print(f"LSTM Agent - Action: {action:.4f}")
    
    print("\nTest complete!")
