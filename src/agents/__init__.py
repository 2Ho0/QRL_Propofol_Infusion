"""
Agents module for Quantum RL Propofol Control
==============================================

Contains:
    - QuantumDDPGAgent: Hybrid Quantum-Classical DDPG agent
    - QuantumPPOAgent: Hybrid Quantum-Classical PPO agent
    - ReplayBuffer: Experience replay buffer for DDPG
    - SequenceReplayBuffer: Sequence-based buffer for temporal models
    - RolloutBuffer: On-policy buffer for PPO
    - EncoderType: Enum for encoder architecture selection

CBIM Paper Implementation:
--------------------------
Both DDPG and PPO are implemented following the paper's formulations:
- State representation: (36)-(39)
- Reward function: (40)
- PPO algorithm: (41)-(49)
- Temporal feature extraction: LSTM/Transformer (Fig.4)
"""

from .quantum_agent import (
    QuantumDDPGAgent, 
    ReplayBuffer, 
    SequenceReplayBuffer,
    EncoderType,
)

from .classical_agent import ClassicalDDPGAgent

__all__ = [
    "QuantumDDPGAgent", 
    "ClassicalDDPGAgent",
    "ReplayBuffer", 
    "SequenceReplayBuffer",
    "EncoderType",
]
