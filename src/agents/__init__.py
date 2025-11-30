"""
Agents module for Quantum RL Propofol Control
==============================================

Contains:
    - QuantumDDPGAgent: Hybrid Quantum-Classical DDPG agent
    - ReplayBuffer: Experience replay buffer
"""

from .quantum_agent import QuantumDDPGAgent, ReplayBuffer

__all__ = ["QuantumDDPGAgent", "ReplayBuffer"]
