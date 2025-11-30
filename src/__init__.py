"""
Quantum Reinforcement Learning for Propofol Infusion Control
============================================================

This package implements a Quantum RL system for closed-loop BIS-guided
propofol infusion control, based on the CBIM paper with Quantum enhancement
using PennyLane.

Modules:
    - environment: Patient simulator and Gym-compatible environment
    - models: VQC (Variational Quantum Circuit) and classical networks
    - agents: Quantum RL agents (DDPG-style)
    - utils: Metrics, visualization, and utilities
"""

__version__ = "0.1.0"
__author__ = "QRL Propofol Team"
