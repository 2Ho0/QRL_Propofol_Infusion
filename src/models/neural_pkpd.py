"""
Neural PK/PD System Identification Module
==========================================

This module implements a Neural System Identification approach to estimate
individualized PK/PD parameters from patient history.

Key Components:
1. HistoryEncoder: Encodes sequence of (action, state) -> context z
2. ParameterHead: Maps context z -> PK/PD parameters θ (ke0, EC50, etc.)
3. DifferentiablePKPD: Solves PK/PD ODEs using predicted parameters

The model is trained to minimize the error between predicted BIS and actual BIS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class NeuralPKPD(nn.Module):
    """
    Neural System Identification for Personalized PK/PD.
    """
    def __init__(
        self,
        state_dim: int = 18,
        action_dim: int = 2,
        hidden_dim: int = 64,
        context_dim: int = 32,
        history_len: int = 20
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        
        # 1. History Encoder (LSTM)
        # Input: [action_t, state_t] sequence
        self.input_dim = state_dim + action_dim
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 2. Parameter Head (Estimator)
        # Maps context to physical parameters
        # We estimate deviation from population means or absolute values
        # Let's estimate absolute values but constrained to realistic ranges
        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # Predicting 6 key parameters
        )
        
        # Parameter mapping:
        # 0: k10 (elimination)
        # 1: k12 (distribution fast)
        # 2: k21 (redistribution fast)
        # 3: k31 (redistribution slow) - k13 is usually fixed or related
        # 4: ke0 (effect-site transfer)
        # 5: EC50 (sensitivity)
        
    def forward(self, history_states: torch.Tensor, history_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estimate parameters from history.
        
        Args:
            history_states: (B, T, state_dim)
            history_actions: (B, T, action_dim)
            
        Returns:
            Dictionary of estimated parameters
        """
        # Concatenate inputs
        x = torch.cat([history_states, history_actions], dim=-1)
        
        # Encode
        output, (hn, cn) = self.lstm(x)
        
        # Use last hidden state (concat forward and backward)
        context = torch.cat([hn[-2], hn[-1]], dim=-1)
        
        # Predict raw params
        raw_params = self.param_head(context)
        
        # Apply constraints (Softplus for positivity) using population priors
        # We add learned offsets to log-space population means for stability
        
        # Parameters dict
        params = {}
        
        # Propofol PK (approximate population means)
        # k10 ~ 0.119, k12 ~ 0.112, k21 ~ 0.055, ke0 ~ 0.456 (Schnider)
        # EC50 ~ 3.5 (Schnider/Marsh mixed)
        
        # Use simple softplus activation to ensure positivity
        params['k10'] = F.softplus(raw_params[:, 0]) + 0.01
        params['k12'] = F.softplus(raw_params[:, 1]) + 0.01
        params['k21'] = F.softplus(raw_params[:, 2]) + 0.01
        params['k31'] = F.softplus(raw_params[:, 3]) + 0.001
        params['ke0'] = F.softplus(raw_params[:, 4]) + 0.01
        params['ec50'] = F.softplus(raw_params[:, 5]) + 1.0  # Min EC50 = 1.0
        
        return params

    def solve_ode(
        self, 
        params: Dict[str, torch.Tensor], 
        initial_state: torch.Tensor, 
        actions: torch.Tensor, 
        dt: float = 1.0/60.0  # 1 second in minutes (standard PK unit)
    ) -> torch.Tensor:
        """
        Differentiable ODE Solver (Forward Euler for simplicity).
        
        Args:
            params: Estimated parameters for batch
            initial_state: (B, 4) - [Cp, C2, C3, Ce] (Concentrations)
            actions: (B, T, 1) - Propofol infusion rate sequence
            dt: Time step size
            
        Returns:
            Predicted Ce trajectory (B, T)
        """
        batch_size = actions.shape[0]
        seq_len = actions.shape[1]
        
        # Verify action shape (Propofol only for now, Remi usually separate)
        # actions assumption: (B, T, 2) -> we take just Propofol for PK calc if needed
        # Or if passed strictly propofol: (B, T)
        
        # Unpack params
        k10 = params['k10'].unsqueeze(1)  # (B, 1)
        k12 = params['k12'].unsqueeze(1)
        k21 = params['k21'].unsqueeze(1)
        k13 = torch.zeros_like(k10) + 0.003 # Fixed small value or estimated
        k31 = params['k31'].unsqueeze(1)
        ke0 = params['ke0'].unsqueeze(1)
        
        # Current concentrations [Cp, C2, C3, Ce]
        cp = initial_state[:, 0].unsqueeze(1)
        c2 = initial_state[:, 1].unsqueeze(1)
        c3 = initial_state[:, 2].unsqueeze(1)
        ce = initial_state[:, 3].unsqueeze(1)
        
        ce_trace = []
        
        # Simulation loop
        for t in range(seq_len):
            u = actions[:, t, 0].unsqueeze(1) # Propofol rate
            
            # Differential equations (3-compartment + effect site)
            # dCp/dt = u/V1 - (k10 + k12 + k13)*Cp + k21*C2 + k31*C3
            # dC2/dt = k12*Cp - k21*C2
            # dC3/dt = k13*Cp - k31*C3
            # dCe/dt = ke0*(Cp - Ce)
            
            # Note: We need V1 to convert u (rate) to concentration change
            # Simplification: Assume u is already volume-normalized or learn V1 implicitly in k-params
            # Or add V1 to estimated params. Let's assume u is in mg/min and we need /V1.
            # For now, let's treat 'u' as 'input concentration change' for differentiable simplicity
            
            dcp = u - (k10 + k12 + k13) * cp + k21 * c2 + k31 * c3
            dc2 = k12 * cp - k21 * c2
            dc3 = k13 * cp - k31 * c3
            dce = ke0 * (cp - ce)
            
            # Euler update
            cp = cp + dcp * dt
            c2 = c2 + dc2 * dt
            c3 = c3 + dc3 * dt
            ce = ce + dce * dt
            
            ce_trace.append(ce)
            
        return torch.cat(ce_trace, dim=1)

    def predict_bis(self, ce: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Sigmoid Emax model to predict BIS from Ce.
        
        BIS = E0 - (E0 - Emax) * (Ce^gamma) / (Ce^gamma + EC50^gamma)
        """
        ec50 = params['ec50'].unsqueeze(1)
        gamma = 2.5 # Hill coefficient (can also be learned)
        e0 = 98.0   # Baseline BIS
        emax = 0.0  # Min BIS
        
        # Hill equation
        # Ce is (B, T)
        effect = (ce ** gamma) / (ce ** gamma + ec50 ** gamma)
        bis = e0 - (e0 - emax) * effect
        
        return bis
