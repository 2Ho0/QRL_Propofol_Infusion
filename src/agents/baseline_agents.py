"""
Baseline Agents for Reviewer Comparison
========================================

Implements non-learning and classical control baselines as requested by reviewers:
1. ClinicalGuidelineAgent  - Rule-based dosing per clinical anesthesia protocols
2. PIDController           - Classic PID feedback controller for BIS tracking

Both agents share the same interface as RL agents (select_action) so they can be
plugged directly into evaluate_agent_on_simulator().
"""

import numpy as np
from collections import deque
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 1. Clinical Guideline Agent (Rule-based)
# ─────────────────────────────────────────────────────────────

class ClinicalGuidelineAgent:
    """
    Rule-based dosing agent based on published clinical anesthesia guidelines.

    Protocol (simplified from ESA/ASA guidelines):
    ─────────────────────────────────────────────
    Induction (BIS > 70):
        Propofol: 10-12 mg/kg/h (high induction dose)
        Remifentanil: 0.25-0.5 μg/kg/min (analgesic cover)

    Maintenance (BIS 40-60):
        Propofol: 4-8 mg/kg/h (titrate per BIS)
        Remifentanil: 0.05-0.25 μg/kg/min (maintenance)

    BIS-Guided Adjustment:
        BIS > 65 → increase propofol by 1 mg/kg/h
        BIS < 35 → decrease propofol by 1 mg/kg/h
        BIS 40-60 → maintain current dose

    Reference: Absalom et al. (2009), Anaesthesia guidelines for TIVA
    """

    def __init__(
        self,
        target_bis: float = 50.0,
        action_dim: int = 2,
        seed: Optional[int] = None,
    ):
        self.target_bis = target_bis
        self.action_dim = action_dim

        # Current infusion rates (physical units)
        self.current_ppf = 6.0    # mg/kg/h  (mid-range maintenance)
        self.current_rftn = 0.15  # μg/kg/min

        # State tracking
        self.prev_bis = None
        self.step_count = 0

        # Hysteresis to prevent rapid oscillation
        self.hold_steps = 0       # Steps to hold current dose before adjusting

        # Action bounds matching environment
        self.ppf_min, self.ppf_max = 0.0, 12.0
        self.rftn_min, self.rftn_max = 0.0, 2.0

        if seed is not None:
            np.random.seed(seed)

    def reset_noise(self):
        """Compatibility shim — no noise in rule-based agent."""
        pass

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Select action based on current BIS error (extracted from state[0]).

        State layout (from DualDrugEnv):
            state[0] = BIS_error = 50 - BIS  → BIS = 50 - state[0]
        """
        # Recover BIS from state
        bis_error = float(state[0])          # e_t = target - BIS
        current_bis = self.target_bis - bis_error

        self.step_count += 1

        # ── BIS-guided propofol adjustment ──
        if self.hold_steps > 0:
            self.hold_steps -= 1
        else:
            if current_bis > 65:               # Too light → increase propofol
                self.current_ppf = min(self.current_ppf + 1.5, self.ppf_max)
                self.hold_steps = 3
            elif current_bis > 60:
                self.current_ppf = min(self.current_ppf + 0.5, self.ppf_max)
                self.hold_steps = 2
            elif current_bis < 35:             # Too deep → decrease propofol
                self.current_ppf = max(self.current_ppf - 1.5, self.ppf_min)
                self.current_rftn = max(self.current_rftn - 0.05, self.rftn_min)
                self.hold_steps = 3
            elif current_bis < 40:
                self.current_ppf = max(self.current_ppf - 0.5, self.ppf_min)
                self.hold_steps = 2
            # else BIS 40-60: maintain current dose

        # ── Remifentanil maintained at low analgesic level during maintenance ──
        # Increase remifentanil slightly if BIS remains high (insufficient analgesia)
        if current_bis > 60 and self.step_count > 5:
            self.current_rftn = min(self.current_rftn + 0.02, 0.5)
        elif current_bis < 40:
            self.current_rftn = max(self.current_rftn - 0.02, self.rftn_min)

        self.prev_bis = current_bis

        return np.array([self.current_ppf, self.current_rftn], dtype=np.float32)

    def reset(self):
        """Reset agent state for a new episode."""
        self.current_ppf = 6.0
        self.current_rftn = 0.15
        self.prev_bis = None
        self.step_count = 0
        self.hold_steps = 0


# ─────────────────────────────────────────────────────────────
# 2. PID Controller
# ─────────────────────────────────────────────────────────────

class PIDController:
    """
    Classic PID feedback controller for BIS tracking.

    Independently controls both propofol and remifentanil using
    two decoupled PID loops on the common BIS error signal.

    Transfer function:
        u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de(t)/dt

    where e(t) = BIS_target − BIS(t)

    Gains are tuned heuristically for the Schnider/Minto PK-PD model
    with dt = 1 min.  Positive error (BIS too high) → increase drug.

    Propofol PID:   Kp=0.25, Ki=0.005, Kd=0.10
    Remifentanil:   Kp=0.02, Ki=0.001, Kd=0.005

    Anti-windup: integral clamped to ±20 for propofol, ±2 for remifentanil.
    """

    def __init__(
        self,
        target_bis: float = 50.0,
        action_dim: int = 2,
        # Propofol PID gains
        kp_ppf: float = 0.25,
        ki_ppf: float = 0.005,
        kd_ppf: float = 0.10,
        # Remifentanil PID gains
        kp_rftn: float = 0.02,
        ki_rftn: float = 0.001,
        kd_rftn: float = 0.005,
        base_ppf: float = 6.0,
        base_rftn: float = 0.1,
        dt: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.target_bis = target_bis
        self.action_dim = action_dim
        self.dt = dt

        # Propofol PID state
        self.kp_ppf = kp_ppf
        self.ki_ppf = ki_ppf
        self.kd_ppf = kd_ppf
        self._integral_ppf = 0.0
        self._prev_error = 0.0

        # Remifentanil PID state
        self.kp_rftn = kp_rftn
        self.ki_rftn = ki_rftn
        self.kd_rftn = kd_rftn
        self._integral_rftn = 0.0

        # Action bounds
        self.ppf_min, self.ppf_max = 0.0, 12.0
        self.rftn_min, self.rftn_max = 0.0, 2.0

        # Start with baseline infusion (safe induction level)
        self._base_ppf = float(base_ppf)
        self._base_rftn = float(base_rftn)

    def reset_noise(self):
        """Compatibility shim."""
        pass

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Compute PID output from BIS error in state[0].

        state[0] = BIS_error = target − BIS → positive means BIS too high.
        """
        # BIS_error from env is (target - BIS). 
        # For PID, we want positive error when BIS is too high (BIS > target), 
        # so we need to negate state[0].
        error = -float(state[0])   # error = BIS - target; positive → increase drug

        # ── Propofol PID ──
        d_error = (error - self._prev_error) / self.dt
        self._integral_ppf += error * self.dt
        self._integral_ppf = np.clip(self._integral_ppf, -20.0, 20.0)  # anti-windup

        ppf_delta = (
            self.kp_ppf * error
            + self.ki_ppf * self._integral_ppf
            + self.kd_ppf * d_error
        )
        ppf_rate = np.clip(self._base_ppf + ppf_delta, self.ppf_min, self.ppf_max)

        # ── Remifentanil PID (same error, smaller gains) ──
        self._integral_rftn += error * self.dt
        self._integral_rftn = np.clip(self._integral_rftn, -2.0, 2.0)

        rftn_delta = (
            self.kp_rftn * error
            + self.ki_rftn * self._integral_rftn
            + self.kd_rftn * d_error
        )
        rftn_rate = np.clip(self._base_rftn + rftn_delta, self.rftn_min, self.rftn_max)

        self._prev_error = error

        return np.array([ppf_rate, rftn_rate], dtype=np.float32)

    def reset(self):
        """Reset integrators for new episode."""
        self._integral_ppf = 0.0
        self._integral_rftn = 0.0
        self._prev_error = 0.0
