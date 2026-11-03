"""Effect-site target-controlled infusion (TCI) baseline (v3, D16).

A clinical TCI pump integrates the covariate-only population model of each drug
(Schnider for propofol, Minto for remifentanil), driven by the infusion it has
delivered, and sets the rate that brings the predicted effect-site concentration
to its target (Shafer & Gregg 1992). It does not read BIS.

At each control decision the pump predicts the effect-site concentration from
its model state and chooses the rate that, held constant, tracks the target
best over the next ``horizon_minutes`` (least squares). The rate is limited so
that this decision's infusion alone cannot push the predicted peak above the
target even if the pump stopped afterwards, and is clipped to the action range.
The action range caps propofol at 12 mg/kg/h, so induction is slower than with
a clinical TCI bolus.

History: v1 read state indices 3 and 6 (dBIS/dt and the propofol dose); the v2
replacement was proportional to the concentration shortfall and stopped at the
target, so it cycled between full and zero infusion.
"""
from typing import Tuple

import numpy as np
from scipy.linalg import expm

from models.pharmacokinetics import MintoModel, PatientParameters, SchniderModel

# BIS 50 in the population Bouillon model (E0 97.4, gamma 1.43, C50 4.47 ug/mL
# and 19.3 ng/mL) with remifentanil at a typical 4 ng/mL.
CE_TARGET_PPF = 3.38      # ug/mL
CE_TARGET_RFTN = 4.0      # ng/mL


def _discrete_system(params, weight: float, minutes: float) -> Tuple[np.ndarray, np.ndarray]:
    """Exact discretisation of PatientSimulator._two_drug_derivatives for one drug.

    State [C1, C2, C3, Ce]. The input is in ug/kg/min (propofol) or ng/kg/min
    (remifentanil), so concentrations are in ug/mL or ng/mL.
    """
    p = params
    a = np.array([
        [-(p.k10 + p.k12 + p.k13), p.k21 * p.V2 / p.V1, p.k31 * p.V3 / p.V1, 0.0],
        [p.k12 * p.V1 / p.V2, -p.k21, 0.0, 0.0],
        [p.k13 * p.V1 / p.V3, 0.0, -p.k31, 0.0],
        [p.ke0, 0.0, 0.0, -p.ke0],
    ])
    b = np.array([weight / (p.V1 * 1000.0), 0.0, 0.0, 0.0])
    augmented = np.zeros((5, 5))
    augmented[:4, :4] = a
    augmented[:4, 4] = b
    transition = expm(augmented * minutes)
    return transition[:4, :4], transition[:4, 4]


class _PumpModel:
    """Population model of one drug on a fixed prediction grid."""

    def __init__(self, params, weight, grid_minutes, steps_per_decision,
                 horizon_steps, peak_steps, rate_max):
        self.phi, self.gamma = _discrete_system(params, weight, grid_minutes)
        self.steps_per_decision = steps_per_decision
        self.horizon_steps = horizon_steps
        self.peak_steps = peak_steps
        self.rate_max = rate_max
        self.state = np.zeros(4)
        n = max(horizon_steps, peak_steps)
        # Effect-site response to a unit rate held throughout, and held for one
        # decision only.
        self.held = self._effect_site(np.zeros(4), 1.0, n, hold=n)[:horizon_steps]
        self.pulse = self._effect_site(np.zeros(4), 1.0, n, hold=steps_per_decision)[:peak_steps]

    def _effect_site(self, state, rate, n, hold):
        ce = np.empty(n)
        for k in range(n):
            state = self.phi @ state + self.gamma * (rate if k < hold else 0.0)
            ce[k] = state[3]
        return ce

    def rate_for(self, target: float) -> float:
        free = self._effect_site(self.state, 0.0, max(self.horizon_steps, self.peak_steps), 0)
        tracking = (np.sum(self.held * (target - free[:self.horizon_steps]))
                    / np.sum(self.held ** 2))
        no_overshoot = np.min((target - free[:self.peak_steps]) / self.pulse)
        return float(np.clip(min(tracking, no_overshoot), 0.0, self.rate_max))

    def advance(self, rate: float) -> None:
        for _ in range(self.steps_per_decision):
            self.state = self.phi @ self.state + self.gamma * rate


class TCIAgent:
    """Population-model effect-site TCI for propofol and remifentanil.

    Args:
        ce_target_ppf: Propofol effect-site target (ug/mL).
        ce_target_rftn: Remifentanil effect-site target (ng/mL).
        control_interval_seconds: Time each returned rate is held.
        grid_seconds: Prediction grid; must divide the control interval.
        horizon_minutes: Least-squares tracking horizon.
        peak_window_minutes: Window searched for the predicted peak.
        ppf_max: Propofol ceiling (mg/kg/h).
        rftn_max: Remifentanil ceiling (ug/kg/min).
    """

    def __init__(self, ce_target_ppf: float = CE_TARGET_PPF,
                 ce_target_rftn: float = CE_TARGET_RFTN,
                 control_interval_seconds: int = 60, grid_seconds: int = 10,
                 horizon_minutes: float = 5.0, peak_window_minutes: float = 10.0,
                 ppf_max: float = 12.0, rftn_max: float = 2.0):
        if control_interval_seconds % grid_seconds:
            raise ValueError("control interval must be a multiple of the prediction grid")
        self.ce_target_ppf = ce_target_ppf
        self.ce_target_rftn = ce_target_rftn
        self.control_interval_seconds = control_interval_seconds
        self.grid_seconds = grid_seconds
        self.horizon_minutes = horizon_minutes
        self.peak_window_minutes = peak_window_minutes
        self.ppf_max = ppf_max
        self.rftn_max = rftn_max
        self._pumps = None
        self._last_rates = None

    def set_patient(self, patient: PatientParameters) -> None:
        """Build the population pump models for ``patient``, starting drug-free."""
        grid = self.grid_seconds / 60.0
        steps = self.control_interval_seconds // self.grid_seconds
        horizon = int(round(self.horizon_minutes * 60 / self.grid_seconds))
        peak = int(round(self.peak_window_minutes * 60 / self.grid_seconds))
        self._pumps = (
            _PumpModel(SchniderModel(patient=patient).params, patient.weight, grid,
                       steps, horizon, peak, self.ppf_max * 1000.0 / 60.0),   # ug/kg/min
            _PumpModel(MintoModel(patient=patient).params, patient.weight, grid,
                       steps, horizon, peak, self.rftn_max * 1000.0),         # ng/kg/min
        )
        self._last_rates = None

    def reset(self) -> None:
        """Start the next episode drug-free."""
        if self._pumps is not None:
            for pump in self._pumps:
                pump.state = np.zeros(4)
        self._last_rates = None

    def reset_noise(self) -> None:
        """TCI is noise-free; present for interface compatibility."""

    @property
    def predicted_ce(self) -> np.ndarray:
        """Pump-model effect-site concentrations at the current decision."""
        if self._pumps is None:
            return np.zeros(2)
        return np.array([pump.state[3] for pump in self._pumps])

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return [propofol mg/kg/h, remifentanil ug/kg/min] for the next decision.

        ``state`` is not read: the pump knows only what it has infused.
        """
        if self._pumps is None:
            raise RuntimeError("TCIAgent needs set_patient(patient) before its first action")
        if self._last_rates is not None:
            for pump, rate in zip(self._pumps, self._last_rates):
                pump.advance(rate)
        ppf, rftn = (pump.rate_for(target) for pump, target in
                     zip(self._pumps, (self.ce_target_ppf, self.ce_target_rftn)))
        self._last_rates = (ppf, rftn)
        return np.array([ppf * 60.0 / 1000.0, rftn / 1000.0], dtype=np.float32)
