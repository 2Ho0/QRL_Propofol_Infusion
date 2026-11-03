"""Locked, clinically motivated stress extension of :mod:`dual_drug_env`.

This environment is deliberately labelled a synthetic control benchmark.  It
adds phase-varying targets, remifentanil-modulated arousal events, hidden PD/PK
mismatch, and a delayed/imperfect BIS sensor while preserving the 13-element
observation and two physical action channels used by the frozen policies.

The numerical protocol is recorded before outcomes in
``logs/challenging_benchmark/locked_protocol.json``.  The additions are not a
validated hemodynamic or nociception model and must not be interpreted as one.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from environment.dual_drug_env import DualDrugEnv
from models.pharmacokinetics import PatientParameters


TARGET_SCHEDULE: Tuple[Tuple[int, int, float], ...] = (
    (0, 20, 55.0),
    (20, 160, 50.0),
    (160, 180, 55.0),
    (180, 200, 65.0),
)
EVENT_CENTRES = (35, 80, 125)
EVENT_DURATIONS = (8, 12, 10)


class ChallengingDualDrugEnv(DualDrugEnv):
    """Dual-drug environment with a deterministic patient-seeded scenario.

    ``scenario_seed`` controls every added perturbation.  Passing the same
    value for a patient to every controller implements common random numbers.
    The controller observes delayed/noisy ``bis`` while outcome metrics should
    use ``true_bis`` and ``active_target`` from ``info``.
    """

    def __init__(
        self,
        patient: PatientParameters,
        scenario_seed: int,
        max_steps: int = 200,
        dt: float = 1.0,
    ) -> None:
        self.scenario_seed = int(scenario_seed)
        self._pd_factors_applied = False
        self.pd_factors: Dict[str, float] = {}
        self.events: List[Dict[str, float]] = []
        self.true_bis = 98.0
        self.outcome_target = self.target_for_step(0)
        self.raw_stimulation = 0.0
        self.residual_stimulation = 0.0
        self.sensor_dropout = False
        self.sensor_artifact = 0.0
        self._sensor_noise = np.zeros(max_steps, dtype=float)
        self._sensor_dropout = np.zeros(max_steps, dtype=bool)
        self._sensor_artifact = np.zeros(max_steps, dtype=float)
        self._sensor_queue: deque[float] = deque(maxlen=3)

        # Parent sensor noise is disabled; the complete pre-generated sensor
        # process below is action-independent and shared across methods.
        super().__init__(
            target_bis=self.target_for_step(0),
            bis_noise_std=0.0,
            max_steps=max_steps,
            dt=dt,
            patient=patient,
            seed=self.scenario_seed,
            reward_type="potential",
        )
        self._apply_hidden_model_mismatch()
        self.reset()

    @staticmethod
    def target_for_step(step: int) -> float:
        """Return the phase-specific target at a zero-based control step."""
        for start, stop, target in TARGET_SCHEDULE:
            if start <= int(step) < stop:
                return target
        return TARGET_SCHEDULE[-1][2]

    def _apply_hidden_model_mismatch(self) -> None:
        if self._pd_factors_applied:
            return
        rng = np.random.default_rng(self.scenario_seed + 17)
        factors = {
            "c50_ppf": float(rng.uniform(0.80, 1.25)),
            "c50_rftn": float(rng.uniform(0.75, 1.30)),
            "gamma": float(rng.uniform(0.90, 1.10)),
            "ke0_ppf": float(rng.uniform(0.80, 1.20)),
            "ke0_rftn": float(rng.uniform(0.80, 1.20)),
        }
        interaction = self.simulator.drug_interaction_params
        interaction.c50_ppf *= factors["c50_ppf"]
        interaction.c50_rftn *= factors["c50_rftn"]
        interaction.gamma *= factors["gamma"]
        self.simulator.ppf_model.params.ke0 *= factors["ke0_ppf"]
        self.simulator.rftn_model.params.ke0 *= factors["ke0_rftn"]
        self.pd_factors = factors
        self._pd_factors_applied = True

    def _make_scenario(self) -> None:
        event_rng = np.random.default_rng(self.scenario_seed + 23)
        jitters = event_rng.integers(-3, 4, size=len(EVENT_CENTRES))
        amplitudes = event_rng.uniform(8.0, 14.0, size=len(EVENT_CENTRES))
        self.events = []
        for centre, duration, jitter, amplitude in zip(
            EVENT_CENTRES, EVENT_DURATIONS, jitters, amplitudes
        ):
            start = int(centre + jitter)
            self.events.append({
                "start": start,
                "stop": start + int(duration),
                "duration": int(duration),
                "amplitude": float(amplitude),
            })

        sensor_rng = np.random.default_rng(self.scenario_seed + 31)
        self._sensor_noise = sensor_rng.normal(0.0, 3.5, self.max_steps)
        self._sensor_dropout = sensor_rng.random(self.max_steps) < 0.05
        artifact_on = sensor_rng.random(self.max_steps) < 0.02
        artifact = np.clip(sensor_rng.normal(0.0, 8.0, self.max_steps), -15.0, 15.0)
        self._sensor_artifact = np.where(artifact_on, artifact, 0.0)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.target_bis = self.target_for_step(0)
        obs, info = super().reset(seed=seed, options=options)
        self._make_scenario()

        self.true_bis = float(self.simulator._compute_bis(
            self.simulator.state_ppf[3], self.simulator.state_rftn[3]))
        self.outcome_target = self.target_for_step(0)
        self.bis = self.true_bis
        self.bis_history = [self.bis]
        self.bis_window.clear()
        self.bis_window.append(self.bis)
        self._sensor_queue = deque([self.true_bis] * 3, maxlen=3)
        self.raw_stimulation = 0.0
        self.residual_stimulation = 0.0
        self.sensor_dropout = False
        self.sensor_artifact = 0.0
        self.episode_history.update({
            "true_bis": [],
            "observed_bis": [],
            "target_bis": [],
            "raw_stimulation": [],
            "residual_stimulation": [],
            "sensor_dropout": [],
            "sensor_artifact": [],
        })
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _stimulation_at(self, step: int) -> float:
        for event in self.events:
            if int(event["start"]) <= step < int(event["stop"]):
                return float(event["amplitude"])
        return 0.0

    def step(self, action: np.ndarray):
        step_index = int(self.current_step)
        self.outcome_target = self.target_for_step(step_index)
        self.target_bis = self.outcome_target
        previous_observed_bis = float(self.bis)

        _, _, terminated, truncated, _ = super().step(action)
        done = bool(terminated or truncated)

        base_true_bis = float(self.simulator._compute_bis(
            self.simulator.state_ppf[3], self.simulator.state_rftn[3]))
        self.raw_stimulation = self._stimulation_at(step_index)
        ce_remi = max(0.0, float(self.simulator.state_rftn[3]))
        self.residual_stimulation = self.raw_stimulation / (1.0 + ce_remi / 4.0)
        self.true_bis = float(np.clip(
            base_true_bis + self.residual_stimulation, 0.0, 100.0))

        self.sensor_artifact = float(self._sensor_artifact[step_index])
        current_sensor = float(np.clip(
            self.true_bis + self._sensor_noise[step_index] + self.sensor_artifact,
            0.0,
            100.0,
        ))
        self._sensor_queue.append(current_sensor)
        delayed_sensor = float(self._sensor_queue[0])
        self.sensor_dropout = bool(self._sensor_dropout[step_index])
        self.bis = previous_observed_bis if self.sensor_dropout else delayed_sensor

        # Replace the parent's undisturbed observation/reward history with the
        # sensor-facing value.  Latent outcome traces remain separate.
        self.bis_history[-1] = self.bis
        self.bis_window[-1] = self.bis
        self.episode_history["bis"][-1] = self.bis
        reward = self._compute_reward(
            float(action[0]), float(action[1]), previous_observed_bis, done=done)
        self.episode_history["reward"][-1] = reward
        self.episode_history["true_bis"].append(self.true_bis)
        self.episode_history["observed_bis"].append(self.bis)
        self.episode_history["target_bis"].append(self.outcome_target)
        self.episode_history["raw_stimulation"].append(self.raw_stimulation)
        self.episode_history["residual_stimulation"].append(
            self.residual_stimulation)
        self.episode_history["sensor_dropout"].append(self.sensor_dropout)
        self.episode_history["sensor_artifact"].append(self.sensor_artifact)

        # Expose the target for the upcoming transition in the returned state;
        # retain ``outcome_target`` for scoring the just-completed transition.
        self.target_bis = self.target_for_step(self.current_step)
        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_info(self) -> dict:
        info = super()._get_info()
        info.update({
            "true_bis": float(self.true_bis),
            "observed_bis": float(self.bis),
            "active_target": float(self.target_bis),
            "outcome_target": float(self.outcome_target),
            "raw_stimulation": float(self.raw_stimulation),
            "residual_stimulation": float(self.residual_stimulation),
            "sensor_dropout": bool(self.sensor_dropout),
            "sensor_artifact": float(self.sensor_artifact),
            "scenario_seed": self.scenario_seed,
            "pd_factors": dict(self.pd_factors),
        })
        return info
