"""Effect-site TCI baseline with a population pump model (v3, D16).

A clinical TCI pump integrates the covariate-only population model and sets the
rate that brings the predicted effect-site concentration to its target without
overshoot (Shafer & Gregg 1992). v1 read the wrong state indices; the v2
proportional pump stopped at the target and restarted below it, so it cycled
between full and zero infusion (test MDAPE 29.1, BIS > 70 for 25.9% of the time).
"""
import numpy as np
import pytest

from agents.tci_agent import TCIAgent
from environment.dual_drug_env import DualDrugEnv
from environment.variability import get_variability_spec
from evaluation.per_patient_eval import evaluate_patient
from models.pharmacokinetics import PatientParameters

PATIENT = PatientParameters(age=50, weight=70.0, height=170, gender="male")
PER_DECISION = 6


def _env(variability):
    return DualDrugEnv(patient=PATIENT, seed=5, dt=10 / 60, max_steps=6000,
                       shaping_interval_steps=PER_DECISION, variability=variability)


def _run(agent, env, decisions):
    """Return per-step plant Ce and per-decision (state, predicted Ce, action)."""
    state, _ = env.reset()
    agent.set_patient(env.patient_obj)
    plant, decisions_log = [], []
    for _ in range(decisions):
        action = agent.select_action(state)
        decisions_log.append((state.copy(), agent.predicted_ce.copy(), action.copy()))
        for _ in range(PER_DECISION):
            state, *_ = env.step(action)
            plant.append((env.simulator.state_ppf[3], env.simulator.state_rftn[3]))
    return np.array(plant), decisions_log


def test_acting_before_set_patient_is_refused():
    with pytest.raises(RuntimeError, match="set_patient"):
        TCIAgent().select_action(np.zeros(13))


def test_pump_model_tracks_the_simulators_population_observer():
    _, log = _run(TCIAgent(), _env(get_variability_spec("literature")), decisions=40)
    for state, predicted, _ in log:
        np.testing.assert_allclose(predicted, state[1:3], rtol=1e-5, atol=1e-5)


def test_population_patient_reaches_and_holds_the_target_without_overshoot():
    agent = TCIAgent()
    plant, log = _run(agent, _env(None), decisions=90)    # plant = population model
    ratio = plant / np.array([agent.ce_target_ppf, agent.ce_target_rftn])
    assert ratio.max() <= 1.001
    assert np.all(np.abs(ratio[20 * PER_DECISION:] - 1.0) < 0.02)    # from 20 min
    maintenance_ppf = np.array([action[0] for _, _, action in log[30:]])
    assert maintenance_ppf.std() < 0.5        # no full/zero cycling (v2: about 5)


def test_default_targets_give_bis_50_in_the_population_model():
    agent = TCIAgent()
    bis = _env(None).simulator._compute_bis(agent.ce_target_ppf, agent.ce_target_rftn)
    assert bis == pytest.approx(50.0, abs=0.05)


def test_set_patient_empties_the_pump_model():
    agent = TCIAgent()
    _run(agent, _env(None), decisions=5)
    agent.set_patient(PatientParameters(age=70, weight=50.0, height=155, gender="female"))
    np.testing.assert_array_equal(agent.predicted_ce, [0.0, 0.0])
    assert agent.select_action(np.zeros(13))[0] == pytest.approx(12.0)   # induction at the cap


def test_evaluate_patient_tells_the_agent_the_patient():
    class _Recorder:
        sequence_length = 1
        seen = None

        def set_patient(self, patient):
            _Recorder.seen = patient

        def select_action(self, state, deterministic=True):
            return np.array([6.0, 0.1])

    evaluate_patient(_Recorder(), PATIENT, seed=1, steps=2, observation_interval_seconds=10,
                     control_interval_seconds=60, variability=None)
    assert _Recorder.seen is PATIENT
