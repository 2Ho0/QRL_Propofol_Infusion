"""Per-patient evaluation producing the results contract.

The unit of analysis is one held-out patient: that patient's real demographics
parameterise the Schnider/Minto simulator, one episode is run per seed, and
seeds are averaged within the patient by the analysis step. This replaces
evaluation over synthetically sampled demographics, which never touched a
held-out patient at all.

Metrics follow Varvel et al.: with performance error
PE_t = 100 (BIS_t - 50) / 50,

    MDPE       median PE_t                       (bias)
    MDAPE      median |PE_t|                     (inaccuracy)
    wobble     median |PE_t - MDPE|              (intra-patient variability)
    divergence slope of |PE_t| against time      (drift, %/min)
"""
from collections import deque
import inspect
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from environment.dual_drug_env import DualDrugEnv
from environment.patient_simulator import PatientParameters
from evaluation.results_contract import PatientResult

BIS_TARGET = 50.0
EPISODE_STEPS = 200          # 200 steps x 1 min = the 200-minute protocol episode


def patient_from_record(rec: Dict) -> PatientParameters:
    """Build simulator patient parameters from a clinical record.

    Args:
        rec: Mapping with age, sex, height, weight. `sex` may be 'M'/'F' or a
            number, where >= 0.5 means male, matching the loader's encoding.

    Returns:
        PatientParameters for the Schnider/Minto models.
    """
    sex = rec["sex"]
    male = (str(sex).upper().startswith("M") if isinstance(sex, str)
            else float(sex) >= 0.5)
    return PatientParameters(
        age=float(rec["age"]),
        weight=float(rec["weight"]),
        height=float(rec["height"]),
        gender="male" if male else "female",
    )


def varvel_metrics(bis: np.ndarray, dt_min: float = 1.0) -> Dict[str, float]:
    """Varvel performance measures for one episode.

    Args:
        bis: BIS trace for the episode.
        dt_min: Control interval in minutes, used for the divergence slope.

    Returns:
        Dict with mdpe, mdape, wobble and divergence.
    """
    pe = 100.0 * (bis - BIS_TARGET) / BIS_TARGET
    mdpe = float(np.median(pe))
    mdape = float(np.median(np.abs(pe)))
    wobble = float(np.median(np.abs(pe - mdpe)))

    t = np.arange(len(pe), dtype=float) * dt_min
    # Divergence is the regression slope of |PE| on time, in %/min.
    divergence = float(np.polyfit(t, np.abs(pe), 1)[0]) if len(pe) > 1 else 0.0
    return {"mdpe": mdpe, "mdape": mdape, "wobble": wobble, "divergence": divergence}


def evaluate_patient(
    agent,
    patient: PatientParameters,
    seed: int,
    reward_type: str = "potential",
    smoothing_beta: float = 0.0,
    steps: int = EPISODE_STEPS,
    observation_interval_seconds: int = 60,
    control_interval_seconds: int = 60,
    *,
    variability,
    env_kwargs: Optional[Dict] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Run one episode for one patient and return metrics and traces.

    Args:
        agent: Anything exposing `select_action(state, deterministic=True)`.
        patient: Simulator parameters for this patient.
        seed: Seed for the environment.
        reward_type: Reward variant, passed to the environment.
        smoothing_beta: EMA action-smoothing factor, 0 to disable.
        steps: Episode length in control steps.
        variability: Plant variability spec (sim v2) or None for the v1
            plant. Required, so that no caller silently evaluates a
            different plant from the one it trained on.
        env_kwargs: Extra DualDrugEnv settings (v3 residual mode); the
            recorded doses are those the environment applied.

    Returns:
        (metrics, bis trace, propofol trace, remifentanil trace).
    """
    if control_interval_seconds % observation_interval_seconds:
        raise ValueError("control interval must be divisible by observation interval")
    observations_per_control = (
        control_interval_seconds // observation_interval_seconds)
    env = DualDrugEnv(
        patient=patient,
        seed=seed,
        reward_type=reward_type,
        dt=observation_interval_seconds / 60.0,
        max_steps=steps * observations_per_control,
        shaping_interval_steps=observations_per_control,
        variability=variability,
        **(env_kwargs or {}),
    )
    state, _ = env.reset()

    # ``torch.no_grad`` in an agent's ``select_action`` is not sufficient to
    # make a policy deterministic: Transformer/LSTM encoders may still have
    # dropout enabled while their modules are in training mode.  Put all
    # policy-side modules in evaluation mode for the episode, then restore the
    # caller's state so this helper is also safe during online training.
    policy_modules = []
    for name in ("actor", "encoder"):
        module = getattr(agent, name, None)
        if module is not None and hasattr(module, "training"):
            policy_modules.append((module, bool(module.training)))
            module.eval()

    # Model-based baselines (the TCI pump) need the patient's covariates.
    if hasattr(agent, "set_patient"):
        agent.set_patient(patient)
    if hasattr(agent, "reset"):
        agent.reset()
    if hasattr(agent, "reset_noise"):
        agent.reset_noise()

    bis_hist, bis_true_hist, ppf_hist, rftn_hist = [], [], [], []
    prev_action = None
    sequence_length = int(getattr(agent, "sequence_length", 1))
    observation_history = deque(maxlen=sequence_length)
    observation_history.append(np.asarray(state, dtype=np.float32).copy())
    supports_explicit_sequence = (
        "state_sequence" in inspect.signature(agent.select_action).parameters)

    def current_sequence() -> np.ndarray:
        observed = list(observation_history)
        return np.asarray(
            [observed[0]] * (sequence_length - len(observed)) + observed,
            dtype=np.float32,
        )

    try:
        for _ in range(steps):
            if (getattr(agent, "encoder", None) is not None
                    and supports_explicit_sequence):
                action = agent.select_action(
                    state, deterministic=True, state_sequence=current_sequence())
            else:
                action = agent.select_action(state, deterministic=True)
            action = np.asarray(action, dtype=np.float64).ravel()[:2]

            if smoothing_beta > 0.0:
                prev_action = action if prev_action is None else (
                    smoothing_beta * prev_action + (1.0 - smoothing_beta) * action)
                action = prev_action

            episode_done = False
            for _ in range(observations_per_control):
                state, _, done, truncated, info = env.step(action)
                observation_history.append(
                    np.asarray(state, dtype=np.float32).copy())
                bis_hist.append(float(env.bis))
                bis_true_hist.append(float(env.simulator.get_bis(add_noise=False)))
                # The dose the environment applied (after the v3 residual and
                # clipping), which equals the command for a plain environment.
                applied = info["applied_action"]
                ppf_hist.append(float(applied[0]))
                rftn_hist.append(float(applied[1]))
                if done or truncated:
                    episode_done = True
                    break

            if episode_done:
                break
    finally:
        for module, was_training in policy_modules:
            module.train(was_training)

    bis = np.asarray(bis_hist)
    bis_true = np.asarray(bis_true_hist)
    ppf = np.asarray(ppf_hist)
    rftn = np.asarray(rftn_hist)
    dt_min = observation_interval_seconds / 60.0
    metrics = varvel_metrics(bis, dt_min=dt_min)
    metrics.update({
        f"{name}_true": value
        for name, value in varvel_metrics(bis_true, dt_min=dt_min).items()})
    metrics["bis_true_trace"] = bis_true
    return metrics, bis, ppf, rftn


def evaluate_cohort(
    agent,
    method: str,
    cohort: Sequence[Tuple[int, PatientParameters]],
    seed: int,
    environment_seed: Optional[int] = None,
    reward_type: str = "potential",
    smoothing_beta: float = 0.0,
    progress: Optional[object] = None,
    observation_interval_seconds: int = 60,
    control_interval_seconds: int = 60,
    *,
    variability,
    env_kwargs: Optional[Dict] = None,
) -> List[PatientResult]:
    """Evaluate one method over a held-out cohort, one row per patient.

    Args:
        agent: Policy or baseline controller.
        method: Arm name recorded in the contract.
        cohort: (caseid, PatientParameters) pairs for the held-out patients.
        seed: Training/run seed recorded in the results contract.
        environment_seed: Optional seed controlling simulator randomness.  Set
            this to the same value across training seeds to use common random
            numbers on the held-out cohort.  Defaults to ``seed`` for backward
            compatibility.
        reward_type: Reward variant.
        smoothing_beta: EMA action-smoothing factor.
        progress: Optional tqdm-like iterable wrapper.

    Returns:
        One PatientResult per patient.
    """
    it = cohort if progress is None else progress(cohort)
    rows: List[PatientResult] = []

    simulator_seed = seed if environment_seed is None else environment_seed
    for caseid, patient in it:
        m, bis, ppf, rftn = evaluate_patient(
            agent, patient, seed=simulator_seed + int(caseid),
            reward_type=reward_type, smoothing_beta=smoothing_beta,
            observation_interval_seconds=observation_interval_seconds,
            control_interval_seconds=control_interval_seconds,
            variability=variability, env_kwargs=env_kwargs)

        rows.append(PatientResult(
            method=method,
            patient_id=int(caseid),
            seed=int(seed),
            mdpe=m["mdpe"],
            mdape=m["mdape"],
            wobble=m["wobble"],
            divergence=m["divergence"],
            tit_45_55=float(100.0 * np.mean((bis >= 45) & (bis <= 55))),
            tit_40_60=float(100.0 * np.mean((bis >= 40) & (bis <= 60))),
            n_bis_lt30=int(np.sum(bis < 30)),
            n_bis_gt70=int(np.sum(bis > 70)),
            # Physical units (mg/kg/h, ug/kg/min), as applied by the
            # environment (the v3 residual included).
            mean_ppf=float(np.mean(ppf)),
            mean_remi=float(np.mean(rftn)),
            mdpe_true=m["mdpe_true"],
            mdape_true=m["mdape_true"],
            wobble_true=m["wobble_true"],
            divergence_true=m["divergence_true"],
        ))

    return rows
