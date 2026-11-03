"""Pilot run: one seed, all arms, emitting the per-patient results contract.

Purpose is threefold: measure the real per-arm training cost before committing
to the full seeded protocol, exercise the results-emission path end to end, and
produce the first numbers under the corrected reward, patient-level splits and
real demographics.

This is a pilot. Per the pre-registration it does not replace the seeded
protocol, and its numbers are reported as a pilot.

Usage:
    python experiments/run_pilot.py --n_cases 3000 --offline_epochs 5 --online_episodes 60
"""
import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Windows may expose a CP949 console even when stdout and stderr are redirected
# to experiment logs.  Training messages contain a few Unicode symbols, so use
# UTF-8 explicitly rather than letting a successful run fail while printing its
# final status line.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from agents.baseline_agents import PIDController, ClinicalGuidelineAgent  # noqa: E402
from agents.classical_agent import ClassicalDDPGAgent                      # noqa: E402
from agents.quantum_agent import QuantumDDPGAgent                          # noqa: E402
from agents.tci_agent import TCIAgent                                      # noqa: E402
from data.patient_splits import sample_online_patients, split_by_patient   # noqa: E402
from data.vitaldb_loader import VitalDBLoader                              # noqa: E402
from evaluation.per_patient_eval import evaluate_cohort, patient_from_record  # noqa: E402
from evaluation.results_contract import append_results                     # noqa: E402
from environment.variability import get_variability_spec                   # noqa: E402
from models.vqc import FourierSurrogatePolicy, QuantumDisabledPolicy, QuantumPolicy  # noqa: E402
from models.quantum_critic import HybridQuantumTwinCritic                  # noqa: E402
from train_hybrid import stage1_offline_pretraining, stage2_online_finetuning  # noqa: E402
from environment.dual_drug_env import DualDrugEnv                          # noqa: E402

STATE_DIM = 13
ACTION_DIM = 2

TRAINED_ARMS = {
    "classical_large":   {"kind": "classical", "hidden_dims": [256, 128, 64]},
    "classical_matched": {"kind": "classical", "hidden_dims": [82, 41, 20]},
    "quantum_full":      {"kind": "quantum", "use_skip": True},
    "quantum_noskip":    {"kind": "quantum", "use_skip": False},
    "quantum_disabled":  {"kind": "disabled", "use_skip": True},
    "quantum_noentangle": {"kind": "quantum", "use_skip": True,
                             "entanglement": False},
    "quantum_2qubit":    {"kind": "quantum", "use_skip": True,
                             "n_qubits": 2},
    "quantum_4qubit":    {"kind": "quantum", "use_skip": True,
                             "n_qubits": 4},
    "quantum_2layer":    {"kind": "quantum", "use_skip": True,
                             "n_layers": 2},
    "quantum_8layer":    {"kind": "quantum", "use_skip": True,
                             "n_layers": 8},
    "quantum_noencoder": {"kind": "quantum", "use_skip": True,
                             "encoder_type": "none"},
    # Reviewer 1, comment 3: other ansatz families in the same policy.
    "ansatz_realamplitudes": {"kind": "quantum", "use_skip": True,
                              "ansatz": "real_amplitudes"},
    "ansatz_efficientsu2":   {"kind": "quantum", "use_skip": True,
                              "ansatz": "efficient_su2"},
    "ansatz_twolocal":       {"kind": "quantum", "use_skip": True,
                              "ansatz": "two_local"},
    # Reviewer 2, comment 5: classical Fourier features on the circuit's
    # frequency set, readout matched to the circuit's parameter budget.
    "fourier_surrogate":     {"kind": "fourier", "use_skip": True},
    # Reviewer 1, comment 1: quantum circuit in the critic, classical actor
    # of the Classical-Matched size.
    "quantum_critic":        {"kind": "classical", "hidden_dims": [82, 41, 20],
                              "quantum_critic": True},
}


def _install_quantum_critic(agent, cfg: Dict, seed: int) -> None:
    """Replace the twin critic by the hybrid quantum twin critic (R1-1)."""
    q = cfg.get("quantum", {})
    hidden = cfg.get("networks", {}).get("critic", {}).get("hidden_dims", [256, 256])
    kw = dict(state_dim=agent.encoded_dim, action_dim=ACTION_DIM,
              hidden_dims=hidden, n_qubits=q.get("n_qubits", 3),
              n_layers=q.get("n_layers", 4), seed=seed)
    agent.critic = HybridQuantumTwinCritic(**kw)
    agent.critic_target = HybridQuantumTwinCritic(**kw)
    agent.critic_target.load_state_dict(agent.critic.state_dict())
    for param in agent.critic_target.parameters():
        param.requires_grad = False
    training = cfg.get("training", {})
    quantum, classical = [], []
    for param_name, param in agent.critic.named_parameters():
        (quantum if param_name.split(".")[-1] in {"weights", "input_scaling"}
         else classical).append(param)
    agent.critic_optimizer = torch.optim.Adam([
        {"params": classical, "lr": training.get("critic_lr", 1e-3),
         "weight_decay": training.get("weight_decay", 1e-5)},
        {"params": quantum, "lr": training.get("critic_lr", 1e-3),
         "weight_decay": 0.0},
    ])


PREVIOUS_OUTPUTS = ("results.csv", "patient_split.json", "run_config.json")


def ensure_fresh_output(out: Path) -> None:
    """Refuse to write into a directory that already holds a run's outputs."""
    existing = [name for name in PREVIOUS_OUTPUTS if (Path(out) / name).exists()]
    if existing:
        raise FileExistsError(
            f"{out} already contains {existing}; choose a new --out")


def build_agent(name: str, spec: Dict, config: Dict, device, seed: int):
    """Construct one arm's agent with the encoder and critic held fixed."""
    cfg = json.loads(json.dumps(config))    # deep copy; arms must not share state
    encoder_type = spec.get(
        "encoder_type",
        cfg.get("experiment_overrides", {}).get("encoder_type", "transformer"))

    if spec["kind"] == "classical":
        cfg.setdefault("networks", {})["actor"] = {
            "hidden_dims": spec["hidden_dims"],
            "activation": "relu",
            "dropout": config.get("experiment_overrides", {}).get(
                "classical_dropout", 0.1),
            "action_prior": config.get("experiment_overrides", {}).get(
                "classical_action_prior"),
            "residual_init_scale": config.get("quantum", {}).get(
                "residual_init_scale", 0.02),
        }
        agent = ClassicalDDPGAgent(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, config=cfg,
            encoder_type=encoder_type, seed=seed,
            sequence_length=cfg.get("encoder", {}).get("sequence_length", 10))
        if spec.get("quantum_critic"):
            _install_quantum_critic(agent, cfg, seed)
    else:
        q = cfg.setdefault("quantum", {})
        if "n_qubits" in spec:
            q["n_qubits"] = spec["n_qubits"]
        if "n_layers" in spec:
            q["n_layers"] = spec["n_layers"]
        if "entanglement" in spec:
            q["entanglement"] = spec["entanglement"]
        if "ansatz" in spec:
            q["ansatz"] = spec["ansatz"]
        agent = QuantumDDPGAgent(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, config=cfg,
            encoder_type=encoder_type, seed=seed,
            sequence_length=cfg.get("encoder", {}).get("sequence_length", 10))
        if not spec["use_skip"] or spec["kind"] in ("disabled", "fourier"):
            q = cfg["quantum"]
            kw = dict(state_dim=agent.encoded_dim, n_qubits=q["n_qubits"],
                      n_layers=q["n_layers"], action_dim=ACTION_DIM,
                      action_scale=1.0, use_classical_skip=False,
                      action_prior=q.get("action_prior"),
                      residual_init_scale=q.get("residual_init_scale", 0.02),
                      shots=q.get("shots"),
                      entanglement=q.get("entanglement", True),
                      ansatz=q.get("ansatz", "strongly_entangling"))
            if spec["kind"] == "disabled":
                kw["use_classical_skip"] = True
                policy_class = QuantumDisabledPolicy
            elif spec["kind"] == "fourier":
                kw["use_classical_skip"] = True
                policy_class = FourierSurrogatePolicy
            else:
                policy_class = QuantumPolicy
            agent.actor = policy_class(**kw, seed=seed)
            agent.actor_target = policy_class(**kw, seed=seed)
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            actor_lr = config["training"].get("quantum_actor_lr", 1e-3)
            vqc_lr = config["training"].get("vqc_lr", actor_lr)
            quantum_params, classical_params = [], []
            for param_name, param in agent.actor.named_parameters():
                (quantum_params if param_name in {
                    "weights", "input_scaling", "surrogate_weights"}
                 else classical_params).append(param)
            agent.actor_optimizer = torch.optim.Adam([
                {"params": classical_params, "lr": actor_lr},
                {"params": quantum_params, "lr": vqc_lr},
            ])

    for m in ("actor", "actor_target", "critic", "critic_target", "encoder",
              "encoder_target"):
        if getattr(agent, m, None) is not None:
            setattr(agent, m, getattr(agent, m).to(device))
    return agent


DEFAULT_RESIDUAL_RANGE = (4.0, 0.05)


def parse_residual_range(text) -> tuple:
    """'ppf,remi' half-ranges (mg/kg/h, ug/kg/min); None gives the v3 range (D14)."""
    if text is None:
        return DEFAULT_RESIDUAL_RANGE
    try:
        values = tuple(float(x) for x in str(text).split(","))
    except ValueError:
        values = ()
    if len(values) != 2 or not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError(f"--residual_range needs two positive values 'ppf,remi', got {text!r}")
    return values


def learned_env_kwargs(args) -> Dict:
    """Environment settings shared by training and evaluation of every learned arm."""
    kwargs = {"reward_w3": args.reward_w3, "reward_w_smooth": args.reward_w_smooth}
    if args.residual_pid:
        document = json.loads(Path(args.residual_pid).read_text(encoding="utf-8"))
        if "parameters" not in document:
            raise ValueError(f"{args.residual_pid} has no 'parameters' block")
        kwargs["residual_pid"] = dict(document["parameters"])
        kwargs["residual_range"] = parse_residual_range(args.residual_range)
    return kwargs


def check_residual_args(args) -> None:
    """Refuse option combinations that have no meaning for a residual policy (D14).

    A residual corrects the PID, so imitating clinician doses has no meaning
    for it, and action smoothing would smooth the residual fraction rather
    than the dose. Checked before any data is loaded or any baseline is run.
    """
    if not args.residual_pid:
        if args.residual_range is not None:
            raise SystemExit("--residual_range has no effect without --residual_pid")
        return
    try:
        parse_residual_range(args.residual_range)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.resume_stage1 or not args.skip_offline or args.online_bc_weight > 0:
        raise SystemExit("residual mode trains online only: pass --skip_offline and "
                         "--online_bc_weight 0, without --resume_stage1")
    if args.smoothing_beta > 0:
        raise SystemExit("residual mode: --smoothing_beta would smooth the residual "
                         "fraction, not the dose; use 0")


def env_kwargs_from_run_config(path) -> Dict:
    """Learned-arm environment settings of a finished run, from its run_config.json.

    Re-evaluating a v3 checkpoint without them would read its residual output
    as an absolute dose. Runs from before v3 lack these keys and get the
    parser defaults, i.e. the plain environment.
    """
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    args = build_parser().parse_args(["--out", "unused"])
    for key, value in saved.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return learned_env_kwargs(args)


def run_config_for_checkpoint(checkpoint) -> Path:
    """run_config.json of the run that wrote <out>/<arm>/<stage>/<name>.pt."""
    path = Path(checkpoint).resolve().parents[2] / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"no run_config.json for {checkpoint} (looked for {path})")
    return path


def apply_quantum_action_prior(config: Dict, text: str) -> None:
    """Set the initial output fractions of every quantum-family actor (D17)."""
    if text == "config":
        return
    prior = [float(x) for x in text.split(",")]
    if len(prior) != ACTION_DIM or not all(0.0 < p < 1.0 for p in prior):
        raise ValueError("--quantum_action_prior needs two values in (0, 1)")
    config.setdefault("quantum", {})["action_prior"] = prior


def apply_actor_learning_rates(config: Dict, args) -> None:
    """Actor learning rates per family: MLP actors and quantum-family actors.

    The quantum-family rate covers the classical parts and the circuit angles
    of the hybrid, circuit-disabled, Fourier and ansatz actors.
    """
    training = config.setdefault("training", {})
    if args.classical_actor_lr is not None:
        training["actor_lr"] = args.classical_actor_lr
    if args.quantum_actor_lr is not None:
        training["quantum_actor_lr"] = args.quantum_actor_lr
        training["vqc_lr"] = args.quantum_actor_lr


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_cases", type=int, default=3000)
    ap.add_argument("--sampling_interval", type=int, default=60)
    ap.add_argument("--data_path", default=None,
                    help="prebuilt offline dataset; avoids rebuilding it per seed")
    ap.add_argument("--clean_temporal_protocol", action="store_true",
                    help="use gap-safe dense observations and subject grouping")
    ap.add_argument("--observation_interval_seconds", type=int, default=60)
    ap.add_argument("--control_interval_seconds", type=int, default=60)
    ap.add_argument("--offline_epochs", type=int, default=5)
    ap.add_argument("--online_episodes", type=int, default=60)
    ap.add_argument("--warmup_episodes", type=int, default=10)
    ap.add_argument("--online_eval_interval", type=int, default=25)
    ap.add_argument("--n_online_eval_patients", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4,
                    help="data-loader workers; use 0 for deterministic debugging")
    ap.add_argument("--bc_weight", type=float, default=0.8)
    ap.add_argument("--sequence_length", type=int, default=10,
                    help="patient-safe temporal context used offline and online")
    ap.add_argument("--online_bc_weight", type=float, default=0.0,
                    help="TD3+BC expert anchor during simulator fine-tuning")
    ap.add_argument("--online_rl_scale", type=float, default=0.05,
                    help="scale-invariant coefficient numerator for online RL")
    ap.add_argument("--remi_bc_weight", type=float, default=4.0,
                    help="relative online BC loss weight for remifentanil")
    # sim v2: no EMA in the primary evaluation. v1 applied beta=0.8 to the
    # learned arms at evaluation only (never during training), which is the
    # train-deploy mismatch Reviewer 2 asked about (R2-9).
    ap.add_argument("--smoothing_beta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=0,
                    help="fixed patient-level split seed (independent of training seed)")
    ap.add_argument("--evaluation_seed", type=int, default=0,
                    help="fixed simulator seed for common-random-number evaluation")
    ap.add_argument("--online_validation_seed", type=int, default=1000,
                    help="fixed synthetic validation cohort used for checkpoints")
    ap.add_argument("--max_test_patients", type=int, default=120,
                    help="cap the evaluation cohort; the pilot measures cost, "
                         "so a subset of held-out patients is sufficient")
    ap.add_argument("--test_offset", type=int, default=0,
                    help="skip this many held-out patient IDs before evaluation; "
                         "use 120 for the untouched 480-patient final cohort")
    ap.add_argument("--skip_baselines", action="store_true",
                    help="do not re-evaluate PID/TCI/guideline for trained-arm ablations")
    ap.add_argument("--eval_split", choices=("test", "validation"), default="test",
                    help="cohort for the final evaluation; pilots and tuning "
                         "must use 'validation' so test patients stay unseen")
    ap.add_argument("--method_suffix", default="",
                    help="appended to trained-arm names in results.csv, e.g. "
                         "'_td3bc' for the standard TD3+BC settings")
    ap.add_argument("--pid_params", default=None,
                    help="selected_pid.json from tune_pid_patient_safe.py; "
                         "adds a 'pid_tuned' baseline with those gains")
    ap.add_argument("--variability", choices=("literature", "none"),
                    default="literature",
                    help="sim v2 plant with literature inter-individual "
                         "variability and a TCI-style observer; 'none' is v1")
    ap.add_argument("--online_patients", choices=("training", "synthetic"),
                    default="training",
                    help="sim v2 draws one training-split subject per simulator "
                         "episode; 'synthetic' is the v1 uniform population")
    ap.add_argument("--classical_dropout", type=float, default=0.1)
    ap.add_argument("--classical_actor_lr", type=float, default=None)
    ap.add_argument("--quantum_actor_lr", type=float, default=None,
                    help="learning rate of every quantum-family actor, circuit "
                         "angles included (default: config/hyperparameters.yaml)")
    ap.add_argument("--classical_action_prior", default="none",
                    help="comma-separated normalized actions, or 'none'")
    ap.add_argument("--encoder_type", choices=("transformer", "none"),
                    default="transformer")
    ap.add_argument("--out", required=True,
                    help="new output directory; existing run outputs are never overwritten")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--resume_stage1", default=None,
                    help="load this checkpoint and skip offline pre-training")
    ap.add_argument("--skip_offline", action="store_true",
                    help="start online-only training from random initialization")
    # v3 (docs/superpowers/plans/2026-10-21-v3-residual-qrl.md, D14-D17).
    ap.add_argument("--residual_pid", default=None,
                    help="JSON whose 'parameters' configure the tuned PID that sets "
                         "the base infusion; learned arms then add a bounded residual")
    ap.add_argument("--residual_range", default=None,
                    help="residual half-ranges 'propofol,remifentanil' "
                         "(mg/kg/h, ug/kg/min); default 4.0,0.05 with --residual_pid")
    ap.add_argument("--reward_w3", type=float, default=0.1,
                    help="dose-cost weight of the reward (D15)")
    ap.add_argument("--reward_w_smooth", type=float, default=2.0,
                    help="smoothness-penalty weight of the reward (D15)")
    ap.add_argument("--quantum_action_prior", default="config",
                    help="'a,b' initial output fractions of the quantum-family "
                         "actors, or 'config' for config/hyperparameters.yaml")
    return ap


def main():
    args = build_parser().parse_args()
    check_residual_args(args)

    out = Path(args.out)
    ensure_fresh_output(out)
    if args.pid_params and args.skip_baselines:
        raise SystemExit("--pid_params has no effect with --skip_baselines")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=2)
    contract = out / "results.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    variability = get_variability_spec(args.variability)
    print(f"variability: {args.variability}")

    config = yaml.safe_load(open("config/hyperparameters.yaml", encoding="utf-8"))
    apply_quantum_action_prior(config, args.quantum_action_prior)
    config.setdefault("encoder", {})["sequence_length"] = args.sequence_length
    apply_actor_learning_rates(config, args)
    classical_prior = None
    if args.classical_action_prior.lower() != "none":
        classical_prior = [float(x) for x in args.classical_action_prior.split(",")]
        if len(classical_prior) != ACTION_DIM:
            raise ValueError("--classical_action_prior requires two comma-separated values")
    config["experiment_overrides"] = {
        "classical_dropout": args.classical_dropout,
        "classical_action_prior": classical_prior,
        "encoder_type": args.encoder_type,
    }
    print(f"quantum action prior: {config.get('quantum', {}).get('action_prior')}")
    training_config = config.get("training", {})
    print(f"actor learning rates: MLP {training_config.get('actor_lr')}, quantum family "
          f"{training_config.get('quantum_actor_lr')} (circuit {training_config.get('vqc_lr')})")
    env_kwargs = learned_env_kwargs(args)
    print("learned-arm environment: "
          + ", ".join(f"{k}={v}" for k, v in env_kwargs.items() if k != "residual_pid")
          + (f", residual PID from {args.residual_pid}" if args.residual_pid else ""))

    # ---- data and patient-level split -----------------------------------
    loader = VitalDBLoader(cache_dir="data/vitaldb_cache")
    if args.data_path:
        with open(args.data_path, "rb") as stream:
            prepared = pickle.load(stream)
        required = {
            "states", "actions", "next_states", "rewards", "dones",
            "caseids", "subjectids",
        }
        missing = required - set(prepared)
        if missing:
            raise ValueError(f"clean dataset is missing fields: {sorted(missing)}")
    elif args.clean_temporal_protocol:
        prepared = loader.prepare_dual_drug_control_data(
            n_cases=args.n_cases,
            observation_interval_seconds=args.observation_interval_seconds,
            control_interval_seconds=args.control_interval_seconds,
            sequence_length=args.sequence_length,
            unique_subjects=True,
        )
    else:
        states, actions, next_states, rewards, dones, caseids = \
            loader.prepare_dual_drug_training_data(
                n_cases=args.n_cases, sampling_interval=args.sampling_interval)
        prepared = {
            "states": states, "actions": actions,
            "next_states": next_states, "rewards": rewards,
            "dones": dones, "caseids": caseids,
            "subjectids": caseids,
        }

    states = prepared["states"]
    actions = prepared["actions"]
    next_states = prepared["next_states"]
    rewards = prepared["rewards"]
    dones = prepared["dones"]
    caseids = prepared["caseids"]
    subjectids = prepared["subjectids"]
    if args.clean_temporal_protocol or args.data_path:
        dataset_observation = int(np.asarray(
            prepared.get("observation_interval_seconds",
                         args.observation_interval_seconds)))
        dataset_control = int(np.asarray(
            prepared.get("control_interval_seconds",
                         args.control_interval_seconds)))
        dataset_sequence = int(np.asarray(
            prepared.get("sequence_length", args.sequence_length)))
        if (dataset_observation != args.observation_interval_seconds or
                dataset_control != args.control_interval_seconds or
                dataset_sequence != args.sequence_length):
            raise ValueError(
                "dataset timing does not match command: "
                f"data=({dataset_observation}s,{dataset_control}s,T={dataset_sequence}) "
                f"args=({args.observation_interval_seconds}s,"
                f"{args.control_interval_seconds}s,T={args.sequence_length})")

    tr, va, te = split_by_patient(
        subjectids, ratios=(0.7, 0.1, 0.2), seed=args.split_seed)
    pack = lambda idx: {"states": states[idx], "actions": actions[idx],
                        "next_states": next_states[idx], "rewards": rewards[idx],
                        "dones": dones[idx], "caseids": caseids[idx],
                        "subjectids": subjectids[idx]}
    train_data, val_data = pack(tr), pack(va)

    eval_partition = te if args.eval_split == "test" else va
    all_test_ids = np.unique(caseids[eval_partition])
    stop = args.test_offset + args.max_test_patients
    test_ids = all_test_ids[args.test_offset:stop]
    if len(test_ids) != args.max_test_patients:
        raise ValueError(
            f"requested {args.max_test_patients} {args.eval_split} patients at "
            f"offset {args.test_offset}, but only {len(test_ids)} are available")
    clinical = loader.load_clinical_info()
    cohort = [(int(c), patient_from_record(clinical.loc[int(c)].to_dict()))
              for c in test_ids]
    print(f"evaluation cohort: {len(cohort)} {args.eval_split} patients")

    online_patients = None
    checkpoint_patients = None
    if args.online_patients == "training" and args.online_episodes > 0:
        online_ids = sample_online_patients(
            np.unique(caseids[tr]), n_episodes=args.online_episodes, seed=args.seed)
        online_patients = [patient_from_record(clinical.loc[int(c)].to_dict())
                           for c in online_ids]
        # Fixed checkpoint cohort from the validation split, identical for
        # every arm and training seed.
        checkpoint_ids = sample_online_patients(
            np.unique(caseids[va]), n_episodes=args.n_online_eval_patients,
            seed=args.online_validation_seed)
        checkpoint_patients = [patient_from_record(clinical.loc[int(c)].to_dict())
                               for c in checkpoint_ids]
    print(f"\ntrain {len(np.unique(subjectids[tr]))} subjects / "
          f"{len(np.unique(caseids[tr]))} cases / {len(tr):,} transitions")
    print(f"val   {len(np.unique(subjectids[va]))} subjects / "
          f"{len(np.unique(caseids[va]))} cases / {len(va):,} transitions")
    print(f"test  {len(np.unique(subjectids[te]))} subjects / "
          f"{len(np.unique(caseids[te]))} cases "
          f"(evaluating {len(cohort)})")

    json.dump({"seed": args.seed,
               "split_seed": args.split_seed,
               "train_caseids": [int(c) for c in np.unique(caseids[tr])],
               "val_caseids": [int(c) for c in np.unique(caseids[va])],
               "test_caseids": [int(c) for c in np.unique(caseids[te])],
               "train_subjectids": [int(c) for c in np.unique(subjectids[tr])],
               "val_subjectids": [int(c) for c in np.unique(subjectids[va])],
               "test_subjectids": [int(c) for c in np.unique(subjectids[te])],
               "evaluation_split": args.eval_split,
               "evaluated_caseids": [int(c) for c in test_ids],
               "online_patients": args.online_patients,
               "online_caseids": ([int(c) for c in online_ids]
                                  if online_patients is not None else []),
               "checkpoint_caseids": ([int(c) for c in checkpoint_ids]
                                      if checkpoint_patients is not None else [])},
              open(out / "patient_split.json", "w"), indent=2)

    # The locked dense-window pickle is roughly 2 GiB.  Advanced indexing has
    # already materialised only the train/validation partitions needed for
    # learning, so release the full arrays before constructing a model.  This
    # keeps concurrent seed jobs within host RAM.
    del prepared, states, actions, next_states, rewards, dones
    gc.collect()

    timings: Dict[str, Dict[str, float]] = {}

    # ---- untrained baselines --------------------------------------------
    # Baselines always run in the plain environment (no env_kwargs): in the
    # v3 residual mode their output would be read as a residual.
    baseline_agents = () if args.skip_baselines else (
        ("pid", PIDController()),
        ("tci", TCIAgent(control_interval_seconds=args.control_interval_seconds)),
        ("guideline", ClinicalGuidelineAgent()))
    if args.pid_params and not args.skip_baselines:
        tuned = json.loads(Path(args.pid_params).read_text())["parameters"]
        print(f"tuned PID from {args.pid_params}: {tuned}")
        baseline_agents = baseline_agents + (("pid_tuned", PIDController(**tuned)),)
    for name, agent in baseline_agents:
        t0 = time.time()
        rows = evaluate_cohort(agent, name, cohort, seed=args.seed,
                               environment_seed=args.evaluation_seed,
                               variability=variability,
                               observation_interval_seconds=args.observation_interval_seconds,
                               control_interval_seconds=args.control_interval_seconds,
                               progress=lambda it: tqdm(it, desc=f"eval {name}",
                                                        leave=False))
        append_results(rows, contract)
        timings[name] = {"eval_s": time.time() - t0}
        md = np.median([r.mdape for r in rows])
        print(f"  {name:<18} MDAPE {md:6.2f}%   ({timings[name]['eval_s']:.0f}s)")

    # ---- trained arms ----------------------------------------------------
    selected = (list(TRAINED_ARMS) if args.arms == "all"
                else [a.strip() for a in args.arms.split(",")])

    for name in selected:
        spec = TRAINED_ARMS[name]
        print(f"\n{'='*66}\nARM: {name}\n{'='*66}")
        dirs = {"stage1": out / name / "stage1", "stage2": out / name / "stage2",
                "base": out / name}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        agent = build_agent(name, spec, config, device, args.seed)
        n_actor = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
        print(f"actor parameters: {n_actor:,}")

        if args.resume_stage1:
            agent.load(args.resume_stage1)
            t_stage1 = 0.0
            print(f"loaded stage-1 checkpoint: {args.resume_stage1}")
        elif args.skip_offline:
            t_stage1 = 0.0
            print("skipping stage 1: online-only random initialization")
        else:
            t0 = time.time()
            agent = stage1_offline_pretraining(
                agent=agent, train_data=train_data, val_data=val_data,
                n_epochs=args.offline_epochs, batch_size=args.batch_size,
                bc_weight=args.bc_weight, dirs=dirs, eval_interval=10,
                device=device, num_workers=args.num_workers)
            t_stage1 = time.time() - t0

        if args.online_episodes > 0:
            t0 = time.time()
            agent = stage2_online_finetuning(
                agent=agent, n_episodes=args.online_episodes,
                warmup_episodes=min(args.warmup_episodes, args.online_episodes),
                dirs=dirs, eval_interval=args.online_eval_interval, seed=args.seed,
                env_class=DualDrugEnv, device=device,
                n_eval_patients=args.n_online_eval_patients,
                offline_bc_data=(train_data if args.online_bc_weight > 0 else None),
                online_bc_weight=args.online_bc_weight,
                online_rl_scale=args.online_rl_scale,
                bc_action_weights=[1.0, args.remi_bc_weight],
                validation_seed=args.online_validation_seed,
                observation_interval_seconds=args.observation_interval_seconds,
                control_interval_seconds=args.control_interval_seconds,
                train_patients=online_patients,
                validation_patients=checkpoint_patients,
                variability=variability,
                env_kwargs=env_kwargs)
            t_stage2 = time.time() - t0
        else:
            t_stage2 = 0.0

        t0 = time.time()
        rows = evaluate_cohort(agent, name + args.method_suffix, cohort, seed=args.seed,
                               environment_seed=args.evaluation_seed,
                               variability=variability,
                               smoothing_beta=args.smoothing_beta,
                               observation_interval_seconds=args.observation_interval_seconds,
                               control_interval_seconds=args.control_interval_seconds,
                               env_kwargs=env_kwargs,
                               progress=lambda it: tqdm(it, desc=f"eval {name}",
                                                        leave=False))
        append_results(rows, contract)
        t_eval = time.time() - t0

        n_critic = sum(p.numel() for p in agent.critic.parameters() if p.requires_grad)
        timings[name + args.method_suffix] = {
            "stage1_s": t_stage1, "stage2_s": t_stage2, "eval_s": t_eval,
            "actor_params": n_actor, "critic_params": n_critic,
            "device": str(device)}
        json.dump(timings, open(out / "timings.json", "w"), indent=2)

        md = np.median([r.mdape for r in rows])
        tit = np.median([r.tit_45_55 for r in rows])
        ppf = np.median([r.mean_ppf for r in rows])
        rft = np.median([r.mean_remi for r in rows])
        print(f"\n{name}: MDAPE {md:.2f}%  TIT[45,55] {tit:.1f}%  "
              f"ppf {ppf:.2f} mg/kg/h  remi {rft:.3f} ug/kg/min")
        print(f"  stage1 {t_stage1/60:.1f} min | stage2 {t_stage2/60:.1f} min "
              f"| eval {t_eval/60:.1f} min")

    print(f"\nContract: {contract}")
    print(f"Timings:  {out/'timings.json'}")


if __name__ == "__main__":
    main()
