"""Closed-loop shot, gate-noise and readout-noise evaluation of trained QRL (sim v2).

Each trained quantum_full policy is loaded, only the circuit execution is
replaced, and the complete 200-minute closed loop is rerun on test patients of
the sim v2 plant (literature variability, TCI observer, 10-s observations,
60-s control, no EMA). The conditions cover finite shots, depolarizing noise
after every layer, symmetric readout error (PennyLane default.mixed
``readout_prob``, applied in each measurement basis) and a combined
near-term setting. This is an in-silico noise study, not a hardware run.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import gpu_units  # noqa: E402
from data.vitaldb_loader import VitalDBLoader  # noqa: E402
from environment.variability import LITERATURE_SPEC  # noqa: E402
from evaluation.per_patient_eval import evaluate_cohort, patient_from_record  # noqa: E402
from evaluation.results_contract import append_results  # noqa: E402
from run_pilot import (  # noqa: E402
    TRAINED_ARMS, build_agent, env_kwargs_from_run_config, run_config_for_checkpoint)

CONDITIONS = {
    "analytic": {"shots": None, "depolarizing": 0.0, "readout": 0.0},
    "shots_1000": {"shots": 1000, "depolarizing": 0.0, "readout": 0.0},
    "shots_100": {"shots": 100, "depolarizing": 0.0, "readout": 0.0},
    "shots_10": {"shots": 10, "depolarizing": 0.0, "readout": 0.0},
    "depolarizing_0.001": {"shots": None, "depolarizing": 0.001, "readout": 0.0},
    "depolarizing_0.005": {"shots": None, "depolarizing": 0.005, "readout": 0.0},
    "depolarizing_0.01": {"shots": None, "depolarizing": 0.01, "readout": 0.0},
    "depolarizing_0.05": {"shots": None, "depolarizing": 0.05, "readout": 0.0},
    "readout_0.01": {"shots": None, "depolarizing": 0.0, "readout": 0.01},
    "readout_0.03": {"shots": None, "depolarizing": 0.0, "readout": 0.03},
    "nisq_typical": {"shots": 1000, "depolarizing": 0.005, "readout": 0.01},
}
CHECKPOINT = "{root}/primary/quantum_full/seed_{seed}/quantum_full/stage2/best_mdape.pt"
SPLIT = "{root}/primary/quantum_full/seed_0/patient_split.json"


class NoisyInferenceVQC:
    """Float64 NumPy QNode replacing the trained circuit at inference only."""

    def __init__(self, n_qubits, n_layers, shots, depolarizing, readout, seed):
        noisy = depolarizing > 0 or readout > 0
        kwargs = {"wires": n_qubits, "shots": shots}
        if noisy:
            kwargs["readout_prob"] = readout if readout > 0 else None
            device = qml.device("default.mixed", **kwargs)
        else:
            device = qml.device("default.qubit", seed=seed, **kwargs)

        @qml.qnode(device, interface=None)
        def circuit(layer_inputs, weights):
            for layer in range(n_layers):
                for wire in range(n_qubits):
                    qml.RX(layer_inputs[layer, wire] * np.pi, wires=wire)
                qml.StronglyEntanglingLayers(weights[layer:layer + 1], wires=range(n_qubits))
                if depolarizing > 0:
                    for wire in range(n_qubits):
                        qml.DepolarizingChannel(depolarizing, wires=wire)
            return [qml.expval(observable(wire)) for wire in range(n_qubits)
                    for observable in (qml.PauliX, qml.PauliY, qml.PauliZ)]

        self.circuit = circuit

    def forward(self, layer_inputs, weights):
        features = layer_inputs.detach().cpu().double().numpy()
        angles = weights.detach().cpu().double().numpy()
        outputs = np.stack([np.asarray(self.circuit(sample, angles), dtype=np.float64)
                            for sample in features])
        return torch.as_tensor(outputs, dtype=layer_inputs.dtype, device=layer_inputs.device)


def training_config() -> dict:
    """The configuration run_pilot builds for the campaign (T = 60)."""
    config = yaml.safe_load((ROOT / "config/hyperparameters.yaml").read_text(encoding="utf-8"))
    config.setdefault("encoder", {})["sequence_length"] = 60
    config.setdefault("training", {})["actor_lr"] = 1e-4
    config["experiment_overrides"] = {"classical_dropout": 0.0,
                                      "classical_action_prior": [0.5, 0.075],
                                      "encoder_type": "transformer"}
    return config


def evaluate_seed(seed, names, ids, root, pattern, device="cpu", progress=None):
    clinical = VitalDBLoader(cache_dir=str(ROOT / "data/vitaldb_cache")).load_clinical_info()
    cohort = [(c, patient_from_record(clinical.loc[c].to_dict())) for c in ids]
    config = training_config()
    checkpoint = pattern.format(root=root, seed=seed)
    # The environment the checkpoint was trained in (v3: residual on the PID).
    env_kwargs = env_kwargs_from_run_config(run_config_for_checkpoint(checkpoint))
    finished = []
    for name in names:
        condition = CONDITIONS[name]
        agent = build_agent("quantum_full", TRAINED_ARMS["quantum_full"],
                            copy.deepcopy(config), torch.device(device), seed)
        agent.load(checkpoint)
        if agent.actor.vqc.ansatz != "strongly_entangling":
            raise RuntimeError("noise study implements the strongly entangling circuit only")
        agent.actor.vqc = NoisyInferenceVQC(
            agent.actor.n_qubits, agent.actor.n_layers, condition["shots"],
            condition["depolarizing"], condition["readout"], seed=10_000 * seed + 20261019)
        agent.actor.eval()
        np.random.seed(10_000 * seed + 20261019)    # default.mixed sampling
        rows = evaluate_cohort(agent, name, cohort, seed=seed, environment_seed=0,
                               smoothing_beta=0.0, observation_interval_seconds=10,
                               control_interval_seconds=60, variability=LITERATURE_SPEC,
                               env_kwargs=env_kwargs, progress=progress)
        finished.append((name, rows))
    return seed, finished


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="logs/revision_v2/campaign")
    parser.add_argument("--checkpoint_pattern", default=CHECKPOINT)
    parser.add_argument("--split_json", default=None)
    parser.add_argument("--seeds", default=",".join(map(str, range(10))))
    parser.add_argument("--max_patients", type=int, default=120,
                        help="first N evaluated test patients")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device of the encoder and readout; the noisy circuit is NumPy")
    parser.add_argument("--one", default=None, help="CONDITION:SEED, run one unit in-process")
    parser.add_argument("--unit_out", default=None, help="folder of the --one unit")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    names = [n.strip() for n in args.conditions.split(",")]
    unknown = set(names) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    split_path = Path(args.split_json or SPLIT.format(root=args.root))
    split = json.loads(split_path.read_text())
    ids = [int(c) for c in split["evaluated_caseids"][:args.max_patients]]
    if len(ids) != args.max_patients:
        raise RuntimeError("not enough evaluated patients in the split file")

    if args.one:  # one (condition, seed) unit, called by the runner below
        name, seed = args.one.split(":")
        _, finished = evaluate_seed(int(seed), [name], ids, args.root, args.checkpoint_pattern,
                                    args.device, progress=gpu_units.progress_printer(len(ids)))
        for name, rows in finished:
            append_results(rows, Path(args.unit_out) / "results.csv")
            print(f"seed {seed} {name}: mean MDAPE {np.mean([r.mdape for r in rows]):.3f}", flush=True)
        return

    # Every (condition, seed) is a GPU subprocess under the stall watchdog
    # (experiments/gpu_units.py); finished units are cached.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps({
        "conditions": {n: CONDITIONS[n] for n in names}, "seeds": seeds,
        "patients": ids, "checkpoint_pattern": args.checkpoint_pattern,
        "root": args.root, "variability": "literature", "device": args.device,
        "observation_interval_seconds": 10, "control_interval_seconds": 60,
        "smoothing_beta": 0.0}, indent=2), encoding="utf-8")
    units = []
    for seed in seeds:
        for name in names:
            folder = out / "units" / f"{name}_seed{seed}"
            command = (sys.executable, str(Path(__file__).resolve()), "--one", f"{name}:{seed}",
                       "--unit_out", str(folder.resolve()), "--device", args.device,
                       "--root", args.root, "--checkpoint_pattern", args.checkpoint_pattern,
                       "--max_patients", str(len(ids)), "--split_json", str(split_path.resolve()))
            units.append(gpu_units.Unit(f"{name}/seed{seed}", command, folder,
                                        frozenset({name}), len(ids)))
    print(f"{len(units)} units on {args.device}", flush=True)
    failures = gpu_units.run_units(units, args.jobs)
    sources = gpu_units.unit_results(out / "units")
    merged = gpu_units.merge(sources, out / "results_all.csv")
    print(f"merged {merged.groupby(['method', 'seed']).ngroups} condition-seed pairs", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} unit(s) failed: {failures}")


if __name__ == "__main__":
    main()
