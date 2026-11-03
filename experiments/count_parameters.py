"""Count the trainable parameters of each model.

Every number is produced by walking the instantiated modules rather than
transcribed by hand.

Three totals are reported because they answer different questions:

  policy head       what a claim about the actor may reference
  end-to-end        encoder + actor + twin critic; what training cost tracks
  deployed          encoder + actor; what an embedded implementation carries

Usage:
    python experiments/count_parameters.py
    python experiments/count_parameters.py --latex
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.classical_agent import MLPActor            # noqa: E402
from agents.quantum_agent import QuantumDDPGAgent      # noqa: E402
from models.vqc import QuantumDisabledPolicy, QuantumPolicy  # noqa: E402

STATE_DIM = 13
ACTION_DIM = 2
CONFIG = "config/hyperparameters.yaml"


def trainable(module: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def actor_budget(policy: QuantumPolicy) -> Dict[str, int]:
    """Decompose a QuantumPolicy into the categories Table II reports.

    Args:
        policy: The actor to measure.

    Returns:
        Counts per category, plus `on_device` (parameters physically residing
        on the quantum device) and `total`.
    """
    skip = trainable(policy.classical_skip) if policy.classical_skip is not None else 0
    budget = {
        "angle_projection": trainable(policy.layer_projections),
        "input_scaling": policy.input_scaling.numel(),
        "pqc_angles": policy.weights.numel(),
        "surrogate_core": 0,
        "readout": trainable(policy.quantum_residual),
        "classical_skip": skip,
        "mlp_body": 0,
    }
    budget["on_device"] = budget["pqc_angles"]
    budget["total"] = trainable(policy)
    return budget


def mlp_actor_budget(actor: MLPActor) -> Dict[str, int]:
    """Budget for a purely classical actor, in the same key space."""
    return {
        "angle_projection": 0,
        "input_scaling": 0,
        "pqc_angles": 0,
        "surrogate_core": 0,
        "readout": 0,
        "classical_skip": 0,
        "mlp_body": trainable(actor),
        "on_device": 0,
        "total": trainable(actor),
    }


def find_matched_hidden_dims(
    target: int, input_dim: int, output_dim: int = ACTION_DIM, tolerance: float = 0.05
) -> Tuple[List[int], int]:
    """Size a classical actor to match `target` parameters within `tolerance`.

    Keeps the 4:2:1 taper of the Classical-Large baseline ([256, 128, 64]) and
    scales it, so capacity is the only difference between the two classical arms.

    Args:
        target: Parameter count to match.
        input_dim: Encoder output width feeding the actor.
        output_dim: Action dimension.
        tolerance: Allowed relative deviation.

    Returns:
        The chosen widths and their exact parameter count.

    Raises:
        ValueError: If no width lands within tolerance.
    """
    best = None
    for base in range(4, 513):
        dims = [base, max(1, base // 2), max(1, base // 4)]
        count = trainable(MLPActor(input_dim=input_dim, output_dim=output_dim,
                                   hidden_dims=dims))
        err = abs(count - target)
        if best is None or err < best[0]:
            best = (err, dims, count)

    err, dims, count = best
    if err / target > tolerance:
        raise ValueError(
            f"no width within {tolerance:.0%} of {target:,}; closest {dims} at {count:,}"
        )
    return dims, count


ROWS = [
    ("encoder", "Temporal encoder (shared)"),
    ("angle_projection", r"Actor: angle projection (per-layer, $L\times$)"),
    ("input_scaling", "Actor: learnable input scaling"),
    ("pqc_angles", r"Actor: PQC angles ($3nL$)"),
    ("surrogate_core", r"Actor: classical surrogate core ($3nL$)"),
    ("readout", "Actor: readout"),
    ("classical_skip", "Actor: classical residual branch"),
    ("mlp_body", "Actor: MLP body"),
    ("policy_head", r"\textbf{Policy head total}"),
    ("critic", "Twin critic"),
    ("end_to_end", r"\textbf{End-to-end trainable}"),
    ("deployed", r"\textbf{Deployed (encoder $+$ actor)}"),
    ("on_device", r"\textbf{Resident on quantum device}"),
]


def build() -> Dict[str, Dict[str, int]]:
    """Instantiate every configuration and return its full budget."""
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))

    agent = QuantumDDPGAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, config=cfg,
                             encoder_type="transformer", seed=0)
    enc = trainable(agent.encoder)
    critic = trainable(agent.critic)
    latent = agent.encoded_dim

    qcfg = cfg["quantum"]
    common = dict(state_dim=latent, n_qubits=qcfg["n_qubits"],
                  n_layers=qcfg["n_layers"], action_dim=ACTION_DIM, seed=0)

    full = actor_budget(QuantumPolicy(**common))
    noskip = actor_budget(QuantumPolicy(**common, use_classical_skip=False))
    disabled_policy = QuantumDisabledPolicy(**common)
    disabled = {
        "angle_projection": trainable(disabled_policy.layer_projections),
        "input_scaling": disabled_policy.input_scaling.numel(),
        "pqc_angles": 0,
        "surrogate_core": disabled_policy.surrogate_weights.numel(),
        "readout": trainable(disabled_policy.quantum_residual),
        "classical_skip": trainable(disabled_policy.classical_skip),
        "mlp_body": 0,
        "on_device": 0,
        "total": trainable(disabled_policy),
    }
    large = mlp_actor_budget(MLPActor(input_dim=latent, output_dim=ACTION_DIM,
                                      hidden_dims=[256, 128, 64]))
    dims, matched_count = find_matched_hidden_dims(full["total"], latent)
    matched = mlp_actor_budget(MLPActor(input_dim=latent, output_dim=ACTION_DIM,
                                        hidden_dims=dims))

    out = {}
    for name, b in (("Cls-Large", large), ("Cls-Matched", matched),
                    ("Proposed", full), ("QuantumDisabled", disabled),
                    ("NoSkip", noskip)):
        b = dict(b)
        b["encoder"] = enc
        b["critic"] = critic
        b["policy_head"] = b["total"]
        b["end_to_end"] = enc + b["total"] + critic
        b["deployed"] = enc + b["total"]
        out[name] = b

    out["_meta"] = {"latent_dim": latent, "matched_hidden_dims": dims,
                    "matched_count": matched_count,
                    "n_qubits": qcfg["n_qubits"], "n_layers": qcfg["n_layers"]}
    return out


def format_table(budgets: Dict[str, Dict[str, int]]) -> str:
    """Render the budget as the body of Table II."""
    cols = ["Cls-Large", "Cls-Matched", "Proposed", "QuantumDisabled", "NoSkip"]
    lines = [" & ".join(["Module"] + cols) + r" \\", r"\midrule"]
    for key, label in ROWS:
        cells = []
        for c in cols:
            v = budgets[c][key]
            cells.append("---" if v == 0 and key not in ("on_device",) else f"{v:,}")
        lines.append(" & ".join([label] + cells) + r" \\")
        if key in ("encoder", "policy_head", "critic"):
            lines.append(r"\midrule")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latex", action="store_true", help="print the LaTeX table body")
    ap.add_argument("--json", default="logs/parameter_budget.json")
    args = ap.parse_args()

    b = build()
    meta = b.pop("_meta")

    print(f"Encoder latent dimension : {meta['latent_dim']}")
    print(f"Qubits / layers          : {meta['n_qubits']} / {meta['n_layers']}")
    print(f"Classical-Matched widths : {meta['matched_hidden_dims']} "
          f"-> {meta['matched_count']:,}")
    print()
    for name, v in b.items():
        frac = 100 * v["on_device"] / v["deployed"] if v["deployed"] else 0
        print(f"{name:<12} head={v['policy_head']:>8,}  deployed={v['deployed']:>8,}  "
              f"end2end={v['end_to_end']:>8,}  on-device={v['on_device']:>4,} ({frac:.3f}%)")

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({**b, "_meta": meta}, open(args.json, "w"), indent=2)
    print(f"\nwrote {args.json}")

    if args.latex:
        print("\n" + format_table(b))


if __name__ == "__main__":
    main()
