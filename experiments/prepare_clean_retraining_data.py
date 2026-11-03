"""Create and lock the subject-disjoint 10-s/60-s/T=60 VitalDB dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.patient_splits import split_by_patient  # noqa: E402
from data.vitaldb_loader import VitalDBLoader      # noqa: E402


DEFAULT_DATA = ROOT / "data/offline_dataset/vitaldb_obs10_ctrl60_t60_subject3000_pkvalid.pkl"
DEFAULT_MANIFEST = ROOT / "logs/clean_retraining_obs10_ctrl60_t60/dataset_manifest_pkvalid.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_caseids(manifest: dict) -> list:
    """Every case id of a locked manifest, ascending (the original scan order)."""
    return sorted(int(caseid) for partition in manifest["partitions"].values()
                  for caseid in partition["caseids"])


def split_differences(manifest: dict, locked: dict) -> list:
    """Partitions whose case ids differ from those of the locked manifest."""
    return [name for name in ("train", "validation", "test")
            if manifest["partitions"][name]["caseids"]
            != locked["partitions"][name]["caseids"]]


def resolve_split_seed(requested, locked):
    """The requested split seed, else the locked manifest's, else 0."""
    if requested is not None:
        return int(requested)
    return int(locked["split_seed"]) if locked else 0


def parse_bis_range(values):
    """``[low, high]`` as a checked float pair, or None for the loader's default."""
    if values is None:
        return None
    low, high = (float(v) for v in values)
    if not (0.0 <= low < high <= 100.0):
        raise ValueError(f"BIS range must satisfy 0 <= low < high <= 100, got {values}")
    return (low, high)


def check_manifest_target(path: Path, validate_existing: bool) -> None:
    """Refuse to overwrite a manifest, which records a locked dataset."""
    if Path(path).exists() and not validate_existing:
        raise FileExistsError(f"refusing to overwrite locked manifest: {path}")


def validate(data: dict, split_seed: int = 0) -> dict:
    required = {
        "states", "next_states", "actions", "rewards", "dones",
        "caseids", "subjectids", "timestamps", "segmentids",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"missing dataset fields: {sorted(missing)}")
    n = len(data["states"])
    if data["states"].shape != (n, 60, 13):
        raise ValueError(f"unexpected state shape {data['states'].shape}")
    if data["next_states"].shape != data["states"].shape:
        raise ValueError("state and next-state windows differ")
    if not all(len(data[key]) == n for key in (
            "actions", "rewards", "dones", "caseids", "subjectids",
            "timestamps", "segmentids")):
        raise ValueError("transition fields have unequal lengths")
    if not (np.isfinite(data["states"]).all()
            and np.isfinite(data["next_states"]).all()
            and np.isfinite(data["actions"]).all()
            and np.isfinite(data["rewards"]).all()):
        raise ValueError("dataset contains non-finite values")

    cases = np.unique(data["caseids"])
    subjects = np.unique(data["subjectids"])
    if len(cases) != 3000 or len(subjects) != 3000:
        raise ValueError(
            f"expected 3000 cases/subjects, got {len(cases)}/{len(subjects)}")
    mapping = {}
    for caseid, subjectid in zip(data["caseids"], data["subjectids"]):
        mapping.setdefault(int(caseid), int(subjectid))
        if mapping[int(caseid)] != int(subjectid):
            raise ValueError("one case maps to multiple subjects")

    tr, va, te = split_by_patient(
        np.asarray(data["subjectids"]), ratios=(0.7, 0.1, 0.2), seed=split_seed)
    partitions = {
        "train": tr,
        "validation": va,
        "test": te,
    }
    partition_manifest = {}
    subject_sets = {}
    for name, indices in partitions.items():
        partition_cases = np.unique(data["caseids"][indices])
        partition_subjects = np.unique(data["subjectids"][indices])
        partition_manifest[name] = {
            "n_cases": int(len(partition_cases)),
            "n_subjects": int(len(partition_subjects)),
            "n_transitions": int(len(indices)),
            "caseids": [int(value) for value in partition_cases],
            "subjectids": [int(value) for value in partition_subjects],
        }
        subject_sets[name] = set(partition_subjects.tolist())
    if [partition_manifest[x]["n_subjects"] for x in partitions] != [2100, 300, 600]:
        raise ValueError("split is not exactly 2100/300/600 subjects")
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        if subject_sets[left] & subject_sets[right]:
            raise ValueError(f"subject leakage between {left} and {right}")

    return {
        "n_transitions": n,
        "state_shape": list(data["states"].shape),
        "n_segments": int(len(np.unique(data["segmentids"]))),
        "observation_interval_seconds": 10,
        "control_interval_seconds": 60,
        "sequence_length": 60,
        "history_minutes": 10,
        "unique_operation_per_subject": True,
        "split_seed": int(split_seed),
        "partitions": partition_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validate_existing", action="store_true")
    parser.add_argument("--split_seed", type=int, default=None,
                        help="default: the --caseids_from manifest's seed, else 0")
    parser.add_argument("--caseids_from", type=Path, default=None,
                        help="rebuild exactly the cases of this locked manifest; "
                             "fails unless every partition keeps the same cases")
    parser.add_argument("--bis_range", type=float, nargs=2, default=None,
                        metavar=("LOW", "HIGH"),
                        help="BIS range of usable samples (loader default: 30 70)")
    args = parser.parse_args()
    bis_range = parse_bis_range(args.bis_range)

    args.out = args.out.resolve()
    args.manifest = args.manifest.resolve()
    locked = (json.loads(args.caseids_from.read_text(encoding="utf-8"))
              if args.caseids_from else None)
    args.split_seed = resolve_split_seed(args.split_seed, locked)
    check_manifest_target(args.manifest, args.validate_existing)
    if args.out.exists():
        if not args.validate_existing:
            raise FileExistsError(
                f"refusing to overwrite locked dataset: {args.out}")
        with args.out.open("rb") as stream:
            data = pickle.load(stream)
    else:
        partial = args.out.with_suffix(args.out.suffix + ".partial")
        if partial.exists():
            raise FileExistsError(f"partial dataset already exists: {partial}")
        loader = VitalDBLoader(cache_dir=str(ROOT / "data/vitaldb_cache"))
        data = loader.prepare_dual_drug_control_data(
            n_cases=3000,
            observation_interval_seconds=10,
            control_interval_seconds=60,
            sequence_length=60,
            unique_subjects=True,
            save_path=str(partial),
            caseids=locked_caseids(locked) if locked else None,
            bis_range=bis_range,
        )
        if locked:
            moved = split_differences(validate(data, split_seed=args.split_seed), locked)
            if moved:
                raise ValueError(
                    f"rebuilt cohort changes the {moved} partitions; kept {partial}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        partial.replace(args.out)

    manifest = validate(data, split_seed=args.split_seed)
    if locked:
        manifest["rebuilt_from_manifest"] = str(args.caseids_from)
    if bis_range is not None:
        manifest["bis_range"] = list(bis_range)
    manifest.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.out),
        "dataset_bytes": args.out.stat().st_size,
        "dataset_sha256": sha256(args.out),
        "protocol_status": "locked_before_training",
    })
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "dataset": str(args.out),
        "sha256": manifest["dataset_sha256"],
        "n_transitions": manifest["n_transitions"],
        "partitions": {
            name: {key: value for key, value in details.items()
                   if key.startswith("n_")}
            for name, details in manifest["partitions"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
