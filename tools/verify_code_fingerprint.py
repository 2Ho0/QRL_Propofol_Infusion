"""Check that this checkout reproduces the code fingerprint of the locked v3 protocol.

The fingerprint is computed with the campaign runner's own function
(experiments/run_revision_campaign.py, code_fingerprint). It covers every
src/**/*.py file, config/hyperparameters.yaml, experiments/run_pilot.py,
train_hybrid.py, run_revision_campaign.py, run_v3_campaign.py and the tuned PID
gains in logs/revision_v2/pid_tuning/selected_pid.json, path and bytes alike.

usage: python tools/verify_code_fingerprint.py [--protocol PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from run_revision_campaign import DEFAULT_PID, code_fingerprint  # noqa: E402
from run_v3_campaign import campaign_fingerprint_files  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "protocols/campaign_locked_protocol.json")
    args = parser.parse_args(argv)

    locked = json.loads(args.protocol.read_text(encoding="utf-8"))["code_fingerprint"]
    files = campaign_fingerprint_files(DEFAULT_PID)
    current = code_fingerprint(files)
    n_src = sum(1 for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts)
    print(f"files  : {n_src} src/**/*.py + {', '.join(files)}")
    print(f"locked : {locked}")
    print(f"current: {current}")
    if current != locked:
        print("MISMATCH: the covered files differ from the locked code. A changed line "
              "ending is the usual cause; .gitattributes stores these files unconverted.")
        return 1
    print("OK: this checkout is the code that produced the v3 campaign.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
