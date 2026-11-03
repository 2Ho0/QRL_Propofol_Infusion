"""Environment seed ranges that keep training runs and cohorts independent.

An environment seed fixes a simulated patient's random effects, its BIS noise
and, through DualDrugEnv's global NumPy seeding, the warm-up exploration
doses. Evaluation uses ``evaluation_seed + caseid`` (VitalDB case ids are
below 10,000), so training episodes and the checkpoint cohort draw from
separate, non-overlapping ranges. Each training seed gets its own block, so
two runs never share a simulated patient.
"""

TRAINING_BASE = 10_000_000
CHECKPOINT_BASE = 20_000_000
BLOCK = 1_000


def training_env_seed(training_seed: int, episode: int) -> int:
    """Seed of simulator episode ``episode`` in training run ``training_seed``."""
    if not 0 <= episode < BLOCK:
        raise ValueError(f"episode index must be in [0, {BLOCK})")
    return TRAINING_BASE + BLOCK * int(training_seed) + int(episode)


def checkpoint_env_seed(validation_seed: int, index: int) -> int:
    """Seed of checkpoint-cohort patient ``index`` (shared by all runs)."""
    if not 0 <= index < BLOCK:
        raise ValueError(f"cohort index must be in [0, {BLOCK})")
    return CHECKPOINT_BASE + BLOCK * int(validation_seed) + int(index)
