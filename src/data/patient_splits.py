"""Patient-level (grouped) data splitting.

Splits must never place transitions from one patient into more than one
partition. Consecutive transitions within a case are highly correlated, so a
transition-level split leaks the test set into training and inflates held-out
performance.

See docs/superpowers/specs/2026-08-09-paper-code-reconciliation-design.md (D3).
"""
from typing import Tuple

import numpy as np


def split_by_patient(
    caseids: np.ndarray,
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split transition indices into train/val/test, grouped by patient.

    Args:
        caseids: Per-transition VitalDB case id, shape (N,).
        ratios: (train, val, test) proportions of *patients*, summing to 1.
        seed: Seed controlling which patients land in which split.

    Returns:
        Three arrays of indices into `caseids`, disjoint and jointly covering
        every transition.

    Raises:
        ValueError: If `ratios` does not sum to 1, or fewer than 3 patients are
            present, since each split must be non-empty.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    caseids = np.asarray(caseids)
    unique = np.unique(caseids)
    if len(unique) < 3:
        raise ValueError(
            f"need at least 3 patients to fill train/val/test, got {len(unique)}"
        )

    shuffled = np.random.default_rng(seed).permutation(unique)
    n = len(shuffled)

    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    # Guarantee every split is non-empty regardless of rounding.
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)

    groups = (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_val],
        shuffled[n_train + n_val:],
    )
    return tuple(np.flatnonzero(np.isin(caseids, g)) for g in groups)


def sample_online_patients(
    train_ids: np.ndarray,
    n_episodes: int,
    seed: int,
) -> np.ndarray:
    """Choose one distinct training case per simulator episode.

    The v1 online stage used synthetic patients with uniform demographics.
    In sim v2 every episode uses a different subject from the training split,
    drawn with a generator seeded by the training seed only, so the choice
    never depends on validation or test data.

    Args:
        train_ids: Case ids of the training partition.
        n_episodes: Number of simulator episodes.
        seed: Training seed.

    Returns:
        ``n_episodes`` distinct case ids, in episode order.
    """
    ids = np.unique(np.asarray(train_ids))
    if n_episodes > len(ids):
        raise ValueError(
            f"{n_episodes} episodes need distinct subjects, only {len(ids)} available")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 20261019]))
    return rng.choice(ids, size=n_episodes, replace=False)
