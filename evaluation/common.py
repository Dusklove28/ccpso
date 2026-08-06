"""Shared validation and metadata helpers for optimization evaluation."""

from collections.abc import Sequence

import numpy as np

from problems.metadata import serialize_problem


STATE_FIELDS = (
    "state_fe_progress",
    "state_recent_progress",
    "state_position_diversity",
    "state_q_diversity",
    "state_movement",
    "state_stagnation",
)


def validate_evaluation_seeds(seeds):
    if isinstance(seeds, np.ndarray):
        if seeds.ndim != 1:
            raise ValueError(
                "seeds must be a non-empty one-dimensional integer sequence"
            )
        seed_values = seeds.tolist()
    elif isinstance(seeds, Sequence) and not isinstance(
        seeds,
        (str, bytes),
    ):
        seed_values = list(seeds)
    else:
        raise ValueError("seeds must be a non-empty integer sequence")

    if not seed_values:
        raise ValueError("seeds must be a non-empty integer sequence")

    validated = []
    for index, seed in enumerate(seed_values):
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
        ):
            raise ValueError(
                "seeds must contain only integers; "
                f"seeds[{index}]={seed!r}"
            )
        validated.append(int(seed))
    return validated
