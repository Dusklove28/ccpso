"""Pure state-feature transforms shared by CCPSO environment modes."""

from numbers import Real

import numpy as np


RELATIVE_LOG_K = 8


def _finite_real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def relative_log_transform(ratio, k=RELATIVE_LOG_K):
    """Map a non-negative ratio to [0, 1] using the relative log scale."""
    ratio = _finite_real("ratio", ratio)
    if ratio < 0.0:
        raise ValueError(f"ratio must be >= 0.0, got {ratio!r}")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"k must be a positive integer, got {k!r}")
    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k!r}")
    if ratio == 0.0:
        return 0.0
    value = 0.5 + np.log10(max(ratio, 10.0 ** (-k))) / (2.0 * k)
    result = float(np.clip(value, 0.0, 1.0))
    if not np.isfinite(result):
        raise FloatingPointError("relative log transform is non-finite")
    return result


def mean_radial_diversity(values):
    """Return mean Euclidean distance from a population's coordinate mean."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(
            "values must have non-empty shape (n, dimensions), "
            f"got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("values contain NaN or Infinity")
    center = np.mean(array, axis=0)
    result = float(np.mean(np.linalg.norm(array - center, axis=1)))
    if not np.isfinite(result) or result < 0.0:
        raise FloatingPointError(f"diversity must be finite and non-negative, got {result!r}")
    return result


def safe_initial_spatial_scale(value, search_diagonal, floor_ratio=1e-12):
    """Return a finite positive reference tied to the search-space size."""
    value = _finite_real("value", value)
    search_diagonal = _finite_real("search_diagonal", search_diagonal)
    floor_ratio = _finite_real("floor_ratio", floor_ratio)
    if value < 0.0:
        raise ValueError(f"value must be >= 0.0, got {value!r}")
    if search_diagonal <= 0.0:
        raise ValueError(
            f"search_diagonal must be > 0.0, got {search_diagonal!r}"
        )
    if floor_ratio <= 0.0:
        raise ValueError(f"floor_ratio must be > 0.0, got {floor_ratio!r}")
    floor = search_diagonal * floor_ratio
    if not np.isfinite(floor) or floor <= 0.0:
        raise FloatingPointError(f"spatial scale floor is invalid: {floor!r}")
    return float(max(value, floor))


def log_stagnation_feature(stagnation_steps, max_episode_updates):
    """Normalize stagnation against the episode's complete-update budget."""
    for name, value in (
        ("stagnation_steps", stagnation_steps),
        ("max_episode_updates", max_episode_updates),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer, got {value!r}")
    stagnation_steps = int(stagnation_steps)
    max_episode_updates = int(max_episode_updates)
    if stagnation_steps < 0:
        raise ValueError(
            f"stagnation_steps must be >= 0, got {stagnation_steps!r}"
        )
    if max_episode_updates <= 0:
        raise ValueError(
            "max_episode_updates must be > 0, "
            f"got {max_episode_updates!r}"
        )
    result = np.log1p(min(stagnation_steps, max_episode_updates)) / np.log1p(
        max_episode_updates
    )
    return float(np.clip(result, 0.0, 1.0))


def build_relative_log_state(
    *,
    fe_progress,
    recent_improvement,
    initial_improvement_scale,
    position_diversity,
    initial_position_scale,
    q_diversity,
    initial_q_scale,
    movement,
    stagnation_steps,
    max_episode_updates,
    k=RELATIVE_LOG_K,
):
    """Build the six-dimensional ``relative_log_v2`` state."""
    fe_progress = _finite_real("fe_progress", fe_progress)
    recent_improvement = _finite_real("recent_improvement", recent_improvement)
    initial_improvement_scale = _finite_real(
        "initial_improvement_scale", initial_improvement_scale
    )
    position_diversity = _finite_real("position_diversity", position_diversity)
    initial_position_scale = _finite_real(
        "initial_position_scale", initial_position_scale
    )
    q_diversity = _finite_real("q_diversity", q_diversity)
    initial_q_scale = _finite_real("initial_q_scale", initial_q_scale)
    movement = _finite_real("movement", movement)
    if min(
        recent_improvement,
        position_diversity,
        q_diversity,
        movement,
    ) < 0.0:
        raise ValueError("relative state magnitudes must be non-negative")
    for name, scale in (
        ("initial_improvement_scale", initial_improvement_scale),
        ("initial_position_scale", initial_position_scale),
        ("initial_q_scale", initial_q_scale),
    ):
        if scale <= 0.0:
            raise ValueError(f"{name} must be > 0.0, got {scale!r}")

    observation = np.asarray(
        [
            np.clip(fe_progress, 0.0, 1.0),
            relative_log_transform(
                recent_improvement / initial_improvement_scale, k
            ),
            relative_log_transform(
                position_diversity / initial_position_scale, k
            ),
            relative_log_transform(q_diversity / initial_q_scale, k),
            relative_log_transform(movement / initial_position_scale, k),
            log_stagnation_feature(stagnation_steps, max_episode_updates),
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(observation)):
        raise FloatingPointError(f"relative_log_v2 state is non-finite: {observation}")
    return np.clip(observation, 0.0, 1.0).astype(np.float32)
