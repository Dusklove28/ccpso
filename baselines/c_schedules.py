"""Deterministic, non-learning schedules for the CCPSO C parameter."""

import numpy as np


def _require_finite_number(name, value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
    ):
        raise ValueError(
            f"{name} must be a finite number, got {value!r}"
        )
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(
            f"{name} must be a finite number, got {value!r}"
        )
    return result


def _require_positive_integer(name, value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer, got {value!r}"
        )
    return int(value)


def make_fixed_c_schedule(c_value, num_steps):
    """Return ``num_steps`` copies of one fixed C value."""
    validated_c = _require_finite_number("c_value", c_value)
    validated_steps = _require_positive_integer("num_steps", num_steps)
    return np.full(
        (validated_steps,),
        validated_c,
        dtype=np.float64,
    )


def make_linear_c_schedule(c_start, c_end, num_steps):
    """Return the discrete schedule named ``linear_c``.

    For step indices ``0, ..., num_steps - 1`` and more than one step,
    ``progress = step_index / (num_steps - 1)`` and
    ``C = c_start + progress * (c_end - c_start)``.
    """
    validated_start = _require_finite_number("c_start", c_start)
    validated_end = _require_finite_number("c_end", c_end)
    validated_steps = _require_positive_integer("num_steps", num_steps)

    if validated_steps == 1:
        return np.array([validated_start], dtype=np.float64)

    step_indices = np.arange(validated_steps, dtype=np.float64)
    progress = step_indices / float(validated_steps - 1)
    with np.errstate(over="ignore", invalid="ignore"):
        schedule = (
            validated_start
            + progress * (validated_end - validated_start)
        )
    if not np.all(np.isfinite(schedule)):
        raise ValueError(
            "linear_c schedule produced non-finite values from "
            f"c_start={validated_start!r}, c_end={validated_end!r}"
        )

    schedule[0] = validated_start
    schedule[-1] = validated_end
    return schedule


def c_value_to_action(c_value, c_min, c_max):
    """Invert the environment's action-to-C mapping without clipping."""
    validated_c = _require_finite_number("c_value", c_value)
    validated_min = _require_finite_number("c_min", c_min)
    validated_max = _require_finite_number("c_max", c_max)

    if validated_min > validated_max:
        raise ValueError(
            "c_min must be less than or equal to c_max, "
            f"got c_min={validated_min!r}, c_max={validated_max!r}"
        )
    if validated_c < validated_min or validated_c > validated_max:
        raise ValueError(
            "c_value must lie within [c_min, c_max], "
            f"got c_value={validated_c!r}, "
            f"c_min={validated_min!r}, c_max={validated_max!r}"
        )

    if validated_min == validated_max:
        return 0.0
    if validated_c == validated_min:
        return -1.0
    if validated_c == validated_max:
        return 1.0

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        action = (
            2.0
            * (validated_c - validated_min)
            / (validated_max - validated_min)
            - 1.0
        )
    action = float(action)
    if not np.isfinite(action) or not -1.0 <= action <= 1.0:
        raise ValueError(
            "C-to-action mapping produced an invalid result from "
            f"c_value={validated_c!r}, c_min={validated_min!r}, "
            f"c_max={validated_max!r}"
        )
    return action
