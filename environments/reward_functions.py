"""不依赖环境状态的候选奖励纯函数。"""

import math
from numbers import Real

import numpy as np


def _require_finite_real(name, value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"{name} 必须是有限实数，实际值为 {value!r}"
        )

    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(
            f"{name} 必须是有限实数，实际值为 {value!r}"
        )
    return converted


def _log_gap(best, optimum):
    """返回 ``log(max(best - optimum, 0))``；零 gap 返回 None。"""
    if best <= optimum:
        return None

    # 仅当 best > 0 > optimum 时，直接减法才可能溢出。
    if best > 0.0 and optimum < 0.0:
        magnitude = max(abs(best), abs(optimum))
        scaled_gap = best / magnitude - optimum / magnitude
        return math.log(magnitude) + math.log(scaled_gap)

    return math.log(best - optimum)


def _log_normalized_gap_plus_epsilon(
    best,
    optimum,
    initial_gap_scale,
    epsilon,
):
    log_epsilon = math.log(epsilon)
    log_gap = _log_gap(best, optimum)
    if log_gap is None:
        return log_epsilon

    log_normalized_gap = log_gap - math.log(initial_gap_scale)
    larger = max(log_normalized_gap, log_epsilon)
    smaller = min(log_normalized_gap, log_epsilon)
    return larger + math.log1p(math.exp(smaller - larger))


def log_gap_reduction_reward(
    old_best,
    new_best,
    optimum,
    initial_gap_scale,
    epsilon=1e-12,
):
    """计算不裁剪、路径无关的归一化 log-gap 减少奖励。"""
    old_best = _require_finite_real("old_best", old_best)
    new_best = _require_finite_real("new_best", new_best)
    optimum = _require_finite_real("optimum", optimum)
    initial_gap_scale = _require_finite_real(
        "initial_gap_scale",
        initial_gap_scale,
    )
    epsilon = _require_finite_real("epsilon", epsilon)

    if initial_gap_scale <= 0.0:
        raise ValueError(
            "initial_gap_scale 必须大于 0，实际值为 "
            f"{initial_gap_scale!r}"
        )
    if epsilon <= 0.0:
        raise ValueError(
            f"epsilon 必须大于 0，实际值为 {epsilon!r}"
        )

    old_log_gap = _log_normalized_gap_plus_epsilon(
        old_best,
        optimum,
        initial_gap_scale,
        epsilon,
    )
    new_log_gap = _log_normalized_gap_plus_epsilon(
        new_best,
        optimum,
        initial_gap_scale,
        epsilon,
    )
    reward = float(old_log_gap - new_log_gap)
    if not math.isfinite(reward):
        raise FloatingPointError(
            f"reward 必须是有限值，实际值为 {reward!r}"
        )
    return reward
