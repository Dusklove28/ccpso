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


def calculate_initial_improvement_scale(
    initial_fitness,
    floor=1e-12,
):
    """由初始种群适应度的稳健离散程度计算改善尺度。"""
    floor = _require_finite_real("floor", floor)
    if floor <= 0.0:
        raise ValueError(
            f"floor 必须大于 0，实际值为 {floor!r}"
        )

    if not isinstance(initial_fitness, np.ndarray):
        raise ValueError(
            "initial_fitness 必须是一维、非空的 float64 NumPy 数组，"
            f"实际类型为 {type(initial_fitness).__name__}"
        )
    if initial_fitness.dtype != np.float64:
        raise ValueError(
            "initial_fitness 必须具有 float64 dtype，实际 dtype 为 "
            f"{initial_fitness.dtype}"
        )
    if initial_fitness.ndim != 1 or initial_fitness.size == 0:
        raise ValueError(
            "initial_fitness 必须是一维且非空，实际 shape 为 "
            f"{initial_fitness.shape}"
        )
    if not np.all(np.isfinite(initial_fitness)):
        invalid_indices = np.argwhere(~np.isfinite(initial_fitness))
        first_index = int(invalid_indices[0, 0])
        raise ValueError(
            "initial_fitness 必须全部有限，"
            f"索引 {first_index} 的值为 "
            f"{initial_fitness[first_index]!r}"
        )

    best = float(np.min(initial_fitness))
    median = float(np.median(initial_fitness))
    with np.errstate(over="ignore", invalid="ignore"):
        percentile_25, percentile_75 = np.percentile(
            initial_fitness,
            [25.0, 75.0],
        )
        median_distance = float(np.subtract(median, best))
        iqr = float(np.subtract(percentile_75, percentile_25))
    if not math.isfinite(median_distance):
        raise FloatingPointError(
            "median - best 计算得到非有限值："
            f"{median_distance!r}"
        )
    if not math.isfinite(iqr):
        raise FloatingPointError(
            f"iqr 计算得到非有限值：{iqr!r}"
        )

    scale = float(max(median_distance, iqr, floor))
    if not math.isfinite(scale) or scale <= 0.0:
        raise FloatingPointError(
            f"initial_improvement_scale 必须有限且大于 0，实际值为 {scale!r}"
        )
    return scale


def linear_improvement_reward(
    old_best,
    new_best,
    initial_improvement_scale,
):
    """计算不裁剪的线性 gbest 改善奖励。"""
    old_best = _require_finite_real("old_best", old_best)
    new_best = _require_finite_real("new_best", new_best)
    initial_improvement_scale = _require_finite_real(
        "initial_improvement_scale",
        initial_improvement_scale,
    )
    if initial_improvement_scale <= 0.0:
        raise ValueError(
            "initial_improvement_scale 必须大于 0，实际值为 "
            f"{initial_improvement_scale!r}"
        )

    with np.errstate(over="ignore", invalid="ignore"):
        improvement = float(np.subtract(old_best, new_best))
    if not math.isfinite(improvement):
        raise FloatingPointError(
            f"improvement 必须是有限值，实际值为 {improvement!r}"
        )

    with np.errstate(over="ignore", invalid="ignore"):
        reward = float(np.divide(improvement, initial_improvement_scale))
    if not math.isfinite(reward):
        raise FloatingPointError(
            f"reward 必须是有限值，实际值为 {reward!r}"
        )
    return reward


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
