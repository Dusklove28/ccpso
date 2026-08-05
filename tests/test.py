import numpy as np

from cec2017.functions import f5

samples = 3
dimension = 50
x = np.random.uniform(-100, 100, size=(samples, dimension))
val = f5(x)
for i in range(samples):
    print(f"f5(x_{i}) = {val[i]:.6f}")
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


def _log_unit_scale(
    value: float,
    floor: float = 1e-8,
) -> float:
    """
    将 [0, 1] 内可能非常小的数值进行对数拉伸。

    floor 及以下映射为 0；
    1 映射为 1。

    用于轨迹移动状态，避免后期大量数值都接近 0。
    """
    value = float(np.clip(value, 0.0, 1.0))

    if value <= floor:
        return 0.0

    log_floor = np.log10(floor)

    return float(
        np.clip(
            (np.log10(value) - log_floor)
            / (0.0 - log_floor),
            0.0,
            1.0,
        )
    )


class CCPSOEnv(gym.Env):
    """
    强化学习控制 CCPSO 收敛性系数 C 的环境。

    一个 RL step 对应 CCPSO 完成一代群体更新。

    状态：
        0. FE 进度
        1. 近期对数最优改善
        2. 粒子位置多样性
        3. Q 多样性
        4. 轨迹移动强度
        5. 停滞比例

    动作：
        Actor 输出 [-1, 1]，映射到 [c_min, c_max]。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        swarm,
        c_min: float = 0.0,
        c_max: float = 1.5,
        optimum: float = 0.0,
        function_id: int | None = None,
        recent_window: int = 5,
        stagnation_horizon: int = 10,
        stagnation_abs_tol: float = 1e-12,
        stagnation_rel_tol: float = 1e-12,
        movement_log_floor: float = 1e-8,
    ):
        super().__init__()


        self.swarm = swarm

        # C 的动作范围
        self.c_min = float(c_min)
        self.c_max = float(c_max)

        # optimum 不进入状态，本版本仅用于最终误差日志
        self.optimum = float(optimum)

        self.function_id = (
            None if function_id is None else int(function_id)
        )

        self.recent_window = int(recent_window)
        self.stagnation_horizon = int(stagnation_horizon)

        self.stagnation_abs_tol = float(stagnation_abs_tol)
        self.stagnation_rel_tol = float(stagnation_rel_tol)

        self.movement_log_floor = float(movement_log_floor)

        # Actor 输出归一化连续动作
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 六个状态全部限制在 [0, 1]
        self.observation_space = gym.spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )

        # 当前状态统计量
        self.mean_movement = 0.0
        self.recent_progress = 0.0
        self.stagnation_steps = 0

        # 只需要保存 recent_window + 1 个 gbest
        self.best_history: list[float] = []

        # 用初始种群适应度分布构造固定尺度
        # 不依赖理论最优值，同时消除 CEC 函数平移常数的影响
        self.initial_fitness_scale = 1.0

    # --------------------------------------------------
    # 动作转换
    # --------------------------------------------------

    def _action_to_c(
        self,
        action: np.ndarray,
    ) -> tuple[float, float]:
        action_array = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action_array.size != 1:
            raise ValueError(
                f"动作必须只包含一个元素，实际形状为 "
                f"{np.asarray(action).shape}"
            )

        raw_action = float(
            np.clip(action_array[0], -1.0, 1.0)
        )

        normalized_action = (raw_action + 1.0) / 2.0

        c_value = (
            self.c_min
            + normalized_action
            * (self.c_max - self.c_min)
        )

        return float(c_value), raw_action

    # --------------------------------------------------
    # Q 准备
    # --------------------------------------------------

    def _prepare_generation(self) -> None:
        """
        在 Actor 观察状态前生成本代 Q。

        必须保证：
            观察到的 Q
            与执行 X = Q + C * P 时使用的 Q
            是同一个 Q。
        """
        if not hasattr(self.swarm, "_calculate_q"):
            raise AttributeError(
                "swarm 必须实现 _calculate_q()"
            )

        if not self.swarm.generation_prepared:
            self.swarm._calculate_q()

        if self.swarm.q_positions is None:
            raise RuntimeError("Q 计算失败：q_positions 为 None")

    # --------------------------------------------------
    # 初始适应度尺度
    # --------------------------------------------------

    def _calculate_initial_fitness_scale(self) -> float:
        """
        使用初始种群的适应度分布确定固定归一化尺度。

        不使用理论最优值。

        IQR 对异常值相对稳健，同时对 CEC 函数中的加性偏移
        具有不变性。
        """
        fitness = np.asarray(
            self.swarm.fitness,
            dtype=np.float64,
        )

        q25, q75 = np.percentile(
            fitness,
            [25.0, 75.0],
        )

        iqr = float(q75 - q25)

        median_to_best = abs(
            float(np.median(fitness))
            - float(self.swarm.gbest_fitness)
        )

        return max(
            iqr,
            median_to_best,
            1e-12,
        )

    # --------------------------------------------------
    # 近期改善
    # --------------------------------------------------

    def _calculate_recent_progress(self) -> float:
        """
        计算最近 W 代的对数最优改善。

        I_t = tanh(
            log(
                1 + (f_{t-W} - f_t)_+ / fitness_scale
            )
        )

        特点：
        1. 不使用理论最优值；
        2. 使用差值，因此不受 CEC 加性偏移影响；
        3. 使用初始适应度分布归一化；
        4. 输出位于 [0, 1)。
        """
        if len(self.best_history) < 2:
            return 0.0

        history_index = max(
            0,
            len(self.best_history) - 1 - self.recent_window,
        )

        old_best = float(
            self.best_history[history_index]
        )

        current_best = float(
            self.best_history[-1]
        )

        improvement = max(
            old_best - current_best,
            0.0,
        )

        scaled_improvement = (
            improvement
            / max(self.initial_fitness_scale, 1e-12)
        )

        log_improvement = np.log1p(
            scaled_improvement
        )

        return float(
            np.clip(
                np.tanh(log_improvement),
                0.0,
                1.0,
            )
        )

    # --------------------------------------------------
    # 停滞判断
    # --------------------------------------------------

    def _is_meaningful_improvement(
        self,
        old_best: float,
        new_best: float,
    ) -> bool:
        """
        判断本代是否发生“有意义”的 gbest 改善。

        不直接使用 new_best < old_best，
        避免浮点数微小波动不断清零停滞计数。
        """
        improvement = float(old_best - new_best)

        threshold = (
            self.stagnation_abs_tol
            + self.stagnation_rel_tol
            * max(self.initial_fitness_scale, 1e-12)
        )

        return improvement > threshold

    def _update_progress_state(
        self,
        old_best: float,
        new_best: float,
    ) -> None:
        """
        在完成一代群体更新后，统一更新：

        1. gbest 历史；
        2. 停滞步数；
        3. 近期对数改善。
        """
        if self._is_meaningful_improvement(
            old_best,
            new_best,
        ):
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1

        self.best_history.append(float(new_best))

        max_history_length = self.recent_window + 1

        if len(self.best_history) > max_history_length:
            self.best_history = self.best_history[
                -max_history_length:
            ]

        self.recent_progress = (
            self._calculate_recent_progress()
        )

    # --------------------------------------------------
    # 状态
    # --------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        positions = np.asarray(
            self.swarm.positions,
            dtype=np.float64,
        )

        q_positions = np.asarray(
            self.swarm.q_positions,
            dtype=np.float64,
        )

        if positions.shape != q_positions.shape:
            raise ValueError(
                "positions 和 q_positions 的形状必须相同，"
                f"实际为 {positions.shape} 与 "
                f"{q_positions.shape}"
            )

        search_diagonal = float(
            np.linalg.norm(
                self.swarm.upper_bound
                - self.swarm.lower_bound
            )
        )

        search_diagonal = max(
            search_diagonal,
            1e-12,
        )

        # state 1：函数评价进度
        fe_progress = np.clip(
            self.swarm.fe_count
            / max(self.swarm.max_fe, 1),
            0.0,
            1.0,
        )

        # state 2：近期对数最优改善
        recent_progress = np.clip(
            self.recent_progress,
            0.0,
            1.0,
        )

        # state 3：粒子位置多样性 D_X
        position_center = np.mean(
            positions,
            axis=0,
        )

        mean_position_distance = float(
            np.mean(
                np.linalg.norm(
                    positions - position_center,
                    axis=1,
