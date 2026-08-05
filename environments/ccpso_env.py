import gymnasium as gym
import numpy as np


# 计算移动程度，对数
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
        self.function_id = None if function_id is None else int(function_id)

        self.recent_window = int(recent_window)
        self.stagnation_horizon = int(stagnation_horizon)

        self.stagnation_abs_tol = float(stagnation_abs_tol)
        self.stagnation_rel_tol = float(stagnation_rel_tol)

        self.movement_log_floor = float(movement_log_floor)

        # Actor 输出归一化连续动作
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
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

    def _action_to_conv(self, action):
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action.size != 1:
            raise ValueError(
                f"动作必须只包含一个元素，实际形状为 "
                f"{np.asarray(action).shape}"
            )

        action_value = float(action[0])
        if not np.isfinite(action_value):
            raise ValueError(
                f"动作必须是有限值，实际值为 {action_value!r}"
            )

        raw_action = float(np.clip(action_value, -1.0, 1.0))

        normalized = (raw_action + 1.0) / 2.0

        conv = (self.c_min+ normalized* (self.c_max - self.c_min))

        return float(conv), raw_action

    # 计算实际使用Q
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

        if not self.swarm.generation_prepared and not self.swarm.done:
            self.swarm._calculate_q()

        if self.swarm.q_positions is None:
            raise RuntimeError("Q 计算失败：q_positions 为 None")

    def _get_info(
            self,
            *,
            conv=None,
            raw_action=None,
            reward_progress=0.0,
            terminal=False,
    ):
        return {
            "fe_count": int(self.swarm.fe_count),
            "gbest_fitness": float(self.swarm.gbest_fitness),
            "gap": max(
                float(self.swarm.gbest_fitness) - self.optimum,
                0.0,
            ),
            "function_id": self.function_id,
            "conv": None if conv is None else float(conv),
            "raw_action": (
                None if raw_action is None
                else float(raw_action)
            ),
            "reward_progress": float(reward_progress),
            "mean_movement": float(self.mean_movement),
            "recent_progress": float(self.recent_progress),
            "stagnation_steps": int(self.stagnation_steps),
            "terminal": bool(terminal),
        }

    # 计算提升归一化的分母
    def _calculate_initial_fitness_scale(self) -> float:
        initial_fitness = np.asarray(
            self.swarm.fitness,
            dtype=np.float64,
        )

        initial_best = float(np.min(initial_fitness))
        initial_median = float(np.median(initial_fitness))

        return max(
            initial_median - initial_best,
            1e-12,
        )

    # 计算最近提升
    def _calculate_recent_progress(self) -> float:
        if len(self.best_history) < 2:
            return 0.0

        start_index = max(
            0,
            len(self.best_history) - 1 - self.recent_window,
        )

        old_best = float(self.best_history[start_index])
        new_best = float(self.best_history[-1])

        improvement = max(
            old_best - new_best,
            0.0,
        )

        normalized_improvement = (
                improvement
                / max(self.initial_fitness_scale, 1e-12)
        )

        return float(
            np.clip(
                np.tanh(np.log1p(normalized_improvement)),
                0.0,
                1.0,
            )
        )

    # 停滞判断
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

    # 统一更新
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

    def _get_observation(self):
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
                )
            )
        )

        position_diversity = np.clip(
            mean_position_distance / search_diagonal,
            0.0,
            1.0,
        )

        # state 4：Q 多样性 D_Q
        q_center = np.mean(
            q_positions,
            axis=0,
        )

        mean_q_distance = float(
            np.mean(
                np.linalg.norm(
                    q_positions - q_center,
                    axis=1,
                )
            )
        )

        q_diversity = np.clip(
            mean_q_distance / search_diagonal,
            0.0,
            1.0,
        )

        # state 5：上一代实际轨迹移动强度
        movement_ratio = np.clip(
            self.mean_movement / search_diagonal,
            0.0,
            1.0,
        )

        movement_state = _log_unit_scale(
            movement_ratio,
            floor=self.movement_log_floor,
        )

        # state 6：停滞比例
        stagnation_ratio = np.clip(
            self.stagnation_steps
            / max(self.stagnation_horizon, 1),
            0.0,
            1.0,
        )

        observation = np.asarray(
            [
                fe_progress,
                recent_progress,
                position_diversity,
                q_diversity,
                movement_state,
                stagnation_ratio,
            ],
            dtype=np.float32,
        )

        if not np.all(np.isfinite(observation)):
            raise FloatingPointError(
                "observation 中出现 NaN 或 Inf："
                f"{observation}"
            )

        return np.clip(
            observation,
            0.0,
            1.0,
        ).astype(np.float32)

    # 重点：奖励函数设计
    def _calculate_reward(self, old_best, new_best) -> tuple[float,float]:
        """
                使用单步对数相对改善作为基础奖励。

                不使用理论最优值。

                后续研究多步信用传播时，只修改 TD target，
                暂时不混入多样性奖励或阶段奖励。
                """
        improvement = max(
            float(old_best - new_best),
            0.0,
        )

        scaled_improvement = (
                improvement
                / max(self.initial_fitness_scale, 1e-12)
        )

        progress = float(
            np.log1p(scaled_improvement)
        )

        reward = float(
            np.clip(
                progress,
                0.0,
                5.0,
            )
        )

        return reward, progress

    # Gymnasium 接口
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.swarm.rng = np.random.default_rng(seed)

        self.swarm.reset()

        # 计算初始适应度归一化尺度
        self.initial_fitness_scale = (
            self._calculate_initial_fitness_scale()
        )

        self.mean_movement = 0.0
        self.recent_progress = 0.0
        self.stagnation_steps = 0

        self.best_history = [
            float(self.swarm.gbest_fitness)
        ]

        # Actor 观察状态前，必须先准备本代 Q
        self._prepare_generation()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        if self.swarm.done:
            raise RuntimeError(
                "Episode 已结束，请先调用 reset()"
            )

        old_best = self.swarm.gbest_fitness

        old_positions = np.asarray(
            self.swarm.positions,
            dtype=np.float64,
        ).copy()

        conv, raw_action = self._action_to_conv(
            action
        )


        #一次PSO种群更新,使用 observation 中对应的同一个 Q
        self.swarm.step(conv)

        new_best = self.swarm.gbest_fitness

        new_positions = np.asarray(
            self.swarm.positions,
            dtype=np.float64,
        )

        # 当前 C 执行后产生的实际平均移动距离
        self.mean_movement = float(
            np.mean(
                np.linalg.norm(
                    new_positions - old_positions,
                    axis=1,
                )
            )
        )

        self._update_progress_state(
            old_best,
            new_best,
        )

        reward, progress = (
            self._calculate_reward(
                old_best,
                new_best,
            )
        )

        # step() 后 generation_prepared=False，
        # 为下一次 Actor 决策准备新的 Q
        self._prepare_generation()
        observation = self._get_observation()

        # FE预算属于当前有限时域优化任务的终止条件
        terminated = bool(self.swarm.done)
        truncated = False

        info = self._get_info(
            conv=conv,
            raw_action=raw_action,
            reward_progress=progress,
            terminal=terminated,
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
