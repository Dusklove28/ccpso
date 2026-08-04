import gymnasium as gym
import numpy as np


class CCPSOEnv(gym.Env):
    def __init__(
        self,
        swarm,
        conv_min=0.0,
        conv_max=1.5,
        optimum=0.0,
        function_id=None,
        target_tolerance=1e-12,
        stagnation_horizon=20,
        stage_fe=100,
        stage_action_mode="c_target_hold",
        stage_smoothing_alpha=0.5,
        stage_max_delta_c=0.2,
    ):
        super().__init__()

        self.swarm = swarm
        self.conv_min = float(conv_min)
        self.conv_max = float(conv_max)
        self.optimum = float(optimum)
        self.function_id = None if function_id is None else int(function_id)
        self.target_tolerance = float(target_tolerance)
        self.stagnation_horizon = int(stagnation_horizon)
        self.stage_fe = int(stage_fe)
        self.stage_action_mode = str(stage_action_mode)
        self.stage_smoothing_alpha = float(stage_smoothing_alpha)
        self.stage_max_delta_c = float(stage_max_delta_c)

        if self.stage_fe <= 0:
            raise ValueError("stage_fe must be positive")
        if self.stage_action_mode != "c_target_hold":
            raise ValueError(
                f"unsupported stage_action_mode: {self.stage_action_mode}"
            )
        if not 0.0 <= self.stage_smoothing_alpha <= 1.0:
            raise ValueError("stage_smoothing_alpha must be in [0, 1]")
        if self.stage_max_delta_c < 0.0:
            raise ValueError("stage_max_delta_c must be non-negative")


        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

        self.current_conv = 0.75
        self.previous_gbest = None
        self.recent_progress = 0.0
        self.stagnation_steps = 0

    def _action_to_conv(self, action):
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action.size != 1:
            raise ValueError(f"action shape error: {action.shape}")

        raw_action = float(np.clip(action[0], -1.0, 1.0))

        normalized = (raw_action + 1.0) / 2.0

        conv = (self.conv_min+ normalized* (self.conv_max - self.conv_min))

        return float(conv), raw_action

    def _get_observation(self):
        # 迭代进度
        fe_progress = np.clip(self.swarm.fe_count / self.swarm.max_fe,0.0,1.0,)

        # 粒子加权中心
        center = np.mean(self.swarm.positions,axis=0,)

        # 离Q的距离
        mean_distance = np.mean(np.linalg.norm(self.swarm.positions - center,axis=1,))

        #
        search_diagonal = np.linalg.norm(self.swarm.upper_bound- self.swarm.lower_bound)

        # 多样性
        diversity = np.clip(mean_distance / max(search_diagonal, 1e-12),0.0,1.0,)

        conv_normalized = (
                (self.current_conv - self.conv_min)
                / max(self.conv_max - self.conv_min, 1e-12)
        )

        stagnation_ratio = np.clip(self.stagnation_steps/ max(self.stagnation_horizon, 1),0.0,1.0,)

        observation = np.array(
            [
                fe_progress,
                np.clip(self.recent_progress, 0.0, 1.0),
                diversity,
                conv_normalized,
                stagnation_ratio,
            ],
            dtype=np.float32,
        )

        return observation

    @property
    def remaining_stage_steps(self):
        remaining_swarm_steps = max(
            0,
            (self.swarm.max_fe - self.swarm.fe_count)
            // self.swarm.particles,
        )
        swarm_steps_per_stage = max(
            1,
            int(np.ceil(self.stage_fe / self.swarm.particles)),
        )
        return int(
            np.ceil(remaining_swarm_steps / swarm_steps_per_stage)
        )

    def _gap_progress(self, old_best, new_best):
        eps = 1e-12

        old_gap = max(
            float(old_best) - self.optimum,
            eps,
        )
        new_gap = max(
            float(new_best) - self.optimum,
            eps,
        )

        return float(
            np.log(
                (old_gap + eps)
                / (new_gap + eps)
            )
        )

    def _instability(self):
        eps = 1e-12

        at_lower = (
                self.swarm.positions
                <= self.swarm.lower_bound + eps
        )
        at_upper = (
                self.swarm.positions
                >= self.swarm.upper_bound - eps
        )

        boundary_ratio = np.mean(
            np.logical_or(at_lower, at_upper)
        )

        # 对齐旧实现中边界项占不稳定性的 0.5。
        return float(0.5 * boundary_ratio)

    # 重点：奖励函数设计
    def _calculate_reward(self, old_best, new_best):
        progress = self._gap_progress(
            old_best,
            new_best,
        )

        instability = self._instability()

        reward_unclipped = (
                progress
                - 0.05 * instability
        )

        reward = np.clip(
            reward_unclipped,
            -5.0,
            5.0,
        )

        return float(reward), progress, instability

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.swarm.rng = np.random.default_rng(seed)

        self.swarm.reset()

        self.current_conv = 0.75
        self.previous_gbest = self.swarm.gbest_fitness
        self.recent_progress = 0.0
        self.stagnation_steps = 0

        observation = self._get_observation()

        info = {
            "fe_count": self.swarm.fe_count,
            "gbest_fitness": self.swarm.gbest_fitness,
            "gap": max(self.swarm.gbest_fitness - self.optimum, 0.0),
            "conv": self.current_conv,
            "function_id": self.function_id,
            "function_optimum": self.optimum,
        }

        return observation, info

    def step(self, action):
        if self.swarm.done:
            raise RuntimeError(
                "Episode already finished; call reset()."
            )

        # old_best = self.swarm.gbest_fitness
        # conv, raw_action = self._action_to_conv(
        #     action
        # )

        # stage前保存数据
        old_best = self.swarm.gbest_fitness
        c_target, raw_action = self._action_to_conv(action)



        # 切换为stage更新
        stage_info = self.swarm.run_stage(
            c_target=c_target,
            stage_fe=self.stage_fe,
            smoothing_alpha=self.stage_smoothing_alpha,
            max_delta_c=self.stage_max_delta_c,
        )

        self.current_conv = stage_info["stage_c_hold"]

        #一次PSO种群更新
        new_best = self.swarm.gbest_fitness

        reward, progress, instability = (
            self._calculate_reward(
                old_best,
                new_best,
            )
        )

        self.recent_progress = max(progress, 0.0)

        if new_best < old_best:
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1

        observation = self._get_observation()

        gap = max(
            new_best - self.optimum,
            0.0,
        )

        terminated = (
                gap <= self.target_tolerance
        )

        truncated = self.swarm.done

        info = {
            "raw_action": raw_action,
            "conv": self.current_conv,
            "fe_count": self.swarm.fe_count,
            "gbest_fitness": new_best,
            "gap": gap,
            "gbest_progress": progress,
            "instability": instability,
            "function_id": self.function_id,
            "function_optimum": self.optimum,
            **stage_info,
        }

        return (
            observation,
            reward,
            bool(terminated),
            bool(truncated),
            info,
        )
