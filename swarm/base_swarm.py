# 统一群体智能算法继承基类，实现初始化
from abc import ABC,abstractmethod

import numpy as np

class BaseSwarm(ABC):
    def __init__(self,
                 particles,
                 dimensions,
                 fun,
                 lower_bound,
                 upper_bound,
                 max_fe,
                 seed=None,):

        self.fun = fun
        self.particles = int(particles)
        self.dimensions = int(dimensions)
        self.lower_bound = np.broadcast_to(
            np.asarray(lower_bound, dtype=np.float64),
            (self.dimensions,),
        ).copy()
        self.upper_bound = np.broadcast_to(
            np.asarray(upper_bound, dtype=np.float64),
            (self.dimensions,),
        ).copy()

        if np.any(self.lower_bound >= self.upper_bound):
            raise ValueError(
                "lower_bound 必须逐维小于 upper_bound"
            )
        self.min_v = None
        self.max_v = None
        self.max_fe = max_fe

        self.rng = np.random.default_rng(seed)

        self.positions = None
        self.fitness = None

        self.pbest_positions = None
        self.pbest_fitness = None

        self.gbest_position = None
        self.gbest_fitness = None

        self.fe_count = 0
        self.part = 0


    def reset(self):
        self.fe_count = 0

        self.positions = self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(
                self.particles,
                self.dimensions,
            ),
        )

        self.fitness = self.evaluate(self.positions)

        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = self.fitness.copy()

        best_index = np.argmin(self.fitness)

        self.gbest_position = self.positions[
            best_index
        ].copy()

        self.gbest_fitness = float(
            self.fitness[best_index]
        )

        return self._reset_algorithm_state()

    def evaluate(self,positions):
        positions = np.asarray(
            positions,
            dtype=np.float64,
        )

        if (
            positions.ndim != 2
            or positions.shape[1] != self.dimensions
        ):
            raise ValueError(
                f"positions 实际 shape 为 {positions.shape}，"
                "期望格式为 (n, dimensions)，"
                f"其中 dimensions={self.dimensions}"
            )

        evaluation_count = positions.shape[0]
        if self.fe_count + evaluation_count > self.max_fe:
            raise RuntimeError(
                "FE消耗完毕"
            )

        fitness = self.fun(positions)
        fitness = np.asarray(fitness, dtype=np.float64)
        if fitness.shape != (evaluation_count,):
            raise RuntimeError(
                "数据异常"
            )

        non_finite_indices = np.flatnonzero(~np.isfinite(fitness))
        if non_finite_indices.size:
            invalid_entries = ", ".join(
                f"{int(index)}={float(fitness[index])!r}"
                for index in non_finite_indices
            )
            raise FloatingPointError(
                "fitness 包含非有限值，异常索引和值: "
                f"{invalid_entries}"
            )

        self.fe_count += evaluation_count
        return fitness

    def update_bests(self, new_fitness):

        improved = new_fitness < self.pbest_fitness

        self.pbest_positions[improved] = (
            self.positions[improved]
        )
        self.pbest_fitness[improved] = (
            new_fitness[improved]
        )

        best_index = np.argmin(self.pbest_fitness)
        best_fitness = float(self.pbest_fitness[best_index])

        if best_fitness < self.gbest_fitness:
            self.gbest_fitness = float(best_fitness)
            self.gbest_position = self.pbest_positions[
                best_index
            ].copy()

    def clip_positions(self):
        return np.clip(
            self.positions,
            self.lower_bound,
            self.upper_bound,
        )

    # 暂时返回算法内部状态字典
    def get_state(self):
        return {
            "粒子位置": self.positions.copy(),
            "当前适应度": self.fitness.copy(),
            "gbest位置": self.gbest_position.copy(),
            "gbest适应度": self.gbest_fitness,
            "已评估fe次数": self.fe_count,
        }

    @property
    def done(self):
        return self.fe_count + self.particles > self.max_fe

    @abstractmethod
    def _reset_algorithm_state(self):
        pass

    @abstractmethod
    def step(self, conv):
        pass
