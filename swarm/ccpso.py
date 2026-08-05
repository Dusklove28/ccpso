from  swarm.base_swarm import BaseSwarm

import numpy as np

W = 0.729844
C1 = 1.496180
C2 = 1.496180

# 1.初始化种群位置、适应度、p/gbest、
# 2.计算榜样项Q、扰动项P
# 3.根据X=Q+P更新新粒子位置
class CCPSOSwarm(BaseSwarm):

    def __init__(self,
                 particles,
                 dimensions,
                 fun,
                 lower_bound,
                 upper_bound,
                 max_fe,
                 seed):
        super().__init__(particles, dimensions, fun, lower_bound, upper_bound, max_fe,seed=seed)
        self.previous_positions = None
        self.fe_count = 0
        self.q_positions = None
        self.c1_r1 = None
        self.c2_r2 = None
        self.c_sum = None
        self.w = W
        self.c1 = C1
        self.c2 = C2
        self.generation_prepared = False
        self.boundary_clip_ratio = 0.0

    # 在reset后，初始化算法特定变量
    def _reset_algorithm_state(self):
        self.previous_positions = self.positions.copy()
        # 初始收敛性系数C
        self.current_conv = 1.5

        self.q_positions=None
        self.c1_r1=None
        self.c2_r2=None
        self.c_sum=None
        self.generation_prepared = False
        self.boundary_clip_ratio = 0.0

    # 3. 计算榜样项Q
    def _calculate_q(self):


        r1 = self.rng.uniform(0, 1, (self.particles, self.dimensions))
        r2 = self.rng.uniform(0, 1, (self.particles, self.dimensions))

        self.c1_r1 = self.c1 * r1
        self.c2_r2 = self.c2 * r2
        self.c_sum = self.c1_r1 + self.c2_r2
        # 3. 计算榜样项Q
        weighted_sum = (
            self.c1_r1 * self.pbest_positions
            + self.c2_r2 * self.gbest_position
        )
        fallback_q = (self.pbest_positions + self.gbest_position) / 2.0
        near_zero = np.isclose(
            self.c_sum,
            0.0,
            rtol=0.0,
            atol=np.finfo(np.float64).eps,
        )
        self.q_positions = fallback_q.copy()
        np.divide(
            weighted_sum,
            self.c_sum,
            out=self.q_positions,
            where=~near_zero,
        )

        self.generation_prepared = True

    def step(self, conv):

        if not self.generation_prepared:
            raise RuntimeError(
                "调用 step() 前必须先计算一次Q"
            )

        # 当前已经消耗FE次数+粒子数
        if self.fe_count + self.particles > self.max_fe:
            raise RuntimeError("剩余 FE 不足以评估完整种群")
        current_positions = self.positions.copy()

        a1 = 1 + self.w - self.c_sum
        a2 = -self.w

        q=self.q_positions # 上一次迭代的Q，初始时会预先计算一次，以便Actor使用
        p = a1 * (self.positions - q) + a2 * (self.previous_positions - q)
        # 4.根据二阶公式生成新位置
        candidate_positions = q + conv * p

        # 5. 边界处理-位置，速度无需考虑
        # implicit_vs = X - self.positions
        # implicit_vs = np.clip(implicit_vs, self.min_v, self.max_v)
        positions = np.clip(
            candidate_positions,
            self.lower_bound,
            self.upper_bound,
        )
        self.boundary_clip_ratio = float(
            np.count_nonzero(candidate_positions != positions)
            / candidate_positions.size
        )

        # 6.更新 previous_positions
        self.previous_positions = current_positions
        self.positions = positions.copy()
        # 7. 计算新fitness
        self.fitness = self.evaluate(positions)
        # 8. 更新pbest&gbest
        self.update_bests(self.fitness)
        # 9.更新标志
        self.generation_prepared = False


