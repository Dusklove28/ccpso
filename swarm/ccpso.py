"""收敛控制粒子群（CCPSO）的群体更新实现。

当前方法将论文式（17）—（18）中的位置更新分解为榜样中心 ``Q`` 与
二阶残差项 ``P``，再以外部给定的收敛控制系数 ``C`` 执行
``X_next = clip(Q + C * P)``。这里的 ``C`` 缩放确定性残差项，并非
高斯采样的标准差。
"""

from  swarm.base_swarm import BaseSwarm

import numpy as np

W = 0.729844
C1 = 1.496180
C2 = 1.496180

class CCPSOSwarm(BaseSwarm):
    """实现 ``X_{t+1} = clip(Q_t + C_t P_t)`` 的 CCPSO 群体。

    ``Q_t`` 是由个体最优与全局最优形成的逐坐标加权中心；``P_t``
    是由当前位置、上一位置和 ``Q_t`` 构成的二阶残差项。环境传入的
    ``conv`` 即研究定义中的 ``C_t``，用于缩放 ``P_t``。
    """

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

    # reset 后清空当代 Q、随机权重与边界投影诊断。
    def _reset_algorithm_state(self):
        self.previous_positions = self.positions.copy()

        self.q_positions=None
        self.c1_r1=None
        self.c2_r2=None
        self.c_sum=None
        self.generation_prepared = False
        self.boundary_clip_ratio = 0.0

    # 计算 Q_t 及 phi_t=c1*r1+c2*r2；权重按坐标独立采样。
    def _calculate_q(self):


        r1 = self.rng.uniform(0, 1, (self.particles, self.dimensions))
        r2 = self.rng.uniform(0, 1, (self.particles, self.dimensions))

        self.c1_r1 = self.c1 * r1
        self.c2_r2 = self.c2 * r2
        self.c_sum = self.c1_r1 + self.c2_r2
        # 正常权重下严格计算 Q_t 的加权中心。
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

        q=self.q_positions # 本次转移使用的 Q_t；预先生成以供动作前状态使用。
        p = a1 * (self.positions - q) + a2 * (self.previous_positions - q)
        # 二阶残差 P_t 由 conv（即 C_t）缩放，形成未投影候选位置。
        candidate_positions = q + conv * p

        # 对 X_{t+1}=Q_t+C_t*P_t 做逐坐标边界投影。
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


