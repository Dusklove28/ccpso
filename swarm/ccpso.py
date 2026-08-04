from learning_ddpg.swarm.base_swarm import BaseSwarm

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
                 max_fe):
        super().__init__(particles, dimensions, fun, lower_bound, upper_bound, max_fe)
        self.previous_positions = None
        self.x_old = None

    # 重置
    def _reset_algorithm_state(self):
        self.previous_positions = self.positions.copy()
        # self.current_conv = 0.75

        # 上一个 Stage 实际使用的 conv
        self.previous_stage_conv = None
        self.stage_index = 0


    def run_stage(self, c_target, stage_fe, smoothing_alpha, max_delta_c):
        c_target = float(c_target)
        stage_fe = int(stage_fe)
        smoothing_alpha = float(smoothing_alpha)
        max_delta_c = float(max_delta_c)

        if stage_fe <= 0:
            raise ValueError("stage_fe must be positive")
        if not 0.0 <= smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in [0, 1]")
        if max_delta_c < 0.0:
            raise ValueError("max_delta_c must be non-negative")

        c_prev = self.previous_stage_conv
        if self.previous_stage_conv is None:
            # 第一个 Stage 没有历史 Stage，不进行平滑
            c_hold = c_target
        else:
            c_hold = (
                    self.previous_stage_conv
                    + smoothing_alpha
                    * (c_target - self.previous_stage_conv)
            )

            delta = float(
                np.clip(
                    c_hold - self.previous_stage_conv,
                    -max_delta_c,
                    max_delta_c,
                )
            )
            c_hold = self.previous_stage_conv + delta

        planned_steps = max(1, int(np.ceil(stage_fe / self.particles)))
        remaining_fe = max(self.max_fe - self.fe_count, 0)
        remaining_steps = max(remaining_fe // self.particles, 0)
        available_steps = min(planned_steps, remaining_steps)
        if available_steps <= 0:
            raise RuntimeError("No FE budget remains for another CCPSO stage")

        old_fe = self.fe_count
        self.stage_index += 1

        for _ in range(available_steps):
            self.step(conv=c_hold)

        self.previous_stage_conv = float(c_hold)

        return {
            "stage_index": int(self.stage_index),
            "stage_fe": int(self.fe_count - old_fe),
            "stage_inner_steps": int(available_steps),
            "stage_c_target": float(c_target),
            "stage_c_prev": (
                None if c_prev is None else float(c_prev)
            ),
            "stage_c_hold": float(c_hold),
            "stage_c_delta": (
                0.0 if c_prev is None
                else float(c_hold - c_prev)
            ),
            "stage_smoothing_alpha": float(smoothing_alpha),
        }

    def step(self, conv):
        current_positions = self.positions.copy()

        if self.fe_count + self.particles > self.max_fe:
            raise RuntimeError("剩余 FE 不足以评估完整种群")

        # 参数初始化
        w = W
        c1 =C1
        c2 = C2
        r1 = self.rng.uniform(0, 1, (self.particles, self.dimensions))
        r2 = self.rng.uniform(0, 1, (self.particles, self.dimensions))

        c1_r1 = c1 * r1
        c2_r2 = c2 * r2
        c_sum = c1_r1 + c2_r2
        # 3. 计算榜样项Q
        q = (c1_r1 * self.pbest_positions + c2_r2 * self.gbest_position) / (c_sum + 1e-16)

        a1 = 1 + w - c_sum
        a2 = -w

        p = a1 * (self.positions - q) + a2 * (self.previous_positions - q)
        # 4.根据二阶公式生成新位置
        x = q + conv * p

        # 5. 边界处理-位置，速度无需考虑
        # implicit_vs = X - self.positions
        # implicit_vs = np.clip(implicit_vs, self.min_v, self.max_v)
        new_x = np.clip(x, self.lower_bound, self.upper_bound)

        # 6.更新 previous_positions
        self.previous_positions = current_positions
        self.positions = new_x.copy()
        # 7. 计算新fitness
        self.fitness = self.evaluate(new_x)
        # 8. 更新pbest&gbest
        self.update_bests(self.fitness)


