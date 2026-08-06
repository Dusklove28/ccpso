# 候选路径无关 log-gap 奖励

## 1. 定义

本轮定义一个尚未接入环境和训练流程的候选纯函数：

```python
log_gap_reduction_reward(
    old_best,
    new_best,
    optimum,
    initial_gap_scale,
    epsilon=1e-12,
)
```

首先定义非负 gap：

\[
\operatorname{gap}(f)=\max(f-f^*,0),
\]

其中 \(f^*\) 是 `optimum`。若 best 因浮点误差略低于理论最优值，
对应 gap 按 0 处理。再使用固定的初始尺度 \(s_0>0\) 归一化：

\[
g_t=\frac{\operatorname{gap}(f_t)}{s_0}.
\]

单步奖励定义为：

\[
r_t=\log(g_t+\epsilon)-\log(g_{t+1}+\epsilon),
\qquad \epsilon>0.
\]

代码不裁剪奖励，也不把负奖励强制置零。因此：

- 新 gap 小于旧 gap 时，\(r_t>0\)；
- gap 不变时，\(r_t=0\)；
- 新 gap 大于旧 gap 时，\(r_t<0\)。

实现使用与上述公式代数等价的稳定对数计算，避免极大有限 gap 与极小
有限尺度相除时先溢出为无穷大；这不改变奖励的数学定义。

## 2. Telescoping 与路径无关性

对任意中间轨迹 \(g_0,g_1,\ldots,g_T\)，未折扣累计奖励为：

\[
\begin{aligned}
\sum_{t=0}^{T-1}
\left[
\log(g_t+\epsilon)-\log(g_{t+1}+\epsilon)
\right]
&=\log(g_0+\epsilon)-\log(g_T+\epsilon).
\end{aligned}
\]

中间项两两抵消。因此在 \(\gamma=1\) 且奖励不裁剪时，累计回报只由
初始 gap 和最终 gap 决定，与一次改善被拆成多少步、经过哪些中间 gap
无关。裁剪单步奖励或把负奖励置零都会破坏这一 telescoping 性质。

当起点相同时，更小的最终 gap 会得到更大的累计回报。若目标函数值、
理论最优值以及 `initial_gap_scale` 同时乘以相同正数，归一化 gap 不变，
因而奖励也不变。

## 3. 与折扣和 n-step 的关系

- 当 \(\gamma<1\) 时，各时刻奖励的权重不同，早期改善仍会获得更高
  权重；此时折扣累计回报不再严格只由起点和终点决定。
- n-step 可以延长奖励向前传播的跨度，改善时间信用传播，但不能自动
  修复奖励目标本身。如果单步奖励偏好错误、被裁剪或丢失负向信息，
  仅增加 n-step 并不会恢复路径无关性。
- 当前 `CCPSOEnv._calculate_reward()` 使用的
  `log1p(step_improvement / scale)` 奖励继续保留为原始一步奖励基线。
- 本文档中的候选函数尚未接入 `CCPSOEnv`、TD3 或任何训练配置，因此
  当前实验与训练行为均不改变。
