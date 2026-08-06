# CCPSO Reward 设计与三种模式

## 1. 研究定位

Sphere FE 敏感性实验用于发现：原始单步 reward 与最终 gap 之间可能存在
目标错位，而且不同 C 策略的排序会随 FE 预算改变。Sphere 是单峰、可
分离函数，不能仅凭该结果决定面向 CEC2017 多峰和混合函数的主奖励。

当前环境支持三个严格名称，不提供别名或大小写兼容：

```text
step_log_improvement
linear_improvement
oracle_log_gap_reduction
```

- `step_log_improvement`：原始短期奖励基线；
- `linear_improvement`：面向 CEC2017 的主候选奖励，不使用理论最优值；
- `oracle_log_gap_reduction`：使用理论最优值的 oracle 诊断消融，不能作为
  默认主奖励。

本轮 reward mode 只接入 `CCPSOEnv` 和环境工厂，尚未进入
`TD3OnlineConfig`、`ClassicTD3ExperimentConfig`、实验 CLI 或正式训练。

## 2. 原始一步奖励：`step_log_improvement`

该模式完整保留现有公式：

\[
\Delta_t=\max(b_t-b_{t+1},0),
\]

\[
p_t=\log\left(1+\frac{\Delta_t}{S_{\mathrm{old}}}\right),
\qquad
r_t=\operatorname{clip}(p_t,0,5).
\]

其中旧尺度仍由初始适应度的 `median - best` 与固定下限构造。这个模式
忽略退化并裁剪大改善，因此不具有严格 telescoping 性质，保留它是为了
形成与既有 TD3 训练完全一致的原始短期奖励基线。

## 3. CEC 主候选：`linear_improvement`

### 3.1 初始改善尺度

给定初始种群适应度 \(F_0\)，定义

\[
b_0=\min(F_0),\qquad
m_0=\operatorname{median}(F_0),
\]

\[
\operatorname{IQR}_0=P_{75}(F_0)-P_{25}(F_0),
\]

\[
S_0=\max(m_0-b_0,\operatorname{IQR}_0,10^{-12}).
\]

每次 reset 都根据该 episode 的初始种群重新计算 \(S_0\)。

### 3.2 奖励

\[
r_t=\frac{b_t-b_{t+1}}{S_0}.
\]

该奖励不执行 `max(improvement, 0)`，也不裁剪：改善为正，不变为零，
退化为负。当前环境的 gbest 按定义单调不增，所以正常完整 episode 中
实际得到的 linear reward 应为非负；纯函数仍保留负向语义，以避免奖励
定义依赖这一实现假设。

`linear_improvement` 的计算路径只读取 \(b_t,b_{t+1},S_0\)，不读取
`optimum`。因此它具有：

- **平移不变性**：目标函数加任意常数时，分子和初始尺度不变；
- **正比例缩放不变性**：目标函数乘正数时，分子和初始尺度同比缩放，
  比值不变。

当 TD3 使用 `discount=1.0` 且尺度在 episode 内固定时：

\[
\sum_{t=0}^{T-1}r_t
=\frac{1}{S_0}
\sum_{t=0}^{T-1}(b_t-b_{t+1})
=\frac{b_0-b_T}{S_0}.
\]

因此累计 reward 只由初始和最终 gbest 决定。当前 TD3 默认
`discount=0.99` 并未在本轮修改；折扣小于 1 时仍会提高早期改善的权重。

## 4. Oracle 诊断：`oracle_log_gap_reduction`

定义

\[
\operatorname{gap}(b)=\max(b-f^*,0),
\qquad
g_t=\frac{\operatorname{gap}(b_t)}{G_0},
\]

其中 \(f^*\) 是理论最优值，reset 时由初始 gbest 构造固定正尺度

\[
G_0=\max(\max(b_0-f^*,0),10^{-12}).
\]

奖励为

\[
r_t=\log(g_t+\epsilon)-\log(g_{t+1}+\epsilon).
\]

它不裁剪，也不丢弃负奖励。在 `discount=1.0` 时：

\[
\sum_{t=0}^{T-1}r_t
=\log(g_0+\epsilon)-\log(g_T+\epsilon).
\]

中间项完全抵消。该模式显式读取理论最优值，只能用于 oracle 消融，
不能用来证明不依赖问题先验的主方法有效。

## 5. 状态、诊断和信用传播边界

当前 reward 不直接奖励 diversity、movement、stagnation 或较大的 C。
这些量只作为 Actor 状态和实验诊断，避免把手工搜索偏好混入优化目标。

n-step 解决的是奖励向前传播的跨度，不能自动修复奖励目标本身。比较
一步 TD 与 n-step 时，必须使用相同 reward mode 和相同 discount，才能
把差异归因于信用传播机制，而不是奖励或折扣变化。

三种模式仅允许改变 reward、`reward_progress` 以及 reward mode/scale
诊断字段；在相同问题、seed 和动作序列下，不应改变位置、fitness、
pbest/gbest、Q、observation、FE、终止信号、边界投影或随机数顺序。
