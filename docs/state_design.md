# CCPSO 六维状态设计

`CCPSOEnv` 提供两种严格命名的状态模式：

- `legacy_v1`：现有状态，仍是默认值，用于复现实验和已有模型。
- `relative_log_v2`：候选相对尺度状态，用于缓解多样性、movement 和 stagnation 过早饱和。

切换状态模式只改变 observation 及对应的 info 诊断字段，不改变 CCPSO 更新、奖励、函数评价预算、终止条件或随机数调用。

## relative_log_v2

定义

\[
L(r;k)=\operatorname{clip}\left(
\frac{1}{2}+\frac{\log_{10}(\max(r,10^{-k}))}{2k},
0,1
\right),\qquad k=8,
\]

并单独规定 \(L(0;k)=0\)。因此比例1映射为0.5；比例小于1位于0.5以下，比例大于1位于0.5以上。比例 \(1,10^{-2},10^{-4},10^{-6},10^{-8},0\) 分别映射为 \(0.5,0.375,0.25,0.125,0,0\)。最后两个值相同是定义本身的结果，但不会像旧的线性归一化那样在 \(10^{-2}\) 至 \(10^{-6}\) 阶段提前全部变成0。

六维状态为：

1. FE进度沿用当前定义 \(FE_t/FE_{max}\)。
2. 最近完整 \(W\) 代改善：
   \[
   L\left(\frac{\max(b_{t-W}-b_t,0)}{S_0};8\right).
   \]
   \(S_0\) 是reset时由初始fitness分布计算的 `initial_improvement_scale`，不使用理论最优值。窗口不足时改善量为0。
3. position diversity：\(L(D_X(t)/D_X(0);8)\)。
4. Q diversity：\(L(D_Q(t)/D_Q(0);8)\)。
5. movement：\(L(M(t)/D_X(0);8)\)。
6. stagnation：
   \[
   \frac{\log(1+k_t)}{\log(1+N_{max})},
   \]
   其中 \(N_{max}\) 是reset初始评价完成后，本episode最多还能执行的完整种群更新次数。

## Reset尺度与不变性

`relative_log_v2` 在每次reset并准备好本代Q之后重新计算 \(D_X(0)\)、\(D_Q(0)\) 和 \(N_{max}\)。两个初始多样性尺度必须有限且为正；若实际多样性为0，则采用“搜索空间对角线乘 \(10^{-12}\)”作为安全下限。

这些定义带来以下性质：

- 目标函数增加常数不改变改善状态；
- 目标函数和 \(S_0\) 同时乘正数不改变改善状态；
- 坐标、边界和movement同时乘正数时，position diversity、Q diversity和movement三个状态不变；
- 初始非零position/Q diversity映射为0.5；后续收缩和扩张分别落在0.5两侧。

`optimum`、function id和CEC函数类别均不参与 `relative_log_v2` 的状态计算。本轮只提供环境级候选状态，不改变训练配置、CLI或已有正式实验。
