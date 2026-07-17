# RGLA-VB 方法规格

## 1. 基线主干

保持原论文复现的主要路径：源域小样本扩增、DDGAN、生成样本伪标签、PGRU 连续 VB 回归和冻结特征提取器后的微调。所有改进必须通过独立开关启用，关闭时应回到 B2：`DDGAN + pseudo-label + PGRU`。

## 2. 生成样本可靠性门控

对生成样本 `x_g`，三头判别器输出：

- 真实性头 `D_real(x_g)`；
- 域一致性头 `D_domain(x_g)`；
- 磨损一致性头 `D_wear(x_g)`。

将每个头转换为 `[0,1]` 置信度 `c_r, c_d, c_w`，联合权重为：

```text
q_g = clip(c_r ^ alpha_r * c_d ^ alpha_d * c_w ^ alpha_w, q_min, 1)
```

默认 `alpha_r = alpha_d = alpha_w = 1/3`，`q_min = 0.05`。伪标签回归损失改为：

```text
L_generated = sum(q_g * huber(y_hat_g, y_pseudo_g)) / sum(q_g)
```

不能用目标真实 `VB` 校准 `q_g`。置信度阈值和温度只能由源域验证工具确定。必须同时记录接受率、`q_g` 分布、各头置信度和伪标签一致性误差。

## 3. 软生命周期表示

每个样本有一个只由可观测进程变量计算的进度 `u in [0,1]`：

- PHM2010：实际 cut 序号相对当前序列可用进程；
- Hannover：累计接触时间优先；
- NASA：case 内实际 elapsed time 或 run 序号。

用 `K=3` 个径向基函数得到软阶段权重：

```text
s_k(u) = exp(-(u - mu_k)^2 / (2 * sigma^2))
p_k(u) = s_k(u) / sum_j s_j(u)
mu = [0.15, 0.50, 0.85], sigma = 0.20
```

`p_k` 只作为条件权重，不生成阶段类别标签。目标域进度不得使用目标寿命终点、最后 cut 或目标测试 `VB` 反推。在线设置中，归一化分母必须来自已观测进程上界或源域固定标尺。

## 4. 生命周期条件回归结构对齐

设编码特征为 `z`，阶段权重为 `p_k`。对源域和目标域分别计算加权特征 Gram 统计 `G_s^k`、`G_t^k`，只在两域有效样本数均达到阈值时计算：

```text
G_d^k = (Z_d^T diag(p_dk) Z_d) / (sum(p_dk) + eps) + eps * I
L_rgla = sum_k gamma_k * distance(G_s^k, G_t^k)
```

首版 `distance` 使用稳定的归一化 Frobenius 距离；正式 DARE-GRAM 对照再实现逆 Gram 的角度与尺度项。`gamma_k` 与两域该阶段的有效质量成正比，并归一化为和 1。

该模块的目的不是让域不可辨别，而是在相似生命周期位置对齐回归几何。若某阶段支持不足，跳过该阶段并写日志，禁止用零矩阵伪造有效损失。

## 5. 总损失

```text
L = L_source
  + lambda_g * L_generated
  + lambda_t * L_target_pseudo
  + lambda_rgla * L_rgla
  + lambda_smooth * L_smooth
```

默认只在单模块验证通过后逐项开启。首轮固定：`lambda_g=1.0`、`lambda_rgla in {0.01, 0.05, 0.1}`、`lambda_smooth` 沿用旧基线。超参数只能用源域分组验证选择。

## 6. 特征注意力位置

轻量特征注意力放在手工特征标准化/选择之后、PGRU 之前：

```text
h = tanh(Wx + b)
a = softmax(v^T h)
x_att = x * (1 + a)
```

`tanh` 压缩非线性特征响应，`softmax` 将各特征的相对重要性转成和为 1 的权重，相乘完成逐特征重标定；残差 `1+a` 避免初期把特征完全压没。注意力必须作为共同骨干或单独消融，不能把它与 RGLA 的收益混为一谈。

## 7. 不变量

- Wear Head 是生成样本筛选信号，不是最终回归结果；最终输出来自 PGRU 回归头。
- 不使用目标真实 `VB` 训练判别器、设阈值或选 checkpoint。
- 不对目标测试 `VB` 做插值后再计算指标。
- 不改变输出单位来制造更小的 MAE；论文同时报告原始单位与跨数据集归一化指标。
