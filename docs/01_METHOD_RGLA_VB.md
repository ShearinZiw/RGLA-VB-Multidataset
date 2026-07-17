# RGLA-VB/RUL 双回归方法规格

## 1. 任务与基线边界

主任务是在目标域标签不可见的 UDA 条件下，联合预测：

- 连续原始后刀面磨损量 `VB`；
- 连续剩余走刀次数 `RUL-cut`。

两者都是回归目标，不生成磨损阶段类别，不以 `VB_norm` 为训练目标。旧 B2：`DDGAN + hard pseudo-label + PGRU` 只作为复现基线保留；完整方法主干是：

```text
生命周期局部增强
  -> 多尺度 1D 特征提取
  -> 生命周期条件 TCN
  -> 共享连续退化表示
  -> 原始 VB / RUL-cut 双回归头
  -> RGLA / DARE-GRAM 回归结构对齐
  +  不确定性感知 EMA Teacher
```

各模块必须有独立开关。关闭增强、对齐和 Teacher 时，得到新的 source-only 双回归骨干 C0；旧 B2 不要求与 C0 结构相同。

## 2. RUL 标签与 mask

对第 `e` 次走刀，若该 lifecycle 存在经过审计的 EOL 位置 `e_EOL`：

```text
rul_cut(e) = max(0, e_EOL - e)
```

每个 lifecycle 同时记录：

```text
eol_origin in {provider_eol, preregistered_threshold, censored, unavailable}
rul_label_mask in {0, 1}
```

规则：

1. 主 EOL 使用数据集提供方定义或数据清单中预注册、可复核的终点。
2. `VB=150 um` 首次穿越只作为单位核验后的阈值敏感性分析，不能无记录地替换主 EOL。
3. 最后一个已观测 run 不能自动视为 EOL。右删失、提前停机和非 run-to-failure 数据的 `rul_label_mask=0`。
4. `rul_contact_s` 可作为 Hannover 等数据集的次要输出，但主跨数据集 RUL 指标使用 `RUL-cut`。
5. 目标域 EOL 与 RUL 仅在训练完成后的评价阶段解封，不得用于窗口构造、进程归一化、Teacher、选模或早停。

## 3. 生命周期局部增强

增强在 identity split 和窗口创建之后执行，输入仅来自 source train。候选样本对必须满足：

- 同一 `dataset_id`、`domain_id` 和 `condition_id`；
- 优先同一 `tool_id/sequence_id`；跨工具时必须工况相同且由配置显式允许；
- 可观测进程差、原始 VB 差和时间间隔均不超过 source-only 预注册邻域；
- 两个样本的 `rul_label_mask` 均为 1 时才插值 RUL，否则合成样本的 RUL mask 为 0。

连续插值为：

```text
x_mix = lambda * x_i + (1 - lambda) * x_j
vb_mix = lambda * vb_i + (1 - lambda) * vb_j
rul_mix = lambda * rul_i + (1 - lambda) * rul_j
```

`lambda ~ Beta(alpha, alpha)`，首轮固定 `alpha=0.4`。不得跨 source/target、train/validation/test 或生命周期远端样本增强。SMOGN 只作为高磨损/低 RUL 稀有区的 source-train 对照；DDGAN/WGAN-GP 仅作为生成式增强基线，不进入完整方法默认路径。

增强审计至少记录样本来源 identity、邻域距离、`lambda`、标签 mask 和拒绝原因。

## 4. 双时间尺度编码器

### 4.1 走刀内多尺度 1D 特征

对第 `e` 次走刀的多通道原始信号 `x_e`，并行 1D 分支提取短、中、长时间尺度模式：

```text
f_e^b = Pool(Conv1D_b(x_e)), b in {short, medium, long}
f_e = Projection(concat(f_e^short, f_e^medium, f_e^long))
```

首版分支使用不同 kernel/dilation，但保持相同输出宽度；具体形状由 T051 固定并用合成信号测试感受野。若某数据路径只有经过审计的缓存特征，则使用共享投影层进入 TCN，并把该运行标记为 `feature_input`，不能声称完成了原始信号多尺度学习。

### 4.2 跨走刀生命周期条件 TCN

TCN 只读取当前及历史走刀：

```text
z_e = TCN_causal([f_(e-L+1), ..., f_e], lifecycle_condition)
```

使用膨胀因果卷积和残差连接。历史长度 `L` 只能从预注册集合 `{8, 16, 32}` 中依据 source grouped validation 选择。

生命周期条件只使用部署时已观测的信息，例如累计走刀次数、累计接触时间和源域固定标尺。软权重继续采用：

```text
s_k(u) = exp(-(u - mu_k)^2 / (2 * sigma^2))
p_k(u) = s_k(u) / sum_j s_j(u)
mu = [0.15, 0.50, 0.85], sigma = 0.20
```

`p_k` 是连续条件权重，不是阶段标签。目标域 `u` 不得除以最终 run、总寿命或由目标 VB/EOL 反推。

## 5. 共享退化表示与双回归头

共享表示 `z_e` 输入两个独立线性/浅层回归头：

```text
vb_hat_e = Head_vb(z_e)
rul_hat_e = softplus(Head_rul(z_e))
```

`softplus` 只保证 RUL 非负；VB 输出保持连续原始单位，不做分类或截断。源监督损失为：

```text
L_vb = mean(huber(vb_hat, vb_raw))
L_rul = sum(rul_label_mask * huber(rul_hat, rul_cut)) / max(sum(rul_label_mask), 1)
L_source = lambda_vb * L_vb + lambda_rul * L_rul
```

首轮固定 `lambda_vb=lambda_rul=1.0`，两个指标分别报告。任何后续权重改变必须仅基于 source grouped validation 预注册，不能按目标结果调权。

可选时间一致性只使用相邻样本顺序：VB 总体不下降、RUL 总体不增加，并允许预注册噪声容差。它不能使用目标 EOL 或完整目标寿命长度。

## 6. DARE-GRAM 与 RGLA

DARE-GRAM 是无生命周期条件的回归专用基线：对齐 source/target 共享表示的逆 Gram 角度与尺度。RGLA 在此基础上按软生命周期权重计算结构统计：

```text
G_d^k = (Z_d^T diag(p_dk) Z_d) / (sum(p_dk) + eps) + eps * I
L_rgla = sum_k gamma_k * distance_regression(G_s^k, G_t^k)
```

`distance_regression` 的普通 Gram 版本与 DARE-GRAM 逆 Gram 版本必须作为不同开关，不得在一个实验 ID 下混用。只有 source 和 target 在阶段 `k` 的有效质量均达到阈值时才计算；否则跳过并写日志，禁止用零矩阵伪造损失。

RGLA 的作用是对齐相近生命周期位置的回归结构，而不是使两个域在全局上不可区分。计算过程只能读取目标特征、可观测进程元数据和 Teacher 预测，不能读取目标真实 VB/RUL。

## 7. 不确定性感知 EMA Teacher

Student 正常反向传播；Teacher 不反向传播，参数按指数移动平均更新：

```text
theta_teacher = ema_decay * theta_teacher
              + (1 - ema_decay) * theta_student
```

首轮固定 `ema_decay=0.999`。Teacher 接收弱扰动目标样本，Student 接收物理允许的较强扰动。Teacher 对每个目标样本做 `M=8` 次 MC Dropout 前向，分别得到 VB 与 RUL 的均值和方差。

```text
u_i = vb_var_i / tau_vb + rul_var_i / tau_rul
w_i = clip(exp(-u_i), w_min, 1)
```

`tau_vb`、`tau_rul` 只由 source validation 的预测方差标尺确定；首轮 `w_min=0`。目标一致性损失为：

```text
L_mt_vb = weighted_huber(student_vb, stopgrad(teacher_vb_mean), w)
L_mt_rul = weighted_huber(student_rul, stopgrad(teacher_rul_mean), w * task_rul_mask)
L_mt = L_mt_vb + L_mt_rul
```

`task_rul_mask=0` 的数据集不启用目标 RUL 一致性，避免让未受监督的 RUL 头自我强化。Teacher 产生的是软连续目标，不写入训练集成为永久硬标签。必须记录权重、方差、有效样本数和不同扰动的一致性。

## 8. 总损失与方法开关

```text
L = L_source
  + lambda_aug * L_augmented_source
  + lambda_da * L_dare_or_rgla
  + lambda_mt * L_mt
  + lambda_mono * L_monotonic
```

首轮只固定 `lambda_aug=1.0`；其余候选值在实验计划中预注册，并只用 source grouped validation 选择。方法开关必须支持：

- `legacy_b2`：DDGAN + hard pseudo-label + PGRU，单 VB 旧基线；
- `c0`：多尺度 1D + lifecycle TCN + VB/RUL 双头，source-only；
- `c0_gru`：与 C0 相同但 TCN 替换为参数量匹配 GRU；
- `local_aug`、`dare_gram`、`rgla`、`mean_teacher`：独立单模块开关；
- `full`：C0 + lifecycle-local augmentation + RGLA + uncertainty-aware EMA Teacher。

## 9. 不变量与审计

- scaler、特征选择器、增强邻域、Teacher 方差标尺和 checkpoint 只使用 source train/validation。
- identity split 必须早于窗口和增强。
- 目标评价 VB/RUL/EOL 不能进入训练对象，也不能用来选择 epoch、权重或历史长度。
- PHM2010/Hannover/NASA 的 RUL mask 和 EOL 来源必须逐 lifecycle 写入输出。
- NASA 非完整寿命样本不得静默生成 RUL；Hannover T8 左截断必须保留并标记。
- 每次实验分别写 VB/RUL 指标、预测、不确定性、有效对齐阶段数和标签可见性审计。
