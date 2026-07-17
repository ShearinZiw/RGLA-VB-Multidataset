# RGLA-VB/RUL 多数据集实验计划

## Material Passport

- Material ID: `RGLA-VB-RUL-EXPERIMENT-PLAN-002`
- Material Type: Code Experiment Plan
- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-17
- Verification Status: UNVERIFIED
- Version Label: v2.0-dual-regression-preregistered

## 1. 目标与假设

**H1 双时间尺度骨干**：多尺度 1D + 生命周期条件 TCN 相对参数量匹配 GRU，在 source-only 条件下同时改善或保持 VB 与 RUL-cut，并减少 seed 方差。

**H2 生命周期局部增强**：只在 source train 生命周期邻域内增强，相对无增强和 DDGAN 增强，减少高磨损/低 RUL 区误差且不增加身份泄漏。

**H3 回归结构对齐**：RGLA 相对无对齐、全局 DANN 和 DARE-GRAM，在相近生命周期位置减少负迁移。

**H4 不确定性感知 Teacher**：EMA Teacher 相对硬伪标签与无 Teacher 对照，降低高不确定目标样本对训练的影响，并稳定 VB/RUL 双头目标一致性。

**H5 完整方法**：生命周期局部增强 + 生命周期条件 TCN + RGLA + uncertainty-aware EMA Teacher 相对旧 B2 和新 C0，在 PHM2010 与可评价的 Hannover 任务上获得方向一致的 VB/RUL 改善。

**H6 外部边界**：NASA 默认无完整 RUL 标签；完整方法在其 8 个跨材料任务上只检验 VB 泛化和删失处理，不据此主张 RUL 改善。

这些是假设而非结果。TCN、Mean Teacher、DARE-GRAM 和 Mixup 本身不是项目原创；主要主张必须依赖模块消融、双任务共同收益和标签可见性审计。

## 2. 方法矩阵

| ID | 方法 | 目的 |
|---|---|---|
| B0-VB | Source-only PGRU，单 VB | 旧无迁移下限 |
| B2-VB | DDGAN + hard pseudo-label + PGRU，单 VB | 旧复现主基线 |
| B3-VB | DARE-GRAM + GRU/PGRU，单 VB | 旧回归域适应基线 |
| C0-GRU | 多尺度 1D + 生命周期条件 GRU + VB/RUL双头，source-only | 隔离 TCN 收益 |
| C0 | 多尺度 1D + 生命周期条件 TCN + VB/RUL双头，source-only | 新共同骨干 |
| A1-local | C0 + 生命周期局部增强 | 检验 H2 |
| A2-DARE | C0 + DARE-GRAM | 回归对齐基线 |
| A3-RGLA | C0 + RGLA，无 Teacher | 无伪标签核心对照 |
| A4-MT | C0 + uncertainty-aware EMA Teacher，无对齐 | 检验 H4 |
| M-noMT | C0 + 生命周期局部增强 + RGLA | 检验 Teacher 增量 |
| M | C0 + 生命周期局部增强 + RGLA + uncertainty-aware EMA Teacher | 完整方法 |
| M-DDGAN | 以 DDGAN 替换 M 的局部增强，其余相同 | 生成式增强敏感性 |

`hard pseudo-label` 不进入新完整方法。A3-RGLA 是强制的无伪标签对照；若 A4-MT 或 M 不能稳定超过对应无 Teacher 方法，正式主方法降级为 M-noMT，而不是按目标结果重新设计阈值。

## 3. 数据与 RUL 可评价门禁

| 数据集 | VB | RUL-cut | 正式用途 |
|---|---|---|---|
| PHM2010 | 可用 | EOL来源表通过后可用 | 双任务开发与完整消融 |
| Hannover | 发布方 VB 可用；measured-only 锚点仍待解决 | 逐工具 EOL来源审计通过后可用 | 跨机床主验证；未过门禁的指标不报告为主结果 |
| NASA | 146 个实测 VB，21 个缺失 | 默认右删失，mask=0 | 只做 VB 跨材料压力测试 |

不得因为 PHM2010 每把刀具有相同 cut 数或 Hannover 文件有最终 run，就自动声明 EOL。每把工具必须写 `eol_origin`、`rul_label_mask` 和删失类型。

## 4. 控制变量与选模

- 同一 dataset/task/seed 的 identity split、窗口索引、source scaler 和输入缓存完全相同。
- 所有增强在 split 和窗口创建之后执行，只读 source train。
- C0-GRU 与 C0 尽量匹配参数量、历史长度候选、batch size、优化器和 epoch 预算。
- A1/A2/A3/A4 每次只增加一个模块；M 的组合收益不能替代单模块结论。
- 所有方法共享预注册 seed，禁止为某个方法挑最好 seed。
- checkpoint 只用 source grouped validation。双任务数据集分别按 source VB MAE 与 RUL-cut MAE 排名，选择平均名次最小的 checkpoint；不把 `VB_norm` 用作训练目标或选模标签。
- 只有 VB 的 NASA 使用 source VB MAE 选择 checkpoint，不使用未监督 RUL 头。

## 5. 分阶段实验

### Phase -1：标签与前向门禁

1. 逐 lifecycle 生成 EOL/RUL 来源表。
2. 验证目标训练对象不暴露 `vb_eval`、`rul_eval`、`eol_eval`。
3. 用合成序列验证 TCN 因果性、双头 shape、RUL mask 和 Teacher stop-gradient。

任何一项失败都不得启动真实训练。

### Phase 0：单 seed smoke

- seed：`20260512`。
- PHM `C4->C6`：B2-VB、C0-GRU、C0、A1-local、A3-RGLA、A4-MT、M。
- Hannover `M1->M2`：C0 只做读取/前向；EOL门禁通过后再运行 M。
- NASA `DOC=0.75, feed=0.25, material1->material2`：B0-VB、C0、M，仅 VB 损失和指标。

退出条件：无 NaN、因果测试通过、RUL mask 生效、输出契约齐全、目标标签访问为零。

### Phase 1：三 seed 单模块筛选

- seeds：`20260512, 20260513, 20260514`。
- PHM 六个方向：C0-GRU、C0、A1-local、A2-DARE、A3-RGLA、A4-MT、M-noMT、M。
- Hannover 六个方向：仅在 VB/EOL 门禁允许的指标上运行 C0、A2-DARE、A3-RGLA、A4-MT、M-noMT、M。
- NASA 不用于调超参数或决定是否保留 RUL 模块。

进入正式方法的模块必须在至少一个主开发数据集改善对应主指标，并且另一主数据集任何可评价任务宏平均恶化不超过 2%。VB 与 RUL 分别判断；不能用一个任务的大幅改善抵消另一个任务明显恶化。

### Phase 2：五 seed 正式实验

- seeds：`20260512, 20260513, 20260514, 20260515, 20260516`。
- PHM：B0-VB、B2-VB、B3-VB、C0-GRU、C0、M-noMT、M、M-DDGAN。
- Hannover：B0-VB、B2-VB、B3-VB、C0、M-noMT、M；未通过 measured-only/EOL 门禁的指标明确降为次要分析。
- NASA 八个跨材料方向：B0-VB、B2-VB、C0、M，只报告 VB。
- 固定 Phase 1 预注册选择，不再按目标结果改超参数、增强邻域或 Teacher 权重。

### Phase 3：敏感性与失败分析

- 历史长度 `L={8,16,32}`，主值只用 source grouped validation 选一次。
- `lambda_rgla={0.01,0.05,0.1}`。
- Teacher `ema_decay={0.99,0.999}`；正式主值预注册为 `0.999`。
- MC Dropout 次数主值 `M=8`，只报告 `M={4,8,16}` 的成本/稳定性敏感性，不按目标指标重选。
- 生命周期局部增强 `alpha={0.2,0.4,0.8}`，正式主值预注册为 `0.4`。
- 主 EOL 与 150 µm 首穿敏感性并列表；不得只保留更有利的 EOL 定义。
- Hannover T8 和混叠通道异常完整保留，排除版本只能作为附加敏感性表。

## 6. 指标

每个目标 tool/case 先计算，再做任务和数据集宏平均。

### VB 主指标

- 原始单位 MAE、RMSE；
- R2、Pearson 作为描述性指标；
- high-wear MAE；
- 单调违例率。

### RUL 主指标

- `RUL-cut` MAE、RMSE；
- 归一化绝对误差只作事后描述，不参与训练或选模；
- 提前/滞后偏差；
- 可评价覆盖率、右删失率和按生命周期 30%/50%/70% 观察点的分层误差。

### 可靠性与成本

- Teacher VB/RUL 方差、有效权重分布和有效样本数；
- RGLA 有效阶段数与跳过原因；
- 增强配对距离、拒绝率和来源 identity；
- 运行时间、峰值显存和参数量。

VB 与 RUL 单独成表并同等进入主结论。跨数据集展示可用 source/target range 的事后 NMAE/NRMSE，但不得把目标评价范围用于训练、Teacher或选模。

## 7. 统计分析

1. 主比较为 M 对 C0；历史比较为 M 对 B2-VB，但 B2 只具备 VB 输出。
2. TCN收益比较 C0 对 C0-GRU；Teacher收益比较 M 对 M-noMT 及 A4-MT 对 C0。
3. 同 task、同 seed 配对，分别报告均值、标准差、配对差中位数、95% 分层 bootstrap CI 和胜率。
4. 分层 bootstrap 顺序为 dataset -> transfer task -> seed，10,000 次。
5. 不把多个 run 当作独立生命周期扩大样本量，不直接池化 Hannover 与 NASA runs。
6. 不以单个最好 seed、单个最好方向或单一混合分数作为主结果。

## 8. 成功标准

完整方法支持主要主张必须同时满足：

1. PHM2010：M 相对 C0 的 VB 与 RUL-cut 宏平均误差均不恶化超过 2%，至少一个任务改善 3% 以上，且两个任务合计至少 4/6 方向同时改善。
2. Hannover：在通过标签门禁的指标上，M 相对 C0 至少 4/6 方向改善；不能用发布方插值 VB 冒充 measured-only 主结论。
3. Teacher：M 相对 M-noMT 的五 seed 配对胜率超过 50%，且不会让任一主任务宏平均恶化超过 2%；否则删除 Teacher，采用 M-noMT。
4. NASA：M 的 VB 宏平均不比 C0 恶化超过 3%，且至少 5/8 方向改善；NASA 不设置 RUL 成功条件。
5. 全部正式运行标签可见性审计为零违规，旧基线与新 C0 回退测试均通过。

如果只在 PHM 成功，只主张跨刀具开发结果；如果 Hannover 标签门禁未解决，不声称完成跨机床双任务验证；如果 NASA 失败，完整报告小样本跨材料限制。

## 9. 输出契约

T054/T030 完成后的预期命令目前不是已实现命令：

```powershell
python -m rgla_vb.experiments.run --config configs/experiments/phase2_final.json
python -m rgla_vb.analysis.aggregate --runs outputs/runs --output outputs/analysis/phase2
```

每个运行目录必须包含：

```text
resolved_config.json
environment.json
label_visibility_audit.json
eol_rul_audit.json
train_log.jsonl
metrics.json
predictions.csv
checkpoint.pt
```

`predictions.csv` 最少包含：

```text
dataset, task, seed, sequence_id, cut,
vb_true, vb_pred, vb_label_origin,
rul_true, rul_pred, rul_label_mask, eol_origin, censoring_type,
teacher_vb_var, teacher_rul_var, teacher_weight
```

论文图表只从该 CSV、`metrics.json` 和审计文件重建。

## 10. 停止规则

- 任一训练访问目标评价 VB/RUL/EOL：立即停止该配置并作废性能结果。
- 三个 seed 中两个出现 NaN/发散：先修数值问题，不扩大矩阵。
- C0 因果测试或 identity split 测试失败：不得运行后续增强、RGLA或Teacher。
- M 在 PHM 的 VB 和 RUL 均比 C0 恶化超过 2%：停止正式扩展，回到单模块消融。
- NASA 单独失败不触发 RUL 方法重设计，只进入 VB 小样本失败分析。
