# RGLA-VB 多数据集实验计划

## Material Passport

- Material ID: `RGLA-VB-EXPERIMENT-PLAN-001`
- Material Type: Code Experiment Plan
- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-17
- Verification Status: UNVERIFIED
- Version Label: v1.0-multidataset-preregistered

## 1. 目标与假设

**H1 生成可靠性**：三头置信度门控相对未加权 DDGAN 伪样本，降低目标域 MAE，并降低 seed 方差。

**H2 生命周期条件化**：软生命周期条件对齐相对全局 DANN，在高磨损区和 endpoint 附近减少负迁移。

**H3 组合收益**：RGLA-VB 相对当前 B2 在 PHM2010 与 Hannover 上取得一致方向的宏平均改善。

**H4 外部方向性**：在 NASA 同工况跨材料任务上，RGLA-VB 相对 B2 的任务宏平均 NRMSE 不恶化，并在多数任务上改善。

**H5 注意力边界**：轻量特征注意力能改善特征利用，但不能单独解释 RGLA 的可靠性与结构对齐收益。

## 2. 方法矩阵

| ID | 方法 | 目的 |
|---|---|---|
| B0 | Source-only PGRU | 无迁移下限 |
| B1 | DANN-PGRU | 全局域对抗基线 |
| B2 | DDGAN + pseudo-label + PGRU | 当前论文复现基线 |
| B3 | DARE-GRAM + GRU/PGRU | 回归专用结构对齐基线 |
| A1 | B2 + generated reliability gate | 检验 H1 |
| A2 | B2 + soft lifecycle conditioning | 检验 H2 的条件化部分 |
| A3 | B2 + lifecycle-conditioned Gram alignment | 检验回归结构对齐 |
| M | A1 + A2 + A3，RGLA-VB | 完整方法 |
| M-attn | M + feature attention | 注意力附加收益 |

PHM2010 完成 B0/B1/B2/B3/A1/A2/A3/M/M-attn；Hannover 完成 B0/B2/B3/A1/A3/M；NASA 只完成 B0/B2/B3/M，避免在小数据集上扩大实验搜索空间。

## 3. 控制变量

- 同一 dataset/task/seed 的原始切分、特征缓存和源域 scaler 完全相同。
- B0/B1/B2/A1/A2/A3/M 使用同一 PGRU 宽度、batch size、优化器和总 epoch 预算。
- 生成样本数量固定，可靠性门控只改变样本权重，不改变生成预算。
- 任何超参数选择仅依据源域 grouped validation。
- 每个 seed 对所有方法共享；禁止为某个方法单独挑 seed。

## 4. 任务与 seed

### Phase 0：基础 smoke

- PHM `C4->C6`，B2 和 M，各 1 seed：`20260512`。
- NASA `DOC=0.75, feed=0.25, material1->material2`，B0 和 B2，各 1 seed。
- Hannover 接入后 `M1->M2`，B0 各 1 seed，只验证读取、划分和前向传播。

退出条件：无 NaN、所有输出契约齐全、标签泄漏测试通过、关闭新开关可复现 B2 容差内结果。

### Phase 1：三 seed 机制筛选

- seeds：`20260512, 20260513, 20260514`。
- PHM 全 6 个方向：B0/B2/A1/A2/A3/M。
- Hannover 全 6 个方向：B0/B2/A1/A3/M。
- 不在这一阶段调 NASA 超参数。

进入正式实验的模块必须在两个数据集中的至少一个达到：相对 B2 宏平均 NRMSE 降低 3%，且另一数据集不恶化超过 2%。

### Phase 2：五 seed 正式实验

- seeds：`20260512, 20260513, 20260514, 20260515, 20260516`。
- PHM：B0/B1/B2/B3/M/M-attn。
- Hannover：B0/B2/B3/M。
- NASA 8 个跨材料方向：B0/B2/B3/M。
- 固定 Phase 1 选择的超参数，不再按目标结果调整。

### Phase 3：敏感性与失败分析

- Hannover 插值标签权重 `{0, 0.25, 1}`；主值 0.25。
- `lambda_rgla={0.01,0.05,0.1}`；仅源验证选择一个主值。
- 可靠性 `q_min={0,0.05,0.1}`；仅在一个开发方向筛选。
- 含/不含 Hannover 已知异常工具的结果并列，完整工具结果为主。

## 5. 指标

每个目标 sequence 先计算，再做目标域宏平均：

- MAE、RMSE、R2、Pearson；
- endpoint absolute error；
- high-wear MAE，阈值由该数据集预注册物理范围或源域定义；
- monotonic violations；
- NMAE 与 NRMSE：用目标评价标签的 `max-min` 仅作事后跨任务尺度归一化，不参与训练或选模。

同时记录运行时间、峰值 GPU 显存、生成样本接受率、可靠性分布、有效 RGLA 阶段数和目标标签访问计数。

## 6. 统计分析

1. 主比较是 M 对 B2，同 task、同 seed 配对。
2. 报告均值、标准差、配对差值中位数、95% bootstrap CI 和胜率。
3. 跨数据集总效应使用分层 bootstrap：dataset -> transfer task -> seed，10,000 次。
4. 不把 Hannover 的 6418 runs 与 NASA 的 167 runs 直接池化。
5. 不以单个最好 seed、单个最好方向或最小 MAE 作为主结果。

## 7. 成功标准

完整方法同时满足以下条件才支持主要主张：

1. Hannover 六方向宏平均 NRMSE 相比 B2 至少降低 5%，且至少 4/6 方向改善。
2. PHM2010 六方向宏平均 NRMSE 相比 B2 至少降低 3%，且 high-wear MAE 不恶化。
3. NASA 八个跨材料任务的宏平均 NRMSE 不高于 B2，且至少 5/8 方向改善。
4. 全部 dataset/task/seed 中至少 60% 配对获胜，没有任一数据集宏平均恶化超过 3%。
5. 标签可见性审计为零违规，关闭新模块时基线复现通过。

若只满足 Hannover，则论文只主张跨机床改进；若 NASA 无效，明确报告极小样本跨材料限制，不更换任务或隐去结果。

## 8. 运行与输出契约

以下是 T030 完成后的预期命令，目前不是已实现命令：

```powershell
python -m rgla_vb.experiments.run --config configs/experiments/phase2_final.json
python -m rgla_vb.analysis.aggregate --runs outputs/runs --output outputs/analysis/phase2
```

每个运行目录必须包含：

```text
resolved_config.json
environment.json
label_visibility_audit.json
train_log.jsonl
metrics.json
predictions.csv
checkpoint.pt
```

`predictions.csv` 最少包含 `dataset, task, seed, sequence_id, cut, y_true, y_pred, label_origin`。论文图只从该 CSV 与 `metrics.json` 重建。

## 9. 停止规则

- 任一训练访问目标评价标签：立即停止整个配置，不保留其性能结果。
- 三个 seed 中出现两个 NaN/发散：先修数值问题，不继续扩大矩阵。
- M 比 B2 在 PHM 和 Hannover 均恶化超过 2%：停止正式扩展，回到 A1/A2/A3 定位。
- NASA 单独失败不触发方法重设计，只触发小样本失败分析。
