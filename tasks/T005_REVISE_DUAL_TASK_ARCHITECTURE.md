# T005：修订 VB/RUL 双回归研究方案与实施任务图

## Dependency

无代码依赖。本任务只修改研究规格、实验预注册和后续原子任务书，不实现训练代码。

## Objective

把项目主方法从“DDGAN + 硬伪标签 + PGRU”升级为：

```text
生命周期局部增强
  -> 多尺度 1D 特征提取
  -> 生命周期条件 TCN
  -> 共享连续退化表示
  -> 原始 VB / RUL-cut 双回归头
  -> RGLA / DARE-GRAM 回归结构对齐
  +  不确定性感知 EMA Teacher
```

旧 B2 路线保留为复现基线。不得在本任务中修改模型或训练代码。

## Allowed files

- `AGENTS.md`
- `README.md`
- `docs/00_PROJECT_OVERVIEW.md`
- `docs/01_METHOD_RGLA_VB.md`
- `docs/02_DATASETS_AND_ADAPTERS.md`
- `docs/03_EXPERIMENT_AGENT_PLAN.md`
- `docs/04_DEEPSEEK_V4_IMPLEMENTATION_GUIDE.md`
- `configs/experiments/protocol.json`
- `configs/experiments/phase2_final.json`
- `tasks/T005_REVISE_DUAL_TASK_ARCHITECTURE.md`
- `tasks/T050_LIFECYCLE_LOCAL_AUGMENTATION.md`
- `tasks/T051_MULTISCALE_1D_ENCODER.md`
- `tasks/T052_LIFECYCLE_TCN_DUAL_HEAD.md`
- `tasks/T053_UNCERTAINTY_MEAN_TEACHER.md`
- `tasks/T054_INTEGRATE_DUAL_TASK_FRAMEWORK.md`
- `outputs/task_reports/T005.json`

## Required behavior

- 连续原始 `VB` 与 `RUL-cut` 提升为同等地位的两个回归目标，不改成分类。
- `VB_norm` 仍不得作为训练目标。
- `RUL-cut` 只在可验证 EOL 的完整寿命数据上监督；右删失或无 EOL 数据不得伪造 RUL 标签。
- 目标域评价标签不得进入增强、Teacher、生命周期条件、对齐、缩放、选模或早停。
- 生命周期局部增强只能在 source train 内、完成 identity split 后执行。
- 旧 DDGAN、硬伪标签和 PGRU 作为 B2 基线保留，不再作为完整方法主干。
- 每个新模块都有独立开关和单模块消融，最终组合不得跳过归因实验。
- 后续实现继续拆成原子任务；本任务不修改 `src/`、`tests/`、`legacy/`。

## Required checks

```powershell
git diff --check
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q
```

并人工确认：

- 活跃方法文档同时包含 `VB`、`RUL-cut`、`EMA Teacher`、`TCN` 和 `RGLA / DARE-GRAM`；
- 实验计划包含 B2 旧基线、无伪标签对照、单模块消融和五 seed 正式实验；
- 没有把 NASA 缺失 EOL 样本改写成完整 RUL 标签；
- `git diff --name-only` 只出现 Allowed files。

## Done criteria

方法、数据、实验、实施指南和任务图相互一致；全部检查通过；完成报告写入 `outputs/task_reports/T005.json`。
