# T022：实现生命周期条件 Gram 对齐

## Dependency

T021 通过。

## Objective

只实现 `docs/01` 第 4 节的加权 Gram 统计和稳定距离，再接入一个训练步骤。不要同时实现 DARE-GRAM 的逆矩阵版本。

## Allowed files

- `src/rgla_vb/losses/rgla.py`（新建）
- `src/rgla_vb/losses/__init__.py`（新建）
- `tests/test_rgla_loss.py`（新建）
- `outputs/task_reports/T022.json`（新建）

## Exact behavior

- `eps=1e-6`，每阶段有效权重和小于 2.0 时跳过。
- `G = Z.T @ diag(p) @ Z / (sum(p)+eps) + eps*I`。
- distance 为 `||Gs-Gt||F / (||Gs||F+||Gt||F+eps)`。
- `gamma_k` 与 `min(source_mass,target_mass)` 成正比，并在有效阶段归一化。
- 无有效阶段时返回可求导的零 loss 和 `valid_stages=0`。

## Required checks

- source 与 target 完全相同时 loss 近 0。
- 共同乘常数后归一化距离保持稳定。
- 阶段交换只要两域同步交换，loss 不变。
- 空阶段不产生 NaN。
- 梯度对 source/target feature 均有限。

## Done criteria

数学测试通过；API 返回每阶段 mass、distance、gamma 和 valid_stages。训练集成和 smoke 留给 T025。
