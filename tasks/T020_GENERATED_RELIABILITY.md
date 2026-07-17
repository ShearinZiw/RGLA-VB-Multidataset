# T020：实现生成样本可靠性门控

## Dependency

T001 通过；旧 B2 基线输出已固定。

## Objective

只实现三头置信度到生成样本回归权重的计算与日志，不实现生命周期对齐。

## Allowed files

- `src/rgla_vb/models/reliability.py`（新建）
- `src/rgla_vb/models/__init__.py`（新建或修改）
- `tests/test_reliability.py`（新建）
- `outputs/task_reports/T020.json`（新建）

## Exact algorithm

```text
q = clip(c_real^(1/3) * c_domain^(1/3) * c_wear^(1/3), 0.05, 1.0)
L = sum(q * huber(pred, pseudo)) / max(sum(q), eps)
```

输入置信度必须在 `[0,1]`；越界抛错。空 batch 抛错。所有 `q` 被 detach，首版不让回归损失反向操纵判别器置信度。

## Required checks

- 三个置信度全 1 时 `q=1`。
- 任一置信度 0 时 `q=0.05`。
- 交换三个头结果不变。
- 加权 Huber 与手算结果一致。
- 日志包含各头均值、q 的 q10/q50/q90、接受率和有效权重和。

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest tests/test_reliability.py -q
```

## Done criteria

只有纯可靠性门控模块；全部数学测试通过；没有读取目标 VB。训练集成留给 T025。
