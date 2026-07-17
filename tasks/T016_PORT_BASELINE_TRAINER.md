# T016：移植 B2 基线训练阶段

## Dependency

T015 通过。

## Objective

在 `src/` 建立数据集无关的 B2 训练接口，按旧脚本顺序执行伪标签、DDGAN、域适应和微调。先只接 PHM C4->C6，关闭所有新方法。

## Allowed files

- `src/rgla_vb/training/baseline.py`（新建）
- `src/rgla_vb/training/contracts.py`（新建）
- `src/rgla_vb/training/__init__.py`（新建）
- `tests/test_training_contracts.py`（新建）
- `tests/test_baseline_smoke.py`（新建）
- `outputs/task_reports/T016.json`（新建）

禁止修改 `legacy/scripts/*`。

## Required work

1. 定义 `SourceBatch`、`TargetUnlabeledBatch`、`TargetEvaluationView`，训练函数签名不得接收 evaluation view。
2. 从 legacy 机械移植训练阶段和默认超参数，禁止顺手调参。
3. 保存 source grouped validation checkpoint，不读取目标 MAE。
4. 新 trainer 在固定 30-cut、同 seed 下与 legacy 比较 stage 数、样本数、输出 shape 和 loss 数量级。
5. 所有新模块开关默认 false。

## Required checks

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\legacy\scripts"
python -m pytest tests/test_training_contracts.py tests/test_baseline_smoke.py -q
```

## Done criteria

新 B2 trainer 完成 C4->C6 30-cut smoke；目标评价标签不能传入训练函数；legacy 零修改。
