# T025：把 A1/A2/A3/B3/M 接入 trainer

## Dependency

T016、T020、T021、T022、T023 全部通过。

## Objective

只做模块编排与开关，把纯模块接入 B2 trainer，形成 B3、A1、A2、A3 和 M。不得在此任务修改数学公式。

## Allowed files

- `src/rgla_vb/training/baseline.py`
- `src/rgla_vb/training/methods.py`（新建）
- `tests/test_method_switches.py`（新建）
- `tests/test_method_smoke.py`（新建）
- `outputs/task_reports/T025.json`（新建）

## Required behavior

- `method=B2`：所有新损失为零，输出与 T016 固定测试一致。
- `method=B3`：只启用 DARE-GRAM，不启用 DDGAN 可靠性或生命周期条件化。
- `method=A1/A2/A3`：每次只启用对应单模块。
- `method=M`：启用 generated reliability、soft lifecycle 和 RGLA alignment。
- 每个 loss 及其有效样本/阶段数单独写日志。
- 所有选择仍基于 source grouped validation。

## Required checks

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest tests/test_method_switches.py tests/test_method_smoke.py -q
```

## Done criteria

每个方法开关组合与计划一致；B2 parity 仍通过；C4->C6 每种方法完成 30-cut 单 seed smoke，无 NaN、无目标标签访问。
