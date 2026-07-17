# T053：不确定性感知 EMA Teacher

## Dependency

T052 通过。

## Objective

实现双回归 Student/Teacher 的 EMA 更新、MC Dropout 不确定性和加权一致性损失。不得接 trainer，不生成永久硬伪标签。

## Allowed files

- `src/rgla_vb/training/__init__.py`（新建）
- `src/rgla_vb/training/mean_teacher.py`（新建）
- `tests/test_mean_teacher.py`（新建）
- `outputs/task_reports/T053.json`（新建）

## Required behavior

- Teacher 参数不参与优化器，按 `theta_t=decay*theta_t+(1-decay)*theta_s` 更新；默认 decay=0.999。
- MC Dropout 默认 8 次，分别返回 VB/RUL 均值和方差；不得用目标标签校准方差。
- `tau_vb/tau_rul` 必须由调用方传入 source-only 标尺。
- 权重为方差的单调递减函数并 clip 到 `[w_min,1]`；首轮 `w_min=0`。
- Student 对 Teacher 均值计算 stop-gradient Huber 一致性。
- `task_rul_mask=0` 时目标 RUL 一致性精确为 0；VB一致性继续有效。
- 返回逐样本方差、权重、有效样本数和非有限预测审计。

## Required checks

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest tests/test_mean_teacher.py -q
```

## Done criteria

EMA 数值、Teacher 无梯度、方差权重单调性、RUL task mask、确定性 seed 和异常输入测试通过；报告写入 `outputs/task_reports/T053.json`。
