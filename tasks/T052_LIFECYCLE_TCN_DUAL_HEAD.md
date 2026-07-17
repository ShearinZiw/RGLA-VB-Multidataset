# T052：生命周期条件 TCN 与 VB/RUL 双回归头

## Dependency

T051 通过。

## Objective

实现跨走刀的因果 TCN、连续生命周期条件和共享表示上的原始 VB/RUL-cut 双回归头。只实现独立模型与损失，不接 trainer。

## Allowed files

- `src/rgla_vb/models/lifecycle_tcn.py`（新建）
- `src/rgla_vb/models/dual_regression_head.py`（新建）
- `tests/test_lifecycle_tcn.py`（新建）
- `tests/test_dual_regression_head.py`（新建）
- `outputs/task_reports/T052.json`（新建）

## Required behavior

- 输入走刀序列 shape 为 `[batch, history, embedding_dim]`；只允许因果访问。
- 使用膨胀卷积和残差块；历史长度由外部配置传入，不在模块内搜索。
- 生命周期条件来自可观测 `progress_value/progress_type` 和固定软基函数；接口不得接收目标最终长度或 EOL。
- 共享表示分别进入 VB 线性连续头和带 softplus 的非负 RUL 头。
- `joint_regression_loss` 在原始 VB 和 RUL-cut 上计算 Huber；RUL 分母只统计 `rul_label_mask=1`。
- 全零 RUL mask 时 RUL loss 精确为 0，VB 梯度保持有效。
- 因果测试必须证明修改未来输入不会改变过去输出。

## Required checks

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest tests/test_lifecycle_tcn.py tests/test_dual_regression_head.py -q
```

## Done criteria

因果性、shape、非负 RUL、mask、双头梯度和错误输入测试全部通过；报告写入 `outputs/task_reports/T052.json`。
