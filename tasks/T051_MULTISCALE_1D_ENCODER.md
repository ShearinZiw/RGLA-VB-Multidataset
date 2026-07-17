# T051：多尺度 1D 走刀内编码器

## Dependency

T015 通过。

## Objective

实现一次走刀内部多通道原始信号的多尺度 1D 编码器。只实现模型模块和合成测试，不接 trainer、不读取数据集。

## Allowed files

- `src/rgla_vb/models/__init__.py`（新建）
- `src/rgla_vb/models/multiscale_1d.py`（新建）
- `tests/test_multiscale_1d.py`（新建）
- `outputs/task_reports/T051.json`（新建）

## Required behavior

- 输入 shape 固定为 `[batch, channels, samples]`，输出 `[batch, embedding_dim]`。
- 三个并行分支使用不同 kernel/dilation，输出宽度相同；分支后做自适应池化再拼接投影。
- 不假定固定采样长度，支持 PHM/Hannover/NASA 不同 run 长度。
- 配置中记录 kernel、dilation、branch width、embedding dim 和 dropout；不自动搜索。
- 非有限输入、错误 channel 数和过短序列显式报错。
- 单元测试验证三种输入长度、梯度可回传、参数量记录和相同 seed 输出一致。

## Required checks

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest tests/test_multiscale_1d.py -q
```

## Done criteria

模块对可变长度合成信号前向/反向均通过，无数据集和训练依赖；报告写入 `outputs/task_reports/T051.json`。
