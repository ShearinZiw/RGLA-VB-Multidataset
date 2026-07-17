# T015：移植模型组件，不改变算法

## Dependency

T001 通过。

## Objective

从冻结 legacy 脚本中提取 PGRU、DDGAN 生成器/判别器、三头输出和特征注意力为可导入模块。只做机械移植，不改公式、默认维度或初始化。

## Allowed files

- `src/rgla_vb/models/pgru.py`（新建）
- `src/rgla_vb/models/ddgan.py`（新建）
- `src/rgla_vb/models/attention.py`（新建）
- `src/rgla_vb/models/__init__.py`
- `tests/test_model_parity.py`（新建）
- `outputs/task_reports/T015.json`（新建）

禁止修改 `legacy/scripts/*`。

## Required work

1. 标注每个类来自哪个 legacy 文件和原始类名。
2. 固定相同 seed 和输入，比较 legacy 类与新类的 state_dict key、shape 和前向输出。
3. PGRU 最终输出仍为一个连续值。
4. 三头判别器 Wear Head 只输出筛选/一致性信号，不替代 PGRU 回归头。
5. 特征注意力默认关闭；关闭时不引入参数或数值变化。

## Required checks

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\legacy\scripts"
python -m pytest tests/test_model_parity.py -q
```

## Done criteria

模型组件 parity 测试通过；legacy 零修改；没有训练循环或数据集逻辑混入模型文件。
