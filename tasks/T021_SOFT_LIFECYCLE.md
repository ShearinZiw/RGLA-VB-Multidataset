# T021：实现软生命周期权重

## Dependency

T010 或 T011 至少一个 adapter 完成。

## Objective

实现数据集无关的进程变量到三阶段软权重映射，不改成阶段分类，不实现 Gram loss。

## Allowed files

- `src/rgla_vb/models/lifecycle.py`（新建）
- `src/rgla_vb/models/__init__.py`
- `tests/test_lifecycle.py`（新建）
- `outputs/task_reports/T021.json`（新建）

## Exact algorithm

`mu=[0.15,0.50,0.85]`、`sigma=0.20`，按 `docs/01_METHOD_RGLA_VB.md` 的 RBF + softmax 公式计算。输入 `u` 必须在 `[0,1]`，每行输出和为 1。

进度构造规则：使用当前可观测 elapsed/contact time 和源域固定标尺；目标域不能使用未来终点或最后一条测试记录归一化。

## Required checks

- `u=mu[k]` 时对应阶段权重最大。
- 每行权重非负且和为 1。
- CPU/GPU 结果容差内一致。
- 越界输入抛错。
- 测试构造一个目标序列前半段，确认加入未来记录不会改变已有记录的 `u`。

## Done criteria

单元测试通过；没有硬阶段标签；没有目标寿命终点依赖；默认开关关闭。
