# T040：聚合指标与论文图

## Dependency

T030 通过且至少有一组完整 smoke 输出。

## Objective

只依赖标准 `predictions.csv` 和 `metrics.json` 重算指标、做配对统计并绘图。不得读取 checkpoint 或训练数据。

## Allowed files

- `src/rgla_vb/analysis/*`（新建）
- `tests/test_analysis.py`（新建）
- `outputs/task_reports/T040.json`（新建）

## Required behavior

- 绘制 y_true、原始 y_pred、平滑 y_pred；平滑只用于显示，指标使用原始 y_pred。
- absolute error 使用 bar，不画成连续折线。
- 按 sequence -> task -> dataset 宏平均。
- 同 task/seed 配对 M 与 B2，输出全部 seed，禁止 best-seed 过滤。
- 分层 bootstrap 10,000 次，固定 bootstrap seed。

## Required checks

- 从手工 CSV 重算 MAE/RMSE 与解析值一致。
- 改变平滑窗口不改变指标。
- 删除任一注册 seed 时分析脚本失败并指出缺失 seed。
- Hannover interpolated 标签不会混入 measured-only 主指标。

## Done criteria

smoke 的表格和图可从 CSV 重建；结果文件记录输入哈希；没有读取训练标签或 checkpoint。
