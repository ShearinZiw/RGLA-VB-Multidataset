# T030：实现统一多数据集实验 runner

## Dependency

T010、T011、T025 全部通过。

## Objective

实现读取 JSON 配置、枚举 dataset/task/method/seed、启动单次训练和写标准输出的 runner。此任务不改模型公式。

## Allowed files

- `src/rgla_vb/experiments/*`（新建）
- `src/rgla_vb/audit/*`（新建）
- `tests/test_runner_contract.py`（新建）
- `tests/test_label_visibility.py`（新建）
- `configs/experiments/*.json`
- `outputs/task_reports/T030.json`（新建）

## Required behavior

1. CLI：`python -m rgla_vb.experiments.run --config <json> [--dry-run]`。
2. dry-run 只打印任务矩阵、数据路径状态和输出目录，不加载目标评价标签。
3. 每次 run 写 `resolved_config.json`、`environment.json`、`label_visibility_audit.json`、`train_log.jsonl`、`metrics.json`、`predictions.csv`、`checkpoint.pt`。
4. checkpoint 只按源 grouped validation 选择。
5. 已完成且 config hash 相同可 resume；hash 不同必须新目录。
6. 捕获异常后写 `failure.json`，不能伪造空 metrics。

## Required checks

- 合成数据 dry-run 枚举数量正确。
- UDA 训练试图读取 target eval view 时测试失败。
- 两个 seed 输出目录不同且配置中 seed 正确。
- predictions 指标能独立重算。
- 中断后 resume 不重复已完成 epoch。

## Done criteria

`phase2_final.json --dry-run` 成功；所有契约测试通过；未启动正式五 seed 训练。
