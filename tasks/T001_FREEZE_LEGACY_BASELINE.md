# T001：冻结并确认 PHM 基线

## Dependency

无。

## Objective

证明迁入 `legacy/scripts/` 的 B2 路径可在新项目中调用，并记录一个 C4->C6 30-cut smoke 的命令、配置和输出契约。不得改变模型行为。

## Allowed files

- `tests/test_legacy_cli.py`（新建）
- `docs/07_LEGACY_BASELINE_COMMANDS.md`（新建）
- `outputs/task_reports/T001.json`（新建）

禁止修改 `legacy/scripts/*`。

## Required work

1. 测试 `phm_paper_regression_pipeline.py --help` 返回 0，且包含 `--wear-target {vb,vb_norm}`、`--feature-attention`、`--target-prefix-strategy`。
2. 文档给出只使用 `--wear-target vb` 的 C4->C6 smoke 命令。
3. 文档说明 `legacy/` 的输出与未来新 runner 输出不是同一 schema。
4. 任务报告记录 Python、PyTorch、CUDA、命令和退出码，不写虚构指标。

## Required checks

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest tests/test_legacy_cli.py -q
python legacy/scripts/phm_paper_regression_pipeline.py --help
```

## Done criteria

测试通过；未修改任何 legacy 文件；报告存在且只包含实际运行信息。
