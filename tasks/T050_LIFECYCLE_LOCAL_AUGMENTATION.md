# T050：生命周期局部增强

## Dependency

T016 通过。

## Objective

实现只在 source train 生命周期局部邻域内工作的连续 VB/RUL 增强器。不得接入 trainer，不实现 GAN、SMOGN 或目标域增强。

## Allowed files

- `src/rgla_vb/data/augmentation.py`（新建）
- `tests/test_lifecycle_local_augmentation.py`（新建）
- `outputs/task_reports/T050.json`（新建）

## Required behavior

- 输入显式携带 `dataset_id/domain_id/condition_id/sequence_id/cut`、原始 VB、RUL、两个 label mask 和 split role。
- 只接受 `split_role=source_train`；任何 target/validation/test 输入立即报错。
- 默认只在同 sequence、同 condition 内配对；跨 sequence 必须显式开启且 condition 相同。
- 配对同时检查进程差、VB差和时间/走刀差，阈值来自调用配置，不能读取目标评价数据估计。
- 使用独立 seeded RNG 和 `Beta(alpha, alpha)`；默认 `alpha=0.4`。
- 只有两个父样本 RUL mask 都为 1 时才生成有效 `rul_mix`，否则输出 `rul_label_mask=0`。
- 返回增强样本及 audit：父 identity、距离、lambda、接受/拒绝和原因。
- 不静默跳过空邻域；返回零样本和结构化原因。

## Required checks

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest tests/test_lifecycle_local_augmentation.py -q
```

## Done criteria

合成数据测试覆盖确定性、跨 split 拒绝、远端生命周期拒绝、VB/RUL同步插值和 RUL mask；报告写入 `outputs/task_reports/T050.json`。
