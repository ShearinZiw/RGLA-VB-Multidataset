# 目标域前 30% 独立 Fine-tune、后 70% 评估

## 协议变化

`--target-prefix-strategy finetune` 使用以下严格顺序：

1. 使用源域 100% VB 标签训练伪标签、DDGAN、source/DANN 和生成样本微调阶段。
2. 上述阶段可使用目标域传感器特征和伪标签，但目标域真实 VB 标签完全不参与。
3. 最后只使用目标域前 30% 真实 VB 做低学习率监督 fine-tune，并使用源域 replay 抑制灾难性遗忘。
4. 指标只计算目标域后 70%。315 个 cut 对应前 95 个训练点、后 220 个测试点。

结果同时包含：

- `proposed_before_target_prefix_finetune`：目标前缀 fine-tune 前。
- `proposed_ddgan_pgru_finetune`：目标前缀 fine-tune 后，也是导出命令使用的最终结果。

因此可直接在 `MULTI_SEED_RESULTS.md` 中判断独立 fine-tune 是否真正改善 MAE、RMSE 和 R2。该方法可能改善域偏移和整体标定，但不能保证后 70% 一定变好；模型选择全过程不读取后 70% 标签。

## C4 100% 预训练 + C6 前 30% Fine-tune，预测 C6 后 70%

```powershell
python scripts\run_paper_regression_multiseed.py `
  --data-root D:\PHM\data `
  --output-root D:\PHM\refine-logs\paper-regression-c4-full-c6-prefix30-supervised-finetune-formal `
  --feature-cache-dir D:\PHM\refine-logs\paper-regression-shared-features `
  --data-sanity-path D:\PHM\refine-logs\paper-regression-c1-c4-full-multiseed-formal\seed_20260510\data_sanity.json `
  --source c4 --target c6 --wear-target vb `
  --target-prefix-fraction 0.30 --target-prefix-strategy finetune `
  --lambda-target-prefix 1.0 `
  --target-prefix-finetune-epochs 100 `
  --lr-target-prefix-finetune 0.0001 `
  --target-prefix-finetune-scope recurrent_head `
  --seeds 20260510 20260511 20260512 `
  --full-lifecycle --augmentation-windows 30 `
  --pseudo-epochs 200 --gan-epochs 400 --adapt-epochs 500 --finetune-epochs 100 `
  --lifecycle-batch-size 5 --pseudo-batch-size 5 `
  --noise-dim 100 --gan-hidden-dim 64 --pgru-hidden1 120 --pgru-hidden2 240 `
  --lr-pseudo 0.001 --lr-g 0.002 --lr-d 0.00002 --lr-reg 0.005 --lr-finetune 0.0005 `
  --lambda-js 1.0 --lambda-domain 0.1 --lambda-s8 0.08 --lambda-coeff-anchor 0.01 `
  --lambda-pseudo-consistency 1.0 --lambda-source-replay 0.5 --lambda-target-pseudo 1.0 `
  --finetune-base source --generated-multiplier 3 `
  --pseudo-filter-quantile 0.70 --pseudo-filter-max-mae 5.0 --pseudo-filter-min-sets 30 `
  --feature-attention --deterministic --resume --log-every 25
```

### 导出 C6 后 70% 最终预测

将 seed 改为汇总结果中希望查看的 seed：

```powershell
python scripts\export_seed_predictions.py `
  --run-root D:\PHM\refine-logs\paper-regression-c4-full-c6-prefix30-supervised-finetune-formal `
  --seed 20260512 --source c4 --target c6 `
  --system proposed_ddgan_pgru_finetune `
  --output D:\PHM\refine-logs\paper-regression-c4-full-c6-prefix30-supervised-finetune-formal\seed_20260512\exports\c6_suffix70_finetuned_predictions.csv
```

### 计算指标并绘制 C6 曲线

Absolute Error 子图自动使用 bar，指标仍基于未平滑的 `y_pred`。

```powershell
python scripts\evaluate_prediction_curves.py `
  --input D:\PHM\refine-logs\paper-regression-c4-full-c6-prefix30-supervised-finetune-formal\seed_20260512\exports\c6_suffix70_finetuned_predictions.csv `
  --output-dir D:\PHM\refine-logs\paper-regression-c4-full-c6-prefix30-supervised-finetune-formal\seed_20260512\exports\c6_suffix70_finetuned_evaluation `
  --wear-target vb --smooth-window 9 `
  --title "C4 100% pretrain + C6 prefix 30% fine-tune -> C6 suffix 70%"
```

## C6 100% 预训练 + C4 前 30% Fine-tune，预测 C4 后 70%

```powershell
python scripts\run_paper_regression_multiseed.py `
  --data-root D:\PHM\data `
  --output-root D:\PHM\refine-logs\paper-regression-c6-full-c4-prefix30-supervised-finetune-formal `
  --feature-cache-dir D:\PHM\refine-logs\paper-regression-shared-features `
  --data-sanity-path D:\PHM\refine-logs\paper-regression-c1-c4-full-multiseed-formal\seed_20260510\data_sanity.json `
  --source c6 --target c4 --wear-target vb `
  --target-prefix-fraction 0.30 --target-prefix-strategy finetune `
  --lambda-target-prefix 1.0 `
  --target-prefix-finetune-epochs 100 `
  --lr-target-prefix-finetune 0.0001 `
  --target-prefix-finetune-scope recurrent_head `
  --seeds 20260510 20260511 20260512 `
  --full-lifecycle --augmentation-windows 30 `
  --pseudo-epochs 200 --gan-epochs 400 --adapt-epochs 500 --finetune-epochs 100 `
  --lifecycle-batch-size 5 --pseudo-batch-size 5 `
  --noise-dim 100 --gan-hidden-dim 64 --pgru-hidden1 120 --pgru-hidden2 240 `
  --lr-pseudo 0.001 --lr-g 0.002 --lr-d 0.00002 --lr-reg 0.005 --lr-finetune 0.0005 `
  --lambda-js 1.0 --lambda-domain 0.1 --lambda-s8 0.08 --lambda-coeff-anchor 0.01 `
  --lambda-pseudo-consistency 1.0 --lambda-source-replay 0.5 --lambda-target-pseudo 1.0 `
  --finetune-base source --generated-multiplier 3 `
  --pseudo-filter-quantile 0.70 --pseudo-filter-max-mae 5.0 --pseudo-filter-min-sets 30 `
  --feature-attention --deterministic --resume --log-every 25
```

### 导出 C4 后 70% 最终预测

```powershell
python scripts\export_seed_predictions.py `
  --run-root D:\PHM\refine-logs\paper-regression-c6-full-c4-prefix30-supervised-finetune-formal `
  --seed 20260512 --source c6 --target c4 `
  --system proposed_ddgan_pgru_finetune `
  --output D:\PHM\refine-logs\paper-regression-c6-full-c4-prefix30-supervised-finetune-formal\seed_20260512\exports\c4_suffix70_finetuned_predictions.csv
```

### 计算指标并绘制 C4 曲线

```powershell
python scripts\evaluate_prediction_curves.py `
  --input D:\PHM\refine-logs\paper-regression-c6-full-c4-prefix30-supervised-finetune-formal\seed_20260512\exports\c4_suffix70_finetuned_predictions.csv `
  --output-dir D:\PHM\refine-logs\paper-regression-c6-full-c4-prefix30-supervised-finetune-formal\seed_20260512\exports\c4_suffix70_finetuned_evaluation `
  --wear-target vb --smooth-window 9 `
  --title "C6 100% pretrain + C4 prefix 30% fine-tune -> C4 suffix 70%"
```

## 检查点

- 日志中的 `pseudo_label`、`source_only`、`domain_adaptation` 和 `generated_target_finetune` 应显示 `target_prefix: 0.0`。
- 之后应出现独立的 `target_prefix_finetune` 阶段。
- `paper_regression_results.json` 中 `target_prefix_ground_truth_training_stage` 应为 `target_prefix_finetune`。
- `target_suffix_ground_truth_used_for_training` 必须为 `false`。
- 导出 CSV 只有 `cut,y_true,y_pred`，并且仅包含 cut 96-315。
- 不设置新参数时默认仍是原来的 `joint` 策略；不设置目标前缀时旧命令仍按完整目标域评估运行。
