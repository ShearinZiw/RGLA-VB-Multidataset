# 验收与完整性测试

## 1. 当前可执行检查

```powershell
cd D:\RGLA-VB-Multidataset
$env:PYTHONPATH = "$PWD\src"
python -m rgla_vb.data.validate_datasets --dataset all
python -m pytest -q
```

预期：PHM 和 NASA 为 `ok`；未设置环境变量时 Hannover 为 `not_configured`；测试全部通过。

## 2. Hannover 接入验收

设置 `HANNOVER_DATA_ROOT` 后必须产生：

- `outputs/data_audit/hannover_inventory.csv`；
- `outputs/data_audit/hannover_hdf5_schema.json`；
- machine/tool/run/VB/contact-time 字段映射；
- 实测、发布方插值、缺失 VB 数量；
- M2 混叠、M3 坐标和 T8 缺失报告。

6418 是发布版本的预期 run 数，不应写成盲目的硬失败：若数量不同，先检查压缩包层级和版本，再由人确认。

## 3. NASA 接入验收

- 文件 SHA-256 等于 `04819835e2747b9951a0d4415f5d0ce9d339ae6a0985ba3d6098c0b69116b1be`；
- 167 行、16 cases、146 个非空 VB、21 个缺失 VB；
- 六路信号数组均为有限一维数组，长度范围 9000-15360；
- 8 个跨材料任务中的 case 清单与文档一致；
- 单位未被权威材料确认前，`vb_um` 保持空值。

## 4. 训练完整性验收

每个方法必须通过：

1. `target_eval_label_reads == 0` during training；
2. scaler 和 selector 的 `fit_group_ids` 都属于源训练组；
3. train/test group identity 交集为空；
4. seed 重新运行时配置哈希相同；
5. 新模块关闭时，B2 的预测在预注册数值容差内一致；
6. 指标可由 `predictions.csv` 独立重算；
7. 最佳 checkpoint 由源验证分数而非目标指标选择。

## 5. 论文结果验收

- 表格同时提供每方向结果和宏平均，不能只给有利方向；
- 五 seed 均报告，不能挑最好 seed；
- Hannover 主指标只用实测点，插值标签另表；
- NASA 缺失标签不进入指标；
- 统计检验以 task/seed 配对为单位，不把每个 run 当独立样本；
- 所有数字能追溯到运行目录、配置哈希和预测 CSV。
