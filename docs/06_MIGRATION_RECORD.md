# 项目迁移记录

## 1. 迁移范围

2026-07-17 从 `D:/PHM` 建立独立项目 `D:/RGLA-VB-Multidataset`。旧项目未被覆盖或删除。

迁入的冻结脚本：

- `phm_bridge.py`
- `phm_proposed_pipeline.py`
- `phm_paper_regression_pipeline.py`
- `run_paper_regression_multiseed.py`
- `export_seed_predictions.py`
- `evaluate_prediction_curves.py`

迁入的研究参考文档位于 `docs/references/`。它们作为历史材料保留，新实施以 `docs/00` 到 `docs/05` 和 `tasks/` 为准。

## 2. 数据策略

- PHM2010 原始数据不复制，默认引用 `D:/PHM/data`。
- Hannover 由用户从其他目录加入，只通过 `HANNOVER_DATA_ROOT` 读取。
- NASA Milling Parquet 位于 `data/raw/nasa_milling/data.parquet`，被 Git 忽略；来源与校验和单独提交。

## 3. 已验证内容

- NASA transport snapshot SHA-256 与清单一致；
- NASA schema 为 167 x 13，包含 6 路数组信号；
- PHM C1/C4/C6 目录存在；
- 数据注册与适配层单元测试通过；
- Hannover 未配置时可安全跳过。

## 4. 未验证内容

- Hannover 用户目录的实际 HDF5 key、filelist 结构和标签来源字段；
- NASA `VB` 的权威单位与 material 数值编码，需要 T010 对照官方说明二次核验；
- 新 RGLA-VB 方法代码与任何新实验结果；
- 旧训练脚本在新根目录下的完整正式训练。

这些未验证项不能在论文中写成已完成结果。
