# RGLA-VB Multi-dataset

面向小样本跨域刀具磨损预测的独立研究项目。任务始终是连续原始磨损量 `VB` 回归，不改为磨损阶段分类，也不把 `VB_norm` 作为训练目标。

## 当前状态

- PHM2010：复用 `D:/PHM/data`，旧训练代码冻结在 `legacy/scripts/`。
- NASA Milling：已取得 167-run Parquet，SHA-256 已核验。
- Hannover：由用户自行准备，通过 `HANNOVER_DATA_ROOT` 以只读方式接入。
- 新方法与正式训练：尚未实施；先完成数据契约、实验预注册和低能力模型可执行任务书。

## 快速检查

```powershell
cd D:\RGLA-VB-Multidataset
$env:PYTHONPATH = "$PWD\src"
python -m rgla_vb.data.validate_datasets --dataset all
python -m pytest -q
```

Hannover 数据加入后：

```powershell
$env:HANNOVER_DATA_ROOT = 'D:\你的\Hannover\数据目录'
python -m rgla_vb.data.validate_datasets --dataset hannover --require-present
```

## 文档入口

1. `docs/00_PROJECT_OVERVIEW.md`：研究问题、范围和目录职责。
2. `docs/02_DATASETS_AND_ADAPTERS.md`：三个数据集的字段、划分与泄漏规则。
3. `docs/03_EXPERIMENT_AGENT_PLAN.md`：experiment-agent 生成的分阶段实验计划。
4. `docs/04_DEEPSEEK_V4_IMPLEMENTATION_GUIDE.md`：面向便宜、较弱模型的实施规约。
5. `tasks/`：一次只交给模型一个的原子任务书。

## 数据许可与边界

仓库不提交 PHM2010 和 Hannover 原始数据。NASA 文件也默认被 `.gitignore` 排除，只提交来源、校验和与字段清单。所有外部数据的使用必须遵守其原始许可证和引用要求。
