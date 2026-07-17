# RGLA-VB Multi-dataset

面向小样本跨域刀具磨损与寿命预测的独立研究项目。主任务是连续原始磨损量 `VB` 与剩余走刀次数 `RUL-cut` 的联合回归，不改为磨损阶段分类，也不把 `VB_norm` 作为训练目标。没有可验证 EOL 的生命周期只训练和评价 `VB`，不得伪造 RUL 标签。

当前预注册主方法为：

```text
生命周期局部增强
  -> 多尺度 1D 特征提取
  -> 生命周期条件 TCN
  -> 共享连续退化表示
  -> 原始 VB / RUL-cut 双回归头
  -> RGLA / DARE-GRAM 回归结构对齐
  +  不确定性感知 EMA Teacher
```

旧 `DDGAN + hard pseudo-label + PGRU` 路线保留为 B2 复现基线，不再作为完整方法主干。

## 当前状态

- PHM2010：复用 `D:/PHM/data`，旧训练代码冻结在 `legacy/scripts/`。
- NASA Milling：已取得 167-run Parquet，SHA-256 已核验。
- Hannover：由用户自行准备，通过 `HANNOVER_DATA_ROOT` 以只读方式接入。
- 新方法与正式训练：尚未实施；双回归方法、实验矩阵和原子实施任务已预注册。

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

1. `docs/00_PROJECT_OVERVIEW.md`：研究问题、双回归框架和范围。
2. `docs/01_METHOD_RGLA_VB.md`：主方法、损失、开关和泄漏边界。
3. `docs/02_DATASETS_AND_ADAPTERS.md`：三个数据集的字段、RUL可用性、划分与泄漏规则。
4. `docs/03_EXPERIMENT_AGENT_PLAN.md`：experiment-agent 生成的分阶段实验计划。
5. `docs/04_DEEPSEEK_V4_IMPLEMENTATION_GUIDE.md`：面向便宜、较弱模型的实施规约。
6. `tasks/`：一次只交给模型一个的原子任务书。

## 数据许可与边界

仓库不提交 PHM2010 和 Hannover 原始数据。NASA 文件也默认被 `.gitignore` 排除，只提交来源、校验和与字段清单。所有外部数据的使用必须遵守其原始许可证和引用要求。
