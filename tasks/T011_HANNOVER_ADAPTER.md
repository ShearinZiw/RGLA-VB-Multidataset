# T011：连接 Hannover 外部数据

## Dependency

用户已设置 `HANNOVER_DATA_ROOT`；T001 通过。

## Objective

读取用户目录的真实 filelist/HDF5 结构，完成 Hannover adapter、标签来源和六个跨机床任务。不得猜测不存在的 key。

## Allowed files

- `src/rgla_vb/data/adapters/hannover.py`
- `src/rgla_vb/data/splits.py`
- `tests/test_hannover_adapter.py`
- `tests/test_hannover_splits.py`（新建）
- `data/manifests/hannover_local_schema.json`（新建，不含用户绝对路径）
- `docs/02_DATASETS_AND_ADAPTERS.md`
- `outputs/data_audit/*`
- `outputs/task_reports/T011.json`（新建）

## Required work

1. 先只清点文件并 probe 一个 M1、M2、M3 文件，生成 schema 报告。
2. 从 filelist/metadata 读取 machine、tool、run、contact time、VB 和标签来源；若标签来源字段不存在，依据发布文档建立可追踪规则并记录。
3. 输出统一生命周期表，不把四刃平均 VB 改为分类。
4. 对 M2 T4-T6 aliasing、M3 T7/T8 rotation、T8 early missing 建立显式标志列。
5. 实现六个有向机床任务；目标标签单独保存在 evaluation view。
6. 实测标签和发布方插值标签可分别筛选，主评价只返回 measured。

## Required checks

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m rgla_vb.data.validate_datasets --dataset hannover --require-present
python -m pytest tests/test_hannover_adapter.py tests/test_hannover_splits.py -q
```

## Stop conditions

若 HDF5 key、filelist 或标签来源无法确定，停止并在 T011 report 列出实际字段；禁止通过硬编码空值继续。

## Done criteria

三台机床和九把工具计数可追溯；六任务无 identity 交叉；异常标志与 measured-only 评价测试通过。
