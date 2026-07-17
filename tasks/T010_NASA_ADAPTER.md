# T010：完成 NASA Milling 适配器

## Dependency

T001 通过。

## Objective

基于权威 NASA 说明核验 `material` 编码、`VB` 单位、time 含义与六路信号含义，完成可用于连续 VB 回归的 NASA adapter 和 8 个跨材料任务清单。

## Allowed files

- `src/rgla_vb/data/adapters/nasa_milling.py`
- `src/rgla_vb/data/splits.py`（新建）
- `tests/test_nasa_adapter.py`
- `tests/test_nasa_splits.py`（新建）
- `data/manifests/nasa_milling.json`
- `docs/02_DATASETS_AND_ADAPTERS.md`
- `outputs/task_reports/T010.json`（新建）

## Required work

1. 用 NASA 官方页面、官方 readme 或原始 archive 核验编码和单位；在 manifest 写来源字段，不得只引用第三方博客。
2. 若确认单位为 mm，将 `vb_um=vb_value*1000`；若不能确认，继续保持 `native_unverified` 并让需要 um 的 API 显式报错。
3. 保留 21 个缺失 VB 行，设置 `label_mask=false`；禁止插值为评价真值。
4. 实现固定 8 个任务，case 配对必须与 `docs/02` 表一致。
5. split 对象必须分开返回 source train labels、target unlabeled rows 和 target evaluation labels。
6. 对每个任务断言 source/target case 无交集、DOC/feed 相同、material 不同。

## Required checks

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest tests/test_nasa_adapter.py tests/test_nasa_splits.py -q
python -m rgla_vb.data.validate_datasets --dataset nasa_milling --require-present
```

## Done criteria

8 个任务全部通过 identity 和泄漏测试；字段单位有权威出处或明确保持未验证；任务报告列出 167/146/21 计数。
