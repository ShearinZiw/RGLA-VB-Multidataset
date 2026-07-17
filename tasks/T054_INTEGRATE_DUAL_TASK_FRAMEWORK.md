# T054：集成 RGLA-VB/RUL 完整方法

## Dependency

T016、T022、T023、T050、T051、T052、T053 全部通过。

## Objective

只做模块编排与方法开关，把生命周期局部增强、多尺度 1D、生命周期条件 TCN、双回归头、DARE/RGLA 和 EMA Teacher 接入 trainer。不得在本任务修改各模块公式。

## Allowed files

- `src/rgla_vb/training/baseline.py`
- `src/rgla_vb/training/methods.py`（新建）
- `tests/test_method_switches.py`（新建）
- `tests/test_method_smoke.py`（新建）
- `tests/test_label_visibility_dual_task.py`（新建）
- `outputs/task_reports/T054.json`（新建）

## Required behavior

- `legacy_b2`：只启用旧 DDGAN + hard pseudo-label + PGRU 单 VB 路径。
- `c0`：多尺度 1D + lifecycle TCN + VB/RUL双头，增强/DA/Teacher损失为零。
- `c0_gru`：与 C0 数据、双头和预算一致，仅以参数量匹配 GRU 替换 TCN。
- `a1_local/a2_dare/a3_rgla/a4_mt`：每次只增加对应单模块。
- `m_no_mt`：C0 + local augmentation + RGLA。
- `full`：C0 + local augmentation + RGLA + uncertainty-aware EMA Teacher。
- 没有 EOL 的 batch 设置 RUL mask，不允许填零标签冒充监督。
- 所有选模继续基于 source grouped validation；目标评价 VB/RUL/EOL 访问计数必须为零。
- 每个 loss、有效 VB/RUL 数、增强接受数、Teacher权重和有效 RGLA 阶段单独写日志。

## Required checks

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest tests/test_method_switches.py tests/test_method_smoke.py tests/test_label_visibility_dual_task.py -q
```

## Done criteria

全部方法开关与实验计划一致；C0 source-only 回退、legacy B2 独立回退和 RUL mask 均通过；PHM C4->C6 每个新方法完成 30-cut 单 seed smoke，无 NaN、无目标评价标签访问；报告写入 `outputs/task_reports/T054.json`。
