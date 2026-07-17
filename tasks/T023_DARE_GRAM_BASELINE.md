# T023：实现 DARE-GRAM 回归对齐基线

## Dependency

T015 和 T016 通过。

## Objective

依据论文与官方实现实现 B3，用作回归专用域适应对照。它不是 RGLA 消融，不与 T022 的简化 Gram 距离共用名称。

## Allowed files

- `src/rgla_vb/losses/dare_gram.py`（新建）
- `src/rgla_vb/losses/__init__.py`
- `tests/test_dare_gram.py`（新建）
- `docs/references/DARE_GRAM_IMPLEMENTATION_NOTES.md`（新建）
- `outputs/task_reports/T023.json`（新建）

## Required work

1. 先记录论文公式、官方仓库 commit、张量维度、截断和正则化细节。
2. 分开实现逆 Gram 的角度项和尺度项，API 返回两项日志。
3. 对病态矩阵使用论文/官方实现规定的正则化与低秩策略，不自行发明默认值。
4. 合成测试覆盖相同域、尺度变化、旋转变化、低秩和有限梯度。
5. 不读取标签，DARE-GRAM 只使用源/目标特征。

## Done criteria

公式与官方实现对照记录完整；数学测试通过；训练集成留给 T025。
