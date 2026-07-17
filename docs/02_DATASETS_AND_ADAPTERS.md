# 数据集与适配协议

## 1. 统一记录

每个适配器输出一行一个 run/cut 的生命周期表，至少包含：

```text
dataset_id, domain_id, machine_id, tool_id, material_id,
condition_id, sequence_id, progress_value, progress_type,
vb_value, vb_unit, vb_um, label_mask, label_origin
```

`vb_value` 保存发布数据中的原值；只有单位由权威说明核验后才填写 `vb_um`。`label_origin` 只能是 `measured`、`provider_interpolated` 或 `missing`。

## 2. PHM2010

- 路径：默认只读复用 `D:/PHM/data`，可由 `PHM2010_DATA_ROOT` 覆盖。
- 域：C1、C4、C6；每个 cutter 是完整生命周期分组。
- 主任务：6 个有向 UDA；另设 Prefix-30 SSDA，训练目标是后 70% 连续 VB。
- 缩放与特征选择：仅用源域训练 cutter 拟合。
- 旧实现：`legacy/scripts/phm_paper_regression_pipeline.py`。

## 3. Hannover 多机床铣削

权威来源：Mendeley Data v3 `10.17632/zpxs87bjt8.3`，数据论文 `10.1016/j.dib.2023.109574`。

### 3.1 接入方式

用户自行准备数据，不复制进项目：

```powershell
$env:HANNOVER_DATA_ROOT = 'D:\path\to\hannover'
$env:PYTHONPATH = 'D:\RGLA-VB-Multidataset\src'
python -m rgla_vb.data.validate_datasets --dataset hannover --require-present
```

当前 `hannover.py` 只做 HDF5 清点和结构探测。连接真实目录后，先生成字段审计报告，再实现最终 reader，不能根据论文描述猜 HDF5 key。

### 3.2 固定域划分

| 机床域 | 工具 |
|---|---|
| M1 | T1, T2, T3 |
| M2 | T4, T5, T6 |
| M3 | T7, T8, T9 |

正式任务为 `M1->M2`、`M1->M3`、`M2->M1`、`M2->M3`、`M3->M1`、`M3->M2`。目标域所有传感器可用于 UDA，所有目标 `VB` 在训练与选择时隐藏。结果先按目标 tool 计算，再对三个目标 tool 宏平均。

### 3.3 标签与异常

- 主要 VB 是四刃平均值，约每 40 个 run 有一次实际测量，其余由发布方按接触时间插值。
- 主指标只在实测 VB 点计算；“实测 + 发布方插值”作为次要分析。
- 源域训练中实测点权重 1.0，发布方插值点预注册权重 0.25；另做权重 0 和 1 的敏感性分析。
- M2 T4-T6 的混叠通道、M3 T7/T8 坐标旋转和 T8 缺失早期运行必须进入审计报告。
- 主表保留全部工具；排除异常工具只能作为敏感性表，不能替代主表。

## 4. NASA Ames Milling

权威页面：`https://data.nasa.gov/dataset/milling-wear`。本项目使用校验和固定的 Parquet 传输镜像，来源和 SHA-256 见 `data/manifests/nasa_milling.json`。

### 4.1 已核验结构

- 16 个 case、167 个 run、146 个实测 VB、21 个缺失 VB；
- 条件字段：material、DOC、feed、time；
- 六路数组信号：主轴电流 AC/DC、工作台/主轴振动、工作台/主轴 AE；
- 每路每 run 长度 9000 到 15360 点；
- material=1 有 109 runs，material=2 有 58 runs，存在明显不平衡。

在权威字段说明被任务 T010 二次核验前，适配器保留 `vb_unit=native_unverified`，禁止猜测后直接转换为 um。

### 4.2 跨材料任务

每个 DOC/feed 组合在两种材料中各有两个完整 case。固定四个同工况组合，每个组合做两个方向，共 8 个任务：

| DOC | feed | material 1 cases | material 2 cases |
|---:|---:|---|---|
| 1.50 | 0.50 | 1, 9 | 5, 16 |
| 1.50 | 0.25 | 4, 10 | 6, 15 |
| 0.75 | 0.50 | 2, 12 | 8, 14 |
| 0.75 | 0.25 | 3, 11 | 7, 13 |

每个方向中，源材料的两个 case 是源域，目标材料的两个 case 是目标域。只在目标域实测 VB 行评价。缺失 VB 行可作为无标签信号参与 UDA，但不能插值后冒充测试真值。

### 4.3 小样本约束

NASA 某些 case 极短，例如 case 6 只有一个 run。因此：

- 不按 run 随机切分；
- 模型选择用 leave-one-source-case-out，最终 epoch 固定为源验证折最佳 epoch 的中位数；
- 正式运行 5 seeds，但报告平均值而不是最佳 seed；
- NASA 只作为方向性外部压力测试，单独失败不推翻跨机床主结论，但必须如实报告。

## 5. 数据泄漏防护

1. 先按完整 sequence/tool/case 划分，再提取窗口。
2. scaler、PCC、注意力初始化统计和伪标签阈值只拟合源训练数据。
3. 数据加载器分别返回 `train_visible_labels` 和 `evaluation_only_labels`。
4. 每次运行写 `label_visibility_audit.json`，记录每个阶段读取的标签行数和 identity。
5. 训练进程若访问目标评价标签文件，测试必须立即失败。

## 6. 2026-07-17 本地数据审计结论

### 6.1 Hannover

- 本地目录包含 6,418 个非空 HDF5 文件，与发布页和项目预期一致；九把刀具均存在，逐刀文件数见 `data/manifests/hannover_local_schema.json`。
- `filelist.csv` 有 8,607 行，但只有 6,418 个唯一且互相一致的元数据行。多出的 2,189 行是完全重复记录，不能静默丢弃；后续适配器必须保留重复计数和排除原因。
- 三台机床的真实标签键均为 `labels/machine`、`labels/tool`、`labels/run`、`labels/cumulated_tool_contact_time` 和 `labels/wear`。M1 提供进给轴扭矩，M2/M3 提供进给轴力，三台机床均提供外置测力计三轴力。
- HDF5 把累计接触时间的属性写成 `min`，但文件名数值、数据论文和发布说明均把 C 定义为当前 run 前的累计切削接触时间，单位为秒。本项目记录此冲突，并采用论文定义的 `s`；不得无记录地信任 HDF5 属性。
- 发布方说明 VB 约每 40 个 run 实测一次，其余 run 按累计接触时间线性插值，但 CSV/HDF5 没有逐 run 的来源字段，也没有公布精确的实测锚点 identity。因此当前无法构造可信的 measured-only 评价视图。禁止按“每 40 行”或曲线折点猜测实测行；T011 在该门禁处暂停，等待发布方的实测锚点清单或可复核推导规则。
- 已知异常必须显式保留：M2 的 `force_axis` 与 `torque_spindle` 受混叠影响；M3 T7/T8 的 XY 坐标系约旋转 1°–2°；T8 的早期运行缺失，首个可用 run 已有累计接触时间 177 s。主表不得删除这些工具。

### 6.2 PHM2010（只读完整性检查）

- C1、C4、C6 各有 315 个连续编号的传感器 CSV 和 315 行磨损表，共 945 个信号文件与 3 个磨损文件；未发现缺号或零字节文件。
- 单个信号 CSV 为无表头七通道数值序列；磨损表字段为 `cut,1,2,3`。本轮只确认结构完整，不启动另一个 adapter 任务，也不改变其原始 VB 回归定义。

## 7. 刀具 RUL 评估扩展方案

### Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-17
- Verification Status: UNVERIFIED（尚未执行；数据结构和 Hannover 来源说明已核验）
- Version Label: tool_rul_plan_v1

### 7.1 目标与任务边界

主训练任务仍是连续原始 `VB` 回归。RUL 只是对预测磨损轨迹的派生评估，不新增 RUL 训练标签、不改成分类、不使用 `VB_norm`。对接触时间为 `c` 的样本，给定预注册失效阈值 `VB_EOL`，定义：

```text
tau_hat(c) = inf {u >= c | VB_hat(u) >= VB_EOL}
RUL_hat(c) = max(0, tau_hat(c) - c)
```

Hannover 的 `VB_EOL` 预注册为发布方定义的约 150 µm；该阈值只用于阈值穿越评估，模型仍输出 µm 单位的连续 `VB`。其他数据集只有在来源单位和失效阈值分别记录后才能启用 RUL 评估，禁止直接照搬 150 µm。

### 7.2 实验设计

1. 先按完整 tool/case/machine identity 划分，再生成窗口；同一刀具生命周期不得跨训练和测试。
2. UDA 训练期间，目标域 VB、失效时刻和 RUL 全部隐藏；scaler、特征选择、轨迹拟合超参数、checkpoint 和 early stopping 只使用源训练/源验证数据。
3. 预测序列先在原始 VB 单位上做非下降轨迹约束，再计算首次阈值穿越；预测在观测范围内不穿越阈值时记为右删失，不强行填入最后一个 run。
4. 预注册观察点为各目标刀具可评价寿命的 30%、50%、70%；这三个位置只在最终评价阶段由目标标签生成，不能用于训练或模型选择。
5. 主指标为 measured-only 的 VB MAE/RMSE 和 RUL MAE（秒）；辅指标为 RUL 相对绝对误差、阈值穿越提前/滞后偏差和删失率。先逐目标 tool 计算，再做宏平均。
6. Hannover 在逐 run 标签来源得到解决前，不得报告 measured-only 主指标。现有发布方插值标签只能作为明确标注的次要敏感性分析，不能替代主结果。

### 7.3 预期产物与审计

每次实验必须写出 resolved config、seed、git status、VB 与 RUL 预测、逐 tool 指标、阈值与单位来源、删失标志，以及 `label_visibility_audit.json`。审计必须证明目标评价标签没有进入缩放、特征选择、伪标签、checkpoint 选择或 early stopping。
