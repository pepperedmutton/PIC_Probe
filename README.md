# PIC_Probe / PICSIMU

PIC_Probe 是圆柱朗缪尔探针的静电 PIC-MCC 仿真软件。

## 当前状态

本仓库当前是 **Ar/Ar+ 1D3V 研究预览**。它不是生产级软件，也没有完成生产验收。

`Config.PHYSICS_RELEASE_READY` 当前为 `False`。因此，`PICSimulation` 会拒绝生产模式。

旧文档中的以下结论已经撤销：

- 氢等离子体是当前验证工况。
- Kakati 2017 的直径归一化比较证明了模型正确。
- 旧 Kakati 归一化结果可以称为“验证通过”。
- 当前程序可以生成生产级数据。

这些旧结论不符合当前物种、几何和绝对电流口径。它们不能作为验证或发布证据。

当前外部比较对象是 Cenian 等人在 2005 年发表的 Ar 探针实验。首轮真实数据预览比较已经完成，但结果不匹配实验，生产验证仍未完成。

## 1. 适用范围

当前恢复和验证范围如下：

- 中性气体：Ar
- 离子：单电荷 Ar+
- 空间：圆柱径向一维
- 速度：`v_r`、`v_theta`、`v_z` 三个分量
- 场：无磁场的静电 Poisson 场
- 探针：位于内边界的完全吸收圆柱
- 外边界：有限半径的等离子体储库

“1D3V”表示只有径向空间网格。每个宏粒子仍保留三个速度分量。

当前代码不是通用多物种等离子体化学平台。配置中的物种名称也不表示该物种已得到验证。

## 2. 已实现模型

### 2.1 场和几何

程序在 `R_MIN <= r <= R_MAX` 上求解圆柱 Poisson 方程：

```text
(1/r) d/dr (r dphi/dr) = -rho/epsilon_0

phi(R_MIN) = V_bias
phi(R_MAX) = V_wall
```

场模块包含以下功能：

- 圆柱控制体积
- CIC 电荷沉积
- 控制体积归一化
- 三对角 Poisson 求解
- 网格电场计算
- 线性场收集
- 可选的守恒电荷密度平滑

验证运行关闭密度平滑。

### 2.2 粒子推进

粒子状态为 `(r, v_r, v_theta, v_z)`。推进器使用径向半步加速和完整漂移。

径向加速度包含电场项和离心项：

```text
a_r = (q/m) E_r + v_theta^2/r
```

圆柱漂移保持轴向角动量 `r*v_theta`。探针吸收进入内边界的粒子。

外边界可以吸收或反射离开域的粒子。主仿真使用 Maxwell 通量分布补充电子和离子。

离子注入可以加入 Bohm 漂移。Cenian 2005 工况明确设置 `ION_INJECTION_BOHM=False`。

### 2.3 中性碰撞

电子-Ar MCC 包含以下过程：

- 弹性散射
- 激发
- 电离
- 多激发和多电离通道
- 电离后的一个二次电子和一个 Ar+ 宏粒子

碰撞算子按粒子停留时间抽样。它支持一个时间步内的多个碰撞事件。

新电离产物继续处理父粒子剩余的碰撞时间。事件缓冲区、事件数和粒子账本都有上限检查。

Ar+-Ar MCC 包含以下过程：

- 共振电荷交换
- 等质量各向同性弹性散射
- 基于上界频率的零碰撞抽样

Phelps `BACKSCATTER` 只有在显式确认对称 Ar+/Ar 映射后才能作为电荷交换。Phelps `ISOTROPIC` 用于等质量弹性散射。

### 2.4 库仑散射

代码保留可选的预览库仑散射算子。该算子不是发布模型。

生产配置会拒绝启用该算子。运行清单也会记录此限制。

### 2.5 随机数、守恒和追踪

中性碰撞使用按根种子、步骤、流和粒子索引派生的计数器随机流。

仿真还记录以下信息：

- 初始、注入、吸收和电离生成的粒子数量
- 电子和离子粒子账本残差
- 各碰撞过程和通道的事件数量
- 截面能量越界
- 稳定性警告和运行时警告
- 配置哈希、状态哈希和输入文件哈希
- NumPy 随机数生成器状态

账本不平衡、事件缓冲区溢出和碰撞事件上限会停止运行。

## 3. 截面数据

### 3.1 宽松模式

研究预览可以使用常数测试截面或本地的不完整旧文件。回归测试使用程序生成的合成截面，不分发第三方原始数据。

### 3.2 严格模式

设置 `CROSS_SECTION_STRICT=True` 后，程序要求同时提供电子和离子截面文件。

严格电子构表要求：

- Ar 靶粒子
- 电子入射粒子
- 弹性或 `EFFECTIVE` 动量转移截面
- 至少一个激发过程
- 至少一个电离过程
- 有限且一致的阈值
- 覆盖配置的最大能量
- collision model 可以表示的电离产物

Phelps `EFFECTIVE` 是总动量转移截面。程序先减去激发和电离截面，再得到弹性部分。

严格离子构表要求：

- 入射物种与 `ION_SPECIES` 完全一致
- 当前验证物种为 Ar+
- 靶粒子为 Ar
- 已确认的电荷交换过程
- 弹性或 `ISOTROPIC` 过程
- 覆盖配置的最大能量

设置 `CONFIRM_SYMMETRIC_BACKSCATTER_AS_CEX=True` 只确认记录中的 Ar+/Ar 对称映射。该标志不能确认其他离子或电荷态。

### 3.3 Phelps 和 Biagi 数据

[截面清单](validation/cross_sections/lxcat_manifest.json) 固定了三个 LXCat 选择：

- Phelps electron-Ar：Cenian 主比较输入
- Phelps Ar+-Ar：主离子输入
- Biagi electron-Ar：截面模型灵敏度输入

当前 Cenian 主运行器使用两个 Phelps 文件。Biagi 文件不进入主比较。

当前工作树不包含原始 LXCat 文件。LXCat 数据贡献者保留各自权利，第三方再分发需要相应许可。旧 Git 历史曾包含一份 LXCat 下载文件；公开发布前还必须清理该历史或取得书面再分发许可。

请从 LXCat 获取文件并接受其条款。然后把文件放入：

```text
.validation_private/lxcat/
```

文件名、SHA-256 和过程数量必须与清单一致。不要把原始 LXCat 文件提交到仓库。

## 4. 当前本地证据

以下结果来自 2026-08-14 的当前本地工作树：

| 检查 | 当前结果 | 说明 |
|---|---:|---|
| Pytest | 58 passed | 配置、场、粒子、随机数、碰撞、严格截面、输出和验证运行器 |
| 快速基准 | 3 passed | 圆柱真空场、常数截面碰撞率、等质量弹性守恒 |
| Python 依赖检查 | passed | 当前虚拟环境没有损坏的依赖 |

三个快速基准的当前数值如下：

| 基准 | 关键结果 |
|---|---|
| 圆柱真空电容器 | 最细网格相对 L-infinity 误差 `4.061e-5`；最低观测阶 `1.982` |
| 常数截面碰撞箱 | 期望事件数 `9000`；观测事件数 `8984`；绝对 z 分数 `0.169` |
| 等质量弹性碰撞 | 相对能量误差 `0`；绝对动量误差 `2.22e-16 kg m/s` |

这些结果检查实现和独立部件。它们不证明完整探针模型已经通过实验验证。

CI 工作流配置了以下矩阵：

- Ubuntu 和 Windows
- Python 3.10 和 3.13
- 测试与快速基准
- sdist 和 wheel 构建
- 干净环境 wheel smoke test
- 验证包数据存在性检查

本节只报告本地结果。只有远程工作流实际完成后，才能报告 GitHub CI 结果。

## 5. Cenian 2005 外部比较

### 5.1 派生实验数据

[实验 CSV](validation/experimental/cenian2005_fig2_case_rp_lambda_0p26.csv)来自 Cenian 等人 2005 年论文的 Figure 2。

来源：

- A. Cenian et al.
- Journal of Applied Physics 97, 123310 (2005)
- DOI: [10.1063/1.1938275](https://doi.org/10.1063/1.1938275)

当前数据集只包含内部一致的 `r_p/lambda_D = 0.26` 工况。

主要实验条件如下：

| 参数 | 值 |
|---|---:|
| 气体 | Ar |
| 压力 | `1.3 mTorr` |
| 电子密度 | `7.15e13 m^-3` |
| 电子温度 | `1.9 eV` |
| 离子温度 | `0.025 eV` |
| 探针半径 | `313 micrometers` |
| 探针长度 | `47 mm` |
| 偏压范围 | `-60 V` 到 `-10 V` |

[来源记录](validation/experimental/cenian2005_fig2_case_rp_lambda_0p26.provenance.json)包含像素标定、数字化方法、文件哈希和使用限制。

仓库不包含 Cenian 2005 原论文 PDF 或页面图像。根目录的 `Guide.pdf` 是另一篇采用 CC BY 4.0 许可的背景文献，不是本次实验数据源。没有找到允许仓库再分发 Cenian 2005 原文的开放许可。

### 5.2 电流符号和单位

核心结果使用：

```text
avg_current = electron-current magnitude - ion-current magnitude
```

因此，在负偏压离子收集区，`avg_current` 应为负值。这与 Cenian Figure 2 的符号一致。

`avg_conventional_current` 的符号相反。不要用它比较该实验 CSV。

宏粒子电流先按单位轴向长度计算。固定偏压运行随后乘以 `probe_length`。

Cenian 运行器使用 `probe_length=0.047 m`。它输出有符号、未缩放的安培值，不拟合比例因子，也不做直径或长度归一化。

### 5.3 运行器和输出

入口为：

```text
python -m validation.run_cenian2005
```

运行器会核对：

- LXCat 数据库角色
- 固定 SHA-256
- 固定过程数量
- 严格电子和离子构表
- 实验 CSV、来源记录和数据清单哈希
- 数值稳定性
- 加速电子 CFL
- Git 提交和工作树状态
- 每次运行必须为 `READY/PASS`，且账本、能量表越界和告警均为零

每次完整运行写入：

- `simulation_points.csv`
- `comparison.csv`
- `metrics.json`
- `manifest.json`

所有输出均标记为 `PREVIEW`。发布判断固定为 `NOT_EVALUATED_RESEARCH_PREVIEW`。

### 5.4 2026-08-14 的实际对比结果

[完整结果、逐点数据和图](validation/results/cenian2005_phelps_pilot_12mm_64c_1024p_3seed/README.md)来自 11 个偏压、3 个独立种子，共 33 次运行。

![Cenian 2005 experiment and PIC-MCC pilot comparison](validation/results/cenian2005_phelps_pilot_12mm_64c_1024p_3seed/comparison_plot.png)

| 指标 | 结果 |
|---|---:|
| 平均偏差 | `-12.282605 µA` |
| RMSE | `13.019556 µA` |
| 归一化 RMSE | `0.891511` |
| 平均绝对相对误差 | `0.896657` |
| 仿真/实验平均幅值比 | `1.897` |
| 文件定义的组合 2σ 内点数 | `0 / 11` |

所有 33 次运行都是 `READY/PASS`。电子和离子账本残差、截面能量越界和告警均为零。但是，仿真电流幅值在所有点都比实验高 `84.2%` 到 `99.9%`。因此，本轮结果不通过实验一致性检查，生产锁保持关闭。

[三点敏感性筛选](validation/results/cenian2005_sensitivity_screen/README.md)使用 `-50 V`、`-30 V` 和 `-10 V`：

- 把外域从 `12 mm` 增加到 `20 mm` 后，前两点的电流幅值下降约 `11.6%`，但 `-10 V` 上升 `15.4%`。
- 把预热从 `0.02` 增加到 `0.10` 个离子渡越时间后，前两点下降约 `8.2%`，但 `-10 V` 上升 `27.8%`。

这两个筛选都不能统一解释接近 `1.9` 倍的幅值。不得用经验比例因子掩盖该偏差。下一步应先审计外边界储库、初始离子密度与速度一致性、探针有效长度和电流归一化。

## 6. 安装和快速命令

### 6.1 环境

支持的 Python 版本为 3.10 到 3.13。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test,ui]"
```

### 6.2 本地测试

```powershell
python -m pytest
python benchmarks.py
```

### 6.3 研究预览计划

```powershell
python run_physics_accurate.py --plan-only
```

### 6.4 截面文件检查

该命令要求清单中的 Phelps 和 Biagi 文件都在本地目录中。

```powershell
python -m validation.cross_sections.validate_lxcat --data-dir .validation_private/lxcat
```

### 6.5 Cenian 计划检查

```powershell
python -m validation.run_cenian2005 --electron-lxcat .validation_private/lxcat/electron_ar_phelps_20260814.txt --ion-lxcat .validation_private/lxcat/ion_ar_phelps_20260814.txt --plan-only
```

该命令检查输入和计划，但不启动仿真。

### 6.6 Cenian 短时 pilot

输出目录必须不存在或为空。

```powershell
python -m validation.run_cenian2005 --electron-lxcat .validation_private/lxcat/electron_ar_phelps_20260814.txt --ion-lxcat .validation_private/lxcat/ion_ar_phelps_20260814.txt --output-dir validation/results/cenian2005_pilot --cells 64 --particles 1024 --dt-s 3e-11 --domain-radius-m 0.012 --seeds 101 202 303 --warmup-steps 31708 --sample-steps 31708
```

该 pilot 只检查数据链路和短时统计。它不是稳态或生产验收运行。

### 6.7 可视化前端

```powershell
streamlit run frontend/app.py
```

前端只提供研究预览。界面中的验证模式仍然锁定。

## 7. 目录

```text
core/                         物理、数值、随机数和追踪核心
frontend/                     Streamlit 研究预览界面
tests/                        自动化回归测试
validation/
  cross_sections/             LXCat 清单、说明和检查器
  experimental/               Cenian 派生 CSV 和来源记录
  results/                    本轮外部比较输出
  run_cenian2005.py           Cenian 运行器
benchmarks.py                 快速基准和收敛数据结构
run_physics_accurate.py       通用研究预览扫描入口
pyproject.toml                依赖、打包和命令行入口
```

安装 wheel 后也可以使用：

- `picsimu-benchmarks`
- `picsimu-preview-study`
- `picsimu-validate-cenian`

## 8. 已知限制

当前限制包括：

1. 生产锁关闭。任何结果都不能标记为生产结果。
2. 空间模型只有圆柱径向一维。
3. 模型没有磁场、有限探针端部和三维几何。
4. 有符号电流通过无限圆柱的单位长度结果乘探针长度得到。
5. 当前验证范围只有单电荷 Ar+ 和等质量 Ar 中性粒子。
6. 模型没有完整的多物种反应网络。
7. 外边界是给定温度和密度的储库，不是完整实验装置。
8. 边界新粒子的位移仍使用碰撞前速度。运行清单明确记录此项。
9. 可选库仑算子是预览模型。
10. 常数截面只适合测试和快速预览。
11. 外部实验比较尚未完成时间步、网格、粒子数、域半径和种子收敛。
12. Cenian 数据来自单人图像数字化，没有独立复核。
13. 实验密度和温度来自同一条探针曲线，因此不是盲验证。
14. 数字化不确定度不包含完整实验系统误差。
15. `-55 V` 和 `-60 V` 可能受有限探针端部电流影响。
16. `12 mm` pilot 域小于论文数值模型在高负偏压使用的最大约 `66.7 mm` 域。
17. 2026 年 LXCat Phelps 检索版本尚未证明与论文使用的 1997 年表逐值相同。
18. GitHub `master` 和旧 Git 历史仍含一份受限 LXCat 下载文件；公开发布前必须取得许可或经明确批准清理历史。
19. 项目代码许可证尚未由权利人选择。

## 9. 生产验证路线

当前完成项：

- [x] 配置、粒子、场、碰撞和追踪回归测试
- [x] 三个快速部件基准
- [x] 严格 Phelps 主截面链路
- [x] Biagi 灵敏度输入清单
- [x] Cenian 派生实验数据和来源记录
- [x] 有符号、未缩放电流比较运行器
- [x] sdist、wheel 和 CI 配置
- [x] 11 点、3 种子 Cenian 真实实验对比和独立复核
- [x] 三点 `20 mm` 外域与长预热方向筛选

仍需完成：

- [ ] 达到足够的预热和采样离子渡越时间
- [ ] 完成时间步收敛
- [ ] 完成网格收敛
- [ ] 完成宏粒子数收敛
- [ ] 按论文工况把域检查扩展到约 `21.8 mm`、`48.5 mm` 和 `66.7 mm`，并确认 `-55/-60 V`
- [ ] 完成多独立种子的统计稳定性检查
- [ ] 完成 Phelps 与 Biagi 截面灵敏度分析
- [ ] 审计边界储库、初始离子状态、电流归一化和有效探针长度
- [ ] 独立复核实验图像数字化
- [ ] 增加至少一个独立实验数据集
- [ ] 定义并通过预先注册的验收阈值
- [ ] 完成远程 CI 和可复现发布构建
- [ ] 处理旧 Git 历史中的 LXCat 文件并由权利人选择代码许可证

只有这些发布门全部通过后，维护者才能审查 `PHYSICS_RELEASE_READY`。不要为了启动生产模式而直接修改该锁。

## 10. 文档入口

- [人类快速说明](README_HUMAN.md)
- [Cenian 2005 运行说明](validation/CENIAN2005.md)
- [实验数据说明](validation/experimental/README.md)
- [截面数据说明](validation/cross_sections/README.md)
