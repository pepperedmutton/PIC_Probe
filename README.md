# PICSIMU 项目文档（Agent 技术主文档）

## 文档策略

本仓库当前只保留两份 README，且统一使用中文：

- `README.md`：面向 Agent/自动化与开发维护的完整技术文档（本文件）
- `README_HUMAN.md`：面向人类使用者的快速说明

维护约定：

1. 所有技术细节先更新 `README.md`，再同步 `README_HUMAN.md` 的摘要。
2. 不新增其它根目录 README 变体，避免文档分叉。
3. 结果声明必须附证据路径（脚本、CSV、图像、说明文档）。

---

## 1. 项目概述

PICSIMU 是一个 1D 圆柱坐标 Particle-in-Cell + Monte Carlo Collisions（PIC-MCC）仿真器，
用于生成高压碰撞鞘层条件下的 Langmuir 探针 I-V 合成数据。

核心目标：

1. 建立可复现的高压等离子体 I-V 数据生成链路。
2. 为参数反演（`n_e`, `T_e`, `V_p`）与机器学习训练提供数据。
3. 在保证物理一致性的前提下，演进到生产级并行/GPU 流水线。

默认单位约定：

- 距离：m
- 速度：m/s
- 电流：A 或 A/m（按 `probe_length`）
- 温度输入：eV

---

## 2. 研究背景与动机

本项目直接承接以下工作并向高压碰撞区扩展：

> Marchand et al., *Beyond analytic approximations with machine learning inference of plasma parameters and confidence intervals*, Journal of Plasma Physics, 89(1), 2023.  
> DOI: [10.1017/S0022377823000041](https://doi.org/10.1017/S0022377823000041)

原论文覆盖范围：

- 碰撞近似：无碰撞（OMT/OML 思路）
- 压力：近 0 Torr
- 密度：`10^10 - 10^12 m^-3`
- 典型场景：空间/低压实验等离子体

PICSIMU 的补充定位：

- 物理模型：从 OMT/OML 扩展到 PIC-MCC
- 压力范围：`1 - 200 Torr`（碰撞主导）
- 密度范围：`10^14 - 10^18 m^-3`
- 应用场景：工业等离子体处理、大气压等离子体、高压放电诊断

技术路线（旧版 README 全量信息保留并整合）：

```text
无碰撞 OMT/OML + 回归推断  --->  高压 PIC-MCC 合成数据  --->  ML 参数反演与部署
```

---

## 3. 总体架构与目录

### 3.1 架构分层

- `core/`：物理内核（Numba 加速，计算主链路）
- `frontend/`：Streamlit 可视化与交互
- 根目录脚本：基准测试、物理测试/数据生成入口

### 3.2 目录结构（按当前仓库）

```text
PICSIMU/
  core/
    config.py
    particles.py
    fields.py
    collisions.py
    simulation.py
    cross_sections.py
    lxcat_parser.py
    cs_txt_adapter.py
    smooth_density_impl.py
  frontend/
    app.py
  benchmarks.py
  run_physics_accurate.py
  CS.txt
  results/
    benchmarks/
    test_runs/
    production/
  README.md
  README_HUMAN.md
```

---

## 4. 物理模型摘要

### 4.1 几何与方程

- 1D 径向圆柱域：`r ∈ [R_MIN, R_MAX]`
- 探针位于 `R_MIN`，外壁位于 `R_MAX`
- 电势满足圆柱 Poisson：

```text
(1/r) * d/dr (r * dphi/dr) = -rho / epsilon_0
```

- Dirichlet 边界：

```text
phi(R_MIN) = V_bias
phi(R_MAX) = V_wall
```

### 4.2 粒子状态与推进

粒子状态：`(r, v_r, v_theta)`  
径向动力学：

```text
dv_r/dt = (q/m) * E_r + v_theta^2 / r
```

角动量守恒：

```text
v_theta_new = v_theta_old * (r_old / r_new)
```

推进器使用二阶速度 Verlet（旧版 README 对应说明已保留）。

### 4.3 电荷加权与体积修正

使用 CIC（Cloud-in-Cell）线性加权，并按圆柱壳体积归一化：

```text
V_j ≈ 2*pi*r_j*dr   (单位长度)
rho_j = q_weighted / V_j
```

### 4.4 碰撞模型（MCC）

离子-中性碰撞：

- CEX（电荷交换）
- 可选弹性散射
- 概率：

```text
P = 1 - exp(-n_g * sigma_total * v * dt)
```

电子-中性碰撞：

- 弹性/激发/电离
- 支持能量依赖截面（LXCat 或 `CS.txt` 适配）
- 无表时退化为常数截面
- 电离后可选生成二次电子/离子宏粒子（`ENABLE_IONIZATION_SECONDARIES`）

可选库仑散射：

- 电子-离子、离子-离子 pitch-angle 近似散射

### 4.5 注入与电流口径

外边界注入采用 Maxwellian 通量估计，径向速度按通量分布采样（Rayleigh 形式）：

```text
P(v_r) ∝ v_r * exp(-m v_r^2 / (2kT)), v_r >= 0
v_r = v_th * sqrt(-2 ln U)
```

离子注入可加 Bohm 漂移（`ION_INJECTION_BOHM=True`）：

```text
u_B = sqrt(e*Te/m_i)
```

电流定义（电子电流按正幅值报告）：

```text
I_electron = (N_e_hit * |q_e|) / dt
I_ion      = (N_i_hit * q_i) / dt
I_total    = I_electron - I_ion
```

---

## 5. 数值方法与离散化细节

### 5.1 网格

```text
N_nodes = N_CELLS + 1
r_j = R_MIN + j*dr
dr = (R_MAX - R_MIN)/N_CELLS
```

### 5.2 粒子推进（velocity-Verlet）

1. 计算旧位置加速度：`a_old = (q/m)E + v_theta^2/r`
2. 更新位置：`r_new = r_old + v_r*dt + 0.5*a_old*dt^2`
3. 角动量守恒更新 `v_theta`
4. 新位置重算 `a_new`
5. 更新速度：`v_r_new = v_r_old + 0.5*(a_old+a_new)*dt`

### 5.3 粒子边界

- `r <= R_MIN`：吸收并计入探针电流
- `r >= R_MAX`：吸收或反射（可配置）
- 死粒子槽位可用于后续注入复用

### 5.4 CIC 加权

```text
xi = (r - R_MIN)/dr
j = floor(xi)
w = xi - j
rho[j]   += q*(1-w)
rho[j+1] += q*w
```

### 5.5 圆柱 Poisson 离散

通量形式离散后得到三对角系统：

```text
a_j*phi_{j-1} + b_j*phi_j + c_j*phi_{j+1} = d_j
```

其中：

```text
a_j = r_{j-1/2}/(r_j*dr^2)
b_j = -(r_{j+1/2}+r_{j-1/2})/(r_j*dr^2)
c_j = r_{j+1/2}/(r_j*dr^2)
d_j = -rho_j/epsilon_0
```

线性系统用 TDMA（Thomas）`O(N)` 求解。

### 5.6 电场

```text
E = -dphi/dr
```

- 内点：中心差分
- 边界：单边差分

### 5.7 初始分布（加速收敛）

采用 Child-Langmuir 风格初始电势剖面，构造接近鞘层的初值，减少 burn-in：

```text
phi(r) = V_bias + (V_wall - V_bias) * ((r-R_MIN)/s)^(4/3)
```

并配套：

- 电子密度：Boltzmann 关系
- 离子密度：连续性近似（Bohm 速度）
- 位置采样按 `n(r)*r` 权重

### 5.8 电压扫描（warm-start）

```text
for V in voltages:
  设置 V_bias
  burn-in
  sampling
  记录 I_total/I_electron/I_ion
```

默认支持从上一偏压态继续（warm-start），并支持 bias ramp 平滑过渡。

### 5.9 单步主循环顺序

1. 外边界注入
2. 粒子推进
3. 电子碰撞
4. 二次粒子生成（可选）
5. 离子碰撞
6. 库仑散射（可选）
7. 电荷加权 +（可选）平滑
8. Poisson 求解 + 电场更新

---

## 6. 数据模型、输出与命名规范

### 6.1 Core 数据模型

粒子数组（扁平 1D NumPy）：

- `r_e`, `vr_e`, `vt_e`
- `r_i`, `vr_i`, `vt_i`

场与网格数组：

- `r_grid`, `phi`, `E`, `rho`, `rho_e`, `rho_i`, `ne`, `ni`

`SimulationResult` 输出：

- `avg_current`
- `r_grid`, `phi`, `ne`, `ni`
- `ion_r`, `ion_vr`

`scan_voltage_range()` 返回：

- `"voltages"`
- `"I_total"`
- `"I_electron"`
- `"I_ion"`

### 6.2 结果目录

- `results/benchmarks/`：`benchmarks.py` 输出（CSV + PNG）
- `results/test_runs/`：开发测试与单次运行输出
- `results/production/`：生产级数据目录（预留）

### 6.3 I-V 文件命名约定

当前主测试脚本采用时间戳命名：

- `iv_curve_YYYYMMDD_HHMMSS.csv`
- `iv_curve_YYYYMMDD_HHMMSS.png`

不再使用参数拼接作为主命名方案。

---

## 7. 稳定性检查与数值约束

`Config.stability_warnings()` 包含以下约束检查：

1. Debye 解析：`dr < lambda_D`
2. 等离子体频率：`dt * omega_pe < 0.2`
3. 电子 CFL：`v_th,e * dt < dr`
4. 离子 CFL：`v_th,i * dt < dr`

触发时给出 RuntimeWarning，提醒可能的噪声或失稳风险。

---

## 8. 当前实现假设与局限

### 8.1 核心假设

1. 静电近似（不含磁场）
2. 1D 径向几何（无轴向/方位角空间分辨）
3. 中性气体温度固定约 `0.026 eV`（~300K）
4. 外边界采用 Maxwellian 储库注入模型
5. 外壁电势可配置，默认 `0 V`
6. 离子注入可启用 Bohm 漂移
7. 截面可用常数或能量依赖表
8. 电离二次粒子、离子弹性、库仑碰撞均为可选开关

### 8.2 局限

1. 空间维度仍为 1D 径向
2. 化学与碰撞过程为简化模型，不是全反应网络
3. 高压强区对微观过程的细节仍受截面质量与模型开关影响

---

## 9. 运行方式

### 9.1 命令行快速测试

```powershell
@'
from core.config import Config
from core.simulation import PICSimulation

cfg = Config()
sim = PICSimulation(cfg, n_particles=2000, v_bias=-10.0, seed=1)
res = sim.run(n_steps=200, n_warmup=100)
print(res.avg_current)
'@ | python -
```

### 9.2 基准与主测试入口

- `python benchmarks.py`
- `python run_physics_accurate.py`

### 9.3 前端

```powershell
streamlit run frontend/app.py
```

### 9.4 使用 LXCat/CS 截面

可在 `Config` 设置：

- `LXCAT_ELECTRON_FILE`
- `LXCAT_ION_FILE`

默认会尝试读取仓库根目录 `CS.txt`。若不希望启用，设为 `None` 即可。  
`core/cross_sections.py`、`core/lxcat_parser.py`、`core/cs_txt_adapter.py` 负责解析与表格插值。

---

## 10. Benchmark 基准算例记录（LabArgon）

旧 README 的基准算例信息完整保留如下：

- 名称：`LabArgon-0p1Torr-2eV-IV`
- 气体：Argon（Ar+，40 AMU）
- `N0 = 1.0e16 m^-3`
- `Te = 2.0 eV`
- `Ti = 0.026 eV`
- `P_Torr = 0.1`
- `R_MIN = 1.5e-4 m`
- `R_MAX = 5.0e-3 m`
- `V_WALL = 0 V`
- `L = 0.01 m`

数值设置：

- `N_CELLS = 100`
- `DT = 20e-12 s`
- `sigma_cex = 8.0e-18 m^2`
- `n_particles = 10000`（每物种）
- `V_start=-40 V`, `V_end=+10 V`, `n_steps=21`
- `n_burn_in=20000`, `n_sampling=20000`

历史参考输出（旧文档记录）：

- `results/iv_data_labargon_posI.csv`
- `results/iv_curve_labargon_posI.png`
- `results/iv_curve_labargon_semilog_posI.png`

预期特征：

- I-V 随电压上升单调上升
- 浮动电位约 `-10 V`
- 半对数电子支线近似线性

---

## 11. 物理模型验证状态（已完成）

### Test 1：真空圆柱电容器

- 目的：验证圆柱 Poisson 求解器
- 结果：最大相对误差约 `0.0017%`
- 状态：通过（2026-01-20）
- 输出：
  - `results/benchmarks/benchmark_test1_vacuum_capacitor.csv`
  - `results/benchmarks/benchmark_test1_vacuum_capacitor.png`

### Test 2：电子温度推断

- 目的：验证电子速度采样与 Boltzmann 关系
- 配置：无碰撞、`Te=2.0 eV`、`V=-10~-2 V`
- 结果：`slope = 0.494 V^-1`, 推断 `Te = 2.02 eV`
- 状态：通过（2026-01-20）
- 输出：
  - `results/benchmarks/benchmark_test2_electron_temperature.csv`
  - `results/benchmarks/benchmark_test2_electron_temperature.png`

### Test 3：OML 离子动力学

- 目的：验证角动量守恒与 `I_i^2 ∝ |V|`
- 配置：`R_MIN=500 um`, `N0=5e15 m^-3`, 无碰撞
- 结果：`R^2 = 0.993`
- 状态：通过（2026-01-20）
- 输出：
  - `results/benchmarks/benchmark_test3_oml_ion.csv`
  - `results/benchmarks/benchmark_test3_oml_ion.png`

### Test 4：碰撞阻尼

- 目的：验证 CEX 导致离子电流随压强抑制
- 结果：`I_ion(10 Torr) / I_ion(0 Torr) = 0.000`
- 状态：通过（2026-01-20）
- 输出：
  - `results/benchmarks/benchmark_test4_collisional_damping.csv`
  - `results/benchmarks/benchmark_test4_collisional_damping.png`

### Test 5：氢等离子体实验对照（直径归一化）

- 目的：与公开氢等离子体实验做同口径量级核对
- 参考文献：Kakati et al., *Scientific Reports* 7, 490 (2017), PMCID: PMC5593904
- 关键事实：
  - 实验探针直径：`0.15 mm`
  - 仿真探针直径：`0.4 mm`
  - `+80 V` clean plasma 读图约：`13.5 mA`
- 直径归一化：

```text
I_norm = I_exp * (d_sim / d_exp)
       = 13.5 mA * (0.4 / 0.15)
       = 36.0 mA
```

按 `L = 10 mm` 换算为每单位长度：`~3.6 A/m`

- 仿真：`I_sim(+80V) ≈ 2.700 A/m`
  - 数据文件：`results/test_runs/iv_curve_20260310_110921.csv`
- 差异：约 `25%`
- 结论：作为一阶量级一致性验证，通过

核心结论（旧版全量结论保留）：

- 圆柱几何项处理正确
- 角动量守恒实现正确
- 速度 Verlet 积分精度满足要求
- CIC + 体积修正正确
- OML 标度律得到验证
- 碰撞阻尼趋势得到验证
- 与公开氢实验量级达到一阶一致

---

## 12. 物理模型校准说明（旧版信息整合）

### 12.1 校准目标

1. 验证场求解：`1/r` 几何项正确性
2. 验证电子统计：`ln(I_e)` 与偏压关系
3. 验证离子动力学：OML 标度与角动量守恒
4. 验证碰撞趋势：CEX 抑制随压强增强

### 12.2 为什么是这四类 benchmark

1. 真空圆柱电容器：解析可比，最低成本验证 Poisson
2. 电子温度检查：直接检验速度采样和通量注入
3. OML 离子动力学：直接检验轨道动力学与守恒项
4. 碰撞阻尼：直接检验碰撞算子与压强响应

### 12.3 OML 测试关键配置（旧版表格保留）

| 参数 | 值 | 说明 |
|---|---|---|
| 探针半径 | `R_MIN = 5.0e-4 m` | 500 um |
| 外壁半径 | `R_MAX = 5.0e-3 m` | 5 mm |
| 密度 | `N0 = 5.0e15 m^-3` | 中等密度 |
| 电子温度 | `Te = 2.0 eV` | 实验室典型 |
| 离子温度 | `Ti = 0.026 eV` | 室温 |
| 宏粒子数 | `20000` | 每个物种 |
| 稳定步数 | `200000` | 充分热化 |
| 采样步数 | `80000` | 提升统计 |
| 扫描区间 | `-50 -> -10 V` | 离子饱和区 |
| 碰撞 | `P=0, sigma_cex=0` | OML 条件 |

Debye 检查（旧版说明保留）：

```text
lambda_D ≈ 149 um,  r_probe = 500 um > lambda_D
```

符合 OML 条件。

### 12.4 OML 样本数据（旧版保留）

| V_bias (V) | \|V\| (V) | I_ion (A) | I_ion^2 (A^2) | 拟合值 (A^2) | 相对误差 |
|---|---:|---:|---:|---:|---:|
| -50 | 50 | 0.02359 | 5.564e-4 | 5.557e-4 | 0.13% |
| -45 | 45 | 0.02212 | 4.895e-4 | 5.053e-4 | 3.13% |
| -40 | 40 | 0.02121 | 4.498e-4 | 4.548e-4 | 1.10% |
| -35 | 35 | 0.02030 | 4.120e-4 | 4.044e-4 | 1.88% |
| -30 | 30 | 0.01903 | 3.623e-4 | 3.540e-4 | 2.33% |
| -25 | 25 | 0.01789 | 3.201e-4 | 3.036e-4 | 5.45% |
| -20 | 20 | 0.01622 | 2.632e-4 | 2.532e-4 | 3.95% |
| -15 | 15 | 0.01404 | 1.972e-4 | 2.027e-4 | 2.74% |
| -10 | 10 | 0.01164 | 1.356e-4 | 1.523e-4 | 10.99% |

残差解读（旧版保留）：

- 高电压端（-50~-35V）：误差 < 3.2%
- 中电压端（-30~-20V）：误差约 2.3%~5.5%
- 低电压端（-15~-10V）：偏离增大（过渡区 + 小电流统计涨落）

### 12.5 碰撞阻尼样本（旧版保留）

| P_Torr | I_ion (A/m) |
|---:|---:|
| 0.0 | 3.355e-3 |
| 0.1 | 1.342e-3 |
| 0.5 | 1.006e-3 |
| 1.0 | 3.355e-4 |
| 3.0 | 0.000e0 |
| 5.0 | 0.000e0 |
| 10.0 | 0.000e0 |

### 12.6 校准总表（旧版保留）

| 测试项目 | 指标 | 结果 | 状态 |
|---|---|---|---|
| Test 1 - Poisson | 最大相对误差 | 0.0017% | 优秀 |
| Test 2 - 电子温度 | 推断 Te | 2.02 eV | 通过 |
| Test 3 - OML | R^2 | 0.993 | 通过 |
| Test 4 - 碰撞阻尼 | 抑制比 | 0.000 | 通过 |
| Test 5 - 氢实验归一化 | +80V 电流量级 | `2.700 A/m` vs `3.6 A/m` | 通过（差异约25%） |

综合结论：

- 当前模型可作为生产级数据生成的物理基线
- 后续优化应保持对该基线的数值一致性回归

---

## 13. 当前主测试口径（最新实现）

`run_physics_accurate.py` 当前主配置（旧信息 + 最新变更统一）：

- 等离子体：氢
- 压力：`0.3 Pa`（约 `2.25e-3 Torr`）
- 密度：`1e16 m^-3`
- 温度：`1 eV`
- 探针直径：`0.4 mm`
- 扫描区间：`-30 -> +100 V`
- 每偏压点：稳定后采样，重复 5 次取均值
- 关闭二次电离电子：`ENABLE_IONIZATION_SECONDARIES=False`
- 输出：`results/test_runs/iv_curve_<timestamp>.csv/png`

---

## 14. Agent 开发约束（由旧版 Agent Guide 合并）

### 14.1 必需技术栈

- Python 3.10+
- NumPy
- Numba（重循环必须 `nopython=True`）
- Streamlit
- Matplotlib

### 14.2 架构硬约束

1. 几何固定为 1D 径向圆柱域
2. 粒子必须保留 `v_theta`
3. 径向推进必须包含离心项 `v_theta^2 / r`
4. 必须显式保持角动量守恒
5. 加权必须做圆柱壳体积归一化
6. Poisson 必须用圆柱拉普拉斯离散 + TDMA
7. 碰撞至少包含离子-中性 CEX 和电子-中性基本过程
8. 核心重循环必须 Numba 化

### 14.3 模块职责

- `core/config.py`：参数常量、稳定性检查
- `core/particles.py`：推进、边界、加权
- `core/fields.py`：Poisson 与电场
- `core/collisions.py`：MCC 与可选库仑散射
- `core/simulation.py`：主循环、注入、扫描、统计
- `core/cross_sections.py`：截面加载与统一网格
- `core/lxcat_parser.py`：文本截面解析
- `core/cs_txt_adapter.py`：`CS.txt` 适配桥接
- `frontend/app.py`：UI 与图形展示

### 14.4 代码行为规范

1. 避免在 jitted 热循环中动态分配
2. 避免在热循环中引入 Python 对象
3. 新增物理功能必须附带基准或回归点
4. 并行化/GPU 化前后必须与 CPU 基线对比

### 14.5 前端行为要求

至少暴露以下输入：

- 压力
- 密度
- 电子温度
- 探针偏压

至少展示：

- I-V 总电流、电子、离子曲线
- 可选 `ln(I_e)` 半对数支线

### 14.6 可扩展方向（旧版保留）

已实现：

- 能量依赖截面支持（LXCat/自定义）
- 可选二次粒子生成

可继续扩展：

- 多离子组分
- 更完整反应网络与诊断量（EEDF、鞘层宽度、能量守恒审计）
- 更高维几何

---

## 15. 下一阶段任务声明（Production + GPU）

下一阶段目标（旧版声明保留并落地）：

1. 构建 production 级大规模合成数据管线（批量参数扫描 + 元数据追踪）。
2. 对偏压点、重复实验、参数批次进行并行化改造。
3. 对热点核函数（推进、碰撞、加权、场计算）进行 GPU 加速。
4. 保证并行/GPU 改造后与当前 CPU 物理基线一致且可复现。

建议验收指标：

- 吞吐提升倍数
- 单点仿真延迟
- 与 CPU 基线电流曲线偏差
- 随机种子可复现性

---

## 16. 结论

本 README 已将旧版文档中的信息完整整合为中文单文档版本，覆盖：

- 研究背景与目标
- 物理模型与数值实现细节
- 数据结构与输出规范
- 全部基准与实验对照结论
- Agent 约束与下一阶段生产路线

该文档可作为当前仓库的单一技术基线。
