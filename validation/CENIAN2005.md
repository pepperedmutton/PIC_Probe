# Cenian 2005 实验验证运行器

该运行器把仿真的有符号、未缩放电流与 Cenian 2005 图 2 的实验电流直接比较。它不拟合比例因子，也不归一化电流。

符号映射使用 `PICSimulation.avg_current`，即电子电流幅值减离子电流幅值。负偏压结果应为负值。不要使用符号相反的 `avg_conventional_current`。

运行器只处理 `r_p/lambda_D = 0.26` 工况。实验偏压范围固定为 `-60 V` 到 `-10 V`。

配置使用 Maxwell 热通量外边界，并设置 `ION_INJECTION_BOHM=False`。探针表面完全吸收粒子。

## 必要输入

必须显式提供两个本地 LXCat 文件：

- Phelps 电子-Ar 截面文件（固定为 2026-08-14 检索版本）
- Phelps Ar+-Ar 截面文件

运行器会核对数据库标识、SHA-256、过程数量和严格构表结果。任一项不同都会停止。

## 只检查计划

```powershell
python validation/run_cenian2005.py `
  --electron-lxcat C:\data\phelps_ar_electron.txt `
  --ion-lxcat C:\data\phelps_ar_ion.txt `
  --plan-only
```

计划会给出配置哈希、输入文件哈希、稳定性参数和预计采样时间。该操作不启动仿真。

## 短时间 pilot

```powershell
python validation/run_cenian2005.py `
  --electron-lxcat C:\data\phelps_ar_electron.txt `
  --ion-lxcat C:\data\phelps_ar_ion.txt `
  --output-dir results\validation\cenian2005_pilot `
  --cells 64 `
  --particles 1024 `
  --dt-s 3e-11 `
  --domain-radius-m 0.012 `
  --seeds 101 202 303 `
  --warmup-steps 31708 `
  --sample-steps 31708
```

该 pilot 只检查数据链路。它的采样时间不足以形成生产验证证据。

运行器写入以下文件：

- `simulation_points.csv`
- `comparison.csv`
- `metrics.json`
- `manifest.json`

所有输出均标记为 `PREVIEW`。运行器不会给出生产 `PASS`。

可以用 `--voltages -50 -30 -10` 选择固定实验网格的子集，用于方向筛选。子集运行仍使用同一个固定实验 CSV 和来源哈希。

## 当前结果

2026-08-14 的 [11 点、3 种子结果](results/cenian2005_phelps_pilot_12mm_64c_1024p_3seed/README.md)给出 `13.019556 µA` 的 RMSE 和 `0.891511` 的归一化 RMSE。仿真电流幅值比实验高 `84.2%` 到 `99.9%`，因此不通过实验一致性检查。

[三点敏感性筛选](results/cenian2005_sensitivity_screen/README.md)表明，20 mm 外域和较长预热都会改变电流，但不能统一解释该系统偏差。

## 后续正式计算

正式实验比较必须增加采样时间和独立种子数。`12 mm` 只用于 pilot。论文模型在 `-10 V`、`-30 V` 和 `-50 V` 分别使用约 `21.8 mm`、`48.5 mm` 和 `66.7 mm` 的外域。正式域检查必须至少覆盖这些范围，并为 `-55 V` 和 `-60 V` 单独确认域收敛。

论文说明，`|U_p| > 50 V` 时，`47 mm` 探针的有限长度端部电流可能显著。当前无限圆柱模型不表示该效应。因此，`-55 V` 和 `-60 V` 点不能进入生产验收。

论文引用 1997 年的 Phelps 电子表。当前输入是 2026 年从 LXCat 检索的同名数据库版本，尚未证明两者数值完全相同。

实验密度和温度来自同一条探针曲线，实验曲线也只有一次数字化结果。因此，该案例不能单独证明诊断结果准确。
