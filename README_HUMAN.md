# PIC_Probe 快速说明

## 先看状态

这是 **Ar/Ar+ 1D3V PIC-MCC 研究预览**。

它不是生产级软件。它也没有完成外部实验验收。

程序中的生产锁当前关闭。生产模式会在仿真开始前停止。

旧文档中的氢工况和 Kakati 2017 归一化比较已经撤销。旧的“验证通过”和“生产级”结论均无效。

## 它能做什么

程序模拟圆柱朗缪尔探针附近的径向等离子体。

空间只有径向一维。每个电子和 Ar+ 宏粒子保留三个速度分量。

当前实现包括：

- 圆柱 Poisson 场
- CIC 电荷沉积
- 电子和 Ar+ 粒子推进
- Maxwell 通量边界注入
- 电子弹性、激发和电离碰撞
- 电离二次电子和离子
- Ar+-Ar 电荷交换
- Ar+-Ar 等质量弹性散射
- 固定偏压和 I-V 扫描
- 电流标准误
- 粒子账本和运行清单

可选库仑散射仍是预览模型。它不能用于生产发布。

## 目前有什么证据

2026-08-14 的当前本地工作树给出：

- 58 项自动化测试通过
- 3 项快速基准通过
- Phelps 和 Biagi 本地文件哈希检查通过
- 严格截面构表检查通过
- wheel 和 CI 流程已经配置

三个快速基准检查：

- 圆柱真空场与解析解
- 常数截面碰撞率
- 等质量弹性碰撞的动量和能量

这些结果说明主要代码路径可以工作。它们不等于完整模型通过实验验证。

## 当前实验比较

当前比较使用 Cenian 等人 2005 年的 Ar 探针数据：

- Journal of Applied Physics 97, 123310
- DOI: [10.1063/1.1938275](https://doi.org/10.1063/1.1938275)
- Figure 2
- `r_p/lambda_D = 0.26`
- 偏压 `-60 V` 到 `-10 V`

仓库包含：

- [派生实验 CSV](validation/experimental/cenian2005_fig2_case_rp_lambda_0p26.csv)
- [实验来源记录](validation/experimental/cenian2005_fig2_case_rp_lambda_0p26.provenance.json)
- [运行说明](validation/CENIAN2005.md)

当前工作树不包含 Cenian 2005 原论文 PDF 或原始 LXCat 文件。根目录的 `Guide.pdf` 是另一篇采用 CC BY 4.0 许可的背景文献。旧 Git 历史曾包含一份 LXCat 下载文件；公开发布前还必须清理该历史或取得书面再分发许可。

实际 11 点、3 种子对比已经完成：[结果和对比图](validation/results/cenian2005_phelps_pilot_12mm_64c_1024p_3seed/README.md)。

![实验与仿真预览对比](validation/results/cenian2005_phelps_pilot_12mm_64c_1024p_3seed/comparison_plot.png)

- 33 次运行全部 `READY/PASS`
- 粒子账本残差、截面能量越界和告警全部为零
- RMSE 为 `13.019556 µA`
- 归一化 RMSE 为 `0.891511`
- 平均绝对相对误差为 `89.67%`
- 仿真电流幅值在所有点都比实验高 `84.2%` 到 `99.9%`

因此，当前结果不通过实验一致性检查。生产锁保持关闭。

[三点敏感性筛选](validation/results/cenian2005_sensitivity_screen/README.md)表明：20 mm 外域和 0.10 个离子渡越时间预热都会改变结果，但都不能统一消除接近 1.9 倍的幅值。不要使用经验比例因子修正。

[当前代码代表点重放](validation/results/cenian2005_phelps_v3_replay_minus30_seed20260814/README.md)把 `-30 V`、种子 20260814 绑定到干净提交 `45ead2e` 和物理模型 v3。电流、标准误、样本数及状态与主对比对应行逐值相同；实验偏差仍然存在。

## 电流怎样解释

程序使用：

```text
avg_current = 电子电流幅值 - 离子电流幅值
```

因此，负偏压离子收集区的结果应为负值。这与 Cenian 图中的符号一致。

`avg_conventional_current` 的符号相反。不要用它比较当前实验 CSV。

Cenian 运行使用 `47 mm` 探针长度。输出单位是 A。

运行器不拟合比例因子。它也不做直径归一化。

## 为什么需要本地截面文件

主比较使用：

- Phelps electron-Ar
- Phelps Ar+-Ar

Biagi electron-Ar 只用于后续灵敏度分析。

LXCat 数据贡献者保留各自权利。仓库不能直接分发这些原始文件。

请从 LXCat 获取文件并接受其条款。然后把文件放入：

```text
.validation_private/lxcat/
```

程序会核对文件哈希、过程数量、物种和能量范围。

回归测试使用程序生成的合成截面，不在当前工作树中保存第三方原始截面。GitHub `master` 和旧 Git 历史仍含一份旧下载文件；公开发布前必须取得许可或经明确批准清理历史。

## 快速开始

### 1. 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test,ui]"
```

支持 Python 3.10 到 3.13。

### 2. 运行本地检查

```powershell
python -m pytest
python benchmarks.py
```

### 3. 检查 Cenian 计划

```powershell
python -m validation.run_cenian2005 --electron-lxcat .validation_private/lxcat/electron_ar_phelps_20260814.txt --ion-lxcat .validation_private/lxcat/ion_ar_phelps_20260814.txt --plan-only
```

该命令检查输入，但不启动仿真。

### 4. 启动短时 pilot

输出目录必须不存在或为空。

```powershell
python -m validation.run_cenian2005 --electron-lxcat .validation_private/lxcat/electron_ar_phelps_20260814.txt --ion-lxcat .validation_private/lxcat/ion_ar_phelps_20260814.txt --output-dir validation/results/cenian2005_pilot --cells 64 --particles 1024 --dt-s 3e-11 --domain-radius-m 0.012 --seeds 101 202 303 --warmup-steps 31708 --sample-steps 31708
```

该运行仍是短时预览。它不会给出生产 `PASS`。

### 5. 启动界面

```powershell
streamlit run frontend/app.py
```

界面只开放研究预览。

## 运行后会看到什么

Cenian 运行器会写入：

- `simulation_points.csv`
- `comparison.csv`
- `metrics.json`
- `manifest.json`

`manifest.json` 保存配置、输入哈希、版本、随机种子和数值状态。

所有当前外部比较输出都标记为 `PREVIEW`。

## 还缺什么

正式验证至少还需要：

- 更长的预热和采样时间
- 时间步收敛
- 网格收敛
- 粒子数收敛
- 外边界半径检查
- 多独立种子
- Phelps 与 Biagi 灵敏度分析
- 独立复核实验数字化
- 第二个独立实验数据集
- 预先确定的验收阈值
- 远程 CI 和可复现发布构建
- 外边界储库、初始离子状态、电流归一化和有效探针长度审计
- 约 `21.8/48.5/66.7 mm` 的论文尺度域检查
- 旧 Git 历史中的 LXCat 文件处理
- 由代码权利人选择项目许可证

模型还忽略三维几何、探针端部和磁场。边界新粒子的位移也仍使用碰撞前速度。

实验密度和温度来自同一条探针曲线。因此，Cenian 案例不是盲验证。

## 更多信息

- [完整技术说明](README.md)
- [Cenian 运行说明](validation/CENIAN2005.md)
- [实验数据说明](validation/experimental/README.md)
- [截面数据说明](validation/cross_sections/README.md)
