# 仿真对照说明（Hydrogen Langmuir Probe）

## 1. 对照目标
将最新仿真结果与已发表的氢等离子体 Langmuir probe 实验数据按同口径字段对照：压力、探针尺寸、`n_e`、`T_e`、I-V 区间。

## 2. 本次仿真数据（基准）
数据文件：`results/test_runs/iv_curve_20260310_110921.csv`

关键统计：
- 电压范围：`-30` 到 `100` V（14 点）
- `I_total` 范围：`-0.092` 到 `2.871` A/m
- 单调性：13 个增量中负斜率 0 个（单调上升）
- 重复仿真离散度：`I_total_std` 平均 `0.021` A/m，最大 `0.046` A/m
- 代表点：`I(-30V)=-0.092` A/m，`I(0V)=0.278` A/m，`I(100V)=2.871` A/m

## 3. 同口径对照表

| 对象 | 压力 | 探针尺寸 | `n_e` | `T_e` | I-V 区间 |
|---|---|---|---|---|---|
| 本次仿真（20260310_110921） | 0.3 Pa | 直径 0.4 mm（半径 0.2 mm） | 设定 `1.0e16 m^-3` | 设定 1.0 eV | `-30` 到 `100` V |
| Kakati et al., Sci Rep 2017 | `4e-4` 到 `2e-3` mbar（约 0.04-0.2 Pa） | 直径 0.15 mm，长度 10 mm | 约 `1.0e16` 到 `4.5e16 m^-3` | 约 0.6-1.2 eV | 文中给 I-V 曲线（摘要未给固定扫压上限） |
| Rousseau et al., JAP 2002 | 33-55 Pa | Shielded probe（摘要未给明确尺寸） | `2e16` 到 `1.4e18 m^-3` | 最高约 10 eV | 摘要未给固定扫压区间 |
| Roychowdhury et al., RSI 2013 | `1e-4` 到 `1e-3` mbar（约 0.01-0.1 Pa） | 自动 LP 系统（摘要未给明确尺寸） | `5.6e16` 到 `3.8e17 m^-3` | 4-14 eV | 摘要未给固定扫压区间 |

## 4. 对比结论（当前版本）
- 形状层面：本次仿真 I-V 已明显稳定，趋势合理（单调上升，重复标准差较小）。
- 量级层面：在设定 `n_e=1e16 m^-3`、`T_e=1 eV` 下，100V 电流 `2.871 A/m` 偏高。
- 按圆柱探针热通量近似估算，当前 100V 点对应“等效密度”约 `8.52e16 m^-3`（在 `T_e=1 eV` 假设下），约为设定值的 8.52 倍。
- 与 Kakati 2017 的低温低压氢实验相比，当前量级更接近其高密度端甚至偏高，与当前设定参数不完全一致。

## 5. 用于对比的文献
1. Kakati et al., Scientific Reports (2017):  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC5593904/
2. Rousseau et al., J. Appl. Phys. (2002), arXiv entry:  
   https://arxiv.org/abs/physics/0206032
3. Roychowdhury et al., Rev. Sci. Instrum. (2013):  
   https://pubmed.ncbi.nlm.nih.gov/23902054/
