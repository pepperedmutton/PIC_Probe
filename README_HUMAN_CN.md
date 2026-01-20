# PICSIMU 人类快速说明

PICSIMU 是一个 1D 圆柱坐标 PIC-MCC 模拟，用于生成高压碰撞鞘层条件下的朗缪尔探针
I-V 曲线。项目目标是生成可靠的合成数据，支撑等离子体参数推断与机器学习诊断。

## 警告

不建议人类使用者自行修改任何代码，代码修改应由最低为 DeepSeekR1 内核的 agent 执行。

## 项目在做什么

- 1D 径向模拟圆柱探针周围鞘层
- 追踪电子/离子并保持角动量守恒
- 求解圆柱 Poisson 方程，加入离子-中性 CEX 与电子-中性简化碰撞
- 输出 I-V 曲线、电势与密度分布

## 为什么要做

- 覆盖高压碰撞 regime (1-200 Torr)，弥补 OML 低压模型的不足
- 为 ML 诊断生成高压 I-V 合成数据
- 面向工业与大气压等离子体应用

## 目录结构

- `core/`：Numba 加速物理核心
- `frontend/`：Streamlit 前端
- `results/`：基准测试输出
- `README.md`：完整技术细节（AI/automation 参考）

## 快速运行

```powershell
streamlit run frontend/app.py
```

命令行快速测试：

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

## 主要输入

- 气压 (Torr)、密度 (m^-3)、电子温度 (eV)
- 探针偏压 (V)
- 数值设置：网格数、时间步长、粒子数

## 输出

- I-V 曲线数据（总电流/电子/离子）
- 电势与密度分布曲线
- `results/` 中的基准测试图

## 验证状态（摘要）

- Poisson 求解器：真空圆柱电容器测试通过
- 电子温度检查：待复核
- OML 离子动力学：通过（I_i^2 vs |V| 线性）

## Benchmark 说明

- 真空圆柱电容器：验证圆柱 Poisson 求解器与 1/r 几何项
- 电子温度检查：验证速度采样与 Boltzmann 关系（retarding 区）
- OML 离子动力学：验证角动量守恒与 I_i^2 ∝ |V| 线性

## 物理模型假设（简要）

- 1D 径向圆柱几何，无轴向/方位角空间变化
- 仅静电场，无磁场
- 物种为电子与单电荷离子（氩）
- 碰撞：离子-中性 CEX；电子-中性简化为能量损失，不生成二次粒子
- 边界：探针/外壁吸收，外边界注入以维持密度

## 局限

- 仅 1D 径向模型（无轴向/方位角空间变化）
- 碰撞模型简化，未生成二次粒子
- 静电近似（无磁场）

完整技术细节请查看 `README.md`。
