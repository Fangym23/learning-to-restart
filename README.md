# first_experiment — Nesterov 重启阈值 α 的可微分析

## 1. 项目目标

研究 Nesterov 加速梯度法中**函数值重启**（function-value restart）的阈值参数 α 对最终优化质量的影响，并通过自定义 autograd 求出 dJ/dα，验证其正确性。

**核心量**：
```
J(α) = ||∇f(x_N)||²    （N 次迭代后的梯度范数平方）
```
目标是计算 dJ/dα 并用有限差分（FD）验证。

---

## 2. 文件说明

| 文件 | 作用 |
|------|------|
| `restart.py` | 自定义 autograd 函数：`EventTimeFunction`（重启时刻导数）、`EndpointMapFunction`（将时刻梯度传回状态） |
| `main.py` | 主实验脚本：构造问题、numpy 密集扫描、autograd 求导、FD 对比、6-panel 画图 |
| `alpha_study_results/` | 输出的 PNG 图片 |

---

## 3. 重启规则

每次迭代检查：
```
V = f(x_new) − α · f(x_old)
```
若 V > 0（函数值未充分下降），则**重启**：丢弃动量，令 y = x_new, θ = 1。

α 越小越容易触发重启，α > 1 几乎不重启。

---

## 4. 输出的 6-panel 图

运行后生成如 `ls_m200_n100_cond12.png`，包含：

| 面板 | 内容 |
|------|------|
| (a) | J(α) 曲线 + autograd 切线箭头（红色箭头指示梯度方向） |
| (b) | 重启次数 vs α |
| (c) | dJ/dα 导数曲线：autograd（红点线）vs 多种步长的有限差分 |
| (d) | J(α) 上的切线段（用 autograd 斜率画出） |
| (e) | 对比表格：AG vs FD(h=0.005/0.01/0.02)，含符号一致性和相对误差 |
| (f) | |AG − FD| 绝对/相对误差 vs α |

---

## 5. 如何运行

```bash
cd "code for codex/first_experiment"

# 快速测试（只跑前 2 个小问题）
python main.py --quick

# 完整运行（12 个问题，较慢）
python main.py

# 自定义参数
python main.py --iters 800 --n-dense 8000 --n-ag 120 --alpha-lo 0.01 --alpha-hi 2.0
```

参数说明：
- `--iters`：Nesterov 迭代次数（默认 500）
- `--n-dense`：numpy 密集扫描点数（默认 5000）
- `--n-ag`：autograd 求导点数（默认 80）
- `--n-table`：对比表格行数（默认 20）
- `--alpha-lo / --alpha-hi`：α 扫描范围（默认 [0.01, 1.5]）
- `--quick`：只跑前 2 个问题

输出保存在 `alpha_study_results/` 目录下。

---

## 6. 旧版代码的梯度为什么不对

旧版代码在 `about_grad.py` 的 `one_step()` 中，梯度与实际 J(α) 曲线对不上。**主要原因是缺少连续重启保护（consecutive-restart guard）**。

### 6.1 问题根源

`EventTimeFunction` 的反向传播公式为：

```
dL/dV = −1 / (V_now − V_prev)
```

当第 k 步和第 k+1 步**连续发生重启**时：
- 第 k 步：`dL/dV_k = -1 / (V_k - V_{k-1})`
- 第 k+1 步：`dL/dV_{k+1} = -1 / (V_{k+1} - V_k)`

这两步的梯度会**链式相乘**。但连续重启时 V_k 和 V_{k+1} 都很小（刚过零），
分母 `V_{k+1} - V_k` 接近 0，导致：

```
梯度 ∝ 1/(V_k - V_{k-1}) × 1/(V_{k+1} - V_k) × ...  →  爆炸
```

### 6.2 旧代码 vs 新代码对比

**旧代码**（`about_grad.py` 原始版本）— 每次重启都走可微路径：

```python
# ❌ 旧代码：无 prev_was_restart 保护
for k in range(iters):
    x_new = y - step * gradient_torch(problem, y)
    ...
    with torch.no_grad():
        V_L0 = objective_torch(problem, x_new) - alpha * objective_torch(problem, x)

    if (k > 1) and (V_L0 > 0):
        # 每次重启都走 EventTimeFunction，包括连续重启！
        V_L0 = objective_torch(problem, x_new) - alpha * objective_torch(problem, x)
        L = EventTimeFunction.apply(V_L0, V_pre.detach(), k+1)
        x_new = EndpointMapFunction.apply(L, x_new, x)
        y_new = x_new.clone()
        theta_new = 1.0

    V_pre = V_L0.detach()
    x, y, theta = x_new, y_new, theta_new
```

**新代码**（修复后）— 只在非连续重启边界走可微路径：

```python
# ✅ 新代码：加入 prev_was_restart 保护
prev_was_restart = False

for k in range(iters):
    x_new = y - step * gradient_torch(problem, y)
    ...
    with torch.no_grad():
        V_L0 = objective_torch(problem, x_new) - alpha * objective_torch(problem, x)

    if k > 1 and V_L0 > 0:
        if not prev_was_restart:
            # ✅ 只在"首次触发"处走可微路径
            V_L0 = objective_torch(problem, x_new) - alpha * objective_torch(problem, x)
            L_val = EventTimeFunction.apply(V_L0, V_pre.detach(), k + 1)
            x_new = EndpointMapFunction.apply(L_val, x_new, x)
        # 连续重启：正常 restart 但不叠加 event-time 微分
        y_new = x_new.clone()
        theta_new = 1.0
        n_restarts += 1
        prev_was_restart = True          # ← 关键：标记本步是重启
    else:
        prev_was_restart = False          # ← 重启链断开，下次可走可微路径

    V_pre = V_L0.detach()
    x, y, theta = x_new, y_new, theta_new
```

### 6.3 差异总结

| | 旧代码 | 新代码 |
|---|--------|--------|
| 连续重启 | 每次都经过 EventTimeFunction | 只在首次触发处经过 |
| 梯度行为 | 连续重启时链式 1/(V-V') 相乘 → 爆炸 | 只保留一个 1/(V-V') → 稳定 |
| `prev_was_restart` | 无 | 有，阻止连续重启叠加微分 |
| beta 计算 | 存入 list 跨 epoch 复用（图已坏） | 每步重新计算，无跨 epoch 问题 |

### 6.4 直觉解释

物理类比：重启是一个"事件"，隐函数定理告诉我们如何在事件边界处求导。但当重启**连续发生**时（k, k+1, k+2 连续重启），这不是真正的"边界"，而是一段连续重启区间。在区间内部对每个点都用隐函数定理求导没有物理意义，只有在区间的**入口**（从不重启变为重启的那个时刻）才需要走可微路径。
