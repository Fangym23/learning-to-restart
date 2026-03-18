"""
Study of d(||∇f(x_final)||²)/d(alpha):
  1. Wide-range, high-density sweep of alpha → plot J(α) = ||∇f(x_final)||²
  2. Autograd derivative at many α points → smooth dJ/dα curve
  3. Numerical finite-difference derivative for comparison
  4. Table: pointwise FD vs autograd comparison with error metrics
  5. Compare on multiple problem scales and types
"""
from __future__ import annotations

import math
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))

import matplotlib
matplotlib.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
})
import matplotlib.pyplot as plt
import numpy as np
import torch
from restart import EventTimeFunction, EndpointMapFunction


# =========================
# Problem definition
# =========================
@dataclass
class Problem:
    """最小二乘问题的数据容器。

    属性:
        A       (m×n ndarray): 设计矩阵
        b       (m   ndarray): 观测向量
        x_star  (n   ndarray): 最小二乘解 (A^T A)^{-1} A^T b
        f_star  (float):       最优残差 ||Ax*-b||²
        L       (float):       Lipschitz 常数 = 2·λ_max(A^T A)
        mu      (float):       强凸参数 = λ_min(A^T A)
        q_star  (float):       μ/L_gram, Nesterov 动量参数
        name    (str):         问题标签（用于图标题）
    """
    A: np.ndarray
    b: np.ndarray
    x_star: np.ndarray
    f_star: float
    L: float
    mu: float
    q_star: float
    name: str = ""


def make_least_squares(
    m: int = 200, n: int = 100, cond: float = 12.0, seed: int = 7, name: str = ""
) -> Problem:
    """构造条件数可控的最小二乘问题 min_x ||Ax-b||²。

    通过 QR 分解 + geomspace 奇异值精确控制 cond(A)。
    输入: m(行), n(列), cond(条件数), seed(随机种子)
    输出: Problem 实例
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    singular_values = np.geomspace(cond, 1.0, n)
    A = U @ np.diag(singular_values) @ V.T

    x_true = rng.standard_normal(n)
    b = A @ x_true + 0.05 * rng.standard_normal(m)

    x_star, *_ = np.linalg.lstsq(A, b, rcond=None)
    residual = A @ x_star - b
    f_star = float(residual @ residual)

    gram = A.T @ A
    eigvals = np.linalg.eigvalsh(gram)
    mu_val = float(np.min(eigvals))
    L_gram = float(np.max(eigvals))
    L_val = 2.0 * L_gram
    q_star = max(mu_val / L_gram, 0.0)

    label = name or f"LS(m={m},n={n},κ={cond:.0f})"
    return Problem(A=A, b=b, x_star=x_star, f_star=f_star, L=L_val, mu=mu_val, q_star=q_star, name=label)


def make_sparse_least_squares(
    m: int = 200, n: int = 100, cond: float = 12.0, density: float = 0.3, seed: int = 42, name: str = ""
) -> Problem:
    """构造稀疏设计矩阵的最小二乘问题。density 控制非零元素比例。"""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    singular_values = np.geomspace(cond, 1.0, n)
    A_dense = U @ np.diag(singular_values) @ V.T
    mask = rng.random((m, n)) < density
    A = A_dense * mask
    gram = A.T @ A
    eigvals = np.linalg.eigvalsh(gram)
    scale = cond / max(np.sqrt(eigvals.max()), 1e-10)
    A = A * scale

    x_true = rng.standard_normal(n)
    b = A @ x_true + 0.05 * rng.standard_normal(m)
    x_star, *_ = np.linalg.lstsq(A, b, rcond=None)
    residual = A @ x_star - b
    f_star = float(residual @ residual)

    gram = A.T @ A
    eigvals = np.linalg.eigvalsh(gram)
    mu_val = float(np.min(eigvals))
    L_gram = float(np.max(eigvals))
    L_val = 2.0 * L_gram
    q_star = max(mu_val / L_gram, 0.0) if L_gram > 0 else 0.0

    label = name or f"SparseLS(m={m},n={n},κ={cond:.0f},d={density})"
    return Problem(A=A, b=b, x_star=x_star, f_star=f_star, L=L_val, mu=mu_val, q_star=q_star, name=label)


def make_correlated_least_squares(
    m: int = 200, n: int = 100, cond: float = 12.0, corr: float = 0.9, seed: int = 13, name: str = ""
) -> Problem:
    """构造列相关设计矩阵的最小二乘。corr 为 Toeplitz 相关系数 ρ。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    C = corr ** np.abs(idx[:, None] - idx[None, :])
    L_chol = np.linalg.cholesky(C)
    Z = rng.standard_normal((m, n))
    A_corr = Z @ L_chol.T

    U, s, Vt = np.linalg.svd(A_corr, full_matrices=False)
    s_new = np.geomspace(cond, 1.0, n)
    A = U @ np.diag(s_new) @ Vt

    x_true = rng.standard_normal(n)
    b = A @ x_true + 0.05 * rng.standard_normal(m)
    x_star, *_ = np.linalg.lstsq(A, b, rcond=None)
    residual = A @ x_star - b
    f_star = float(residual @ residual)

    gram = A.T @ A
    eigvals = np.linalg.eigvalsh(gram)
    mu_val = float(np.min(eigvals))
    L_gram = float(np.max(eigvals))
    L_val = 2.0 * L_gram
    q_star = max(mu_val / L_gram, 0.0)

    label = name or f"CorrLS(m={m},n={n},κ={cond:.0f},ρ={corr})"
    return Problem(A=A, b=b, x_star=x_star, f_star=f_star, L=L_val, mu=mu_val, q_star=q_star, name=label)


# =========================
# Objective / gradient
# =========================
def objective(problem: Problem, x: np.ndarray) -> float:
    """f(x) = ||Ax - b||²"""
    residual = problem.A @ x - problem.b
    return float(residual @ residual)


def gradient_np(problem: Problem, x: np.ndarray) -> np.ndarray:
    """∇f(x) = 2A^T(Ax - b)"""
    return 2.0 * (problem.A.T @ (problem.A @ x - problem.b))


def theta_update(theta_prev: float, q: float) -> float:
    """Nesterov 动量参数 θ 的更新公式。q=0 退化为标准加速。"""
    b = theta_prev * theta_prev - q
    disc = b * b + 4.0 * theta_prev * theta_prev
    return 0.5 * (-b + math.sqrt(disc))


# =========================
# Numpy sweep
# =========================
def nesterov_restart_numpy(
    problem: Problem, iters: int, step: float, q: float, alpha: float
) -> tuple[np.ndarray, float, int]:
    """Numpy 版 Nesterov + 函数值重启。纯前向，无自动微分。

    重启条件: f(x_new) > α·f(x)  (函数值未下降时重启)
    输入: problem, iters(迭代次数), step(步长=1/L), q(动量参数), alpha(重启阈值)
    输出: (x_final, ||∇f(x_final)||², 重启次数)
    """
    n = problem.A.shape[1]
    x = np.zeros(n)
    y = x.copy()
    theta = 1.0
    n_restarts = 0

    for k in range(1, iters + 1):
        x_new = y - step * gradient_np(problem, y)
        theta_new = theta_update(theta, q)
        beta = theta * (1.0 - theta) / (theta * theta + theta_new)
        y_new = x_new + beta * (x_new - x)

        if k > 2 and objective(problem, x_new) > alpha * objective(problem, x):
            y_new = x_new.copy()
            theta_new = 1.0
            n_restarts += 1

        x, y, theta = x_new, y_new, theta_new

    g = gradient_np(problem, x)
    return x, float(g @ g), n_restarts


def sweep_alpha_numpy(
    problem: Problem, iters: int, step: float, q: float, alpha_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对 alpha_grid 中每个 α 跑一遍 Nesterov，收集 J(α) 和重启次数。"""
    gns = np.empty(len(alpha_grid))
    rcounts = np.empty(len(alpha_grid), dtype=int)
    for i, a in enumerate(alpha_grid):
        _, gns[i], rcounts[i] = nesterov_restart_numpy(problem, iters, step, q, float(a))
    return alpha_grid, gns, rcounts


def numerical_derivative(alphas: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从密集扫描数据计算有限差分导数 dJ/dα。输出: (中点α数组, 导数数组)。"""
    mid = (alphas[:-1] + alphas[1:]) / 2.0
    dv = np.diff(values) / np.diff(alphas)
    return mid, dv


# =========================
# Torch autograd
# =========================
@dataclass
class TorchProblem:
    """Problem 的 PyTorch 版本，保持数据一致，支持自动微分。"""
    A: torch.Tensor
    b: torch.Tensor
    x_star: torch.Tensor
    f_star: float
    L: float
    q_star: float


def make_torch_problem(p: Problem) -> TorchProblem:
    """numpy Problem → torch TorchProblem（float64，数据完全一致）。"""
    return TorchProblem(
        A=torch.from_numpy(p.A.copy()),
        b=torch.from_numpy(p.b.copy()),
        x_star=torch.from_numpy(p.x_star.copy()),
        f_star=p.f_star, L=p.L, q_star=p.q_star,
    )


def objective_torch(tp: TorchProblem, x: torch.Tensor) -> torch.Tensor:
    residual = tp.A @ x - tp.b
    return residual @ residual


def gradient_torch(tp: TorchProblem, x: torch.Tensor) -> torch.Tensor:
    return 2.0 * (tp.A.T @ (tp.A @ x - tp.b))


def one_step_alpha(
    tp: TorchProblem, iters: int, step: float, q: float, alpha: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """
    硬阈值 restart + EventTimeFunction/EndpointMapFunction 求导。
    重启规则：V = f(x_new) - α·f(x) > 0 时触发 restart（不变）。
    求导规则：按理论文档中的 EventTimeFunction / EndpointMapFunction（不变）。
    唯一修复：连续 restart 段内只在首次触发处走可微路径，
    避免 1/(V_k - V_{k-1}) 链式相乘导致梯度爆炸。
    """
    n = tp.A.shape[1]
    x = torch.zeros(n, dtype=tp.A.dtype)
    y = x.clone()
    theta = 1.0
    V_pre = torch.tensor(0.0, dtype=tp.A.dtype)
    n_restarts = 0
    prev_was_restart = False

    for k in range(iters):
        x_new = y - step * gradient_torch(tp, y)
        theta_new = theta_update(theta, q)
        beta = theta * (1.0 - theta) / (theta * theta + theta_new)
        y_new = x_new + beta * (x_new - x)

        with torch.no_grad():
            V_L0 = objective_torch(tp, x_new) - alpha * objective_torch(tp, x)

        if k > 1 and V_L0 > 0:
            if not prev_was_restart:
                # 非连续 restart 边界：走可微路径（理论公式）
                V_L0 = objective_torch(tp, x_new) - alpha * objective_torch(tp, x)
                L_val = EventTimeFunction.apply(V_L0, V_pre.detach(), k + 1)
                x_new = EndpointMapFunction.apply(L_val, x_new, x)
            # 连续 restart：正常 restart 但不叠加 event-time 微分
            y_new = x_new.clone()
            theta_new = 1.0
            n_restarts += 1
            prev_was_restart = True
        else:
            prev_was_restart = False

        V_pre = V_L0.detach()
        x, y, theta = x_new, y_new, theta_new

    return torch.sum(gradient_torch(tp, x) ** 2), n_restarts


def autograd_at_one_alpha(
    tp: TorchProblem, iters: int, step: float, q: float, alpha_val: float,
) -> tuple[float, float, int]:
    """
    对单个 alpha 做一次前向+反向，返回 (||∇f||², d(||∇f||²)/dα, num_restarts).
    """
    alpha = torch.tensor(alpha_val, dtype=tp.A.dtype, requires_grad=True)
    result, rc = one_step_alpha(tp, iters, step, q, alpha)
    if result.requires_grad:
        result.backward()
        grad_val = alpha.grad.item() if alpha.grad is not None else None
    else:
        grad_val = None  # 无restart → 无梯度，显示None
    return result.item(), grad_val, rc


# =========================
# Coarse FD at one α
# =========================
def coarse_fd_at(problem: Problem, iters: int, step: float, q: float,
                 alpha: float, h: float) -> float:
    """中心差分: dJ/dα ≈ [J(α+h) − J(α−h)] / (2h)。用于和 autograd 对比。"""
    _, jp, _ = nesterov_restart_numpy(problem, iters, step, q, alpha + h)
    _, jm, _ = nesterov_restart_numpy(problem, iters, step, q, alpha - h)
    return (jp - jm) / (2.0 * h)


# =========================
# Plotting (6-panel + table)
# =========================
def plot_study(
    problem_name: str,
    # dense sweep
    alphas_np: np.ndarray,    # shape (N,): α 扫描网格
    gns_np: np.ndarray,       # shape (N,): 对应的 J(α) = ||∇f||²
    rcounts_np: np.ndarray,   # shape (N,): 每个 α 的重启次数
    fd_alphas: np.ndarray,    # shape (N-1,): FD 导数对应的 α 中点
    fd_deriv: np.ndarray,     # shape (N-1,): 密集扫描的 FD 导数
    # autograd at many points
    ag_alphas: np.ndarray,    # shape (M,): autograd 计算导数的 α 点
    ag_vals: np.ndarray,      # shape (M,): autograd 的 J(α) 值
    ag_derivs: np.ndarray,    # shape (M,): autograd dJ/dα（NaN=无重启）
    ag_rcounts: np.ndarray,   # shape (M,): autograd 的重启次数
    # comparison table rows: (α, AG, FD_h1, FD_h2, FD_h3, #restarts)
    table_rows: list,         # 长度 n_table 的列表，每行6个值
    save_prefix: str,         # 输出文件路径前缀（不含.png）
):
    """绘制 6-panel 对比图并保存为 PNG。

    面板:
      (a) J(α) 曲线 + autograd 切线箭头
      (b) 重启次数 vs α
      (c) dJ/dα 曲线: autograd vs 多种 FD
      (d) J(α) 上的 autograd 切线
      (e) 对比表格: AG vs FD，含符号一致性和相对误差
      (f) |AG − FD| 绝对/相对误差 vs α
    """
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.30)

    # ── (a) J(α) wide range + autograd points ──
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(alphas_np, gns_np, "-", color="#4477AA", linewidth=0.5, alpha=0.6,
            label=f"numpy sweep ({len(alphas_np)} pts)")
    ax.plot(ag_alphas, ag_vals, "r.", markersize=3, zorder=5,
            label=f"autograd ({len(ag_alphas)} pts)")
    # tangent arrows at 10 evenly-spaced autograd points
    n_arrow = min(12, len(ag_alphas))
    idx_show = np.linspace(0, len(ag_alphas)-1, n_arrow, dtype=int)
    for ii in idx_show:
        a0, j0, slope = ag_alphas[ii], ag_vals[ii], ag_derivs[ii]
        if np.isnan(slope):
            continue  # None grad → skip arrow
        da = 0.012 * (alphas_np[-1] - alphas_np[0])
        ax.annotate("", xy=(a0+da, j0+slope*da), xytext=(a0, j0),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$J(\alpha) = \|\nabla f(x_N)\|^2$")
    ax.set_title(r"(a) $J(\alpha)$ landscape + autograd tangent arrows")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.2)

    # ── (b) Restart count vs α ──
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(alphas_np, rcounts_np, "-", color="#44AA99", linewidth=0.5, alpha=0.6,
            label="numpy sweep")
    ax.plot(ag_alphas, ag_rcounts, "r.", markersize=3, zorder=5,
            label="autograd points")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Number of restarts")
    ax.set_title("(b) Restart count vs α")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.2)

    # ── (c) dJ/dα: autograd smooth curve + FD overlays ──
    ax = fig.add_subplot(gs[1, 0])
    valid_ag = ~np.isnan(ag_derivs)
    ax.plot(ag_alphas[valid_ag], ag_derivs[valid_ag], "r.-", markersize=3, lw=1.0,
            label="Autograd (hard)")
    ax.plot(fd_alphas, fd_deriv, "-", color="#88CCEE", lw=0.6, alpha=0.5,
            label="Dense FD (numpy sweep)")
    # coarse FD at autograd points with multiple h
    h_vals = [0.005, 0.01, 0.02]
    for h, ls, c in zip(h_vals, [":", "--", "-."],
                        ["#44AA99", "#332288", "#BBBBBB"]):
        fd_coarse = np.array([t[2 + h_vals.index(h)] for t in table_rows])
        ax.plot([t[0] for t in table_rows], fd_coarse, ls, color=c, lw=0.8,
                alpha=0.7, label=f"FD (h={h})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$dJ/d\alpha$")
    ax.set_title(r"(c) Derivative curve: Autograd vs FD")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, ls="--", alpha=0.2)

    # ── (d) Tangent lines on J(α) at a few points ──
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(alphas_np, gns_np, "-", color="#4477AA", linewidth=0.5, alpha=0.4)
    n_tangent = min(8, len(ag_alphas))
    idx_tan = np.linspace(0, len(ag_alphas)-1, n_tangent, dtype=int)
    cmap = plt.cm.Reds(np.linspace(0.3, 1.0, n_tangent))
    delta = 0.02 * (alphas_np[-1] - alphas_np[0])
    for j, ii in enumerate(idx_tan):
        a0, j0, slope = ag_alphas[ii], ag_vals[ii], ag_derivs[ii]
        if np.isnan(slope):
            ax.plot(a0, j0, "x", color=cmap[j], ms=6, zorder=5)  # None标记
            continue
        tx = np.array([a0 - delta, a0 + delta])
        ty = j0 + slope * (tx - a0)
        ax.plot(tx, ty, "-", color=cmap[j], lw=1.5)
        ax.plot(a0, j0, "o", color=cmap[j], ms=4, zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$J(\alpha)$")
    ax.set_title("(d) Autograd tangent lines on J(α)")
    ax.grid(True, which="both", ls="--", alpha=0.2)

    # ── (e) Comparison table ──
    ax = fig.add_subplot(gs[2, 0])
    ax.axis("off")
    col_labels = [r"$\alpha$", r"$\#$rst", r"AG $dJ/d\alpha$",
                  "FD (h=.005)", "FD (h=.01)", "FD (h=.02)",
                  "Sign", "Rel.Err (h=.01)"]
    cell_text = []
    n_sign_agree = 0
    n_nonzero = 0
    for row in table_rows:
        a_val, ag_d, fd1, fd2, fd3, nrs = row[0], row[1], row[2], row[3], row[4], row[5]
        is_none = np.isnan(ag_d)
        if is_none:
            sign_ok = "—"
            rel_str = "—"
            ag_str = "None"
        else:
            sign_ok = "Y" if (np.sign(ag_d) == np.sign(fd2) and abs(fd2) > 1e-20) else "N"
            if abs(fd2) > 1e-20:
                n_nonzero += 1
                if np.sign(ag_d) == np.sign(fd2):
                    n_sign_agree += 1
            if abs(ag_d) > 1e-30 and abs(fd2) > 1e-20:
                rel_err = abs(ag_d - fd2) / max(abs(ag_d), abs(fd2))
                rel_str = f"{rel_err:.2e}"
            else:
                rel_str = "—"
            ag_str = f"{ag_d:+.4e}"
        cell_text.append([
            f"{a_val:.4f}", f"{nrs}", ag_str,
            f"{fd1:+.4e}", f"{fd2:+.4e}", f"{fd3:+.4e}",
            sign_ok, rel_str,
        ])
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.05, 1.25)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9e2f3")
            cell.set_text_props(weight="bold")
        if col == 6 and row > 0:
            if cell_text[row-1][6] == "Y":
                cell.set_facecolor("#c6efce")
            elif cell_text[row-1][6] == "N":
                cell.set_facecolor("#ffc7ce")
    sign_pct = n_sign_agree / n_nonzero * 100 if n_nonzero > 0 else 0
    ax.set_title(f"(e) Autograd vs FD comparison (sign agree: "
                 f"{n_sign_agree}/{n_nonzero} = {sign_pct:.0f}%)",
                 pad=15, fontsize=11)

    # ── (f) Absolute error |AG - FD| vs α ──
    ax = fig.add_subplot(gs[2, 1])
    t_alphas = np.array([r[0] for r in table_rows])
    t_ag = np.array([r[1] for r in table_rows])
    t_fd_01 = np.array([r[3] for r in table_rows])
    abs_err = np.abs(t_ag - t_fd_01)
    # avoid log(0)
    mask = abs_err > 0
    if mask.any():
        ax.semilogy(t_alphas[mask], abs_err[mask], "o-", color="darkred",
                     ms=3, lw=0.8, label="|AG − FD(h=.01)|")
    # also plot relative error
    ax2 = ax.twinx()
    rel_err_arr = np.where(
        np.maximum(np.abs(t_ag), np.abs(t_fd_01)) > 1e-20,
        np.abs(t_ag - t_fd_01) / np.maximum(np.abs(t_ag), np.abs(t_fd_01)),
        np.nan,
    )
    valid = ~np.isnan(rel_err_arr) & (rel_err_arr > 0)
    if valid.any():
        ax2.semilogy(t_alphas[valid], rel_err_arr[valid], "s-", color="navy",
                      ms=2, lw=0.6, alpha=0.6, label="Relative error")
        ax2.set_ylabel("Relative error", color="navy")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("|AG − FD| (absolute)", color="darkred")
    ax.set_title("(f) Error: |Autograd − FD| vs α")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.2)

    fig.suptitle(problem_name, fontsize=14, y=1.0)
    fig.savefig(f"{save_prefix}.png")
    plt.close(fig)
    print(f"  Saved: {save_prefix}.png")


# =========================
# Run study for one problem
# =========================
def run_one_problem(
    problem: Problem,
    iters: int = 500,           # Nesterov 迭代次数
    alpha_lo: float = 0.01,     # α 扫描下界
    alpha_hi: float = 1.5,      # α 扫描上界
    n_alpha_dense: int = 5000,  # numpy 密集扫描点数
    n_alpha_ag: int = 80,       # autograd 求导点数
    n_table: int = 20,          # 对比表格行数
    save_prefix: str = "alpha_study",  # 输出路径前缀
):
    """对单个问题执行完整实验流程:
    1. numpy 密集扫描 J(α) (n_alpha_dense 个点)
    2. 密集扫描的 FD 导数
    3. autograd 在 n_alpha_ag 个 α 点求 dJ/dα
    4. 构建 AG vs FD 对比表（3种 FD 步长）
    5. 绘制 6-panel 图并保存
    """
    step = 1.0 / problem.L
    q = 0.0

    print(f"\n{'='*60}")
    print(f"Problem: {problem.name}")
    print(f"  shape=({problem.A.shape[0]},{problem.A.shape[1]}), L={problem.L:.4e}, "
          f"μ={problem.mu:.4e}, q*={problem.q_star:.6f}")
    print(f"  f*={problem.f_star:.6e}, iters={iters}, "
          f"α∈[{alpha_lo},{alpha_hi}]")

    # 1) Dense numpy sweep (wide range, fine grid)
    alpha_dense = np.linspace(alpha_lo, alpha_hi, n_alpha_dense)
    print(f"  [1/4] Numpy sweep ({n_alpha_dense} points)...")
    alphas_np, gns_np, rcounts_np = sweep_alpha_numpy(
        problem, iters, step, q, alpha_dense)

    # 2) FD derivative from dense sweep
    fd_alphas, fd_deriv = numerical_derivative(alphas_np, gns_np)

    # 3) Autograd at many α → smooth derivative curve
    tp = make_torch_problem(problem)
    margin = 0.02 * (alpha_hi - alpha_lo)
    ag_alphas = np.linspace(alpha_lo + margin, alpha_hi - margin, n_alpha_ag)
    ag_vals = np.empty(n_alpha_ag)
    ag_derivs = np.full(n_alpha_ag, np.nan)  # None → NaN
    ag_rcounts = np.empty(n_alpha_ag, dtype=int)
    print(f"  [2/4] Autograd at {n_alpha_ag} α points...")
    for i, a in enumerate(ag_alphas):
        v, d, rc = autograd_at_one_alpha(tp, iters, step, q, float(a))
        ag_vals[i] = v
        ag_derivs[i] = d if d is not None else np.nan
        ag_rcounts[i] = rc

    # 4) Build comparison table (n_table points, with 3 FD step sizes)
    print(f"  [3/4] Building comparison table ({n_table} points, 3 FD widths)...")
    table_idx = np.linspace(0, n_alpha_ag - 1, n_table, dtype=int)
    h_vals = [0.005, 0.01, 0.02]
    table_rows = []
    for ii in table_idx:
        a = float(ag_alphas[ii])
        ag_d = ag_derivs[ii]
        fds = [coarse_fd_at(problem, iters, step, q, a, h) for h in h_vals]
        table_rows.append((a, ag_d, fds[0], fds[1], fds[2], ag_rcounts[ii]))

    # Summary
    print(f"\n  {'α':>8s}  {'AG dJ/dα':>12s}  {'FD(h=.01)':>12s}  {'Sign':>5s}  {'#rst':>5s}")
    print("  " + "─" * 50)
    for row in table_rows:
        a_v, ag_d, _, fd2, _, nrs = row
        if np.isnan(ag_d):
            print(f"  {a_v:8.4f}  {'None':>12s}  {fd2:+12.4e}  {'—':>5s}  {nrs:5d}")
        else:
            sign = "Y" if (np.sign(ag_d) == np.sign(fd2) and abs(fd2) > 1e-20) else "N"
            print(f"  {a_v:8.4f}  {ag_d:+12.4e}  {fd2:+12.4e}  {sign:>5s}  {nrs:5d}")

    # 5) Plot
    print(f"  [4/4] Plotting 6-panel figure...")
    plot_study(
        problem.name, alphas_np, gns_np, rcounts_np, fd_alphas, fd_deriv,
        ag_alphas, ag_vals, ag_derivs, ag_rcounts, table_rows, save_prefix,
    )


# =========================
# Problem configurations
# =========================
def build_problem_suite() -> list[tuple[Problem, str]]:
    """构建 12 个问题的测试集:
    - 4种尺寸 (50×20 到 1000×300)
    - 4种条件数 (κ=3 到 200)
    - 2种稀疏度 (30%, 10%)
    - 2种相关度 (ρ=0.5, 0.9)
    返回: [(Problem, 文件名标签), ...]
    """
    problems = []

    # --- Different scales ---
    for m, n in [(50, 20), (200, 100), (500, 200), (1000, 300)]:
        p = make_least_squares(m=m, n=n, cond=12.0, seed=7)
        problems.append((p, f"ls_m{m}_n{n}_cond12"))

    # --- Different condition numbers ---
    for cond in [3.0, 12.0, 50.0, 200.0]:
        p = make_least_squares(m=200, n=100, cond=cond, seed=7,
                               name=f"LS(200×100, κ={cond:.0f})")
        problems.append((p, f"ls_cond{int(cond)}"))

    # --- Sparse design matrix ---
    for density in [0.3, 0.1]:
        p = make_sparse_least_squares(m=200, n=100, cond=12.0, density=density, seed=42)
        problems.append((p, f"sparse_d{int(density*100)}"))

    # --- Correlated design matrix ---
    for corr in [0.5, 0.9]:
        p = make_correlated_least_squares(m=200, n=100, cond=12.0, corr=corr, seed=13)
        problems.append((p, f"corr_rho{int(corr*100)}"))

    return problems


# =========================
# Main
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--n-dense", type=int, default=5000, help="numpy sweep density")
    parser.add_argument("--n-ag", type=int, default=80, help="autograd derivative points")
    parser.add_argument("--n-table", type=int, default=20, help="comparison table rows")
    parser.add_argument("--alpha-lo", type=float, default=0.01)
    parser.add_argument("--alpha-hi", type=float, default=1.5)
    parser.add_argument("--quick", action="store_true", help="只跑前 2 个小问题")
    args = parser.parse_args()

    suite = build_problem_suite()
    if args.quick:
        suite = suite[:2]

    out_dir = os.path.join(SCRIPT_DIR, "alpha_study_results")
    os.makedirs(out_dir, exist_ok=True)

    for problem, tag in suite:
        run_one_problem(
            problem,
            iters=args.iters,
            alpha_lo=args.alpha_lo,
            alpha_hi=args.alpha_hi,
            n_alpha_dense=args.n_dense,
            n_alpha_ag=args.n_ag,
            n_table=args.n_table,
            save_prefix=os.path.join(out_dir, tag),
        )

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
