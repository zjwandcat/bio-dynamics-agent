# BioDynamics Agent v4 - DDE 求解器封装
# 对应 PART C5 + 审计 §3.2 无 DDE 支持。
#
# 设计原则：
# 1. 数值 RHS 使用 method-of-steps 求解（转录延迟真实生效）
# 2. jitcdde 仅保留为可选依赖探针；通用 NumPy RHS 不能直接符号化
# 3. 接口与 scipy.solve_ivp 对齐，便于 v4 模板无缝替换
# 4. 不修改 sandbox.py（不可碰清单）

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# jitcdde try-import
# =============================================================================
try:
    from jitcdde import jitcdde, y as jitcdde_y, t as jitcdde_t
    _JITCDDE_AVAILABLE = True
    _JITCDDE_VERSION = getattr(__import__("jitcdde"), "__version__", "unknown")
except ImportError:
    _JITCDDE_AVAILABLE = False
    _JITCDDE_VERSION = None
    warnings.warn(
        "jitcdde 不可用，DDE 将降级为 ODE 求解（延迟项近似为 y(t)）。"
        "如需真实 DDE 求解，请安装：pip install jitcdde",
        RuntimeWarning,
        stacklevel=2,
    )


def is_dde_available() -> bool:
    """Return whether the supported numerical DDE backend is available."""
    try:
        import scipy.integrate  # noqa: F401
        return True
    except ImportError:
        return False


def solve_dde(
    rhs: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: np.ndarray | list[float],
    delay: float,
    t_eval: np.ndarray | None = None,
    history: Callable[[float], np.ndarray] | None = None,
    max_step: float = 0.5,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """求解 DDE 系统。

    方程形式：dy/dt = f(t, y(t), y(t-τ))

    Args:
        rhs: 右端函数 f(t, y, y_delayed)，y_delayed 为 y(t-τ)
        t_span: 时间区间 (t_start, t_end)
        y0: 初始状态 y(0)
        delay: 延迟 τ（与 t_span 同单位）
        t_eval: 输出时间点
        history: 历史函数 y(t) for t < 0，默认常数 y0
        max_step: 最大步长
        rtol: 相对容差
        atol: 绝对容差

    Returns:
        dict: {
            "t": 时间序列,
            "y": 状态序列 (N_time x N_species),
            "dde_used": 是否真实使用 DDE,
            "delay": 实际使用的延迟（DDE=delay, ODE=0）,
            "solver": 使用的求解器名,
        }
    """
    y0_arr = np.asarray(y0, dtype=float)
    t_start, t_end = t_span

    if delay > 0:
        # Generic agent templates provide numerical NumPy RHS functions.  They
        # cannot be converted safely to jitcdde symbolic expressions, so use a
        # standard method-of-steps integrator with dense history interpolation.
        try:
            return _solve_dde_method_of_steps(
                rhs, t_span, y0_arr, delay, t_eval, history,
                max_step, rtol, atol,
            )
        except Exception as e:
            # Preserve operational behavior, but make the scientific downgrade
            # explicit in both logs and result metadata.
            logger.error(
                "method-of-steps DDE 求解失败 (%s)，降级为 ODE（延迟失效）", e
            )
            return _solve_ode_fallback(rhs, t_span, y0_arr, t_eval,
                                        max_step, rtol, atol, delay)
    else:
        # No delay requested: the DDE contract reduces exactly to an ODE.
        return _solve_ode_fallback(rhs, t_span, y0_arr, t_eval,
                                    max_step, rtol, atol, 0.0)


def _solve_dde_method_of_steps(
    rhs: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: np.ndarray,
    delay: float,
    t_eval: np.ndarray | None,
    history: Callable[[float], np.ndarray] | None,
    max_step: float,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Solve a constant-delay DDE with the method of steps.

    Each integration segment is no longer than ``delay``. Therefore every
    delayed lookup in the current segment lies in the supplied history or a
    previously completed dense-output segment.
    """
    from scipy.integrate import solve_ivp

    if not np.isfinite(delay) or delay <= 0:
        raise ValueError(f"delay must be positive and finite, got {delay!r}")

    t_start, t_end = map(float, t_span)
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError(f"invalid t_span={t_span!r}")

    if t_eval is None:
        eval_points = np.linspace(t_start, t_end, 200)
    else:
        eval_points = np.asarray(t_eval, dtype=float)
        if eval_points.ndim != 1 or eval_points.size == 0:
            raise ValueError("t_eval must be a non-empty one-dimensional array")
        if np.any(np.diff(eval_points) < 0):
            raise ValueError("t_eval must be sorted")
        if eval_points[0] < t_start or eval_points[-1] > t_end:
            raise ValueError("t_eval points must lie within t_span")

    history_fn = history or (lambda _t: y0)
    completed: list[tuple[float, float, Any]] = []

    def past_value(query_time: float) -> np.ndarray:
        if query_time <= t_start + 1e-12:
            value = np.asarray(history_fn(query_time), dtype=float)
            if value.shape != y0.shape:
                raise ValueError("history function returned an incompatible shape")
            return value
        for seg_start, seg_end, dense in reversed(completed):
            if seg_start - 1e-10 <= query_time <= seg_end + 1e-10:
                return np.asarray(dense(query_time), dtype=float)
        raise RuntimeError(f"delayed state unavailable at t={query_time:.9g}")

    current_t = t_start
    current_y = np.asarray(y0, dtype=float).copy()
    while current_t < t_end - 1e-12:
        segment_end = min(current_t + delay, t_end)

        def segment_rhs(t: float, y: np.ndarray) -> np.ndarray:
            delayed = past_value(t - delay)
            value = np.asarray(rhs(t, y, delayed), dtype=float)
            if value.shape != current_y.shape:
                raise ValueError("rhs returned an incompatible shape")
            return value

        sol = solve_ivp(
            segment_rhs,
            (current_t, segment_end),
            current_y,
            method="LSODA",
            dense_output=True,
            max_step=min(max_step, delay),
            rtol=rtol,
            atol=atol,
        )
        if not sol.success or sol.sol is None:
            raise RuntimeError(sol.message or "DDE segment integration failed")
        completed.append((current_t, segment_end, sol.sol))
        current_t = segment_end
        current_y = np.asarray(sol.y[:, -1], dtype=float)

    values: list[np.ndarray] = []
    for point in eval_points:
        if point <= t_start + 1e-12:
            values.append(np.asarray(y0, dtype=float).copy())
            continue
        values.append(past_value(float(point)))

    return {
        "t": eval_points,
        "y": np.vstack(values),
        "dde_used": True,
        "delay": float(delay),
        "solver": f"scipy.solve_ivp method-of-steps (LSODA, delay={delay:.6g})",
        "backend": "method_of_steps",
        "jitcdde_available": _JITCDDE_AVAILABLE,
    }


def _solve_dde_jitcdde(
    rhs: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: np.ndarray,
    delay: float,
    t_eval: np.ndarray | None,
    max_step: float,
    rtol: float,
    atol: float,
    history: Callable[[float], np.ndarray] | None,
) -> dict[str, Any]:
    """IB-006 修复：使用 jitcdde 执行真实 DDE 求解。

    jitcdde 使用 lambda-based API，将 rhs 包装为符号化形式。
    历史函数默认为常数 y0。
    """
    from jitcdde import jitcdde, y as jitcdde_y, t as jitcdde_t

    n = len(y0)
    t_start, t_end = t_span

    # 构建 jitcdde 的 lambda f 函数列表
    # jitcdde 接受 lambda 函数列表，每个 lambda 返回第 i 个 dy/dt
    # 使用 y(jitcdde_t - delay) 获取延迟状态
    def make_f_component(i: int):
        def f_i():
            # 当前状态
            y_current = np.array([jitcdde_y(j) for j in range(n)])
            # 延迟状态
            y_delayed = np.array([jitcdde_y(j, jitcdde_t - delay) for j in range(n)])
            # 调用 rhs
            dydt = rhs(float(jitcdde_t), y_current, y_delayed)
            return dydt[i]
        return f_i

    f_list = [make_f_component(i) for i in range(n)]

    dde = jitcdde(f_list, max_delay=delay)

    # 设置历史函数
    if history is not None:
        dde.past_from_function(history)
    else:
        # 常数历史：y(t) = y0 for t < 0
        dde.constant_past(y0.tolist())

    # 积分到 t_end
    dde.step_on_discontinuities()

    # 在 t_eval 点采样
    if t_eval is None:
        t_eval = np.linspace(t_start, t_end, 200)

    t_out = []
    y_out = []
    for t_val in t_eval:
        if t_val <= t_start:
            t_out.append(t_val)
            y_out.append(y0.copy())
        else:
            state = dde.integrate(t_val)
            t_out.append(t_val)
            y_out.append(np.array(state))

    t_arr = np.array(t_out)
    y_arr = np.array(y_out)

    return {
        "t": t_arr,
        "y": y_arr,
        "dde_used": True,
        "delay": delay,
        "solver": f"jitcdde (real DDE, delay={delay:.2f})",
    }


def _solve_ode_fallback(
    rhs: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: np.ndarray,
    t_eval: np.ndarray | None,
    max_step: float,
    rtol: float,
    atol: float,
    delay: float,
) -> dict[str, Any]:
    """ODE 降级求解（将 DDE rhs 包装为 ODE rhs）。"""
    from scipy.integrate import solve_ivp

    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        """ODE rhs：延迟项近似为 y(t)（即 y(t-τ) ≈ y(t)）。"""
        return rhs(t, y, y)

    sol = solve_ivp(
        ode_rhs, t_span, y0, t_eval=t_eval,
        method="LSODA", max_step=max_step, rtol=rtol, atol=atol,
    )

    return {
        "t": sol.t,
        "y": sol.y.T,
        "dde_used": False,
        "delay": 0.0,  # ODE 降级，延迟失效
        "solver": "scipy.solve_ivp (LSODA, DDE downgraded to ODE)",
        "original_delay": delay,
    }


def solve_dde_simple(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: np.ndarray | list[float],
    t_eval: np.ndarray | None = None,
    max_step: float = 0.5,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """简单 ODE 求解（无延迟，用于不需要 DDE 的通路）。

    接口与 solve_dde 对齐，便于统一调用。
    """
    from scipy.integrate import solve_ivp

    y0_arr = np.asarray(y0, dtype=float)
    sol = solve_ivp(
        rhs, t_span, y0_arr, t_eval=t_eval,
        method="LSODA", max_step=max_step, rtol=rtol, atol=atol,
    )

    return {
        "t": sol.t,
        "y": sol.y.T,
        "dde_used": False,
        "delay": 0.0,
        "solver": "scipy.solve_ivp (LSODA)",
    }


__all__ = [
    "is_dde_available",
    "solve_dde",
    "solve_dde_simple",
]
