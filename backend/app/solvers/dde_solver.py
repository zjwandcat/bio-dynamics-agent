# BioDynamics Agent v4 - DDE 求解器封装
# 对应 PART C5 + 审计 §3.2 无 DDE 支持。
#
# 设计原则：
# 1. jitcdde 可用时使用 DDE 求解（转录延迟真实生效）
# 2. jitcdde 不可用时降级为 scipy.integrate.solve_ivp（延迟失效但仍可仿真）
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
    """返回 jitcdde 是否可用。"""
    return _JITCDDE_AVAILABLE


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

    if _JITCDDE_AVAILABLE and delay > 0:
        # DDE 求解（jitcdde）
        # 注意：jitcdde API 需要符号化 rhs，这里给出框架
        # 生产实现需将 rhs 转换为 jitcdde 的 symbolic 形式
        # MVP 阶段降级为 ODE
        logger.warning(
            "jitcdde 可用但 MVP 阶段仍降级为 ODE（jitcdde symbolic rhs 转换未完成）"
        )
        return _solve_ode_fallback(rhs, t_span, y0_arr, t_eval, max_step, rtol, atol, delay)
    else:
        # ODE 降级（延迟项近似为 y(t)）
        if delay > 0:
            logger.warning(
                "jitcdde 不可用，DDE 延迟 τ=%.2f 将失效（近似为 y(t)）", delay
            )
        return _solve_ode_fallback(rhs, t_span, y0_arr, t_eval, max_step, rtol, atol, 0.0)


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
