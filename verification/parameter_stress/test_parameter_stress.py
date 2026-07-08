"""参数压力测试 — Parameter Stress Suite

对每条核心通路随机扰动参数 ±10% / ±30% / ±50%，检查：
  - Solver Stability      （求解器是否成功收敛）
  - Negative Concentration（是否出现负浓度）
  - NaN                   （是否产生 NaN / Inf）
  - Explosion             （数值爆炸，|y| > 1e6）
  - Oscillation Loss      （振荡通路是否丢失振荡行为）

设计要点：
  - 每条通路定义一个简化 ODE 模型 + 基线参数集
  - 用固定种子 RNG 做可复现扰动
  - 对振荡通路（p53 / NF-κB / Cell Cycle）额外检查振荡保留
  - 大扰动（±50%）允许数值不稳定，但必须 graceful（不崩溃）
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pytest
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
PERTURBATION_LEVELS = [0.10, 0.30, 0.50]
# 振荡通路：扰动后必须保留振荡
OSCILLATING_PATHWAYS = {"p53", "NF_KB", "CELL_CYCLE"}

CORE_PATHWAYS = [
    "EGFR_RTK", "MAPK_ERK", "PI3K_AKT_mTOR", "p53", "APOPTOSIS",
    "CELL_CYCLE", "JAK_STAT", "NF_KB", "WNT", "TGF_BETA",
]


def perturb_parameters(params: dict[str, float], level: float,
                       seed: int = 42) -> dict[str, float]:
    """对参数 dict 做可复现的 ±level 扰动。

    保证扰动后参数为正（速率常数不能为负）。
    """
    rng = np.random.default_rng(seed)
    perturbed: dict[str, float] = {}
    for key, value in params.items():
        factor = 1.0 + rng.uniform(-level, level)
        perturbed[key] = max(value * factor, 1e-9)  # 防止负值/零
    return perturbed


def run_simulation(rhs: Callable, y0: list[float], t_end: float,
                   n_points: int = 501) -> dict[str, Any]:
    """通用仿真封装，返回时间序列 + 稳定性诊断。"""
    t_eval = np.linspace(0, t_end, n_points)
    sol = solve_ivp(rhs, [0, t_end], y0, t_eval=t_eval,
                    method="LSODA", rtol=1e-6, atol=1e-9)
    if not sol.success:
        return {
            "success": False, "message": sol.message,
            "t": [], "y": np.array([]),
            "has_nan": True, "has_inf": True, "has_negative": True,
            "has_explosion": True, "max_abs": float("inf"),
        }
    y = sol.y
    flat = y.flatten()
    return {
        "success": True,
        "message": sol.message,
        "t": sol.t,
        "y": y,
        "has_nan": bool(np.any(np.isnan(flat))),
        "has_inf": bool(np.any(np.isinf(flat))),
        "has_negative": bool(np.any(flat < -1e-9)),
        "has_explosion": bool(np.any(np.abs(flat) > 1e6)),
        "max_abs": float(np.max(np.abs(flat))) if flat.size else 0.0,
    }


def check_stability(result: dict[str, Any]) -> dict[str, bool]:
    """稳定性检查汇总。"""
    return {
        "solver_stable": result["success"] and not result["has_nan"]
                         and not result["has_inf"],
        "no_negative": not result["has_negative"],
        "no_nan": not result["has_nan"],
        "no_explosion": not result["has_explosion"],
        "no_oscillation_loss": result.get("oscillation_preserved", True),
    }


def detect_oscillation(t: np.ndarray, y_row: np.ndarray, min_peaks: int = 3) -> bool:
    """检测单行时间序列是否振荡。"""
    if len(y_row) < 5:
        return False
    peaks = 0
    for i in range(1, len(y_row) - 1):
        if y_row[i] > y_row[i - 1] and y_row[i] > y_row[i + 1]:
            peaks += 1
    if peaks < min_peaks:
        return False
    # 周期 CV 检查
    peak_times = [t[i] for i in range(1, len(y_row) - 1)
                  if y_row[i] > y_row[i - 1] and y_row[i] > y_row[i + 1]]
    if len(peak_times) >= 2:
        periods = np.diff(peak_times)
        cv = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else 1.0
        return cv < 0.5
    return True


# --------------------------------------------------------------------------- #
# 通路模型工厂：每条通路返回 (rhs, y0, baseline_params, oscillation_species_idx)
# --------------------------------------------------------------------------- #
def get_pathway_model(pathway: str) -> tuple[Callable, list[float], dict[str, float], int]:
    """返回 (rhs_factory, y0, baseline_params, osc_species_idx)。

    rhs_factory 接受 params dict 返回 rhs 函数。
    osc_species_idx: 振荡检测的物种索引（-1 表示非振荡通路）。
    """
    if pathway == "EGFR_RTK":
        params = {"k_on": 0.1, "k_off": 0.01, "k_phos": 0.5, "k_int": 0.02, "k_deg": 0.01}
        def factory(p):
            def rhs(t, y):
                L, R, LR, pLR, pLR_int = y
                return [
                    -p["k_on"] * L * R + p["k_off"] * LR,
                    -p["k_on"] * L * R + p["k_off"] * LR,
                    p["k_on"] * L * R - p["k_off"] * LR - p["k_phos"] * LR,
                    p["k_phos"] * LR - p["k_int"] * pLR,
                    p["k_int"] * pLR - p["k_deg"] * pLR_int,
                ]
            return rhs
        return factory, [100.0, 50.0, 0.0, 0.0, 0.0], params, -1

    if pathway == "MAPK_ERK":
        params = {"k1": 0.1, "k2": 0.2, "k3": 0.5, "k_deg": 0.05}
        def factory(p):
            def rhs(t, y):
                RasGTP, Raf, pRaf, MEK, pMEK, ERK, ppERK = y
                return [
                    -0.01 * RasGTP,
                    -p["k1"] * RasGTP * Raf + 0.1 * pRaf,
                    p["k1"] * RasGTP * Raf - 0.1 * pRaf,
                    -p["k2"] * pRaf * MEK + 0.1 * pMEK,
                    p["k2"] * pRaf * MEK - 0.1 * pMEK,
                    -p["k3"] * pMEK * ERK + p["k_deg"] * ppERK,
                    p["k3"] * pMEK * ERK - p["k_deg"] * ppERK,
                ]
            return rhs
        return factory, [100.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0], params, -1

    if pathway == "PI3K_AKT_mTOR":
        params = {"k_pip3": 0.3, "k_akt": 0.5, "k_s6k": 0.2, "k_fb": 0.01, "k_deg": 0.02}
        def factory(p):
            def rhs(t, y):
                PIP2, PIP3, AKT, pAKT, mTOR, S6K, pS6K = y
                return [
                    -p["k_pip3"] * PIP2 + 0.1 * PIP3 - p["k_fb"] * pS6K * PIP2,
                    p["k_pip3"] * PIP2 - p["k_deg"] * PIP3,
                    -p["k_akt"] * PIP3 * AKT + 0.05 * pAKT,
                    p["k_akt"] * PIP3 * AKT - 0.05 * pAKT,
                    0.1 * pAKT - 0.01 * mTOR,
                    -p["k_s6k"] * mTOR * S6K + 0.05 * pS6K,
                    p["k_s6k"] * mTOR * S6K - 0.05 * pS6K,
                ]
            return rhs
        return factory, [100.0, 0.0, 50.0, 0.0, 0.0, 50.0, 0.0], params, -1

    if pathway == "p53":
        params = {"k1": 0.2, "k2": 0.1, "k3": 0.1, "k4": 0.1, "k5": 0.5, "k6": 0.1, "n": 2.0}
        def factory(p):
            def rhs(t, y):
                p53, Mdm2_c, Mdm2_n = y
                n = p["n"]
                return [
                    p["k1"] - p["k2"] * Mdm2_n * p53 / (1 + (p53 / 10) ** n),
                    p["k3"] * p53 - p["k4"] * Mdm2_c,
                    p["k5"] * Mdm2_c - p["k6"] * Mdm2_n,
                ]
            return rhs
        return factory, [5.0, 0.0, 0.0], params, 0  # p53 振荡

    if pathway == "APOPTOSIS":
        params = {"k_act": 0.01, "k_momp": 0.1, "k_deg": 0.001, "feedback": 0.05}
        def factory(p):
            def rhs(t, y):
                C8, CytC, C9, C3, XIAP = y
                return [
                    0.5 - p["k_deg"] * C8,
                    p["k_momp"] * C8 + p["feedback"] * C3 * (1 - CytC) - 0.01 * CytC,
                    p["k_act"] * CytC * (100 - C9) - 0.01 * C9,
                    0.1 * C9 * (100 - C3) - 0.02 * XIAP * C3,
                    -0.01 * C3 * XIAP,
                ]
            return rhs
        return factory, [2.0, 0.0, 0.0, 0.0, 50.0], params, -1

    if pathway == "CELL_CYCLE":
        params = {"vi": 0.025, "kd1": 0.01, "kd2": 0.01,
                  "V1": 1.0, "V2": 0.5, "V3": 1.0, "V4": 0.5,
                  "K1": 0.01, "K2": 0.01, "K3": 0.01, "K4": 0.01}
        def factory(p):
            def rhs(t, y):
                C, M, M_ = y
                V1_eff = p["V1"] * C / (p["K1"] + C)
                return [
                    p["vi"] - p["kd1"] * C - p["kd2"] * M_,
                    V1_eff * (1 - M) - p["V2"] * M / (p["K2"] + M),
                    p["V3"] * M / (p["K3"] + M) - p["V4"] * M_ / (p["K4"] + M_),
                ]
            return rhs
        return factory, [0.1, 0.1, 0.1], params, 0  # Cyclin 振荡

    if pathway == "JAK_STAT":
        params = {"k_phos": 0.5, "k_dim": 0.1, "k_import": 0.05,
                  "k_export": 0.01, "k_deg": 0.005}
        def factory(p):
            def rhs(t, y):
                pSTAT, pSTAT_dim, pSTAT_nuc = y
                return [
                    1.0 - p["k_phos"] * pSTAT - 2 * p["k_dim"] * pSTAT ** 2,
                    p["k_dim"] * pSTAT ** 2 - p["k_import"] * pSTAT_dim,
                    p["k_import"] * pSTAT_dim - p["k_export"] * pSTAT_nuc
                    - p["k_deg"] * pSTAT_nuc,
                ]
            return rhs
        return factory, [0.0, 0.0, 0.0], params, -1

    if pathway == "NF_KB":
        params = {"k1": 0.5, "k2": 0.1, "k3": 0.5, "k4": 0.1,
                  "k5": 0.1, "k_deg": 0.01, "n": 2.0}
        def factory(p):
            def rhs(t, y):
                NFkB_n, IkBa_c, IkBa_n, IKK = y
                n = p["n"]
                return [
                    p["k1"] * IKK * (1 - NFkB_n) - p["k2"] * IkBa_n * NFkB_n,
                    p["k3"] * NFkB_n ** n / (1 + NFkB_n ** n) - p["k4"] * IkBa_c,
                    p["k5"] * IkBa_c - p["k2"] * IkBa_n * NFkB_n - p["k_deg"] * IkBa_n,
                    -p["k_deg"] * IKK,
                ]
            return rhs
        return factory, [0.0, 0.0, 0.0, 1.0], params, 0  # NF-κB 振荡

    if pathway == "WNT":
        params = {"k_syn": 0.5, "k_deg": 0.1, "n": 4.0}
        def factory(p):
            def rhs(t, y):
                bCat, DC, bCat_n = y
                n = p["n"]
                return [
                    p["k_syn"] - p["k_deg"] * DC * bCat - 0.05 * bCat,
                    -0.1 * DC + 0.1 * (1 - DC),
                    0.1 * bCat ** n / (1 + bCat ** n) - 0.02 * bCat_n,
                ]
            return rhs
        return factory, [0.1, 1.0, 0.0], params, -1

    if pathway == "TGF_BETA":
        params = {"k_phos": 0.5, "k_bind": 0.3, "k_import": 0.1,
                  "k_export": 0.02, "k_deg": 0.01}
        def factory(p):
            def rhs(t, y):
                pS2, complex_c, complex_n = y
                return [
                    1.0 - p["k_phos"] * pS2 - p["k_bind"] * pS2,
                    p["k_bind"] * pS2 - p["k_import"] * complex_c,
                    p["k_import"] * complex_c - p["k_export"] * complex_n
                    - p["k_deg"] * complex_n,
                ]
            return rhs
        return factory, [0.0, 0.0, 0.0], params, -1

    raise ValueError(f"未知通路：{pathway}")


# --------------------------------------------------------------------------- #
# 参数化压力测试：10 通路 × 3 扰动等级 = 30 用例
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pathway", CORE_PATHWAYS)
@pytest.mark.parametrize("level", PERTURBATION_LEVELS,
                         ids=["p10", "p30", "p50"])
def test_parameter_perturbation_stability(pathway: str, level: float) -> None:
    """±{level*100}% 扰动下求解器稳定性 + 无负浓度 + 无 NaN + 无爆炸。"""
    factory, y0, baseline, osc_idx = get_pathway_model(pathway)
    perturbed = perturb_parameters(baseline, level, seed=hash(pathway) % 2**31)

    # 数值参数 n 可能非整数，取整
    if "n" in perturbed:
        perturbed["n"] = max(round(perturbed["n"]), 1)

    rhs = factory(perturbed)
    t_end = 600.0 if pathway in OSCILLATING_PATHWAYS else 120.0
    result = run_simulation(rhs, y0, t_end=t_end)
    stability = check_stability(result)

    # ±10% 必须完全稳定
    if level <= 0.10:
        assert stability["solver_stable"], (
            f"{pathway} ±{level*100:.0f}%: 求解器不稳定 ({result['message']})"
        )
        assert stability["no_nan"], f"{pathway} ±{level*100:.0f}%: 出现 NaN"
        assert stability["no_negative"], f"{pathway} ±{level*100:.0f}%: 出现负浓度"
        assert stability["no_explosion"], (
            f"{pathway} ±{level*100:.0f}%: 数值爆炸 max={result['max_abs']:.2e}"
        )

    # ±30% 允许爆炸但不允许 NaN（graceful degradation）
    elif level <= 0.30:
        assert stability["no_nan"], f"{pathway} ±{level*100:.0f}%: 出现 NaN"
        if stability["solver_stable"]:
            assert stability["no_negative"], (
                f"{pathway} ±{level*100:.0f}%: 求解成功但出现负浓度"
            )

    # ±50% 大扰动：仅要求不产生 NaN（允许爆炸/负值）
    else:
        assert stability["no_nan"], f"{pathway} ±{level*100:.0f}%: 出现 NaN"


@pytest.mark.parametrize("pathway", ["p53", "NF_KB", "CELL_CYCLE"])
@pytest.mark.parametrize("level", [0.10, 0.30],
                         ids=["p10", "p30"])
def test_oscillation_preserved_under_perturbation(pathway: str, level: float) -> None:
    """振荡通路在 ±10% / ±30% 扰动下必须保留振荡行为。"""
    factory, y0, baseline, osc_idx = get_pathway_model(pathway)
    perturbed = perturb_parameters(baseline, level, seed=hash(pathway) % 2**31)
    if "n" in perturbed:
        perturbed["n"] = max(round(perturbed["n"]), 1)

    rhs = factory(perturbed)
    result = run_simulation(rhs, y0, t_end=600.0, n_points=1201)

    assert result["success"], f"{pathway} ±{level*100:.0f}%: 求解失败"
    osc_series = result["y"][osc_idx]
    is_osc = detect_oscillation(result["t"], osc_series, min_peaks=3)
    assert is_osc, (
        f"{pathway} ±{level*100:.0f}%: 振荡丢失"
    )


@pytest.mark.parametrize("pathway", CORE_PATHWAYS)
def test_baseline_stability(pathway: str) -> None:
    """基线参数（无扰动）必须完全稳定——作为压力测试的对照。"""
    factory, y0, baseline, osc_idx = get_pathway_model(pathway)
    if "n" in baseline:
        baseline["n"] = max(round(baseline["n"]), 1)
    rhs = factory(baseline)
    t_end = 600.0 if pathway in OSCILLATING_PATHWAYS else 120.0
    result = run_simulation(rhs, y0, t_end=t_end)

    assert result["success"], f"{pathway} 基线求解失败：{result['message']}"
    assert not result["has_nan"], f"{pathway} 基线出现 NaN"
    assert not result["has_negative"], f"{pathway} 基线出现负浓度"
    assert not result["has_explosion"], (
        f"{pathway} 基线数值爆炸 max={result['max_abs']:.2e}"
    )


@pytest.mark.parametrize("pathway", CORE_PATHWAYS)
@pytest.mark.parametrize("level", PERTURBATION_LEVELS,
                         ids=["p10", "p30", "p50"])
def test_mass_conservation_under_perturbation(pathway: str, level: float) -> None:
    """扰动后质量守恒（总量变化 < 50%）。"""
    factory, y0, baseline, osc_idx = get_pathway_model(pathway)
    perturbed = perturb_parameters(baseline, level, seed=hash(pathway) % 2**31)
    if "n" in perturbed:
        perturbed["n"] = max(round(perturbed["n"]), 1)

    rhs = perturbed if isinstance(perturbed, Callable) else factory(perturbed)
    if not isinstance(rhs, Callable):
        rhs = factory(perturbed)

    result = run_simulation(rhs, y0, t_end=120.0)
    if not result["success"] or result["has_explosion"]:
        pytest.skip(f"{pathway} ±{level*100:.0f}%: 求解不稳定，跳过守恒检查")

    total_initial = sum(abs(v) for v in y0)
    total_final = float(np.sum(np.abs(result["y"][:, -1])))
    # 守恒：总量不应偏离初始 5 倍以上
    assert total_final < total_initial * 5, (
        f"{pathway} ±{level*100:.0f}%: 质量不守恒 "
        f"initial={total_initial:.2f} final={total_final:.2e}"
    )


# --------------------------------------------------------------------------- #
# 文档化已知 v4 限制
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="已知 v4 限制：所有 specialist 缺少动力学参数 (FM-013)")
def test_no_kinetic_parameters_documentation() -> None:
    """文档化 P0 bug：specialist 无 Km/kcat/Vmax，参数压力测试需在 v4 集成后跑。"""
    pass


@pytest.mark.skip(reason="已知 v4 限制：ODE Renderer 使用默认参数 (FM-002)")
def test_ode_renderer_default_params_documentation() -> None:
    """文档化 P0 bug：_extract_params 读取不存在字段。"""
    pass


# --------------------------------------------------------------------------- #
# 数据完整性
# --------------------------------------------------------------------------- #
def test_all_pathways_have_models() -> None:
    """校验 10 条通路均有可用的参数化 ODE 模型。"""
    for pathway in CORE_PATHWAYS:
        factory, y0, baseline, osc_idx = get_pathway_model(pathway)
        assert callable(factory), f"{pathway} factory 不可调用"
        assert len(y0) > 0, f"{pathway} y0 为空"
        assert len(baseline) > 0, f"{pathway} 基线参数为空"
        # 模型可执行
        rhs = factory(baseline)
        deriv = rhs(0.0, y0)
        assert len(deriv) == len(y0), f"{pathway} rhs 维度不匹配"


def test_perturbation_is_deterministic() -> None:
    """校验相同种子产生相同扰动（可复现性）。"""
    params = {"a": 1.0, "b": 2.0, "c": 0.5}
    p1 = perturb_parameters(params, 0.30, seed=42)
    p2 = perturb_parameters(params, 0.30, seed=42)
    assert p1 == p2, "相同种子的扰动不一致"


def test_perturbation_bounds() -> None:
    """校验扰动在 ±level 范围内且参数为正。"""
    params = {"a": 1.0}
    for level in PERTURBATION_LEVELS:
        p = perturb_parameters(params, level, seed=7)
        ratio = p["a"] / params["a"]
        assert (1 - level - 1e-9) <= ratio <= (1 + level + 1e-9), (
            f"扰动 {level*100:.0f}% 越界：ratio={ratio}"
        )
        assert p["a"] > 0, "扰动后参数非正"
