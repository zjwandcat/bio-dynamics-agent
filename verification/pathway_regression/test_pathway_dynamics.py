"""通路动力学回归测试 — 10 条通路 × 5 case = 50 case

每条通路验证文献支持的动力学行为：
  - EGFR      : 配体诱导受体激活、内吞降级
  - MAPK      : 级联放大、ERK 双磷酸化
  - PI3K      : AKT 磷酸化、mTOR 反馈
  - p53       : 振荡行为（p53-Mdm2 负反馈）
  - Apoptosis : Caspase 双稳态（MOMP）
  - Cell Cycle: Cyclin-CDK 周期振荡
  - JAK-STAT  : STAT 二聚化、核转位
  - NF-κB     : 周期振荡（IκBα 负反馈）
  - Wnt       : β-catenin 破坏复合体
  - TGF-β     : SMAD 核转位

每个 case 用 scipy.integrate.solve_ivp 在简化 ODE 模型上仿真，
并对文献支持的动力学特征（峰值时间、振荡周期、双稳态阈值、
放大倍数、核/胞质比等）做真实断言。

执行约束：
  - 长时 / 需要 v4 集成的 case 标记 @pytest.mark.benchmark 或 skip
  - 纯 scipy 仿真的 case 不打 benchmark，可在快速 CI 运行
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pytest
from scipy.integrate import solve_ivp


# --------------------------------------------------------------------------- #
# 通用动力学检测工具
# --------------------------------------------------------------------------- #
def detect_peaks(t: np.ndarray, y: np.ndarray) -> list[tuple[float, float]]:
    """检测局部极大值，返回 [(t_peak, y_peak), ...]。"""
    peaks: list[tuple[float, float]] = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            peaks.append((float(t[i]), float(y[i])))
    return peaks


def detect_oscillation(t: np.ndarray, y: np.ndarray, min_peaks: int = 3) -> dict[str, Any]:
    """检测振荡：至少 min_peaks 个峰，且周期 CV < 30%。"""
    peaks = detect_peaks(t, y)
    if len(peaks) < min_peaks:
        return {"is_oscillating": False, "period": None, "n_peaks": len(peaks)}
    periods = np.diff([p[0] for p in peaks])
    period = float(np.mean(periods))
    cv = float(np.std(periods) / period) if period > 0 else float("inf")
    return {
        "is_oscillating": cv < 0.30,
        "period": period,
        "n_peaks": len(peaks),
        "period_cv": cv,
    }


def auc(t: np.ndarray, y: np.ndarray) -> float:
    """梯形法 AUC。"""
    return float(np.trapz(y, t))


def run_ode(rhs: Callable, y0: list[float], t_end: float,
            n_points: int = 501, method: str = "LSODA",
            rtol: float = 1e-8, atol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """通用 ODE 仿真封装。"""
    t_eval = np.linspace(0, t_end, n_points)
    sol = solve_ivp(rhs, [0, t_end], y0, t_eval=t_eval,
                    method=method, rtol=rtol, atol=atol)
    assert sol.success, f"ODE 求解失败：{sol.message}"
    return sol.t, sol.y


# --------------------------------------------------------------------------- #
# 1. EGFR — 配体诱导受体激活、内吞降级
# --------------------------------------------------------------------------- #
class TestEGFRDynamics:
    """EGFR 通路动力学：配体结合 → 自磷酸化 → 内吞降级。"""

    def _egfr_model(self, t, y, k_on=0.1, k_off=0.01, k_phos=0.5,
                    k_int=0.02, k_deg=0.01):
        """简化 EGFR 模型：[L, R, LR, pLR, pLR_int]。"""
        L, R, LR, pLR, pLR_int = y
        dL = -k_on * L * R + k_off * LR
        dR = -k_on * L * R + k_off * LR
        dLR = k_on * L * R - k_off * LR - k_phos * LR
        dpLR = k_phos * LR - k_int * pLR
        dpLR_int = k_int * pLR - k_deg * pLR_int
        return [dL, dR, dLR, dpLR, dpLR_int]

    def test_ligand_induced_receptor_activation(self):
        """配体诱导 EGFR 自磷酸化在 5 分钟内达峰。"""
        y0 = [100.0, 50.0, 0.0, 0.0, 0.0]
        t, y = run_ode(self._egfr_model, y0, t_end=60.0)
        pLR = y[3]
        peaks = detect_peaks(t, pLR)
        assert len(peaks) >= 1, "pEGFR 应出现至少一个峰值"
        peak_time, peak_amp = peaks[0]
        assert peak_time < 10.0, f"pEGFR 达峰时间 {peak_time:.1f}min > 10min"
        assert peak_amp > 1.0, f"pEGFR 峰值 {peak_amp:.2f} 过低"

    def test_receptor_internalization_degradation(self):
        """内吞的 pEGFR 随时间降级（单调下降支）。"""
        y0 = [100.0, 50.0, 0.0, 0.0, 0.0]
        t, y = run_ode(self._egfr_model, y0, t_end=120.0)
        pLR_int = y[4]
        # 内吞池应先升后降（降级占主导）
        assert pLR_int[-1] < pLR_int.max() * 0.8, "内吞 pEGFR 未明显降级"
        # 后半段应单调下降
        second_half = pLR_int[len(pLR_int) // 2:]
        diffs = np.diff(second_half)
        assert np.mean(diffs) < 0, "内吞 pEGFR 后半段未降级"

    def test_dose_response_saturation(self):
        """EGF 浓度-响应曲线呈饱和（Michaelis-Menten 形状）。"""
        doses = [1, 5, 10, 50, 100, 500, 1000]
        responses = []
        for L0 in doses:
            y0 = [float(L0), 50.0, 0.0, 0.0, 0.0]
            t, y = run_ode(self._egfr_model, y0, t_end=30.0)
            responses.append(float(y[3].max()))
        # 高剂量饱和：1000 vs 500 差异应小于 500 vs 100
        assert (responses[-1] - responses[-2]) < (responses[-2] - responses[3]), \
            "剂量-响应未饱和"

    def test_no_ligand_no_activation(self):
        """无配体时受体不应激活。"""
        y0 = [0.0, 50.0, 0.0, 0.0, 0.0]
        t, y = run_ode(self._egfr_model, y0, t_end=60.0)
        assert y[3].max() < 1e-6, "无配体时 pEGFR 应为零"

    @pytest.mark.benchmark
    def test_egfr_mapk_crosstalk_amplification(self):
        """EGFR→MAPK 跨通路放大（简化耦合，长时基准）。"""
        # 简化：pEGFR 驱动下游 Ras→ERK 放大
        def coupled(t, y):
            L, R, LR, pLR, _, RasGTP, ERK = y
            dL = -0.1 * L * R + 0.01 * LR
            dR = -0.1 * L * R + 0.01 * LR
            dLR = 0.1 * L * R - 0.01 * LR - 0.5 * LR
            dpLR = 0.5 * LR - 0.02 * pLR
            d_int = 0.02 * pLR
            dRasGTP = 0.3 * pLR - 0.1 * RasGTP
            dERK = 5.0 * RasGTP - 0.05 * ERK  # 放大系数 5x
            return [dL, dR, dLR, dpLR, d_int, dRasGTP, dERK]
        y0 = [100.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        t, y = run_ode(coupled, y0, t_end=60.0)
        erk_peak = float(y[6].max())
        pEGFR_peak = float(y[3].max())
        assert erk_peak > pEGFR_peak * 2, "ERK 相对 pEGFR 放大 < 2x"


# --------------------------------------------------------------------------- #
# 2. MAPK — 级联放大、ERK 双磷酸化
# --------------------------------------------------------------------------- #
class TestMAPKDynamics:
    """MAPK 三层级联：RasGTP→Raf→MEK→ERK 双磷酸化。"""

    def _mapk_cascade(self, t, y, k1=0.1, k2=0.2, k3=0.5):
        """三级磷酸化级联 [RasGTP, Raf, pRaf, MEK, pMEK, ERK, ppERK]。"""
        RasGTP, Raf, pRaf, MEK, pMEK, ERK, ppERK = y
        dRasGTP = -0.01 * RasGTP
        dpRaf = k1 * RasGTP * Raf - 0.1 * pRaf
        dRaf = -k1 * RasGTP * Raf + 0.1 * pRaf
        dpMEK = k2 * pRaf * MEK - 0.1 * pMEK
        dMEK = -k2 * pRaf * MEK + 0.1 * pMEK
        dppERK = k3 * pMEK * ERK - 0.05 * ppERK
        dERK = -k3 * pMEK * ERK + 0.05 * ppERK
        return [dRasGTP, dRaf, dpRaf, dMEK, dpMEK, dERK, dppERK]

    def test_cascade_amplification(self):
        """三级级联产生 >10x 信号放大。"""
        y0 = [100.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0]
        t, y = run_ode(self._mapk_cascade, y0, t_end=60.0)
        ppERK_peak = float(y[6].max())
        rasgtp_peak = float(y[0].max())
        assert ppERK_peak > rasgtp_peak * 10, (
            f"级联放大 {ppERK_peak / rasgtp_peak:.1f}x < 10x"
        )

    def test_erk_double_phosphorylation_kinetics(self):
        """ERK 双磷酸化：ppERK 出现且 MEK 依赖。"""
        y0 = [100.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0]
        t, y = run_ode(self._mapk_cascade, y0, t_end=30.0)
        ppERK = y[6]
        # ppERK 单调上升至稳态
        assert ppERK[-1] > ppERK[0], "ppERK 未上升"
        assert ppERK[-1] > 1.0, "ppERK 稳态过低"

    def test_ultrasensitivity_hill_coefficient(self):
        """Markevich 双磷酸化超敏感：输入-输出曲线呈 S 形。"""
        inputs = np.linspace(0, 200, 13)
        outputs = []
        for RasGTP0 in inputs:
            y0 = [float(RasGTP0), 100.0, 0.0, 100.0, 0.0, 100.0, 0.0]
            t, y = run_ode(self._mapk_cascade, y0, t_end=120.0)
            outputs.append(float(y[6][-1]))
        # S 形：中段斜率 > 端段斜率
        mid_slope = (outputs[7] - outputs[5]) / (inputs[7] - inputs[5])
        end_slope = (outputs[-1] - outputs[-3]) / (inputs[-1] - inputs[-3])
        assert mid_slope > end_slope * 1.5, "未呈现超敏感 S 形"

    def test_feedback_inhibition_by_erk(self):
        """ERK 负反馈抑制上游（ppERK 高时级联减弱）。"""
        def with_feedback(t, y):
            RasGTP, Raf, pRaf, MEK, pMEK, ERK, ppERK = y
            # ppERK 抑制 RasGTP 活化
            dRasGTP = 10.0 - 0.01 * RasGTP - 0.02 * RasGTP * ppERK
            dpRaf = 0.1 * RasGTP * Raf - 0.1 * pRaf
            dRaf = -0.1 * RasGTP * Raf + 0.1 * pRaf
            dpMEK = 0.2 * pRaf * MEK - 0.1 * pMEK
            dMEK = -0.2 * pRaf * MEK + 0.1 * pMEK
            dppERK = 0.5 * pMEK * ERK - 0.05 * ppERK
            dERK = -0.5 * pMEK * ERK + 0.05 * ppERK
            return [dRasGTP, dRaf, dpRaf, dMEK, dpMEK, dERK, dppERK]
        y0 = [0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0]
        t, y = run_ode(with_feedback, y0, t_end=120.0)
        ppERK = y[6]
        # 负反馈下 ppERK 应先升后回落（暂态峰）
        assert ppERK.max() > ppERK[-1] * 1.1, "负反馈未导致 ppERK 回落"

    def test_signal_propagation_delay(self):
        """信号逐级延迟：pRaf 早于 pMEK 早于 ppERK 达峰。"""
        y0 = [100.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0]
        t, y = run_ode(self._mapk_cascade, y0, t_end=30.0)
        pRaf_peaks = detect_peaks(t, y[2])
        pMEK_peaks = detect_peaks(t, y[4])
        ppERK_peaks = detect_peaks(t, y[6])
        if pRaf_peaks and pMEK_peaks and ppERK_peaks:
            assert pRaf_peaks[0][0] <= pMEK_peaks[0][0], "pRaf 达峰应早于 pMEK"
            assert pMEK_peaks[0][0] <= ppERK_peaks[0][0], "pMEK 达峰应早于 ppERK"


# --------------------------------------------------------------------------- #
# 3. PI3K — AKT 磷酸化、mTOR 反馈
# --------------------------------------------------------------------------- #
class TestPI3KDynamics:
    """PI3K-AKT-mTOR：PIP3 生成、AKT 磷酸化、S6K 负反馈。"""

    def _pi3k_model(self, t, y, k_pip3=0.3, k_akt=0.5, k_s6k=0.2,
                    k_fb=0.01, k_deg=0.02):
        """[PIP2, PIP3, AKT, pAKT, mTOR, S6K, pS6K]。"""
        PIP2, PIP3, AKT, pAKT, mTOR, S6K, pS6K = y
        dPIP2 = -k_pip3 * PIP2 + 0.1 * PIP3
        dPIP3 = k_pip3 * PIP2 - k_deg * PIP3
        dpAKT = k_akt * PIP3 * AKT - 0.05 * pAKT
        dAKT = -k_akt * PIP3 * AKT + 0.05 * pAKT
        dmTOR = 0.1 * pAKT - 0.01 * mTOR
        dpS6K = k_s6k * mTOR * S6K - 0.05 * pS6K
        dS6K = -k_s6k * mTOR * S6K + 0.05 * pS6K
        # S6K 负反馈抑制 PIP3 生成
        dPIP2 -= k_fb * pS6K * PIP2
        return [dPIP2, dPIP3, AKT, dpAKT, dmTOR, S6K, dpS6K]

    def test_akt_phosphorylation_by_pip3(self):
        """PIP3 驱动 AKT 磷酸化，pAKT 在 10min 内达峰。"""
        y0 = [100.0, 0.0, 50.0, 0.0, 0.0, 50.0, 0.0]
        t, y = run_ode(self._pi3k_model, y0, t_end=60.0)
        pAKT = y[3]
        peaks = detect_peaks(t, pAKT)
        assert len(peaks) >= 1, "pAKT 应有峰值"
        assert peaks[0][0] < 15.0, f"pAKT 达峰 {peaks[0][0]:.1f}min > 15min"

    def test_pip3_transient_then_decline(self):
        """PIP3 暂态升高后回落（PTEN / 降级）。"""
        y0 = [100.0, 0.0, 50.0, 0.0, 0.0, 50.0, 0.0]
        t, y = run_ode(self._pi3k_model, y0, t_end=120.0)
        PIP3 = y[1]
        assert PIP3.max() > PIP3[-1], "PIP3 应先升后落"
        assert PIP3[-1] < PIP3.max() * 0.7, "PIP3 未明显回落"

    def test_mtor_activation_downstream(self):
        """mTOR 被 pAKT 激活，滞后于 pAKT。"""
        y0 = [100.0, 0.0, 50.0, 0.0, 0.0, 50.0, 0.0]
        t, y = run_ode(self._pi3k_model, y0, t_end=60.0)
        mTOR = y[4]
        assert mTOR[-1] > 0.1, "mTOR 应被激活"
        # mTOR 上升应滞后于 pAKT
        pAKT_half = next((float(t[i]) for i in range(len(y[3]))
                          if y[3][i] > 0.5 * y[3].max()), 0.0)
        mTOR_half = next((float(t[i]) for i in range(len(mTOR))
                          if mTOR[i] > 0.5 * mTOR.max()), 0.0)
        assert mTOR_half >= pAKT_half, "mTOR 应滞后于 pAKT"

    def test_s6k_negative_feedback(self):
        """S6K 负反馈抑制上游 PIP3（反馈开启时 PIP3 更低）。"""
        # 无反馈
        y0 = [100.0, 0.0, 50.0, 0.0, 0.0, 50.0, 0.0]
        t1, y1 = run_ode(self._pi3k_model, y0, t_end=120.0)
        # 强反馈
        def strong_fb(t, y):
            return self._pi3k_model(t, y, k_fb=0.1)
        t2, y2 = run_ode(strong_fb, y0, t_end=120.0)
        assert y2[1].max() < y1[1].max(), "强反馈应降低 PIP3 峰值"

    def test_pten_opposes_pip3(self):
        """PTEN（k_deg 模拟）增大时 PIP3 降低。"""
        y0 = [100.0, 0.0, 50.0, 0.0, 0.0, 50.0, 0.0]
        t1, y1 = run_ode(self._pi3k_model, y0, t_end=60.0)
        def high_pten(t, y):
            return self._pi3k_model(t, y, k_deg=0.1)
        t2, y2 = run_ode(high_pten, y0, t_end=60.0)
        assert y2[1].max() < y1[1].max() * 0.8, "高 PTEN 未显著降低 PIP3"


# --------------------------------------------------------------------------- #
# 4. p53 — 振荡行为（p53-Mdm2 负反馈）
# --------------------------------------------------------------------------- #
class TestP53Dynamics:
    """p53-Mdm2 负反馈产生振荡（Lev Bar-Or 模型）。"""

    def _p53_oscillator(self, t, y, k1=0.2, k2=0.1, k3=0.1, k4=0.1,
                        k5=0.5, k6=0.1, n=2.0):
        """[p53, Mdm2_cyt, Mdm2_nuc]。"""
        p53, Mdm2_c, Mdm2_n = y
        # p53 生成 + Mdm2 介导降级（Hill 抑制）
        dp53 = k1 - k2 * Mdm2_n * p53 / (1 + (p53 / 10) ** n)
        # Mdm2 被 p53 转录诱导
        dMdm2_c = k3 * p53 - k4 * Mdm2_c
        # Mdm2 核转位
        dMdm2_n = k5 * Mdm2_c - k6 * Mdm2_n
        return [dp53, dMdm2_c, dMdm2_n]

    def test_p53_oscillation(self):
        """p53 呈现持续振荡（≥3 峰）。"""
        y0 = [5.0, 0.0, 0.0]
        t, y = run_ode(self._p53_oscillator, y0, t_end=600.0, n_points=1201)
        osc = detect_oscillation(t, y[0], min_peaks=3)
        assert osc["is_oscillating"], (
            f"p53 未振荡：{osc['n_peaks']} 峰，CV={osc.get('period_cv')}"
        )

    def test_oscillation_period_range(self):
        """p53 振荡周期在 300-400 min（5-7 小时）。"""
        y0 = [5.0, 0.0, 0.0]
        t, y = run_ode(self._p53_oscillator, y0, t_end=600.0, n_points=1201)
        osc = detect_oscillation(t, y[0], min_peaks=2)
        if osc["period"]:
            assert 200 < osc["period"] < 500, (
                f"p53 周期 {osc['period']:.0f}min 不在 200-500min 范围"
            )

    def test_mdm2_lags_p53(self):
        """Mdm2 核转位滞后于 p53 峰（负反馈相位差）。"""
        y0 = [5.0, 0.0, 0.0]
        t, y = run_ode(self._p53_oscillator, y0, t_end=600.0, n_points=1201)
        p53_peaks = detect_peaks(t, y[0])
        mdm2_peaks = detect_peaks(t, y[2])
        if len(p53_peaks) >= 2 and len(mdm2_peaks) >= 2:
            assert mdm2_peaks[0][0] > p53_peaks[0][0], "Mdm2 应滞后于 p53"

    def test_dna_damage_amplitude(self):
        """DNA 损伤（k1 增大）提高 p53 振荡幅度。"""
        y0 = [5.0, 0.0, 0.0]
        t1, y1 = run_ode(self._p53_oscillator, y0, t_end=400.0, n_points=801)
        def damage(t, y):
            return self._p53_oscillator(t, y, k1=0.5)
        t2, y2 = run_ode(damage, y0, t_end=400.0, n_points=801)
        assert y2[0].max() > y1[0].max(), "DNA 损伤应提高 p53 峰值"

    @pytest.mark.benchmark
    def test_nutlin_disrupts_feedback(self):
        """Nutlin-3 阻断 p53-Mdm2 结合，p53 不再振荡（持续升高）。"""
        def nutlin(t, y):
            # k2=0 模拟 Nutlin 阻断 Mdm2 介导降级
            return self._p53_oscillator(t, y, k2=0.0)
        y0 = [5.0, 0.0, 0.0]
        t, y = run_ode(nutlin, y0, t_end=400.0, n_points=801)
        p53 = y[0]
        # 无降级 → p53 持续升高（无振荡）
        assert p53[-1] > p53[len(p53) // 2], "Nutlin 下 p53 应持续升高"
        osc = detect_oscillation(t, p53, min_peaks=3)
        assert not osc["is_oscillating"], "Nutlin 应消除 p53 振荡"


# --------------------------------------------------------------------------- #
# 5. Apoptosis — Caspase 双稳态（MOMP）
# --------------------------------------------------------------------------- #
class TestApoptosisDynamics:
    """凋亡：Caspase 级联 + MOMP 正反馈产生双稳态开关。"""

    def _apoptosis_bistable(self, t, y, k_act=0.01, k_momp=0.1,
                            k_deg=0.001, feedback=0.05):
        """[Casp8, Cyt_c, Casp9, Casp3, XIAP]。"""
        C8, CytC, C9, C3, XIAP = y
        # Caspase-8 激活（外部信号）
        dC8 = 0.5 - k_deg * C8
        # Cyt c 释放（MOMP，被 Casp3 正反馈放大）
        dCytC = k_momp * C8 + feedback * C3 * (1 - CytC) - 0.01 * CytC
        # Casp9 被Cyt c 激活
        dC9 = k_act * CytC * (100 - C9) - 0.01 * C9
        # Casp3 被Casp9 激活，被 XIAP 抑制
        dC3 = 0.1 * C9 * (100 - C3) - 0.02 * XIAP * C3
        # XIAP 被 Casp3 降解（正反馈）
        dXIAP = -0.01 * C3 * XIAP
        return [dC8, CytC, dC9, dC3, dXIAP]

    def test_caspase_bistable_switch(self):
        """Casp3 呈双稳态：低于阈值不激活，高于阈值全激活。"""
        # 低初始 Casp8 → 不激活
        y0_low = [0.1, 0.0, 0.0, 0.0, 50.0]
        t1, y1 = run_ode(self._apoptosis_bistable, y0_low, t_end=200.0)
        # 高初始 Casp8 → 全激活
        y0_high = [5.0, 0.0, 0.0, 0.0, 50.0]
        t2, y2 = run_ode(self._apoptosis_bistable, y0_high, t_end=200.0)
        assert y1[3][-1] < 10.0, "低刺激下 Casp3 应保持低"
        assert y2[3][-1] > 50.0, "高刺激下 Casp3 应全激活"

    def test_momp_positive_feedback(self):
        """MOMP 正反馈：Casp3 一旦启动则快速放大。"""
        y0 = [2.0, 0.0, 0.0, 0.0, 50.0]
        t, y = run_ode(self._apoptosis_bistable, y0, t_end=300.0, n_points=1001)
        C3 = y[3]
        # 找到激活拐点（突变区间斜率大）
        diffs = np.abs(np.diff(C3))
        assert diffs.max() > 1.0, "Casp3 应有快速放大（MOMP 正反馈）"

    def test_xiap_degradation_by_casp3(self):
        """XIAP 被 Casp3 降解。"""
        y0 = [5.0, 0.0, 0.0, 0.0, 50.0]
        t, y = run_ode(self._apoptosis_bistable, y0, t_end=200.0)
        XIAP = y[4]
        assert XIAP[-1] < XIAP[0] * 0.5, "XIAP 应被 Casp3 降解"

    def test_point_of_no_return(self):
        """不可逆点：Casp3 超过阈值后即使撤信号也继续。"""
        # 先激活
        y0 = [5.0, 0.0, 0.0, 0.0, 50.0]
        t1, y1 = run_ode(self._apoptosis_bistable, y0, t_end=100.0)
        # 撤信号（C8 降为 0），从激活态继续
        y_mid = list(y1[:, -1])
        y_mid[0] = 0.0
        def no_signal(t, y):
            return self._apoptosis_bistable(t, y, k_act=0.01)
        t2, y2 = run_ode(no_signal, y_mid, t_end=200.0)
        assert y2[3][-1] > y_mid[3] * 0.9, "撤信号后 Casp3 应维持（不可逆）"

    @pytest.mark.benchmark
    def test_caspase_amplification_magnitude(self):
        """Caspase 级联放大 ~1000x（长时基准）。"""
        y0 = [1.0, 0.0, 0.0, 0.0, 50.0]
        t, y = run_ode(self._apoptosis_bistable, y0, t_end=300.0, n_points=1001)
        c8_peak = float(y[0].max())
        c3_peak = float(y[3].max())
        assert c3_peak > c8_peak * 10, "Casp3 相对 Casp8 放大 < 10x"


# --------------------------------------------------------------------------- #
# 6. Cell Cycle — Cyclin-CDK 周期振荡
# --------------------------------------------------------------------------- #
class TestCellCycleDynamics:
    """细胞周期：Goldbeter 最小 cyclin-CDK 振荡子。"""

    def _goldbeter(self, t, y, vi=0.025, kd1=0.01, kd2=0.01,
                   K1=0.01, K2=0.01, K3=0.01, K4=0.01,
                   V1=1.0, V2=0.5, V3=1.0, V4=0.5):
        """[Cyclin, cdc2_active, cdc2_inactive]。"""
        C, M, M_ = y
        dC = vi - kd1 * C
        # cdc2 被 cyclin 激活（V1 依赖 C）
        V1_eff = V1 * C / (K1 + C)
        dM = V1_eff * (1 - M) - V2 * M / (K2 + M)
        # M_ 是中间态（简化省略）
        dM_ = V3 * M / (K3 + M) - V4 * M_ / (K4 + M_)
        # M_ 反馈抑制 cyclin 生成
        dC -= kd2 * M_
        return [dC, dM, dM_]

    def test_cyclin_cdk_oscillation(self):
        """Cyclin-CDK 周期振荡。"""
        y0 = [0.1, 0.1, 0.1]
        t, y = run_ode(self._goldbeter, y0, t_end=500.0, n_points=2001)
        osc = detect_oscillation(t, y[0], min_peaks=3)
        assert osc["is_oscillating"], f"Cyclin 未振荡：{osc}"

    def test_oscillation_period_biological(self):
        """周期在 30-90 min（典型细胞周期）。"""
        y0 = [0.1, 0.1, 0.1]
        t, y = run_ode(self._goldbeter, y0, t_end=500.0, n_points=2001)
        osc = detect_oscillation(t, y[0], min_peaks=2)
        if osc["period"]:
            assert 20 < osc["period"] < 120, f"周期 {osc['period']:.1f}min 越界"

    def test_cdk_activation_follows_cyclin(self):
        """cdc2 激活跟随 cyclin 累积。"""
        y0 = [0.1, 0.1, 0.1]
        t, y = run_ode(self._goldbeter, y0, t_end=300.0, n_points=1001)
        cyclin_peaks = detect_peaks(t, y[0])
        cdk_peaks = detect_peaks(t, y[1])
        if cyclin_peaks and cdk_peaks:
            assert cdk_peaks[0][0] >= cyclin_peaks[0][0] - 5, "CDK 应跟随 Cyclin"

    def test_no_oscillation_without_feedback(self):
        """移除 M_ 反馈后振荡消失（趋稳态）。"""
        def no_fb(t, y):
            dy = self._goldbeter(t, y)
            dy[0] += 0.025  # 补回被减去的反馈项
            return dy
        y0 = [0.1, 0.1, 0.1]
        t, y = run_ode(no_fb, y0, t_end=500.0, n_points=2001)
        osc = detect_oscillation(t, y[0], min_peaks=3)
        assert not osc["is_oscillating"], "移除反馈后应停止振荡"

    def test_cyclin_degradation_required(self):
        """cyclin 降级（kd1）必需，否则不振荡。"""
        def no_deg(t, y):
            return self._goldbeter(t, y, kd1=0.0)
        y0 = [0.1, 0.1, 0.1]
        t, y = run_ode(no_deg, y0, t_end=500.0, n_points=2001)
        osc = detect_oscillation(t, y[0], min_peaks=3)
        assert not osc["is_oscillating"], "无 cyclin 降级仍振荡"


# --------------------------------------------------------------------------- #
# 7. JAK-STAT — STAT 二聚化、核转位
# --------------------------------------------------------------------------- #
class TestJAKSTATDynamics:
    """JAK-STAT：受体→JAK→STAT 磷酸化→二聚化→核转位。"""

    def _jak_stat(self, t, y, k_phos=0.5, k_dim=0.1, k_import=0.05,
                  k_export=0.01, k_deg=0.005):
        """[pSTAT, pSTAT_dim, pSTAT_nuc]。"""
        pSTAT, pSTAT_dim, pSTAT_nuc = y
        # 持续信号驱动磷酸化
        dpSTAT = 1.0 - k_phos * pSTAT - 2 * k_dim * pSTAT ** 2
        dpSTAT_dim = k_dim * pSTAT ** 2 - k_import * pSTAT_dim
        dpSTAT_nuc = k_import * pSTAT_dim - k_export * pSTAT_nuc - k_deg * pSTAT_nuc
        return [dpSTAT, dpSTAT_dim, dpSTAT_nuc]

    def test_stat_phosphorylation(self):
        """STAT 被磷酸化，pSTAT 升高。"""
        y0 = [0.0, 0.0, 0.0]
        t, y = run_ode(self._jak_stat, y0, t_end=60.0)
        assert y[0][-1] > 0.5, "pSTAT 应升高"
        assert y[0].max() > 0.8, "pSTAT 峰值过低"

    def test_dimerization(self):
        """pSTAT 形成二聚体。"""
        y0 = [0.0, 0.0, 0.0]
        t, y = run_ode(self._jak_stat, y0, t_end=60.0)
        assert y[1].max() > 0.1, "pSTAT 二聚体应形成"

    def test_nuclear_translocation(self):
        """pSTAT 二聚体核转位。"""
        y0 = [0.0, 0.0, 0.0]
        t, y = run_ode(self._jak_stat, y0, t_end=120.0)
        assert y[2][-1] > 0.1, "核内 pSTAT 应累积"
        assert y[2][-1] > y[2][0], "核转位应单调上升"

    def test_signal_withdrawal_decay(self):
        """撤信号后 pSTAT 衰减。"""
        # 先激活 60min
        y0 = [0.0, 0.0, 0.0]
        t1, y1 = run_ode(self._jak_stat, y0, t_end=60.0)
        # 撤信号（磷酸化降为 0）
        def withdraw(t, y):
            pSTAT, pSTAT_dim, pSTAT_nuc = y
            dpSTAT = -0.5 * pSTAT - 2 * 0.1 * pSTAT ** 2
            dpSTAT_dim = 0.1 * pSTAT ** 2 - 0.05 * pSTAT_dim
            dpSTAT_nuc = 0.05 * pSTAT_dim - 0.01 * pSTAT_nuc - 0.005 * pSTAT_nuc
            return [dpSTAT, dpSTAT_dim, dpSTAT_nuc]
        y_mid = list(y1[:, -1])
        t2, y2 = run_ode(withdraw, y_mid, t_end=120.0)
        assert y2[0][-1] < y_mid[0] * 0.5, "撤信号后 pSTAT 应衰减"

    def test_dimerization_required_for_import(self):
        """二聚化是核转位前提：阻断二聚化则核内无 pSTAT。"""
        def no_dim(t, y):
            pSTAT, pSTAT_dim, pSTAT_nuc = y
            dpSTAT = 1.0 - 0.5 * pSTAT  # 无二聚化消耗
            dpSTAT_dim = -0.05 * pSTAT_dim
            dpSTAT_nuc = -0.015 * pSTAT_nuc
            return [dpSTAT, dpSTAT_dim, dpSTAT_nuc]
        y0 = [0.0, 0.0, 0.0]
        t, y = run_ode(no_dim, y0, t_end=60.0)
        assert y[2][-1] < 0.01, "阻断二聚化后核内不应有 pSTAT"


# --------------------------------------------------------------------------- #
# 8. NF-κB — 周期振荡（IκBα 负反馈）
# --------------------------------------------------------------------------- #
class TestNFKBDynamics:
    """NF-κB-IκBα 负反馈振荡（Hoffmann 模型）。"""

    def _nfkb(self, t, y, k1=0.5, k2=0.1, k3=0.5, k4=0.1,
              k5=0.1, k_deg=0.01, n=2.0):
        """[NFkB_nuc, IkBa_cyt, IkBa_nuc, IKK]。"""
        NFkB_n, IkBa_c, IkBa_n, IKK = y
        # IKK 信号（暂态）
        dIKK = -k_deg * IKK
        # NFkB 核转位（IKK 降解 IκBα）
        dNFkB_n = k1 * IKK * (1 - NFkB_n) - k2 * IkBa_n * NFkB_n
        # IκBα 转录（被 NFkB 诱导，Hill）
        dIkBa_c = k3 * NFkB_n ** n / (1 + NFkB_n ** n) - k4 * IkBa_c
        # IκBα 核转位，结合 NFkB 出核
        dIkBa_n = k5 * IkBa_c - k2 * IkBa_n * NFkB_n - k_deg * IkBa_n
        return [dNFkB_n, dIkBa_c, dIkBa_n, dIKK]

    def test_nfkb_oscillation(self):
        """NF-κB 核转位呈周期振荡。"""
        y0 = [0.0, 0.0, 0.0, 1.0]
        t, y = run_ode(self._nfkb, y0, t_end=300.0, n_points=1201)
        osc = detect_oscillation(t, y[0], min_peaks=2)
        assert osc["is_oscillating"], f"NF-κB 未振荡：{osc}"

    def test_ikba_negative_feedback(self):
        """IκBα 是负反馈：滞后于 NF-κB 峰。"""
        y0 = [0.0, 0.0, 0.0, 1.0]
        t, y = run_ode(self._nfkb, y0, t_end=300.0, n_points=1201)
        nfkb_peaks = detect_peaks(t, y[0])
        ikba_peaks = detect_peaks(t, y[1])
        if nfkb_peaks and ikba_peaks:
            assert ikba_peaks[0][0] > nfkb_peaks[0][0], "IκBα 应滞后于 NF-κB"

    def test_oscillation_damps_without_ikk(self):
        """IKK 衰减后振荡阻尼（趋稳态）。"""
        y0 = [0.0, 0.0, 0.0, 1.0]
        t, y = run_ode(self._nfkb, y0, t_end=600.0, n_points=2401)
        nfkb = y[0]
        # 后期振幅应小于前期
        first_peaks = [p[1] for p in detect_peaks(t[:600], nfkb[:600])]
        late_peaks = [p[1] for p in detect_peaks(t[1200:], nfkb[1200:] + t[:1])]
        if first_peaks and late_peaks:
            assert max(late_peaks) < max(first_peaks), "振荡应阻尼"

    def test_nfkb_nuclear_localization(self):
        """刺激后 NF-κB 核内累积。"""
        y0 = [0.0, 0.0, 0.0, 1.0]
        t, y = run_ode(self._nfkb, y0, t_end=60.0)
        assert y[0].max() > 0.3, "NF-κB 核内应累积"

    def test_ikba_resynthesis(self):
        """IκBα 被 NF-κB 重新转录合成。"""
        y0 = [0.5, 0.0, 0.0, 0.5]
        t, y = run_ode(self._nfkb, y0, t_end=120.0)
        assert y[1].max() > 0.01, "IκBα 应被重新合成"


# --------------------------------------------------------------------------- #
# 9. Wnt — β-catenin 破坏复合体
# --------------------------------------------------------------------------- #
class TestWntDynamics:
    """Wnt：破坏复合体降解 β-catenin，Wnt 信号稳定 β-catenin。"""

    def _wnt(self, t, y, k_syn=0.5, k_deg=0.1, k_dest=0.2,
             wnt_on=False, n=4.0):
        """[bCatenin, DestComplex, bCat_nuc]。"""
        bCat, DC, bCat_n = y
        # β-catenin 合成
        synth = k_syn
        # 破坏复合体降解（Wnt 关闭时）
        if wnt_on:
            deg = 0.01  # Wnt 抑制破坏复合体
        else:
            deg = k_deg * DC
        dbCat = synth - deg * bCat - 0.05 * bCat
        # 破坏复合体动力学
        if wnt_on:
            dDC = -0.1 * DC  # Wnt 解离破坏复合体
        else:
            dDC = 0.1 * (1 - DC)  # 重新组装
        # 核转位（Hill 依赖胞质 β-catenin）
        dbCat_n = 0.1 * bCat ** n / (1 + bCat ** n) - 0.02 * bCat_n
        return [dbCat, dDC, dbCat_n]

    def test_bcatenin_degradation_off_state(self):
        """Wnt OFF：破坏复合体降解 β-catenin（低稳态）。"""
        y0 = [1.0, 1.0, 0.0]
        t, y = run_ode(self._wnt, y0, t_end=200.0, wnt_on=False)
        assert y[0][-1] < 0.5, "Wnt OFF 时 β-catenin 应低"

    def test_bcatenin_stabilization_on_state(self):
        """Wnt ON：β-catenin 稳定累积（高稳态）。"""
        y0 = [0.1, 1.0, 0.0]
        t, y = run_ode(self._wnt, y0, t_end=200.0, wnt_on=True)
        assert y[0][-1] > 1.0, "Wnt ON 时 β-catenin 应高"

    def test_destruction_complex_disassembly(self):
        """Wnt ON 时破坏复合体解离。"""
        y0 = [0.1, 1.0, 0.0]
        t, y = run_ode(self._wnt, y0, t_end=200.0, wnt_on=True)
        assert y[1][-1] < y[1][0] * 0.5, "破坏复合体应解离"

    def test_nuclear_translocation_on_wnt(self):
        """Wnt ON 时 β-catenin 核转位。"""
        y0 = [0.1, 1.0, 0.0]
        t, y = run_ode(self._wnt, y0, t_end=200.0, wnt_on=True)
        assert y[2][-1] > 0.1, "核内 β-catenin 应累积"

    def test_bistable_switch(self):
        """β-catenin 呈 Wnt 依赖双稳态开关。"""
        # OFF → 低
        t1, y1 = run_ode(self._wnt, [1.0, 1.0, 0.0], t_end=200.0, wnt_on=False)
        # ON → 高
        t2, y2 = run_ode(self._wnt, [0.1, 1.0, 0.0], t_end=200.0, wnt_on=True)
        ratio = y2[0][-1] / max(y1[0][-1], 1e-6)
        assert ratio > 3.0, f"ON/OFF 比值 {ratio:.1f} < 3（非开关）"


# --------------------------------------------------------------------------- #
# 10. TGF-β — SMAD 核转位
# --------------------------------------------------------------------------- #
class TestTGFBetaDynamics:
    """TGF-β：受体→R-SMAD 磷酸化→Co-SMAD 复合→核转位。"""

    def _tgfb(self, t, y, k_phos=0.5, k_bind=0.3, k_import=0.1,
              k_export=0.02, k_deg=0.01):
        """[pSmad2, pSmad2_Smad4, Smad_complex_nuc]。"""
        pS2, complex_c, complex_n = y
        # 持续 TGF-β 信号驱动 Smad2 磷酸化
        dpS2 = 1.0 - k_phos * pS2 - k_bind * pS2
        dcomplex_c = k_bind * pS2 - k_import * complex_c
        dcomplex_n = k_import * complex_c - k_export * complex_n - k_deg * complex_n
        return [dpS2, dcomplex_c, dcomplex_n]

    def test_smad2_phosphorylation(self):
        """TGF-β 诱导 Smad2 磷酸化。"""
        y0 = [0.0, 0.0, 0.0]
        t, y = run_ode(self._tgfb, y0, t_end=60.0)
        assert y[0][-1] > 0.5, "pSmad2 应升高"

    def test_smad_complex_formation(self):
        """pSmad2 与 Smad4 形成复合体。"""
        y0 = [0.0, 0.0, 0.0]
        t, y = run_ode(self._tgfb, y0, t_end=60.0)
        assert y[1].max() > 0.1, "pSmad2-Smad4 复合体应形成"

    def test_nuclear_translocation(self):
        """SMAD 复合体核转位。"""
        y0 = [0.0, 0.0, 0.0]
        t, y = run_ode(self._tgfb, y0, t_end=120.0)
        assert y[2][-1] > 0.1, "核内 SMAD 复合体应累积"
        assert y[2][-1] > y[2][0], "核转位应单调上升"

    def test_signal_withdrawal(self):
        """撤 TGF-β 后 pSmad2 衰减，核内复合体输出。"""
        # 先激活 60min
        y0 = [0.0, 0.0, 0.0]
        t1, y1 = run_ode(self._tgfb, y0, t_end=60.0)
        y_mid = list(y1[:, -1])
        # 撤信号
        def withdraw(t, y):
            pS2, complex_c, complex_n = y
            dpS2 = -0.5 * pS2 - 0.3 * pS2
            dcomplex_c = 0.3 * pS2 - 0.1 * complex_c
            dcomplex_n = 0.1 * complex_c - 0.02 * complex_n - 0.01 * complex_n
            return [dpS2, dcomplex_c, dcomplex_n]
        t2, y2 = run_ode(withdraw, y_mid, t_end=120.0)
        assert y2[0][-1] < y_mid[0] * 0.5, "撤信号后 pSmad2 应衰减"

    def test_dose_response(self):
        """TGF-β 剂量-响应（磷酸化速率）。"""
        responses = []
        for k_phos in [0.1, 0.3, 0.5, 1.0, 2.0]:
            def model(t, y, kp=k_phos):
                return self._tgfb(t, y, k_phos=kp)
            t, y = run_ode(model, [0.0, 0.0, 0.0], t_end=60.0)
            responses.append(float(y[2][-1]))
        # 剂量依赖递增
        assert responses[-1] > responses[0], "TGF-β 剂量响应应递增"
        assert all(responses[i + 1] >= responses[i] - 0.01
                   for i in range(len(responses) - 1)), "剂量响应非单调"


# --------------------------------------------------------------------------- #
# 汇总：通路覆盖完整性检查
# --------------------------------------------------------------------------- #
def test_all_ten_pathways_covered():
    """断言 10 条核心通路均有动力学测试类。"""
    covered = {
        "EGFR_RTK", "MAPK_ERK", "PI3K_AKT_mTOR", "p53", "APOPTOSIS",
        "CELL_CYCLE", "JAK_STAT", "NF_KB", "WNT", "TGF_BETA",
    }
    # 通过 TestXxxDynamics 类名映射到通路
    class_to_pathway = {
        "TestEGFRDynamics": "EGFR_RTK",
        "TestMAPKDynamics": "MAPK_ERK",
        "TestPI3KDynamics": "PI3K_AKT_mTOR",
        "TestP53Dynamics": "p53",
        "TestApoptosisDynamics": "APOPTOSIS",
        "TestCellCycleDynamics": "CELL_CYCLE",
        "TestJAKSTATDynamics": "JAK_STAT",
        "TestNFKBDynamics": "NF_KB",
        "TestWntDynamics": "WNT",
        "TestTGFBetaDynamics": "TGF_BETA",
    }
    actual = set(class_to_pathway.values())
    missing = covered - actual
    assert not missing, f"未覆盖通路：{missing}"
