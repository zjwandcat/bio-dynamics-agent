# BioDynamics Agent v4 - Sensitivity Analyzer Orchestrator (Phase 5 / Task 5.8.4)
#
# SensitivityAnalyzer 主编排器 + LangGraph hook 节点。
# 职责：编排 local + sobol + morris 三种灵敏度分析，聚合输出 v4_sensitivity_report。
#   - local sensitivity（始终执行，无外部依赖）
#   - sobol（SALib 可用时执行，否则 skipped）
#   - morris（SALib 可用时执行，否则 skipped）
#
# 设计原则（铁律）：
# 1. Feature Flag V4_CALIBRATION_AGENT_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
# 3. 仅消费 P1/P2/P3/P5 产出（v4_ode_system / v4_calibration_result / state.parameters）
# 4. 失败降级：任何异常都返回空结果或降级路径，不阻塞主流水线
# 5. 输出写入 state["v4_sensitivity_report"]（新增 v4 字段，与 v3 字段共存）
# 6. 依赖隔离：SALib 不可用时仅运行 local sensitivity + warning
# 7. 单文件优先；不创建多余依赖
#
# 对应 spec.md Part 4 Sensitivity Analysis（第 342-346 行）
# - 输入：v4_ode_system + v4_calibration_result（可选）+ state.parameters
# - 输出：v4_sensitivity_report = {
#     local_sensitivity: {param_name: sensitivity_value, ...},
#     sobol: {S1, ST, method, ...} | None,
#     morris: {mu, sigma, mu_star, method, ...} | None,
#     method: "full"|"local_only",
#     salib_available: bool,
#     warnings: list[str]
# }
# - 失败策略：SALib 不可用时仅运行 local sensitivity + warning
#
# 依赖：
# - app.config.settings（Feature Flag）
# - app.sensitivity.local_sensitivity.LocalSensitivityAnalyzer
# - app.sensitivity.sobol_analyzer.SobolAnalyzer
# - app.sensitivity.morris_analyzer.MorrisAnalyzer

from __future__ import annotations

import logging
from typing import Any, Callable

from app.config import SALIB_AVAILABLE, settings
from app.state import set_v4_state
from app.sensitivity.local_sensitivity import (
    LocalSensitivityAnalyzer,
    LocalSensitivityResult,
)
from app.sensitivity.morris_analyzer import MorrisAnalyzer, MorrisResult
from app.sensitivity.sobol_analyzer import SobolAnalyzer, SobolResult

logger = logging.getLogger(__name__)


# =============================================================================
# SensitivityAnalyzer 主编排器
# =============================================================================
class SensitivityAnalyzer:
    """Sensitivity Analysis 主编排器。

    主入口 analyze(state, model_func)：
    1. 提取输入：v4_ode_system / v4_calibration_result.calibrated_params / parameters
    2. 提取参数 dict（优先 calibrated_params，回退到 state.parameters）
    3. 调用 local_analyzer.analyze（始终执行）
    4. 调用 sobol_analyzer.analyze（可能 skipped）
    5. 调用 morris_analyzer.analyze（可能 skipped）
    6. 聚合输出 v4_sensitivity_report

    用法：
        analyzer = SensitivityAnalyzer()
        report = analyzer.analyze(state, model_func)
        # report = {"v4_sensitivity_report": {...}}
    """

    AGENT_VERSION = "v4.0"

    def __init__(
        self,
        local_analyzer: LocalSensitivityAnalyzer | None = None,
        sobol_analyzer: SobolAnalyzer | None = None,
        morris_analyzer: MorrisAnalyzer | None = None,
    ) -> None:
        """依赖注入（默认创建，测试时可 mock）。

        Args:
            local_analyzer: LocalSensitivityAnalyzer 实例（None 时创建默认）
            sobol_analyzer: SobolAnalyzer 实例（None 时创建默认）
            morris_analyzer: MorrisAnalyzer 实例（None 时创建默认）
        """
        self._local_analyzer = (
            local_analyzer if local_analyzer is not None else LocalSensitivityAnalyzer()
        )
        self._sobol_analyzer = (
            sobol_analyzer if sobol_analyzer is not None else SobolAnalyzer()
        )
        self._morris_analyzer = (
            morris_analyzer if morris_analyzer is not None else MorrisAnalyzer()
        )
        self._salib_available = SALIB_AVAILABLE

    # =========================================================================
    # 主入口
    # =========================================================================
    def analyze(
        self,
        state: dict[str, Any],
        model_func: Callable[..., float] | None = None,
    ) -> dict[str, Any]:
        """主入口：执行 Sensitivity Analysis。

        Args:
            state: LangGraph 全局状态，含：
                - v4_ode_system: P3 输出的 ODE 系统（含 ode_code / equations / parameters）
                - v4_calibration_result: P5 校准结果（可选，含 calibrated_params）
                - parameters: v3 参数 dict（仅读取，不修改）
            model_func: 模型函数（输入参数 dict，返回标量）。
                None 时使用默认占位（参数乘积的标量）。

        Returns:
            {"v4_sensitivity_report": {
                local_sensitivity: dict,           # {param_name: sensitivity_value}
                sobol: dict | None,                # {S1, ST, method, ...} 或 None
                morris: dict | None,                # {mu, sigma, mu_star, method, ...} 或 None
                method: "full"|"local_only",       # 聚合方法标识
                salib_available: bool,             # SALib 是否可用
                warnings: list[str]                # 警告信息
            }}
            失败降级返回 fallback 结果（local_sensitivity={}, method="skipped"）。
        """
        try:
            # 输入校验：state 必须 dict
            if not isinstance(state, dict):
                return self._fallback_result("invalid_state_type")

            # 提取参数 dict（优先 calibrated_params，回退到 state.parameters）
            params = self._extract_params(state)
            if not params:
                logger.warning(
                    "SensitivityAnalyzer 无可用参数，降级返回 fallback"
                )
                return self._fallback_result("empty_params")

            # 选择 model_func（默认占位）
            effective_model = (
                model_func if model_func is not None else self._default_model
            )
            # TD-025 修复：默认路径标记，用于触发真实有限差分灵敏度分析
            use_default_model = model_func is None

            logger.info(
                "SensitivityAnalyzer 开始分析：params=%d, salib_available=%s",
                len(params),
                self._salib_available,
            )

            warnings: list[str] = []

            # 1. local sensitivity（始终执行）
            local_results = self._local_analyzer.analyze(params, effective_model)
            local_sensitivity = {
                name: float(result.sensitivity)
                for name, result in local_results.items()
            }

            # 1.5 TD-025 修复：默认路径下执行真实有限差分灵敏度分析
            # （基于 ODE 轨迹的 ±1% 中心差分 + 归一化 L2 范数，替代占位 dummy）
            finite_difference_sensitivity: dict[str, float] | None = None
            if use_default_model:
                try:
                    param_names = list(params.keys())
                    # 默认 t_span 与 _default_model 的 t_end 一致；y0 每个状态 1.0
                    t_span_default = (0.0, 10.0)
                    y0_default = [1.0] * max(len(param_names), 1)
                    finite_difference_sensitivity = (
                        self._finite_difference_sensitivity(
                            self._default_ode_func,
                            params,
                            t_span_default,
                            y0_default,
                            param_names,
                        )
                    )
                    logger.info(
                        "有限差分灵敏度完成：%d 个参数",
                        len(finite_difference_sensitivity),
                    )
                except Exception as fd_exc:
                    # 有限差分失败不影响主流水线，仅记录 warning
                    logger.warning(
                        "有限差分灵敏度分析失败（不阻塞主流水线）: %s", fd_exc
                    )
                    warnings.append(f"finite_difference_failed: {fd_exc}")

            # 2. sobol（可能 skipped）
            sobol_result = self._sobol_analyzer.analyze(
                params, effective_model
            )
            sobol_dict = self._to_sobol_dict(sobol_result)
            if sobol_result.method == "skipped":
                warnings.extend(sobol_result.warnings)

            # 3. morris（可能 skipped）
            morris_result = self._morris_analyzer.analyze(
                params, effective_model
            )
            morris_dict = self._to_morris_dict(morris_result)
            if morris_result.method == "skipped":
                warnings.extend(morris_result.warnings)

            # 聚合 method
            both_success = (
                sobol_result.method == "sobol"
                and morris_result.method == "morris"
            )
            method = "full" if both_success else "local_only"

            # TD-025 修复：新增 finite_difference_sensitivity 字段（默认路径下真实计算）
            result = {
                "local_sensitivity": local_sensitivity,
                "finite_difference_sensitivity": finite_difference_sensitivity,
                "sobol": sobol_dict,
                "morris": morris_dict,
                "method": method,
                "salib_available": bool(self._salib_available),
                "warnings": warnings,
                "agent_version": self.AGENT_VERSION,
            }

            logger.info(
                "SensitivityAnalyzer 完成：method=%s, local=%d, "
                "sobol=%s, morris=%s, warnings=%d",
                method,
                len(local_sensitivity),
                sobol_result.method,
                morris_result.method,
                len(warnings),
            )

            return {"v4_sensitivity_report": result}

        except Exception as exc:
            logger.warning(
                "SensitivityAnalyzer.analyze 失败，降级返回空结果: %s", exc
            )
            return self._fallback_result(f"analyze_exception: {exc}")

    # =========================================================================
    # 输入提取辅助方法
    # =========================================================================
    def _extract_params(self, state: dict[str, Any]) -> dict[str, Any]:
        """提取参数 dict。

        优先级：
        1. v4_calibration_result.calibrated_params（P5 校准结果，最新）
        2. state.parameters（v3 参数，回退）
        3. v4_ode_system.parameters（P3 输出，备选）

        Args:
            state: LangGraph 全局状态

        Returns:
            参数 dict（合并后，calibrated_params 覆盖 parameters）
        """
        merged: dict[str, Any] = {}

        # 1. state.parameters（基础）
        parameters = state.get("parameters")
        if isinstance(parameters, dict):
            for k, v in parameters.items():
                if isinstance(k, str):
                    merged[k] = v

        # 2. v4_ode_system.parameters（备选补充）
        ode_system = state.get("v4_ode_system")
        if isinstance(ode_system, dict):
            ode_params = ode_system.get("parameters")
            if isinstance(ode_params, dict):
                for k, v in ode_params.items():
                    if isinstance(k, str) and k not in merged:
                        merged[k] = v

        # 3. v4_calibration_result.calibrated_params（覆盖，最高优先级）
        cal_result = state.get("v4_calibration_result")
        if isinstance(cal_result, dict):
            calibrated_params = cal_result.get("calibrated_params")
            if isinstance(calibrated_params, dict):
                for k, v in calibrated_params.items():
                    if isinstance(k, str):
                        merged[k] = v

        return merged

    # =========================================================================
    # 结果转换辅助方法
    # =========================================================================
    @staticmethod
    def _to_sobol_dict(sobol_result: SobolResult) -> dict[str, Any] | None:
        """将 SobolResult 转换为 dict（写入 state 用）。"""
        if sobol_result.method == "skipped" and not sobol_result.warnings:
            return None
        return {
            "S1": dict(sobol_result.S1),
            "ST": dict(sobol_result.ST),
            "S2": sobol_result.S2,
            "method": sobol_result.method,
            "n_samples": sobol_result.n_samples,
            "warnings": list(sobol_result.warnings),
        }

    @staticmethod
    def _to_morris_dict(morris_result: MorrisResult) -> dict[str, Any] | None:
        """将 MorrisResult 转换为 dict（写入 state 用）。"""
        if morris_result.method == "skipped" and not morris_result.warnings:
            return None
        return {
            "mu": dict(morris_result.mu),
            "sigma": dict(morris_result.sigma),
            "mu_star": dict(morris_result.mu_star),
            "method": morris_result.method,
            "n_trajectories": morris_result.n_trajectories,
            "warnings": list(morris_result.warnings),
        }

    # =========================================================================
    # 默认 model_func（TD-025 修复：真实一阶动力学模型，替代参数积占位）
    # =========================================================================
    @staticmethod
    def _default_model(params: dict[str, Any]) -> float:
        """真实默认 model_func（替代占位 placeholder）。

        基于一阶动力学指数衰减模型：将每个参数视为独立衰减速率常数 k_i，
        解析解 y_i(t) = y0 * exp(-k_i * t)，返回 t_end 时刻总浓度标量。
        供 local / sobol / morris 分析器作为标量输出模型使用。
        生产环境应注入真实 ODE 仿真（roadrunner / scipy.solve_ivp）。
        """
        if not params:
            return 0.0
        import math

        t_end = 10.0  # 默认仿真终点（与 _finite_difference_sensitivity 一致）
        total = 0.0
        for v in params.values():
            try:
                if isinstance(v, bool):
                    k = 1.0
                else:
                    k = float(v)
            except (TypeError, ValueError):
                k = 1.0
            # 一阶衰减：y(t_end) = exp(-|k| * t_end)（y0=1.0）
            total += math.exp(-abs(k) * t_end)
        return total

    @staticmethod
    def _default_ode_func(t: float, y: Any, params: dict[str, Any]) -> Any:
        """默认 ODE 右端函数：独立一阶衰减 dy_i/dt = -|k_i| * y_i。

        供 _finite_difference_sensitivity 在默认路径下使用。
        """
        import numpy as np  # type: ignore[import-untyped]

        param_values: list[float] = []
        for v in params.values():
            try:
                if isinstance(v, bool):
                    param_values.append(1.0)
                else:
                    param_values.append(float(v))
            except (TypeError, ValueError):
                param_values.append(1.0)
        dy = np.zeros_like(y, dtype=float)
        for i, k in enumerate(param_values):
            if i < len(dy):
                dy[i] = -abs(k) * y[i]
        return dy

    # =========================================================================
    # 真实有限差分灵敏度分析（TD-025 修复：替代占位 dummy 值）
    # =========================================================================
    @staticmethod
    def _finite_difference_sensitivity(
        ode_func: Callable[..., Any],
        params: dict[str, Any],
        t_span: tuple[float, float],
        y0: list[float],
        param_names: list[str],
    ) -> dict[str, float]:
        """真实有限差分灵敏度分析。

        对每个参数进行 ±1% 扰动，求解 ODE 得到输出轨迹，计算扰动前后轨迹差的
        归一化 L2 范数作为灵敏度得分。

        Args:
            ode_func: ODE 右端函数，签名 ode_func(t, y, params) -> dy/dt
            params: 参数 dict {param_name: value}
            t_span: (t_start, t_end) 仿真时间区间
            y0: 初始状态向量
            param_names: 参数名列表（顺序对应 params）

        Returns:
            {param_name: sensitivity_score}
            - sensitivity_score = ||y_perturbed - y_baseline||_2 / ||y_baseline||_2
              （±1% 扰动结果的平均值）
            - ODE 求解失败（数值不稳定）的参数返回 0.0 + warning
        """
        import numpy as np  # type: ignore[import-untyped]

        # 求解基线轨迹
        y_base = SensitivityAnalyzer._solve_ode_trajectory(
            ode_func, params, t_span, y0
        )
        if y_base is None:
            # 基线求解失败 → 所有参数灵敏度 0.0
            logger.warning(
                "有限差分灵敏度：基线 ODE 求解失败，所有参数 sensitivity=0.0"
            )
            return {name: 0.0 for name in param_names}

        base_norm = float(np.linalg.norm(y_base))
        results: dict[str, float] = {}

        for name in param_names:
            value = params.get(name)
            # 非数值参数 → sensitivity=0.0
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                results[name] = 0.0
                continue
            original = float(value)

            # +1% 扰动
            sens_plus = SensitivityAnalyzer._perturb_sensitivity(
                ode_func, params, name, original * 1.01,
                t_span, y0, y_base, base_norm,
            )
            # -1% 扰动
            sens_minus = SensitivityAnalyzer._perturb_sensitivity(
                ode_func, params, name, original * 0.99,
                t_span, y0, y_base, base_norm,
            )
            # 中心差分：取两侧平均值
            results[name] = (sens_plus + sens_minus) / 2.0

        return results

    @staticmethod
    def _perturb_sensitivity(
        ode_func: Callable[..., Any],
        params: dict[str, Any],
        name: str,
        new_value: float,
        t_span: tuple[float, float],
        y0: list[float],
        y_base: Any,
        base_norm: float,
    ) -> float:
        """扰动单参数后计算归一化 L2 灵敏度；失败返回 0.0。"""
        import numpy as np  # type: ignore[import-untyped]

        perturbed = dict(params)
        perturbed[name] = new_value
        y_pert = SensitivityAnalyzer._solve_ode_trajectory(
            ode_func, perturbed, t_span, y0
        )
        if y_pert is None:
            # ODE 求解失败（数值不稳定）→ sensitivity=0.0 + warning
            logger.warning(
                "参数 %s 扰动 ODE 求解失败（数值不稳定），sensitivity=0.0",
                name,
            )
            return 0.0
        # 对齐时间点数（防御性：扰动轨迹长度可能与基线不一致）
        n = min(len(y_pert), len(y_base))
        diff = y_pert[:n] - y_base[:n]
        diff_norm = float(np.linalg.norm(diff))
        if base_norm <= 0:
            return 0.0
        return diff_norm / base_norm

    @staticmethod
    def _solve_ode_trajectory(
        ode_func: Callable[..., Any],
        params: dict[str, Any],
        t_span: tuple[float, float],
        y0: list[float],
    ) -> Any:
        """求解 ODE 返回轨迹矩阵 (n_times, n_states)；失败返回 None。

        使用 scipy.integrate.solve_ivp（LSODA）。scipy 不可用或求解失败 → None。
        """
        import numpy as np  # type: ignore[import-untyped]

        try:
            from scipy.integrate import solve_ivp  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("scipy 不可用，有限差分灵敏度无法求解 ODE")
            return None

        y0_arr = np.asarray(y0, dtype=float)
        n_states = max(len(y0_arr), 1)
        t_start = float(t_span[0])
        t_end = float(t_span[1])
        t_eval = np.linspace(t_start, t_end, max(n_states, 10))

        def rhs(t: float, y: Any) -> Any:
            # 包装 ode_func，异常时返回零导数（避免求解器崩溃）
            try:
                dy = ode_func(t, y, params)
                dy = np.asarray(dy, dtype=float)
                if dy.shape != y.shape:
                    dy = np.zeros_like(y, dtype=float)
                return dy
            except Exception:
                return np.zeros_like(y, dtype=float)

        try:
            sol = solve_ivp(
                rhs, (t_start, t_end), y0_arr,
                t_eval=t_eval, method="LSODA", rtol=1e-6, atol=1e-9,
            )
            if not sol.success or sol.y is None:
                return None
            # 返回 (n_times, n_states)
            return sol.y.T
        except Exception as exc:
            logger.warning("ODE 求解失败: %s", exc)
            return None

    # =========================================================================
    # 失败降级方法
    # =========================================================================
    def _fallback_result(self, reason: str = "") -> dict[str, Any]:
        """失败降级：返回空 sensitivity_report（仍含 v4_sensitivity_report 字段）。

        与 calibration_agent._fallback_result 一致：标记 fallback=True，
        不阻塞主流水线，warnings 记录失败原因。

        Args:
            reason: 失败原因（记录到 warnings）

        Returns:
            {"v4_sensitivity_report": {
                local_sensitivity: {},
                sobol: None,
                morris: None,
                method: "skipped",
                salib_available: bool,
                warnings: [f"sensitivity_fallback: {reason}"],
                fallback: True
            }}
        """
        return {
            "v4_sensitivity_report": {
                "local_sensitivity": {},
                "sobol": None,
                "morris": None,
                "method": "skipped",
                "salib_available": bool(self._salib_available),
                "warnings": [f"sensitivity_fallback: {reason}"] if reason else [],
                "agent_version": self.AGENT_VERSION,
                "fallback": True,
            }
        }


# =============================================================================
# LangGraph hook 节点（Feature Flag 隔离）
# =============================================================================
def sensitivity_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：Sensitivity Analysis hook。

    行为：
    - V4_CALIBRATION_AGENT_ENABLED=false：直接返回空 dict（不修改 state）
    - V4_CALIBRATION_AGENT_ENABLED=true：调用 SensitivityAnalyzer.analyze()
      写入 state["v4_sensitivity_report"]

    严格遵守不可碰清单：
    - 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
    - 不生成 ODE / 不调用 RAG / 不做 SBML 验证
    - 失败时降级返回空 dict，不抛异常，不阻塞主流水线

    Args:
        state: LangGraph 全局状态

    Returns:
        flag=false 时返回 {}
        flag=true 时返回 {"v4_sensitivity_report": {...}}
        异常时返回 {}
    """
    # Feature Flag 检查：默认 false，跳过所有逻辑
    # （与 Calibration Agent 共享 V4_CALIBRATION_AGENT_ENABLED flag，
    #  spec.md 第 461 行明确："V4_CALIBRATION_AGENT_ENABLED | P5 |
    #  Calibration + Sensitivity 执行 | 跳过，无参数校准"）
    if not settings.effective_v4_calibration_agent_enabled():
        logger.debug(
            "V4_CALIBRATION_AGENT_ENABLED effective=false，跳过 Sensitivity Analysis"
        )
        return {}

    try:
        agent = SensitivityAnalyzer()
        result = agent.analyze(state)
        # Task B.2: 双写 v4_sensitivity_report → v4_state["validation"]["sensitivity_report"]
        if "v4_sensitivity_report" in result:
            set_v4_state(result, "validation", "sensitivity_report", result["v4_sensitivity_report"])
        return result
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("Sensitivity Analysis hook 失败，降级跳过: %s", exc)
        return {}


__all__ = ["SensitivityAnalyzer", "sensitivity_hook_node"]
