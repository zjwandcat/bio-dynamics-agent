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

            result = {
                "local_sensitivity": local_sensitivity,
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
    # 默认 model_func（placeholder：参数积近似）
    # =========================================================================
    @staticmethod
    def _default_model(params: dict[str, Any]) -> float:
        """默认 placeholder model：返回参数乘积的标量。

        生产环境应注入真实 ODE 仿真（roadrunner / scipy.solve_ivp），
        此 placeholder 仅用于接口完整性 + 测试，不保证生物学意义。
        """
        if not params:
            return 0.0
        product = 1.0
        for v in params.values():
            try:
                if isinstance(v, bool):
                    continue
                product *= float(v)
            except (TypeError, ValueError):
                continue
        return product

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
    if not getattr(settings, "V4_CALIBRATION_AGENT_ENABLED", False):
        logger.debug(
            "V4_CALIBRATION_AGENT_ENABLED=false，跳过 Sensitivity Analysis"
        )
        return {}

    try:
        agent = SensitivityAnalyzer()
        return agent.analyze(state)
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("Sensitivity Analysis hook 失败，降级跳过: %s", exc)
        return {}


__all__ = ["SensitivityAnalyzer", "sensitivity_hook_node"]
