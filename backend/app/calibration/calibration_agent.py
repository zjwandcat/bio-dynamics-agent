# BioDynamics Agent v4 - Calibration Agent (Phase 5 / Task 5.7.1)
#
# CalibrationAgent 主类 + LangGraph hook 节点。
# 职责：用 BioModels reference 或用户实验数据拟合 ODE 系统参数，
# 输出 calibrated_params + confidence_intervals + uncalifiable。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_CALIBRATION_AGENT_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
# 3. 仅消费 P1/P2/P3 产出（v4_ode_system / v4_reaction_ir / state.parameters）
# 4. 失败降级：任何异常都返回空更新，不阻塞主流水线
# 5. 输出写入 state["v4_calibration_result"]（新增 v4 字段，与 v3 字段共存）
# 6. 依赖隔离：lmfit 不可用时降级到 scipy.optimize.least_squares
#
# 对应 spec.md Part 4 Calibration Agent（第 335-340 行）
# - 输入：v4_ode_system / state.parameters / experimental_data（可选）
# - 输出：v4_calibration_result: {calibrated_params, confidence_intervals,
#         uncalifiable, method, agent_version, warnings}
# - 失败策略：拟合失败 → 标记 uncalibrated=True（不阻塞流水线）
#
# 依赖：
# - app.config.settings（Feature Flag）
# - app.calibration.least_squares_fitter.LeastSquaresFitter / FitResult
# - app.calibration.confidence_interval.ConfidenceIntervalEstimator

from __future__ import annotations

import logging
from typing import Any

from app.calibration.confidence_interval import ConfidenceIntervalEstimator
from app.calibration.least_squares_fitter import FitResult, LeastSquaresFitter
from app.config import settings
from app.state import set_v4_state

logger = logging.getLogger(__name__)


# =============================================================================
# CalibrationAgent 主类
# =============================================================================
class CalibrationAgent:
    """Calibration Agent 主类。

    主入口 calibrate(state) 执行参数校准流程：
    1. 提取输入：v4_ode_system / sbml_model_id / state.parameters / experimental_data
    2. 提取待拟合参数列表（从 v4_grounding_ledger 的 parameter_ids
       或 state.parameters 的 keys）
    3. 提取参考数据（experimental_data.user_data 或空）
    4. 调用 LeastSquaresFitter.fit() 拟合
    5. 调用 ConfidenceIntervalEstimator.estimate() 计算置信区间
    6. 失败参数标记 uncalibrated=True 写入 uncalifiable 列表
    7. 返回 {"v4_calibration_result": {...}}

    失败策略（spec.md 第 340 行）：
    - 拟合失败 → 标记 uncalibrated=True，不阻塞流水线

    用法：
        agent = CalibrationAgent()
        result = agent.calibrate(state)
        # result = {"v4_calibration_result": {calibrated_params, ...}}
    """

    AGENT_VERSION = "v4.0"
    # bootstrap 默认重采样次数（CI 估计用）
    DEFAULT_CI_N_SAMPLES = 100

    def __init__(
        self,
        fitter: LeastSquaresFitter | None = None,
        ci_estimator: ConfidenceIntervalEstimator | None = None,
    ) -> None:
        """依赖注入（默认创建，测试时可 mock）。

        Args:
            fitter: LeastSquaresFitter 实例（None 时创建默认）
            ci_estimator: ConfidenceIntervalEstimator 实例（None 时创建默认 95% CI）
        """
        self._fitter = fitter if fitter is not None else LeastSquaresFitter()
        self._ci_estimator = (
            ci_estimator
            if ci_estimator is not None
            else ConfidenceIntervalEstimator(confidence_level=0.95)
        )

    # =========================================================================
    # 主入口
    # =========================================================================
    def calibrate(self, state: dict[str, Any]) -> dict[str, Any]:
        """主入口：执行参数校准。

        Args:
            state: LangGraph 全局状态，含：
                - v4_ode_system: P3 输出的 ODE 系统（含 ode_code / equations / parameters）
                - sbml_model_id: BioModels 模型 ID（如 BIOMD0000000205）
                - parameters: v3 参数 dict（仅读取，不修改）
                - experimental_data: 用户提供的实验数据 dict（可选），含：
                    - user_data: {observations: list[float], ...}
                    - observations: list[float]（备选）
                - v4_grounding_ledger: P4 输出（可选，提供 parameter_ids）

        Returns:
            {"v4_calibration_result": {
                calibrated_params: dict,        # 拟合后的参数
                confidence_intervals: dict,      # 每个参数的 CI
                uncalifiable: list,             # 不可校准参数名列表
                method: "lmfit"|"least_squares",# 拟合方法
                agent_version: str,             # "v4.0"
                warnings: list                  # 警告信息
            }}
            失败降级返回空更新（仍含 v4_calibration_result 字段，标记 fallback）。
        """
        try:
            # 输入校验：state 必须 dict
            if not isinstance(state, dict):
                return self._fallback_result("invalid_state_type")

            # 提取输入
            ode_system = state.get("v4_ode_system") or {}
            sbml_model_id = state.get("sbml_model_id", "")
            parameters = state.get("parameters") or {}

            logger.info(
                "CalibrationAgent 开始校准：sbml_model_id=%s, "
                "ode_system=%s, parameters=%d",
                sbml_model_id,
                "present" if ode_system else "missing",
                len(parameters) if isinstance(parameters, dict) else 0,
            )

            # 提取待拟合参数列表
            target_params = self._extract_target_params(state)
            if not target_params:
                logger.warning(
                    "CalibrationAgent 无待拟合参数（target_params 为空），降级"
                )
                return self._fallback_result("empty_target_params")

            # 提取参考数据
            reference_data = self._extract_reference_data(state)

            # 调用 fitter 拟合
            fit_result = self._fitter.fit(target_params, reference_data)

            # 调用 CI estimator
            confidence_intervals = self._ci_estimator.estimate(
                fit_result, n_samples=self.DEFAULT_CI_N_SAMPLES
            )

            # 标记不可校准参数
            uncalifiable: list[str] = []
            warnings: list[str] = []

            if not fit_result.success:
                # 整体拟合失败 → 全部参数标记 uncalibrated
                uncalifiable.extend(target_params)
                warnings.append(
                    f"fit_failed: {fit_result.message or 'unknown'}"
                )
                # 为所有参数标记 uncalibrated
                for name in target_params:
                    if name not in confidence_intervals:
                        confidence_intervals[name] = {
                            "lower": 0.0,
                            "upper": 0.0,
                            "std_error": 0.0,
                            "method": "none",
                            "uncalibrated": True,
                        }
                    else:
                        confidence_intervals[name]["uncalibrated"] = True
            else:
                # 拟合成功但部分参数 CI 估计失败 → 标记 uncalibrated
                for name in target_params:
                    ci = confidence_intervals.get(name)
                    if ci is None or ci.get("uncalibrated") is True:
                        if name not in uncalifiable:
                            uncalifiable.append(name)
                        warnings.append(
                            f"ci_estimate_failed: {name}"
                        )

            # 构造 calibrated_params
            calibrated_params = (
                dict(fit_result.params) if fit_result.success else {}
            )

            result = {
                "calibrated_params": calibrated_params,
                "confidence_intervals": confidence_intervals,
                "uncalifiable": uncalifiable,
                "method": fit_result.method,
                "agent_version": self.AGENT_VERSION,
                "warnings": warnings,
            }

            logger.info(
                "CalibrationAgent 完成：method=%s, success=%s, "
                "calibrated=%d, uncalifiable=%d, warnings=%d",
                fit_result.method,
                fit_result.success,
                len(calibrated_params),
                len(uncalifiable),
                len(warnings),
            )

            # IB-019 修复：calibrated_params 回写 state.parameters
            # 仅在拟合成功（calibrated_params 非空）时回写，避免空 dict 覆盖原参数
            update_dict: dict[str, Any] = {"v4_calibration_result": result}
            if calibrated_params:
                update_dict["parameters"] = calibrated_params
            return update_dict

        except Exception as exc:
            logger.warning(
                "CalibrationAgent.calibrate 失败，降级返回空结果: %s", exc
            )
            return self._fallback_result(f"calibrate_exception: {exc}")

    # =========================================================================
    # 输入提取辅助方法
    # =========================================================================
    def _extract_target_params(self, state: dict[str, Any]) -> list[str]:
        """提取待拟合参数列表。

        优先级：
        1. v4_grounding_ledger 中的 parameter_ids（去重，P4 输出）
        2. state.parameters 的 keys（v3 参数名）
        3. v4_ode_system.parameters 的 keys（P3 输出，备选）

        Args:
            state: LangGraph 全局状态

        Returns:
            去重后的参数名列表（保持插入顺序）
        """
        seen: set[str] = set()
        result: list[str] = []

        # 1. v4_grounding_ledger.parameter_ids（可能来自 ode_equations[].parameter_ids）
        ledger = state.get("v4_grounding_ledger")
        if isinstance(ledger, dict):
            ode_equations = ledger.get("ode_equations")
            if isinstance(ode_equations, list):
                for eq in ode_equations:
                    if not isinstance(eq, dict):
                        continue
                    param_ids = eq.get("parameter_ids")
                    if isinstance(param_ids, list):
                        for p in param_ids:
                            if isinstance(p, str) and p and p not in seen:
                                seen.add(p)
                                result.append(p)

        # 2. state.parameters 的 keys
        parameters = state.get("parameters")
        if isinstance(parameters, dict):
            for key in parameters.keys():
                if isinstance(key, str) and key and key not in seen:
                    seen.add(key)
                    result.append(key)

        # 3. v4_ode_system.parameters 的 keys
        ode_system = state.get("v4_ode_system")
        if isinstance(ode_system, dict):
            ode_params = ode_system.get("parameters")
            if isinstance(ode_params, dict):
                for key in ode_params.keys():
                    if isinstance(key, str) and key and key not in seen:
                        seen.add(key)
                        result.append(key)

        return result

    def _extract_reference_data(self, state: dict[str, Any]) -> dict[str, Any]:
        """提取参考数据（experimental_data.user_data 或空）。

        Args:
            state: LangGraph 全局状态

        Returns:
            参考 data dict（含 observations / user_data / values 等），
            缺失时返回空 dict（fitter 会处理为空 observations → 拟合失败）。
        """
        experimental_data = state.get("experimental_data")
        if not isinstance(experimental_data, dict):
            return {}

        # 优先 user_data
        user_data = experimental_data.get("user_data")
        if isinstance(user_data, dict) and user_data:
            # 同时保留 observations 字段（fitter 兼容）
            merged = dict(user_data)
            if "observations" not in merged and "observations" in experimental_data:
                merged["observations"] = experimental_data["observations"]
            return merged

        # 直接返回 experimental_data
        return experimental_data

    # =========================================================================
    # 失败降级方法
    # =========================================================================
    def _fallback_result(self, reason: str = "") -> dict[str, Any]:
        """失败降级：返回空 calibrated_result（仍含 v4_calibration_result 字段）。

        与 grounder_agent._fallback_ledger 一致：标记 fallback=True，
        不阻塞主流水线，warnings 记录失败原因。

        Args:
            reason: 失败原因（记录到 warnings）

        Returns:
            {"v4_calibration_result": {
                calibrated_params: {},
                confidence_intervals: {},
                uncalifiable: [],
                method: "none",
                agent_version: "v4.0",
                warnings: [f"calibration_fallback: {reason}"],
                fallback: True
            }}
        """
        return {
            "v4_calibration_result": {
                "calibrated_params": {},
                "confidence_intervals": {},
                "uncalifiable": [],
                "method": "none",
                "agent_version": self.AGENT_VERSION,
                "warnings": [f"calibration_fallback: {reason}"] if reason else [],
                "fallback": True,
            }
        }


# =============================================================================
# LangGraph hook 节点（Feature Flag 隔离）
# =============================================================================
def calibration_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：Calibration Agent hook。

    行为：
    - V4_CALIBRATION_AGENT_ENABLED=false：直接返回空 dict（不修改 state）
    - V4_CALIBRATION_AGENT_ENABLED=true：调用 CalibrationAgent.calibrate()
      写入 state["v4_calibration_result"]

    严格遵守不可碰清单：
    - 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
    - 不生成 ODE / 不调用 RAG / 不做 SBML 验证
    - 失败时降级返回空 dict，不抛异常，不阻塞主流水线

    Args:
        state: LangGraph 全局状态

    Returns:
        flag=false 时返回 {}
        flag=true 时返回 {"v4_calibration_result": {...}}
        异常时返回 {}
    """
    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not settings.effective_v4_calibration_agent_enabled():
        logger.debug("V4_CALIBRATION_AGENT_ENABLED effective=false，跳过 Calibration Agent")
        return {}

    try:
        agent = CalibrationAgent()
        result = agent.calibrate(state)
        # Task B.2: 双写 v4_calibration_result → v4_state["validation"]["calibration_result"]
        if "v4_calibration_result" in result:
            set_v4_state(result, "validation", "calibration_result", result["v4_calibration_result"])
        return result
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("Calibration Agent hook 失败，降级跳过: %s", exc)
        return {}


__all__ = ["CalibrationAgent", "calibration_hook_node"]
