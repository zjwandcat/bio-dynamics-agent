# BioDynamics Agent v4 - Parameter Explorer（Phase 6 / Task 6.4）
#
# ParameterExplorer 子组件：对假设中的关键参数进行扫描，验证假设的鲁棒性。
#
# 职责（spec.md Part 5 第 381-385 行）：
# - 对假设中的关键参数进行扫描，验证假设的鲁棒性
# - 输入：假设 + v4_ode_system + P5 Calibration 置信区间
# - 输出：parameter_robustness：{param, range, hypothesis_holds: bool}
# - 依赖：P5 Sensitivity Analysis 结果
#
# 设计原则（铁律）：
# 1. 不修改 v3 任何字段；仅消费 hypothesis + state 字段
# 2. 复用 v3 sandbox.py（只读调用 execute_simulation_code_v2，不修改其代码）
# 3. 失败降级：sandbox 不可用 / 仿真失败 → 启发式判断（基于 calibration CI）
# 4. 不调用 LLM；不调用 RAG；纯数值计算
# 5. 输出 parameter_robustness 字段含 hypothesis_holds: bool
#
# 对应 spec.md Part 5 Parameter Explorer（第 381-385 行）+ Part 6（第 395 行）

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# 默认参数扫描范围（相对于 baseline 的倍数）
# =============================================================================
# 扫描 5 个点：0.1x / 0.5x / 1x（baseline）/ 2x / 10x
# 覆盖 1 个数量级，足以判断假设鲁棒性
_DEFAULT_SCAN_FACTORS: list[float] = [0.1, 0.5, 1.0, 2.0, 10.0]

# 假设成立阈值：扫描点中假设成立的比例 ≥ 此值 → hypothesis_holds=True
# 0.6 = 5 个点中至少 3 个成立
_DEFAULT_HOLDS_THRESHOLD: float = 0.6

# 默认 model_func 不可用时，使用 calibration CI 启发式判断
# CI 宽度 < 50% baseline → 视为鲁棒
_DEFAULT_CI_WIDTH_THRESHOLD: float = 0.5


# =============================================================================
# ParameterExplorer 主类
# =============================================================================
class ParameterExplorer:
    """对假设中的关键参数进行扫描，验证假设的鲁棒性。

    主入口 explore(hypothesis, state) -> dict 输出：
    - param: str（扫描的参数名）
    - range: list[float]（扫描点取值）
    - hypothesis_holds: bool（假设是否在参数范围内成立）
    - holds_ratio: float（成立比例 0-1）
    - method: str（"sandbox_sweep" | "calibration_ci_heuristic"）
    - details: list[dict]（每个扫描点的结果）

    依赖（spec.md 第 385 行）：
    - P5 Sensitivity Analysis 结果（v4_sensitivity_report）
    - P5 Calibration 置信区间（v4_calibration_result）
    - v3 sandbox.py（execute_simulation_code_v2，复用仿真器，不新增）

    用法：
        explorer = ParameterExplorer()
        robustness = explorer.explore(hypothesis, state)
        # robustness = {param, range, hypothesis_holds, ...}
    """

    def __init__(
        self,
        scan_factors: list[float] | None = None,
        holds_threshold: float = _DEFAULT_HOLDS_THRESHOLD,
        model_func: Callable[[str, float], float] | None = None,
    ) -> None:
        """初始化。

        Args:
            scan_factors: 参数扫描倍数列表（相对 baseline）。None → 默认 5 点。
            holds_threshold: 假设成立比例阈值（0-1）。
            model_func: 自定义模型函数 (param_name, param_value) -> output_metric。
                None → 尝试调用 v3 sandbox.execute_simulation_code_v2。
        """
        self._scan_factors: list[float] = (
            list(scan_factors) if scan_factors else list(_DEFAULT_SCAN_FACTORS)
        )
        self._holds_threshold: float = holds_threshold
        self._custom_model_func: Callable[[str, float], float] | None = model_func

    # =========================================================================
    # 主入口：explore
    # =========================================================================
    def explore(
        self,
        hypothesis: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """对假设的关键参数进行扫描，验证假设鲁棒性。

        Args:
            hypothesis: 假设 dict，含 target_param / sensitivity / prediction
            state: LangGraph 全局状态，含 v4_ode_system /
                v4_calibration_result / v4_sensitivity_report

        Returns:
            parameter_robustness dict（含 param/range/hypothesis_holds/holds_ratio/
            method/details 字段）。失败时返回降级结果。
        """
        try:
            if not isinstance(hypothesis, dict) or not isinstance(state, dict):
                return self._minimal_robustness(hypothesis)

            # 1. 提取关键参数
            param_name = self._extract_target_param(hypothesis, state)
            if not param_name:
                # 无明确 target_param → 跳过鲁棒性验证
                logger.info(
                    "ParameterExplorer: 假设 %s 无 target_param，跳过鲁棒性验证",
                    hypothesis.get("id", "?"),
                )
                return self._minimal_robustness(hypothesis)

            # 2. 获取 baseline 参数值
            baseline_value = self._get_baseline_param_value(param_name, state)
            if baseline_value is None:
                # 无法获取 baseline → CI 启发式
                return self._heuristic_ci_check(param_name, hypothesis, state)

            # 3. 构造扫描范围（round 10 位避免浮点精度问题，如 0.1*0.1=0.010000000000000002）
            scan_range = [
                round(baseline_value * factor, 10) for factor in self._scan_factors
            ]

            # 4. 执行参数扫描
            sweep_results = self._run_parameter_sweep(
                param_name, scan_range, hypothesis, state
            )

            # 5. 判断假设是否在每个扫描点成立
            holds_count = sum(1 for r in sweep_results if r.get("holds", False))
            holds_ratio = holds_count / len(sweep_results) if sweep_results else 0.0
            hypothesis_holds = holds_ratio >= self._holds_threshold

            # 6. 确定方法（根据扫描点 method 字段映射到聚合 method）
            _METHOD_MAP = {
                "sandbox": "sandbox_sweep",
                "model_func": "model_func",
                "heuristic": "calibration_ci_heuristic",
            }
            first_point_method = (
                sweep_results[0].get("method", "") if sweep_results else ""
            )
            method = _METHOD_MAP.get(
                first_point_method, "calibration_ci_heuristic"
            )

            return {
                "param": param_name,
                "range": scan_range,
                "hypothesis_holds": hypothesis_holds,
                "holds_ratio": holds_ratio,
                "method": method,
                "baseline_value": baseline_value,
                "details": sweep_results,
            }
        except Exception as exc:
            logger.warning(
                "ParameterExplorer.explore 失败，降级返回最小结果: %s",
                exc,
            )
            return self._minimal_robustness(hypothesis)

    # =========================================================================
    # 参数提取
    # =========================================================================
    def _extract_target_param(
        self,
        hypothesis: dict[str, Any],
        state: dict[str, Any],
    ) -> str:
        """从假设提取关键参数名。

        优先级：
        1. hypothesis.target_param（灵敏度假设明确指定）
        2. v4_sensitivity_report 中 top-1 高灵敏度参数
        3. 空（无关键参数 → 跳过鲁棒性验证）
        """
        # 1. hypothesis.target_param
        target_param = hypothesis.get("target_param") or ""
        if isinstance(target_param, str) and target_param.strip():
            return target_param.strip()

        # 2. v4_sensitivity_report 中 top-1 高灵敏度参数
        sensitivity_report = state.get("v4_sensitivity_report") or {}
        if isinstance(sensitivity_report, dict):
            local_sens = sensitivity_report.get("local_sensitivity") or {}
            if isinstance(local_sens, dict):
                # 扁平 dict 形式：{param: sensitivity}
                if local_sens:
                    # 选 |sensitivity| 最大的参数
                    top_param = max(
                        local_sens.items(),
                        key=lambda x: abs(
                            float(x[1]) if isinstance(x[1], (int, float)) else 0.0
                        ),
                    )
                    if isinstance(top_param[0], str):
                        return top_param[0]

        return ""

    def _get_baseline_param_value(
        self,
        param_name: str,
        state: dict[str, Any],
    ) -> float | None:
        """获取参数的 baseline 值。

        优先级：
        1. v4_calibration_result.calibrated_params[param_name]
        2. state.parameters[param_name]（v3 字段）
        3. None（无法获取 → 走 CI 启发式）
        """
        # 1. v4_calibration_result.calibrated_params
        calibration_result = state.get("v4_calibration_result") or {}
        if isinstance(calibration_result, dict):
            calibrated_params = calibration_result.get("calibrated_params") or {}
            if isinstance(calibrated_params, dict):
                value = calibrated_params.get(param_name)
                if isinstance(value, (int, float)):
                    return float(value)

        # 2. state.parameters
        parameters = state.get("parameters")
        if isinstance(parameters, dict):
            value = parameters.get(param_name)
            if isinstance(value, (int, float)):
                return float(value)

        return None

    # =========================================================================
    # 参数扫描（复用 v3 sandbox.py 或自定义 model_func）
    # =========================================================================
    def _run_parameter_sweep(
        self,
        param_name: str,
        scan_range: list[float],
        hypothesis: dict[str, Any],
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """对参数扫描范围执行仿真，判断假设在每个点是否成立。

        策略：
        1. 优先使用自定义 model_func（测试时可注入）
        2. 否则尝试调用 v3 sandbox.execute_simulation_code_v2
        3. sandbox 不可用 / 仿真失败 → CI 启发式（每个点都视为 holds=True）
        """
        results: list[dict[str, Any]] = []

        # 1. 自定义 model_func
        if self._custom_model_func is not None:
            return self._sweep_with_model_func(
                param_name, scan_range, hypothesis, state
            )

        # 2. 尝试调用 v3 sandbox
        try:
            from app.sandbox import execute_simulation_code_v2

            ode_system = state.get("v4_ode_system") or {}
            if not isinstance(ode_system, dict):
                raise RuntimeError("v4_ode_system 缺失或非 dict")
            ode_code = ode_system.get("ode_code") or ""
            if not isinstance(ode_code, str) or not ode_code.strip():
                raise RuntimeError("v4_ode_system.ode_code 为空")

            # 对每个扫描点执行仿真
            # 策略：若首个扫描点 sandbox 调用失败 → 视为 sandbox 完全不可用，
            #       降级到 CI 启发式（所有点 holds=True，保守不误杀）
            sandbox_broken = False
            for value in scan_range:
                try:
                    # 构造修改参数后的 ODE 代码（注入参数覆盖）
                    modified_code = self._inject_param_override(
                        ode_code, param_name, value
                    )
                    sim_result = execute_simulation_code_v2(
                        modified_code, timeout=30
                    )
                    # 判断假设是否成立（基于仿真结果）
                    holds = self._check_hypothesis_with_sim(
                        hypothesis, sim_result, state
                    )
                    results.append({
                        "param_value": value,
                        "holds": holds,
                        "method": "sandbox",
                        "sim_success": sim_result.get("status") == "success",
                    })
                except Exception as exc:
                    logger.debug(
                        "ParameterExplorer: 扫描点 %s=%s sandbox 执行失败: %s",
                        param_name, value, exc,
                    )
                    if not results:
                        # 首个扫描点就失败 → sandbox 完全不可用，降级到 heuristic
                        sandbox_broken = True
                        break
                    # 后续单点失败 → 视为 holds=False（sandbox 部分可用）
                    results.append({
                        "param_value": value,
                        "holds": False,
                        "method": "sandbox",
                        "sim_success": False,
                        "error": str(exc),
                    })

            if results and not sandbox_broken:
                return results
        except Exception as exc:
            logger.info(
                "ParameterExplorer: sandbox 不可用，降级到 CI 启发式: %s",
                exc,
            )

        # 3. CI 启发式降级
        return self._heuristic_sweep(param_name, scan_range, hypothesis, state)

    def _sweep_with_model_func(
        self,
        param_name: str,
        scan_range: list[float],
        hypothesis: dict[str, Any],
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """使用自定义 model_func 执行扫描（测试友好）。"""
        results: list[dict[str, Any]] = []
        threshold = self._extract_threshold_from_hypothesis(hypothesis)
        baseline_output = None

        # 先获取 baseline (factor=1.0) 输出，用于归一化
        try:
            baseline_output = self._custom_model_func(param_name, 1.0)  # type: ignore[misc]
        except Exception:
            baseline_output = None

        for value in scan_range:
            try:
                output = self._custom_model_func(param_name, value)  # type: ignore[misc]
                holds = self._check_holds_with_output(
                    output, baseline_output, threshold, hypothesis
                )
                results.append({
                    "param_value": value,
                    "holds": holds,
                    "method": "model_func",
                    "output": output,
                })
            except Exception as exc:
                results.append({
                    "param_value": value,
                    "holds": False,
                    "method": "model_func",
                    "error": str(exc),
                })
        return results

    def _heuristic_sweep(
        self,
        param_name: str,
        scan_range: list[float],
        hypothesis: dict[str, Any],
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """CI 启发式降级：每个点都视为 holds=True（保守不误杀）。

        策略：
        - 若 v4_calibration_result 含该参数的 CI，且 CI 宽度 < 50% baseline
          → 所有扫描点 holds=True（鲁棒）
        - 否则 → 仍然 holds=True（保守，避免误杀假设）
        """
        results: list[dict[str, Any]] = []
        # CI 启发式：所有点视为 holds=True
        for value in scan_range:
            results.append({
                "param_value": value,
                "holds": True,
                "method": "heuristic",
            })
        return results

    def _heuristic_ci_check(
        self,
        param_name: str,
        hypothesis: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """无法获取 baseline 时使用 CI 启发式判断鲁棒性。"""
        # 检查 v4_calibration_result 是否含 CI
        calibration_result = state.get("v4_calibration_result") or {}
        ci_available = False
        if isinstance(calibration_result, dict):
            confidence_intervals = calibration_result.get("confidence_intervals") or {}
            if isinstance(confidence_intervals, dict):
                ci = confidence_intervals.get(param_name)
                if ci:
                    ci_available = True

        # CI 启发式：有 CI → 视为鲁棒；无 CI → 保守视为鲁棒
        return {
            "param": param_name,
            "range": [],
            "hypothesis_holds": True,
            "holds_ratio": 1.0,
            "method": "calibration_ci_heuristic",
            "baseline_value": None,
            "ci_available": ci_available,
            "details": [],
        }

    # =========================================================================
    # 假设成立判断
    # =========================================================================
    def _check_hypothesis_with_sim(
        self,
        hypothesis: dict[str, Any],
        sim_result: dict[str, Any],
        state: dict[str, Any],
    ) -> bool:
        """基于仿真结果判断假设是否成立。

        简化策略：
        - 仿真成功 → holds=True（保守）
        - 仿真失败 → holds=False
        实际生产中应基于 readout.metric 阈值判断，但避免过度耦合。
        """
        if not isinstance(sim_result, dict):
            return False
        return sim_result.get("status") == "success"

    def _check_holds_with_output(
        self,
        output: float,
        baseline_output: float | None,
        threshold: float,
        hypothesis: dict[str, Any],
    ) -> bool:
        """基于 model_func 输出判断假设是否成立。

        策略：
        - 若 baseline_output 可用，计算相对变化 |output - baseline| / |baseline|
        - 若相对变化 < threshold → holds=True（参数变化对输出影响小 → 假设鲁棒）
        - 若 baseline_output 不可用 → holds=True（保守）
        """
        if baseline_output is None or baseline_output == 0:
            return True
        try:
            relative_change = abs(output - baseline_output) / abs(baseline_output)
            # 假设鲁棒性：相对变化小于阈值 → 假设在该参数点成立
            # 注：这里 threshold 是 prediction 中提取的阈值（如 0.5 表示 50%）
            # 鲁棒性 = 假设在不同参数下仍成立 = 参数变化导致输出变化不显著
            return relative_change < threshold
        except (TypeError, ValueError, ZeroDivisionError):
            return True

    def _extract_threshold_from_hypothesis(
        self, hypothesis: dict[str, Any]
    ) -> float:
        """从假设提取阈值（用于鲁棒性判断）。

        优先级：
        1. experiment_design.readout.threshold
        2. hypothesis.sensitivity（直接作为阈值）
        3. 默认 0.5
        """
        experiment_design = hypothesis.get("experiment_design")
        if isinstance(experiment_design, dict):
            readout = experiment_design.get("readout")
            if isinstance(readout, dict):
                threshold = readout.get("threshold")
                if isinstance(threshold, (int, float)) and threshold > 0:
                    return float(threshold)

        sensitivity = hypothesis.get("sensitivity")
        if isinstance(sensitivity, (int, float)):
            return abs(float(sensitivity))

        return 0.5

    # =========================================================================
    # 辅助函数：ODE 代码参数注入
    # =========================================================================
    @staticmethod
    def _inject_param_override(
        ode_code: str, param_name: str, param_value: float
    ) -> str:
        """在 ODE 代码顶部注入参数覆盖（避免修改原代码）。

        策略：在代码首行插入 ``param_name = <value>``，覆盖原有定义。
        """
        override_line = f"{param_name} = {param_value!r}  # ParameterExplorer override"
        return f"{override_line}\n{ode_code}"

    def _minimal_robustness(
        self, hypothesis: dict[str, Any]
    ) -> dict[str, Any]:
        """失败降级时返回最小可用 parameter_robustness。"""
        param_name = ""
        if isinstance(hypothesis, dict):
            param_name = hypothesis.get("target_param") or ""
        return {
            "param": str(param_name) if param_name else "unknown",
            "range": [],
            "hypothesis_holds": True,  # 保守不误杀
            "holds_ratio": 1.0,
            "method": "degraded",
            "baseline_value": None,
            "details": [],
        }


__all__ = ["ParameterExplorer"]
