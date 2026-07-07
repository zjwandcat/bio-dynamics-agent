# BioDynamics Agent v4 - Sensitivity Planner（Phase 6 / Task 6.4）
#
# SensitivityPlanner 子组件：规划灵敏度分析的范围与目标。
#
# 职责（spec.md Part 5 第 387-391 行）：
# - 规划灵敏度分析的范围与目标
# - 输入：假设列表 + v4_pathway_graph（识别关键节点）
# - 输出：sensitivity_plan：{target_params, method: "sobol"|"morris"|"local",
#   sample_size}
# - 依赖：P5 Sensitivity Analysis 框架
#
# 设计原则（铁律）：
# 1. 不修改 v3 任何字段；仅消费 hypothesis list + v4_pathway_graph
# 2. 不调用 LLM；不调用 RAG；纯规则匹配
# 3. 失败降级：任何异常返回默认 plan（local 方法 + sample_size=100）
# 4. 不修改假设字段（只读 + 输出 plan）
# 5. method 选择规则：假设数 ≥3 → sobol；假设数 1-2 → morris；无假设 → local
#
# 对应 spec.md Part 5 Sensitivity Planner（第 387-391 行）

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# 默认参数与阈值
# =============================================================================
# 灵敏度分析方法枚举
METHOD_LOCAL: str = "local"
METHOD_MORRIS: str = "morris"
METHOD_SOBOL: str = "sobol"

# 假设数量 → 方法映射
# 假设数 ≥ 3 → sobol（全局灵敏度，计算开销大但信息丰富）
# 假设数 1-2 → morris（路径灵敏度，开销中等）
# 假设数 0 → local（局部灵敏度，开销最小）
_SOBOLE_MIN_HYPOTHESES: int = 3
_MORRIS_MIN_HYPOTHESES: int = 1

# 默认采样规模
# local: 1（仅 baseline 偏导）
# morris: 10（轨迹数）
# sobol: 100（采样数）
_DEFAULT_SAMPLE_SIZE: dict[str, int] = {
    METHOD_LOCAL: 1,
    METHOD_MORRIS: 10,
    METHOD_SOBOL: 100,
}

# 最大目标参数数（避免 plan 过大）
_MAX_TARGET_PARAMS: int = 10


# =============================================================================
# SensitivityPlanner 主类
# =============================================================================
class SensitivityPlanner:
    """规划灵敏度分析的范围与目标（spec.md Part 5 第 387-391 行）。

    主入口 plan(hypotheses, state) -> dict 输出：
    - target_params: list[str]（待分析的参数名列表）
    - method: str（"sobol" | "morris" | "local"）
    - sample_size: int（采样规模）
    - rationale: str（方法选择理由）

    依赖（spec.md 第 391 行）：
    - P5 Sensitivity Analysis 框架（v4_sensitivity_report）
    - v4_pathway_graph（识别关键节点）

    用法：
        planner = SensitivityPlanner()
        plan = planner.plan(hypotheses, state)
        # plan = {target_params, method, sample_size, rationale}
    """

    def __init__(
        self,
        max_target_params: int = _MAX_TARGET_PARAMS,
        sample_size_map: dict[str, int] | None = None,
    ) -> None:
        """初始化。

        Args:
            max_target_params: 最大目标参数数（避免 plan 过大）。
            sample_size_map: 方法 → 采样规模映射覆盖。
        """
        self._max_target_params: int = max_target_params
        self._sample_size_map: dict[str, int] = dict(_DEFAULT_SAMPLE_SIZE)
        if sample_size_map:
            self._sample_size_map.update(sample_size_map)

    # =========================================================================
    # 主入口：plan
    # =========================================================================
    def plan(
        self,
        hypotheses: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """规划灵敏度分析的范围与目标。

        Args:
            hypotheses: 假设列表（含 target_param / feedback_node / threshold_node）
            state: LangGraph 全局状态，含 v4_pathway_graph /
                v4_sensitivity_report / v4_calibration_result

        Returns:
            sensitivity_plan dict（含 target_params / method / sample_size /
            rationale 字段）。失败时返回降级 plan。
        """
        try:
            if not isinstance(hypotheses, list) or not isinstance(state, dict):
                return self._default_plan()

            # 1. 提取目标参数（从假设 + pathway_graph + sensitivity_report）
            target_params = self._collect_target_params(hypotheses, state)

            # 2. 选择方法（基于假设数量）
            method = self._select_method(len(hypotheses), len(target_params))

            # 3. 确定采样规模
            sample_size = self._select_sample_size(method, len(target_params))

            # 4. 构造方法选择理由
            rationale = self._build_rationale(
                method, len(hypotheses), len(target_params)
            )

            plan = {
                "target_params": target_params,
                "method": method,
                "sample_size": sample_size,
                "rationale": rationale,
                "n_hypotheses": len(hypotheses),
                "n_target_params": len(target_params),
            }
            logger.info(
                "SensitivityPlanner: 生成灵敏度分析 plan (method=%s, "
                "n_params=%d, sample_size=%d)",
                method,
                len(target_params),
                sample_size,
            )
            return plan
        except Exception as exc:
            logger.warning(
                "SensitivityPlanner.plan 失败，降级返回默认 plan: %s",
                exc,
            )
            return self._default_plan()

    # =========================================================================
    # 目标参数收集
    # =========================================================================
    def _collect_target_params(
        self,
        hypotheses: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> list[str]:
        """收集待分析的参数列表。

        来源优先级：
        1. 假设中的 target_param（去重）
        2. v4_sensitivity_report.local_sensitivity 中 top-K 高灵敏度参数
        3. v4_pathway_graph 中的关键节点（feedback/threshold 节点对应的参数）

        Returns:
            参数名列表（去重，最多 max_target_params 个）
        """
        params: list[str] = []
        seen: set[str] = set()

        # 1. 从假设提取 target_param
        for hyp in hypotheses:
            if not isinstance(hyp, dict):
                continue
            target_param = hyp.get("target_param")
            if isinstance(target_param, str) and target_param.strip():
                param = target_param.strip()
                if param not in seen:
                    seen.add(param)
                    params.append(param)
            if len(params) >= self._max_target_params:
                return params

        # 2. 从 v4_sensitivity_report 提取 top-K 高灵敏度参数
        sensitivity_report = state.get("v4_sensitivity_report") or {}
        if isinstance(sensitivity_report, dict):
            local_sens = sensitivity_report.get("local_sensitivity") or {}
            if isinstance(local_sens, dict):
                # 按 |sensitivity| 降序排序
                sorted_params = sorted(
                    local_sens.items(),
                    key=lambda x: abs(
                        float(x[1]) if isinstance(x[1], (int, float)) else 0.0
                    ),
                    reverse=True,
                )
                for param_name, _ in sorted_params:
                    if not isinstance(param_name, str) or not param_name.strip():
                        continue
                    param = param_name.strip()
                    if param not in seen:
                        seen.add(param)
                        params.append(param)
                    if len(params) >= self._max_target_params:
                        return params

        # 3. 从 v4_pathway_graph 提取关键节点（feedback / threshold）
        pathway_graph = state.get("v4_pathway_graph") or {}
        if isinstance(pathway_graph, dict):
            # feedback_loops 中的节点
            feedback_loops = pathway_graph.get("feedback_loops") or []
            if isinstance(feedback_loops, list):
                for loop in feedback_loops:
                    if not isinstance(loop, dict):
                        continue
                    # feedback_loop 通常含 source / target 字段
                    for key in ("source", "target", "node"):
                        node = loop.get(key)
                        if isinstance(node, str) and node.strip():
                            # 构造参数名：k_<node>_act / k_<node>_inact
                            for suffix in ("_act", "_inact", "_phos", "_dephos"):
                                param = f"k_{node}{suffix}"
                                if param not in seen:
                                    seen.add(param)
                                    params.append(param)
                                if len(params) >= self._max_target_params:
                                    return params

        return params

    # =========================================================================
    # 方法选择
    # =========================================================================
    def _select_method(
        self,
        n_hypotheses: int,
        n_target_params: int,
    ) -> str:
        """选择灵敏度分析方法（基于假设数量）。

        规则：
        - 假设数 ≥ 3 → sobol（全局灵敏度，计算开销大但信息丰富）
        - 假设数 1-2 → morris（路径灵敏度，开销中等）
        - 假设数 0 → local（局部灵敏度，开销最小）

        特殊情况：n_target_params=0 → local（无参数可分析）
        """
        if n_target_params == 0:
            return METHOD_LOCAL
        if n_hypotheses >= _SOBOLE_MIN_HYPOTHESES:
            return METHOD_SOBOL
        if n_hypotheses >= _MORRIS_MIN_HYPOTHESES:
            return METHOD_MORRIS
        return METHOD_LOCAL

    def _select_sample_size(
        self,
        method: str,
        n_target_params: int,
    ) -> int:
        """根据方法 + 参数数量选择采样规模。

        策略：
        - local: 固定 1
        - morris: max(10, n_params * 2)
        - sobol: max(100, n_params * 10)
        """
        base_size = self._sample_size_map.get(method, 100)
        if method == METHOD_LOCAL:
            return 1
        if method == METHOD_MORRIS:
            return max(base_size, n_target_params * 2)
        if method == METHOD_SOBOL:
            return max(base_size, n_target_params * 10)
        return base_size

    # =========================================================================
    # 辅助函数
    # =========================================================================
    def _build_rationale(
        self,
        method: str,
        n_hypotheses: int,
        n_target_params: int,
    ) -> str:
        """构造方法选择理由。"""
        if method == METHOD_SOBOL:
            return (
                f"选择 sobol 方法：假设数 {n_hypotheses} ≥ 3，"
                f"目标参数 {n_target_params} 个，需要全局灵敏度分析"
                f"（Sobol 指数可量化参数间相互作用）"
            )
        if method == METHOD_MORRIS:
            return (
                f"选择 morris 方法：假设数 {n_hypotheses} 为 1-2，"
                f"目标参数 {n_target_params} 个，路径灵敏度分析"
                f"（Morris 轨迹法开销中等，可识别非单调参数）"
            )
        if method == METHOD_LOCAL:
            if n_target_params == 0:
                return (
                    "选择 local 方法：无目标参数，仅做局部灵敏度分析"
                )
            return (
                f"选择 local 方法：假设数 {n_hypotheses} = 0，"
                f"目标参数 {n_target_params} 个，局部灵敏度分析（开销最小）"
            )
        return f"默认方法: {method}"

    def _default_plan(self) -> dict[str, Any]:
        """失败降级时返回默认 plan。"""
        return {
            "target_params": [],
            "method": METHOD_LOCAL,
            "sample_size": 1,
            "rationale": "降级默认 plan（异常或无效输入）",
            "n_hypotheses": 0,
            "n_target_params": 0,
        }


__all__ = ["SensitivityPlanner", "METHOD_LOCAL", "METHOD_MORRIS", "METHOD_SOBOL"]
