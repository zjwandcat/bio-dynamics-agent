# BioDynamics Agent v4 - Validation Agent 主逻辑（Phase 5 / Task 5.9）
#
# ValidationAgent 编排 Level 1→2→3→4→5 五层验证：
#   Level 1: Internal Consistency（mass conservation / non-negative / steady state）
#   Level 2: SBML/BioModels（Track A roadrunner / Track B 结构相似度 / skipped）
#   Level 3: Cross-Pathway（crosstalk consistency / shared species conservation）
#   Level 4: Benchmark（EGFR/MAPK/NF-κB/p53/Wnt 通路特异阈值）
#   Level 5: Hypothesis Validation（P6 未启用时 skipped pass=True）
#
# 设计原则（铁律）：
# 1. Feature Flag V4_VALIDATION_PYRAMID_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
# 3. 仅消费 P1/P2/P3/P4/P5 产出（v4_ode_system / v4_reaction_ir / v4_pathway_graph /
#    v4_grounding_ledger / v4_hypothesis_list）
# 4. 失败降级：任何异常都返回 overall_pass=False，但不抛异常
# 5. 输出写入 state["v4_validation_report"]（含 level1~level5 + overall_pass + short_circuit）
# 6. overall_pass=False 时设置 pending_clarification 触发 clarification_needed SSE
#
# 对应 spec.md Part 4 Validation Pyramid（第 276-317 行）
# 对应 tasks.md SubTask 5.9.1-5.9.4
#
# 依赖：
# - app.validation_v2.level1_internal.Level1InternalValidator
# - app.validation_v2.level2_sbml.Level2SBMLValidator
# - app.validation_v2.level3_crosstalk.Level3CrossPathwayValidator
# - app.validation_v2.level4_benchmark.Level4BenchmarkValidator
# - app.validation_v2.level5_hypothesis.Level5HypothesisValidator
# - app.config.settings（Feature Flag）

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.state import set_v4_state
from app.validation_v2.level1_internal import Level1InternalValidator
from app.validation_v2.level2_sbml import Level2SBMLValidator
from app.validation_v2.level3_crosstalk import Level3CrossPathwayValidator
from app.validation_v2.level4_benchmark import Level4BenchmarkValidator
from app.validation_v2.level5_hypothesis import Level5HypothesisValidator

logger = logging.getLogger(__name__)


# =============================================================================
# ValidationAgent 主类
# =============================================================================
class ValidationAgent:
    """Validation Pyramid 编排 Agent。

    主入口 validate(state) 顺序执行 Level 1→2→3→4→5，聚合 overall_pass：
    1. Level 1 Internal Consistency：mass conservation / non-negative / steady state
    2. Level 2 SBML/BioModels：Track A roadrunner / Track B 结构相似度 / skipped
    3. Level 3 Cross-Pathway：单通路 skipped / 多通路 shared species 守恒
    4. Level 4 Benchmark：5 通路 benchmark（EGFR/MAPK/NF-κB/p53/Wnt）
    5. Level 5 Hypothesis Validation：P6 未启用 skipped pass=True

    聚合规则（对应 spec.md 第 280-317 行）：
    - overall_pass = L1.pass AND L2.pass AND L3.pass AND L4.pass AND L5.pass
    - 任一 Level pass=False → overall_pass=False（短路）
    - skipped 状态按各 Level 定义（L3/L5 单通路/P6 未启用 skipped pass=True；
      L2 skipped pass=False 修复审计 §7.2）

    失败短路信号（对应 tasks.md SubTask 5.9.3）：
    - overall_pass=False 时设置 pending_clarification（触发 clarification_needed SSE）
    - clarification context="validation_failed"，含失败 Level 与原因
    - 不阻塞 graph_v3 主流程（pending_clarification 由 Supervisor 决策）

    用法：
        agent = ValidationAgent()
        report = agent.validate(state)
        # report = {level1: {...}, level2: {...}, ..., overall_pass: bool,
        #           short_circuit: bool, failed_levels: list}
    """

    def __init__(self) -> None:
        self._level1 = Level1InternalValidator()
        self._level2 = Level2SBMLValidator()
        self._level3 = Level3CrossPathwayValidator()
        self._level4 = Level4BenchmarkValidator()
        self._level5 = Level5HypothesisValidator()

    def validate(self, state: dict[str, Any]) -> dict[str, Any]:
        """主入口：编排 Level 1→2→3→4→5，聚合 overall_pass。

        Args:
            state: LangGraph 全局状态，含：
                - v4_ode_system / v4_reaction_ir / v4_pathway_graph（Level 1 输入）
                - v4_pathway_class / sbml_model_id（Level 2 输入）
                - v4_crosstalk_edges / v4_shared_species（Level 3 输入）
                - metrics / feature_metadata（Level 4 输入）
                - v4_hypothesis_list（Level 5 输入，P6 未启用时为空）

        Returns:
            完整 v4_validation_report dict：
            {
                level1: {...},           # Level 1 报告
                level2: {...},           # Level 2 报告
                level3: {...},           # Level 3 报告
                level4: {...},           # Level 4 报告
                level5: {...},           # Level 5 报告
                overall_pass: bool,      # 全部 Level pass 的逻辑与
                short_circuit: bool,     # overall_pass=False 时为 True
                failed_levels: list[str],# pass=False 的 Level 名称列表
                agent_version: str       # Agent 版本标识
            }
            异常时返回 overall_pass=False（铁律 #4），不抛异常。
        """
        # 收集已有 level 报告（避免覆盖 Level hook 已经写入的部分）
        existing_report: dict[str, Any] = {}
        if isinstance(state, dict):
            existing = state.get("v4_validation_report")
            if isinstance(existing, dict):
                existing_report = existing

        report: dict[str, Any] = {
            "agent_version": "ValidationAgent-v1.0",
        }

        # 顺序执行 5 层验证（每层独立 try-except，单层失败不影响其他层）
        # Level 4 需要仿真 metrics（N8 输出），若 metrics 未计算则 skipped pass=True
        report["level1"] = self._run_level("level1", self._level1, state, existing_report)
        report["level2"] = self._run_level("level2", self._level2, state, existing_report)
        report["level3"] = self._run_level("level3", self._level3, state, existing_report)
        report["level4"] = self._run_level4(state, existing_report)
        report["level5"] = self._run_level("level5", self._level5, state, existing_report)

        # 聚合 overall_pass（任一 Level pass=False → overall_pass=False）
        failed_levels = []
        overall_pass = True
        for level_name in ("level1", "level2", "level3", "level4", "level5"):
            level_report = report.get(level_name) or {}
            level_pass = bool(level_report.get("pass", False))
            if not level_pass:
                overall_pass = False
                failed_levels.append(level_name)

        report["overall_pass"] = overall_pass
        report["short_circuit"] = not overall_pass
        report["failed_levels"] = failed_levels

        return report

    def build_clarification_signal(self, report: dict[str, Any]) -> dict[str, Any] | None:
        """根据验证报告构造 pending_clarification 信号（触发 clarification_needed SSE）。

        对应 tasks.md SubTask 5.9.3：失败短路信号。
        overall_pass=True 时返回 None（无需 clarification）；
        overall_pass=False 时返回 pending_clarification dict。

        Args:
            report: validate() 返回的完整验证报告

        Returns:
            None（无需短路）或 pending_clarification dict：
            {
                context: "validation_failed",
                question: str,                      # 失败原因摘要
                options: list[dict],                # 用户可选选项
                failed_levels: list[str],           # 失败 Level 列表
                validation_report_summary: dict     # 各 Level pass 状态摘要
            }
        """
        if report.get("overall_pass", False):
            return None

        failed_levels = report.get("failed_levels", []) or []
        summary = {
            lvl: bool((report.get(lvl) or {}).get("pass", False))
            for lvl in ("level1", "level2", "level3", "level4", "level5")
        }

        # 构造失败原因摘要
        reasons = []
        for lvl in failed_levels:
            lvl_report = report.get(lvl) or {}
            reason = lvl_report.get("reason") or lvl_report.get("method") or "unknown"
            reasons.append(f"{lvl}({reason})")

        question = (
            f"v4 Validation Pyramid 验证失败：{', '.join(reasons)}。"
            "是否继续生成报告（标记 low_confidence），或停止本次仿真？"
        )

        return {
            "context": "validation_failed",
            "question": question,
            "options": [
                {
                    "key": "continue",
                    "label": "继续生成报告（标记 low_confidence）",
                    "description": "保留验证失败标记，继续生成报告供人工 review",
                },
                {
                    "key": "stop",
                    "label": "停止本次仿真",
                    "description": "终止当前仿真流程，不生成报告",
                },
            ],
            "failed_levels": failed_levels,
            "validation_report_summary": summary,
        }

    # =========================================================================
    # 内部辅助：执行单层验证（含异常降级）
    # =========================================================================
    def _run_level(
        self,
        level_name: str,
        validator: Any,
        state: dict[str, Any],
        existing_report: dict[str, Any],
    ) -> dict[str, Any]:
        """执行单层验证，异常时降级返回 pass=False 报告。

        Args:
            level_name: Level 名称（"level1" / "level2" / ...）
            validator: 对应 Level 验证器实例
            state: LangGraph 全局状态
            existing_report: 已有 v4_validation_report（优先复用，避免重复执行）

        Returns:
            Level 报告 dict（含 pass 字段）
        """
        # 优先复用已有报告（避免 hook 已执行时重复计算）
        existing_level = existing_report.get(level_name)
        if isinstance(existing_level, dict) and existing_level:
            return existing_level

        try:
            return validator.validate(state)
        except Exception as exc:
            logger.warning(
                "ValidationAgent %s 异常，降级 pass=False: %s", level_name, exc
            )
            return {
                "pass": False,
                "skipped": False,
                "reason": f"{level_name}_exception: {exc}",
            }

    def _run_level4(
        self, state: dict[str, Any], existing_report: dict[str, Any]
    ) -> dict[str, Any]:
        """执行 Level 4 Benchmark 验证。

        Level 4 需要仿真 metrics（N8 scientific_features 输出）。
        当 metrics 未计算（Pyramid hook 在 worker_report 前注入）时，
        自动 skipped pass=True，不阻塞流水线。

        Args:
            state: LangGraph 全局状态
            existing_report: 已有 v4_validation_report

        Returns:
            Level 4 报告 dict
        """
        # 优先复用已有报告
        existing_level = existing_report.get("level4")
        if isinstance(existing_level, dict) and existing_level:
            return existing_level

        # 检查 metrics 是否可用（N8 输出）
        metrics = state.get("metrics") if isinstance(state, dict) else None
        if not isinstance(metrics, dict) or not metrics:
            # TD-017 修复（硬门）：metrics 未计算 → pass=False（不再 pass=True 软门放水）
            logger.debug(
                "Level 4 Benchmark skipped: metrics 未计算（N8 未执行）→ pass=False（硬门）"
            )
            return {
                "pass": False,
                "skipped": True,
                "reason": "metrics_not_computed_yet",
                "benchmarks": [],
                "method": "skipped_no_metrics",
            }

        try:
            return self._level4.validate(state)
        except Exception as exc:
            logger.warning(
                "ValidationAgent level4 异常，降级 pass=False: %s", exc
            )
            return {
                "pass": False,
                "skipped": False,
                "reason": f"level4_exception: {exc}",
            }


# =============================================================================
# LangGraph hook 节点（Feature Flag 隔离）
# =============================================================================
def validation_pyramid_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：Validation Pyramid 编排 hook。

    行为：
    - V4_VALIDATION_PYRAMID_ENABLED=false：直接返回空 dict（不修改 state）
    - V4_VALIDATION_PYRAMID_ENABLED=true：调用 ValidationAgent.validate()
      写入 state["v4_validation_report"]；overall_pass=False 时设置
      pending_clarification 触发 clarification_needed SSE

    严格遵守不可碰清单：
    - 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
    - 不生成 ODE / 不调用 RAG / 不做 SBML 验证（仅编排验证器）
    - 失败时降级返回空 dict，不抛异常，不阻塞主流水线

    Args:
        state: LangGraph 全局状态

    Returns:
        flag=false 时返回 {}
        flag=true overall_pass=True 时返回 {"v4_validation_report": {...}}
        flag=true overall_pass=False 时返回 {"v4_validation_report": {...},
                                            "pending_clarification": {...}}
        异常时返回 {}
    """
    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not settings.effective_v4_validation_pyramid_enabled():
        logger.debug("V4_VALIDATION_PYRAMID_ENABLED effective=false，跳过 Validation Pyramid")
        return {}

    try:
        agent = ValidationAgent()
        report = agent.validate(state)

        update: dict[str, Any] = {}
        # Task B.2: 双写 v4_validation_report → v4_state["validation"]["report"]
        set_v4_state(update, "validation", "report", report)

        # 失败短路：overall_pass=False 时设置 pending_clarification + 硬门字段
        if not report.get("overall_pass", False):
            clarification = agent.build_clarification_signal(report)
            if clarification is not None:
                update["pending_clarification"] = clarification
                logger.info(
                    "Validation Pyramid 短路：overall_pass=False, failed_levels=%s",
                    report.get("failed_levels", []),
                )
            # TD-019 (IB-058) 修复：整个验证金字塔为软门不阻断 ——
            # 原 overall_pass=False 仅设置 pending_clarification（软信号），
            # 下游消费者无法显式查询硬门失败状态。新增硬门字段：
            # - validation_hard_gate_failed: True（硬门失败标志，可查询）
            # - validation_failed_levels: 失败 Level 列表（供下游决策）
            # - validation_block_reason: 失败原因标识（固定 "validation_pyramid_failed"）
            # 注意：不抛异常（避免破坏 graph），仅显式标记失败状态供下游消费。
            update["validation_hard_gate_failed"] = True
            update["validation_failed_levels"] = report.get("failed_levels", [])
            update["validation_block_reason"] = "validation_pyramid_failed"
            logger.info(
                "TD-019: Validation Pyramid 硬门失败，"
                "validation_hard_gate_failed=True, failed_levels=%s",
                update["validation_failed_levels"],
            )

        return update
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("Validation Pyramid hook 失败，降级跳过: %s", exc)
        return {}


__all__ = ["ValidationAgent", "validation_pyramid_hook_node"]
