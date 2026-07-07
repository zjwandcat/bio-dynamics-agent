# BioDynamics Agent v4 - Level 5 Hypothesis Validation (Phase 5 / Task 5.6)
#
# Level5HypothesisValidator 主类 + LangGraph hook 节点。
# 职责：假设验证（接口定义，由 P6 Hypothesis Agent 调用）。
#   消费 P6 输出的 v4_hypothesis_list + 可选实验数据，输出假设验证报告：
#   - 统计 validated / falsified 计数
#   - 收集 evidence_support（文献 PMID + 实验数据支持）
#   - 无任何支持的假设标记 low_confidence（不阻塞，仅写入报告）
#
# 设计原则（铁律）：
# 1. Feature Flag V4_VALIDATION_PYRAMID_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
# 3. 仅消费 P6 产出（v4_hypothesis_list）+ 用户提供的 experimental_data
# 4. 失败降级：任何异常都返回 pass=False，但不抛异常
# 5. 输出写入 state["v4_validation_report"]["level5"]（新增 v4 字段）
# 6. P6 未启用时（V4_HYPOTHESIS_AGENT_ENABLED=false）Level 5 自动 skipped（pass=True）
#
# 对应 spec.md Part 4 Level 5（第 313-317 行）
# - 输入：P6 Hypothesis Agent 输出的 v4_hypothesis_list + 实验数据（用户提供，可选）
# - 输出：v4_validation_report.level5: {pass, hypotheses_validated,
#         hypotheses_falsified, evidence_support}
# - 失败策略：假设无任何文献支持且无实验数据 → 标记 low_confidence，不阻塞但写入报告
# - 回滚策略：P6 未启用时 Level 5 自动 skipped（pass=True）
#
# 依赖：
# - app.config.settings（Feature Flag）
#
# 注意：本模块是接口定义，由 P6 Hypothesis Agent 调用。P6 当前未实现，
#       V4_HYPOTHESIS_AGENT_ENABLED 默认 false，Level 5 自动 skipped。

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Level5HypothesisValidator 主类
# =============================================================================
class Level5HypothesisValidator:
    """Level 5 Hypothesis Validation 验证器（接口定义，由 P6 调用）。

    主入口 validate(state) 执行假设验证流程：
    1. 检查 P6 是否启用（V4_HYPOTHESIS_AGENT_ENABLED）
       - 未启用 → 返回 skipped（pass=True，不阻塞）
       - 启用但 v4_hypothesis_list 为空 → 返回 skipped（pass=True）
    2. 遍历每个假设，检查文献支持（supporting_pmids）与实验数据支持
    3. 聚合 validated / falsified 计数
    4. 无任何支持的假设 → low_confidence=True（不阻塞，pass=True）
    5. 被实验数据证伪的假设 → hypotheses_falsified += 1（不阻塞，pass=True）

    失败策略（对应 spec.md 第 316 行）：
    - 假设无任何文献支持且无实验数据 → 标记 low_confidence，不阻塞但写入报告

    回滚策略（对应 spec.md 第 317 行）：
    - P6 未启用时 Level 5 自动 skipped（pass=True）

    用法：
        validator = Level5HypothesisValidator()
        report = validator.validate(state)
        # report = {pass, hypotheses_validated, hypotheses_falsified,
        #           evidence_support, low_confidence}
    """

    def validate(self, state: dict[str, Any]) -> dict[str, Any]:
        """主入口：执行 Level 5 Hypothesis Validation。

        Args:
            state: LangGraph 全局状态，含：
                - v4_hypothesis_list: P6 输出的假设列表，每条含：
                    - hypothesis_id: str
                    - statement: str（假设陈述）
                    - supporting_pmids: list[str]（文献支持，可选）
                    - falsifying_pmids: list[str]（文献证伪，可选）
                - experimental_data: 用户提供的实验数据 dict，可选，含：
                    - validated_hypothesis_ids: list[str]（实验支持的假设 ID）
                    - falsified_hypothesis_ids: list[str]（实验证伪的假设 ID）
                    - observations: list[dict]（实验观测，可选）

        Returns:
            Level 5 报告 dict（对应 spec.md 第 315 行）：
            {
                pass: bool,                       # 始终 True（不阻塞，spec 要求）
                hypotheses_validated: int,        # 已验证假设数
                hypotheses_falsified: int,        # 已证伪假设数
                evidence_support: list,           # 每条假设的支持证据
                low_confidence: bool,             # 是否有低置信度假设
                skipped: bool (可选),             # P6 未启用或无假设时为 True
                reason: str (可选)                # skipped 原因
            }
            异常时返回 pass=False（铁律 #5），不抛异常。
        """
        try:
            if not isinstance(state, dict):
                return self._build_failure_report(reason="invalid_state_type")

            # P6 未启用 → 自动 skipped（pass=True）
            if not self._is_p6_enabled(state):
                return self._run_skipped("P6_hypothesis_agent_not_enabled")

            # P6 启用但 v4_hypothesis_list 为空 → skipped（pass=True）
            hypotheses = self._extract_hypothesis_list(state)
            if not hypotheses:
                return self._run_skipped("empty_hypothesis_list")

            experimental_data = self._extract_experimental_data(state)

            # 遍历所有假设，聚合 validated/falsified 计数
            return self._validate_hypothesis_list(hypotheses, experimental_data)
        except Exception as exc:
            # 铁律 #5：失败降级返回 pass=False，但不抛异常
            logger.warning(
                "Level5HypothesisValidator.validate 失败，降级 pass=False: %s", exc
            )
            return self._build_failure_report(
                reason=f"validation_exception: {exc}"
            )

    # =========================================================================
    # SubTask 5.6.2: 接口定义 - 单个假设支持检查
    # =========================================================================
    def _check_hypothesis_support(
        self, hypothesis: dict[str, Any], experimental_data: dict[str, Any]
    ) -> dict[str, Any]:
        """检查单个假设的文献支持与实验数据支持。

        检查策略：
        1. 从 hypothesis.supporting_pmids 提取文献支持（PMID 列表）
        2. 从 experimental_data.validated_hypothesis_ids 检查是否实验支持
        3. 从 experimental_data.falsified_hypothesis_ids 检查是否实验证伪
        4. 聚合 evidence_support 列表（含文献 + 实验证据）

        判定规则：
        - validated=True：有文献支持（supporting_pmids 非空）OR 实验数据支持
        - falsified=True：实验数据证伪（优先级高于 validated）
        - low_confidence=True：无任何文献支持 AND 无实验数据支持

        Args:
            hypothesis: 单个假设 dict，含：
                - hypothesis_id: str
                - statement: str
                - supporting_pmids: list[str]（可选）
                - falsifying_pmids: list[str]（可选）
            experimental_data: 实验数据 dict，含：
                - validated_hypothesis_ids: list[str]
                - falsified_hypothesis_ids: list[str]
                - observations: list[dict]

        Returns:
            单个假设的验证结果 dict：
            {
                hypothesis_id: str,
                validated: bool,         # 是否被支持
                falsified: bool,         # 是否被证伪
                evidence_support: list,  # 支持证据列表
                low_confidence: bool     # 是否低置信度
            }
        """
        hypothesis_id = self._get_field(hypothesis, "hypothesis_id", "")
        statement = self._get_field(hypothesis, "statement", "")
        supporting_pmids = self._get_field(hypothesis, "supporting_pmids", []) or []
        falsifying_pmids = self._get_field(hypothesis, "falsifying_pmids", []) or []

        # 提取实验数据支持
        validated_ids = (
            self._get_field(experimental_data, "validated_hypothesis_ids", []) or []
        )
        falsified_ids = (
            self._get_field(experimental_data, "falsified_hypothesis_ids", []) or []
        )

        # 收集 evidence_support（文献 + 实验）
        evidence_support: list[dict[str, Any]] = []

        # 文献支持（supporting_pmids）
        has_literature_support = False
        for pmid in supporting_pmids:
            if isinstance(pmid, str) and pmid.strip():
                evidence_support.append({
                    "type": "literature",
                    "source": pmid,
                    "support": "supporting",
                    "hypothesis_id": hypothesis_id,
                })
                has_literature_support = True

        # 文献证伪（falsifying_pmids）—— 记录但不直接判定 falsified
        # 实验数据证伪优先级更高，文献证伪仅作为参考证据
        has_literature_falsification = False
        for pmid in falsifying_pmids:
            if isinstance(pmid, str) and pmid.strip():
                evidence_support.append({
                    "type": "literature",
                    "source": pmid,
                    "support": "falsifying",
                    "hypothesis_id": hypothesis_id,
                })
                has_literature_falsification = True

        # 实验数据支持
        has_experimental_support = False
        if hypothesis_id and hypothesis_id in validated_ids:
            evidence_support.append({
                "type": "experimental",
                "source": "user_provided_data",
                "support": "supporting",
                "hypothesis_id": hypothesis_id,
            })
            has_experimental_support = True

        # 实验数据证伪（优先级最高）
        is_falsified = False
        if hypothesis_id and hypothesis_id in falsified_ids:
            evidence_support.append({
                "type": "experimental",
                "source": "user_provided_data",
                "support": "falsifying",
                "hypothesis_id": hypothesis_id,
            })
            is_falsified = True

        # 判定 validated / low_confidence
        is_validated = has_literature_support or has_experimental_support
        is_low_confidence = (
            not has_literature_support
            and not has_experimental_support
            and not is_falsified
        )

        return {
            "hypothesis_id": hypothesis_id,
            "statement": statement,
            "validated": is_validated,
            "falsified": is_falsified,
            "evidence_support": evidence_support,
            "low_confidence": is_low_confidence,
        }

    # =========================================================================
    # SubTask 5.6.2: 接口定义 - 假设列表聚合验证
    # =========================================================================
    def _validate_hypothesis_list(
        self, hypotheses: list[Any], experimental_data: dict[str, Any]
    ) -> dict[str, Any]:
        """遍历所有假设，聚合 validated/falsified 计数。

        聚合规则：
        - hypotheses_validated = sum(validated=True 且 falsified=False)
        - hypotheses_falsified = sum(falsified=True)
        - evidence_support = 所有假设的 evidence_support 拼接
        - low_confidence = 任一假设 low_confidence=True
        - pass=True（spec 要求：low_confidence / falsified 都不阻塞）

        失败策略（对应 spec.md 第 316 行）：
        - 假设无任何文献支持且无实验数据 → low_confidence=True，不阻塞（pass=True）
        - 假设被实验数据证伪 → hypotheses_falsified += 1，不阻塞（pass=True）

        Args:
            hypotheses: 假设列表（每项为 dict 或对象）
            experimental_data: 实验数据 dict

        Returns:
            Level 5 报告 dict：
            {
                pass: True,                       # 始终 True（不阻塞）
                hypotheses_validated: int,
                hypotheses_falsified: int,
                evidence_support: list,
                low_confidence: bool
            }
        """
        hypotheses_validated = 0
        hypotheses_falsified = 0
        evidence_support: list[dict[str, Any]] = []
        low_confidence = False

        for hypothesis in hypotheses:
            # 兼容 dict / 对象
            hyp_dict = self._to_dict(hypothesis)
            if not hyp_dict:
                continue

            result = self._check_hypothesis_support(hyp_dict, experimental_data)

            if result.get("falsified"):
                # 被实验数据证伪 → hypotheses_falsified += 1
                hypotheses_falsified += 1
            elif result.get("validated"):
                # 被支持 → hypotheses_validated += 1
                hypotheses_validated += 1

            # 聚合 evidence_support
            ev_list = result.get("evidence_support", []) or []
            evidence_support.extend(ev_list)

            # 任一假设 low_confidence=True → 整体 low_confidence=True
            if result.get("low_confidence"):
                low_confidence = True

        # spec 要求：low_confidence / falsified 都不阻塞（pass=True）
        return {
            "pass": True,
            "hypotheses_validated": hypotheses_validated,
            "hypotheses_falsified": hypotheses_falsified,
            "evidence_support": evidence_support,
            "low_confidence": low_confidence,
        }

    # =========================================================================
    # SubTask 5.6.3: P6 未启用 skipped
    # =========================================================================
    def _is_p6_enabled(self, state: dict[str, Any]) -> bool:
        """检查 P6 Hypothesis Agent 是否启用。

        检查 V4_HYPOTHESIS_AGENT_ENABLED flag（P6 的 flag，当前默认 False）。
        P6 未实现，flag 默认 False，Level 5 自动 skipped。

        Args:
            state: LangGraph 全局状态（保留参数，预留扩展）

        Returns:
            True 表示 P6 已启用；False 表示未启用
        """
        return bool(settings.effective_v4_hypothesis_enabled())

    def _run_skipped(self, reason: str) -> dict[str, Any]:
        """P6 未启用或无假设时 skipped 状态。

        spec.md 第 317 行：P6 未启用时 Level 5 自动 skipped（pass=True）。
        与 Level 3 单通路 skipped 一致：skipped pass=True 不阻塞流水线。

        Args:
            reason: skipped 原因（记录到报告）

        Returns:
            Skipped 报告 dict：
            {
                pass: True,                          # skipped pass=True
                hypotheses_validated: 0,
                hypotheses_falsified: 0,
                evidence_support: [],
                skipped: True,
                reason: "P6_hypothesis_agent_not_enabled"
                       | "empty_hypothesis_list"
            }
        """
        return {
            "pass": True,  # spec 要求 skipped pass=True（第 317 行）
            "hypotheses_validated": 0,
            "hypotheses_falsified": 0,
            "evidence_support": [],
            "low_confidence": False,
            "skipped": True,
            "reason": reason,
        }

    # =========================================================================
    # 辅助函数：提取输入字段
    # =========================================================================
    def _extract_hypothesis_list(self, state: dict[str, Any]) -> list[Any]:
        """从 state 提取 v4_hypothesis_list。

        优先从 state["v4_hypothesis_list"] 取（P6 输出）；
        缺失或非列表时返回空列表。
        """
        hypotheses = state.get("v4_hypothesis_list")
        if isinstance(hypotheses, list):
            return hypotheses
        return []

    def _extract_experimental_data(self, state: dict[str, Any]) -> dict[str, Any]:
        """从 state 提取 experimental_data（用户提供的实验数据，可选）。

        缺失或非 dict 时返回空 dict（视为无实验数据支持）。
        """
        experimental_data = state.get("experimental_data")
        if isinstance(experimental_data, dict):
            return experimental_data
        return {}

    @staticmethod
    def _get_field(obj: Any, field: str, default: Any = None) -> Any:
        """从 dict 或对象提取字段值。"""
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

    @staticmethod
    def _to_dict(obj: Any) -> dict[str, Any]:
        """将 dict 或对象转为 dict（兼容对象输入）。"""
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            return vars(obj)
        return {}

    def _build_failure_report(self, reason: str) -> dict[str, Any]:
        """构造失败降级报告（pass=False）。

        失败降级时 pass=False（阻塞流水线，铁律 #5），与 skipped 不同。
        skipped 是 P6 未启用/无假设时的正常跳过，pass=True。
        """
        return {
            "pass": False,
            "hypotheses_validated": 0,
            "hypotheses_falsified": 0,
            "evidence_support": [],
            "low_confidence": False,
            "reason": reason,
        }


# =============================================================================
# LangGraph hook 节点（Feature Flag 隔离）
# =============================================================================
def level5_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：Level 5 Hypothesis Validation hook。

    行为：
    - V4_VALIDATION_PYRAMID_ENABLED=false：直接返回空 dict（不修改 state）
    - V4_VALIDATION_PYRAMID_ENABLED=true：调用 Level5HypothesisValidator.validate()
      写入 state["v4_validation_report"]["level5"]
    - P6 未启用（V4_HYPOTHESIS_AGENT_ENABLED=false）时 validate() 内部返回
      skipped（pass=True），hook 仍写入 level5 报告（包含 skipped 字段）

    严格遵守不可碰清单：
    - 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
    - 不生成 ODE / 不调用 RAG / 不做 SBML 验证
    - 失败时降级返回空 dict，不抛异常，不阻塞主流水线

    Args:
        state: LangGraph 全局状态

    Returns:
        flag=false 时返回 {}
        flag=true 时返回 {"v4_validation_report": {"level5": {...}}}
        异常时返回 {}
    """
    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not settings.effective_v4_validation_pyramid_enabled():
        logger.debug("V4_VALIDATION_PYRAMID_ENABLED effective=false，跳过 Level 5 validation")
        return {}

    try:
        validator = Level5HypothesisValidator()
        level5_report = validator.validate(state)
        # 与现有 v4_validation_report 合并，不覆盖 level1/level2/level3/level4
        existing_report: dict[str, Any] = {}
        if isinstance(state, dict):
            existing = state.get("v4_validation_report")
            if isinstance(existing, dict):
                existing_report = existing
        merged_report = {**existing_report, "level5": level5_report}
        return {"v4_validation_report": merged_report}
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("Level 5 validation hook 失败，降级跳过: %s", exc)
        return {}


__all__ = ["Level5HypothesisValidator", "level5_hook_node"]
