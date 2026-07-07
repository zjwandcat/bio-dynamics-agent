# BioDynamics Agent v4 - Hypothesis Agent（Phase 6 / Task 6.1）
#
# HypothesisAgent 主 Agent：协调子组件生成 v4_hypothesis_list。
#
# 职责（spec.md Part 5 第 352-358 行）：
# - 基于仿真结果 + 科学特征 + 验证报告，生成可验证的实验假设
# - 输入：state.metrics / state.feature_metadata / v4_validation_report /
#   v4_grounding_ledger / v4_sensitivity_report / v4_pathway_class
# - 输出：v4_hypothesis_list: list[Hypothesis]，每个假设含
#   {id, statement, prediction, experiment_design, validation_method,
#    expected_result, falsifiable, supporting_pmids, contradicting_pmids}
# - 依赖：P5 Validation Report / P5 Grounding Ledger / 文献检索（复用 rag_client.py）
#
# 设计原则（铁律）：
# 1. Feature Flag V4_HYPOTHESIS_AGENT_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段；仅消费 v4_* 字段 + metrics / feature_metadata
# 3. 与 Validation 交互：仅在 v4_validation_report.overall_pass=True 时执行
#    （Validation 失败短路到错误报告，不进入 Hypothesis）
# 4. 失败降级：任何异常都返回空 v4_hypothesis_list，不阻塞报告生成
# 5. 不修改 P5 Validation Report（只读）
# 6. 假设生成失败 → 输出空列表 + warning（spec.md 第 357 行）
#
# 对应 spec.md Part 5（第 350-398 行）+ Part 6（第 393-398 行交互）

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.hypothesis.hypothesis_generator import HypothesisGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# HypothesisAgent 主类
# =============================================================================
class HypothesisAgent:
    """Hypothesis Agent 主类（spec.md Part 5 第 352-358 行）。

    主入口 generate(state) 协调子组件生成 v4_hypothesis_list：
    1. 检查 v4_validation_report.overall_pass=True（Validation 失败短路）
    2. 调用 HypothesisGenerator 生成候选假设（振荡/双稳态/灵敏度）
    3. 为每个假设填充 experiment_design（Task 6.2 ExperimentPlanner 钩子）
    4. 调用 FalsifiabilityChecker 复核 falsifiable（Task 6.3 钩子）
    5. 调用 ParameterExplorer 验证参数鲁棒性（Task 6.4 钩子）
    6. 通过 rag_client 检索文献支持/证伪（Task 6.1.4）
    7. 返回 v4_hypothesis_list

    子组件依赖注入：
    - experiment_designer: Task 6.2 ExperimentPlanner（默认 None → 空 experiment_design）
    - falsifiability_checker: Task 6.3 FalsificationChecker（默认 None → 不复核）
    - parameter_explorer: Task 6.4 ParameterExplorer（默认 None → 不验证鲁棒性）
    - rag_client: 文献检索（默认 None → 延迟创建，失败降级到无文献支持）

    用法：
        agent = HypothesisAgent()
        hypothesis_list = agent.generate(state)
        # hypothesis_list = [{id, statement, prediction, ...}, ...]
    """

    def __init__(
        self,
        generator: HypothesisGenerator | None = None,
        experiment_designer: Any = None,
        falsifiability_checker: Any = None,
        parameter_explorer: Any = None,
        rag_client: Any = None,
    ) -> None:
        """初始化。

        Args:
            generator: HypothesisGenerator 实例（默认 None → 创建默认实例）
            experiment_designer: Task 6.2 ExperimentPlanner 实例（默认 None）
            falsifiability_checker: Task 6.3 FalsificationChecker 实例（默认 None）
            parameter_explorer: Task 6.4 ParameterExplorer 实例（默认 None）
            rag_client: RagClient 实例（默认 None → 延迟创建，失败降级）
        """
        self._generator = generator or HypothesisGenerator()
        self._experiment_designer = experiment_designer
        self._falsifiability_checker = falsifiability_checker
        self._parameter_explorer = parameter_explorer
        self._rag_client = rag_client

    # =========================================================================
    # 主入口：generate
    # =========================================================================
    def generate(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """主入口：生成 v4_hypothesis_list。

        Args:
            state: LangGraph 全局状态，含：
                - metrics: 仿真结果指标（n8_scientific_features 输出）
                - feature_metadata: 特征元数据
                - v4_validation_report: P5 Validation 报告（含 overall_pass）
                - v4_grounding_ledger: P5 SBML Grounder 溯源账本
                - v4_sensitivity_report: P5 Sensitivity 报告
                - v4_pathway_class: P4 通路类别

        Returns:
            假设列表（每项为 Hypothesis dict），失败时返回空列表。
            spec.md 第 357 行：假设生成失败 → 输出空列表 + warning，不阻塞报告生成
        """
        try:
            if not isinstance(state, dict):
                return []

            # 1. Validation 失败短路（spec.md 第 394 行）
            #    v4_validation_report.overall_pass=False 时不进入 Hypothesis
            if not self._is_validation_passed(state):
                logger.info(
                    "HypothesisAgent: Validation 未通过或未启用，短路返回空假设列表"
                )
                return []

            # 2. 提取输入字段
            metrics = state.get("metrics") or {}
            if not isinstance(metrics, dict):
                metrics = {}
            feature_metadata = state.get("feature_metadata") or {}
            if not isinstance(feature_metadata, dict):
                feature_metadata = {}
            v4_sensitivity_report = state.get("v4_sensitivity_report") or {}
            if not isinstance(v4_sensitivity_report, dict):
                v4_sensitivity_report = {}
            pathway_class = state.get("v4_pathway_class") or ""

            # 3. 调用 HypothesisGenerator 生成候选假设
            candidates = self._generator.generate(
                metrics=metrics,
                feature_metadata=feature_metadata,
                v4_sensitivity_report=v4_sensitivity_report,
                pathway_class=pathway_class,
            )

            if not candidates:
                logger.info("HypothesisAgent: 候选假设为空，返回空列表")
                return []

            # 4. 填充 experiment_design（Task 6.2 ExperimentPlanner 钩子）
            if self._experiment_designer is not None:
                candidates = self._fill_experiment_designs(candidates, state)

            # 5. 复核 falsifiable（Task 6.3 FalsificationChecker 钩子）
            if self._falsifiability_checker is not None:
                candidates = self._check_falsifiability(candidates)

            # 6. 验证参数鲁棒性（Task 6.4 ParameterExplorer 钩子）
            if self._parameter_explorer is not None:
                candidates = self._check_parameter_robustness(candidates, state)

            # 7. 文献检索填充 supporting_pmids / contradicting_pmids
            candidates = self._search_literature_for_hypotheses(candidates, state)

            logger.info(
                "HypothesisAgent: 生成 %d 条假设（pathway_class=%s）",
                len(candidates),
                pathway_class,
            )
            return candidates
        except Exception as exc:
            # 铁律 #4：失败降级返回空列表，不阻塞报告生成
            logger.warning(
                "HypothesisAgent.generate 失败，降级返回空列表: %s", exc
            )
            return []

    # =========================================================================
    # SubTask 6.1.4: 文献检索验证（复用 rag_client.py）
    # =========================================================================
    def _search_literature_for_hypotheses(
        self,
        hypotheses: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """为每个假设检索文献支持/证伪（spec.md 第 356 行 + 第 397 行）。

        复用 rag_client.RagClient.search_params 接口（仅 import 调用，不修改其代码）。
        如果 rag_client 不可用或检索失败，假设保留空 supporting_pmids /
        contradicting_pmids（不阻塞）。

        检索策略：
        - 用 hypothesis.statement + prediction 构造 query
        - 调用 rag_client.search_params 检索 top-5 相关文献
        - 从检索结果提取 source/pmid 字段
        - 简单分类：含假设关键词的 → supporting；含相反词（not / against / contradict）
          的 → contradicting

        与 SBML Grounder 交互（spec.md 第 397 行）：
        - 假设的 supporting_pmids 通过 Grounding Ledger 溯源
        - 若 v4_grounding_ledger 含相同 PMID，标记为 grounded=True

        Args:
            hypotheses: 候选假设列表
            state: LangGraph 全局状态（用于提取 v4_grounding_ledger）

        Returns:
            填充文献支持后的假设列表
        """
        client = self._get_rag_client()
        if client is None:
            # rag_client 不可用 → 假设保留空 pmids，不阻塞
            return hypotheses

        grounding_ledger = state.get("v4_grounding_ledger") or {}
        if not isinstance(grounding_ledger, dict):
            grounding_ledger = {}
        # 提取 grounding ledger 中的已知 PMID 集合（用于溯源交叉验证）
        grounded_pmids = self._extract_grounded_pmids(grounding_ledger)

        for hyp in hypotheses:
            try:
                statement = hyp.get("statement", "")
                prediction = hyp.get("prediction", "")
                query = f"{statement} {prediction}".strip()
                if not query:
                    continue

                # 调用 rag_client.search_params 检索 top-5 文献
                results = client.search_params(query=query, top_k=5)
                if not results:
                    continue

                supporting_pmids: list[str] = []
                contradicting_pmids: list[str] = []

                for rec in results:
                    if not isinstance(rec, dict):
                        continue
                    source = rec.get("source") or rec.get("pmid") or ""
                    if not isinstance(source, str) or not source.strip():
                        continue
                    source = source.strip()

                    # 简单分类：含相反词 → contradicting；否则 → supporting
                    rec_text = (
                        str(rec.get("text", "")) + " " + str(rec.get("summary", ""))
                    ).lower()
                    contradict_keywords = [
                        " not ", " against ", " contradict", " refute",
                        " disprove", " inconsistent",
                    ]
                    if any(kw in rec_text for kw in contradict_keywords):
                        contradicting_pmids.append(source)
                    else:
                        # 标记是否在 grounding_ledger 中溯源
                        is_grounded = source in grounded_pmids
                        if is_grounded:
                            source = f"{source}|grounded"
                        supporting_pmids.append(source)

                hyp["supporting_pmids"] = supporting_pmids
                hyp["contradicting_pmids"] = contradicting_pmids
                # Level 5 兼容字段
                hyp["falsifying_pmids"] = contradicting_pmids
            except Exception as exc:
                # 单个假设检索失败不影响其他假设
                logger.debug(
                    "假设 %s 文献检索失败，保留空 pmids: %s",
                    hyp.get("id", "?"),
                    exc,
                )
                continue

        return hypotheses

    # =========================================================================
    # SubTask 6.1.3: Validation 失败短路检查
    # =========================================================================
    def _is_validation_passed(self, state: dict[str, Any]) -> bool:
        """检查 P5 Validation 是否通过（spec.md 第 394 行）。

        spec.md 第 394 行：Hypothesis Agent 仅在 v4_validation_report.overall_pass=True
        时执行（Validation 失败短路到错误报告，不进入 Hypothesis）。

        特殊处理：
        - V4_VALIDATION_PYRAMID_ENABLED=false（P5 未启用）→ 视为 pass=True
          （P6 可独立于 P5 启用，只要 Validation 未明确失败就进入 Hypothesis）
        - v4_validation_report 缺失或非 dict → 视为 pass=True（保守不阻塞）
        - v4_validation_report.overall_pass=False → 短路返回空列表
        - v4_validation_report.overall_pass=True → 进入 Hypothesis

        Args:
            state: LangGraph 全局状态

        Returns:
            True 表示 Validation 通过或未启用；False 表示 Validation 明确失败
        """
        # P5 Validation 未启用 → 不阻塞（视为 pass=True）
        if not getattr(settings, "V4_VALIDATION_PYRAMID_ENABLED", False):
            return True

        validation_report = state.get("v4_validation_report")
        if not isinstance(validation_report, dict):
            # Validation 报告缺失 → 保守视为 pass=True（不阻塞）
            return True

        overall_pass = validation_report.get("overall_pass")
        # overall_pass 缺失 → 保守视为 pass=True
        if overall_pass is None:
            return True
        return bool(overall_pass)

    # =========================================================================
    # Task 6.2/6.3/6.4 钩子：子组件填充（默认 None 时跳过）
    # =========================================================================
    def _fill_experiment_designs(
        self,
        hypotheses: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Task 6.2 ExperimentPlanner 钩子：为每个假设填充 experiment_design。

        若 experiment_designer 为 None（Task 6.2 未实现），保留空 experiment_design。
        """
        try:
            designer = self._experiment_designer
            if designer is None:
                return hypotheses
            for hyp in hypotheses:
                try:
                    design = designer.design(hyp, state)
                    if isinstance(design, dict):
                        hyp["experiment_design"] = design
                except Exception as exc:
                    logger.debug(
                        "假设 %s experiment_design 填充失败: %s",
                        hyp.get("id", "?"),
                        exc,
                    )
            return hypotheses
        except Exception as exc:
            logger.warning("_fill_experiment_designs 失败，保留空 design: %s", exc)
            return hypotheses

    def _check_falsifiability(
        self, hypotheses: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Task 6.3 FalsificationChecker 钩子：复核每个假设的 falsifiable 字段。

        若 falsifiability_checker 为 None（Task 6.3 未实现），保留默认 falsifiable=True。
        """
        try:
            checker = self._falsifiability_checker
            if checker is None:
                return hypotheses
            for hyp in hypotheses:
                try:
                    result = checker.check(hyp)
                    if isinstance(result, dict):
                        hyp["falsifiable"] = bool(result.get("falsifiable", True))
                        if "falsification_criteria" in result:
                            hyp["falsification_criteria"] = result[
                                "falsification_criteria"
                            ]
                except Exception as exc:
                    logger.debug(
                        "假设 %s falsifiability 复核失败: %s",
                        hyp.get("id", "?"),
                        exc,
                    )
            return hypotheses
        except Exception as exc:
            logger.warning("_check_falsifiability 失败，保留默认 falsifiable: %s", exc)
            return hypotheses

    def _check_parameter_robustness(
        self,
        hypotheses: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Task 6.4 ParameterExplorer 钩子：验证假设的参数鲁棒性。

        若 parameter_explorer 为 None（Task 6.4 未实现），保留默认无鲁棒性验证。
        """
        try:
            explorer = self._parameter_explorer
            if explorer is None:
                return hypotheses
            for hyp in hypotheses:
                try:
                    robustness = explorer.explore(hyp, state)
                    if isinstance(robustness, dict):
                        hyp["parameter_robustness"] = robustness
                except Exception as exc:
                    logger.debug(
                        "假设 %s parameter_robustness 验证失败: %s",
                        hyp.get("id", "?"),
                        exc,
                    )
            return hypotheses
        except Exception as exc:
            logger.warning("_check_parameter_robustness 失败: %s", exc)
            return hypotheses

    # =========================================================================
    # 辅助函数：rag_client 延迟创建 + grounding ledger PMID 提取
    # =========================================================================
    def _get_rag_client(self) -> Any:
        """延迟创建 RagClient 实例（失败时返回 None，不阻塞）。

        复用 rag_client.RagClient（仅 import 调用，不修改其代码）。
        """
        if self._rag_client is not None:
            return self._rag_client
        try:
            from app.rag_client import RagClient
            self._rag_client = RagClient()
            return self._rag_client
        except Exception as exc:
            logger.warning(
                "RagClient 创建失败，假设将无文献支持: %s", exc
            )
            return None

    @staticmethod
    def _extract_grounded_pmids(grounding_ledger: dict[str, Any]) -> set[str]:
        """从 v4_grounding_ledger 提取已知 PMID 集合（用于溯源交叉验证）。

        grounding_ledger 结构（P5 SBML Grounder 输出）：
        - {entries: [{pmid, ...}, ...], integrity: bool, ...}
        - 或扁平 {pmids: [...], ...}

        Returns:
            PMID 字符串集合（用于 _search_literature_for_hypotheses 标记 grounded）
        """
        pmids: set[str] = set()
        if not isinstance(grounding_ledger, dict):
            return pmids

        # 结构化形式：entries 列表
        entries = grounding_ledger.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                pmid = entry.get("pmid") or entry.get("source") or ""
                if isinstance(pmid, str) and pmid.strip():
                    pmids.add(pmid.strip())
            if pmids:
                return pmids

        # 扁平形式：pmids 列表
        pmid_list = grounding_ledger.get("pmids")
        if isinstance(pmid_list, list):
            for pmid in pmid_list:
                if isinstance(pmid, str) and pmid.strip():
                    pmids.add(pmid.strip())

        return pmids


# =============================================================================
# LangGraph hook 节点（Feature Flag 隔离）
# =============================================================================
def hypothesis_agent_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：Hypothesis Agent hook（spec.md Part 5 第 352-358 行）。

    行为：
    - V4_HYPOTHESIS_AGENT_ENABLED=false：直接返回空 dict（不修改 state）
    - V4_HYPOTHESIS_AGENT_ENABLED=true：调用 HypothesisAgent.generate()
      写入 state["v4_hypothesis_list"]

    严格遵守不可碰清单：
    - 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
    - 不生成 ODE / 不做 SBML 验证 / 不修改 P5 Validation Report
    - 失败时降级返回空 dict，不抛异常，不阻塞主流水线

    依赖检查（spec.md 第 471 行）：
    - V4_HYPOTHESIS_AGENT_ENABLED=true 隐含 V4_VALIDATION_PYRAMID_ENABLED=true
      （Hypothesis 依赖 Validation pass）
    - 若 V4_VALIDATION_PYRAMID_ENABLED=false，Hypothesis 仍可执行，但
      _is_validation_passed() 视为 pass=True（不阻塞）

    Args:
        state: LangGraph 全局状态

    Returns:
        flag=false 时返回 {}
        flag=true 时返回 {"v4_hypothesis_list": [...], "v4_hypothesis_generated": True}
        异常时返回 {}
    """
    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not getattr(settings, "V4_HYPOTHESIS_AGENT_ENABLED", False):
        logger.debug("V4_HYPOTHESIS_AGENT_ENABLED=false，跳过 Hypothesis Agent")
        return {}

    try:
        agent = HypothesisAgent()
        hypothesis_list = agent.generate(state)
        # v4_hypothesis_generated 标志位（Task 6.7 SSE 事件消费）
        return {
            "v4_hypothesis_list": hypothesis_list,
            "v4_hypothesis_generated": True,
        }
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("Hypothesis Agent hook 失败，降级跳过: %s", exc)
        return {}


__all__ = ["HypothesisAgent", "hypothesis_agent_hook_node"]
