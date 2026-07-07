# BioDynamics Agent v4 - SBML Grounder Agent (Phase 5 / Task 5.1.1)
#
# SBML Grounder Agent 主类 + LangGraph hook 节点。
# 职责：建立 ODE ↔ Reaction ↔ SBML ↔ Parameter ↔ PMID 五级映射链。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_SBML_GROUNDER_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段（network_json / parameters / ode_model 等）
# 3. 仅消费 P1/P2/P3 产出（v4_ontology_entities / v4_reaction_ir / v4_ode_system）
# 4. 失败降级：任何异常都返回空更新，不阻塞主流水线
# 5. 输出写入 state["v4_grounding_ledger"]（新增 v4 字段）
#
# 依赖：
# - app.config.settings（Feature Flag）
# - app.sbml_grounder.sbml_parser_v2.SBMLParserV2
# - app.sbml_grounder.five_level_mapping.FiveLevelMapper
# - app.sbml_grounder.canonical_species.CanonicalSpeciesResolver
# - app.sbml_grounder.ontology_grounding.OntologyGrounder
# - app.sbml_grounder.alias_resolution.AliasResolver

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.state import set_v4_state
from app.sbml_grounder.alias_resolution import AliasResolver
from app.sbml_grounder.canonical_species import CanonicalSpeciesResolver
from app.sbml_grounder.five_level_mapping import FiveLevelMapper
from app.sbml_grounder.ontology_grounding import OntologyGrounder
from app.sbml_grounder.sbml_parser_v2 import SBMLDocument, SBMLParserV2

logger = logging.getLogger(__name__)


# =============================================================================
# SBMLGrounderAgent 主类
# =============================================================================
class SBMLGrounderAgent:
    """SBML Grounder Agent 主类。

    主入口 ground(state) 构建 ODE ↔ Reaction ↔ SBML ↔ Parameter ↔ PMID 五级映射链，
    输出 v4_grounding_ledger。

    输入（来自 state）：
        - v4_ode_system: P3 输出的 ODE 系统（含 ode_code / equations）
        - v4_reaction_ir: P2 输出的 Reaction IR（含 reactions + provenance）
        - sbml_model_id: BioModels 模型 ID（如 BIOMD0000000205）
        - sbml_model_text: SBML XML 文本（用于解析）
        - parameters: state.parameters dict
        - v4_ontology_entities: P1 Ontology Agent 输出（可选）

    输出：
        v4_grounding_ledger: {
            ode_equations: [{eq_id, reaction_id, sbml_reaction_id,
                             parameter_ids, pmids, species_ids}],
            species_mapping: [...],
            integrity: bool,
            warnings: [...],
            statistics: {...}
        }

    用法：
        agent = SBMLGrounderAgent()
        result = agent.ground(state)
        # result = {"v4_grounding_ledger": {...}}
    """

    def __init__(
        self,
        sbml_parser: SBMLParserV2 | None = None,
        mapper: FiveLevelMapper | None = None,
    ) -> None:
        # 依赖注入（默认创建，测试时可 mock）
        self._sbml_parser = sbml_parser or SBMLParserV2()
        if mapper is not None:
            self._mapper = mapper
        else:
            # 默认创建带 alias_resolver 的 mapper 链
            alias_resolver = AliasResolver()
            canonical_resolver = CanonicalSpeciesResolver(alias_resolver)
            ontology_grounder = OntologyGrounder(alias_resolver)
            self._mapper = FiveLevelMapper(
                canonical_resolver=canonical_resolver,
                ontology_grounder=ontology_grounder,
            )

    def ground(self, state: dict[str, Any]) -> dict[str, Any]:
        """主入口：构建五级映射链。

        Args:
            state: LangGraph 全局状态，含 v4_ode_system / v4_reaction_ir /
                sbml_model_id / sbml_model_text / parameters / v4_ontology_entities

        Returns:
            {"v4_grounding_ledger": {ode_equations, species_mapping,
                                       integrity, warnings, statistics}}
            失败降级返回空 ledger：{"v4_grounding_ledger": {integrity: False, ...}}
        """
        try:
            # 提取输入
            ode_system = state.get("v4_ode_system") or {}
            reaction_ir = state.get("v4_reaction_ir") or {}
            sbml_model_id = state.get("sbml_model_id", "")
            sbml_model_text = state.get("sbml_model_text", "")
            parameters = state.get("parameters") or {}
            p1_ontology_entities = state.get("v4_ontology_entities")

            logger.info(
                "SBMLGrounderAgent 开始五级映射：sbml_model_id=%s, "
                "ode_equations=%s, reactions=%s",
                sbml_model_id,
                "present" if ode_system else "missing",
                "present" if reaction_ir else "missing",
            )

            # 解析 SBML（若提供文本）
            sbml_document: SBMLDocument | None = None
            if sbml_model_text:
                sbml_document = self._sbml_parser.parse(sbml_model_text)
                logger.info(
                    "SBML 解析完成：backend=%s, species=%d, reactions=%d, "
                    "parameters=%d, integrity=%s",
                    sbml_document.parser_backend,
                    len(sbml_document.species),
                    len(sbml_document.reactions),
                    len(sbml_document.parameters),
                    sbml_document.integrity,
                )
            else:
                logger.warning(
                    "sbml_model_text 为空，五级映射将缺失 SBML Reaction 层"
                )

            # 构建五级映射
            ledger = self._mapper.build_mapping(
                ode_system=ode_system,
                reaction_ir=reaction_ir,
                sbml_document=sbml_document,
                parameters=parameters,
                p1_ontology_entities=p1_ontology_entities,
            )

            # 补充元信息
            ledger["sbml_model_id"] = sbml_model_id
            ledger["parser_backend"] = (
                sbml_document.parser_backend if sbml_document else "none"
            )
            ledger["agent_version"] = "v4.0"

            logger.info(
                "SBMLGrounderAgent 完成：integrity=%s, ode_equations=%d, "
                "species_verified=%d/%d, warnings=%d",
                ledger.get("integrity", False),
                len(ledger.get("ode_equations", [])),
                ledger.get("statistics", {}).get("verified_species", 0),
                ledger.get("statistics", {}).get("total_species", 0),
                len(ledger.get("warnings", [])),
            )

            return {"v4_grounding_ledger": ledger}

        except Exception as exc:
            logger.warning(
                "SBMLGrounderAgent.ground 失败，降级返回空 ledger: %s", exc
            )
            return self._fallback_ledger(str(exc))

    def _fallback_ledger(self, reason: str = "") -> dict[str, Any]:
        """失败降级：返回空 ledger（integrity=False）。"""
        return {
            "v4_grounding_ledger": {
                "ode_equations": [],
                "species_mapping": [],
                "integrity": False,
                "warnings": [f"grounder_fallback: {reason}"] if reason else [],
                "statistics": {},
                "agent_version": "v4.0",
                "fallback": True,
            }
        }


# =============================================================================
# LangGraph hook 节点（Feature Flag 隔离）
# =============================================================================
def sbml_grounder_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：SBML Grounder hook。

    行为：
    - V4_SBML_GROUNDER_ENABLED=false：直接返回空 dict（不修改 state）
    - V4_SBML_GROUNDER_ENABLED=true：调用 SBMLGrounderAgent.ground() 写入
      state["v4_grounding_ledger"]

    严格遵守不可碰清单：
    - 不修改 v3 任何字段（network_json / parameters / ode_model 等）
    - 不生成 ODE / 不调用 RAG / 不做 SBML 验证
    - 失败时降级返回空 dict，不抛异常，不阻塞主流水线

    Args:
        state: LangGraph 全局状态

    Returns:
        flag=false 时返回 {}
        flag=true 时返回 {"v4_grounding_ledger": {...}}
        异常时返回 {}
    """
    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not settings.effective_v4_sbml_grounder_enabled():
        logger.debug("V4_SBML_GROUNDER_ENABLED effective=false，跳过 SBML Grounder")
        return {}

    try:
        agent = SBMLGrounderAgent()
        result = agent.ground(state)
        # Task B.2: 双写 v4_grounding_ledger → v4_state["grounding"]["ledger"]
        if "v4_grounding_ledger" in result:
            set_v4_state(result, "grounding", "ledger", result["v4_grounding_ledger"])
        return result
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("SBML Grounder hook 失败，降级跳过: %s", exc)
        return {}


__all__ = ["SBMLGrounderAgent", "sbml_grounder_hook_node"]
