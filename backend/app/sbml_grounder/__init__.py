# BioDynamics Agent v4 - SBML Grounder Agent (Phase 5 / Task 5.1)
#
# 职责：建立 ODE ↔ Reaction ↔ SBML ↔ Parameter ↔ PMID 五级映射链
# （修复审计 §7.4 五级缺失）
#
# 设计原则（铁律）：
# 1. Feature Flag V4_SBML_GROUNDER_ENABLED=false 时完全不执行，系统行为同 v3
# 2. 不修改 v3 任何字段（network_json / parameters / ode_model 等）
# 3. 仅消费 P1/P2/P3 产出（v4_ontology_entities / v4_reaction_ir / v4_ode_system）
# 4. 失败降级：任何异常都返回空更新，不阻塞主流水线
# 5. 依赖隔离：lxml 不可用时降级到 xml.etree.ElementTree（标准库）
#
# 模块结构：
#   - sbml_parser_v2.py      : 真正 XML 解析（替代 v3 LLM 解析）
#   - canonical_species.py   : canonical species 解析（HGNC/UniProt 提取）
#   - ontology_grounding.py   : HGNC/UniProt/ChEBI 对齐（复用 P1）
#   - alias_resolution.py     : UniProt 多字段对齐（EGFR/ERBB1/HER1）
#   - five_level_mapping.py   : 五级映射链构建
#   - grounder_agent.py       : SBMLGrounderAgent 主类 + LangGraph hook

from __future__ import annotations

from app.sbml_grounder.grounder_agent import SBMLGrounderAgent, sbml_grounder_hook_node
from app.sbml_grounder.sbml_parser_v2 import SBMLDocument, SBMLParserV2
from app.sbml_grounder.five_level_mapping import FiveLevelMapper
from app.sbml_grounder.canonical_species import CanonicalSpeciesResolver
from app.sbml_grounder.ontology_grounding import OntologyGrounder
from app.sbml_grounder.alias_resolution import AliasResolver

__all__ = [
    "SBMLGrounderAgent",
    "sbml_grounder_hook_node",
    "SBMLParserV2",
    "SBMLDocument",
    "FiveLevelMapper",
    "CanonicalSpeciesResolver",
    "OntologyGrounder",
    "AliasResolver",
]
