# BioDynamics Agent v4 - 本体层（Ontology Layer）
# Phase 1 新增模块：提供 HGNC / UniProt / ChEBI / GO / SBO / KEGG / Reactome 标准化能力。
# 所有新代码仅放在本目录下，通过 Feature Flag V4_ONTOLOGY_AGENT_ENABLED 控制启用，
# 默认 false 时系统行为与 v3 完全一致，不阻塞主流水线。

from app.ontology.ontology_agent import OntologyAgent, ontology_hook_node
from app.ontology.sbo_terms import SBOTerms, MECHANISM_TO_SBO, SBO_TO_MECHANISM
from app.ontology.pathway_registry import PATHWAY_REGISTRY, lookup_pathway

__all__ = [
    "OntologyAgent",
    "ontology_hook_node",
    "SBOTerms",
    "MECHANISM_TO_SBO",
    "SBO_TO_MECHANISM",
    "PATHWAY_REGISTRY",
    "lookup_pathway",
]
