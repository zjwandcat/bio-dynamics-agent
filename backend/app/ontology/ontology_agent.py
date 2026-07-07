# BioDynamics Agent v4 - Ontology Agent 主逻辑
# Phase 1 核心：将用户问题中的生物实体标准化到 HGNC / UniProt / ChEBI / GO / SBO ID。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_ONTOLOGY_AGENT_ENABLED=false 时完全不执行，系统行为同 v3
# 2. 任何 API 失败都降级返回 verified=false，不抛异常，不阻塞主流水线
# 3. 缓存优先（TTL 7 天），减少外部 API 调用
# 4. 实体抽取优先复用 v3 NER 结果（state["entities"]），无则用关键词匹配兜底
# 5. 输出写入 state["v4_ontology_entities"]，v3 字段完全不触碰

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import settings
from app.state import set_v4_state
from app.ontology.chebi_client import query_chebi
from app.ontology.go_client import query_go
from app.ontology.hgnc_client import query_hgnc
from app.ontology.pathway_registry import lookup_pathway
from app.ontology.sbo_terms import get_sbo_term
from app.ontology.uniprot_client import query_uniprot

logger = logging.getLogger(__name__)


# =============================================================================
# 实体抽取：从用户文本提取候选生物实体名
# =============================================================================
# 已知蛋白/基因大写缩写集合（兜底用，避免依赖 v3 NER）
_KNOWN_PROTEINS = {
    # EGFR 通路
    "EGF", "EGFR", "ERBB1", "ERBB2", "HER2", "SHC", "GRB2", "SOS", "RAS",
    "RAF", "BRAF", "MEK", "MAP2K1", "ERK", "MAPK", "MAPK1", "MAPK3",
    # PI3K 通路
    "PI3K", "AKT", "MTOR", "PTEN", "PDK1", "TSC", "RHEB", "S6K",
    # p53 通路
    "TP53", "MDM2", "P21", "CDKN1A", "BAX", "PUMA", "NOXA",
    # 凋亡
    "BCL2", "BAX", "BAK", "CASP3", "CASP8", "CASP9", "FASL", "FAS",
    # 细胞周期
    "CCND1", "CDK4", "CDK6", "CDKN2A", "RB1", "E2F",
    # JAK-STAT
    "JAK1", "JAK2", "STAT1", "STAT3", "STAT5",
    # NF-κB
    "NFKB1", "RELA", "IKBKB", "CHUK", "NFKBIA",
    # Wnt
    "CTNNB1", "APC", "AXIN1", "GSK3B",
    # TGF-β
    "TGFB1", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4",
}

# 已知化学实体（配体/药物）集合
_KNOWN_CHEMICALS = {
    "EGF", "imatinib", "dasatinib", "gefitinib", "erlotinib", "lapatinib",
    "trastuzumab", "cetuximab", "doxorubicin", "paclitaxel", "cisplatin",
    "5-FU", "fluorouracil", "methotrexate", "rapamycin", "wortmannin",
    "LY294002", "U0126", "PD98059", "SB203580", "TNF", "LPS", "IFN",
}

# 物种类型分类
_SPECIES_TYPES = {"gene", "protein", "chemical", "pathway", "unknown"}


def _extract_entities_from_text(text: str) -> list[dict[str, str]]:
    """从文本中提取候选生物实体（兜底实现，无 LLM 依赖）。

    策略：
    1. 匹配已知蛋白/基因缩写（全词匹配，大小写敏感）
    2. 匹配已知化学实体名
    3. 匹配大写连续字母（≥2 字符，疑似蛋白缩写）

    Args:
        text: 用户输入文本

    Returns:
        实体候选列表，每项 {name, species_type, source}
    """
    if not text:
        return []
    entities: list[dict[str, str]] = []
    seen: set[str] = set()

    # 1. 匹配已知蛋白/基因
    for protein in _KNOWN_PROTEINS:
        if re.search(rf"\b{re.escape(protein)}\b", text):
            if protein not in seen:
                entities.append({
                    "name": protein,
                    "species_type": "protein",
                    "source": "known_protein_set",
                })
                seen.add(protein)

    # 2. 匹配已知化学实体
    for chem in _KNOWN_CHEMICALS:
        pattern = rf"\b{re.escape(chem)}\b"
        if re.search(pattern, text, re.IGNORECASE):
            if chem not in seen:
                entities.append({
                    "name": chem,
                    "species_type": "chemical",
                    "source": "known_chemical_set",
                })
                seen.add(chem)

    # 3. 兜底：连续大写字母（疑似蛋白缩写），长度 2-6
    # 仅补充未被前两步命中的实体
    for match in re.finditer(r"\b([A-Z][A-Z0-9]{1,5})\b", text):
        name = match.group(1)
        if name not in seen and name not in {"DNA", "RNA", "mRNA", "GTP", "GDP", "ATP", "ADP", "AMP"}:
            entities.append({
                "name": name,
                "species_type": "protein",
                "source": "regex_uppercase",
            })
            seen.add(name)

    return entities


def _merge_with_v3_entities(
    v3_entities: list[dict] | None,
    extracted: list[dict[str, str]],
) -> list[dict[str, str]]:
    """合并 v3 NER 实体与本模块抽取的实体，去重。

    v3 NER 实体格式：{entity_id, name, type, aliases, canonical_id}
    本模块格式：{name, species_type, source}

    合并策略：v3 实体优先（已 LLM 抽取），本模块补缺；按 name 去重。

    Args:
        v3_entities: v3 NER 输出的实体列表（可能为 None）
        extracted: 本模块兜底抽取的实体列表

    Returns:
        合并去重后的实体列表，统一为 {name, species_type, source}
    """
    merged: list[dict[str, str]] = []
    seen: set[str] = set()

    # v3 实体优先
    if v3_entities:
        for ent in v3_entities:
            name = (ent.get("name") or "").strip()
            if not name or name in seen:
                continue
            # v3 的 type 字段可能为 "protein" / "chemical" / "pathway" 等
            species_type = (ent.get("type") or "unknown").lower()
            if species_type not in _SPECIES_TYPES:
                species_type = "unknown"
            merged.append({
                "name": name,
                "species_type": species_type,
                "source": "v3_ner",
            })
            seen.add(name)

    # 补充本模块抽取的实体
    for ent in extracted:
        name = ent["name"]
        if name not in seen:
            merged.append(ent)
            seen.add(name)

    return merged


# =============================================================================
# Ontology Agent 主逻辑
# =============================================================================
class OntologyAgent:
    """Ontology Agent：将生物实体标准化到 HGNC/UniProt/ChEBI/GO/SBO ID。

    使用方式：
        agent = OntologyAgent()
        result = agent.annotate(user_input, v3_entities=state.get("entities"))

    返回结构：
        {
            "entities": [
                {
                    "name": "EGFR",
                    "hgnc_id": "HGNC:3236",
                    "uniprot_id": "P00533",
                    "chebi_id": "",
                    "go_terms": [{"go_id": "GO:0007179", ...}, ...],
                    "sbo_term": "SBO:0000216",   # 若识别出机制
                    "species_type": "protein",
                    "verified": True,
                    "source": "v3_ner" | "known_protein_set" | ...
                },
                ...
            ],
            "pathway_class": "EGFR_RTK",  # 通路识别结果（来自 pathway_registry）
            "warnings": ["..."],          # 降级 warning 列表
        }
    """

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def annotate(
        self,
        user_input: str,
        v3_entities: list[dict] | None = None,
    ) -> dict[str, Any]:
        """对用户输入进行本体标注。

        Args:
            user_input: 用户原始问题文本
            v3_entities: v3 NER 抽取的实体列表（可选，优先复用）

        Returns:
            标注结果字典（结构见类文档字符串）
        """
        self.warnings = []

        # 1. 实体抽取：优先复用 v3 NER，兜底用关键词匹配
        extracted = _extract_entities_from_text(user_input)
        entities = _merge_with_v3_entities(v3_entities, extracted)

        if not entities:
            logger.info("Ontology Agent 未抽取出任何实体")
            return {
                "entities": [],
                "pathway_class": lookup_pathway(user_input),
                "warnings": ["no_entities_extracted"],
            }

        # 2. 逐个实体查本体 API
        annotated: list[dict[str, Any]] = []
        for ent in entities:
            try:
                annotated.append(self._annotate_single(ent))
            except Exception as exc:
                # 单个实体失败不阻塞，记录 warning 并降级
                self.warnings.append(f"{ent['name']}: {exc}")
                annotated.append({
                    "name": ent["name"],
                    "hgnc_id": "",
                    "uniprot_id": "",
                    "chebi_id": "",
                    "go_terms": [],
                    "sbo_term": "",
                    "species_type": ent.get("species_type", "unknown"),
                    "verified": False,
                    "source": ent.get("source", "unknown"),
                    "error": str(exc),
                })

        # 3. 通路识别
        pathway_class = lookup_pathway(user_input)

        result: dict[str, Any] = {
            "entities": annotated,
            "pathway_class": pathway_class,
            "warnings": self.warnings,
        }
        logger.info(
            "Ontology Agent 标注完成：%d 个实体，%d 个 verified，通路=%s",
            len(annotated),
            sum(1 for e in annotated if e.get("verified")),
            pathway_class,
        )
        return result

    def _annotate_single(self, ent: dict[str, str]) -> dict[str, Any]:
        """对单个实体查询所有本体 API。

        Args:
            ent: {name, species_type, source}

        Returns:
            标注后的实体字典
        """
        name = ent["name"]
        species_type = ent.get("species_type", "unknown")
        verified = True

        # HGNC 查询（gene / protein 类）
        hgnc_id = ""
        uniprot_id = ""
        if species_type in ("gene", "protein", "unknown"):
            hgnc_result = query_hgnc(name)
            if hgnc_result:
                hgnc_id = hgnc_result.get("hgnc_id", "")
                uniprot_id = hgnc_result.get("uniprot_id", "")
                # 若 HGNC 命中，修正 species_type
                if species_type == "unknown":
                    species_type = "protein"

        # UniProt 查询（若 HGNC 未给出 uniprot_id）
        if not uniprot_id and species_type in ("protein", "unknown"):
            uniprot_result = query_uniprot(name)
            if uniprot_result:
                uniprot_id = uniprot_result.get("accession", "")

        # ChEBI 查询（chemical 类）
        chebi_id = ""
        if species_type == "chemical":
            chebi_result = query_chebi(name)
            if chebi_result:
                chebi_id = chebi_result.get("chebi_id", "")

        # GO 查询（protein / gene 类）
        go_terms: list[dict[str, Any]] = []
        if species_type in ("protein", "gene") and (hgnc_id or uniprot_id):
            go_terms = query_go(name)

        # SBO term：根据文本上下文识别机制（简化版，仅基于关键词）
        sbo_term = self._infer_sbo_from_context(name, species_type)

        # verified 判定：至少有一个 ID 命中
        if not any([hgnc_id, uniprot_id, chebi_id]):
            verified = False
            self.warnings.append(f"{name}: no canonical ID resolved")

        return {
            "name": name,
            "hgnc_id": hgnc_id,
            "uniprot_id": uniprot_id,
            "chebi_id": chebi_id,
            "go_terms": go_terms,
            "sbo_term": sbo_term,
            "species_type": species_type,
            "verified": verified,
            "source": ent.get("source", "unknown"),
        }

    def _infer_sbo_from_context(self, name: str, species_type: str) -> str:
        """根据实体类型推断可能的 SBO term（简化版，仅 P1 占位）。

        P1 阶段不做精确的机制识别（那是 P2 Reaction IR 的工作），
        仅根据 species_type 给出粗粒度的默认 SBO term。

        Args:
            name: 实体名
            species_type: 实体类型

        Returns:
            SBO term 字符串（如 "SBO:0000216"），无法推断返回空串
        """
        # 化学实体 → binding（配体-受体结合）
        if species_type == "chemical":
            return get_sbo_term("binding") or ""
        # 蛋白 → 暂不绑定具体机制（P2 Reaction IR 阶段细化）
        return ""


# =============================================================================
# LangGraph Hook 节点
# =============================================================================
# ontology_agent 单例（避免每次请求重新创建）
_ontology_agent: OntologyAgent | None = None


def _get_ontology_agent() -> OntologyAgent:
    """获取 Ontology Agent 单例（延迟初始化）。"""
    global _ontology_agent
    if _ontology_agent is None:
        _ontology_agent = OntologyAgent()
    return _ontology_agent


def ontology_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：在 pre_router 前执行的本体标注 hook。

    行为：
    - V4_ONTOLOGY_AGENT_ENABLED=false：直接返回空 dict（不修改 state，不执行任何逻辑）
    - V4_ONTOLOGY_AGENT_ENABLED=true：调用 OntologyAgent，结果写入 state["v4_ontology_entities"]

    严格遵守 P1 不可碰清单：
    - 不修改 v3 任何字段（network_json / parameters / entities 等）
    - 不改变路由逻辑（返回的 dict 不含 execution_plan / next_worker 等）
    - 失败时降级返回 verified=false，不抛异常

    Args:
        state: LangGraph 全局状态

    Returns:
        状态更新字典。flag=false 时返回空 dict（无更新）；
        flag=true 时返回 {"v4_ontology_entities": {...}}
    """
    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not getattr(settings, "V4_ONTOLOGY_AGENT_ENABLED", False):
        logger.debug("V4_ONTOLOGY_AGENT_ENABLED=false，跳过 Ontology Agent")
        return {}

    # 执行 Ontology Agent
    try:
        user_input = state.get("user_input", "")
        v3_entities = state.get("entities")  # 复用 v3 NER 结果（若有）
        agent = _get_ontology_agent()
        result = agent.annotate(user_input, v3_entities=v3_entities)
        logger.info(
            "Ontology Agent hook 完成：%d 实体，通路=%s",
            len(result.get("entities", [])),
            result.get("pathway_class"),
        )
        # 仅写入 v4 字段，不触碰任何 v3 字段
        # Task B.2: 双写 v4_ontology_entities → v4_state["ontology"]["entities"]
        result_update: dict[str, Any] = {}
        set_v4_state(result_update, "ontology", "entities", result)
        return result_update
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("Ontology Agent hook 失败，降级跳过: %s", exc)
        return {}
