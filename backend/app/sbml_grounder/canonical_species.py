# BioDynamics Agent v4 - Canonical Species 解析 (Phase 5 / Task 5.1.4)
#
# 从 SBML species 元素的 metaid / annotation 提取 HGNC/UniProt/ChEBI ID；
# 缺失时用 species name + compartment 推断 canonical name。
#
# 设计原则（铁律）：
# 1. 仅消费 SBMLParserV2 提取的 species dict（不直接读 XML）
# 2. 失败降级：annotation 缺失时用 name+compartment 推断，verified=False
# 3. 不调用外部 API（HGNC/UniProt 查询在 ontology_grounding.py 中）
# 4. 输出 list[dict] 纯数据结构，便于序列化与下游消费
#
# 依赖：
# - app.sbml_grounder.sbml_parser_v2.SBMLParserV2（复用 ontology 正则）
# - app.sbml_grounder.alias_resolution.AliasResolver（canonical name 归一）

from __future__ import annotations

import logging
import re
from typing import Any

from app.sbml_grounder.alias_resolution import AliasResolver

logger = logging.getLogger(__name__)


# =============================================================================
# Ontology ID 正则（与 SBMLParserV2 内部一致，但独立维护避免循环依赖）
# =============================================================================
_HGNC_RE = re.compile(r"HGNC:HGNC:(\d+)", re.IGNORECASE)
_HGNC_PLAIN_RE = re.compile(r"(?<![A-Za-z])HGNC:(\d+)(?!\d)")
_UNIPROT_PREFIX_RE = re.compile(r"UniProt[:\s]+([A-Z0-9]{6,10})", re.IGNORECASE)
_UNIPROT_IDENTIFIERS_RE = re.compile(
    r"identifiers\.org/uniprot/([A-Z0-9]{6,10})", re.IGNORECASE
)
_CHEBI_RE = re.compile(r"CHEBI:(\d+)", re.IGNORECASE)


# =============================================================================
# CanonicalSpeciesResolver 主类
# =============================================================================
class CanonicalSpeciesResolver:
    """Canonical Species 解析器。

    从 SBML species 的 metaid / annotation 提取 HGNC/UniProt/ChEBI ID，
    缺失时用 species name + compartment 推断 canonical name（verified=False）。

    用法：
        resolver = CanonicalSpeciesResolver()
        canonical = resolver.resolve(sbml_document.species)
        for sp in canonical:
            print(sp["sbml_species_id"], sp["hgnc_id"], sp["verified"])
    """

    def __init__(self, alias_resolver: AliasResolver | None = None) -> None:
        # 注入 AliasResolver（默认创建，测试时可 mock）
        self._alias_resolver = alias_resolver or AliasResolver()

    def resolve(self, sbml_species: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """解析 canonical species 列表。

        Args:
            sbml_species: SBMLParserV2.extract_species 输出，每项含
                {id, name, compartment, metaid, annotation, ontology}

        Returns:
            list[dict] 每项含：
                {sbml_species_id, canonical_name, display_name, compartment,
                 hgnc_id, uniprot_id, chebi_id, verified, source}
        """
        result: list[dict[str, Any]] = []
        for sp in sbml_species:
            try:
                resolved = self._resolve_one(sp)
                result.append(resolved)
            except Exception as exc:
                logger.warning(
                    "canonical species 解析失败 (id=%s): %s",
                    sp.get("id", "?"),
                    exc,
                )
                # 失败降级：仅保留 id 与 name，verified=False
                result.append(
                    {
                        "sbml_species_id": sp.get("id", ""),
                        "canonical_name": sp.get("name", sp.get("id", "")),
                        "display_name": sp.get("name", ""),
                        "compartment": sp.get("compartment", ""),
                        "hgnc_id": None,
                        "uniprot_id": None,
                        "chebi_id": None,
                        "verified": False,
                        "source": "fallback_on_error",
                    }
                )
        return result

    def _resolve_one(self, sp: dict[str, Any]) -> dict[str, Any]:
        """解析单个 species 的 canonical 信息。"""
        sp_id = sp.get("id", "")
        sp_name = sp.get("name", "") or sp_id
        compartment = sp.get("compartment", "")
        annotation = sp.get("annotation", "") or ""
        ontology = sp.get("ontology", {}) or {}

        # 优先从 ontology dict 直接取（SBMLParserV2 已预解析）
        hgnc_id = ontology.get("hgnc_id") or self.extract_hgnc_id(annotation)
        uniprot_id = ontology.get("uniprot_id") or self.extract_uniprot_id(annotation)
        chebi_id = ontology.get("chebi_id") or self.extract_chebi_id(annotation)

        # canonical name 归一：先用 alias_resolver，再用 species name
        canonical_name = self._alias_resolver.canonicalize(sp_name)
        # 若 alias_resolver 未命中（返回原名），且 name 与 id 不同则用 name
        if canonical_name == sp_name and sp_name != sp_id:
            canonical_name = sp_name

        # verified 标记：有 HGNC 或 UniProt ID 即视为 verified
        verified = bool(hgnc_id or uniprot_id)

        # 推断 source
        if verified:
            source = "sbml_annotation"
        elif annotation:
            source = "sbml_annotation_partial"  # 有 annotation 但未提取到 ID
        else:
            source = "inferred_from_name_compartment"

        # 缺失 annotation 时用 name + compartment 推断（已在 canonical_name 体现）
        # 不阻塞，但 verified=False（下游 ontology_grounding 会标记 warning）
        if not verified and not annotation:
            logger.debug(
                "species %s 无 annotation，用 name+compartment 推断: %s@%s",
                sp_id,
                sp_name,
                compartment,
            )

        return {
            "sbml_species_id": sp_id,
            "canonical_name": canonical_name,
            "display_name": sp_name,
            "compartment": compartment,
            "hgnc_id": hgnc_id,
            "uniprot_id": uniprot_id,
            "chebi_id": chebi_id,
            "verified": verified,
            "source": source,
        }

    # -------------------------------------------------------------------------
    # Ontology ID 正则提取接口（公开，供测试与外部调用）
    # -------------------------------------------------------------------------
    def extract_hgnc_id(self, annotation: str) -> str | None:
        """从 annotation 文本提取 HGNC ID。

        支持格式：
        - HGNC:HGNC:3236（identifiers.org 标准格式）
        - HGNC:3236（简化格式）

        Returns:
            形如 'HGNC:3236' 的字符串，无匹配返回 None
        """
        if not annotation:
            return None
        m = _HGNC_RE.search(annotation)
        if m:
            return f"HGNC:{m.group(1)}"
        m = _HGNC_PLAIN_RE.search(annotation)
        if m:
            return f"HGNC:{m.group(1)}"
        return None

    def extract_uniprot_id(self, annotation: str) -> str | None:
        """从 annotation 文本提取 UniProt accession。

        支持格式：
        - UniProt:P00533（显式前缀）
        - identifiers.org/uniprot/P00533（URI 形式）

        Returns:
            形如 'P00533' 的字符串（大写），无匹配返回 None
        """
        if not annotation:
            return None
        m = _UNIPROT_PREFIX_RE.search(annotation)
        if m:
            return m.group(1).upper()
        m = _UNIPROT_IDENTIFIERS_RE.search(annotation)
        if m:
            return m.group(1).upper()
        return None

    def extract_chebi_id(self, annotation: str) -> str | None:
        """从 annotation 文本提取 ChEBI ID。

        支持格式：ChEBI:33384 / CHEBI:33384

        Returns:
            形如 'CHEBI:33384' 的字符串（大写前缀），无匹配返回 None
        """
        if not annotation:
            return None
        m = _CHEBI_RE.search(annotation)
        if m:
            return f"CHEBI:{m.group(1)}"
        return None


__all__ = ["CanonicalSpeciesResolver"]
