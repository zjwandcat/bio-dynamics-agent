# BioDynamics Agent v4 - Ontology Grounding (Phase 5 / Task 5.1.5)
#
# 为 canonical species 添加 ontology ID（HGNC/UniProt/ChEBI）。
# 所有 species 必须有 HGNC 或 UniProt ID；药物用 ChEBI ID；
# 无 ID 的 species 标记 unverified=True，不阻塞但 warning。
#
# 设计原则（铁律）：
# 1. 不调用外部 API（HGNC/UniPot 查询为可选增强，默认仅本地对齐）
# 2. 与 P1 OntologyRef 对齐（复用 v4_ontology_entities 输出）
# 3. 失败降级：无 ID 的 species 标记 unverified=True，不阻塞
# 4. 仅消费 canonical_species.py 的输出，不修改其结构
#
# 依赖：
# - app.sbml_grounder.canonical_species.CanonicalSpeciesResolver 输出
# - app.sbml_grounder.alias_resolution.AliasResolver（推断 ontology）

from __future__ import annotations

import logging
from typing import Any

from app.sbml_grounder.alias_resolution import AliasResolver

logger = logging.getLogger(__name__)


# =============================================================================
# 本地 ontology 知识库（硬编码常见蛋白的 HGNC/UniProt ID）
# =============================================================================
# 当 SBML annotation 缺失或无法提取 ID 时，用本地知识库兜底。
# 数据来源：UniProt Homo sapiens reference proteome（仅收录 10 通路核心蛋白）。
#
# key = canonical_name（与 AliasResolver.canonicalize 输出对齐）
# value = {hgnc_id, uniprot_id, chebi_id, species_type}
_LOCAL_ONTOLOGY: dict[str, dict[str, str]] = {
    # EGFR 通路
    "EGFR": {"hgnc_id": "HGNC:3236", "uniprot_id": "P00533", "species_type": "protein"},
    "ERBB2": {"hgnc_id": "HGNC:3430", "uniprot_id": "P04626", "species_type": "protein"},
    "ERBB3": {"hgnc_id": "HGNC:3431", "uniprot_id": "P21860", "species_type": "protein"},
    "EGF": {"hgnc_id": "HGNC:3229", "uniprot_id": "P01133", "species_type": "ligand"},
    "GRB2": {"hgnc_id": "HGNC:4566", "uniprot_id": "P62993", "species_type": "protein"},
    "SOS1": {"hgnc_id": "HGNC:11187", "uniprot_id": "Q07889", "species_type": "protein"},
    "SOS2": {"hgnc_id": "HGNC:11188", "uniprot_id": "Q07890", "species_type": "protein"},
    "SHC1": {"hgnc_id": "HGNC:10840", "uniprot_id": "P29353", "species_type": "protein"},
    "HRAS": {"hgnc_id": "HGNC:5173", "uniprot_id": "P01112", "species_type": "protein"},
    "KRAS": {"hgnc_id": "HGNC:6407", "uniprot_id": "P01116", "species_type": "protein"},
    "NRAS": {"hgnc_id": "HGNC:7989", "uniprot_id": "P01111", "species_type": "protein"},
    "RAF1": {"hgnc_id": "HGNC:9829", "uniprot_id": "P04049", "species_type": "protein"},
    "BRAF": {"hgnc_id": "HGNC:1097", "uniprot_id": "P15056", "species_type": "protein"},
    "ARAF": {"hgnc_id": "HGNC:672", "uniprot_id": "P10398", "species_type": "protein"},
    # MAPK 通路
    "MAPK1": {"hgnc_id": "HGNC:6871", "uniprot_id": "P28482", "species_type": "protein"},
    "MAPK3": {"hgnc_id": "HGNC:6877", "uniprot_id": "P27361", "species_type": "protein"},
    "MAP2K1": {"hgnc_id": "HGNC:6840", "uniprot_id": "Q02750", "species_type": "protein"},
    "MAP2K2": {"hgnc_id": "HGNC:6841", "uniprot_id": "P36507", "species_type": "protein"},
    # PI3K-AKT-mTOR 通路
    "PIK3CA": {"hgnc_id": "HGNC:8975", "uniprot_id": "P42336", "species_type": "protein"},
    "PIK3CB": {"hgnc_id": "HGNC:8976", "uniprot_id": "P42338", "species_type": "protein"},
    "PIK3R1": {"hgnc_id": "HGNC:8979", "uniprot_id": "P27986", "species_type": "protein"},
    "AKT1": {"hgnc_id": "HGNC:391", "uniprot_id": "P31749", "species_type": "protein"},
    "AKT2": {"hgnc_id": "HGNC:392", "uniprot_id": "P31751", "species_type": "protein"},
    "MTOR": {"hgnc_id": "HGNC:3942", "uniprot_id": "P42345", "species_type": "protein"},
    "PTEN": {"hgnc_id": "HGNC:9588", "uniprot_id": "P60484", "species_type": "protein"},
    "RPS6KB1": {"hgnc_id": "HGNC:10431", "uniprot_id": "P23443", "species_type": "protein"},
    # p53 通路
    "TP53": {"hgnc_id": "HGNC:11998", "uniprot_id": "P04637", "species_type": "protein"},
    "MDM2": {"hgnc_id": "HGNC:6973", "uniprot_id": "Q00987", "species_type": "protein"},
    "CDKN1A": {"hgnc_id": "HGNC:1784", "uniprot_id": "P38936", "species_type": "protein"},
    "BAX": {"hgnc_id": "HGNC:959", "uniprot_id": "Q07812", "species_type": "protein"},
    "BCL2": {"hgnc_id": "HGNC:990", "uniprot_id": "P10415", "species_type": "protein"},
    # 凋亡
    "CASP3": {"hgnc_id": "HGNC:1504", "uniprot_id": "P42574", "species_type": "protein"},
    "CASP8": {"hgnc_id": "HGNC:1509", "uniprot_id": "Q14790", "species_type": "protein"},
    "CASP9": {"hgnc_id": "HGNC:1511", "uniprot_id": "P55211", "species_type": "protein"},
    "FAS": {"hgnc_id": "HGNC:11920", "uniprot_id": "P25445", "species_type": "protein"},
    # 细胞周期
    "CCND1": {"hgnc_id": "HGNC:1582", "uniprot_id": "P24385", "species_type": "protein"},
    "CDK4": {"hgnc_id": "HGNC:1773", "uniprot_id": "P11802", "species_type": "protein"},
    "CDK6": {"hgnc_id": "HGNC:1779", "uniprot_id": "Q00534", "species_type": "protein"},
    "RB1": {"hgnc_id": "HGNC:9928", "uniprot_id": "P06400", "species_type": "protein"},
    "E2F1": {"hgnc_id": "HGNC:3114", "uniprot_id": "Q01094", "species_type": "protein"},
    # JAK-STAT
    "JAK1": {"hgnc_id": "HGNC:6190", "uniprot_id": "P23458", "species_type": "protein"},
    "JAK2": {"hgnc_id": "HGNC:6192", "uniprot_id": "O60674", "species_type": "protein"},
    "STAT1": {"hgnc_id": "HGNC:11364", "uniprot_id": "P42224", "species_type": "protein"},
    "STAT3": {"hgnc_id": "HGNC:11364", "uniprot_id": "P40763", "species_type": "protein"},
    "STAT5": {"hgnc_id": "HGNC:11366", "uniprot_id": "P42229", "species_type": "protein"},
    # NF-κB
    "NFKB1": {"hgnc_id": "HGNC:7794", "uniprot_id": "P19838", "species_type": "protein"},
    "RELA": {"hgnc_id": "HGNC:9955", "uniprot_id": "P04637", "species_type": "protein"},
    "IKBKB": {"hgnc_id": "HGNC:5767", "uniprot_id": "O14920", "species_type": "protein"},
    "CHUK": {"hgnc_id": "HGNC:1974", "uniprot_id": "O15111", "species_type": "protein"},
    "NFKBIA": {"hgnc_id": "HGNC:7797", "uniprot_id": "P25963", "species_type": "protein"},
    # Wnt
    "CTNNB1": {"hgnc_id": "HGNC:2514", "uniprot_id": "P35222", "species_type": "protein"},
    "APC": {"hgnc_id": "HGNC:583", "uniprot_id": "P25054", "species_type": "protein"},
    "AXIN1": {"hgnc_id": "HGNC:903", "uniprot_id": "O15169", "species_type": "protein"},
    "GSK3B": {"hgnc_id": "HGNC:4617", "uniprot_id": "P49841", "species_type": "protein"},
    # TGF-β
    "TGFB1": {"hgnc_id": "HGNC:11766", "uniprot_id": "P01137", "species_type": "protein"},
    "TGFBR1": {"hgnc_id": "HGNC:11772", "uniprot_id": "P36897", "species_type": "protein"},
    "TGFBR2": {"hgnc_id": "HGNC:11773", "uniprot_id": "P37173", "species_type": "protein"},
    "SMAD2": {"hgnc_id": "HGNC:6768", "uniprot_id": "Q15796", "species_type": "protein"},
    "SMAD3": {"hgnc_id": "HGNC:6769", "uniprot_id": "P84022", "species_type": "protein"},
    "SMAD4": {"hgnc_id": "HGNC:6770", "uniprot_id": "Q13485", "species_type": "protein"},
    # 常见药物/配体
    "IMATINIB": {"chebi_id": "CHEBI:45783", "species_type": "drug"},
    "GEFITINIB": {"chebi_id": "CHEBI:49668", "species_type": "drug"},
    "ERLOTINIB": {"chebi_id": "CHEBI:114331", "species_type": "drug"},
    "RAPAMYCIN": {"chebi_id": "CHEBI:9168", "species_type": "drug"},
}


# =============================================================================
# OntologyGrounder 主类
# =============================================================================
class OntologyGrounder:
    """Ontology Grounding 处理器。

    为 canonical species 添加 ontology ID：
    - 所有 species 必须有 HGNC 或 UniProt ID
    - 药物用 ChEBI ID
    - 无 ID 的 species 标记 unverified=True，不阻塞但 warning
    - 与 P1 OntologyRef 对齐（复用 v4_ontology_entities 输出）

    用法：
        grounder = OntologyGrounder()
        grounded = grounder.ground_species(canonical_species)
        for sp in grounded:
            print(sp["species_id"], sp["ontology_ref"], sp["verified"])
    """

    def __init__(self, alias_resolver: AliasResolver | None = None) -> None:
        self._alias_resolver = alias_resolver or AliasResolver()
        # P1 ontology_entities 反向索引（canonical_name → ontology dict）
        self._p1_index: dict[str, dict[str, Any]] = {}

    def ground_species(
        self,
        canonical_species: list[dict[str, Any]],
        p1_ontology_entities: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """为每个 species 添加 ontology ID。

        Args:
            canonical_species: CanonicalSpeciesResolver.resolve 输出，每项含
                {sbml_species_id, canonical_name, hgnc_id, uniprot_id, chebi_id, verified}
            p1_ontology_entities: 可选，P1 Ontology Agent 输出的 v4_ontology_entities，
                用于交叉验证与补充 ontology ID

        Returns:
            list[dict] 每项含：
                {species_id, canonical_name, ontology_ref, verified, warnings, source}
        """
        # 构建 P1 ontology 索引（若提供）
        if p1_ontology_entities:
            self._p1_index = self.align_with_p1_ontology(p1_ontology_entities)
        else:
            self._p1_index = {}

        result: list[dict[str, Any]] = []
        for sp in canonical_species:
            try:
                grounded = self._ground_one(sp)
                result.append(grounded)
            except Exception as exc:
                logger.warning(
                    "ontology grounding 失败 (species_id=%s): %s",
                    sp.get("sbml_species_id", "?"),
                    exc,
                )
                result.append(
                    {
                        "species_id": sp.get("sbml_species_id", ""),
                        "canonical_name": sp.get("canonical_name", ""),
                        "ontology_ref": {},
                        "verified": False,
                        "warnings": [f"grounding_failed: {exc}"],
                        "source": "fallback_on_error",
                    }
                )
        return result

    def _ground_one(self, sp: dict[str, Any]) -> dict[str, Any]:
        """为单个 species 添加 ontology ID。"""
        sp_id = sp.get("sbml_species_id", "")
        canonical_name = sp.get("canonical_name", "")
        warnings: list[str] = []

        # 1. 优先用 species 自带的 ontology（从 SBML annotation 提取）
        ontology_ref: dict[str, Any] = {}
        if sp.get("hgnc_id"):
            ontology_ref["hgnc_id"] = sp["hgnc_id"]
        if sp.get("uniprot_id"):
            ontology_ref["uniprot_id"] = sp["uniprot_id"]
        if sp.get("chebi_id"):
            ontology_ref["chebi_id"] = sp["chebi_id"]

        verified = sp.get("verified", False)
        source = sp.get("source", "")

        # 2. 若 SBML annotation 未提取到 ID，查本地知识库
        if not verified and canonical_name:
            local_entry = _LOCAL_ONTOLOGY.get(canonical_name.upper())
            if local_entry:
                if "hgnc_id" in local_entry and "hgnc_id" not in ontology_ref:
                    ontology_ref["hgnc_id"] = local_entry["hgnc_id"]
                if "uniprot_id" in local_entry and "uniprot_id" not in ontology_ref:
                    ontology_ref["uniprot_id"] = local_entry["uniprot_id"]
                if "chebi_id" in local_entry and "chebi_id" not in ontology_ref:
                    ontology_ref["chebi_id"] = local_entry["chebi_id"]
                ontology_ref["species_type"] = local_entry.get(
                    "species_type", "protein"
                )
                verified = True
                source = "local_ontology_kb"
                warnings.append(
                    f"ontology ID from local KB (canonical_name={canonical_name})"
                )

        # 3. 进一步查 P1 ontology_entities 索引（若提供）
        if not verified and self._p1_index:
            p1_entry = self._p1_index.get(canonical_name.upper())
            if p1_entry:
                if "hgnc_id" in p1_entry and "hgnc_id" not in ontology_ref:
                    ontology_ref["hgnc_id"] = p1_entry["hgnc_id"]
                if "uniprot_id" in p1_entry and "uniprot_id" not in ontology_ref:
                    ontology_ref["uniprot_id"] = p1_entry["uniprot_id"]
                if "chebi_id" in p1_entry and "chebi_id" not in ontology_ref:
                    ontology_ref["chebi_id"] = p1_entry["chebi_id"]
                ontology_ref["species_type"] = p1_entry.get(
                    "species_type", "protein"
                )
                verified = True
                source = "p1_ontology_agent"
                warnings.append(
                    f"ontology ID from P1 Ontology Agent (canonical_name={canonical_name})"
                )

        # 4. 仍未 verified：标记 unverified，不阻塞但 warning
        if not verified:
            warnings.append(
                f"no ontology ID found for species (canonical_name={canonical_name}, "
                f"sbml_species_id={sp_id}); marked unverified"
            )
            if not source:
                source = "unverified"

        # 5. 校验：protein 必须有 HGNC 或 UniProt；drug 必须有 ChEBI
        species_type = ontology_ref.get("species_type", "protein")
        if species_type == "drug":
            if "chebi_id" not in ontology_ref:
                warnings.append(
                    f"drug species missing ChEBI ID (canonical_name={canonical_name})"
                )
                verified = False
        else:
            # protein / ligand / gene 等：必须有 HGNC 或 UniProt
            if "hgnc_id" not in ontology_ref and "uniprot_id" not in ontology_ref:
                if verified:
                    # 仅在原本 verified=True 但 ID 实际缺失时降级
                    verified = False
                    warnings.append(
                        f"protein species missing HGNC and UniProt ID "
                        f"(canonical_name={canonical_name}); downgraded to unverified"
                    )

        return {
            "species_id": sp_id,
            "canonical_name": canonical_name,
            "ontology_ref": ontology_ref,
            "verified": verified,
            "warnings": warnings,
            "source": source,
        }

    def align_with_p1_ontology(
        self, ontology_entities: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """与 P1 OntologyRef 对齐（复用 P1 ontology_agent 输出）。

        P1 v4_ontology_entities 结构：
            {entities: [{name, hgnc_id, uniprot_id, chebi_id, go_terms, sbo_term,
                         species_type, verified, source}], pathway_class, warnings}

        本方法提取 entities 列表，构建 canonical_name → ontology dict 反向索引。

        Args:
            ontology_entities: P1 v4_ontology_entities dict 或 entities 列表

        Returns:
            {canonical_name_upper: ontology_dict}
        """
        if not ontology_entities:
            return {}

        # 兼容两种输入：dict（含 entities key）或直接 list[dict]
        if isinstance(ontology_entities, dict):
            entities = ontology_entities.get("entities", [])
        else:
            entities = list(ontology_entities)

        index: dict[str, dict[str, Any]] = {}
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = entity.get("name", "")
            if not name:
                continue
            ont: dict[str, Any] = {}
            if entity.get("hgnc_id"):
                ont["hgnc_id"] = entity["hgnc_id"]
            if entity.get("uniprot_id"):
                ont["uniprot_id"] = entity["uniprot_id"]
            if entity.get("chebi_id"):
                ont["chebi_id"] = entity["chebi_id"]
            ont["species_type"] = entity.get("species_type", "protein")
            ont["verified"] = entity.get("verified", False)
            # canonical_name 通过 alias_resolver 归一
            canonical = self._alias_resolver.canonicalize(name)
            index[canonical.upper()] = ont
        return index


__all__ = ["OntologyGrounder"]
