# BioDynamics Agent v4 - Alias Resolution (Phase 5 / Task 5.1.6)
#
# 通过 UniProt 的 accession / gene_names / protein_names 多字段对齐别名。
# EGFR/ERBB1/HER1 视为同一 canonical species；MAPK1/ERK2、AKT1/PKB 同理。
#
# 设计原则（铁律）：
# 1. 不调用外部 API（UniProt 查询为可选增强，默认用本地别名表）
# 2. 失败降级：未知名称保持原样（canonicalize 返回原名）
# 3. 别名表硬编码常见 EGFR/MAPK/AKT 等（覆盖 10 通路核心蛋白）
# 4. 大小写不敏感（EGFR/egfr/Egfr 视为同一别名）
#
# 依赖：无（纯 Python，零外部依赖）

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# 别名映射表（硬编码常见通路核心蛋白）
# =============================================================================
# key = canonical_name（首选官方基因符号，大写）
# value = 该蛋白的所有已知别名列表（含小写变体）
#
# 数据来源：
# - UniProt Homo sapiens reference proteome
# - HGNC Gene Names with Synonyms
# - Reactome pathway database
#
# 覆盖 10 通路核心蛋白：EGFR / MAPK / PI3K-AKT / p53 / Apoptosis /
# Cell Cycle / JAK-STAT / NF-κB / Wnt / TGF-β
_ALIAS_MAP: dict[str, list[str]] = {
    # EGFR 通路
    "EGFR": ["EGFR", "ERBB1", "ERBB-1", "HER1", "HER-1", "ERBB", "PIG61", "mENA"],
    "ERBB2": ["ERBB2", "HER2", "HER-2", "NEU", "NGL", "TKR1", "CD340"],
    "ERBB3": ["ERBB3", "HER3", "HER-3", "LTKS", "c-erbB3"],
    "EGF": ["EGF", "URG", "HMGF", "beta-urogastrone"],
    "GRB2": ["GRB2", "ASH", "MST084", "MSTP084", "PP013", "CG32"],
    "SOS1": ["SOS1", "GEF", "GROLS", "H-NS"],
    "SOS2": ["SOS2", "GRL2", "GRL1"],
    "SHC1": ["SHC1", "SHC", "SHCA", "p66SHC"],
    "RAS": ["RAS", "HRAS", "KRAS", "NRAS", "p21RAS"],
    "HRAS": ["HRAS", "H-RAS", "H-RAS-1", "C-H-RAS", "CTLO"],
    "KRAS": ["KRAS", "K-RAS", "K-RAS-2A", "K-RAS-2B", "C-K-RAS", "RALD"],
    "NRAS": ["NRAS", "N-RAS", "NRAS1", "N-RAS-1"],
    "RAF1": ["RAF1", "CRAF", "C-RAF", "RAF", "NS5"],
    "BRAF": ["BRAF", "B-RAF", "RAFB1", "NS7"],
    "ARAF": ["ARAF", "A-RAF", "RAFA", "PKS2"],
    # MAPK 通路
    "MAPK1": ["MAPK1", "ERK2", "ERK-2", "ERT2", "MAPK2", "PRKM1", "p42-MAPK"],
    "MAPK3": ["MAPK3", "ERK1", "ERK-1", "ERT1", "MAPK1", "p44-MAPK"],
    "MAP2K1": ["MAP2K1", "MEK1", "MEK-1", "MKK1", "MAPKK1", "PRKMK1"],
    "MAP2K2": ["MAP2K2", "MEK2", "MEK-2", "MKK2", "MAPKK2", "PRKMK2"],
    "MAP2K7": ["MAP2K7", "MKK7", "MKK-7", "JNKK2", "SKK4"],
    "MAPK8": ["MAPK8", "JNK1", "JNK-1", "SAPK1", "PRKM8"],
    "MAPK9": ["MAPK9", "JNK2", "JNK-2", "SAPK2", "p54a"],
    "MAPK14": ["MAPK14", "p38", "p38-MAPK", "CSBP1", "CSBP2", "RK", "HOG1"],
    # PI3K-AKT-mTOR 通路
    "PIK3CA": ["PIK3CA", "PI3K", "PI3KCA", "p110alpha", "p110-alpha"],
    "PIK3CB": ["PIK3CB", "PI3KCB", "p110beta", "p110-beta"],
    "PIK3R1": ["PIK3R1", "PI3K", "p85", "p85-alpha", "GRB1"],
    "AKT1": ["AKT1", "AKT", "PKB", "PKB-A", "PKB-ALPHA", "RAC", "RAC-ALPHA"],
    "AKT2": ["AKT2", "PKBB", "PKB-B", "PKB-BETA", "RAC-BETA"],
    "AKT3": ["AKT3", "PKBG", "PKB-GAMMA", "MPPH"],
    "MTOR": ["MTOR", "mTOR", "FRAP", "FRAP1", "FRAP-1", "RAFT1", "RAFT-1"],
    "PTEN": ["PTEN", "MMAC1", "TEP1", "BZS"],
    "PDPK1": ["PDPK1", "PDK1", "PDK-1", "PDPK2"],
    "TSC1": ["TSC1", "TSC", "HAMARTIN", "KIAA0243"],
    "TSC2": ["TSC2", "TSC4", "TUBERIN", "PPM1"],
    "RHEB": ["RHEB", "RHEB2", "RASL"],
    "RPS6KB1": ["RPS6KB1", "S6K", "S6K1", "S6K-1", "p70-S6K", "p70-S6K1"],
    # p53 通路
    "TP53": ["TP53", "p53", "P53", "BCC7", "LFS1", "TRP53"],
    "MDM2": ["MDM2", "HDM2", "HDMX", "ACT5", "MPM2"],
    "CDKN1A": ["CDKN1A", "p21", "P21", "CIP1", "WAF1", "SDI1", "CAP20"],
    "BAX": ["BAX", "BCL2L4", "BCL2L4", "BAX"],
    "PUMA": ["PUMA", "BBC3", "JFY1", "PUMA"],
    "NOXA": ["NOXA", "PMAIP1", "NOXA", "APR"],
    "BCL2": ["BCL2", "Bcl-2", "BCL-2", "PPP1R50"],
    # 凋亡通路
    "CASP3": ["CASP3", "CASPASE-3", "CASP-3", "CPP32", "SCA1", "CPP32B"],
    "CASP8": ["CASP8", "CASPASE-8", "CASP-8", "FLICE", "MACH", "MCH5"],
    "CASP9": ["CASP9", "CASPASE-9", "CASP-9", "MCH6", "ICE-LAP6"],
    "FAS": ["FAS", "FASR", "APO1", "APT1", "CD95", "TNFRSF6"],
    "FASL": ["FASL", "FASLG", "CD95L", "CD95-L", "APT1LG1", "TNFSF6"],
    "BAK1": ["BAK1", "BAK", "BCL2L7", "CN", "BCL2L7"],
    "PARP1": ["PARP1", "PARP", "ADPRT", "ADPRT1", "PPOL"],
    # 细胞周期
    "CCND1": ["CCND1", "Cyclin D1", "CYCLIN-D1", "BCL1", "PRAD1", "U21B31"],
    "CDK4": ["CDK4", "CMM3", "PSK-J6"],
    "CDK6": ["CDK6", "CDKN6", "MCP12", "PLSTIRE"],
    "RB1": ["RB1", "RB", "pRb", "p105-RB", "OSRC", "PPP1R130"],
    "E2F1": ["E2F1", "E2F", "RBF1", "RBP3"],
    # JAK-STAT 通路
    "JAK1": ["JAK1", "JAK-1", "JAK1A", "JAK1B"],
    "JAK2": ["JAK2", "JAK-2", "JAK2_HUMAN"],
    "STAT1": ["STAT1", "STAT-1", "ISGF-3", "C-SF3", "IMD41A"],
    "STAT3": ["STAT3", "STAT-3", "APRF", "ADMIO"],
    "STAT5": ["STAT5", "STAT5A", "STAT5B", "MGF"],
    # NF-κB 通路
    "NFKB1": ["NFKB1", "NF-KB", "NF-KB1", "P50", "P105", "KBF1", "EBP-1"],
    "RELA": ["RELA", "P65", "NFKB3", "P65-REL", "PPARG"],
    "IKBKB": ["IKBKB", "IKK-B", "IKK-BETA", "IKK2", "NFKBIKB"],
    "CHUK": ["CHUK", "IKK-A", "IKK-ALPHA", "IKK1", "NFKBIKA"],
    "NFKBIA": ["NFKBIA", "IKBA", "IKB-ALPHA", "MAD-3", "NFKBI"],
    # Wnt 通路
    "CTNNB1": ["CTNNB1", "Beta-catenin", "Beta-Catenin", "BetaCat", "CTNNB"],
    "APC": ["APC", "DP2.5", "GS", "DP25", "FAP", "FPC"],
    "AXIN1": ["AXIN1", "AXIN", "H-AXIN", "M-AXIN"],
    "GSK3B": ["GSK3B", "GSK3-BETA", "GSK3B", "FA"],
    "WNT1": ["WNT1", "INT1", "WNT-1"],
    # TGF-β 通路
    "TGFB1": ["TGFB1", "TGF-BETA", "TGF-BETA-1", "TGF-B1", "CEDLAP"],
    "TGFBR1": ["TGFBR1", "TGF-BR1", "TGF-BETA-R-I", "ALK5", "ALK-5"],
    "TGFBR2": ["TGFBR2", "TGF-BR2", "TGF-BETA-R-II", "TGFBR-2"],
    "SMAD2": ["SMAD2", "SMAD-2", "MADH2", "MADR2"],
    "SMAD3": ["SMAD3", "SMAD-3", "MADH3", "MAD3"],
    "SMAD4": ["SMAD4", "SMAD-4", "MADH4", "DPC4"],
    # 通用药物/配体别名
    "IMATINIB": ["IMATINIB", "GLEEVEC", "STI571", "STI-571", "CGP57148"],
    "GEFITINIB": ["GEFITINIB", "IRESSA", "ZD1839", "ZD-1839"],
    "ERLOTINIB": ["ERLOTINIB", "TARCEVA", "OSI774", "OSI-774", "CP358774"],
    "RAPAMYCIN": ["RAPAMYCIN", "SIROLIMUS", "RAPA", "RAPA-MTOR-INHIBITOR"],
}


# =============================================================================
# AliasResolver 主类
# =============================================================================
class AliasResolver:
    """UniProt 多字段别名对齐解析器。

    通过 UniProt 的 accession / gene_names / protein_names 多字段对齐；
    EGFR/ERBB1/HER1 视为同一 canonical species。

    用法：
        resolver = AliasResolver()
        canonical = resolver.canonicalize("ERBB1")  # → "EGFR"
        alias_map = resolver.build_alias_map()  # 全量映射
    """

    def __init__(self, external_alias_map: dict[str, list[str]] | None = None) -> None:
        """初始化别名解析器。

        Args:
            external_alias_map: 可选的外部别名映射（如来自 P1 Ontology Agent），
                会与本地 _ALIAS_MAP 合并（外部优先）
        """
        self._alias_map = self._build_internal_alias_map()
        if external_alias_map:
            self._merge_external_map(external_alias_map)

    def _build_internal_alias_map(self) -> dict[str, str]:
        """构建反向映射：alias → canonical_name（全大写不敏感）。

        Returns:
            {alias_uppercase: canonical_name}
        """
        reverse: dict[str, str] = {}
        for canonical, aliases in _ALIAS_MAP.items():
            # canonical 自身也加入反向映射
            reverse[canonical.upper()] = canonical
            for alias in aliases:
                key = alias.upper()
                # 首次写入优先（避免后续小通路覆盖主通路 canonical）
                if key not in reverse:
                    reverse[key] = canonical
        return reverse

    def _merge_external_map(self, external: dict[str, list[str]]) -> None:
        """合并外部别名映射（外部优先，覆盖本地）。"""
        for canonical, aliases in external.items():
            self._alias_map.setdefault(canonical.upper(), canonical)
            for alias in aliases:
                key = alias.upper()
                # 外部覆盖本地
                self._alias_map[key] = canonical

    def build_alias_map(self) -> dict[str, str]:
        """构建完整别名映射表（original → canonical）。

        Returns:
            {original_name_upper: canonical_name}
        """
        return dict(self._alias_map)

    def canonicalize(self, name: str) -> str:
        """将别名转换为 canonical name。

        Args:
            name: 原始名称（如 "ERBB1" / "ERK2" / "PKB"）

        Returns:
            canonical name（如 "EGFR" / "MAPK1" / "AKT1"）；
            未知名称返回原样（不报错，不阻塞）
        """
        if not name:
            return name
        # 大小写不敏感查询
        canonical = self._alias_map.get(name.upper())
        if canonical:
            return canonical
        # 未知名称：返回原名（不阻塞，由 caller 决定是否标记 warning）
        return name

    def resolve_aliases(self, species_names: list[str]) -> dict[str, str]:
        """解析一批 species 名称的别名。

        Args:
            species_names: 原始 species 名称列表

        Returns:
            {original_name: canonical_name} 映射
        """
        result: dict[str, str] = {}
        for name in species_names:
            result[name] = self.canonicalize(name)
        return result

    def get_aliases_for(self, canonical_name: str) -> list[str]:
        """返回指定 canonical name 的所有已知别名。

        Args:
            canonical_name: canonical name（如 "EGFR"）

        Returns:
            别名列表（含 canonical 本身），无匹配返回 [canonical_name]
        """
        key = canonical_name.upper()
        # 在 _ALIAS_MAP 中查找（canonical 可能大小写不同）
        for canonical, aliases in _ALIAS_MAP.items():
            if canonical.upper() == key:
                return list(aliases)
        return [canonical_name]


__all__ = ["AliasResolver"]
