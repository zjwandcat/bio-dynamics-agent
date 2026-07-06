# BioDynamics Agent v4 - SBO Term 常量定义
# 来源：v4 Scientific Architecture Part 4 的 17 类机制 + SBO term 映射表。
# SBO（Systems Biology Ontology）是 SBML 标准的机制本体，所有 reaction 必须携带 SBO term
# 以保证模型可追溯、可交换。本文件为常量定义，无外部依赖。

from __future__ import annotations

from enum import Enum


class SBOTerms(str, Enum):
    """17 类机制对应的 SBO term 常量。

    继承 str + Enum 使得既可作字符串使用（"SBO:0000216"），
    又可通过 SBOTerms.PHOSPHORYLATION 引用，兼顾可读性与类型安全。
    """

    # —— 修饰类（Modification）——
    PHOSPHORYLATION = "SBO:0000216"          # 激酶-底物磷酸化（强制 Michaelis-Menten）
    DEPHOSPHORYLATION = "SBO:0000330"        # 磷酸酶-底物去磷酸化
    UBIQUITINATION = "SBO:0000218"           # 泛素化（E3 依赖，如 p53-Mdm2）

    # —— 结合/组装类（Binding/Assembly）——
    BINDING = "SBO:0000177"                  # 配体-受体、蛋白-蛋白结合（可逆 mass-action）
    DIMERIZATION = "SBO:0000434"             # 受体二聚化（2A → A2）
    COMPLEX_FORMATION = "SBO:0000526"        # 多组分复合物组装（如 destruction complex）
    SEQUESTRATION = "SBO:0000169"            # 屏蔽结合（如 Bad-Bcl-2 sequestration）

    # —— 切割/交换类（Cleavage/Exchange）——
    CLEAVAGE = "SBO:0000213"                 # Caspase 切割、Notch NICD 释放（不可逆 MM）
    GTP_GDP_EXCHANGE = "SBO:0000174"         # Ras-RasGTP / Ras-RasGDP 转换（GEF/GAP 依赖）

    # —— 基因表达类（Gene Expression）——
    TRANSCRIPTION = "SBO:0000183"            # 转录（Hill 动力学，n=1-4）
    TRANSLATION = "SBO:0000184"              # 翻译（mRNA 依赖一级反应）

    # —— 转运类（Transport）——
    NUCLEAR_IMPORT = "SBO:0000186"           # 入核转运（NF-κB、STAT、SMAD）
    NUCLEAR_EXPORT = "SBO:0000187"           # 出核转运（NF-κB-IκBα）
    CYTOPLASM_TRANSLOCATION = "SBO:0000186"  # 胞质内转运（复用入核 SBO，无独立 term）

    # —— 降解类（Degradation）——
    DEGRADATION_TERM = "SBO:0000179"         # 蛋白自发降解（一级反应）
    PROTEASOMAL_DEGRADATION = "SBO:0000218"  # 泛素-蛋白酶体降解（复用 ubiquitination term）

    # —— 调控类（Regulation）——
    INHIBITION = "SBO:0000169"               # 抑制（竞争/别构，药物抑制、负反馈）
    ACTIVATION = "SBO:0000170"               # 激活（多种形式）

    def __str__(self) -> str:
        """直接返回 SBO ID 字符串，便于 json 序列化与模板拼接。"""
        return self.value


# 机制名 → SBO term 的字典视图，便于从机制名快速查 SBO ID
# 注意：degradation 使用 DEGRADATION_TERM 成员（避免与机制名混淆）
MECHANISM_TO_SBO: dict[str, str] = {
    "phosphorylation": SBOTerms.PHOSPHORYLATION.value,
    "dephosphorylation": SBOTerms.DEPHOSPHORYLATION.value,
    "ubiquitination": SBOTerms.UBIQUITINATION.value,
    "binding": SBOTerms.BINDING.value,
    "dimerization": SBOTerms.DIMERIZATION.value,
    "complex_formation": SBOTerms.COMPLEX_FORMATION.value,
    "sequestration": SBOTerms.SEQUESTRATION.value,
    "cleavage": SBOTerms.CLEAVAGE.value,
    "gtp_gdp_exchange": SBOTerms.GTP_GDP_EXCHANGE.value,
    "transcription": SBOTerms.TRANSCRIPTION.value,
    "translation": SBOTerms.TRANSLATION.value,
    "nuclear_import": SBOTerms.NUCLEAR_IMPORT.value,
    "nuclear_export": SBOTerms.NUCLEAR_EXPORT.value,
    "cytoplasm_translocation": SBOTerms.CYTOPLASM_TRANSLOCATION.value,
    "degradation": SBOTerms.DEGRADATION_TERM.value,
    "proteasomal_degradation": SBOTerms.PROTEASOMAL_DEGRADATION.value,
    "inhibition": SBOTerms.INHIBITION.value,
    "activation": SBOTerms.ACTIVATION.value,
}

# 反向映射：SBO term → 机制名，便于从 SBML 解析结果反查机制类型
SBO_TO_MECHANISM: dict[str, str] = {v: k for k, v in MECHANISM_TO_SBO.items()}


# 每类机制的默认动力学（供 P2/P3 Reaction IR 与模板引擎使用）
# P1 阶段仅定义常量，不在主流程中读取
DEFAULT_KINETICS: dict[str, str] = {
    "phosphorylation": "michaelis_menten",
    "dephosphorylation": "michaelis_menten",
    "ubiquitination": "mass_action_e3",
    "binding": "mass_action_reversible",
    "dimerization": "mass_action_dimer",
    "complex_formation": "mass_action_multi",
    "sequestration": "mass_action",
    "cleavage": "michaelis_menten_irreversible",
    "gtp_gdp_exchange": "mass_action_gef_gap",
    "transcription": "hill",
    "translation": "first_order_mrna",
    "nuclear_import": "first_order_cargo",
    "nuclear_export": "first_order_cargo",
    "cytoplasm_translocation": "first_order",
    "degradation": "first_order",
    "proteasomal_degradation": "mass_action_proteasome",
    "inhibition": "mixed",
    "activation": "mixed",
}


# 机制层级分类（对应 v4 架构 Part 4 的本体树）
MECHANISM_HIERARCHY: dict[str, list[str]] = {
    "modification": ["phosphorylation", "dephosphorylation", "ubiquitination"],
    "binding_assembly": [
        "binding", "dimerization", "complex_formation", "sequestration",
    ],
    "cleavage_exchange": ["cleavage", "gtp_gdp_exchange"],
    "gene_expression": ["transcription", "translation"],
    "transport": [
        "nuclear_import", "nuclear_export", "cytoplasm_translocation",
    ],
    "degradation": ["degradation", "proteasomal_degradation"],
    "regulation": ["inhibition", "activation"],
}


def get_sbo_term(mechanism: str) -> str | None:
    """根据机制名查询 SBO term。

    Args:
        mechanism: 机制类型名（如 "phosphorylation"）

    Returns:
        对应的 SBO ID 字符串（如 "SBO:0000216"），未知机制返回 None
    """
    return MECHANISM_TO_SBO.get(mechanism.lower())


def get_mechanism_name(sbo_term: str) -> str | None:
    """根据 SBO term 反查机制名。

    Args:
        sbo_term: SBO ID 字符串（如 "SBO:0000216"）

    Returns:
        机制类型名（如 "phosphorylation"），未知 term 返回 None
    """
    return SBO_TO_MECHANISM.get(sbo_term)
