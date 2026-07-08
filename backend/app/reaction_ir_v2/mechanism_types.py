# BioDynamics Agent v4 - 19 类机制枚举
# 对应 v4 Scientific Architecture Part 4 §4.3 的 19 类机制 + SBO term + 默认动力学映射表。
# TD-042 (IB-042): 实际枚举值为 19（17 类基础机制 + INHIBITION/ACTIVATION 调控类），原注释误写为 17。
#
# 设计原则：
# 1. 复用 P1 已定义的 app.ontology.sbo_terms 常量，不重复定义 SBO term
# 2. 提供机制 → 默认动力学映射，供 Mechanism Layer（P3）使用
# 3. 提供机制层级分类，供 Reaction Builder 识别机制类别
# 4. 提供 v3 interaction（activation/inhibition）→ v4 mechanism 的映射，供 Adapter 使用

from __future__ import annotations

from enum import Enum

from app.ontology.sbo_terms import (
    DEFAULT_KINETICS,
    MECHANISM_HIERARCHY,
    MECHANISM_TO_SBO,
    SBO_TO_MECHANISM,
    get_mechanism_name,
    get_sbo_term,
)


class MechanismType(str, Enum):
    """19 类机制枚举（对应架构 §4.3 表格）。

    继承 str + Enum 使得既可作字符串使用（"phosphorylation"），
    又可通过 MechanismType.PHOSPHORYLATION 引用，兼顾可读性与类型安全。
    顺序按架构表格：修饰 → 结合/组装 → 切割/交换 → 基因表达 → 转运 → 降解 → 调控。
    """

    # —— 修饰类（Modification）——
    PHOSPHORYLATION = "phosphorylation"          # 激酶-底物磷酸化（强制 MM）
    DEPHOSPHORYLATION = "dephosphorylation"      # 磷酸酶-底物去磷酸化
    UBIQUITINATION = "ubiquitination"            # 泛素化（E3 依赖）

    # —— 结合/组装类（Binding/Assembly）——
    BINDING = "binding"                          # 配体-受体、蛋白-蛋白结合
    DISSOCIATION = "dissociation"                # 复合物解离
    DIMERIZATION = "dimerization"                # 受体二聚化
    COMPLEX_FORMATION = "complex_formation"      # 多组分复合物组装
    SEQUESTRATION = "sequestration"              # 屏蔽结合

    # —— 切割/交换类（Cleavage/Exchange）——
    CLEAVAGE = "cleavage"                        # Caspase 切割、Notch NICD 释放
    GTP_GDP_EXCHANGE = "gtp_gdp_exchange"        # Ras-RasGTP / Ras-RasGDP 转换

    # —— 基因表达类（Gene Expression）——
    TRANSCRIPTION = "transcription"              # 转录（Hill 动力学）
    TRANSLATION = "translation"                  # 翻译（mRNA 依赖一级反应）

    # —— 转运类（Transport）——
    NUCLEAR_IMPORT = "nuclear_import"            # 入核转运
    NUCLEAR_EXPORT = "nuclear_export"            # 出核转运
    CYTOPLASM_TRANSLOCATION = "cytoplasm_translocation"  # 胞质内转运

    # —— 降解类（Degradation）——
    DEGRADATION = "degradation"                  # 蛋白自发降解
    PROTEASOMAL_DEGRADATION = "proteasomal_degradation"  # 泛素-蛋白酶体降解

    # —— 调控类（Regulation）——
    INHIBITION = "inhibition"                    # 抑制（药物、负反馈）
    ACTIVATION = "activation"                    # 激活（多种形式）

    def __str__(self) -> str:
        """直接返回机制名字符串，便于 json 序列化与日志输出。"""
        return self.value

    @property
    def sbo_term(self) -> str:
        """本机制对应的 SBO term（来自 P1 sbo_terms 常量）。"""
        sbo = get_sbo_term(self.value)
        return sbo or "SBO:0000176"  # 未知时用 SBO:0000176（biochemical reaction）兜底

    @property
    def default_kinetics(self) -> str:
        """本机制的默认动力学类型（来自 P1 DEFAULT_KINETICS 映射）。

        返回值映射到 ReactionV2.kinetics_type 的合法值：
          - michaelis_menten / mass_action_reversible / first_order / hill / mixed
          → 统一映射到 ReactionV2.kinetics_type 的 5 类：
            mass_action / Michaelis_Menten / Hill / Boolean / hybrid
        """
        raw = DEFAULT_KINETICS.get(self.value, "mass_action")
        return _normalize_kinetics_name(raw)


# =============================================================================
# 动力学名称归一化
# =============================================================================
def _normalize_kinetics_name(raw: str) -> str:
    """将 P1 DEFAULT_KINETICS 的细粒度动力学名归一化到 ReactionV2.kinetics_type 的 5 类。

    P1 定义的细粒度名（如 michaelis_menten / mass_action_reversible / first_order_mrna）
    需要归一化到 ReactionV2 schema 接受的 5 类：
      - mass_action / Michaelis_Menten / Hill / Boolean / hybrid

    归一化规则：
      - michaelis_menten / michaelis_menten_irreversible → Michaelis_Menten
      - mass_action / mass_action_reversible / mass_action_dimer / mass_action_multi
        / mass_action_e3 / mass_action_gef_gap / mass_action_proteasome → mass_action
      - hill → Hill
      - first_order / first_order_mrna / first_order_cargo → mass_action
        （v4 schema 不接受 first_order，统一归到 mass_action；ODE 模板仍可按一阶处理）
      - mixed → hybrid
      - 其他未知值 → mass_action（最安全默认）
    """
    if not raw:
        return "mass_action"
    r = raw.lower().strip()
    if "michaelis" in r:
        return "Michaelis_Menten"
    if r == "hill":
        return "Hill"
    if r in ("mixed",):
        return "hybrid"
    if r in ("boolean",):
        return "Boolean"
    # first_order / mass_action_* / 其他全部归到 mass_action
    return "mass_action"


# =============================================================================
# v3 interaction → v4 mechanism 映射（供 Adapter 使用）
# =============================================================================
# v3 的 network_json edges 仅含 interaction 字段（activation/inhibition），
# Adapter 需要根据 interaction 推断 v4 的 mechanism。
# 这是降级映射：v3 的 activation 一律映射到 v4 的 activation 机制（不强制 phosphorylation），
# 避免审计报告 §4.2 的"activation→phosphorylation 强制映射"语义错误。
_V3_INTERACTION_TO_V4_MECHANISM: dict[str, MechanismType] = {
    # 调控类
    "activation": MechanismType.ACTIVATION,
    "inhibition": MechanismType.INHIBITION,
    # 修饰类
    "phosphorylation": MechanismType.PHOSPHORYLATION,
    "dephosphorylation": MechanismType.DEPHOSPHORYLATION,
    "ubiquitination": MechanismType.UBIQUITINATION,
    # 结合/组装类
    "binding": MechanismType.BINDING,
    "dissociation": MechanismType.DISSOCIATION,
    "dimerization": MechanismType.DIMERIZATION,
    "complex_formation": MechanismType.COMPLEX_FORMATION,
    "sequestration": MechanismType.SEQUESTRATION,
    # 切割/交换类
    "cleavage": MechanismType.CLEAVAGE,
    "gtp_gdp_exchange": MechanismType.GTP_GDP_EXCHANGE,
    # 基因表达类
    "transcription": MechanismType.TRANSCRIPTION,
    "translation": MechanismType.TRANSLATION,
    # 转运类
    "nuclear_import": MechanismType.NUCLEAR_IMPORT,
    "nuclear_export": MechanismType.NUCLEAR_EXPORT,
    "cytoplasm_translocation": MechanismType.CYTOPLASM_TRANSLOCATION,
    "transport": MechanismType.NUCLEAR_IMPORT,  # v3 transport 统一映射到 nuclear_import
    # 降解类
    "degradation": MechanismType.DEGRADATION,
    "proteasomal_degradation": MechanismType.PROTEASOMAL_DEGRADATION,
}


def v3_interaction_to_mechanism(interaction: str) -> MechanismType:
    """将 v3 的 interaction 字段映射为 v4 的 MechanismType。

    未知 interaction 默认映射到 activation（最安全的正向调控语义）。
    显式禁止 v3 activation → v4 phosphorylation 的强制映射（审计 §4.2 修复）。
    """
    if not interaction:
        return MechanismType.ACTIVATION
    return _V3_INTERACTION_TO_V4_MECHANISM.get(
        interaction.lower().strip(), MechanismType.ACTIVATION
    )


# =============================================================================
# 机制类别查询
# =============================================================================
def get_mechanism_category(mechanism: str) -> str:
    """查询机制所属的类别（对应架构 §4.6 本体树）。

    Args:
        mechanism: 机制名，如 "phosphorylation"

    Returns:
        类别名："modification" / "binding_assembly" / "cleavage_exchange" /
        "gene_expression" / "transport" / "degradation" / "regulation" /
        "unknown"
    """
    for category, members in MECHANISM_HIERARCHY.items():
        if mechanism.lower() in members:
            return category
    return "unknown"


def is_enzymatic_mechanism(mechanism: str) -> bool:
    """判断是否为酶催化机制（强制 Michaelis-Menten，审计 §3.1 修复）。

    磷酸化 / 去磷酸化 / 切割 必须用 MM，禁止降级为 mass-action。
    """
    return mechanism.lower() in {
        "phosphorylation", "dephosphorylation", "cleavage",
    }


def is_transport_mechanism(mechanism: str) -> bool:
    """判断是否为转运机制（跨区室反应校验用，Validation Rule 4）。"""
    return mechanism.lower() in {
        "nuclear_import", "nuclear_export", "cytoplasm_translocation",
    }


def is_degradation_mechanism(mechanism: str) -> bool:
    """判断是否为降解机制。"""
    return mechanism.lower() in {"degradation", "proteasomal_degradation"}


__all__ = [
    "MechanismType",
    "v3_interaction_to_mechanism",
    "get_mechanism_category",
    "is_enzymatic_mechanism",
    "is_transport_mechanism",
    "is_degradation_mechanism",
]
