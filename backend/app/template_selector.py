# BioDynamics Agent - Template Selector Skill（规则引擎）
# 对应 修复提示词1.md §二.2 与 EGF-EGFR错误结论根因与后续修复计划报告.md §5.5
#
# 目标：让模板选择不依赖 LLM 单点决策，而是基于规则引擎：
#   1. 关键词匹配（最高优先级）
#   2. mechanism 投票（基于边机制分布）
#   3. SBML grounding（用户输入含 BIOMD*）
#   4. LLM 输出复核（与规则冲突时，规则置信度 > 0.8 时以规则为准）
#
# 不再使用 LLM 直接选择模板，规避 Cascade_Activation 被错误用于 EGF-EGFR 级联的问题。

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 模板白名单（修复提示词1.md §二.2）
# -----------------------------------------------------------------------------
TEMPLATE_WHITELIST: tuple[str, ...] = (
    "Signaling_Cascade_Phos",  # 信号级联磷酸化（EGF-EGFR 默认）
    "Simple_Inhibition",       # 单药物抑制
    "Simple_Activation",       # 单激活子激活
    "Cascade_Activation",      # 通用激活级联（需补质量守恒补丁）
    "Cascade_Inhibition",      # 级联抑制
    "PKPD_OneCompartment",     # 一室 PK/PD
    "PKPD_TwoCompartment",     # 二室 PK/PD
    "Combination",             # 联合用药
    "DoseSweep",               # 剂量递增
)

# 信号级联磷酸化专用模板集合
PHOS_CASCADE_TEMPLATES: frozenset[str] = frozenset({"Signaling_Cascade_Phos"})

# 通用级联模板集合（缺乏质量守恒，仅作降级）
CASCADE_TEMPLATES: frozenset[str] = frozenset({
    "Cascade_Activation", "Cascade_Inhibition",
})

# PK/PD 模板集合
PKPD_TEMPLATES: frozenset[str] = frozenset({
    "PKPD_OneCompartment", "PKPD_TwoCompartment",
})

# 模板降级映射：原模板不可用时回退到兼容模板
TEMPLATE_FALLBACK_MAP: dict[str, str] = {
    "Signaling_Cascade_Phos": "Cascade_Activation",  # 需补质量守恒
    "PKPD_TwoCompartment": "PKPD_OneCompartment",
}


# -----------------------------------------------------------------------------
# 深度审核报告 §2.4 模板复杂度分层（Tier 1/2/3）
# -----------------------------------------------------------------------------
# Tier 1: Simple Reaction (< 5 物种，单一反应类型)
# Tier 2: Pathway Chain (线性通路，2-5 个反应步骤)
# Tier 3: Cascade System (复杂级联，如 MAPK，多级磷酸化/多药物组合)
TEMPLATE_TIER_1: frozenset[str] = frozenset({
    "Simple_Inhibition", "Simple_Activation",
})
TEMPLATE_TIER_2: frozenset[str] = frozenset({
    "Cascade_Activation", "Cascade_Inhibition",
    "PKPD_OneCompartment", "PKPD_TwoCompartment",
})
TEMPLATE_TIER_3: frozenset[str] = frozenset({
    "Signaling_Cascade_Phos", "Combination", "DoseSweep",
})

# 单模板文件行数上限（深度审核报告 §2.4）
TEMPLATE_MAX_LINES: int = 300


def get_template_tier(template_name: str) -> int:
    """返回模板的复杂度分层（1/2/3）。

    Tier 1: Simple Reaction (< 5 物种) — 单一反应类型，线性动力学。
    Tier 2: Pathway Chain (线性通路) — 2-5 步串联反应，无分支。
    Tier 3: Cascade System (复杂级联) — 多级磷酸化/多药物组合/剂量递增，
            必须拆分为子模块通过 Jinja include 组合。

    未注册的模板默认为 Tier 1（最低复杂度）。
    """
    if template_name in TEMPLATE_TIER_3:
        return 3
    if template_name in TEMPLATE_TIER_2:
        return 2
    return 1


def check_template_line_limits(template_dir: Any = None) -> dict[str, Any]:
    """检查所有 .j2 模板文件是否超过行数上限。

    深度审核报告 §2.4：单个模板文件禁止超过 300 行。
    对于 Tier 3 系统，主模板通过 Jinja include 组合子模块，
    子模块（以 _ 前缀命名）同样受行数限制。

    Returns:
        {"passed": bool, "violations": [{template, lines, limit}], "checked": int}
    """
    from pathlib import Path
    if template_dir is None:
        template_dir = Path(__file__).resolve().parent / "ode_templates"
    else:
        template_dir = Path(template_dir)

    violations: list[dict[str, Any]] = []
    checked = 0
    for j2_file in sorted(template_dir.glob("*.j2")):
        try:
            with open(j2_file, encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
        except Exception as exc:
            logger.warning("模板行数检查失败 (%s)：%s", j2_file.name, exc)
            continue
        checked += 1
        if line_count > TEMPLATE_MAX_LINES:
            violations.append({
                "template": j2_file.name,
                "lines": line_count,
                "limit": TEMPLATE_MAX_LINES,
            })
            logger.warning(
                "模板行数超限：%s (%d > %d)", j2_file.name, line_count, TEMPLATE_MAX_LINES
            )
    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "checked": checked,
    }


# -----------------------------------------------------------------------------
# 关键词规则引擎
# -----------------------------------------------------------------------------
# 信号级联磷酸化关键词：触发即强制 Signaling_Cascade_Phos
_SIGNALING_CASCADE_KEYWORDS: tuple[str, ...] = (
    # 配体-受体-磷酸化关键词
    "EGF", "EGFR", "pEGFR", "phosphorylation", "磷酸化", "受体", "receptor",
    "配体", "ligand", "二聚化", "dimerization", "自磷酸化", "autophosphorylation",
    # 下游级联
    "Shc", "Grb2", "SOS", "Ras", "Raf", "MEK", "MAPK", "ERK",
    "pShc", "pRaf", "pMEK", "pMAPK", "pERK",
    # 通用信号级联
    "signaling cascade", "信号级联", "信号通路", "signaling pathway",
    "kinase cascade", "激酶级联",
    # 文献引用
    "BIOMD0000000205", "BIOMD0000000010", "BIOMD0000000056",
)

# 药物抑制关键词：触发即强制 Simple_Inhibition 或 PKPD 系列
_DRUG_INHIBITION_KEYWORDS: tuple[str, ...] = (
    "inhibitor", "抑制", "IC50", "EC50", "Ki", "drug", "药物",
    "dose", "剂量", "给药", "administration", "treatment",
    "加拉替尼", "吉非替尼", "gefitinib", "galunisertib", "imatinib",
)

# PK/PD 强信号关键词（需房室模型）
_PKPD_KEYWORDS: tuple[str, ...] = (
    "PK/PD", "PKPD", "pharmacokinetics", "药代动力学",
    "compartment", "房室", "IV", "oral", "静脉", "口服",
    "Emax", "Hill equation", "剂量递增", "dose-response",
    "临床剂量", "clinical dose",
)

# 联合用药关键词
_COMBINATION_KEYWORDS: tuple[str, ...] = (
    "combination", "联合用药", "协同", "synergy", "antagonism", "拮抗",
    "Chou-Talalay", "combination index", "CI",
)


@dataclass(frozen=True)
class TemplateSelection:
    """模板选择结果。"""
    template: str
    confidence: float              # 0.0-1.0
    reason: str                    # 规则命中说明
    rule_source: str               # "keyword" | "mechanism_vote" | "sbml_grounding" | "llm"
    override_llm: bool = False     # 是否覆盖 LLM 的选择


# -----------------------------------------------------------------------------
# TemplateSelectorSkill
# -----------------------------------------------------------------------------
class TemplateSelectorSkill:
    """规则引擎：根据用户输入与 KG 边机制，输出推荐模板与置信度。

    规则优先级（从高到低）：
    1. 关键词匹配（强信号优先）；
    2. mechanism 投票（基于边 mechanism 字段分布）；
    3. SBML grounding（用户输入含 BIOMD* 时）；
    4. 单 inhibition 边强制 Simple_Inhibition；
    5. LLM 输出（最低优先级，仅作 tie-breaker）。
    """

    def select(
        self,
        user_input: str,
        edges: list[dict[str, Any]] | None = None,
        llm_template: str = "",
        sbml_model_id: str = "",
        pkpd_profile: dict[str, Any] | None = None,
    ) -> TemplateSelection:
        """根据多源输入选择模板。

        Args:
            user_input: 用户原始输入文本。
            edges: 知识图谱边列表（含 source/target/interaction/mechanism 字段）。
            llm_template: N2 LLM 输出的 template（最低优先级，仅作 tie-breaker）。
            sbml_model_id: 用户输入显式提到的 BIOMD*/MODEL* ID。
            pkpd_profile: worker_pkpd 推断的 PK/PD 模型（若已激活）。

        Returns:
            TemplateSelection 对象。
        """
        edges = edges or []
        text = user_input or ""
        text_lower = text.lower()

        # === 规则 1: PK/PD 优先（最高优先级，因 worker_pkpd 已决策）===
        if pkpd_profile and pkpd_profile.get("drug_name") and pkpd_profile.get("drug_target"):
            compartment = str(pkpd_profile.get("compartment", "1-compartment")).lower()
            if "2" in compartment or "two" in compartment:
                return TemplateSelection(
                    template="PKPD_TwoCompartment",
                    confidence=0.95,
                    reason="PK/PD 推断已激活，房室模型=2-compartment",
                    rule_source="keyword",
                    override_llm=True,
                )
            return TemplateSelection(
                template="PKPD_OneCompartment",
                confidence=0.95,
                reason="PK/PD 推断已激活，房室模型=1-compartment",
                rule_source="keyword",
                override_llm=True,
            )

        # === 规则 2: 联合用药（≥2 药物 + inhibition 边 ≥2）===
        inhibition_edges = [
            e for e in edges
            if str(e.get("interaction", "")).lower() == "inhibition"
        ]
        if _matches_any(text_lower, _COMBINATION_KEYWORDS) and len(inhibition_edges) >= 2:
            return TemplateSelection(
                template="Combination",
                confidence=0.9,
                reason="用户输入含联合用药关键词 + ≥2 inhibition 边",
                rule_source="keyword",
                override_llm=True,
            )

        # === 规则 3: 信号级联磷酸化（EGF-EGFR 等受体信号级联）===
        # 关键：避免 LLM 误选 Cascade_Activation
        if _matches_any(text_lower, _SIGNALING_CASCADE_KEYWORDS):
            # 进一步检查是否有 binding + phosphorylation 边
            mechanisms = [str(e.get("mechanism", "")).lower() for e in edges]
            has_binding = any(m == "binding" for m in mechanisms)
            has_phosphorylation = any(m == "phosphorylation" for m in mechanisms)
            if has_binding or has_phosphorylation or len(edges) >= 3:
                return TemplateSelection(
                    template="Signaling_Cascade_Phos",
                    confidence=0.92,
                    reason=(
                        "用户输入命中信号级联磷酸化关键词 + "
                        f"KG 含 binding/phosphorylation 边或边数≥3（{len(edges)} 条）"
                    ),
                    rule_source="keyword",
                    override_llm=True,
                )

        # === 规则 4: 单 inhibition 边 → Simple_Inhibition ===
        if len(edges) == 1 and len(inhibition_edges) == 1:
            return TemplateSelection(
                template="Simple_Inhibition",
                confidence=0.85,
                reason="单 inhibition 边场景，强制 Simple_Inhibition",
                rule_source="keyword",
                override_llm=True,
            )

        # === 规则 5: 药物抑制但边数 > 1 → Cascade_Inhibition 或 Simple_Inhibition ===
        if _matches_any(text_lower, _DRUG_INHIBITION_KEYWORDS):
            if len(inhibition_edges) >= 2:
                return TemplateSelection(
                    template="Cascade_Inhibition",
                    confidence=0.75,
                    reason=f"药物抑制关键词 + {len(inhibition_edges)} inhibition 边",
                    rule_source="keyword",
                    override_llm=True,
                )
            if len(inhibition_edges) == 1 and len(edges) == 1:
                # 已在规则 4 处理，此处兜底
                return TemplateSelection(
                    template="Simple_Inhibition",
                    confidence=0.85,
                    reason="单药物抑制边",
                    rule_source="keyword",
                    override_llm=True,
                )

        # === 规则 6: SBML grounding（用户输入含 BIOMD*）===
        if sbml_model_id:
            # 检查 SBML ID 是否为信号级联模型
            if sbml_model_id.upper() in {
                "BIOMD0000000205",  # EGF-EGFR (Schoeberl/Ung)
                "BIOMD0000000010",   # EGF-EGFR (Schoeberl 2002)
                "BIOMD0000000056",   # MAPK cascade
                "BIOMD0000000567",
            }:
                return TemplateSelection(
                    template="Signaling_Cascade_Phos",
                    confidence=0.95,
                    reason=f"SBML grounding: {sbml_model_id} 是已知的信号级联模型",
                    rule_source="sbml_grounding",
                    override_llm=True,
                )

        # === 规则 7: mechanism 投票（基于边 mechanism 分布）===
        vote_result = self._vote_by_mechanism(edges)
        if vote_result:
            return vote_result

        # === 规则 8: LLM 输出（最低优先级，仅作 tie-breaker）===
        if llm_template and llm_template in TEMPLATE_WHITELIST:
            return TemplateSelection(
                template=llm_template,
                confidence=0.5,
                reason=f"无规则命中，采用 LLM 输出: {llm_template}",
                rule_source="llm",
                override_llm=False,
            )

        # 兜底：Simple_Inhibition
        return TemplateSelection(
            template="Simple_Inhibition",
            confidence=0.3,
            reason="无任何规则命中，回退到 Simple_Inhibition",
            rule_source="fallback",
            override_llm=False,
        )

    # -------------------------------------------------------------------------
    # 内部辅助：mechanism 投票
    # -------------------------------------------------------------------------
    @staticmethod
    def _vote_by_mechanism(edges: list[dict[str, Any]]) -> TemplateSelection | None:
        """基于边 mechanism 字段投票选择模板。

        - binding + phosphorylation 占比 > 50% → Signaling_Cascade_Phos
        - inhibition 占比 > 60% → Cascade_Inhibition
        - 全部 activation → Cascade_Activation
        """
        if not edges:
            return None
        mechanisms = [
            str(e.get("mechanism", "activation")).lower() for e in edges
        ]
        total = len(mechanisms)
        binding_count = sum(1 for m in mechanisms if m == "binding")
        phos_count = sum(1 for m in mechanisms if m == "phosphorylation")
        inhib_count = sum(1 for m in mechanisms if m == "inhibition")
        activation_count = sum(
            1 for m in mechanisms if m == "activation"
        )

        # 信号级联磷酸化：binding + phosphorylation 占比 > 50%
        if (binding_count + phos_count) / total > 0.5:
            return TemplateSelection(
                template="Signaling_Cascade_Phos",
                confidence=0.85,
                reason=(
                    f"mechanism 投票：binding={binding_count} + "
                    f"phosphorylation={phos_count} 占比 > 50%"
                ),
                rule_source="mechanism_vote",
                override_llm=True,
            )

        # 级联抑制：inhibition 占比 > 60%
        if inhib_count / total > 0.6:
            return TemplateSelection(
                template="Cascade_Inhibition",
                confidence=0.75,
                reason=f"mechanism 投票：inhibition={inhib_count} 占比 > 60%",
                rule_source="mechanism_vote",
                override_llm=True,
            )

        # 全部 activation：Cascade_Activation（注意：此模板缺质量守恒，需谨慎）
        if activation_count == total and total >= 2:
            return TemplateSelection(
                template="Cascade_Activation",
                confidence=0.6,
                reason=f"mechanism 投票：全部 activation（{total} 条边）",
                rule_source="mechanism_vote",
                override_llm=False,  # 让 LLM 有机会改为更优模板
            )

        return None


# -----------------------------------------------------------------------------
# 便捷函数
# -----------------------------------------------------------------------------
def select_template(
    user_input: str,
    edges: list[dict[str, Any]] | None = None,
    llm_template: str = "",
    sbml_model_id: str = "",
    pkpd_profile: dict[str, Any] | None = None,
) -> TemplateSelection:
    """全局入口：调用 TemplateSelectorSkill 选择模板。"""
    skill = TemplateSelectorSkill()
    return skill.select(
        user_input=user_input,
        edges=edges,
        llm_template=llm_template,
        sbml_model_id=sbml_model_id,
        pkpd_profile=pkpd_profile,
    )


def _matches_any(text_lower: str, keywords: tuple[str, ...]) -> bool:
    """检查文本是否匹配任一关键词（大小写不敏感）。"""
    return any(kw.lower() in text_lower for kw in keywords)


# -----------------------------------------------------------------------------
# 时间尺度分层（修复提示词1.md §5.1.2）
# -----------------------------------------------------------------------------
def get_simulation_time_scale(template_name: str) -> tuple[float, int, str]:
    """根据模板返回 (t_end, n_eval, time_unit)。

    确保不同模板的仿真时长与生物学时间尺度匹配：
    - Signaling_Cascade_Phos: 120 min（pEGFR 5-10 min + MAPK 60-120 min）
    - Cascade_*: 60 min（默认分钟，避免 48h 错误）
    - Simple_*: 48 h（药物代谢小时级）
    - PKPD_*: 48 h（房室模型小时级）
    """
    if template_name in PHOS_CASCADE_TEMPLATES:
        return 120.0, 300, "min"
    if template_name in CASCADE_TEMPLATES:
        # 级联模板也用分钟，避免 EGFR 磷酸化被错误用 48h
        return 60.0, 200, "min"
    if template_name in PKPD_TEMPLATES:
        return 48.0, 200, "h"
    if template_name == "Combination":
        return 48.0, 200, "h"
    if template_name == "DoseSweep":
        return 48.0, 100, "h"
    # Simple_Inhibition / Simple_Activation 等
    return 48.0, 200, "h"
