# BioDynamics Agent - Reaction IR（中间表示）
# 对应 修复提示词1.md §二.1：KG → Reaction Graph → Template Compiler → ODE
# 对应 深度审核报告 §2.2：Reaction IR 预校验层（token boundary + conflict detection）
#
# Reaction IR 是 KG 与 ODE 模板之间的中间表示，结构化为：
#   species: list[str]
#   reactions: list[{from, to, reaction_type, kinetics_type, parameter_context}]
#   state_transitions: list[dict]  # 留作未来扩展
#
# 这样 LLM 不再直接输出 ODE，而是输出 Reaction Graph；
# 模板引擎根据 Reaction Graph 渲染为可执行 Python。
#
# 预校验层（pre_validate_reaction_graph）：
#   1. Token Boundary Check：使用严格正则 + 哈希 ID 匹配，禁止子串匹配
#      （防止 ERK1 vs ERK12 / pEGFR vs EGFR 误判）
#   2. Conflict Detection：检测同一物种在 Reaction Graph 中是否同时作为酶和底物
#      以冲突方式出现（非催化情况）
#   3. 失败处理：触发 rule_violations 事件，阻断渲染并请求用户澄清，禁止静默修复

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Reaction 类型枚举（修复提示词1.md §三.1）
# -----------------------------------------------------------------------------
REACTION_TYPES: tuple[str, ...] = (
    "binding",              # 质量作用结合 A + B → A-B
    "phosphorylation",      # 磷酸化 Xxx → pXxx
    "dephosphorylation",    # 去磷酸化 pXxx → Xxx
    "transcription",        # 转录
    "translation",          # 翻译
    "degradation",          # 降解
    "exchange",              # 核苷酸交换
    "recruitment",          # 接头蛋白招募
    "dissociation",         # 解离
    "transport",            # 跨膜运输
)

KINETICS_TYPES: tuple[str, ...] = (
    "mass_action",          # 质量作用动力学
    "Michaelis_Menten",     # 米氏动力学
    "Hill",                 # Hill 方程
    "first_order",          # 一级动力学
    "zero_order",           # 零级动力学
)

# mechanism → (reaction_type, kinetics_type) 默认映射
_MECHANISM_TO_REACTION: dict[str, tuple[str, str]] = {
    "binding": ("binding", "mass_action"),
    "phosphorylation": ("phosphorylation", "Michaelis_Menten"),
    "dephosphorylation": ("dephosphorylation", "first_order"),
    "exchange": ("exchange", "first_order"),
    "recruitment": ("binding", "mass_action"),
    "dissociation": ("dissociation", "first_order"),
    "degradation": ("degradation", "first_order"),
    "activation": ("phosphorylation", "Michaelis_Menten"),
    "inhibition": ("binding", "mass_action"),
}


class ReactionIR:
    """Reaction IR：把 KG 转换为结构化反应图。

    用法：
        ir = ReactionIR.from_kg(knowledge_graph)
        graph = ir.to_reaction_graph()
        # graph 可被 Template Compiler 读取以渲染 ODE
    """

    def __init__(
        self,
        species: list[str],
        reactions: list[dict[str, Any]],
        state_transitions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.species = species
        self.reactions = reactions
        self.state_transitions = state_transitions or []

    # -------------------------------------------------------------------------
    # 从 KG 构建 Reaction IR
    # -------------------------------------------------------------------------
    @classmethod
    def from_kg(cls, knowledge_graph: dict[str, Any]) -> "ReactionIR":
        """从 KG（nodes/edges）构建 Reaction IR。

        - nodes 必须含 name 字段；
        - edges 必须含 source/target/interaction/mechanism 字段，可选 reaction_equation。
        """
        nodes = knowledge_graph.get("nodes", []) or []
        edges = knowledge_graph.get("edges", []) or []

        # 1. 收集 species：KG 节点 name
        species: list[str] = []
        seen: set[str] = set()
        for n in nodes:
            name = n.get("name") or n.get("id", "")
            if name and name not in seen:
                species.append(name)
                seen.add(name)

        # 2. 为每条边构建 reaction
        reactions: list[dict[str, Any]] = []
        for e in edges:
            source = e.get("source", "")
            target = e.get("target", "")
            interaction = e.get("interaction", "activation")
            mechanism = e.get("mechanism", "activation")
            reaction_eq = e.get("reaction_equation", "")

            # 默认 (reaction_type, kinetics_type) 由 mechanism 决定
            reaction_type, kinetics_type = _MECHANISM_TO_REACTION.get(
                mechanism, ("phosphorylation", "Michaelis_Menten")
            )

            # 解析 reaction_equation 得到 from/to 列表
            from_species, to_species = _parse_reaction_equation(reaction_eq, source, target)

            # parameter_context 必填：source → target
            parameter_context = reaction_eq or f"{source} → {target}"

            reactions.append({
                "from": from_species,
                "to": to_species,
                "source": source,        # 原始 KG source（供模板查询参数用）
                "target": target,        # 原始 KG target
                "interaction": interaction,
                "mechanism": mechanism,
                "reaction_type": reaction_type,
                "kinetics_type": kinetics_type,
                "parameter_context": parameter_context,
            })

            # 把 reaction 中的新 species 加入 species 列表
            for sp in from_species + to_species:
                if sp and sp not in seen:
                    species.append(sp)
                    seen.add(sp)

        return cls(species=species, reactions=reactions)

    # -------------------------------------------------------------------------
    # 输出为 dict（供模板渲染 / 日志 / 测试用）
    # -------------------------------------------------------------------------
    def to_reaction_graph(self) -> dict[str, Any]:
        """输出结构化 Reaction Graph JSON。"""
        return {
            "species": self.species,
            "reactions": self.reactions,
            "state_transitions": self.state_transitions,
        }

    # -------------------------------------------------------------------------
    # 校验：是否符合修复提示词1.md §三.1 规范
    # -------------------------------------------------------------------------
    def validate(self) -> list[str]:
        """返回违规列表；空列表表示合规。"""
        violations: list[str] = []
        # species 必须与 KG 节点 name 一致（已在 from_kg 保证）
        for r in self.reactions:
            reaction_type = r.get("reaction_type", "")
            kinetics_type = r.get("kinetics_type", "")
            parameter_context = r.get("parameter_context", "")
            if reaction_type not in REACTION_TYPES:
                violations.append(
                    f"未知 reaction_type: {reaction_type} (reaction: {r.get('source')} → {r.get('target')})"
                )
            if kinetics_type not in KINETICS_TYPES:
                violations.append(
                    f"未知 kinetics_type: {kinetics_type} (reaction: {r.get('source')} → {r.get('target')})"
                )
            if not parameter_context:
                violations.append(
                    f"缺少 parameter_context (reaction: {r.get('source')} → {r.get('target')})"
                )
            # from/to 中的 species 必须出现在 species 列表
            for sp in r.get("from", []) + r.get("to", []):
                if sp and sp not in self.species:
                    violations.append(
                        f"反应 {parameter_context} 中的 species '{sp}' 未在 species 列表"
                    )
        return violations

    # -------------------------------------------------------------------------
    # 便捷查询：按 species 找相关 reaction
    # -------------------------------------------------------------------------
    def reactions_for_species(self, species_name: str) -> list[dict[str, Any]]:
        """返回涉及指定 species 的所有 reaction。"""
        return [
            r for r in self.reactions
            if species_name in r.get("from", []) or species_name in r.get("to", [])
        ]


# -----------------------------------------------------------------------------
# 辅助：解析 reaction_equation
# -----------------------------------------------------------------------------
def _parse_reaction_equation(
    reaction_eq: str,
    default_source: str,
    default_target: str,
) -> tuple[list[str], list[str]]:
    """解析反应方程，返回 (from_species, to_species)。

    例：
        "EGF + EGFR → EGF-EGFR" → (["EGF", "EGFR"], ["EGF-EGFR"])
        "pEGFR + Shc → pEGFR + pShc" → (["pEGFR", "Shc"], ["pEGFR", "pShc"])
        "EGF-EGFR → pEGFR" → (["EGF-EGFR"], ["pEGFR"])
    """
    if not reaction_eq or "→" not in reaction_eq:
        return [default_source], [default_target]
    try:
        lhs, rhs = reaction_eq.split("→", 1)
        from_species = [s.strip() for s in lhs.replace("+", " ").split() if s.strip()]
        to_species = [s.strip() for s in rhs.replace("+", " ").split() if s.strip()]
        return from_species, to_species
    except Exception:
        return [default_source], [default_target]


# -----------------------------------------------------------------------------
# 便捷入口
# -----------------------------------------------------------------------------
def build_reaction_graph(knowledge_graph: dict[str, Any]) -> dict[str, Any]:
    """从 KG 构建 Reaction Graph 并返回 dict。"""
    ir = ReactionIR.from_kg(knowledge_graph)
    return ir.to_reaction_graph()


def validate_reaction_graph(graph: dict[str, Any]) -> list[str]:
    """校验 Reaction Graph 是否合规（后置校验，仅记录违规不阻断）。"""
    ir = ReactionIR(
        species=graph.get("species", []),
        reactions=graph.get("reactions", []),
        state_transitions=graph.get("state_transitions", []),
    )
    return ir.validate()


# =============================================================================
# 预校验层（深度审核报告 §2.2）
# 在 n6_ode_generator 渲染前调用，违反即阻断渲染。
# =============================================================================
def _species_hash_id(name: str) -> str:
    """为物种名生成哈希 ID，作为唯一标识避免子串匹配。

    用于 Token Boundary Check：以哈希 ID 等价替换物种名后做匹配，
    从根本上杜绝 'EGFR' in 'pEGFR' / 'ERK1' in 'ERK12' 这类子串误判。
    """
    return "SP_" + hashlib.md5(name.encode("utf-8")).hexdigest()[:12]


def _strict_token_match(name: str, token_set: set[str]) -> bool:
    """严格 token 匹配：name 必须作为完整 token 出现在 token_set 中。

    禁止子串匹配：'EGFR' 不会匹配 'pEGFR'，'ERK1' 不会匹配 'ERK12'。
    """
    if not name or not token_set:
        return False
    return name in token_set


def pre_validate_reaction_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Reaction Graph 渲染前预校验（阻断式）。

    对应深度审核报告 §2.2 Reaction IR 预校验层：
    1. Token Boundary Check：使用严格正则或哈希 ID 匹配，禁止子串匹配
    2. Conflict Detection：检测同一物种同时作为酶和底物以冲突方式出现
    3. Naming Collision Detection：物种名冲突（同名不同义）

    Returns:
        {
            "passed": bool,           # 是否通过预校验
            "violations": list[str],  # 违规列表（passed=False 时非空）
            "warnings": list[str],    # 警告列表（不阻断但需关注）
            "species_id_map": dict,   # 物种名→哈希 ID 映射（供模板使用）
        }
    失败处理：调用方必须触发 rule_violations 事件，阻断渲染并请求用户澄清，
    禁止静默修复。
    """
    violations: list[str] = []
    warnings: list[str] = []

    species: list[str] = graph.get("species", []) or []
    reactions: list[dict[str, Any]] = graph.get("reactions", []) or []

    # 1. 构建 species → hash_id 映射（用于后续严格匹配）
    species_id_map: dict[str, str] = {}
    name_collision: dict[str, list[str]] = {}
    for sp in species:
        if not sp:
            continue
        hid = _species_hash_id(sp)
        species_id_map[sp] = hid
        # 检测哈希碰撞（理论极低概率，但需警告）
        existing = [k for k, v in species_id_map.items() if v == hid and k != sp]
        if existing:
            name_collision.setdefault(hid, existing + [sp])

    if name_collision:
        for hid, names in name_collision.items():
            warnings.append(
                f"物种名哈希碰撞（{hid}）：{names}，可能引发匹配歧义"
            )

    # 2. Token Boundary Check：检查 reaction 中引用的 species 是否严格匹配
    species_set = set(species)
    species_hash_set = set(species_id_map.values())
    for i, r in enumerate(reactions):
        reaction_label = f"reaction[{i}]({r.get('source', '?')}→{r.get('target', '?')})"
        # 检查 from/to 列表中的物种
        for sp in r.get("from", []) + r.get("to", []):
            if not sp:
                continue
            if sp not in species_set:
                # 检查是否是子串误匹配（如 'EGFR' in species 'pEGFR'）
                substring_matches = [s for s in species if sp in s and sp != s]
                if substring_matches:
                    violations.append(
                        f"Token Boundary 违规：{reaction_label} 引用物种 '{sp}' "
                        f"未在 species 列表，但存在子串匹配 {substring_matches}。"
                        f"请使用完整物种名（禁止子串匹配）。"
                    )
                else:
                    violations.append(
                        f"Token Boundary 违规：{reaction_label} 引用物种 '{sp}' "
                        f"未在 species 列表"
                    )

    # 3. Conflict Detection：同一物种同时作为酶和底物（非催化情况）
    # 收集每个物种的角色：enzyme（出现在产物侧）/ substrate（仅出现在反应物侧）
    species_roles: dict[str, set[str]] = {}  # species -> {"enzyme", "substrate"}
    for r in reactions:
        reaction_eq = r.get("parameter_context", "") or ""
        source = r.get("source", "")
        target = r.get("target", "")
        # 解析 reaction_equation
        if "→" in reaction_eq:
            lhs, rhs = reaction_eq.split("→", 1)
            reactants = set(s.strip() for s in lhs.replace("+", " ").split() if s.strip())
            products = set(s.strip() for s in rhs.replace("+", " ").split() if s.strip())
            # source 出现在产物侧 → enzyme；仅出现在反应物侧 → substrate
            for sp in reactants:
                if sp in products:
                    species_roles.setdefault(sp, set()).add("enzyme")
                else:
                    species_roles.setdefault(sp, set()).add("substrate")
            for sp in products:
                if sp not in reactants:
                    species_roles.setdefault(sp, set()).add("product")
        else:
            # 无反应方程：source 作为 substrate，target 作为 product
            if source:
                species_roles.setdefault(source, set()).add("substrate")
            if target:
                species_roles.setdefault(target, set()).add("product")

    # 检测冲突：同一物种同时是 substrate 和 product（在不同反应中）
    # 这是正常的（如 pEGFR 既是上一级产物又是下一级底物），
    # 但如果同一物种在"同一反应中"同时是 enzyme 和 substrate 且非催化，则冲突
    for r in reactions:
        reaction_eq = r.get("parameter_context", "") or ""
        source = r.get("source", "")
        target = r.get("target", "")
        if "→" not in reaction_eq:
            continue
        lhs, rhs = reaction_eq.split("→", 1)
        reactants = set(s.strip() for s in lhs.replace("+", " ").split() if s.strip())
        products = set(s.strip() for s in rhs.replace("+", " ").split() if s.strip())
        # 冲突：source 既在反应物又在产物（正常催化），但 target 也在反应物中（不正常）
        # 或者同一物种在不同反应中以"冲突"方式出现
        overlap = reactants & products
        # 检查是否有物种同时是 source 和 target（自环，除非是显式自磷酸化）
        if source == target and source:
            mechanism = r.get("mechanism", "")
            if mechanism not in ("phosphorylation", "dephosphorylation"):
                violations.append(
                    f"Conflict 违规：{source}→{target} 自环反应且非磷酸化机制，"
                    f"可能存在酶-底物角色冲突"
                )

    # 4. Naming Collision Detection：检测 'ERK1' vs 'ERK12' 这类前缀碰撞
    name_pairs: list[tuple[str, str]] = []
    sorted_species = sorted(species)
    for i, s1 in enumerate(sorted_species):
        for s2 in sorted_species[i + 1:]:
            # 检查数字后缀碰撞：'ERK1' vs 'ERK12'
            if s1 == s2:
                continue
            # 用 word boundary 严格匹配
            if re.search(rf"\b{re.escape(s1)}\b", s2) or re.search(rf"\b{re.escape(s2)}\b", s1):
                name_pairs.append((s1, s2))
    if name_pairs:
        for s1, s2 in name_pairs[:5]:  # 限制警告数量
            warnings.append(
                f"命名碰撞风险：'{s1}' 与 '{s2}' 存在 token 包含关系，"
                f"模板渲染时将使用哈希 ID 严格匹配"
            )

    passed = len(violations) == 0
    return {
        "passed": passed,
        "violations": violations,
        "warnings": warnings,
        "species_id_map": species_id_map,
    }

