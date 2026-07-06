# BioDynamics Agent v4 - 5 类约束检查逻辑
# 对应 v4 Scientific Architecture Part 4 §4.4 的 5 类约束：
#   1. Mass Conservation（质量守恒）
#   2. Steady State（稳态约束）
#   3. Non-negative（非负约束）
#   4. Enzymatic（酶不被消耗）
#   5. Thermodynamic（热力学一致性）
#
# 设计原则：
# 1. 约束检查不阻塞流水线，仅返回 violations 列表
# 2. 约束可由 Reaction Builder 自动生成（如受体总量守恒），也可由用户显式注入
# 3. 约束表达式为字符串，P3 ODE 渲染时解析为数值检查

from __future__ import annotations

from typing import Any

from app.reaction_ir_v2.schema import (
    Constraint,
    ReactionIRv2,
    ReactionV2,
    SpeciesV2,
)


# =============================================================================
# 1. Mass Conservation（质量守恒）
# =============================================================================
def check_mass_conservation(ir: ReactionIRv2) -> list[str]:
    """检查质量守恒约束是否在初始条件下满足。

    约束示例："EGFR + pEGFR + EGF-EGFR = EGFR_total"

    实现策略：
    - 对每个 mass_conservation 约束，解析表达式左侧各物种的初始浓度之和
    - 与右侧总量（若可解析）比较，误差超过 tolerance 即记 violation
    - 无法解析的表达式记 warning（不阻断）

    Args:
        ir: ReactionIRv2 对象

    Returns:
        violation 描述列表（空列表表示通过）
    """
    violations: list[str] = []
    for c in ir.constraints:
        if c.type != "mass_conservation":
            continue
        # 简化实现：仅检查表达式中的物种名是否都存在于 ir.species
        # 完整的数值检查在 P3 ODE 渲染后由 Simulation Layer 执行
        species_names = {sp.canonical_name for sp in ir.species}
        # 提取表达式中可能出现的物种名（粗略：按 + / = 分割后的 token）
        tokens = (
            c.expression.replace("+", " ").replace("=", " ")
            .replace("-", " ").split()
        )
        for token in tokens:
            token = token.strip()
            if not token or token.isdigit():
                continue
            # 跳过数字与运算符
            try:
                float(token)
                continue
            except ValueError:
                pass
            if token not in species_names and not token.endswith("_total"):
                violations.append(
                    f"Mass Conservation 约束引用了未知物种 '{token}'：{c.expression}"
                )
    return violations


def auto_generate_mass_conservation(
    species_list: list[SpeciesV2],
    reactions: list[ReactionV2],
) -> list[Constraint]:
    """从反应列表自动生成质量守恒约束。

    策略：识别"受体/蛋白池"模式——若某物种同时以未磷酸化/磷酸化形式出现，
    则生成守恒约束。例如 EGFR + pEGFR = EGFR_total。

    Args:
        species_list: 物种列表
        reactions: 反应列表

    Returns:
        自动生成的约束列表
    """
    constraints: list[Constraint] = []
    # 收集所有磷酸化反应的底物与产物
    # 模式：X → pX（phosphorylation），则 X + pX 守恒
    seen_pairs: set[tuple[str, str]] = set()
    for rxn in reactions:
        if rxn.reaction_type != "phosphorylation":
            continue
        for r in rxn.reactants:
            for p in rxn.products:
                rname = _species_name(species_list, r.species_id)
                pname = _species_name(species_list, p.species_id)
                if not rname or not pname:
                    continue
                # 检测 pX / p_X 前缀模式
                if pname.lower().startswith("p" + rname.lower()):
                    pair = (rname, pname)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    constraints.append(Constraint(
                        type="mass_conservation",
                        scope="species",
                        expression=f"{rname} + {pname} = {rname}_total",
                        tolerance=0.05,
                        provenance="auto_generated:phosphorylation_pair",
                    ))
    return constraints


# =============================================================================
# 2. Steady State（稳态约束）
# =============================================================================
def check_steady_state(ir: ReactionIRv2) -> list[str]:
    """检查稳态约束：未刺激状态下某些物种应接近 0 或基线。

    示例：未刺激时 pEGFR < 1% EGFR_total。

    实现策略：检查 initial_concentration 是否满足稳态约束的预期。
    完整稳态验证在 P5 Validation Layer 执行。
    """
    violations: list[str] = []
    for c in ir.constraints:
        if c.type != "steady_state":
            continue
        # 简化：仅记录约束存在，不数值检查（P5 Validation 负责）
        # 但检查表达式非空
        if not c.expression.strip():
            violations.append(
                f"Steady State 约束表达式为空：{c.provenance or 'unknown'}"
            )
    return violations


# =============================================================================
# 3. Non-negative（非负约束）
# =============================================================================
def check_non_negative(ir: ReactionIRv2) -> list[str]:
    """检查所有物种初始浓度非负（数值保护）。"""
    violations: list[str] = []
    for sp in ir.species:
        if sp.initial_concentration < 0:
            violations.append(
                f"Non-negative 违规：物种 {sp.canonical_name} 初始浓度为负数 "
                f"({sp.initial_concentration})"
            )
    return violations


# =============================================================================
# 4. Enzymatic（酶不被消耗）
# =============================================================================
def check_enzymatic(ir: ReactionIRv2) -> list[str]:
    """检查酶催化反应中酶同时出现在 reactants 与 products（不被消耗）。

    对应 Validation Rule 8：标记为 enzyme 的 species 必须同时出现在 reactants 与 products。
    """
    violations: list[str] = []
    for rxn in ir.reactions:
        # 收集所有 enzyme 角色的 species_id
        enzyme_ids = {
            ref.species_id for ref in rxn.reactants if ref.role == "enzyme"
        }
        enzyme_ids |= {
            ref.species_id for ref in rxn.products if ref.role == "enzyme"
        }
        enzyme_ids |= {
            mod.species_id for mod in rxn.modifiers
            if mod.modifier_type == "catalytic"
        }
        if not enzyme_ids:
            continue
        # 酶必须同时出现在 reactants 与 products（不被消耗）
        reactant_ids = {ref.species_id for ref in rxn.reactants}
        product_ids = {ref.species_id for ref in rxn.products}
        # 如果酶只作为 modifier 出现（catalytic），不在 reactants/products 中也算合规
        modifier_ids = {mod.species_id for mod in rxn.modifiers}
        pure_modifier_enzymes = enzyme_ids & modifier_ids
        consumed_enzymes = enzyme_ids & reactant_ids - product_ids - pure_modifier_enzymes
        for eid in consumed_enzymes:
            violations.append(
                f"Enzymatic 违规：反应 {rxn.id} 中酶 {eid} 出现在 reactants 但不在 products，"
                f"酶被消耗（应同时出现或仅作 modifier）"
            )
    return violations


# =============================================================================
# 5. Thermodynamic（热力学一致性）
# =============================================================================
def check_thermodynamic(ir: ReactionIRv2) -> list[str]:
    """检查可逆反应的热力学一致性：K_eq = k_forward / k_reverse。

    实现策略：约束表达式应包含 K_eq / k_forward / k_reverse 等关键字。
    完整数值验证在 P3 ODE 渲染时执行（需要参数值）。
    """
    violations: list[str] = []
    for c in ir.constraints:
        if c.type != "thermodynamic":
            continue
        if not c.expression.strip():
            violations.append(
                f"Thermodynamic 约束表达式为空：{c.provenance or 'unknown'}"
            )
        # 检查表达式是否包含热力学关键字
        keywords = ["k_eq", "k_forward", "k_reverse", "Kd", "kon", "koff"]
        if not any(kw in c.expression for kw in keywords):
            violations.append(
                f"Thermodynamic 约束表达式缺少热力学关键字（k_eq/k_forward/k_reverse/Kd/kon/koff）："
                f"{c.expression}"
            )
    return violations


# =============================================================================
# 统一入口：执行全部 5 类约束检查
# =============================================================================
def check_all_constraints(ir: ReactionIRv2) -> dict[str, Any]:
    """执行全部 5 类约束检查，返回汇总报告。

    Args:
        ir: ReactionIRv2 对象

    Returns:
        {
            "passed": bool,                # 是否全部通过（无 violation）
            "violations": list[str],       # 全部 violation 列表
            "by_type": {                   # 按约束类型分组
                "mass_conservation": [...],
                "steady_state": [...],
                "non_negative": [...],
                "enzymatic": [...],
                "thermodynamic": [...],
            },
        }
    """
    by_type = {
        "mass_conservation": check_mass_conservation(ir),
        "steady_state": check_steady_state(ir),
        "non_negative": check_non_negative(ir),
        "enzymatic": check_enzymatic(ir),
        "thermodynamic": check_thermodynamic(ir),
    }
    all_violations: list[str] = []
    for v in by_type.values():
        all_violations.extend(v)
    return {
        "passed": len(all_violations) == 0,
        "violations": all_violations,
        "by_type": by_type,
    }


# =============================================================================
# 辅助函数
# =============================================================================
def _species_name(species_list: list[SpeciesV2], species_id: str) -> str:
    """根据 species_id 查找规范名。"""
    for sp in species_list:
        if sp.id == species_id:
            return sp.canonical_name
    return ""


__all__ = [
    "check_mass_conservation",
    "auto_generate_mass_conservation",
    "check_steady_state",
    "check_non_negative",
    "check_enzymatic",
    "check_thermodynamic",
    "check_all_constraints",
]
