# BioDynamics Agent v4 - Level 1 Internal Consistency Validation (Phase 5 / Task 5.2)
#
# Level1InternalValidator 主类 + LangGraph hook 节点。
# 职责：内部一致性验证（mass conservation / non-negative / steady state /
#   numerical stability / constraint satisfaction）。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_VALIDATION_PYRAMID_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段（network_json / parameters / ode_model 等）
# 3. 仅消费 P1/P2/P3 产出（v4_ode_system / v4_reaction_ir / v4_pathway_graph）
# 4. 失败降级：任何异常都返回 pass=False（阻塞流水线），但不抛异常
# 5. 输出写入 state["v4_validation_report"]["level1"]（新增 v4 字段）
#
# 对应 spec.md Part 4 Level 1（第 278-282 行）
#
# 依赖：
# - app.config.settings（Feature Flag）
# - app.reaction_ir_v2.schema（Constraint / SpeciesV2 / ReactionIRv2，仅用于类型提示）

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Level1InternalValidator 主类
# =============================================================================
class Level1InternalValidator:
    """Level 1 内部一致性验证器。

    主入口 validate(state) 执行 5 项检查：
    1. mass conservation（受体/蛋白池守恒，误差阈值 5%）
    2. non-negative concentration（负浓度风险检测）
    3. steady state（未刺激状态稳态可达性）
    4. numerical stability（NaN/Inf + stiff system 检测）
    5. constraint satisfaction（P2 Constraint schema 全部校验）

    失败策略（对应 spec.md 第 281 行）：
    - mass conservation 误差 > 5% → pass=False，阻塞流水线
    - 出现 NaN/Inf 风险 → pass=False，阻塞流水线
    - 其他检查失败 → 记录 violations 但不影响 pass（仅 mass + numerical 决定 pass）

    用法：
        validator = Level1InternalValidator()
        report = validator.validate(state)
        # report = {pass, mass_conservation_error, ...}
    """

    # 误差阈值：5%（对应 spec.md 第 281 行）
    MASS_CONSERVATION_THRESHOLD: float = 0.05
    # Stiff system 判定：max_rate / min_rate > 1e6
    STIFF_SYSTEM_RATIO: float = 1e6

    def validate(self, state: dict[str, Any]) -> dict[str, Any]:
        """主入口：执行 Level 1 全部 5 项检查。

        Args:
            state: LangGraph 全局状态，含 v4_ode_system / v4_reaction_ir / v4_pathway_graph

        Returns:
            Level 1 报告 dict（对应 spec.md 第 280 行）：
            {
                pass: bool,
                mass_conservation_error: float,
                non_negative_violations: list,
                steady_state_check: bool,
                numerical_stability: bool,
                constraint_violations: list
            }
            异常时返回 pass=False（阻塞流水线），不抛异常（铁律 #6）。
        """
        try:
            ode_system = state.get("v4_ode_system") or {}
            reaction_ir = state.get("v4_reaction_ir") or {}

            ode_code = ""
            if isinstance(ode_system, dict):
                ode_code = ode_system.get("ode_code", "") or ""

            # 执行 5 项检查
            mass_error, _mass_violations = self._check_mass_conservation(
                ode_code, reaction_ir
            )
            non_neg_violations = self._check_non_negative(ode_code)
            steady_state_ok = self._check_steady_state(ode_code)
            numerical_stable, _stability_violations = self._check_numerical_stability(
                ode_code
            )
            constraint_violations = self._check_constraint_satisfaction(reaction_ir)

            # 失败策略：mass conservation 误差 > 5% 或 NaN/Inf → pass=False
            mass_pass = mass_error <= self.MASS_CONSERVATION_THRESHOLD
            pass_flag = bool(mass_pass and numerical_stable)

            return {
                "pass": pass_flag,
                "mass_conservation_error": mass_error,
                "non_negative_violations": non_neg_violations,
                "steady_state_check": steady_state_ok,
                "numerical_stability": numerical_stable,
                "constraint_violations": constraint_violations,
            }
        except Exception as exc:
            # 铁律 #6：失败降级返回 pass=False（阻塞流水线），但不抛异常
            logger.warning(
                "Level1InternalValidator.validate 失败，降级 pass=False: %s", exc
            )
            return {
                "pass": False,
                "mass_conservation_error": 1.0,
                "non_negative_violations": [],
                "steady_state_check": False,
                "numerical_stability": False,
                "constraint_violations": [],
                "error": str(exc),
            }

    # =========================================================================
    # SubTask 5.2.2: Mass Conservation 检查
    # =========================================================================
    def _check_mass_conservation(
        self, ode_code: str, reaction_ir: dict
    ) -> tuple[float, list]:
        """检查质量守恒（受体总量 / 蛋白池守恒）。

        检查策略：
        - 解析 reaction_ir.constraints 中 type=mass_conservation 的约束
        - 对每个约束表达式（如 "EGFR + pEGFR + EGF_EGFR = EGFR_total"）：
          - 提取 LHS 各物种的初始浓度之和
          - 与 RHS 总量比较，计算相对误差
        - 误差阈值 5%（sum of fractions should be <= 1.0 + 5% tolerance）

        Args:
            ode_code: ODE 代码字符串（用于辅助分析，当前主要依赖 reaction_ir）
            reaction_ir: Reaction IR dict（含 constraints + species）

        Returns:
            (max_error, violation_list)
            - max_error: 所有质量守恒约束中的最大相对误差（0.0 表示无误差或无约束）
            - violation_list: 每条含 constraint_name / expected / actual / diff / reason
        """
        max_error = 0.0
        violations: list[dict[str, Any]] = []

        constraints = self._get_constraints(reaction_ir)
        species_conc = self._get_species_concentrations(reaction_ir)

        for c in constraints:
            if self._get_field(c, "type", "") != "mass_conservation":
                continue
            expr = self._get_field(c, "expression", "")
            tolerance = self._get_field(c, "tolerance", self.MASS_CONSERVATION_THRESHOLD)
            provenance = self._get_field(c, "provenance", "")

            # 解析表达式 "A + B + C = D_total"
            if "=" not in expr:
                violations.append({
                    "constraint_name": provenance or expr,
                    "expected": None,
                    "actual": None,
                    "diff": 1.0,
                    "reason": f"Invalid mass_conservation expression (no '='): {expr}",
                })
                max_error = max(max_error, 1.0)
                continue

            lhs, rhs = expr.split("=", 1)
            lhs_names = [s.strip().replace("-", "_") for s in lhs.split("+") if s.strip()]
            rhs_name = rhs.strip().replace("-", "_")

            # 汇总 LHS 各物种初始浓度
            lhs_sum = sum(species_conc.get(name, 0.0) for name in lhs_names)

            # 获取 RHS 总量
            rhs_total = species_conc.get(rhs_name, None)
            if rhs_total is None:
                # RHS 不是已知物种（可能是参数），无法数值校验，跳过
                logger.debug(
                    "mass_conservation: RHS '%s' 不是已知物种，跳过数值校验",
                    rhs_name,
                )
                continue

            if rhs_total == 0:
                # 总量为 0 时，误差 = LHS 求和绝对值
                error = abs(lhs_sum)
            else:
                error = abs(lhs_sum - rhs_total) / abs(rhs_total)

            max_error = max(max_error, error)

            if error > tolerance:
                violations.append({
                    "constraint_name": provenance or expr,
                    "expected": rhs_total,
                    "actual": lhs_sum,
                    "diff": error,
                    "reason": f"Mass conservation violated: {expr} "
                              f"(expected {rhs_total}, got {lhs_sum})",
                })

        return (max_error, violations)

    # =========================================================================
    # SubTask 5.2.3: Non-Negative Concentration 检查
    # =========================================================================
    def _check_non_negative(self, ode_code: str) -> list:
        """检查是否有 species 浓度可能变为负数。

        检查策略：
        - 解析 ODE 代码中的 dX/dt = ... 方程
        - 检测常量降解项（如 dX/dt = ... - 0.5，不与 X 相乘）→ 可能导致负浓度
        - 比例降解（如 dX/dt = ... - k_deg * X）是安全的（渐近趋于 0），不报违规

        Args:
            ode_code: ODE 代码字符串

        Returns:
            violation_list，每条含 species_name / equation / reason
        """
        violations: list[dict[str, str]] = []
        if not ode_code:
            return violations

        for line in ode_code.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # 匹配 dX/dt = ... 或 dX_dt = ... 格式
            match = re.match(r"d(\w+)/dt\s*=\s*(.+)", stripped)
            if not match:
                continue

            species = match.group(1)
            rhs = match.group(2)

            # 检测常量降解项：- 数字（不跟 *）
            # 例如 dX/dt = X - 0.5 中的 -0.5
            # 但排除 - 0.5 * X（比例降解，安全）
            for num_match in re.finditer(
                r"-\s*(\d+\.?\d*(?:[eE][-+]?\d+)?)", rhs
            ):
                # 检查该数字后面是否紧跟 *（比例项）
                end_pos = num_match.end()
                after = rhs[end_pos:end_pos + 5].lstrip()
                if after.startswith("*"):
                    continue  # 比例降解，安全
                violations.append({
                    "species_name": species,
                    "equation": stripped,
                    "reason": (
                        f"Constant degradation term '{num_match.group(0).strip()}' "
                        "may cause negative concentration when species is small"
                    ),
                })
                break  # 每条方程只报一次

        return violations

    # =========================================================================
    # SubTask 5.2.4: Steady State 检查
    # =========================================================================
    def _check_steady_state(self, ode_code: str) -> bool:
        """检查未刺激状态下（ligand=0）系统是否能达到稳态。

        简化实现（对应 SubTask 5.2.4）：
        - 检查每个 dX/dt 方程是否含自降解项（-k*X 或 -X）
        - 若所有方程都有自降解项，稳态可达（返回 True）
        - 若有方程无自降解项，可能无法达到稳态（返回 False）

        注意：这是非充分必要条件的简化检查，能捕获明显错误。

        Args:
            ode_code: ODE 代码字符串

        Returns:
            True 表示稳态检查通过
        """
        if not ode_code:
            return True

        equation_count = 0
        all_have_degradation = True

        for line in ode_code.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            match = re.match(r"d(\w+)/dt\s*=\s*(.+)", stripped)
            if not match:
                continue

            equation_count += 1
            species = match.group(1)
            rhs = match.group(2)

            # 检测自降解项：-k*X / -X*k / -X
            patterns = [
                rf"-\s*\w+\s*\*\s*{re.escape(species)}\b",  # -k*X
                rf"-\s*{re.escape(species)}\s*\*\s*\w+",     # -X*k
                rf"-\s*{re.escape(species)}\b",               # -X (implicit k=1)
            ]
            has_degradation = any(re.search(p, rhs) for p in patterns)
            if not has_degradation:
                all_have_degradation = False

        if equation_count == 0:
            return True
        return all_have_degradation

    # =========================================================================
    # SubTask 5.2.5: Numerical Stability 检查
    # =========================================================================
    def _check_numerical_stability(self, ode_code: str) -> tuple[bool, list]:
        """检查数值稳定性（NaN/Inf 检测 + stiff system 检测）。

        检查策略：
        1. 检测除零风险：ODE 代码中 / variable（变量作除数）
        2. 检测 log(0) 风险：log(expression)
        3. 检测 stiff system：max_rate / min_rate > 1e6

        Args:
            ode_code: ODE 代码字符串

        Returns:
            (is_stable, violation_list)
            - is_stable: True 表示无数值稳定性风险
            - violation_list: 每条含 type / expression / reason
        """
        violations: list[dict[str, str]] = []
        if not ode_code:
            return (True, violations)

        # 1. 检测除零风险：/ variable
        for match in re.finditer(r"/\s*([a-zA-Z_]\w*)", ode_code):
            divisor = match.group(1)
            # 跳过已知安全变量
            if divisor in ("dt", "t", "time"):
                continue
            violations.append({
                "type": "division_risk",
                "expression": match.group(0),
                "reason": f"Division by variable '{divisor}' may cause divide-by-zero",
            })

        # 2. 检测 log(0) 风险：log(expression)
        for match in re.finditer(r"\blog\s*\(([^)]+)\)", ode_code):
            violations.append({
                "type": "log_risk",
                "expression": match.group(0),
                "reason": "log() of expression may cause log(0) or log(negative)",
            })

        # 3. 检测 stiff system：提取速率常数，计算 max/min 比率
        rates: list[float] = []
        for match in re.finditer(
            r"\bk\w*\s*=\s*(\d+\.?\d*(?:[eE][-+]?\d+)?)", ode_code
        ):
            try:
                val = float(match.group(1))
                if val > 0:
                    rates.append(val)
            except ValueError:
                pass

        if len(rates) >= 2:
            max_rate = max(rates)
            min_rate = min(rates)
            if min_rate > 0 and max_rate / min_rate > self.STIFF_SYSTEM_RATIO:
                ratio = max_rate / min_rate
                violations.append({
                    "type": "stiff_system",
                    "expression": f"max_rate={max_rate}, min_rate={min_rate}",
                    "reason": f"Stiff system: rate ratio {ratio:.2e} > {self.STIFF_SYSTEM_RATIO}",
                })

        is_stable = len(violations) == 0
        return (is_stable, violations)

    # =========================================================================
    # SubTask 5.2.6: Constraint Satisfaction 检查
    # =========================================================================
    def _check_constraint_satisfaction(self, reaction_ir: dict) -> list:
        """校验 P2 Constraint schema 全部约束。

        检查所有声明的约束是否被满足：
        - mass_conservation：LHS 求和 vs RHS 总量
        - steady_state：表达式非空
        - non_negative：所有物种初始浓度 >= 0
        - enzymatic：表达式非空
        - thermodynamic：表达式含热力学关键字

        Args:
            reaction_ir: Reaction IR dict（含 constraints + species）

        Returns:
            violation_list，每条含 constraint_name / expected / actual / diff / reason
        """
        violations: list[dict[str, Any]] = []

        constraints = self._get_constraints(reaction_ir)
        species_conc = self._get_species_concentrations(reaction_ir)

        for c in constraints:
            ctype = self._get_field(c, "type", "")
            expr = self._get_field(c, "expression", "")
            tolerance = self._get_field(c, "tolerance", self.MASS_CONSERVATION_THRESHOLD)
            provenance = self._get_field(c, "provenance", "")

            if ctype == "mass_conservation":
                self._check_mass_conservation_constraint(
                    expr, species_conc, tolerance, provenance, violations
                )
            elif ctype == "steady_state":
                if not expr.strip():
                    violations.append({
                        "constraint_name": provenance or "steady_state",
                        "expected": "non-empty expression",
                        "actual": "empty",
                        "diff": 1.0,
                        "reason": "Steady state constraint expression is empty",
                    })
            elif ctype == "non_negative":
                for sp_name, conc in species_conc.items():
                    if conc < 0:
                        violations.append({
                            "constraint_name": provenance or "non_negative",
                            "expected": ">= 0",
                            "actual": conc,
                            "diff": abs(conc),
                            "reason": f"Species '{sp_name}' has negative concentration",
                        })
            elif ctype == "enzymatic":
                if not expr.strip():
                    violations.append({
                        "constraint_name": provenance or "enzymatic",
                        "expected": "non-empty expression",
                        "actual": "empty",
                        "diff": 1.0,
                        "reason": "Enzymatic constraint expression is empty",
                    })
            elif ctype == "thermodynamic":
                keywords = ["k_eq", "k_forward", "k_reverse", "Kd", "kon", "koff"]
                if not any(kw in expr for kw in keywords):
                    violations.append({
                        "constraint_name": provenance or "thermodynamic",
                        "expected": f"one of {keywords}",
                        "actual": expr,
                        "diff": 1.0,
                        "reason": "Thermodynamic constraint missing keywords",
                    })

        return violations

    def _check_mass_conservation_constraint(
        self,
        expr: str,
        species_conc: dict[str, float],
        tolerance: float,
        provenance: str,
        violations: list,
    ) -> None:
        """检查单个 mass_conservation 约束是否满足（辅助函数）。"""
        if "=" not in expr:
            violations.append({
                "constraint_name": provenance or expr,
                "expected": "valid expression with '='",
                "actual": expr,
                "diff": 1.0,
                "reason": f"Invalid mass_conservation expression (no '='): {expr}",
            })
            return

        lhs, rhs = expr.split("=", 1)
        lhs_names = [s.strip().replace("-", "_") for s in lhs.split("+") if s.strip()]
        rhs_name = rhs.strip().replace("-", "_")

        lhs_sum = sum(species_conc.get(name, 0.0) for name in lhs_names)
        rhs_total = species_conc.get(rhs_name, None)

        if rhs_total is None:
            # RHS 不是已知物种，检查 LHS 是否引用未知物种
            missing = [n for n in lhs_names if n not in species_conc]
            if missing:
                violations.append({
                    "constraint_name": provenance or expr,
                    "expected": rhs_name,
                    "actual": lhs_sum,
                    "diff": 0.0,
                    "reason": f"Constraint references unknown species: {missing}",
                })
            return

        if rhs_total == 0:
            diff = abs(lhs_sum)
        else:
            diff = abs(lhs_sum - rhs_total) / abs(rhs_total)

        if diff > tolerance:
            violations.append({
                "constraint_name": provenance or expr,
                "expected": rhs_total,
                "actual": lhs_sum,
                "diff": diff,
                "reason": f"Mass conservation violated: {expr}",
            })

    # =========================================================================
    # 辅助函数
    # =========================================================================
    def _get_constraints(self, reaction_ir: Any) -> list:
        """从 reaction_ir 提取 constraints 列表（兼容 dict / ReactionIRv2）。"""
        if isinstance(reaction_ir, dict):
            return reaction_ir.get("constraints", []) or []
        return list(getattr(reaction_ir, "constraints", []) or [])

    def _get_species_concentrations(self, reaction_ir: Any) -> dict[str, float]:
        """从 reaction_ir 提取 species_name -> initial_concentration 映射。"""
        result: dict[str, float] = {}
        if isinstance(reaction_ir, dict):
            species_list = reaction_ir.get("species", []) or []
        else:
            species_list = list(getattr(reaction_ir, "species", []) or [])

        for sp in species_list:
            if isinstance(sp, dict):
                name = sp.get("canonical_name", "")
                conc = sp.get("initial_concentration", 0.0)
            else:
                name = getattr(sp, "canonical_name", "")
                conc = getattr(sp, "initial_concentration", 0.0)
            if name:
                result[name] = float(conc)
        return result

    @staticmethod
    def _get_field(obj: Any, field: str, default: Any = None) -> Any:
        """从 dict 或对象提取字段值。"""
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)


# =============================================================================
# LangGraph hook 节点（Feature Flag 隔离）
# =============================================================================
def level1_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：Level 1 Internal Consistency Validation hook。

    行为：
    - V4_VALIDATION_PYRAMID_ENABLED=false：直接返回空 dict（不修改 state）
    - V4_VALIDATION_PYRAMID_ENABLED=true：调用 Level1InternalValidator.validate()
      写入 state["v4_validation_report"]["level1"]

    严格遵守不可碰清单：
    - 不修改 v3 任何字段（network_json / parameters / ode_model 等）
    - 不生成 ODE / 不调用 RAG / 不做 SBML 验证
    - 失败时降级返回空 dict，不抛异常，不阻塞主流水线

    Args:
        state: LangGraph 全局状态

    Returns:
        flag=false 时返回 {}
        flag=true 时返回 {"v4_validation_report": {"level1": {...}}}
        异常时返回 {}
    """
    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not settings.effective_v4_validation_pyramid_enabled():
        logger.debug("V4_VALIDATION_PYRAMID_ENABLED effective=false，跳过 Level 1 validation")
        return {}

    try:
        validator = Level1InternalValidator()
        level1_report = validator.validate(state)
        # 与现有 v4_validation_report 合并，不覆盖 level2/level3
        # 安全读取 existing_report（state 可能为 None 或非 dict）
        existing_report: dict[str, Any] = {}
        if isinstance(state, dict):
            existing = state.get("v4_validation_report")
            if isinstance(existing, dict):
                existing_report = existing
        merged_report = {**existing_report, "level1": level1_report}
        return {"v4_validation_report": merged_report}
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("Level 1 validation hook 失败，降级跳过: %s", exc)
        return {}


__all__ = ["Level1InternalValidator", "level1_hook_node"]
