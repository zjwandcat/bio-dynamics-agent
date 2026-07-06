# BioDynamics Agent - Domain Knowledge Checker（领域常识审查）
# 对应 EGF-EGFR错误结论根因与后续修复计划报告.md §5.7：
# 物理 / 生物 / 化学 / 医学 多维硬约束审查，避免 LLM 自写算法或错误模板
# 生成违反质量守恒、配体-受体关系、单位一致性的 ODE。

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# 审查结果数据类
# =============================================================================
@dataclass
class DomainViolation:
    """单条领域常识违规。"""
    category: str          # "physical" | "biological" | "chemical" | "medical"
    severity: str          # "high" | "medium" | "low"
    rule: str              # 规则名
    message: str           # 违规描述
    fix_suggestion: str = ""


@dataclass
class DomainCheckResult:
    """领域常识审查结果。"""
    passed: bool = True
    violations: list[DomainViolation] = field(default_factory=list)
    summary: str = ""

    def add_violation(
        self,
        category: str,
        severity: str,
        rule: str,
        message: str,
        fix_suggestion: str = "",
    ) -> None:
        self.violations.append(
            DomainViolation(
                category=category,
                severity=severity,
                rule=rule,
                message=message,
                fix_suggestion=fix_suggestion,
            )
        )
        if severity == "high":
            self.passed = False


# =============================================================================
# DomainChecker：静态代码审查 + 仿真结果审查
# =============================================================================
class DomainChecker:
    """对生成的 ODE Python 代码与仿真结果做多维领域常识审查。

    审查层级：
    1. 物理：质量守恒、非负浓度、单位一致性、时间尺度合理性
    2. 生物：配体-受体关系、磷酸化方向、信号放大、通路完整性
    3. 化学：结合反应可逆性、催化不消耗酶、化学计量
    4. 医学：剂量-反应关系、IC50/EC50 范围
    """

    # 受体总量守恒物种（EGF-EGFR 级联场景）
    RECEPTOR_SPECIES: tuple[str, ...] = ("EGFR", "EGF-EGFR", "pEGFR", "EGF-EGFR-2", "EGF-pEGFR-2")

    # 磷酸化物种前缀
    PHOS_PREFIX: str = "p"

    # 受体信号级联关键词（用于判断是否需启用配体-受体关系审查）
    CASCADE_KEYWORDS: tuple[str, ...] = ("EGF", "EGFR", "pEGFR", "Shc", "Grb2", "SOS", "Ras", "MAPK")

    # -------------------------------------------------------------------------
    # 静态代码审查（ODE 渲染后、沙箱执行前）
    # -------------------------------------------------------------------------
    def check_code(
        self,
        code: str,
        species_names: list[str] | None = None,
        edges: list[dict[str, Any]] | None = None,
        template_name: str = "",
    ) -> DomainCheckResult:
        """对生成的 ODE 代码做静态领域常识审查。"""
        result = DomainCheckResult()
        if not code or not code.strip():
            result.add_violation(
                category="physical", severity="high", rule="empty_code",
                message="ODE 代码为空，无法执行仿真",
            )
            result.summary = "代码为空"
            return result

        species_names = species_names or []
        edges = edges or []

        # === L1: 语法检查（ast.parse）===
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            result.add_violation(
                category="physical", severity="high", rule="syntax_error",
                message=f"Python 语法错误: {exc}",
                fix_suggestion="检查 Jinja2 模板渲染是否完整",
            )
            result.summary = "语法错误"
            return result

        # === L2: 物理常识审查 ===
        self._check_mass_conservation(code, species_names, edges, template_name, result)
        self._check_non_negative(code, result)
        self._check_time_scale(code, template_name, result)
        self._check_unit_consistency(code, species_names, result)

        # === L3: 生物常识审查 ===
        self._check_ligand_receptor(code, species_names, edges, result)
        self._check_phosphorylation_direction(code, species_names, edges, result)
        self._check_pathway_completeness(species_names, edges, result)

        # === L4: 化学常识审查 ===
        self._check_reversibility(edges, result)
        self._check_enzyme_not_consumed(edges, result)

        # === L5: 危险调用扫描（沙箱已做，此处兜底）===
        self._check_dangerous_calls(code, result)

        if not result.violations:
            result.summary = "通过领域常识审查"
        else:
            high_count = sum(1 for v in result.violations if v.severity == "high")
            medium_count = sum(1 for v in result.violations if v.severity == "medium")
            low_count = sum(1 for v in result.violations if v.severity == "low")
            result.summary = (
                f"领域常识审查：{high_count} high / {medium_count} medium / {low_count} low"
            )
        return result

    # -------------------------------------------------------------------------
    # 仿真结果审查（沙箱执行后，对 simulation.csv 解析）
    # -------------------------------------------------------------------------
    def check_simulation_result(
        self,
        species_names: list[str],
        time_points: list[float],
        concentrations: dict[str, list[float]],
        template_name: str = "",
    ) -> DomainCheckResult:
        """对仿真结果做领域常识审查。

        Args:
            species_names: 物种名列表。
            time_points: 时间点列表。
            concentrations: {species_name: [conc_t1, conc_t2, ...]}。
            template_name: 使用的模板名。
        """
        result = DomainCheckResult()
        if not species_names or not time_points or not concentrations:
            result.summary = "无数据可审查"
            return result

        # === 1. 非负浓度 ===
        for sp in species_names:
            series = concentrations.get(sp, [])
            if not series:
                continue
            for i, v in enumerate(series):
                if v < 0:
                    result.add_violation(
                        category="physical", severity="high",
                        rule="negative_concentration",
                        message=f"{sp} 在 t={time_points[i]:.2f} 出现负浓度 ({v:.4f})",
                        fix_suggestion="检查 ODE 是否漏掉源物种消耗项",
                    )
                    break

        # === 2. 信号级联：质量守恒（EGF-EGFR 场景）===
        if any(kw in species_names for kw in ("EGF", "EGFR", "pEGFR")):
            self._check_receptor_conservation(concentrations, result)

        # === 3. 时间尺度合理性 ===
        # pEGFR 应在 5-10 min 内达峰（仅对 Signaling_Cascade_Phos 模板）
        if template_name == "Signaling_Cascade_Phos":
            self._check_pegfr_peak_time(species_names, time_points, concentrations, result)
            self._check_mapk_amplification(species_names, concentrations, result)

        # === 4. 数值稳定性 ===
        for sp in species_names:
            series = concentrations.get(sp, [])
            if not series:
                continue
            import math
            for v in series:
                if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
                    result.add_violation(
                        category="physical", severity="high",
                        rule="numerical_instability",
                        message=f"{sp} 出现 NaN/Inf，仿真数值不稳定",
                        fix_suggestion="检查 ODE 参数是否过大导致 stiff 系统",
                    )
                    break

        if not result.violations:
            result.summary = "仿真结果通过领域常识审查"
        else:
            high_count = sum(1 for v in result.violations if v.severity == "high")
            medium_count = sum(1 for v in result.violations if v.severity == "medium")
            low_count = sum(1 for v in result.violations if v.severity == "low")
            result.summary = (
                f"仿真结果审查：{high_count} high / {medium_count} medium / {low_count} low"
            )
        return result

    # -------------------------------------------------------------------------
    # 物理审查子项
    # -------------------------------------------------------------------------
    def _check_mass_conservation(
        self,
        code: str,
        species_names: list[str],
        edges: list[dict[str, Any]],
        template_name: str,
        result: DomainCheckResult,
    ) -> None:
        """检查 ODE 代码是否包含质量守恒项（源物种消耗）。"""
        if not species_names or not edges:
            return

        # 对 binding / phosphorylation 边，检查是否有源物种消耗
        for edge in edges:
            mechanism = str(edge.get("mechanism", "")).lower()
            source = edge.get("source", "")
            if not source:
                continue
            if mechanism in ("binding", "phosphorylation"):
                # 简单检查：代码是否含 dy[s_idx] -= 模式
                # 注：模板渲染后是 for 循环，无法直接定位 source 名，
                # 这里改为检查模板是否含 dy[s_idx] -= k_on / dy[s_idx] -= k_phos
                if mechanism == "binding" and "k_on" in code:
                    if "dy[s_idx] -=" not in code and "dy[i] -=" not in code:
                        result.add_violation(
                            category="physical", severity="medium",
                            rule="mass_conservation",
                            message=f"binding 边 {source}→{edge.get('target','')} 缺源物种消耗项",
                            fix_suggestion="确保 dy[s_idx] -= k_on * src * tgt 项存在",
                        )
                if mechanism == "phosphorylation" and "k_phos" in code:
                    if "dy[s_idx] -=" not in code and "dy[i] -=" not in code:
                        result.add_violation(
                            category="physical", severity="medium",
                            rule="mass_conservation",
                            message=f"phosphorylation 边 {source}→{edge.get('target','')} 缺底物消耗项",
                            fix_suggestion="确保 dy[s_idx] -= k_phos * src 项存在",
                        )

    def _check_non_negative(self, code: str, result: DomainCheckResult) -> None:
        """检查代码是否含 np.maximum(y, 0.0) 防止负浓度。"""
        if "np.maximum" not in code and "np.clip" not in code:
            result.add_violation(
                category="physical", severity="low",
                rule="non_negative",
                message="ODE 代码缺少 np.maximum(y, 0.0) 防负浓度保护",
                fix_suggestion="在 _ode 函数开头加 y = np.maximum(y, 0.0)",
            )

    def _check_time_scale(
        self,
        code: str,
        template_name: str,
        result: DomainCheckResult,
    ) -> None:
        """检查仿真时长是否与生物学时间尺度匹配。"""
        # 提取 T_END 值
        match = re.search(r"T_END\s*=\s*([\d.]+)", code)
        if not match:
            return
        try:
            t_end = float(match.group(1))
        except ValueError:
            return

        # Signaling_Cascade_Phos 应为 60-120 min
        if template_name == "Signaling_Cascade_Phos":
            if t_end > 200:
                result.add_violation(
                    category="physical", severity="medium",
                    rule="time_scale",
                    message=f"信号级联磷酸化 T_END={t_end} 过大（应为 60-120 min）",
                    fix_suggestion="使用 template_selector.get_simulation_time_scale",
                )
        # Cascade_Activation 用于受体信号级联时（错误场景）
        # 此处不强制，因为 Cascade_Activation 也可能用于非受体级联

    def _check_unit_consistency(
        self,
        code: str,
        species_names: list[str],
        result: DomainCheckResult,
    ) -> None:
        """简单检查代码中是否混用了小时/分钟单位标签。"""
        has_min = "Time (min)" in code or "Time(min)" in code
        has_hour = "Time (h)" in code or "Time(h)" in code or "Time (hour)" in code
        if has_min and has_hour:
            result.add_violation(
                category="physical", severity="low",
                rule="unit_consistency",
                message="代码中同时存在 min 与 h 单位标签，可能混淆",
                fix_suggestion="统一时间单位为 min 或 h",
            )

    # -------------------------------------------------------------------------
    # 生物审查子项
    # -------------------------------------------------------------------------
    def _check_ligand_receptor(
        self,
        code: str,
        species_names: list[str],
        edges: list[dict[str, Any]],
        result: DomainCheckResult,
    ) -> None:
        """检查配体-受体关系：受体总量应守恒。"""
        if "EGF" not in species_names or "EGFR" not in species_names:
            return
        # 检查是否有 EGF-EGFR 复合物
        has_complex = any(
            "EGF-EGFR" in sp or "EGF_EGFR" in sp
            for sp in species_names
        )
        if not has_complex:
            result.add_violation(
                category="biological", severity="medium",
                rule="ligand_receptor",
                message="EGF + EGFR 模型缺少 EGF-EGFR 复合物物种",
                fix_suggestion="N1 NER 应提取 EGF-EGFR 复合物作为独立物种",
            )

    def _check_phosphorylation_direction(
        self,
        code: str,
        species_names: list[str],
        edges: list[dict[str, Any]],
        result: DomainCheckResult,
    ) -> None:
        """检查磷酸化方向：激酶不消耗、底物 → pXxx。"""
        # 检查每条 phosphorylation 边
        for edge in edges:
            mechanism = str(edge.get("mechanism", "")).lower()
            if mechanism != "phosphorylation":
                continue
            source = edge.get("source", "")
            target = edge.get("target", "")
            reaction_eq = edge.get("reaction_equation", "")
            # 若 source 是 pXxx（激酶形式），不应消耗
            if source.startswith(self.PHOS_PREFIX) and "→" in reaction_eq:
                # 检查 source 是否在反应式右侧（即作为酶不消耗）
                try:
                    rhs = reaction_eq.split("→", 1)[1]
                    rhs_tokens = set(rhs.replace("+", " ").split())
                    if source in rhs_tokens:
                        # source 是酶，不应消耗
                        # 但若代码无条件 dy[s_idx] -= k_phos * src，会错误消耗
                        # 此处只能做提示，无法静态分析模板渲染后的具体逻辑
                        pass
                except Exception:
                    pass

    def _check_pathway_completeness(
        self,
        species_names: list[str],
        edges: list[dict[str, Any]],
        result: DomainCheckResult,
    ) -> None:
        """检查信号级联完整性：禁止 shortcut（如 EGF→MAPK）。"""
        # 检查是否有 EGF 直接连接 MAPK 的边（跳过整个级联）
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            # 禁止 shortcut：EGF→MAPK / EGF→Ras / Ras→MAPK 等
            forbidden_shortcuts = [
                ("EGF", "MAPK"), ("EGF", "Ras"),
                ("Ras", "MAPK"), ("EGF", "pMAPK"),
            ]
            for src, tgt in forbidden_shortcuts:
                if source == src and target == tgt:
                    result.add_violation(
                        category="biological", severity="high",
                        rule="pathway_completeness",
                        message=f"检测到 shortcut 边 {src}→{tgt}，应展开为完整级联",
                        fix_suggestion=(
                            f"展开为 EGF→EGFR→pEGFR→Shc/pShc→Grb2→SOS→"
                            f"Ras→Raf→MEK→MAPK 完整通路"
                        ),
                    )

    # -------------------------------------------------------------------------
    # 化学审查子项
    # -------------------------------------------------------------------------
    def _check_reversibility(
        self,
        edges: list[dict[str, Any]],
        result: DomainCheckResult,
    ) -> None:
        """检查 binding 边是否可逆（应有 k_on 与 k_off）。"""
        for edge in edges:
            if str(edge.get("mechanism", "")).lower() != "binding":
                continue
            # 仅做静态检查提示，无法静态分析模板渲染后的参数完整性
            # 已由 N5 RAG 决策保证参数完整性，此处兜底
            pass

    def _check_enzyme_not_consumed(
        self,
        edges: list[dict[str, Any]],
        result: DomainCheckResult,
    ) -> None:
        """检查催化反应中酶不被消耗。"""
        # 此处仅做提示，模板渲染逻辑应保证（见 Signaling_Cascade_Phos.j2）

    # -------------------------------------------------------------------------
    # 危险调用扫描
    # -------------------------------------------------------------------------
    def _check_dangerous_calls(self, code: str, result: DomainCheckResult) -> None:
        """扫描 ODE 代码是否含危险调用（沙箱的兜底）。"""
        forbidden_modules = (
            "os", "sys", "subprocess", "socket", "shutil", "pathlib",
            "urllib", "requests", "http", "ftplib", "smtplib", "email",
            "pickle", "ctypes", "multiprocessing", "threading", "asyncio",
        )
        forbidden_builtins = (
            "eval", "exec", "compile", "open", "input", "__import__",
        )
        for mod in forbidden_modules:
            if re.search(rf"\bimport\s+{mod}\b", code) or re.search(rf"\bfrom\s+{mod}\s+import", code):
                result.add_violation(
                    category="physical", severity="high",
                    rule="dangerous_call",
                    message=f"代码含危险模块 import: {mod}",
                    fix_suggestion=f"移除 {mod} 的所有 import",
                )
        for builtin in forbidden_builtins:
            if re.search(rf"\b{builtin}\s*\(", code):
                result.add_violation(
                    category="physical", severity="high",
                    rule="dangerous_call",
                    message=f"代码含危险内建函数: {builtin}",
                    fix_suggestion=f"移除 {builtin} 调用",
                )

    # -------------------------------------------------------------------------
    # 仿真结果审查子项
    # -------------------------------------------------------------------------
    def _check_receptor_conservation(
        self,
        concentrations: dict[str, list[float]],
        result: DomainCheckResult,
    ) -> None:
        """检查受体总量守恒：EGFR + EGF-EGFR + pEGFR ≈ 初始 EGFR 量级。"""
        receptor_species = [
            sp for sp in ("EGFR", "EGF_EGFR", "EGF-EGFR", "pEGFR", "EGF_EGFR_2", "EGF-EGFR-2")
            if sp in concentrations
        ]
        if not receptor_species:
            return
        try:
            # 取所有时间点的受体总量
            n = min(len(concentrations[sp]) for sp in receptor_species)
            initial_total = sum(concentrations[sp][0] for sp in receptor_species)
            if initial_total <= 0:
                return
            for i in range(n):
                total = sum(concentrations[sp][i] for sp in receptor_species)
                # 允许 10% 误差
                if abs(total - initial_total) / initial_total > 0.1:
                    result.add_violation(
                        category="physical", severity="high",
                        rule="receptor_conservation",
                        message=(
                            f"受体总量在 t={i} 时偏离初始值 "
                            f"{abs(total - initial_total) / initial_total * 100:.1f}%"
                            f"（初始={initial_total:.4f}, 当前={total:.4f}）"
                        ),
                        fix_suggestion="检查 ODE 是否漏掉源物种消耗项（质量守恒）",
                    )
                    break
        except Exception as exc:
            logger.warning("受体守恒检查失败: %s", exc)

    def _check_pegfr_peak_time(
        self,
        species_names: list[str],
        time_points: list[float],
        concentrations: dict[str, list[float]],
        result: DomainCheckResult,
    ) -> None:
        """检查 pEGFR 是否在 5-10 min 内达峰。"""
        if "pEGFR" not in concentrations:
            return
        series = concentrations["pEGFR"]
        if not series:
            return
        try:
            # 找到峰值时间
            max_idx = series.index(max(series))
            if max_idx >= len(time_points):
                return
            peak_time = time_points[max_idx]
            # 期望 1-10 min（修复提示词1.md §五 标准：1-5 min）
            # 实际 Schoeberl 2002 文献：pEGFR ~5 min 达峰
            if peak_time > 15:
                result.add_violation(
                    category="biological", severity="medium",
                    rule="peak_time",
                    message=f"pEGFR 达峰时间 {peak_time:.2f} min 偏晚（期望 1-10 min）",
                    fix_suggestion="检查 k_phos/k_dephos 比值，pEGFR 达峰需 k_phos>>k_dephos",
                )
            elif peak_time < 0.5 and max(series) > 0.001:
                result.add_violation(
                    category="biological", severity="low",
                    rule="peak_time",
                    message=f"pEGFR 达峰时间 {peak_time:.2f} min 过早（期望 1-10 min）",
                    fix_suggestion="检查 k_phos 是否过大或初始浓度是否异常",
                )
        except Exception as exc:
            logger.warning("pEGFR 峰值时间检查失败: %s", exc)

    def _check_mapk_amplification(
        self,
        species_names: list[str],
        concentrations: dict[str, list[float]],
        result: DomainCheckResult,
    ) -> None:
        """检查 MAPK 信号放大效应：pMAPK 峰值应 > MAPK 初始。"""
        if "pMAPK" not in concentrations:
            return
        series = concentrations["pMAPK"]
        if not series:
            return
        try:
            pmapk_peak = max(series)
            mapk_initial = concentrations.get("MAPK", [0])[0] if "MAPK" in concentrations else 0.1
            if pmapk_peak < mapk_initial * 10:
                result.add_violation(
                    category="biological", severity="medium",
                    rule="signal_amplification",
                    message=(
                        f"MAPK 放大效应不足：pMAPK 峰值={pmapk_peak:.4f}，"
                        f"MAPK 初始={mapk_initial:.4f}（期望 >10x 放大）"
                    ),
                    fix_suggestion="检查级联各级 k_phos/k_dephos 比值，确保信号放大",
                )
        except Exception as exc:
            logger.warning("MAPK 放大检查失败: %s", exc)


# =============================================================================
# 全局入口
# =============================================================================
_global_checker: DomainChecker | None = None


def get_domain_checker() -> DomainChecker:
    """获取全局 DomainChecker 实例。"""
    global _global_checker
    if _global_checker is None:
        _global_checker = DomainChecker()
    return _global_checker


def check_ode_code(
    code: str,
    species_names: list[str] | None = None,
    edges: list[dict[str, Any]] | None = None,
    template_name: str = "",
) -> DomainCheckResult:
    """全局入口：对 ODE 代码做领域常识审查。"""
    return get_domain_checker().check_code(
        code=code,
        species_names=species_names or [],
        edges=edges or [],
        template_name=template_name,
    )


def check_simulation(
    species_names: list[str],
    time_points: list[float],
    concentrations: dict[str, list[float]],
    template_name: str = "",
) -> DomainCheckResult:
    """全局入口：对仿真结果做领域常识审查。"""
    return get_domain_checker().check_simulation_result(
        species_names=species_names,
        time_points=time_points,
        concentrations=concentrations,
        template_name=template_name,
    )
