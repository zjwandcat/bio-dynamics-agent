# BioDynamics Agent v4 - ODE Templates v2（Phase 3 新增）
# 对应 v4 Scientific Architecture Part 4 + Migration Plan §3。
#
# 职责：从 Reaction IR v2 / Pathway Graph 渲染 ODE 代码，使用 v4 模板。
#
# v4 模板与 v3 模板的关键差异（审计 §3 修复）：
# 1. _mechanism_phosphorylation_mm.j2：恢复 Michaelis-Menten（v3 错误地移除）
# 2. oscillatory_feedback.j2：转录延迟反馈（p53/NF-κB/TGF-β/JAK-STAT），支持 DDE
# 3. bistable_switch.j2：双稳态开关（Apoptosis/Cell Cycle），含正反馈
# 4. _dde_helpers.j2：DDE 求解器辅助（jitcdde try-import 降级）
#
# P3 模板补全（spec.md Part 5 要求 9 个新模板）：
# 5. transcriptional_delay.j2：转录延迟（p53/NF-κB/TGF-β/JAK-STAT）
# 6. nuclear_transport.j2：核质转运（STAT/NF-κB/SMAD/β-catenin/p53）
# 7. ubiquitination_cascade.j2：泛素化级联（p53-Mdm2/IκBα/β-catenin）
# 8. destruction_complex.j2：破坏复合体（Wnt β-catenin 五步耦合）
# 9. caspase_cascade.j2：Caspase 级联（Apoptosis + MOMP）
# 10. cyclin_cdk_toggle.j2：Cyclin-CDK toggle（Cell Cycle + APC/C）
# 11. transcription_factor.j2：转录因子通路（STAT/SMAD/β-catenin/NF-κB）
#
# 设计原则（铁律）：
# 1. Feature Flag V4_ODE_TEMPLATE_V2_ENABLED=false 时完全不使用，仍走 v3 ode_templates/
# 2. v4 模板不修改 v3 ode_templates/ 任何文件（不可碰清单）
# 3. v4 模板渲染产物仍调用 sandbox.py 执行（沙盒不变）
# 4. v4 模板从 ReactionIRv2 + PathwayGraph 派生，不再依赖 LLM 直接生成 ODE

__all__ = [
    # P3 基础 4 模板
    "_mechanism_phosphorylation_mm",
    "oscillatory_feedback",
    "bistable_switch",
    "_dde_helpers",
    # P3 补全 7 模板（spec.md Part 5）
    "transcriptional_delay",
    "nuclear_transport",
    "ubiquitination_cascade",
    "destruction_complex",
    "caspase_cascade",
    "cyclin_cdk_toggle",
    "transcription_factor",
]
