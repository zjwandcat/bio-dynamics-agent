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
# 设计原则（铁律）：
# 1. Feature Flag V4_ODE_TEMPLATE_V2_ENABLED=false 时完全不使用，仍走 v3 ode_templates/
# 2. v4 模板不修改 v3 ode_templates/ 任何文件（不可碰清单）
# 3. v4 模板渲染产物仍调用 sandbox.py 执行（沙盒不变）
# 4. v4 模板从 ReactionIRv2 + PathwayGraph 派生，不再依赖 LLM 直接生成 ODE

__all__ = [
    "_mechanism_phosphorylation_mm",
    "oscillatory_feedback",
    "bistable_switch",
    "_dde_helpers",
]
