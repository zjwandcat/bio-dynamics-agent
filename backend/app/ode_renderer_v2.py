# BioDynamics Agent v4 - ODE Renderer v2
# 对应 PART C5：ODE Template 不再是入口，Pathway Graph → generates ODE。
#
# 职责：从 ReactionIRv2 + PathwayGraph 选择并渲染 v4 ODE 模板，生成可执行 ODE 代码。
#
# 数据流（v4 vs v3 对比）：
#   v3: user input → LLM → network_json → ode_templates/ → ODE code
#   v4: user input → Ontology(P1) → PathwayGraph(P3) → ReactionIRv2(P2) → ode_templates_v2/ → ODE code
#
# 设计原则（铁律）：
# 1. Feature Flag V4_ODE_TEMPLATE_V2_ENABLED=false 时完全不走 v4 路径，仍走 v3 ode_templates/
# 2. 不修改 v3 ode_templates/ 任何文件（不可碰清单）
# 3. 渲染产物仍调用 sandbox.py 执行（沙盒不变）
# 4. 模板选择基于通路类别 + PathwayGraph.feedback_loops + PathwayGraph.temporal
# 5. 不调用 LLM，纯规则渲染（从 ReactionIRv2 派生）

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# v4 ODE 模板目录
# =============================================================================
_V4_TEMPLATES_DIR = Path(__file__).parent / "ode_templates_v2"

# =============================================================================
# pathway_class → v4 ODE 模板路由表（P0-2 修复：11 个模板均可达）
# =============================================================================
# 覆盖 10 个 Pathway Specialist 的 pathway_class 值 +
# pathway_graph/initializer.py 的通路键（两套命名约定的并集）。
# 大小写不敏感（查找时统一 upper()），兼容 specialist 与 initializer 两种命名。
#
# 路由依据（与各 specialist 的 ODE 动力学需求对齐）：
#   - EGFR / MAPK / PI3K-AKT-mTOR：磷酸化级联 → oscillatory_feedback
#     （注意：_mechanism_phosphorylation_mm.j2 是 {% include %} 片段而非独立
#     模板，无法被 ODERendererV2.render() 顶层渲染；oscillatory_feedback.j2
#     的 _ode_rhs 内含 phosphorylation 分支（Michaelis-Menten），可正确处理
#     磷酸化级联动力学，故作为这三条通路的承载模板。）
#   - p53 / NF-κB：转录延迟振荡（DDE 脉冲振荡）→ transcriptional_delay
#   - JAK-STAT / TGF-β：转录因子 + 核转运 → transcription_factor
#   - Wnt：β-catenin 破坏复合体 → destruction_complex
#   - Apoptosis：Caspase 级联（含 MOMP 双稳态）→ caspase_cascade
#   - Cell Cycle：Cyclin-CDK toggle（双稳态开关）→ cyclin_cdk_toggle
#
# 次要特征说明（主模板已覆盖次要特征检测）：
#   - p53 同时具备振荡 + 转录延迟：transcriptional_delay.detect_delay_effect
#     检测延迟脉冲，覆盖振荡表征。
#   - Apoptosis 同时具备 bistability + caspase cascade：
#     caspase_cascade.detect_momp_and_bistability 覆盖双稳态检测。
#   - Cell Cycle 同时具备 bistability + toggle：
#     cyclin_cdk_toggle.detect_toggle_and_oscillation 覆盖双稳态检测。
_PATHWAY_TEMPLATE_MAP: dict[str, str] = {
    # 磷酸化级联（承载模板：oscillatory_feedback.j2，含 phosphorylation 分支）
    "EGFR_RTK": "oscillatory_feedback.j2",
    "MAPK_ERK": "oscillatory_feedback.j2",
    "PI3K_AKT_MTOR": "oscillatory_feedback.j2",
    # 转录延迟振荡器（DDE 脉冲振荡）
    "P53": "transcriptional_delay.j2",
    "P53_SIGNALING": "transcriptional_delay.j2",
    "NF_KB": "transcriptional_delay.j2",
    # 转录因子 + 核转运
    "JAK_STAT": "transcription_factor.j2",
    "TGF_BETA": "transcription_factor.j2",
    # β-catenin 破坏复合体
    "WNT": "destruction_complex.j2",
    # Caspase 级联（Apoptosis，含 MOMP 双稳态）
    "APOPTOSIS": "caspase_cascade.j2",
    # Cyclin-CDK toggle（Cell Cycle，双稳态开关）
    "CELL_CYCLE": "cyclin_cdk_toggle.j2",
}


class ODERendererV2:
    """v4 ODE 渲染器。

    从 ReactionIRv2 + PathwayGraph 选择 v4 模板并渲染为可执行 ODE 代码。

    使用方式::

        renderer = ODERendererV2()
        ode_code = renderer.render(
            pathway_class="p53_signaling",
            reaction_ir={...},
            pathway_graph={...},
            species_names=["p53", "Mdm2_mRNA", "Mdm2"],
            y0=[0.1, 0.0, 0.0],
            params={...},
            t_end=360.0,
        )
        # ode_code 传入 sandbox.py 执行
    """

    def __init__(self) -> None:
        if not _V4_TEMPLATES_DIR.exists():
            raise FileNotFoundError(
                f"v4 ODE 模板目录不存在: {_V4_TEMPLATES_DIR}。"
                f"请确认 Phase 3 已正确实施。"
            )
        # Jinja2 环境（不使用 StrictUndefined，避免模板变量缺失时崩溃）
        self.env = Environment(
            loader=FileSystemLoader(str(_V4_TEMPLATES_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    # -------------------------------------------------------------------------
    # 主渲染入口
    # -------------------------------------------------------------------------
    def render(
        self,
        pathway_class: str,
        reaction_ir: dict[str, Any],
        pathway_graph: dict[str, Any] | None = None,
        species_names: list[str] | None = None,
        y0: list[float] | None = None,
        params: dict[str, dict[str, float]] | None = None,
        t_end: float | None = None,
        n_eval: int = 200,
    ) -> str:
        """渲染 v4 ODE 代码。

        Args:
            pathway_class: 通路类别（用于模板选择）
            reaction_ir: P2 Reaction IR v2 输出
            pathway_graph: P3 Pathway Graph 输出（可选，用于提取 feedback/temporal）
            species_names: 物种名列表（覆盖 reaction_ir 的 species）
            y0: 初始浓度列表
            params: 动力学参数 {species_name: {param_key: value}}
            t_end: 仿真总时长（分钟，None 时从 pathway_graph.temporal 提取）
            n_eval: 输出时间点数

        Returns:
            str: 可执行 ODE Python 代码
        """
        # 1. 从 reaction_ir 提取 species / edges
        if species_names is None:
            species_names = self._extract_species_names(reaction_ir)
        if y0 is None:
            y0 = self._extract_y0(reaction_ir, species_names)
        if params is None:
            params = self._extract_params(reaction_ir)

        # 2. 从 reaction_ir 提取 edges（适配 v4 模板格式）
        edges = self._extract_edges(reaction_ir)

        # 3. 从 pathway_graph 提取 temporal / DDE 信息
        temporal = None
        requires_dde = False
        dde_delay = 0.0
        if pathway_graph:
            temporal = pathway_graph.get("temporal")
            if temporal:
                requires_dde = temporal.get("requires_dde", False)
                dde_delay = temporal.get("dde_delay_minutes", 0.0)
                if t_end is None:
                    t_end = temporal.get("t_end_minutes", 60.0)

        # 4. 默认 t_end
        if t_end is None:
            t_end = 60.0

        # 5. 选择模板
        template_name = self._select_template(pathway_class, requires_dde)

        # 6. 渲染
        template = self.env.get_template(template_name)
        rendered = template.render(
            species_names=species_names,
            y0=y0,
            edges_json=edges,
            params_json=params,
            t_end=t_end,
            n_eval=n_eval,
            dde_delay_minutes=dde_delay,
            requires_dde=requires_dde,
            pathway_class=pathway_class,
        )

        logger.info(
            "ODE v2 rendered: pathway=%s template=%s species=%d t_end=%.1f dde=%s",
            pathway_class, template_name, len(species_names), t_end, requires_dde,
        )
        return rendered

    # -------------------------------------------------------------------------
    # 模板选择
    # -------------------------------------------------------------------------
    def _select_template(self, pathway_class: str, requires_dde: bool) -> str:
        """根据通路类别与 DDE 需求选择 v4 模板。

        P0-2 修复：通过 ``_PATHWAY_TEMPLATE_MAP`` 路由表覆盖全部 10 个
        Pathway Specialist + pathway_graph/initializer.py 通路键，使 11 个
        v4 模板均可在运行时被选中（原实现仅路由到 oscillatory_feedback /
        bistable_switch 两个模板，7 个 P3 模板不可达）。

        选择规则：
          1. 已知 pathway_class → 查 ``_PATHWAY_TEMPLATE_MAP``（大小写不敏感），
             返回该通路的主模板。
          2. 未知 pathway_class + requires_dde=True → ``oscillatory_feedback.j2``
             （DDE 振荡模板，向后兼容）。
          3. 未知 pathway_class + requires_dde=False → ``oscillatory_feedback.j2``
             （默认模板，支持所有机制类型，向后兼容）。

        Args:
            pathway_class: 通路类别键（specialist 的 ``pathway_class`` 属性
                或 initializer 的通路键）。
            requires_dde: 是否需要延迟微分方程（DDE）求解。

        Returns:
            v4 ODE 模板文件名（含 ``.j2`` 后缀）。
        """
        # 1. 查找 pathway_class → template 路由表（大小写不敏感）
        key = (pathway_class or "").upper()
        if key and key in _PATHWAY_TEMPLATE_MAP:
            template = _PATHWAY_TEMPLATE_MAP[key]
            # 次要特征提示（debug 级，不阻塞流程）
            if key in ("P53", "P53_SIGNALING", "NF_KB"):
                logger.debug(
                    "pathway_class='%s' 路由到 %s（振荡为次要特征，"
                    "由 transcriptional_delay 模板覆盖）",
                    pathway_class, template,
                )
            return template

        # 2. 向后兼容：未知 pathway_class 的回退逻辑（与原实现一致）
        # requires_dde=True 或默认均回退到 oscillatory_feedback.j2
        # （它支持全部机制类型 + DDE 降级，是安全的默认选择）
        return "oscillatory_feedback.j2"

    # -------------------------------------------------------------------------
    # 从 Reaction IR v2 提取数据
    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_species_names(reaction_ir: dict[str, Any]) -> list[str]:
        """从 reaction_ir.species 提取物种名列表。"""
        species_list = reaction_ir.get("species", [])
        names: list[str] = []
        seen: set[str] = set()
        for sp in species_list:
            name = sp.get("name") or sp.get("id", "")
            if name and name not in seen:
                names.append(name)
                seen.add(name)
        return names

    @staticmethod
    def _extract_y0(reaction_ir: dict[str, Any], species_names: list[str]) -> list[float]:
        """从 reaction_ir.species 提取初始浓度。"""
        species_list = reaction_ir.get("species", [])
        sp_map = {sp.get("name"): sp for sp in species_list if isinstance(sp, dict)}
        y0: list[float] = []
        for name in species_names:
            sp = sp_map.get(name, {})
            ic = sp.get("initial_concentration", 0.0)
            try:
                y0.append(float(ic))
            except (TypeError, ValueError):
                y0.append(0.0)
        return y0

    @staticmethod
    def _extract_params(reaction_ir: dict[str, Any]) -> dict[str, dict[str, float]]:
        """从 reaction_ir 提取动力学参数。

        返回格式：{species_name: {param_key: value}}
        """
        params: dict[str, dict[str, float]] = {}
        # 从 species.parameters 提取
        for sp in reaction_ir.get("species", []):
            if not isinstance(sp, dict):
                continue
            name = sp.get("name") or sp.get("id", "")
            if not name:
                continue
            sp_params = sp.get("parameters", {})
            if isinstance(sp_params, dict):
                params[name] = {k: float(v) for k, v in sp_params.items()
                                if isinstance(v, (int, float))}
        # 从 reactions.parameters 提取（按 target 归类）
        for rxn in reaction_ir.get("reactions", []):
            if not isinstance(rxn, dict):
                continue
            tgt = rxn.get("target") or rxn.get("products", [{}])[0] if isinstance(rxn.get("products"), list) else ""
            if isinstance(tgt, dict):
                tgt = tgt.get("name", "")
            if not tgt:
                continue
            rxn_params = rxn.get("parameters", {})
            if isinstance(rxn_params, dict):
                if tgt not in params:
                    params[tgt] = {}
                params[tgt].update({
                    k: float(v) for k, v in rxn_params.items()
                    if isinstance(v, (int, float))
                })
        return params

    @staticmethod
    def _extract_edges(reaction_ir: dict[str, Any]) -> list[dict[str, Any]]:
        """从 reaction_ir.reactions 提取边列表（v4 模板格式）。"""
        edges: list[dict[str, Any]] = []
        for rxn in reaction_ir.get("reactions", []):
            if not isinstance(rxn, dict):
                continue
            src = rxn.get("source") or rxn.get("reactants", [{}])[0] if isinstance(rxn.get("reactants"), list) else ""
            tgt = rxn.get("target") or rxn.get("products", [{}])[0] if isinstance(rxn.get("products"), list) else ""
            if isinstance(src, dict):
                src = src.get("name", "")
            if isinstance(tgt, dict):
                tgt = tgt.get("name", "")
            mechanism = rxn.get("reaction_type") or rxn.get("mechanism", "activation")
            reaction_eq = rxn.get("reaction_eq") or rxn.get("equation", "")
            edges.append({
                "source": src,
                "target": tgt,
                "mechanism": mechanism,
                "reaction_eq": reaction_eq,
                "sbo_term": rxn.get("sbo_term"),
            })
        return edges


# =============================================================================
# 顶层便捷函数
# =============================================================================
def render_ode_v2(
    pathway_class: str,
    reaction_ir: dict[str, Any],
    pathway_graph: dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    """渲染 v4 ODE 代码的便捷函数。

    仅当 V4_ODE_TEMPLATE_V2_ENABLED=true 时使用 v4 渲染器；
    否则返回空字符串（调用方应回退到 v3 ode_templates/）。
    """
    if not settings.effective_v4_ode_template_v2_enabled():
        logger.debug("V4_ODE_TEMPLATE_V2_ENABLED effective=false，跳过 v4 渲染")
        return ""

    renderer = ODERendererV2()
    return renderer.render(
        pathway_class=pathway_class,
        reaction_ir=reaction_ir,
        pathway_graph=pathway_graph,
        **kwargs,
    )


def is_v4_ode_available() -> bool:
    """返回 v4 ODE 渲染器是否可用（feature flag + 模板目录）。"""
    if not settings.effective_v4_ode_template_v2_enabled():
        return False
    return _V4_TEMPLATES_DIR.exists() and any(_V4_TEMPLATES_DIR.glob("*.j2"))


__all__ = [
    "ODERendererV2",
    "render_ode_v2",
    "is_v4_ode_available",
]
