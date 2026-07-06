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

# 振荡通路类别（需要 oscillatory_feedback.j2）
_OSCILLATORY_PATHWAYS = {"p53_signaling", "NF_kB", "TGF_beta", "JAK_STAT"}

# 双稳态通路类别（需要 bistable_switch.j2）
_BISTABLE_PATHWAYS = {"Apoptosis", "Cell_Cycle"}


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

        选择规则：
          1. 振荡通路（p53/NF-κB/TGF-β/JAK-STAT）→ oscillatory_feedback.j2
          2. 双稳态通路（Apoptosis/Cell_Cycle）→ bistable_switch.j2
          3. 其他通路 → 默认 oscillatory_feedback.j2（含完整机制支持）
        """
        if pathway_class in _OSCILLATORY_PATHWAYS or requires_dde:
            return "oscillatory_feedback.j2"
        elif pathway_class in _BISTABLE_PATHWAYS:
            return "bistable_switch.j2"
        else:
            # 默认使用 oscillatory_feedback.j2（它支持所有机制类型）
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
    if not getattr(settings, "V4_ODE_TEMPLATE_V2_ENABLED", False):
        logger.debug("V4_ODE_TEMPLATE_V2_ENABLED=false，跳过 v4 渲染")
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
    if not getattr(settings, "V4_ODE_TEMPLATE_V2_ENABLED", False):
        return False
    return _V4_TEMPLATES_DIR.exists() and any(_V4_TEMPLATES_DIR.glob("*.j2"))


__all__ = [
    "ODERendererV2",
    "render_ode_v2",
    "is_v4_ode_available",
]
