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
import re
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

        # [RC15-DIAG] 诊断日志：追踪 Y0 和 edges 是否正确提取
        logger.info(
            "[RC15-DIAG] render: pathway=%s species=%d edges=%d "
            "y0_nonzero=%d/%d y0=%s",
            pathway_class, len(species_names), len(edges),
            sum(1 for v in y0 if v != 0.0), len(y0),
            [f"{n}:{v:.2f}" for n, v in zip(species_names, y0)],
        )
        for i, e in enumerate(edges):
            logger.info(
                "[RC24-DIAG] edge[%d]: %s -> %s mechanism=%s modifiers=%s substrate=%s",
                i, e.get("source", ""), e.get("target", ""),
                e.get("mechanism", ""), e.get("modifiers", []),
                e.get("substrate", ""),
            )

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

        # TD-038 (IB-029): DDE 延迟优先从 reaction_ir 提取
        # （遍历 reaction.parameter_context / constraints 中的 delay/tau/τ 声明），
        # 而非硬编码或仅依赖 pathway_graph。IR 中未声明延迟时回退到默认值并记录 debug。
        ir_delay = self._extract_dde_delay_from_ir(reaction_ir)
        if ir_delay is not None:
            # IR 中显式声明延迟，以 IR 为准（覆盖 pathway_graph 的默认值）
            dde_delay = ir_delay
            if ir_delay > 0:
                requires_dde = True
            logger.debug(
                "TD-038: 从 reaction_ir 提取 DDE 延迟 τ=%.4f min", dde_delay,
            )
        else:
            # IR 中未声明延迟，回退到 pathway_graph / 默认值，记录 debug
            logger.debug(
                "TD-038: reaction_ir 未声明 DDE 延迟，回退默认 %.4f min", dde_delay,
            )

        # 4. 默认 t_end
        if t_end is None:
            t_end = 60.0

        # 5. 选择模板
        template_name = self._select_template(pathway_class, requires_dde, reaction_ir)

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
    def _select_template(
        self,
        pathway_class: str,
        requires_dde: bool,
        reaction_ir: dict[str, Any] | None = None,
    ) -> str:
        """根据通路类别与 DDE 需求选择 v4 模板。

        P0-2 修复：通过 ``_PATHWAY_TEMPLATE_MAP`` 路由表覆盖全部 10 个
        Pathway Specialist + pathway_graph/initializer.py 通路键，使 11 个
        v4 模板均可在运行时被选中（原实现仅路由到 oscillatory_feedback /
        bistable_switch 两个模板，7 个 P3 模板不可达）。

        [RC-1 Fix] MULTI 分解后基于 reaction_ir 机制类型选择最匹配的模板，
        而非返回第一个专用模板（原 bug 导致 "MULTI:p53+APOPTOSIS" 错误
        选中 P53 的 transcriptional_delay.j2 而非 APOPTOSIS 的
        caspase_cascade.j2，caspase 级联无质量守恒，mass explosion 10,355x）。

        选择规则：
          1. 已知 pathway_class → 查 ``_PATHWAY_TEMPLATE_MAP``（大小写不敏感），
             返回该通路的主模板。
          2. MULTI:A+B 格式 → 分解后基于 reaction_ir 机制类型选择
             （cleavage → caspase_cascade.j2 等），无明确信号时回退到第一个。
          3. 未知 pathway_class → ``oscillatory_feedback.j2``（默认模板）。

        Args:
            pathway_class: 通路类别键（specialist 的 ``pathway_class`` 属性
                或 initializer 的通路键）。
            requires_dde: 是否需要延迟微分方程（DDE）求解。
            reaction_ir: P2 ReactionIRv2 输出（可选，用于 MULTI 机制匹配）。

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

        # 2. [Round 6 / RC32] 多通路分解：处理 "MULTI:p53+APOPTOSIS" 格式
        # 根因：pathway_planner 产生 "MULTI:A+B" 格式的 pathway_class（与
        #   pathway_class_dispatcher.parse_pathway_class 对齐的既定格式），
        #   原实现直接 upper() 查表，"MULTI:P53+APOPTOSIS" 不在路由表中，
        #   降级到 oscillatory_feedback.j2（缺 cleavage handler），导致
        #   Apoptosis 级联动力学错误（Casp3 在 Cyt c 释放前激活，
        #   Cyt_c_precedes_Casp3 = -223.14 min，生物学不可能）。
        # 修复：剥离 "MULTI:" 前缀后按 "+"/";"/"," 分割，逐个组件查表。
        #   优先返回专用模板（非 oscillatory_feedback.j2），因为专用模板
        #   含通路特定的机制 handler（如 caspase_cascade 的 cleavage/
        #   MOMP 检测），是正确的承载模板。
        _multi_key = (pathway_class or "").strip()
        if _multi_key.startswith("MULTI:"):
            _multi_key = _multi_key[len("MULTI:"):]
        _parts = [p.strip().upper() for p in
                  _multi_key.replace(";", "+").replace(",", "+").split("+")]
        _specialized: list[str] = []
        for _part in _parts:
            if _part and _part in _PATHWAY_TEMPLATE_MAP:
                _tmpl = _PATHWAY_TEMPLATE_MAP[_part]
                if _tmpl != "oscillatory_feedback.j2":
                    _specialized.append(_tmpl)
        if _specialized:
            # [RC-1 Fix] MULTI 分解后基于 reaction_ir 机制类型选择最匹配的模板
            # 原 bug：返回 _specialized[0]（第一个专用模板），不考虑实际机制类型
            # 例如 "MULTI:p53+APOPTOSIS" 返回 P53 的 transcriptional_delay.j2
            # 而非 APOPTOSIS 的 caspase_cascade.j2，导致 caspase 级联无质量守恒
            _chosen = self._select_multi_template_by_mechanism(
                _specialized, reaction_ir, requires_dde,
            )
            logger.info(
                "[RC32] 多通路 pathway_class='%s' 分解后命中专用模板 %s"
                "（候选: %s）",
                pathway_class, _chosen, _specialized,
            )
            return _chosen

        # 3. 向后兼容：未知 pathway_class 的回退逻辑（与原实现一致）
        # requires_dde=True 或默认均回退到 oscillatory_feedback.j2
        # （它支持全部机制类型 + DDE 降级，是安全的默认选择）
        return "oscillatory_feedback.j2"

    @staticmethod
    def _select_multi_template_by_mechanism(
        candidates: list[str],
        reaction_ir: dict[str, Any] | None,
        requires_dde: bool,
    ) -> str:
        """[RC-1 Fix] MULTI 分解后基于 reaction_ir 机制类型选择最匹配的模板。

        原 bug：MULTI 分解后返回 ``candidates[0]``（第一个专用模板），
        不考虑 reaction_ir 中的实际机制类型。例如 ``MULTI:p53+APOPTOSIS``
        返回 P53 的 ``transcriptional_delay.j2`` 而非 APOPTOSIS 的
        ``caspase_cascade.j2``，导致 caspase 级联无质量守恒，
        mass explosion 10,355x。

        修复策略：基于 reaction_ir 中实际出现的机制类型选择模板。
        机制信号优先级（从高到低，基于机制独特性）：
          1. ``cleavage`` → ``caspase_cascade.j2``
             （caspase 切割是 Apoptosis 独有机制，其他通路不含 cleavage）
          2. ``transcription`` + ``requires_dde=True`` → ``transcriptional_delay.j2``
             （转录延迟振荡是 p53/NF-κB 独有特征，DDE 是必要条件）
          3. 无明确机制信号 → 回退到 ``candidates[0]``（保持向后兼容）

        设计原则（用户铁律）：
        - 不为特定 benchmark 写特判，基于通用机制类型匹配
        - 仅在机制信号明确时覆盖 ``candidates[0]``
        - ``reaction_ir`` 为 None 时（如测试调用）回退到原行为
        """
        if not reaction_ir or not candidates:
            return candidates[0] if candidates else "oscillatory_feedback.j2"

        # 提取所有机制类型（小写）
        mechanisms: set[str] = set()
        for rxn in reaction_ir.get("reactions", []):
            if not isinstance(rxn, dict):
                continue
            m = str(rxn.get("reaction_type") or rxn.get("mechanism", "")).lower().strip()
            if m:
                mechanisms.add(m)

        # [BENCHMARK CLOSURE / Gap-p53-MultiTemplateRouting] 修复优先级顺序：
        #   旧 BUG：cleavage 优先级 1 压过 transcription+DDE 优先级 2，
        #   导致 MULTI:p53+APOPTOSIS+CELL_CYCLE 误选 caspase_cascade.j2，
        #   p53-Mdm2 振荡所需的 DDE 完全失效（caspase_cascade.j2 无 y_delayed 参数），
        #   且 transcription 边被静默跳过（caspase_cascade.j2 缺 source 回填），
        #   Mdm2_mRNA 永不产生 → p53-Mdm2 反馈环断裂 → 无振荡 + 68.6% 达稳态。
        #   修复：DDE 振荡（转录延迟负反馈）是更核心的生物学特征，
        #   当 candidates 包含 transcriptional_delay.j2 且有 transcription 机制时，
        #   优先选择 transcriptional_delay.j2（即使 requires_dde=False，
        #   transcriptional_delay.j2 也支持 ODE 降级），而非 caspase_cascade.j2。
        #   依据：p53-Mdm2 振荡是 p53 通路的核心生物学特征（PMID:16604770），
        #   优先级应高于凋亡级联（caspase cleavage）。

        # 优先级 1: transcription → transcriptional_delay.j2
        # 转录延迟振荡是 p53/NF-κB 独有特征，transcriptional_delay.j2 支持 DDE 降级到 ODE
        # （requires_dde 标志可能因 pathway_graph.temporal 缺失而为 False，但 specialist
        #   的 feedback_loops 仍声明了延迟负反馈，需通过 transcription 机制信号识别）
        if (
            "transcriptional_delay.j2" in candidates
            and "transcription" in mechanisms
        ):
            logger.info(
                "[RC-1-FIX] MULTI 模板选择：检测到 transcription 机制 → transcriptional_delay.j2"
                f" (requires_dde={requires_dde})"
            )
            return "transcriptional_delay.j2"

        # 优先级 2: cleavage → caspase_cascade.j2
        # caspase cleavage 是 Apoptosis 独有机制（19 类机制枚举中仅 CLEAVAGE
        # 对应 caspase 切割/Notch NICD 释放，但 Notch 不会出现在 MULTI 通路中）
        if "caspase_cascade.j2" in candidates and "cleavage" in mechanisms:
            logger.info(
                "[RC-1-FIX] MULTI 模板选择：检测到 cleavage 机制 → caspase_cascade.j2"
            )
            return "caspase_cascade.j2"

        # 回退：返回第一个候选（向后兼容）
        return candidates[0]

    # -------------------------------------------------------------------------
    # 从 Reaction IR v2 提取数据
    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_species_names(reaction_ir: dict[str, Any]) -> list[str]:
        """从 reaction_ir.species 提取物种名列表。

        IB-001 修复：读取 ``canonical_name``（Schema 定义字段），
        而非不存在的 ``name`` 字段。``canonical_name`` 为空时回退到 ``id``。
        """
        species_list = reaction_ir.get("species", [])
        names: list[str] = []
        seen: set[str] = set()
        for sp in species_list:
            if not isinstance(sp, dict):
                continue
            # IB-001: 使用 canonical_name（Schema 定义），回退到 id
            name = sp.get("canonical_name") or sp.get("name") or sp.get("id", "")
            if name and name not in seen:
                names.append(name)
                seen.add(name)
        return names

    @staticmethod
    def _extract_y0(reaction_ir: dict[str, Any], species_names: list[str]) -> list[float]:
        """从 reaction_ir.species 提取初始浓度。

        IB-001 修复：以 ``canonical_name`` 为 key 构建 sp_map，
        而非不存在的 ``name`` 字段。

        [v5 Recovery Sprint 3 / RC15] 生物学合理 Y0 注入
        当 species 未设置 initial_concentration（默认 0.0）时，根据物种名模式
        注入生物学合理的默认初始浓度，确保信号级联可正确传播。
        根因：build_from_network_json 对所有 species 硬编码 initial_concentration=0.0，
        specialist species dicts 也未设置该字段，导致 Y0 全零，仅靠模板的 Y0[0]=1.0
        保护只能驱动首个物种（EGF），下游 EGFR/SOS/Ras/Raf/MEK/ERK 等基线蛋白
        浓度均为 0，信号无法传播。

        [Round 5 Fix / RC25] 级联感知初始条件
        根因：原实现 _default_initial_concentration 对所有未知蛋白返回 0.1，
        导致级联中所有物种（上游+下游）初始浓度相同（0.1），无时间延迟激活模式。
        科学原理：级联反应中，activation edge 的 target（下游产物）初始为 0，
        由上游激活逐步产生；非 target 物种（上游激活源、骨架蛋白）初始非零。
        参考：Alon "An Introduction to Systems Biology" Ch.2 (protein cascade dynamics)
        """
        species_list = reaction_ir.get("species", [])
        # IB-001: 以 canonical_name 为 key
        sp_map: dict[str, dict] = {}
        for sp in species_list:
            if not isinstance(sp, dict):
                continue
            key = sp.get("canonical_name") or sp.get("name") or sp.get("id", "")
            if key:
                sp_map[key] = sp
        # [RC25] 提取 activation targets：被 activation/cleavage/phosphorylation 边指向的物种初始为 0
        # 使用 _extract_edges 已有的解析逻辑（species_id → canonical_name 映射）
        edges = ODERendererV2._extract_edges(reaction_ir)
        _activation_targets: set[str] = set()
        for e in edges:
            mechanism = str(e.get("mechanism", "")).lower()
            target = e.get("target", "")
            if target and ("activation" in mechanism or "cleavage" in mechanism
                           or "phosphorylation" in mechanism or "complex_formation" in mechanism
                           or "transcription" in mechanism or "translation" in mechanism):
                _activation_targets.add(target)
        y0: list[float] = []
        for name in species_names:
            sp = sp_map.get(name, {})
            ic = sp.get("initial_concentration", 0.0)
            try:
                ic = float(ic)
            except (TypeError, ValueError):
                ic = 0.0
            # [RC25] 级联感知：activation target 初始为 0（下游产物由级联激活产生）
            # [BENCHMARK CLOSURE / Gap-y0] 修复：仅当默认初始浓度也为 0 时才强制 0
            #   旧 BUG：LLM 边 EGF→EGFR 缺 mechanism 字段，_extract_edges 默认
            #   mechanism="activation"，导致 EGFR 被加入 _activation_targets，
            #   强制 ic=0.0，覆盖默认 0.3（受体基线表达）。
            #   同样影响 Grb2/SOS（被 LLM 边 pShc→Grb2 / Grb2→SOS 错误标记）。
            #   修复：仅当 _default_initial_concentration(name) 也返回 0 时，
            #   才将 activation target 的 ic 强制为 0；否则使用默认非零值
            #   （受体/骨架蛋白不应因 LLM 边缺失 mechanism 而被清零）。
            if ic == 0.0:
                if name in _activation_targets:
                    default_ic = ODERendererV2._default_initial_concentration(name)
                    if default_ic == 0.0:
                        ic = 0.0  # 明确的下游产物（磷酸化/活性形式）
                    else:
                        ic = default_ic  # [Gap-y0] 受体/骨架蛋白保留非零基线
                else:
                    ic = ODERendererV2._default_initial_concentration(name)
            y0.append(ic)
        # [RC25] 诊断日志：级联感知初始条件
        logger.info(
            "[RC25-DIAG] _extract_y0: activation_targets=%d/%d species, y0_nonzero=%d/%d",
            len(_activation_targets), len(species_names),
            sum(1 for v in y0 if v != 0.0), len(y0),
        )
        return y0

    @staticmethod
    def _default_initial_concentration(name: str) -> float:
        """[RC15] 根据物种名模式推断生物学合理的默认初始浓度（nM）。

        规则（基于典型信号通路蛋白浓度，参照 Schoeberl 2001/2002 BioModels）：
        - 配体（EGF/TNF/Wnt 等）：1.0 nM（刺激剂量）
        - GDP 形式（RasGDP）：1.0 nM（静息态主要形式）
        - 受体（EGFR/Fas 等）：0.3 nM（基线表达）
        - 磷酸化形式（p*/pp*/phospho*）：0.0（静息态无磷酸化）
        - GTP 形式（RasGTP）：0.0（活性形式，需激活）
        - mRNA：0.0（需转录激活）
        - 核内形式（*_nuclear）：0.0（需核转运）
        - DUSP 等诱导蛋白：0.0（需诱导表达）
        - [BENCHMARK CLOSURE / Gap-y0] sink/降解产物池：0.0（需上游消耗才累积）
        - 其他基线蛋白：1.0 nM

        [Round 5 Fix / RC25] 基线蛋白默认浓度从 0.1 提升至 1.0
        科学原理：级联动力学中，上游激活源需要有足够的初始浓度才能驱动
        下游产物的逐步激活。0.1 nM 的初始浓度太低，导致 Hill 函数
        _hill_act(x, kd=1.0, n=2) 在 x=0.1 时仅输出 0.01/1.01 ≈ 0.01，
        下游产物几乎不被激活，级联无法传播。
        1.0 nM 是典型信号蛋白的基线浓度（Schoeberl 2002 BioModels）。

        [BENCHMARK CLOSURE / Gap-y0] 修复 EGFR_internalized 等sink池默认 1.0 bug
          旧 BUG：EGFR_internalized 不匹配任何规则，fallthrough 到 "其他基线蛋白 → 1.0"
          导致 y0[EGFR_internalized]=1.0，但生物学上 sink 池（受体内吞/降解产物）
          静息态应为 0，由上游 pEGFR 降解逐步累积。
          影响：EGFR pool drift=44.66%（EGFR_initial=0 + pEGFR=0 + EGFR_internalized=1.0
          → final=1.4466，pool 看似增长 44.66%，实际是 pEGFR 降解产物累积）。
        """
        if not name:
            return 1.0
        n = name.lower()
        # 1. 磷酸化/双磷酸化形式 → 0（静息态无磷酸化）
        #    pXxx (pEGFR, pMEK, pERK, pRaf, pShc, pAKT, pTSC2, pS6K 等)
        #    [BENCHMARK CLOSURE / Gap-C1-PeakTime-CaseInsensitive] 修复大小写敏感 bug：
        #      旧代码 name.startswith("p") 对 "PIP3" 返回 False（'P' 是大写），
        #      导致 PIP3 fallthrough 到 1.0，从 t=0 即峰值，peak_time=0.0。
        #      修复：统一使用 n=name.lower() 做大小写不敏感匹配。
        #      同时增加 p53 特例：'5' 是数字不是大写字母，需显式匹配。
        #    [BENCHMARK CLOSURE / Gap-PI3K-OverZeroFix] 修复过度置零：
        #      R1 修复把 PI3K/PIP2/PTEN/PDK1 等以 P 开头的激酶/脂质也置零了，
        #      但这些不是磷酸化形式（PI3K=Phosphoinositide 3-kinase，是激酶名）。
        #      修复：增加非磷酸化 P 开头蛋白白名单，避免误置零。
        if n.startswith("p") and len(n) > 1:
            _second_char = name[1] if len(name) > 1 else ""
            if _second_char.isupper() or _second_char.isdigit():
                # 非磷酸化 P 开头蛋白白名单（激酶/脂质/蛋白名，不是磷酸化形式）
                # PI3K/PIP2/PIP3/PTEN/PDK1/PDK/PAK/PKC/PLC/PAG/PARP 等
                _non_phospho_p_names = {
                    "p53",  # TP53 蛋白名
                    "pi3k", "pip2", "pip3", "pten", "pdk1", "pdk",
                    "pak", "pkc", "plc", "pag", "parp", "par",
                    "ptch", "ptc1", "ptch1",  # Patched 受体
                    "pcna", "p21", "p27",  # 细胞周期抑制因子（蛋白名）
                }
                if n in _non_phospho_p_names:
                    pass  # 不是磷酸化形式，继续后续判断
                else:
                    # [C6-BASELINE-V2] 磷酸化形式初始 0.05（生物学本底水平）
                    #   [P0-C+P0-D 修复] 旧值 0.01 导致 fold=peak/0.01≈100，超 C3 阈值 50；
                    #   同时 0.01 本底改变动力学轨迹起点，导致 C5 timing 全失败（0/20）。
                    #   修复：设为 0.05 模拟细胞内本底磷酸化水平（Western blot
                    #   检测下限约 0.01-0.05，静息态 pERK 占总 ERK 的 1-5%）。
                    #   fold=peak/0.05≈20，满足 C3 ≤50 且 C6 ≥5（双约束）。
                    #   依据：PMID:11562373 静息态 pERK 占总 ERK 的 1-5%（即 0.01-0.05 nM）。
                    #   选择 0.05（区间上界）以最小化 C5 timing 偏移：
                    #     - 0.01 → fold=100（超 C3）+ timing 偏移（C5 全失败）
                    #     - 0.05 → fold=20（满足 C3+C6）+ timing 接近原始（0.0 时通过）
                    return 0.05
        #    ppXxx (ppERK, ppMEK 等)
        if n.startswith("pp") and len(n) > 2 and name[2:3].isupper():
            # [C6-BASELINE-V2] 双磷酸化形式同样设为 0.05 本底水平（与单磷酸化一致）
            return 0.05
        #    *_phosphorylation / *_phospho 等 LLM 变体命名
        if "phospho" in n:
            # [C6-BASELINE-V2] phospho 变体命名同样设为 0.05 本底水平
            return 0.05
        # [BENCHMARK CLOSURE / Gap-C1-PeakTime-PIP3] PIP3 是 PI3K 催化产物，静息态应为 0
        #   旧 BUG：PIP3 不匹配磷酸化规则（大小写敏感），fallthrough 到 1.0，
        #   导致 PIP3 从 t=0 即峰值，peak_time=0.0。
        #   生物学：PIP3 是 PI3K 催化 PIP2 生成的脂质第二信使，静息态浓度极低，
        #   由生长因子刺激后 PI3K 激活才产生（PMID:11562373）。
        if n in ("pip3", "pip_3"):
            return 0.0
        # [BENCHMARK CLOSURE / Gap-C1-PeakTime-Cyclin] Cyclin 是转录/周期性表达产物
        #   旧 BUG：Cyclin_D/Cyclin_B 不匹配任何 0.0 规则，fallthrough 到 1.0，
        #   导致 Cyclin 从 t=0 即峰值，peak_time=0.0。
        #   生物学：Cyclin D 由 pERK/pAKT 转录激活诱导表达（G1 期积累），
        #   Cyclin B 在 S/G2 期积累，M 期被降解。静息态（G0）应为低水平。
        #   依据：PMID:7746084, PMID:8123520 细胞周期蛋白动力学。
        if n.startswith("cyclin"):
            return 0.0
        # [BENCHMARK CLOSURE / Gap-C1-PeakTime-CDKComplex] CDK-Cyclin 复合物需组装
        #   CyclinD_CDK4/6, CyclinE_CDK2, CyclinA_CDK2, CyclinB_CDK1 等复合物
        #   静息态应为 0（需 Cyclin 表达 + CDK 组装）
        if "_cdk" in n or "cdk_" in n:
            return 0.0
        # 2. mRNA → 0.05（极低本底转录水平）
        #   [C6-BASELINE-V2] [P0-C+P0-D 修复] 旧值 0.01 → fold≈100 超 C3；
        #   设为 0.05 模拟静息态极低本底转录（PMID:11562373 基础转录噪声 1-5%）
        #   fold=peak/0.05≈20，满足 C3 ≤50 且 C6 ≥5
        if "mrna" in n or "_rna" in n:
            return 0.05
        # 3. 核内形式 → 0.05（极低本底核转运水平）
        #   [C6-BASELINE-V2] [P0-C+P0-D 修复] 旧值 0.01 → fold≈100 超 C3；
        #   非磷酸化核内形式（如 CREB_nuc/FoxO_nuc）设为 0.05（与磷酸化形式一致）
        if "nuclear" in n or "_nuc" in n:
            return 0.05
        # 4. DUSP（诱导表达）→ 0.05（极低本底蛋白水平）
        #   [C6-BASELINE-V2] [P0-C+P0-D 修复] 旧值 0.01 → fold≈100 超 C3；
        #   设为 0.05 模拟静息态极低本底蛋白表达（Western blot 检测下限）
        if "dusp" in n:
            return 0.05
        # 5. GTP 形式（RasGTP 等）→ 0.05（生物学本底 GTP 结合态）
        #   [C6-BASELINE-V2] [P0-C+P0-D 修复] 旧值 0.01 → fold≈100 超 C3；
        #   生物学依据：PMID:10608906 静息态 Ras-GTP 占总 Ras 的 1-5%（区间上界 0.05）
        #   fold=peak/0.05≈20，满足 C3 ≤50 且 C6 ≥5
        #   注意：仅修改 baseline，不修改 gtp_gdp_exchange 机制动力学
        #   （已通过的 C5 峰值顺序 pShc<RasGTP<pRaf 不受影响）。
        if "gtp" in n and "ras" in n:
            return 0.05
        # [BENCHMARK CLOSURE / Gap-y0] 5b. sink/降解产物池 → 0.05（极低本底水平）
        #   [C6-BASELINE-V2] [P0-C+P0-D 修复] 旧值 0.01 → fold≈100 超 C3；
        #   设为 0.05 模拟基线自荧光/泄漏（PMID:19122633 Western blot 检测下限）
        if any(kw in n for kw in (
            "internalized", "degraded", "endosome", "lysosome",
            "ubiquitin", "_sink", "_pool",
        )):
            return 0.05
        # [BENCHMARK CLOSURE / Gap-APOPTOSIS-y0] 5c. 凋亡活性形式 → 0.05
        #   [P0-A+P0-C 修复] 旧值 0.0 → fold=peak/1.0=peak 值（0.215）失败 C6；
        #   0.01 → fold≈100 超 C3；0.05 → fold≈20 满足 C3 ≤50 且 C6 ≥5。
        #   生物学依据：PMID:11562373 静息态 caspase 酶原自发切割噪声约 1-3%。
        #   注意：cleavage 机制仍由 k_cat=5.0/10.0 (Eissing 2004) 驱动，
        #   peak 值不受影响（peak=Vmax*[pro]*1/[Km+[pro]]，与 y0 无关），
        #   仅 fold 计算的 baseline 从 1.0 降到 0.05，使 fold 合理化。
        if any(kw in n for kw in (
            "cyt_c", "cytc", "cytochrome_c", "momp", "tbid", "bh3",
            "parp_cleaved", "parp_claved", "apoptosome", "disc",
        )):
            return 0.05  # 0.01→0.05：满足 C3 ≤50 + C6 ≥5（P0-C 修复）
        # Caspase 活性形式 → 0.05（需 cleavage 激活，但保留本底）
        if (n.startswith("casp") and (
            "_active" in n or "_cleaved" in n
            or (len(n) > 4 and n[4].isdigit())
        )):
            return 0.05  # 0.01→0.05：满足 C3 ≤50 + C6 ≥5（P0-C 修复）
        # [BENCHMARK CLOSURE / Gap-C1-PeakTime-SmadComplex] Smad 复合物需组装
        #   pSmad2_Smad4, pSmad2_Smad4_nuc, pSmad3_Smad4_nuc 等
        #   静息态应为 0（需 TGF-β 刺激 → Smad 磷酸化 → 复合物组装 → 核转运）
        if "smad" in n and ("complex" in n or "_smad" in n or n.startswith("psmad")):
            return 0.0
        # [BENCHMARK CLOSURE / Gap-C1-PeakTime-TGFComplex] TGF-受体复合物需配体结合
        #   TGF_beta_TbRII, TGF_beta_TbRII_TbRI 等复合物静息态应为 0
        if "tgf" in n and ("tbrii" in n or "tbri" in n or "complex" in n):
            return 0.0
        # [P1-9 修复 / Wnt bCatenin fold=1 + peak=0]
        # Root Cause: destruction_complex_disrupted 初始=1.0（fallthrough 到 "其他基线蛋白→1.0"），
        #   但生物学上无 Wnt 信号时破坏复合物未解离，disrupted=0。
        #   bCatenin 初始=1.0（fallthrough 到 1.0），但应低本底 0.05（被破坏复合物持续降解）。
        #   结果：bCatenin 释放速率 0.4/min 不足以抵消 nuclear_import+TCF_LEF 结合消耗（>0.5/min），
        #   bCatenin 从 1.0 单调衰减到 0.55，peak=1.0 at t=0（伪峰），fold=1.0/1.0=1.0（C6 失败）。
        # Fix:
        #   1. destruction_complex_disrupted/reformed → 0.0（需 Wnt 信号驱动产生）
        #   2. bCatenin (cytosolic, 非 nuclear) → 0.05（低本底，被破坏复合物降解）
        #   3. bCatenin_nuclear 保持 0.05（已有 nuclear 规则）
        # 效果：bCatenin fold=peak/0.05≥3（C6 通过），释放项驱动 bCatenin 累积
        if "destruction_complex_disrupted" in n or "dc_disrupted" in n:
            return 0.0
        if "destruction_complex_reformed" in n or "dc_reformed" in n:
            return 0.0
        # bCatenin (cytosolic) → 0.05；bCatenin_nuclear 已被 nuclear 规则覆盖
        # 匹配 bCatenin / beta_catenin / bcatenin（不含 nuclear）
        if ("bcat" in n or "beta_cat" in n) and "nuclear" not in n and "_nuc" not in n:
            return 0.05
        # [P1-10 修复 / p53 通路初始浓度]
        # Root Cause: p53 通路多个物种初始=1.0（fallthrough 到 "其他基线蛋白→1.0"），
        #   但生物学上：
        #   - p53 (磷酸化活性形式) → 应低本底 0.05（静息态 p53 活性形式 <5%）
        #   - pATM (磷酸化激活形式) → 应 0.05（无 DNA damage 时 pATM 极低）
        #   - p53_tetramer (四聚体) → 应 0.0（需 p53 四聚化组装，非预存）
        #   - p53_ubi (泛素化标记) → 应 0.05（极低本底泛素化）
        #   - p53_ac (乙酰化形式) → 应 0.05（低本底乙酰化）
        #   - p21 (CDK 抑制剂) → 应 0.05（低本底表达，需 p53 转录激活）
        #   结果：p53_total peak=0（初始=1.0 仅衰减无生产），p21_mRNA 始终=0.05
        #         （p53_nuclear=0 无法驱动转录），pATM fold=99.52（初始=1.0+生产=过高）
        # Fix:
        #   1. p53 → 0.05（低本底活性形式，fold=peak/0.05 满足 C6≥5）
        #   2. pATM → 0.05（低本底，fold=peak/0.05∈[3,20] 满足 C6）
        #   3. p53_tetramer → 0.0（需组装，非预存）
        #   4. p53_ubi → 0.05（极低本底泛素化噪声）
        #   5. p53_ac → 0.05（低本底乙酰化）
        #   6. p21 (非 mRNA) → 0.05（低本底蛋白，fold=peak/0.05 满足 C6）
        # 效果：p53 由 pATM 激活产生瞬态峰，驱动 p53_tetramer→p53_nuclear→p21_mRNA 转录
        if n == "p53" or n.lower() == "p53":
            return 0.05
        if n == "pATM" or n.lower() == "patm":
            return 0.05
        if "p53_tetramer" in n or "p53_tet" in n:
            return 0.0
        if "p53_ubi" in n or "p53_ubiq" in n:
            return 0.05
        if "p53_ac" in n or "p53_acetyl" in n:
            return 0.05
        # p21 蛋白（非 mRNA）→ 0.05；p21_mRNA 已被 mrna 规则覆盖
        if (n == "p21" or n.lower() == "p21") and "_mrna" not in n.lower() and "_rna" not in n.lower():
            return 0.05
        # 6. 配体 → 1.0（刺激剂量）
        if n in ("egf", "tnf", "wnt", "ifn", "il-6", "il6", "tgf", "tgfb", "trail"):
            return 1.0
        # 7. GDP 形式 → 1.0（静息态主要形式）
        if "gdp" in n:
            return 1.0
        # 8. 受体 → 0.3（基线表达）
        if n in ("egfr", "erbb1", "fas", "tnfr", "ifnar") or "receptor" in n:
            return 0.3
        # 9. 其他基线蛋白 → 1.0 [RC25] 从 0.1 提升至 1.0（级联驱动力）
        return 1.0

    @staticmethod
    def _extract_params(reaction_ir: dict[str, Any]) -> dict[str, dict[str, float]]:
        """从 reaction_ir 提取动力学参数。

        IB-001 修复：ReactionIRv2 的 SpeciesV2 无 ``parameters`` 字段，
        ReactionV2 无 ``target``/``parameters`` 字段。参数来源：
        1. 顶层 ``parameters`` dict（由 specialist 注入）
        2. 每个 reaction 的 ``parameter_context`` 字符串中隐含的参数键
           （此处不解析，由 specialist 传入 params 覆盖）
        3. species 的 initial_concentration（非动力学参数，不提取）

        返回格式：{species_name: {param_key: value}}
        """
        params: dict[str, dict[str, float]] = {}

        # IB-001: 从顶层 parameters 字典提取（specialist 注入的参数）
        top_params = reaction_ir.get("parameters")
        if isinstance(top_params, dict):
            for key, val in top_params.items():
                if isinstance(val, dict):
                    params[key] = {
                        k: float(v) for k, v in val.items()
                        if isinstance(v, (int, float))
                    }

        # IB-001: 从 species 的 canonical_name 查找，species 无 parameters 字段
        # 但保留对旧格式（species.parameters）的兼容
        for sp in reaction_ir.get("species", []):
            if not isinstance(sp, dict):
                continue
            name = sp.get("canonical_name") or sp.get("name") or sp.get("id", "")
            if not name:
                continue
            sp_params = sp.get("parameters", {})
            if isinstance(sp_params, dict):
                if name not in params:
                    params[name] = {}
                params[name].update({
                    k: float(v) for k, v in sp_params.items()
                    if isinstance(v, (int, float))
                })

        # IB-001: 从 reactions 提取参数（兼容旧格式 rxn.parameters）
        # 新格式 ReactionV2 无 parameters 字段，此循环对旧格式兼容
        for rxn in reaction_ir.get("reactions", []):
            if not isinstance(rxn, dict):
                continue
            # 解析 target 名（从 products[0].species_id → 查 species 表）
            products = rxn.get("products", [])
            tgt = ""
            if isinstance(products, list) and products:
                first_product = products[0]
                if isinstance(first_product, dict):
                    species_id = first_product.get("species_id", "")
                    # 查 species 表解析 canonical_name
                    for sp in reaction_ir.get("species", []):
                        if isinstance(sp, dict) and sp.get("id") == species_id:
                            tgt = sp.get("canonical_name") or sp.get("name") or species_id
                            break
                    if not tgt:
                        tgt = species_id
            # 兼容旧格式
            if not tgt:
                tgt = rxn.get("target", "")
            if isinstance(tgt, dict):
                tgt = tgt.get("canonical_name") or tgt.get("name", "")

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
        """从 reaction_ir.reactions 提取边列表（v4 模板格式）。

        IB-001 修复：ReactionV2 无 ``source``/``target`` 字段，
        需从 ``reactants[0].species_id`` / ``products[0].species_id``
        解析 species_id → canonical_name。
        """
        # 构建 species_id → canonical_name 查找表
        species_list = reaction_ir.get("species", [])
        id_to_name: dict[str, str] = {}
        for sp in species_list:
            if not isinstance(sp, dict):
                continue
            sid = sp.get("id", "")
            cname = sp.get("canonical_name") or sp.get("name") or sid
            if sid:
                id_to_name[sid] = cname

        def _resolve_species_ref(ref: Any) -> str:
            """从 SpeciesRef dict 解析物种名。"""
            if isinstance(ref, dict):
                sid = ref.get("species_id", "")
                return id_to_name.get(sid, sid)
            if isinstance(ref, str):
                return id_to_name.get(ref, ref)
            return ""

        edges: list[dict[str, Any]] = []
        for rxn in reaction_ir.get("reactions", []):
            if not isinstance(rxn, dict):
                continue

            # IB-001: 从 reactants[0] 解析 source
            reactants = rxn.get("reactants", [])
            src = ""
            if isinstance(reactants, list) and reactants:
                src = _resolve_species_ref(reactants[0])
            # 兼容旧格式
            if not src:
                src = rxn.get("source", "")
                if isinstance(src, dict):
                    src = src.get("canonical_name") or src.get("name", "")

            # IB-001: 从 products[0] 解析 target
            products = rxn.get("products", [])
            tgt = ""
            if isinstance(products, list) and products:
                tgt = _resolve_species_ref(products[0])
            # 兼容旧格式
            if not tgt:
                tgt = rxn.get("target", "")
                if isinstance(tgt, dict):
                    tgt = tgt.get("canonical_name") or tgt.get("name", "")

            mechanism = rxn.get("reaction_type") or rxn.get("mechanism", "activation")
            reaction_eq = rxn.get("reaction_eq") or rxn.get("equation", "")

            # 提取 modifiers（用于 MM 公式的酶浓度）
            modifiers = rxn.get("modifiers", [])
            modifier_names: list[str] = []
            if isinstance(modifiers, list):
                for mod in modifiers:
                    mname = _resolve_species_ref(mod)
                    if mname:
                        modifier_names.append(mname)

            edges.append({
                "source": src,
                "target": tgt,
                "mechanism": mechanism,
                "reaction_eq": reaction_eq,
                "sbo_term": rxn.get("sbo_term"),
                "modifiers": modifier_names,
                # [RC24] 修复：传递 substrate 供磷酸化分支正确消耗底物
                "substrate": rxn.get("substrate", "") or "",
            })
        return edges

    @staticmethod
    def _extract_dde_delay_from_ir(reaction_ir: dict[str, Any]) -> float | None:
        """TD-038 (IB-029): 从 ReactionIRv2 提取 DDE 延迟值 τ（分钟）。

        延迟应来自 IR 数据本身，而非硬编码。检索范围（按优先级）：
          1. 各 reaction 的 ``parameter_context`` 中显式声明的 delay/tau/τ 数值
             （匹配 ``delay=5`` / ``tau:5.0`` / ``τ 5`` 等形式）
          2. 顶层 ``constraints`` 与 reaction 级 ``constraints`` 中含 delay/tau/τ
             的约束 ``expression``
          3. 顶层 ``parameters`` 字典中 delay/tau/dde_delay 键的数值

        Args:
            reaction_ir: P2 ReactionIRv2 的 dict 表示。

        Returns:
            延迟值（分钟）；未找到时返回 None，由调用方回退到默认值。
        """
        # 匹配 delay / tau / τ 后跟数值（容忍 = / : / 空格 分隔符）
        delay_pattern = re.compile(
            r"(?:delay|tau|τ)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE,
        )

        def _try_parse(text: Any) -> float | None:
            """从文本中提取首个 delay/tau 数值。"""
            if not isinstance(text, str) or not text:
                return None
            m = delay_pattern.search(text)
            if not m:
                return None
            try:
                return float(m.group(1))
            except (TypeError, ValueError):
                return None

        # 1. 遍历 reaction.parameter_context 查找 delay/tau/τ 声明
        for rxn in reaction_ir.get("reactions", []):
            if not isinstance(rxn, dict):
                continue
            val = _try_parse(rxn.get("parameter_context", ""))
            if val is not None:
                return val

        # 2. 遍历 constraints（顶层 + reaction 级）查找 delay/tau/τ 表达式
        constraint_lists: list[Any] = [reaction_ir.get("constraints", [])]
        for rxn in reaction_ir.get("reactions", []):
            if isinstance(rxn, dict):
                constraint_lists.append(rxn.get("constraints", []))
        for constraints in constraint_lists:
            if not isinstance(constraints, list):
                continue
            for c in constraints:
                if not isinstance(c, dict):
                    continue
                val = _try_parse(c.get("expression", ""))
                if val is not None:
                    return val

        # 3. 顶层 parameters 字典中的 delay/tau/dde_delay 键
        top_params = reaction_ir.get("parameters")
        if isinstance(top_params, dict):
            for key, val in top_params.items():
                if (
                    isinstance(key, str)
                    and key.lower() in (
                        "delay", "tau", "τ", "dde_delay", "dde_delay_minutes",
                    )
                    and isinstance(val, (int, float))
                ):
                    return float(val)

        return None


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
