# BioDynamics Agent v4 - Level 3 Cross-Pathway Validation (Phase 5 / Task 5.4)
#
# Level3CrossPathwayValidator 主类 + LangGraph hook 节点。
# 职责：跨通路一致性验证（cross-talk consistency / shared species conservation /
#   time-scale alignment）。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_VALIDATION_PYRAMID_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
# 3. 仅消费 P1/P2/P3 + P4 Coordinator 产出（v4_pathway_graph / v4_shared_species /
#    v4_crosstalk_edges / v4_pathway_class / v4_time_scale_alignment / v4_specialist_outputs）
# 4. 失败降级：任何异常都返回 pass=False，但不抛异常
# 5. 输出写入 state["v4_validation_report"]["level3"]（新增 v4 字段）
# 6. 单通路场景自动 skipped（pass=True，因为无 cross-talk）
#
# 对应 spec.md Part 4 Level 3（第 295-299 行）
#
# 依赖：
# - app.config.settings（Feature Flag）
# - app.pathways.pathway_planner.parse_pathway_class（单通路/多通路判断）

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Level3CrossPathwayValidator 主类
# =============================================================================
class Level3CrossPathwayValidator:
    """Level 3 Cross-Pathway Validation 验证器。

    主入口 validate(state) 执行 3 项检查：
    1. cross-talk consistency：cross-talk edges 在两通路行为一致
    2. shared species conservation：跨通路守恒（阈值 10%）
    3. time-scale alignment：多通路时间尺度对齐

    失败策略（对应 spec.md 第 298 行）：
    - shared species 跨通路守恒误差 > 10% → pass=False

    回滚策略（对应 spec.md 第 299 行）：
    - 单通路场景 Level 3 自动 skipped（pass=True，因为无 cross-talk）

    用法：
        validator = Level3CrossPathwayValidator()
        report = validator.validate(state)
        # report = {pass, crosstalk_consistency, shared_species_conservation,
        #           time_scale_alignment}
    """

    # shared species 跨通路守恒误差阈值：10%（对应 spec.md 第 298 行）
    SHARED_SPECIES_CONSERVATION_THRESHOLD: float = 0.10
    # 数值零保护：避免除零
    _EPSILON: float = 1e-9

    def validate(self, state: dict[str, Any]) -> dict[str, Any]:
        """主入口：执行 Level 3 Cross-Pathway 验证。

        Args:
            state: LangGraph 全局状态，含：
                - v4_pathway_graph: 通路图（含 cross_talk_edges）
                - v4_shared_species: shared species 名列表
                - v4_crosstalk_edges: cross-talk edges 列表
                - v4_pathway_class: 通路类别字符串
                - v4_time_scale_alignment: 时间尺度对齐结果
                - v4_specialist_outputs: Specialist 输出列表（多通路仿真结果）

        Returns:
            Level 3 报告 dict（对应 spec.md 第 297 行）：
            {
                pass: bool,
                crosstalk_consistency: bool,
                shared_species_conservation: float,
                time_scale_alignment: bool
            }
            异常时返回 pass=False，不抛异常（铁律 #5）。
        """
        try:
            if not isinstance(state, dict):
                return self._build_failure_report(
                    crosstalk_consistency=False,
                    shared_species_conservation=1.0,
                    time_scale_alignment=False,
                    reason="invalid_state_type",
                )

            pathway_class = state.get("v4_pathway_class", "") or ""

            # 单通路场景：自动 skipped（pass=True，因为无 cross-talk）
            if self._is_single_pathway(pathway_class):
                return self._run_skipped("single_pathway")

            # 多通路场景：执行 3 项检查
            crosstalk_edges = self._extract_crosstalk_edges(state)
            shared_species = self._extract_shared_species(state)
            time_scale_alignment = state.get("v4_time_scale_alignment") or {}
            specialist_outputs = state.get("v4_specialist_outputs", []) or []

            # SubTask 5.4.2: cross-talk consistency 检查
            crosstalk_ok, _ct_violations = self._check_crosstalk_consistency(
                crosstalk_edges, specialist_outputs
            )

            # SubTask 5.4.3: shared species conservation 检查
            max_conservation_error, _ss_violations = (
                self._check_shared_species_conservation(
                    shared_species, specialist_outputs
                )
            )

            # SubTask 5.4.4: time-scale alignment 检查
            time_aligned = self._check_time_scale_alignment(time_scale_alignment)

            # 失败策略：shared species 守恒误差 > 10% → pass=False
            conservation_pass = (
                max_conservation_error <= self.SHARED_SPECIES_CONSERVATION_THRESHOLD
            )
            pass_flag = bool(
                crosstalk_ok and conservation_pass and time_aligned
            )

            return {
                "pass": pass_flag,
                "crosstalk_consistency": crosstalk_ok,
                "shared_species_conservation": max_conservation_error,
                "time_scale_alignment": time_aligned,
            }
        except Exception as exc:
            # 铁律 #5：失败降级返回 pass=False，但不抛异常
            logger.warning(
                "Level3CrossPathwayValidator.validate 失败，降级 pass=False: %s",
                exc,
            )
            return self._build_failure_report(
                crosstalk_consistency=False,
                shared_species_conservation=1.0,
                time_scale_alignment=False,
                reason=f"validation_exception: {exc}",
            )

    # =========================================================================
    # SubTask 5.4.5: 单通路 skipped
    # =========================================================================
    def _is_single_pathway(self, pathway_class: str) -> bool:
        """检查 pathway_class 是否为单通路。

        判断规则（与 CrossTalkCoordinator 一致）：
        - 不含 "MULTI:" 前缀视为单通路（含 UNKNOWN / 空字符串 / 单通路类别）
        - 含 "MULTI:" 前缀视为多通路

        Args:
            pathway_class: 通路类别字符串，如 "EGFR_RTK" / "MULTI:EGFR_RTK+PI3K_AKT_mTOR"

        Returns:
            True 表示单通路；False 表示多通路
        """
        if not pathway_class or not isinstance(pathway_class, str):
            return True
        return "MULTI:" not in pathway_class

    def _run_skipped(self, reason: str) -> dict[str, Any]:
        """单通路场景 skipped 状态。

        spec.md 第 299 行：单通路场景 Level 3 自动 skipped（pass=True，
        因为无 cross-talk）。

        注意：与 Level 2 的 skipped 状态不同。Level 2 的 skipped = pass=False
        （强制阻塞流水线），Level 3 的 skipped = pass=True（无 cross-talk，
        不阻塞）。

        Args:
            reason: skipped 原因（记录到报告）

        Returns:
            Skipped 报告 dict：
            {
                pass: True,                          # 单通路无 cross-talk，pass=True
                crosstalk_consistency: True,
                shared_species_conservation: 0.0,
                time_scale_alignment: True,
                skipped: True,
                reason: "single_pathway"
            }
        """
        return {
            "pass": True,  # 单通路无 cross-talk，pass=True（spec.md 第 299 行）
            "crosstalk_consistency": True,
            "shared_species_conservation": 0.0,
            "time_scale_alignment": True,
            "skipped": True,
            "reason": reason,
        }

    # =========================================================================
    # SubTask 5.4.2: cross-talk consistency 检查
    # =========================================================================
    def _check_crosstalk_consistency(
        self,
        crosstalk_edges: list[dict[str, Any]],
        specialist_outputs: list[dict[str, Any]],
    ) -> tuple[bool, list[dict[str, Any]]]:
        """检查 cross-talk edges 在两通路行为一致。

        一致性规则：
        - 每条 cross-talk edge 应在源通路与目标通路两侧都被接受为输入
        - 例如：EGFR→PI3K 的 pEGFR→PI3K 激活边在 EGFR 通路激活 PI3K，
          在 PI3K 通路也应被接受为输入
        - 若 source_pathway 的 species 中存在 source_node，且 target_pathway
          的 species 中存在 target_node → 一致
        - 若 target_pathway 的 species 中不存在 target_node → 违规

        Args:
            crosstalk_edges: cross-talk edges 列表，每条含：
                - source_pathway / target_pathway
                - source_node / target_node
                - mechanism / shared_species
            specialist_outputs: Specialist 输出列表，每条含：
                - pathway_class / species / reactions

        Returns:
            (is_consistent, violation_list)
            - is_consistent: True 表示所有 edges 一致
            - violation_list: 每条含 edge_id / reason / source_pathway / target_pathway
        """
        violations: list[dict[str, Any]] = []

        if not crosstalk_edges:
            # 无 cross-talk edges：一致（无矛盾点）
            return (True, [])

        if not specialist_outputs:
            # 有 edges 但无 Specialist 输出：无法验证一致性，视为一致（保守）
            logger.debug(
                "_check_crosstalk_consistency: 有 edges 但无 specialist_outputs，跳过"
            )
            return (True, [])

        # 构建每个通路的 species 集合（用于检查 target_node 是否被接受为输入）
        pathway_species: dict[str, set[str]] = {}
        for output in specialist_outputs:
            if not isinstance(output, dict):
                continue
            pc = output.get("pathway_class", "")
            if not pc:
                continue
            species_set: set[str] = set()
            # 从 species 字段收集
            for sp in output.get("species", []) or []:
                if isinstance(sp, dict):
                    name = sp.get("name") or sp.get("canonical_name") or ""
                    if name:
                        species_set.add(name)
                elif isinstance(sp, str) and sp:
                    species_set.add(sp)
            # 从 crosstalk_reactions 的 source/target 收集
            for rxn in output.get("crosstalk_reactions", []) or []:
                if not isinstance(rxn, dict):
                    continue
                for field in ("source", "target", "source_node", "target_node"):
                    val = rxn.get(field, "")
                    if isinstance(val, str) and val:
                        species_set.add(val)
            pathway_species[pc] = species_set

        for edge in crosstalk_edges:
            if not isinstance(edge, dict):
                continue
            edge_id = edge.get("id", "")
            source_pathway = edge.get("source_pathway", "")
            target_pathway = edge.get("target_pathway", "")
            source_node = edge.get("source_node", "")
            target_node = edge.get("target_node", "")

            # 校验：target_pathway 的 species 中应存在 target_node
            # （即目标通路接受此 cross-talk edge 的输入）
            if target_pathway and target_node:
                target_species = pathway_species.get(target_pathway, set())
                if target_species and target_node not in target_species:
                    violations.append({
                        "edge_id": edge_id,
                        "reason": (
                            f"target_pathway '{target_pathway}' species 中不存在 "
                            f"target_node '{target_node}'，cross-talk edge 未被接受为输入"
                        ),
                        "source_pathway": source_pathway,
                        "target_pathway": target_pathway,
                        "source_node": source_node,
                        "target_node": target_node,
                    })

        is_consistent = len(violations) == 0
        return (is_consistent, violations)

    # =========================================================================
    # SubTask 5.4.3: shared species conservation 检查
    # =========================================================================
    def _check_shared_species_conservation(
        self,
        shared_species: list[str],
        specialist_outputs: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """检查 shared species 跨通路守恒。

        守恒规则（spec.md 第 296-298 行）：
        - shared species 在跨通路场景下应守恒
        - 例如：RasGTP 在 EGFR 通路产生，在 MAPK 通路消耗，总量应守恒
        - 误差 = |产生量 - 消耗量| / max(产生量, 消耗量)
        - 阈值 10%（spec.md 第 298 行）

        实现策略：
        1. 遍历每个 shared species
        2. 从 specialist_outputs 中收集该 species 的产生量与消耗量
           （从 reactions 的 product / substrate 中提取）
        3. 计算守恒误差
        4. 取所有 shared species 的最大误差

        Args:
            shared_species: shared species 名列表（如 ["RasGTP", "AKT"]）
            specialist_outputs: Specialist 输出列表，每条含 reactions

        Returns:
            (max_error, violation_list)
            - max_error: 所有 shared species 中的最大守恒误差（0.0 表示守恒或无 shared species）
            - violation_list: 每条含 species / produced / consumed / error / reason
        """
        if not shared_species:
            # 无 shared species：守恒（误差 0.0）
            return (0.0, [])

        if not specialist_outputs:
            # 有 shared species 但无 Specialist 输出：无法计算守恒，返回 0.0（保守）
            logger.debug(
                "_check_shared_species_conservation: 有 shared_species 但无 specialist_outputs，跳过"
            )
            return (0.0, [])

        # 收集每个 shared species 的产生量与消耗量
        # produced[species] = 总产生量（sum of products across pathways）
        # consumed[species] = 总消耗量（sum of substrates across pathways）
        produced: dict[str, float] = {sp: 0.0 for sp in shared_species}
        consumed: dict[str, float] = {sp: 0.0 for sp in shared_species}

        for output in specialist_outputs:
            if not isinstance(output, dict):
                continue
            reactions = output.get("reactions", []) or []
            crosstalk_reactions = output.get("crosstalk_reactions", []) or []
            for rxn in reactions + crosstalk_reactions:
                if not isinstance(rxn, dict):
                    continue
                # 提取 stoichiometry：从 substrate / product 字段
                # substrate → 消耗，product → 产生
                self._accumulate_stoichiometry(
                    rxn, "substrate", shared_species, consumed
                )
                self._accumulate_stoichiometry(
                    rxn, "product", shared_species, produced
                )
                # 也从 reactants / products（列表形式）提取
                for field in ("reactants", "substrates"):
                    self._accumulate_stoichiometry_list(
                        rxn, field, shared_species, consumed
                    )
                for field in ("products",):
                    self._accumulate_stoichiometry_list(
                        rxn, field, shared_species, produced
                    )

        violations: list[dict[str, Any]] = []
        max_error = 0.0

        for sp in shared_species:
            p = produced.get(sp, 0.0)
            c = consumed.get(sp, 0.0)
            # 守恒误差 = |产生 - 消耗| / max(产生, 消耗, epsilon)
            denom = max(abs(p), abs(c), self._EPSILON)
            error = abs(p - c) / denom
            # 若产生与消耗均为 0，则该 shared species 在 reactions 中未出现，
            # 视为守恒（误差 0.0）
            if abs(p) < self._EPSILON and abs(c) < self._EPSILON:
                error = 0.0
            max_error = max(max_error, error)
            if error > self.SHARED_SPECIES_CONSERVATION_THRESHOLD:
                violations.append({
                    "species": sp,
                    "produced": p,
                    "consumed": c,
                    "error": error,
                    "reason": (
                        f"shared species '{sp}' 跨通路守恒误差 {error:.2%} > "
                        f"阈值 {self.SHARED_SPECIES_CONSERVATION_THRESHOLD:.2%}"
                    ),
                })

        return (max_error, violations)

    def _accumulate_stoichiometry(
        self,
        reaction: dict[str, Any],
        field: str,
        shared_species: list[str],
        accumulator: dict[str, float],
    ) -> None:
        """从 reaction 的 substrate/product 字段累计 stoichiometry。

        支持两种格式：
        - 字符串："substrate": "RasGTP"（按 stoichiometry=1.0 计）
        - dict：{"substrate": {"species": "RasGTP", "stoichiometry": 2.0}}
        """
        val = reaction.get(field)
        if val is None:
            return
        if isinstance(val, str) and val in shared_species:
            accumulator[val] = accumulator.get(val, 0.0) + 1.0
        elif isinstance(val, dict):
            sp_name = val.get("species") or val.get("name") or ""
            if sp_name in shared_species:
                stoich = val.get("stoichiometry", 1.0)
                try:
                    stoich = float(stoich)
                except (TypeError, ValueError):
                    stoich = 1.0
                accumulator[sp_name] = accumulator.get(sp_name, 0.0) + stoich

    def _accumulate_stoichiometry_list(
        self,
        reaction: dict[str, Any],
        field: str,
        shared_species: list[str],
        accumulator: dict[str, float],
    ) -> None:
        """从 reaction 的 reactants/products 列表字段累计 stoichiometry。

        支持列表格式：
        - ["RasGTP", "ATP"]（按 stoichiometry=1.0 计）
        - [{"species": "RasGTP", "stoichiometry": 2.0}]
        """
        val = reaction.get(field)
        if not isinstance(val, list):
            return
        for item in val:
            if isinstance(item, str) and item in shared_species:
                accumulator[item] = accumulator.get(item, 0.0) + 1.0
            elif isinstance(item, dict):
                sp_name = item.get("species") or item.get("name") or ""
                if sp_name in shared_species:
                    stoich = item.get("stoichiometry", 1.0)
                    try:
                        stoich = float(stoich)
                    except (TypeError, ValueError):
                        stoich = 1.0
                    accumulator[sp_name] = accumulator.get(sp_name, 0.0) + stoich

    # =========================================================================
    # SubTask 5.4.4: time-scale alignment 检查
    # =========================================================================
    def _check_time_scale_alignment(
        self, time_scale_alignment: dict[str, Any]
    ) -> bool:
        """检查多通路时间尺度是否对齐。

        检查规则：
        - 消费 v4_time_scale_alignment（来自 Cross-talk Coordinator）
        - 检查 unified_max_step 是否合理（非 None / 非 0 / 非负）
        - 缺失字段或无效值 → 未对齐（False）

        Args:
            time_scale_alignment: 时间尺度对齐结果 dict，含：
                - unified_max_step: float（统一最大步长）
                - pathway_time_scales: list（每通路时间尺度信息）
                - alignment_strategy: str（对齐策略）

        Returns:
            is_aligned: True 表示时间尺度对齐
        """
        if not isinstance(time_scale_alignment, dict):
            return False
        if not time_scale_alignment:
            # 空 dict → 未对齐
            return False

        unified_max_step = time_scale_alignment.get("unified_max_step")
        if unified_max_step is None:
            return False

        # 数值校验：非 0 / 非负 / 可转为 float
        try:
            step = float(unified_max_step)
        except (TypeError, ValueError):
            return False

        if step <= 0:
            # 0 或负值视为未对齐
            return False

        return True

    # =========================================================================
    # 辅助函数：提取输入字段
    # =========================================================================
    def _extract_crosstalk_edges(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """从 state 提取 cross-talk edges。

        优先从 v4_crosstalk_edges 字段取（CrossTalkCoordinator 输出）；
        其次从 v4_pathway_graph.cross_talk_edges 取（PathwayGraph 字段）；
        最后从 v4_pathway_graph.crosstalk_edges 取（Pathway Planner 预识别）。
        """
        edges = state.get("v4_crosstalk_edges")
        if isinstance(edges, list) and edges:
            return edges

        pathway_graph = state.get("v4_pathway_graph")
        if isinstance(pathway_graph, dict):
            # cross_talk_edges（PathwayGraph schema 字段名）
            ct = pathway_graph.get("cross_talk_edges")
            if isinstance(ct, list) and ct:
                return ct
            # crosstalk_edges（Pathway Planner 预识别字段名）
            ct2 = pathway_graph.get("crosstalk_edges")
            if isinstance(ct2, list) and ct2:
                return ct2

        return []

    def _extract_shared_species(self, state: dict[str, Any]) -> list[str]:
        """从 state 提取 shared species 列表。

        优先从 v4_shared_species 字段取（CrossTalkCoordinator 输出）；
        其次从 v4_pathway_graph.shared_species 取。
        """
        ss = state.get("v4_shared_species")
        if isinstance(ss, list) and ss:
            return [str(s) for s in ss if isinstance(s, (str, int, float))]

        pathway_graph = state.get("v4_pathway_graph")
        if isinstance(pathway_graph, dict):
            ss2 = pathway_graph.get("shared_species")
            if isinstance(ss2, list) and ss2:
                return [str(s) for s in ss2 if isinstance(s, (str, int, float))]

        return []

    def _build_failure_report(
        self,
        crosstalk_consistency: bool,
        shared_species_conservation: float,
        time_scale_alignment: bool,
        reason: str = "",
    ) -> dict[str, Any]:
        """构建失败降级报告（pass=False）。

        Args:
            crosstalk_consistency: cross-talk 一致性检查结果
            shared_species_conservation: shared species 守恒误差
            time_scale_alignment: 时间尺度对齐结果
            reason: 失败原因

        Returns:
            失败报告 dict（pass=False）
        """
        report: dict[str, Any] = {
            "pass": False,
            "crosstalk_consistency": crosstalk_consistency,
            "shared_species_conservation": shared_species_conservation,
            "time_scale_alignment": time_scale_alignment,
        }
        if reason:
            report["reason"] = reason
        return report


# =============================================================================
# LangGraph hook 节点（Feature Flag 隔离）
# =============================================================================
def level3_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：Level 3 Cross-Pathway Validation hook。

    行为：
    - V4_VALIDATION_PYRAMID_ENABLED=false：直接返回空 dict（不修改 state）
    - V4_VALIDATION_PYRAMID_ENABLED=true：调用 Level3CrossPathwayValidator.validate()
      写入 state["v4_validation_report"]["level3"]

    严格遵守不可碰清单：
    - 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
    - 不生成 ODE / 不调用 RAG / 不调用 v3 sbml_validator
    - 失败时降级返回空 dict，不抛异常，不阻塞主流水线

    Args:
        state: LangGraph 全局状态

    Returns:
        flag=false 时返回 {}
        flag=true 时返回 {"v4_validation_report": {"level3": {...}}}
        异常时返回 {}
    """
    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not settings.effective_v4_validation_pyramid_enabled():
        logger.debug("V4_VALIDATION_PYRAMID_ENABLED effective=false，跳过 Level 3 validation")
        return {}

    try:
        validator = Level3CrossPathwayValidator()
        level3_report = validator.validate(state)
        # 与现有 v4_validation_report 合并，不覆盖 level1/level2/level4/level5
        existing_report: dict[str, Any] = {}
        if isinstance(state, dict):
            existing = state.get("v4_validation_report")
            if isinstance(existing, dict):
                existing_report = existing
        merged_report = {**existing_report, "level3": level3_report}
        return {"v4_validation_report": merged_report}
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("Level 3 validation hook 失败，降级跳过: %s", exc)
        return {}


__all__ = ["Level3CrossPathwayValidator", "level3_hook_node"]
