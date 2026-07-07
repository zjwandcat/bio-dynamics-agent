# BioDynamics Agent v4 - Cross-talk Coordinator Agent (Phase 4 / Task 4.13)
# 协调多通路 shared species 与 cross-talk edges；防止 cross-pathway parameter
# contamination（强制 pathway_tag 隔离）；处理 cross-talk 时间尺度对齐。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_CROSSTALK_COORDINATOR_ENABLED=false 时返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段；不生成 ODE；不做 SBML 验证（职责边界严格）
# 3. 不修改 Specialist 内部 Reaction（仅消费其输出）
# 4. 失败降级：任何异常都返回空更新，不阻塞主流水线
# 5. 单通路场景（pathway_class 不含 "MULTI:"）返回空列表 + 空 sync 策略
# 6. 不重新设计 P1/P2/P3，仅消费其产出
#
# 依赖：
# - P4 10 Specialist 的 apply_crosstalk 输出
# - P2 schema（PathwayGraph / CrossTalkEdge）
# - P3 CrossTalkEdge schema
#
# 参考：
# - spec.md Part 3 Cross-talk Coordinator Agent（第 262-272 行）
# - tasks.md SubTask 4.13.1 / 4.13.2 / 4.13.5 / 4.13.7

from __future__ import annotations

import logging
from typing import Any

from app.crosstalk.crosstalk_edges import CrossTalkEdgeInjector
from app.crosstalk.shared_species_sync import SharedSpeciesSync
from app.crosstalk.time_scale_aligner import TimeScaleAligner
from app.state import set_v4_state

logger = logging.getLogger(__name__)


# =============================================================================
# Specialist 输出字段约定
# =============================================================================
# specialist_outputs: list[dict]，每条 dict 可含：
#   - pathway_class: str（通路类别键，如 "EGFR_RTK"）
#   - crosstalk_reactions: list[dict]（apply_crosstalk 输出，本通路侧 Reaction 片段）
#   - crosstalk_edges: list[dict]（可选，本通路相关的预识别 cross-talk edges）
#   - shared_species: list[str]（CrosstalkModuleData.shared_species）
#   - coordination_strategy: str（"merge" / "alias" / "separate"）
#   - species: list[dict]（apply_core 输出的物种列表，含 name / shared 字段）
#   - reactions: list[dict]（apply_core 输出的反应列表）
#   - kinetics_overrides: dict（apply_core 的动力学参数覆盖）
#   - max_step: float（可选，该通路推荐 max_step，分钟）
#   - time_scale: str（可选，"fast" / "medium" / "slow"）
#   - t_end: float（可选，仿真总时长，分钟）


class CrossTalkCoordinator:
    """Cross-talk Coordinator Agent。

    协调多通路 shared species 与 cross-talk edges，职责包括：
    1. 识别 shared species（跨通路共享，如 Ras/AKT/MEK）
    2. 计算 shared species 同步策略（同一 ODE 变量映射）
    3. 注入 cross-talk edges 到 PathwayGraph
    4. 强制 pathway_tag 隔离（防止 cross-pathway parameter contamination）
    5. 时间尺度对齐（多通路 max_step 统一）

    职责边界（禁止）：
    - 不修改 Specialist 内部 Reaction
    - 不生成 ODE
    - 不做 SBML 验证
    """

    def __init__(self) -> None:
        self._edge_injector = CrossTalkEdgeInjector()
        self._sync = SharedSpeciesSync()
        self._time_aligner = TimeScaleAligner()

    def coordinate(
        self,
        specialist_outputs: list[dict],
        pathway_class: str,
    ) -> dict:
        """主入口：协调多通路 cross-talk。

        Args:
            specialist_outputs: 多个 Specialist 的 apply_crosstalk 输出列表
                （每条 dict 结构见模块顶部字段约定）。
            pathway_class: 通路类别字符串，如 "MULTI:EGFR_RTK+PI3K_AKT_mTOR"
                或单通路 "EGFR_RTK"。

        Returns:
            dict 含：
            - ``v4_crosstalk_edges``: list[dict]（cross-talk edges 列表）
            - ``v4_shared_species``: list[str]（shared species 名列表）
            - ``v4_shared_species_sync``: dict（同步策略）
            - ``v4_time_scale_alignment``: dict（时间尺度对齐结果）
            - ``v4_pathway_tag_isolation``: dict（pathway_tag 隔离结果）
            异常时返回空更新（各字段为空）。
        """
        try:
            # 单通路场景：返回空列表 + 空 sync 策略
            if not pathway_class or "MULTI:" not in pathway_class:
                return self._empty_result()

            if not specialist_outputs or len(specialist_outputs) < 2:
                logger.info(
                    "CrossTalkCoordinator: 多通路场景但 Specialist 输出不足 2 个，返回空"
                )
                return self._empty_result()

            # 1. 识别 shared species
            shared_species = self._identify_shared_species(specialist_outputs)

            # 2. 收集 + 校验 cross-talk edges
            crosstalk_edges = self._collect_crosstalk_edges(specialist_outputs)

            # 3. 计算 shared species 同步策略
            sync_strategy = self._sync.compute_sync_strategy(
                shared_species, specialist_outputs
            )

            # 4. 时间尺度对齐
            time_alignment = self._time_aligner.align_time_scales(
                specialist_outputs
            )

            # 5. pathway_tag 隔离强制
            tag_isolation = self._enforce_pathway_tag_isolation(
                specialist_outputs
            )

            logger.info(
                "CrossTalkCoordinator 完成：pathway_class=%s, shared_species=%d, "
                "crosstalk_edges=%d",
                pathway_class,
                len(shared_species),
                len(crosstalk_edges),
            )

            return {
                "v4_crosstalk_edges": crosstalk_edges,
                "v4_shared_species": shared_species,
                "v4_shared_species_sync": sync_strategy,
                "v4_time_scale_alignment": time_alignment,
                "v4_pathway_tag_isolation": tag_isolation,
            }
        except Exception as exc:
            logger.warning(
                "CrossTalkCoordinator.coordinate 失败，降级返回空: %s", exc
            )
            return self._empty_result()

    def _empty_result(self) -> dict:
        """返回空更新（各字段为空）。"""
        return {
            "v4_crosstalk_edges": [],
            "v4_shared_species": [],
            "v4_shared_species_sync": {},
            "v4_time_scale_alignment": {},
            "v4_pathway_tag_isolation": {
                "tagged_parameters": [],
                "isolation_violations": [],
            },
        }

    def _identify_shared_species(
        self,
        specialist_outputs: list[dict],
    ) -> list[str]:
        """识别跨通路 shared species。

        规则：
        1. 收集每个 Specialist 的 CrosstalkModuleData.shared_species 字段
        2. 跨通路出现的 species（出现在 ≥2 个 Specialist 输出中）标记为 shared
        3. 也包括 Specialist 在 species 中标记 is_shared=True 的（如 RasGTP、p53、p21、AKT、β-catenin）

        Args:
            specialist_outputs: Specialist 输出列表。

        Returns:
            去重的 shared species 名列表（如 ["RasGTP", "AKT", "MEK", "p53", "p21"]）。
        """
        species_count: dict[str, set[str]] = {}  # species → 出现的通路集合
        # species 标记 is_shared=True 的（spec: 也包括 is_shared=True 的，
        # 即使仅出现在 1 个通路也标记为 shared）
        is_shared_marked: set[str] = set()

        for output in specialist_outputs or []:
            if not isinstance(output, dict):
                continue
            pathway_class = output.get("pathway_class", "UNKNOWN")

            # 1. 收集 CrosstalkModuleData.shared_species
            shared_list = output.get("shared_species", []) or []
            for sp in shared_list:
                if isinstance(sp, str) and sp:
                    species_count.setdefault(sp, set()).add(pathway_class)

            # 2. 收集 species 中标记 shared=True 的
            species_list = output.get("species", []) or []
            for sp in species_list:
                if not isinstance(sp, dict):
                    continue
                if sp.get("shared") or sp.get("is_shared"):
                    name = sp.get("name", "")
                    if name:
                        species_count.setdefault(name, set()).add(pathway_class)
                        # spec: 也包括 is_shared=True 的（无论出现在几个通路）
                        is_shared_marked.add(name)

            # 3. 收集 crosstalk_reactions 中的 shared_species 字段
            crosstalk_reactions = output.get("crosstalk_reactions", []) or []
            for rxn in crosstalk_reactions:
                if not isinstance(rxn, dict):
                    continue
                rxn_shared = rxn.get("shared_species", []) or []
                for sp in rxn_shared:
                    if isinstance(sp, str) and sp:
                        species_count.setdefault(sp, set()).add(pathway_class)

        # 跨通路出现（≥2 个通路）的 species 标记为 shared
        # 也包括 is_shared=True 的（spec: 即使仅出现在 1 个通路）
        shared: list[str] = []
        for species, pathways in species_count.items():
            if len(pathways) >= 2 or species in is_shared_marked:
                shared.append(species)

        # 保持稳定顺序（按首次出现）
        return sorted(shared)

    def _collect_crosstalk_edges(
        self,
        specialist_outputs: list[dict],
    ) -> list[dict]:
        """收集并校验 cross-talk edges。

        来源：
        1. 每个 Specialist 输出的 ``crosstalk_edges`` 字段（预识别 edges）
        2. 从 ``crosstalk_reactions`` 派生的 edges（source_pathway 从 pathway_class 推断）

        Args:
            specialist_outputs: Specialist 输出列表。

        Returns:
            校验 + 去重后的 cross-talk edges 列表。
        """
        all_edges: list[dict] = []

        for output in specialist_outputs or []:
            if not isinstance(output, dict):
                continue
            pathway_class = output.get("pathway_class", "")

            # 1. 收集预识别 edges
            pre_edges = output.get("crosstalk_edges", []) or []
            for edge in pre_edges:
                if isinstance(edge, dict):
                    all_edges.append(edge)

            # 2. 从 crosstalk_reactions 派生 edges
            crosstalk_reactions = output.get("crosstalk_reactions", []) or []
            for rxn in crosstalk_reactions:
                if not isinstance(rxn, dict):
                    continue
                derived = self._derive_edge_from_reaction(rxn, pathway_class)
                if derived:
                    all_edges.append(derived)

        # 校验 + 去重
        validated = self._edge_injector.validate_edges(all_edges)
        deduped = self._edge_injector.deduplicate_edges(validated)
        return deduped

    def _derive_edge_from_reaction(
        self,
        reaction: dict,
        source_pathway: str,
    ) -> dict | None:
        """从 cross-talk Reaction 派生 cross-talk edge。

        Args:
            reaction: cross-talk Reaction IR 片段，含 source / target / mechanism 等。
            source_pathway: 源通路类别。

        Returns:
            派生的 cross-talk edge dict，无法派生时返回 None。
        """
        source_node = reaction.get("source", "")
        target_node = reaction.get("target", "")
        mechanism = reaction.get("mechanism", "activation")
        description = reaction.get("description", "")
        shared_species = reaction.get("shared_species", []) or []
        site = reaction.get("site")

        if not source_node or not target_node or not source_pathway:
            return None

        # target_pathway 无法从 reaction 准确推断，使用占位符（Coordinator 不阻塞，
        # 真正的 target_pathway 由 Pathway Planner 预识别 edges 提供）
        target_pathway = reaction.get("target_pathway", "UNKNOWN")

        edge_id = reaction.get("id") or (
            f"CT_DERIVED_{source_pathway}_{source_node}_{target_node}"
        )

        return {
            "id": edge_id,
            "source_pathway": source_pathway,
            "target_pathway": target_pathway,
            "source_node": source_node,
            "target_node": target_node,
            "mechanism": mechanism,
            "shared_species": list(shared_species),
            "site": site,
            "description": description,
        }

    def _enforce_pathway_tag_isolation(
        self,
        specialist_outputs: list[dict],
    ) -> dict:
        """强制 pathway_tag 隔离，防止 cross-pathway parameter contamination。

        规则：
        1. 遍历每个 Specialist 的 reactions 和 kinetics_overrides
        2. 标记 cross-talk 相关参数为 ``CROSSTALK_A_B``（A、B 为通路名）
        3. 防止参数跨通路污染：通路 A 的 kcat 不可被通路 B 的 RAG 检索结果覆盖
        4. 检测隔离违规（同一参数被多个通路覆盖）

        Args:
            specialist_outputs: Specialist 输出列表。

        Returns:
            dict 含：
            - ``tagged_parameters``: list，每条含 parameter_name / pathway_tag / is_crosstalk
            - ``isolation_violations``: list，违规情况（正常应为空）
        """
        tagged_parameters: list[dict[str, Any]] = []
        # 参数 → 出现的通路集合（用于检测违规）
        param_pathways: dict[str, set[str]] = {}

        # 收集所有通路名（用于生成 CROSSTALK_A_B 标签）
        pathway_names = [
            out.get("pathway_class", "")
            for out in specialist_outputs or []
            if isinstance(out, dict) and out.get("pathway_class")
        ]

        for output in specialist_outputs or []:
            if not isinstance(output, dict):
                continue
            pathway_class = output.get("pathway_class", "")
            if not pathway_class:
                continue

            # 1. 遍历 reactions，标记 cross-talk 相关反应的参数
            reactions = output.get("reactions", []) or []
            crosstalk_reactions = output.get("crosstalk_reactions", []) or []

            for rxn in reactions + crosstalk_reactions:
                if not isinstance(rxn, dict):
                    continue
                # 判断是否为 cross-talk 反应（含 shared_species 或跨通路标记）
                is_crosstalk = bool(
                    rxn.get("shared_species")
                    or rxn.get("is_crosstalk")
                    or rxn.get("cross_talk_to")
                )
                # 标记反应的 pathway_tag
                rxn_tag = rxn.get("pathway_tag", pathway_class)

                # 若是 cross-talk 反应，标记为 CROSSTALK_A_B
                if is_crosstalk:
                    # 找出与本通路交互的其他通路
                    other_pathways = [
                        p for p in pathway_names if p != pathway_class
                    ]
                    if other_pathways:
                        # 取首个交互通路生成 CROSSTALK_A_B 标签
                        crosstalk_tag = (
                            f"CROSSTALK_{pathway_class}_{other_pathways[0]}"
                        )
                    else:
                        crosstalk_tag = f"CROSSTALK_{pathway_class}"
                else:
                    crosstalk_tag = rxn_tag

                # 收集反应中的参数（substrate/product/modifier 作为参数名）
                for param_field in ("substrate", "product", "modifier"):
                    param_val = rxn.get(param_field)
                    if param_val and isinstance(param_val, str):
                        param_name = f"{rxn.get('source', '')}_{param_field}_{param_val}"
                        tagged_parameters.append({
                            "parameter_name": param_name,
                            "pathway_tag": crosstalk_tag,
                            "is_crosstalk": is_crosstalk,
                            "reaction_id": rxn.get("id", ""),
                        })
                        param_pathways.setdefault(param_name, set()).add(pathway_class)

                # 对 cross-talk 反应，也收集 source/target 作为参数
                # （crosstalk_reactions 通常只有 source/target，无 substrate/product）
                if is_crosstalk:
                    for param_field in ("source", "target"):
                        param_val = rxn.get(param_field)
                        if param_val and isinstance(param_val, str):
                            param_name = f"{rxn.get('source', '')}_{param_field}_{param_val}"
                            tagged_parameters.append({
                                "parameter_name": param_name,
                                "pathway_tag": crosstalk_tag,
                                "is_crosstalk": is_crosstalk,
                                "reaction_id": rxn.get("id", ""),
                            })
                            param_pathways.setdefault(param_name, set()).add(pathway_class)

            # 2. 遍历 kinetics_overrides，标记参数
            kinetics_overrides = output.get("kinetics_overrides", {}) or {}
            if isinstance(kinetics_overrides, dict):
                for param_name, value in kinetics_overrides.items():
                    # 判断是否为 cross-talk 参数（仅参数名含 crosstalk/cross_talk 字样；
                    # "shared" 不触发，避免误判普通参数为 crosstalk 参数）
                    name_lower = param_name.lower()
                    is_crosstalk_param = any(
                        kw in name_lower
                        for kw in ("crosstalk", "cross_talk")
                    )
                    if is_crosstalk_param:
                        other_pathways = [
                            p for p in pathway_names if p != pathway_class
                        ]
                        if other_pathways:
                            tag = f"CROSSTALK_{pathway_class}_{other_pathways[0]}"
                        else:
                            tag = f"CROSSTALK_{pathway_class}"
                    else:
                        tag = pathway_class

                    tagged_parameters.append({
                        "parameter_name": param_name,
                        "pathway_tag": tag,
                        "is_crosstalk": is_crosstalk_param,
                        "value": value,
                    })
                    param_pathways.setdefault(param_name, set()).add(pathway_class)

        # 3. 检测隔离违规：同一非 crosstalk 参数被多个通路覆盖
        isolation_violations: list[dict[str, Any]] = []
        for param_name, pathways in param_pathways.items():
            if len(pathways) > 1:
                # 检查是否所有通路都标记为 crosstalk（若是则不算违规）
                param_tags = [
                    tp["pathway_tag"]
                    for tp in tagged_parameters
                    if tp["parameter_name"] == param_name
                ]
                all_crosstalk = all(
                    tag.startswith("CROSSTALK_") for tag in param_tags
                )
                if not all_crosstalk:
                    isolation_violations.append({
                        "parameter_name": param_name,
                        "pathways": sorted(pathways),
                        "violation_type": "cross_pathway_parameter_contamination",
                        "description": (
                            f"参数 {param_name} 被多个通路覆盖: "
                            f"{sorted(pathways)}，存在参数污染风险"
                        ),
                    })

        return {
            "tagged_parameters": tagged_parameters,
            "isolation_violations": isolation_violations,
        }


# =============================================================================
# LangGraph 节点 hook（feature flag 隔离）
# =============================================================================
def crosstalk_coordinator_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：Cross-talk Coordinator hook。

    行为：
    - V4_CROSSTALK_COORDINATOR_ENABLED=false：直接返回空 dict（不修改 state）
    - V4_CROSSTALK_COORDINATOR_ENABLED=true + 单通路：返回空列表 + 空 sync 策略
    - V4_CROSSTALK_COORDINATOR_ENABLED=true + 多通路：调用 coordinate() 合并

    严格遵守不可碰清单：
    - 不修改 v3 任何字段
    - 不修改 Specialist 内部 Reaction
    - 不生成 ODE / 不做 SBML 验证
    - 失败时降级返回空 dict，不抛异常

    Args:
        state: LangGraph 全局状态，读取：
            - ``v4_pathway_class``: 通路类别字符串
            - ``v4_specialist_outputs``: Specialist 输出列表（可选）
            - ``v4_pathway_graph``: 通路图（可选，含预识别 crosstalk_edges）

    Returns:
        flag=false 时返回 {}
        flag=true 单通路时返回空列表字段
        flag=true 多通路时返回 v4_crosstalk_edges / v4_shared_species /
        v4_shared_species_sync / v4_time_scale_alignment 字段
    """
    # 延迟导入 config 避免循环依赖
    from app.config import settings

    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not getattr(settings, "V4_CROSSTALK_COORDINATOR_ENABLED", False):
        logger.debug("V4_CROSSTALK_COORDINATOR_ENABLED=false，跳过 Coordinator")
        return {}

    try:
        pathway_class = state.get("v4_pathway_class", "") or ""
        specialist_outputs = state.get("v4_specialist_outputs", []) or []

        # 单通路场景：返回空列表字段（保持 state 结构一致）
        if not pathway_class or "MULTI:" not in pathway_class:
            logger.debug(
                "CrossTalkCoordinator hook: 单通路场景 %s，返回空列表", pathway_class
            )
            return {
                "v4_crosstalk_edges": [],
                "v4_shared_species": [],
                "v4_shared_species_sync": {},
                "v4_time_scale_alignment": {},
            }

        coordinator = CrossTalkCoordinator()
        result = coordinator.coordinate(specialist_outputs, pathway_class)

        logger.info(
            "CrossTalkCoordinator hook 完成：pathway_class=%s, shared_species=%d, "
            "crosstalk_edges=%d",
            pathway_class,
            len(result.get("v4_shared_species", [])),
            len(result.get("v4_crosstalk_edges", [])),
        )

        # 仅返回 v4 字段（不含 v4_pathway_tag_isolation，避免污染 state）
        # Task B.2: 双写 4 个 crosstalk 字段 → v4_state["specialist"][*]
        result_update: dict[str, Any] = {}
        set_v4_state(result_update, "specialist", "crosstalk_edges", result.get("v4_crosstalk_edges", []))
        set_v4_state(result_update, "specialist", "shared_species", result.get("v4_shared_species", []))
        set_v4_state(result_update, "specialist", "shared_species_sync", result.get("v4_shared_species_sync", {}))
        set_v4_state(result_update, "specialist", "time_scale_alignment", result.get("v4_time_scale_alignment", {}))
        return result_update
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("CrossTalkCoordinator hook 失败，降级跳过: %s", exc)
        return {}


__all__ = [
    "CrossTalkCoordinator",
    "crosstalk_coordinator_hook_node",
]
