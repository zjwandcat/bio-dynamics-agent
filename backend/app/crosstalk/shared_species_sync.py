# BioDynamics Agent v4 - Shared Species Sync (Phase 4 / Task 4.13.3)
# Shared species 同步策略：同一 shared species 在多个通路 ODE 中映射到同一变量。
#
# 设计原则（铁律）：
# 1. 不修改 v3 任何字段；不生成 ODE；不做 SBML 验证（职责边界严格）
# 2. 失败降级：任何异常返回空 dict，不阻塞主流水线
# 3. 主导通路选择规则：物种在该通路中是 "produced"（Core 模块）的优先；否则按通路优先级
# 4. 冲突解决：参数差异时取中位数或通路特异参数标记 pathway_tag（避免参数污染）
#
# 参考：
# - spec.md Part 3 Cross-talk Coordinator Agent（第 262-272 行）
# - tasks.md SubTask 4.13.3

from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)


class SharedSpeciesSync:
    """Shared species 同步策略计算器。

    为多通路场景中的 shared species 计算同步策略：
    - ``sync_map``: species_name → canonical_ode_variable（同一 ODE 变量映射）
    - ``pathway_assignments``: species → 主导通路
    - ``conflict_resolution``: 冲突解决策略

    职责边界：
    - 不修改 Specialist 内部 Reaction
    - 不生成 ODE
    - 不做 SBML 验证
    """

    def compute_sync_strategy(
        self,
        shared_species: list[str],
        specialist_outputs: list[dict],
    ) -> dict:
        """计算 shared species 同步策略。

        Args:
            shared_species: shared species 名列表（如 ["RasGTP", "AKT"]）。
            specialist_outputs: 多通路 Specialist 输出列表，每条 dict 可含：
                - ``pathway_class``: 通路类别键
                - ``species``: 该通路物种列表（每条含 name / shared 等字段）
                - ``reactions``: 该通路反应列表（用于判断 produced/consumed）
                - ``crosstalk_reactions``: cross-talk 反应列表
                - ``kinetics_overrides``: 动力学参数覆盖

        Returns:
            dict 含：
            - ``sync_map``: {species_name: canonical_ode_variable}
            - ``pathway_assignments``: {species_name: dominant_pathway_class}
            - ``conflict_resolution``: {species_name: {"strategy": str, "params": dict}}
            异常时返回空 dict。
        """
        try:
            if not shared_species:
                return {
                    "sync_map": {},
                    "pathway_assignments": {},
                    "conflict_resolution": {},
                }

            sync_map: dict[str, str] = {}
            pathway_assignments: dict[str, str] = {}
            conflict_resolution: dict[str, dict[str, Any]] = {}

            for species in shared_species:
                # 1. 计算 canonical ODE 变量（与 species 同名，确保同一变量）
                sync_map[species] = species

                # 2. 选择主导通路
                dominant = self._select_dominant_pathway(
                    species, specialist_outputs
                )
                pathway_assignments[species] = dominant

                # 3. 冲突解决
                conflict_resolution[species] = self._resolve_conflicts(
                    species, specialist_outputs
                )

            return {
                "sync_map": sync_map,
                "pathway_assignments": pathway_assignments,
                "conflict_resolution": conflict_resolution,
            }
        except Exception as exc:
            logger.warning(
                "SharedSpeciesSync.compute_sync_strategy 失败: %s", exc
            )
            return {}

    def _select_dominant_pathway(
        self,
        species: str,
        specialist_outputs: list[dict],
    ) -> str:
        """选择 shared species 的主导通路。

        规则（优先级递减）：
        1. 物种在该通路中是 "produced"（作为 core reaction 的 product）的优先
        2. 物种在该通路的核心 species 列表中（Core 模块成员）
        3. 否则按通路优先级（首个出现的通路）

        Args:
            species: shared species 名。
            specialist_outputs: Specialist 输出列表。

        Returns:
            主导通路 pathway_class。无匹配返回空字符串。
        """
        produced_in: list[str] = []
        in_core_species: list[str] = []
        appears_in: list[str] = []

        for output in specialist_outputs or []:
            pathway_class = output.get("pathway_class", "")
            if not pathway_class:
                continue

            # 检查 species 是否出现在该通路的物种列表中
            species_list = output.get("species", []) or []
            reactions = output.get("reactions", []) or []
            crosstalk_reactions = output.get("crosstalk_reactions", []) or []

            # 物种出现在该通路（species name 匹配）
            appears = self._species_appears(species, species_list, reactions, crosstalk_reactions)
            if appears:
                appears_in.append(pathway_class)

            # 物种在该通路的核心 species 列表中（Core 模块成员）
            for sp in species_list:
                if isinstance(sp, dict) and sp.get("name") == species:
                    in_core_species.append(pathway_class)
                    break
                elif isinstance(sp, str) and sp == species:
                    in_core_species.append(pathway_class)
                    break

            # 物种在该通路中是 produced（作为 core reaction 的 product）
            if self._species_is_produced(species, reactions, crosstalk_reactions):
                produced_in.append(pathway_class)

        # 优先级 1：produced 的通路（首个）
        if produced_in:
            return produced_in[0]

        # 优先级 2：在核心 species 列表中的通路（首个）
        if in_core_species:
            return in_core_species[0]

        # 否则返回首个出现的通路
        if appears_in:
            return appears_in[0]

        return ""

    def _species_appears(
        self,
        species: str,
        species_list: list[dict],
        reactions: list[dict],
        crosstalk_reactions: list[dict],
    ) -> bool:
        """检查 species 是否出现在通路的物种/反应中。"""
        # 物种列表中直接匹配 name
        for sp in species_list:
            if isinstance(sp, dict):
                if sp.get("name") == species:
                    return True
                # 也检查 shared 标记
                if sp.get("shared") and sp.get("name") == species:
                    return True
            elif isinstance(sp, str) and sp == species:
                return True

        # 反应中作为 source/target/substrate/product/modifier 出现
        for rxn in reactions + crosstalk_reactions:
            if not isinstance(rxn, dict):
                continue
            for key in ("source", "target", "substrate", "product", "modifier"):
                if rxn.get(key) == species:
                    return True
            # shared_species 字段
            if species in (rxn.get("shared_species") or []):
                return True
        return False

    def _species_is_produced(
        self,
        species: str,
        reactions: list[dict],
        crosstalk_reactions: list[dict],
    ) -> bool:
        """检查 species 是否在该通路中作为 product 被生成。"""
        for rxn in reactions + crosstalk_reactions:
            if not isinstance(rxn, dict):
                continue
            # product 字段匹配
            if rxn.get("product") == species:
                return True
            # target 字段匹配（部分反应用 target 表示产物）
            if rxn.get("target") == species and rxn.get("mechanism") in (
                "phosphorylation", "activation", "gtp_gdp_exchange",
                "complex_formation", "binding", "transcription", "translation",
            ):
                return True
        return False

    def _resolve_conflicts(
        self,
        species: str,
        specialist_outputs: list[dict],
    ) -> dict[str, Any]:
        """解决 shared species 参数冲突。

        策略：
        - 收集各通路对该 species 的 kinetics 参数
        - 参数差异时取中位数（数值参数）或标记 pathway_tag（避免污染）
        - 无冲突时返回 no_conflict

        Args:
            species: shared species 名。
            specialist_outputs: Specialist 输出列表。

        Returns:
            dict 含 strategy / params / pathway_tags。
        """
        param_values: list[float] = []
        pathway_tags: list[str] = []

        for output in specialist_outputs or []:
            pathway_class = output.get("pathway_class", "")
            if not pathway_class:
                continue

            kinetics_overrides = output.get("kinetics_overrides", {}) or {}
            if not isinstance(kinetics_overrides, dict):
                continue

            # 收集与该 species 相关的参数（key 含 species 名）
            for param_name, value in kinetics_overrides.items():
                if species.lower() in param_name.lower():
                    if isinstance(value, (int, float)):
                        param_values.append(float(value))
                    pathway_tags.append(pathway_class)

        if not param_values:
            return {
                "strategy": "no_conflict",
                "params": {},
                "pathway_tags": list(set(pathway_tags)),
            }

        # 有数值参数：取中位数
        median_val = statistics.median(param_values) if param_values else None

        return {
            "strategy": "median_value" if len(set(param_values)) > 1 else "no_conflict",
            "params": {
                "median": median_val,
                "values": param_values,
            },
            "pathway_tags": list(set(pathway_tags)),
        }


__all__ = ["SharedSpeciesSync"]
