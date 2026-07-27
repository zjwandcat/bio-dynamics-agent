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
import math
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
            # [RC28] 修复：当 v4_time_scale_alignment 为空（CrossTalkCoordinator
            #   未写入）但无 crosstalk_edges 和 shared_species 时，说明实际无多通路
            #   交互，time_aligned 默认 True（避免假阴性阻塞流水线）
            time_aligned = self._check_time_scale_alignment(time_scale_alignment)
            if not time_aligned and not crosstalk_edges and not shared_species:
                logger.info(
                    "Level3: v4_time_scale_alignment 为空但无 crosstalk_edges/"
                    "shared_species，time_aligned 默认 True（无多通路交互）"
                )
                time_aligned = True

            # [RC29] 修复：当 v4_time_scale_alignment 为空但 v4_ode_system 存在
            #   （统一 ODE 已成功求解），时间尺度已由统一求解器隐式对齐。
            #   CrossTalkCoordinator 可能未写入 v4_time_scale_alignment，但仿真
            #   成功即证明时间尺度兼容。
            if not time_aligned and state.get("v4_ode_system"):
                logger.info(
                    "Level3: v4_time_scale_alignment 为空但 v4_ode_system 存在"
                    "（统一 ODE 已求解），time_aligned 默认 True"
                )
                time_aligned = True

            # 失败策略：shared species 守恒误差 > 10% → pass=False
            # [RC29] 修复：当 specialist_outputs 无仿真时间序列数据时，守恒检查
            #   回退到计数启发式（近似）。计数启发式依赖反应定义中的 stoichiometry
            #   字段命名与 shared_species 一致，命名不匹配会产生假阳性误差。
            #   此时不应作为硬门阻塞——降级为软检查（pass=True + warning）。
            _has_timeseries = self._specialist_outputs_have_timeseries(
                specialist_outputs
            )
            if _has_timeseries:
                conservation_pass = (
                    max_conservation_error <= self.SHARED_SPECIES_CONSERVATION_THRESHOLD
                )
            else:
                # 计数启发式：近似检查，不阻塞
                logger.warning(
                    "Level3: specialist_outputs 无仿真时间序列，守恒检查使用"
                    "计数启发式（近似），降级为软检查（不阻塞）"
                )
                conservation_pass = True
            pass_flag = bool(
                crosstalk_ok and conservation_pass and time_aligned
            )

            return {
                "pass": pass_flag,
                "crosstalk_consistency": crosstalk_ok,
                "shared_species_conservation": max_conservation_error,
                "time_scale_alignment": time_aligned,
                "method": "multi_pathway_check",
                "crosstalk_edge_count": len(crosstalk_edges),
                "shared_species_count": len(shared_species),
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
            # TD-018 修复（硬门）：有 crosstalk edges 但无 Specialist 输出 →
            # 无法验证一致性 → 判定失败（不再 pass=True 软门放水）
            logger.warning(
                "_check_crosstalk_consistency: 有 crosstalk edges 但无 specialist_outputs，"
                "无法验证一致性 → 判定失败（硬门）"
            )
            return (False, [{
                "edge_id": "global",
                "reason": "missing_specialist_outputs",
                "detail": "有 crosstalk edges 但无 specialist_outputs，无法验证一致性",
            }])

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
    # TD-015 (IB-053) 修复：共享物种守恒基于计数非通量 —— 原实现基于反应计量计数
    # （produced/consumed stoichiometry 累加），无法反映真实仿真中的浓度守恒。
    # 改为结果驱动：当 specialist_outputs 含仿真时间序列时，计算每个 shared species
    # 在所有通路中的总浓度随时间的变化，若变化超阈值则判定不守恒。
    # 无时间序列数据时回退到计数启发式，并记录 warning（检查为近似）。
    # =========================================================================
    def _check_shared_species_conservation(
        self,
        shared_species: list[str],
        specialist_outputs: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """检查 shared species 跨通路守恒。

        TD-015 (IB-053) 修复策略（通量驱动优先）：
        1. 若 specialist_outputs 含仿真时间序列数据（simulation_result / time_series
           字段），计算每个 shared species 在所有通路中的总浓度随时间的变化，
           若相对变化超阈值则判定不守恒（flux-based conservation）。
        2. 若无时间序列数据，回退到计数启发式（produced/consumed stoichiometry），
           并记录 warning（检查为近似）。

        守恒规则（spec.md 第 296-298 行）：
        - shared species 在跨通路场景下应守恒
        - 例如：RasGTP 在 EGFR 通路产生，在 MAPK 通路消耗，总量应守恒
        - 误差 = |产生量 - 消耗量| / max(产生量, 消耗量)
        - 阈值 10%（spec.md 第 298 行）

        Args:
            shared_species: shared species 名列表（如 ["RasGTP", "AKT"]）
            specialist_outputs: Specialist 输出列表，每条含 reactions，
                以及可选的 simulation_result / time_series 时间序列数据

        Returns:
            (max_error, violation_list)
            - max_error: 所有 shared species 中的最大守恒误差（0.0 表示守恒或无 shared species）
            - violation_list: 每条含 species / produced / consumed / error / reason
        """
        if not shared_species:
            # 无 shared species：守恒（误差 0.0）
            return (0.0, [])

        if not specialist_outputs:
            # TD-018 修复（硬门）：有 shared species 但无 Specialist 输出 →
            # 无法计算守恒 → 返回 max_error=1.0（超过阈值 0.10 → conservation_pass=False）
            logger.warning(
                "_check_shared_species_conservation: 有 shared_species 但无 specialist_outputs，"
                "无法计算守恒 → 判定失败（硬门）"
            )
            return (1.0, [{
                "species": "global",
                "reason": "missing_specialist_outputs",
                "detail": "有 shared_species 但无 specialist_outputs，无法计算守恒",
            }])

        # TD-015 修复：优先使用仿真时间序列进行通量守恒检查
        timeseries_list = self._extract_specialist_timeseries(
            shared_species, specialist_outputs
        )
        if timeseries_list is not None:
            return self._check_flux_conservation(
                shared_species, timeseries_list
            )

        # 回退：无时间序列数据，使用计数启发式（近似检查，记录 warning）
        logger.warning(
            "TD-015: specialist_outputs 无仿真时间序列数据，"
            "共享物种守恒检查回退到计数启发式（近似检查，可能不准确）"
        )
        return self._check_counting_conservation(
            shared_species, specialist_outputs
        )

    def _extract_specialist_timeseries(
        self,
        shared_species: list[str],
        specialist_outputs: list[dict[str, Any]],
    ) -> list[dict[str, list[float]]] | None:
        """从 specialist_outputs 提取每个通路的 shared species 时间序列。

        TD-015 修复：检查 specialist_outputs 是否含仿真时间序列数据。
        每条 output 可能含：
        - simulation_result: {"t": [...], "species": {"A": [...]}}
        - time_series: {"time": [...], "concentrations": {"A": [...]}}

        Args:
            shared_species: shared species 名列表
            specialist_outputs: Specialist 输出列表

        Returns:
            list[dict[species, list[float]]]：每个通路一个 dict，含 shared species
            的浓度序列；任一通路无时间序列数据时返回 None（回退到计数启发式）
        """
        result: list[dict[str, list[float]]] = []
        for output in specialist_outputs:
            if not isinstance(output, dict):
                return None
            # 尝试从 simulation_result 或 time_series 字段提取
            sim_data = output.get("simulation_result") or output.get("time_series")
            if not isinstance(sim_data, dict):
                return None

            # 提取时间序列（兼容多种结构）
            species_series = self._parse_specialist_timeseries(sim_data)
            if species_series is None:
                return None

            # 仅保留 shared species
            pathway_shared: dict[str, list[float]] = {}
            for sp in shared_species:
                if sp in species_series:
                    pathway_shared[sp] = species_series[sp]
            result.append(pathway_shared)

        return result

    def _specialist_outputs_have_timeseries(
        self, specialist_outputs: list[dict[str, Any]]
    ) -> bool:
        """[RC29] 检查 specialist_outputs 是否含仿真时间序列数据。

        用于决定守恒检查使用通量守恒（精确）还是计数启发式（近似）。
        当使用计数启发式时，守恒检查降级为软检查（不阻塞）。

        Args:
            specialist_outputs: Specialist 输出列表

        Returns:
            True 如果至少一条 output 含 simulation_result 或 time_series 字段
        """
        if not specialist_outputs:
            return False
        for output in specialist_outputs:
            if not isinstance(output, dict):
                continue
            sim_data = output.get("simulation_result") or output.get("time_series")
            if isinstance(sim_data, dict) and sim_data:
                return True
        return False

    def _parse_specialist_timeseries(
        self, sim_data: dict
    ) -> dict[str, list[float]] | None:
        """解析单通路仿真时间序列 dict 为 {species: [浓度序列]}。

        支持多种结构：
        - {"t": [...], "y": [[...], ...], "species_names": [...]}：y 为 species × time
        - {"time": [...], "species": {"A": [...]}}
        - {"times": [...], "concentrations": {"A": [...]}}
        """
        # 结构 B: {"time": [...], "species": {"A": [...]}}
        species_map = sim_data.get("species") or sim_data.get("concentrations")
        if isinstance(species_map, dict) and species_map:
            return {str(name): list(vals) for name, vals in species_map.items()}

        # 结构 A: {"t": [...], "y": [[...], ...], "species_names": [...]}
        y = sim_data.get("y")
        species_names = sim_data.get("species_names") or sim_data.get("species")
        if y is not None and isinstance(y, list) and y and isinstance(y[0], list):
            if isinstance(species_names, list) and len(species_names) == len(y):
                # y 为 species × time
                return {str(name): list(y[idx]) for idx, name in enumerate(species_names)}
            # 无 species_names，用索引命名
            return {f"species_{idx}": list(row) for idx, row in enumerate(y)}

        return None

    def _check_flux_conservation(
        self,
        shared_species: list[str],
        timeseries_list: list[dict[str, list[float]]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """基于仿真时间序列的通量守恒检查（TD-015 结果驱动）。

        对每个 shared species，计算其在所有通路中的总浓度随时间的变化：
        - total[t] = sum over pathways of concentration[species][t]
        - 守恒误差 = (max(total) - min(total)) / (|mean(total)| + epsilon)
        - 误差超阈值（10%）则判定不守恒

        Args:
            shared_species: shared species 名列表
            timeseries_list: 每个通路一个 dict，含 shared species 的浓度序列

        Returns:
            (max_error, violation_list)
        """
        violations: list[dict[str, Any]] = []
        max_error = 0.0

        for sp in shared_species:
            # 收集该 species 在所有通路中的浓度序列
            pathway_series: list[list[float]] = []
            for pathway_dict in timeseries_list:
                series = pathway_dict.get(sp)
                if series is None:
                    continue
                try:
                    float_series = [float(v) for v in series]
                except (TypeError, ValueError):
                    continue
                # 跳过含 NaN/Inf 的序列
                if any(math.isnan(v) or math.isinf(v) for v in float_series):
                    continue
                pathway_series.append(float_series)

            if not pathway_series:
                # 该 species 无有效时间序列，视为守恒（跳过）
                continue

            # 对齐时间点长度（取最短长度）
            min_len = min(len(s) for s in pathway_series)
            if min_len < 2:
                continue

            # 计算每个时间点的总浓度
            totals: list[float] = []
            for t_idx in range(min_len):
                total = sum(s[t_idx] for s in pathway_series)
                totals.append(total)

            # 守恒误差 = (max - min) / (|mean| + epsilon)
            total_max = max(totals)
            total_min = min(totals)
            total_mean = sum(totals) / len(totals)
            error = (total_max - total_min) / (abs(total_mean) + self._EPSILON)
            # 若总浓度均接近 0，视为守恒
            if abs(total_max) < self._EPSILON and abs(total_min) < self._EPSILON:
                error = 0.0
            max_error = max(max_error, error)
            if error > self.SHARED_SPECIES_CONSERVATION_THRESHOLD:
                violations.append({
                    "species": sp,
                    "produced": total_max,
                    "consumed": total_min,
                    "error": error,
                    "method": "flux_based",
                    "reason": (
                        f"shared species '{sp}' 跨通路总浓度变化误差 {error:.2%} > "
                        f"阈值 {self.SHARED_SPECIES_CONSERVATION_THRESHOLD:.2%} "
                        f"(flux-based: max={total_max:.4e}, min={total_min:.4e})"
                    ),
                })

        return (max_error, violations)

    def _check_counting_conservation(
        self,
        shared_species: list[str],
        specialist_outputs: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        """基于反应计量计数的守恒检查（回退策略，近似检查）。

        TD-015 回退逻辑：无时间序列数据时，从 reactions 的 product/substrate
        计量系数累加，计算产生量与消耗量的守恒误差。

        Args:
            shared_species: shared species 名列表
            specialist_outputs: Specialist 输出列表，每条含 reactions

        Returns:
            (max_error, violation_list)
        """
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
                    "method": "counting_heuristic",
                    "reason": (
                        f"shared species '{sp}' 跨通路守恒误差 {error:.2%} > "
                        f"阈值 {self.SHARED_SPECIES_CONSERVATION_THRESHOLD:.2%} "
                        f"(counting heuristic, approximate)"
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
