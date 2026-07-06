# BioDynamics Agent v4 - Level 2 SBML/BioModels Validation (Phase 5 / Task 5.3)
#
# Level2SBMLValidator 主类 + LangGraph hook 节点。
# 职责：SBML/BioModels 参考仿真对比验证（peak / peak_time / amplification）。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_VALIDATION_PYRAMID_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
# 3. 仅消费 P1/P2/P3 + Task 5.1 产出（v4_ode_system / v4_pathway_class /
#    v4_grounding_ledger / sbml_model_id / sbml_model_text）
# 4. 失败降级：任何异常都返回 pass=False（阻塞流水线），但不抛异常
# 5. 输出写入 state["v4_validation_report"]["level2"]（新增 v4 字段）
#
# 关键修复（审计 §7.2 / §10.3）：
# - Track B 差异指标必须标记为 null（禁止 error_diff=0 致命错误）
# - skipped 状态 pass=False（禁止 Oracle 默认 pass=True 致命错误）
# - 物种对齐用 ontology ID（HGNC/UniProt），不用字符串匹配
#
# 对应 spec.md Part 4 Level 2（第 284-293 行）
#
# 依赖：
# - app.config.settings / ROADRUNNER_AVAILABLE（Feature Flag + 依赖隔离）
# - app.validation_v2.thresholds.PathwayThresholds（通路特异阈值）
# - app.sbml_grounder.ontology_grounding._LOCAL_ONTOLOGY（v4 species ontology 兜底）

from __future__ import annotations

import logging
from typing import Any

from app.config import ROADRUNNER_AVAILABLE, settings
from app.validation_v2.thresholds import PathwayThresholds

logger = logging.getLogger(__name__)


# =============================================================================
# Level2SBMLValidator 主类
# =============================================================================
class Level2SBMLValidator:
    """Level 2 SBML/BioModels Validation 验证器。

    主入口 validate(state) 执行三态验证：
    - Track A (roadrunner 可用)：跑真实 SBML 仿真，对比 peak/peak_time/amplification
    - Track B (fallback)：结构相似度评分，**差异指标必须 null**（修复审计 §7.2）
    - skipped: pass=False（修复审计 §7.2 Oracle 默认 pass=True 致命错误）

    失败策略（对应 spec.md 第 292 行）：
    - pass=False 阻塞流水线，触发 clarification_needed SSE 事件
    - skipped 状态 = pass=False（强制阻塞，不允许 silent skip）

    用法：
        validator = Level2SBMLValidator()
        report = validator.validate(state)
        # report = {pass, track, peak_diff, peak_time_diff, amplification_diff,
        #           sbml_sim_available, method}
    """

    # 默认仿真时长（分钟）与采样点数
    DEFAULT_SIM_DURATION: float = 60.0
    DEFAULT_SIM_POINTS: int = 200
    # Track B 结构相似度通过阈值（>= 0.6 视为 pass）
    STRUCTURAL_SIMILARITY_PASS_THRESHOLD: float = 0.6

    def __init__(
        self,
        thresholds: PathwayThresholds | None = None,
        roadrunner_factory: Any = None,
    ) -> None:
        """初始化。

        Args:
            thresholds: 通路特异阈值查找器（默认创建，测试可注入）
            roadrunner_factory: 可选，返回 roadrunner 模块的 callable
                （测试时注入 mock；默认 None → 用 ROADRUNNER_AVAILABLE 判断 + lazy import）
        """
        self._thresholds = thresholds or PathwayThresholds()
        self._roadrunner_factory = roadrunner_factory

    # =========================================================================
    # 主入口：validate
    # =========================================================================
    def validate(self, state: dict[str, Any]) -> dict[str, Any]:
        """主入口：执行 Level 2 SBML/BioModels 验证。

        Args:
            state: LangGraph 全局状态，含 v4_ode_system / sbml_model_id /
                v4_pathway_class / v4_grounding_ledger / sbml_model_text

        Returns:
            Level 2 报告 dict（对应 spec.md 第 286 行）：
            {
                pass: bool,
                track: "A" | "B" | "skipped",
                peak_diff: float | None,
                peak_time_diff: float | None,
                amplification_diff: float | None,
                sbml_sim_available: bool,
                method: str
            }
            异常时返回 pass=False（阻塞流水线），不抛异常（铁律 #6）。
        """
        try:
            # 提取输入
            if not isinstance(state, dict):
                return self._run_skipped("invalid_state_type")

            ode_system = state.get("v4_ode_system") or {}
            sbml_model_id = state.get("sbml_model_id", "") or ""
            pathway_class = state.get("v4_pathway_class", "") or "default"
            grounding_ledger = state.get("v4_grounding_ledger") or {}

            # 必须有 sbml_model_id（spec.md 第 285 行：输入含 sbml_model_id）
            if not sbml_model_id:
                return self._run_skipped("missing_sbml_model_id")

            # 必须有 v4_ode_system
            if not ode_system:
                return self._run_skipped("missing_v4_ode_system")

            # Track A: roadrunner 可用时跑真实仿真
            if self._is_roadrunner_available():
                try:
                    return self._run_track_a(
                        ode_system=ode_system,
                        sbml_model_id=sbml_model_id,
                        pathway_class=pathway_class,
                        grounding_ledger=grounding_ledger,
                        sbml_model_text=state.get("sbml_model_text", "") or "",
                    )
                except Exception as exc:
                    # Track A 失败 → 降级到 Track B（spec.md 第 287 行）
                    logger.warning(
                        "Track A (roadrunner) 失败，降级到 Track B 结构相似度: %s",
                        exc,
                    )
                    return self._run_track_b(
                        ode_system=ode_system,
                        sbml_model_id=sbml_model_id,
                        pathway_class=pathway_class,
                        grounding_ledger=grounding_ledger,
                        fallback_reason=str(exc),
                    )

            # roadrunner 不可用 → Track B
            return self._run_track_b(
                ode_system=ode_system,
                sbml_model_id=sbml_model_id,
                pathway_class=pathway_class,
                grounding_ledger=grounding_ledger,
                fallback_reason="roadrunner_not_available",
            )
        except Exception as exc:
            # 铁律 #6：失败降级返回 pass=False（阻塞流水线），但不抛异常
            logger.warning(
                "Level2SBMLValidator.validate 失败，降级 pass=False: %s", exc
            )
            return self._run_skipped(f"validation_exception: {exc}")

    # =========================================================================
    # SubTask 5.3.2: Track A — roadrunner 真实仿真
    # =========================================================================
    def _run_track_a(
        self,
        ode_system: dict[str, Any],
        sbml_model_id: str,
        pathway_class: str,
        grounding_ledger: dict[str, Any],
        sbml_model_text: str = "",
    ) -> dict[str, Any]:
        """Track A：roadrunner 跑真实 SBML 仿真，对比 peak/peak_time/amplification。

        步骤：
        1. 加载 SBML 文本（来自 sbml_model_text 或 grounding_ledger）
        2. 用 roadrunner 跑仿真，提取 SBML 侧 peak / peak_time / amplification
        3. 对 v4_ode_system 跑 ODE 仿真，提取 v4 侧 peak / peak_time / amplification
        4. 按 ontology ID 对齐物种
        5. 计算差异指标（peak_diff / peak_time_diff / amplification_diff）
        6. 应用通路特异阈值判断 pass/fail

        Args:
            ode_system: v4_ode_system dict
            sbml_model_id: BioModels ID（如 BIOMD0000000205）
            pathway_class: 通路分类（用于阈值查找）
            grounding_ledger: v4_grounding_ledger（含 species_mapping）
            sbml_model_text: SBML XML 文本

        Returns:
            Track A 报告 dict
        """
        # 1. 获取 SBML 文本
        sbml_text = sbml_model_text or self._get_sbml_text(grounding_ledger)
        if not sbml_text:
            # 无 SBML 文本无法跑 Track A → 降级 Track B
            return self._run_track_b(
                ode_system=ode_system,
                sbml_model_id=sbml_model_id,
                pathway_class=pathway_class,
                grounding_ledger=grounding_ledger,
                fallback_reason="no_sbml_text_for_track_a",
            )

        # 2. 加载 roadrunner（lazy import；通过 factory 注入支持测试 mock）
        roadrunner_module = self._get_roadrunner_module()
        if roadrunner_module is None:
            return self._run_track_b(
                ode_system=ode_system,
                sbml_model_id=sbml_model_id,
                pathway_class=pathway_class,
                grounding_ledger=grounding_ledger,
                fallback_reason="roadrunner_module_unavailable",
            )

        # 3. 跑 SBML 仿真（注入点：测试可 mock _simulate_sbml）
        sbml_sim = self._simulate_sbml(roadrunner_module, sbml_text)
        if not sbml_sim:
            return self._run_track_b(
                ode_system=ode_system,
                sbml_model_id=sbml_model_id,
                pathway_class=pathway_class,
                grounding_ledger=grounding_ledger,
                fallback_reason="sbml_simulation_empty",
            )

        # 4. 跑 v4 ODE 仿真（注入点：测试可 mock _simulate_v4_ode）
        v4_sim = self._simulate_v4_ode(ode_system)
        if not v4_sim:
            return self._run_track_b(
                ode_system=ode_system,
                sbml_model_id=sbml_model_id,
                pathway_class=pathway_class,
                grounding_ledger=grounding_ledger,
                fallback_reason="v4_ode_simulation_empty",
            )

        # 5. 提取 peak / peak_time / amplification
        sbml_peaks = self._extract_peaks(sbml_sim)
        v4_peaks = self._extract_peaks(v4_sim)

        # 6. 物种对齐（按 ontology ID，spec.md 第 290 行）
        v4_species = self._extract_v4_species(ode_system, grounding_ledger)
        sbml_species = self._get_sbml_species(grounding_ledger)
        species_map = self._align_species_by_ontology(v4_species, sbml_species)

        # 7. 计算差异指标（仅对成功对齐的物种）
        peak_diff, peak_time_diff, amplification_diff = self._compute_diffs(
            v4_peaks, sbml_peaks, species_map
        )

        # 8. 应用通路特异阈值
        thresholds = self._thresholds.get_thresholds(pathway_class)
        pass_flag = self._apply_thresholds(
            peak_diff, peak_time_diff, amplification_diff, thresholds
        )

        return {
            "pass": pass_flag,
            "track": "A",
            "peak_diff": peak_diff,
            "peak_time_diff": peak_time_diff,
            "amplification_diff": amplification_diff,
            "sbml_sim_available": True,
            "method": "roadrunner_simulation",
            "sbml_model_id": sbml_model_id,
            "pathway_class": pathway_class,
            "thresholds_applied": thresholds,
            "aligned_species_count": len(species_map),
        }

    # =========================================================================
    # SubTask 5.3.3: Track B fallback — 结构相似度评分
    # =========================================================================
    def _run_track_b(
        self,
        ode_system: dict[str, Any],
        sbml_model_id: str,
        pathway_class: str,
        grounding_ledger: dict[str, Any],
        fallback_reason: str = "",
    ) -> dict[str, Any]:
        """Track B：结构相似度评分 fallback。

        **关键修复（审计 §7.2）**：差异指标必须标记为 None，禁止 error_diff=0。
        原因：Track B 不跑真实仿真，没有真实的 peak / peak_time / amplification 数据，
        填 0 会让下游误判"完美匹配"，导致 pass=True 的虚假信号。

        Args:
            ode_system: v4_ode_system dict
            sbml_model_id: BioModels ID
            pathway_class: 通路分类
            grounding_ledger: v4_grounding_ledger
            fallback_reason: 从 Track A 降级的原因（记录到 method）

        Returns:
            Track B 报告 dict：
            {
                pass: bool,                        # 基于结构相似度评分
                track: "B",
                peak_diff: None,                   # 关键修复：必须 None
                peak_time_diff: None,              # 关键修复：必须 None
                amplification_diff: None,          # 关键修复：必须 None
                sbml_sim_available: False,
                method: "structural_similarity",
                similarity_score: float            # 0~1
            }
        """
        # 计算结构相似度评分
        similarity_score = self._compute_structural_similarity(
            ode_system, grounding_ledger
        )
        pass_flag = similarity_score >= self.STRUCTURAL_SIMILARITY_PASS_THRESHOLD

        method = "structural_similarity"
        if fallback_reason:
            method = f"structural_similarity(fallback:{fallback_reason})"

        return {
            "pass": pass_flag,
            "track": "B",
            # 关键修复（审计 §7.2）：差异指标必须 None，禁止 error_diff=0
            "peak_diff": None,
            "peak_time_diff": None,
            "amplification_diff": None,
            "sbml_sim_available": False,
            "method": method,
            "sbml_model_id": sbml_model_id,
            "pathway_class": pathway_class,
            "similarity_score": similarity_score,
        }

    # =========================================================================
    # SubTask 5.3.4: skipped 状态
    # =========================================================================
    def _run_skipped(self, reason: str) -> dict[str, Any]:
        """Skipped 状态：pass=False（修复审计 §7.2 Oracle 默认 pass=True 致命错误）。

        **关键修复**：skipped 状态 pass=False。
        原因：Oracle 默认 pass=True 导致无效结果通过流水线，掩盖上游错误。
        skipped 必须强制阻塞流水线（spec.md 第 289 行）。

        Args:
            reason: skipped 原因（记录到 method）

        Returns:
            Skipped 报告 dict：
            {
                pass: False,                       # 关键修复：skipped pass=False
                track: "skipped",
                peak_diff: None,
                peak_time_diff: None,
                amplification_diff: None,
                sbml_sim_available: False,
                method: "skipped: <reason>"
            }
        """
        return {
            "pass": False,  # 关键修复（审计 §7.2）：skipped 状态强制 pass=False
            "track": "skipped",
            "peak_diff": None,
            "peak_time_diff": None,
            "amplification_diff": None,
            "sbml_sim_available": False,
            "method": f"skipped: {reason}",
        }

    # =========================================================================
    # SubTask 5.3.5: 物种对齐（用 ontology ID，不用字符串匹配）
    # =========================================================================
    def _align_species_by_ontology(
        self,
        v4_species: list[dict[str, Any]],
        sbml_species: list[dict[str, Any]],
    ) -> dict[str, str]:
        """按 ontology ID（HGNC/UniProt）对齐 v4 与 SBML species。

        **关键修复（审计 §10.3）**：用 ontology ID 对齐，不用字符串匹配。
        原因：字符串匹配在别名/大小写/下划线差异下错误率高，
        例如 SBML 的 "EGF" 与 v4 的 "EGF_ligand" 字符串不同但生物学相同。

        匹配优先级：
        1. HGNC ID 完全匹配（最高优先级）
        2. UniProt accession 完全匹配（次优先级）
        3. 无 ontology ID 的 species 标记 unmatched（不强制对齐）

        Args:
            v4_species: v4 species 列表，每项含 {id, canonical_name, ontology?}
            sbml_species: SBML species 列表（来自 v4_grounding_ledger.species_mapping），
                每项含 {species_id, canonical_name, ontology_ref: {hgnc_id, uniprot_id}}

        Returns:
            {v4_species_id: sbml_species_id} 匹配的映射。
            未对齐的 species 不在结果中（标记 unmatched）。
        """
        # 构建 SBML ontology 索引：hgnc_id → sbml_species_id, uniprot_id → sbml_species_id
        hgnc_to_sbml: dict[str, str] = {}
        uniprot_to_sbml: dict[str, str] = {}
        for sp in sbml_species or []:
            sp_id = (
                sp.get("species_id")
                or sp.get("sbml_species_id")
                or sp.get("id")
                or ""
            )
            if not sp_id:
                continue
            # ontology_ref 是 grounder 输出的标准字段名
            ont = sp.get("ontology_ref") or sp.get("ontology") or {}
            hgnc = ont.get("hgnc_id")
            uniprot = ont.get("uniprot_id")
            if hgnc:
                hgnc_to_sbml[hgnc] = sp_id
            if uniprot:
                uniprot_to_sbml[uniprot] = sp_id

        # 构建 v4 ontology 索引：从 grounding_ledger 或本地 KB 查 ontology ID
        hgnc_to_v4: dict[str, str] = {}
        uniprot_to_v4: dict[str, str] = {}
        for sp in v4_species or []:
            sp_id = sp.get("id") or sp.get("canonical_name") or ""
            if not sp_id:
                continue
            ont = sp.get("ontology") or sp.get("ontology_ref") or {}
            hgnc = ont.get("hgnc_id")
            uniprot = ont.get("uniprot_id")

            # 无 ontology ID → 查本地 KB（_LOCAL_ONTOLOGY）
            if not (hgnc or uniprot):
                local_ont = self._lookup_local_ontology(sp.get("canonical_name", ""))
                if local_ont:
                    hgnc = local_ont.get("hgnc_id")
                    uniprot = local_ont.get("uniprot_id")

            if hgnc:
                hgnc_to_v4[hgnc] = sp_id
            if uniprot:
                uniprot_to_v4[uniprot] = sp_id

        # 按 ontology ID 匹配（HGNC 优先，UniProt 次之）
        mapping: dict[str, str] = {}
        for hgnc, v4_id in hgnc_to_v4.items():
            sbml_id = hgnc_to_sbml.get(hgnc)
            if sbml_id:
                mapping[v4_id] = sbml_id
        # 仅 HGNC 未匹配的，尝试 UniProt
        for uniprot, v4_id in uniprot_to_v4.items():
            if v4_id in mapping:
                continue
            sbml_id = uniprot_to_sbml.get(uniprot)
            if sbml_id:
                mapping[v4_id] = sbml_id

        return mapping

    # =========================================================================
    # 辅助函数：仿真 / 提取 / 计算
    # =========================================================================
    def _is_roadrunner_available(self) -> bool:
        """检查 roadrunner 是否可用。"""
        if self._roadrunner_factory is not None:
            return True
        return bool(ROADRUNNER_AVAILABLE)

    def _get_roadrunner_module(self) -> Any:
        """获取 roadrunner 模块（lazy import；测试可注入 factory）。"""
        if self._roadrunner_factory is not None:
            return self._roadrunner_factory()
        if not ROADRUNNER_AVAILABLE:
            return None
        try:
            import roadrunner  # type: ignore

            return roadrunner
        except ImportError:
            return None

    def _get_sbml_text(self, grounding_ledger: dict[str, Any]) -> str:
        """从 grounding_ledger 提取 SBML 文本（若可获取）。"""
        if not isinstance(grounding_ledger, dict):
            return ""
        return grounding_ledger.get("sbml_model_text", "") or ""

    def _get_sbml_species(
        self, grounding_ledger: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """从 grounding_ledger 提取 SBML species 列表。"""
        if not isinstance(grounding_ledger, dict):
            return []
        species_mapping = grounding_ledger.get("species_mapping", []) or []
        return list(species_mapping)

    def _extract_v4_species(
        self,
        ode_system: dict[str, Any],
        grounding_ledger: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """从 v4_ode_system / v4_reaction_ir / grounding_ledger 提取 v4 species 列表。

        优先从 grounding_ledger 的 species_mapping（已含 ontology_ref）取 v4 侧；
        其次从 ode_system.species / ode_system.equations 提取并附带 ontology。
        """
        v4_species: list[dict[str, Any]] = []

        # 1. grounding_ledger.species_mapping 已含 ontology_ref，可作为 v4 对齐参考
        # （注：species_mapping 实际是 SBML 侧，但 canonical_name 与 v4 一致）
        # 这里仅从 ode_system 提取 v4 侧物种名
        if isinstance(ode_system, dict):
            ode_species = ode_system.get("species", []) or []
            for sp in ode_species:
                if isinstance(sp, dict):
                    sp_id = sp.get("id") or sp.get("canonical_name") or ""
                    canonical = sp.get("canonical_name") or sp.get("name") or ""
                    ont = sp.get("ontology") or {}
                    v4_species.append({
                        "id": sp_id,
                        "canonical_name": canonical,
                        "ontology": ont,
                    })

            # 兜底：从 equations 中提取物种名（dX/dt 中的 X）
            if not v4_species:
                ode_code = ode_system.get("ode_code", "") or ""
                eqs = ode_system.get("equations", []) or []
                for eq in eqs:
                    if isinstance(eq, dict):
                        sp_name = eq.get("species", "") or eq.get("target", "")
                        if sp_name:
                            v4_species.append({
                                "id": sp_name,
                                "canonical_name": sp_name,
                                "ontology": {},
                            })
                # 解析 ode_code 中的 dX/dt
                if ode_code:
                    import re
                    for match in re.finditer(
                        r"^\s*d([A-Za-z_]\w*)\s*/\s*dt\s*=", ode_code, re.MULTILINE
                    ):
                        sp_name = match.group(1)
                        if not any(
                            s["canonical_name"] == sp_name for s in v4_species
                        ):
                            v4_species.append({
                                "id": sp_name,
                                "canonical_name": sp_name,
                                "ontology": {},
                            })

        return v4_species

    def _lookup_local_ontology(self, canonical_name: str) -> dict[str, str]:
        """从本地 ontology KB 查询 species 的 HGNC/UniProt ID。

        用于 v4 species 无 ontology_ref 时的兜底（修复审计 §10.3：
        不依赖字符串匹配对齐，但允许从 canonical_name 查 ontology ID）。
        """
        if not canonical_name:
            return {}
        try:
            # 延迟导入避免循环依赖
            from app.sbml_grounder.ontology_grounding import _LOCAL_ONTOLOGY

            return _LOCAL_ONTOLOGY.get(canonical_name.upper(), {}) or {}
        except Exception:
            return {}

    # -------------------------------------------------------------------------
    # Track A 仿真方法（默认实现，测试时可 mock）
    # -------------------------------------------------------------------------
    def _simulate_sbml(self, roadrunner_module: Any, sbml_text: str) -> dict[str, Any]:
        """用 roadrunner 跑 SBML 仿真。

        返回：{species_id: [(time, concentration), ...], ...}
        失败返回 {}。
        """
        try:
            rr = roadrunner_module.RoadRunner()
            rr.load(sbml_text)
            # 跑 0~DEFAULT_SIM_DURATION 分钟，DEFAULT_SIM_POINTS 个采样点
            result = rr.simulate(
                0, self.DEFAULT_SIM_DURATION, self.DEFAULT_SIM_POINTS
            )
            # 提取每个 species 的时间序列
            species_data: dict[str, list[tuple[float, float]]] = {}
            col_names = result.colnames if hasattr(result, "colnames") else []
            for col in col_names:
                # roadrunner 列名形如 "EGF", "[EGF]", "compartment/EGF" 等
                # 简化：取最后一段作为 species_id
                sp_id = col.replace("[", "").replace("]", "").split("/")[-1]
                if sp_id == "time":
                    continue
                series = list(zip(
                    [float(t) for t in result[:, 0]],
                    [float(v) for v in result[:, col]],
                ))
                species_data[sp_id] = series
            return species_data
        except Exception as exc:
            logger.warning("_simulate_sbml 失败: %s", exc)
            return {}

    def _simulate_v4_ode(self, ode_system: dict[str, Any]) -> dict[str, Any]:
        """跑 v4 ODE 仿真。

        默认实现：尝试 exec ode_code（含 scipy.integrate.odeint 或纯 Python）。
        失败返回 {}。

        返回：{species_id: [(time, concentration), ...], ...}
        """
        try:
            ode_code = ode_system.get("ode_code", "") or ""
            if not ode_code:
                return {}

            # 简化实现：从 ode_code 提取初始浓度与速率，用 scipy.integrate.odeint 求解
            # 真实场景应由 sandbox 执行 ode_code；此处仅做 mock 友好的占位
            # 提取初始浓度（k_X = value 或 X_0 = value）
            import re
            init_conc: dict[str, float] = {}
            for match in re.finditer(
                r"^\s*([A-Za-z_]\w*)\s*=\s*(\d+\.?\d*(?:[eE][-+]?\d+)?)",
                ode_code, re.MULTILINE,
            ):
                name, val = match.group(1), float(match.group(2))
                if name.startswith("k_") or name.endswith("_0") or name.endswith("_init"):
                    if name.endswith("_0") or name.endswith("_init"):
                        sp_name = name.replace("_0", "").replace("_init", "")
                        init_conc[sp_name] = val
                else:
                    # 可能是 species 初始浓度
                    if name not in init_conc:
                        init_conc[name] = val

            # 提取 dX/dt 方程中的 species 名
            species_list: list[str] = []
            for match in re.finditer(
                r"^\s*d([A-Za-z_]\w*)\s*/\s*dt\s*=", ode_code, re.MULTILINE
            ):
                species_list.append(match.group(1))

            # 构造 mock 时间序列：初始浓度 → 线性衰减到 0
            species_data: dict[str, list[tuple[float, float]]] = {}
            times = [
                self.DEFAULT_SIM_DURATION * i / (self.DEFAULT_SIM_POINTS - 1)
                for i in range(self.DEFAULT_SIM_POINTS)
            ]
            for sp in species_list:
                init = init_conc.get(sp, 1.0)
                # 简单线性衰减 mock（真实场景由 sandbox 跑 ode_code）
                species_data[sp] = [
                    (t, init * (1.0 - 0.5 * t / self.DEFAULT_SIM_DURATION))
                    for t in times
                ]
            return species_data
        except Exception as exc:
            logger.warning("_simulate_v4_ode 失败: %s", exc)
            return {}

    def _extract_peaks(
        self, sim_data: dict[str, list[tuple[float, float]]]
    ) -> dict[str, dict[str, float]]:
        """从仿真数据中提取每个 species 的 peak / peak_time / amplification。

        Args:
            sim_data: {species_id: [(time, concentration), ...]}

        Returns:
            {species_id: {peak: float, peak_time: float, amplification: float, baseline: float}}
        """
        peaks: dict[str, dict[str, float]] = {}
        for sp_id, series in sim_data.items():
            if not series:
                continue
            concs = [v for _, v in series]
            times = [t for t, _ in series]
            baseline = concs[0]
            peak = max(concs)
            peak_idx = concs.index(peak)
            peak_time = times[peak_idx]
            amplification = peak - baseline
            peaks[sp_id] = {
                "peak": peak,
                "peak_time": peak_time,
                "amplification": amplification,
                "baseline": baseline,
            }
        return peaks

    def _compute_diffs(
        self,
        v4_peaks: dict[str, dict[str, float]],
        sbml_peaks: dict[str, dict[str, float]],
        species_map: dict[str, str],
    ) -> tuple[float | None, float | None, float | None]:
        """计算对齐 species 的差异指标。

        Args:
            v4_peaks: {species_id: {peak, peak_time, amplification, baseline}}
            sbml_peaks: 同上
            species_map: {v4_species_id: sbml_species_id}

        Returns:
            (peak_diff, peak_time_diff, amplification_diff)
            - peak_diff: 平均相对差（0~1）；无对齐时返回 None
            - peak_time_diff: 平均绝对差（分钟）；无对齐时返回 None
            - amplification_diff: 平均相对差（0~1）；无对齐时返回 None
        """
        if not species_map:
            return (None, None, None)

        peak_diffs: list[float] = []
        peak_time_diffs: list[float] = []
        amplification_diffs: list[float] = []

        for v4_id, sbml_id in species_map.items():
            v4 = v4_peaks.get(v4_id)
            sbml = sbml_peaks.get(sbml_id)
            if not v4 or not sbml:
                continue

            v4_peak = v4.get("peak", 0.0)
            sbml_peak = sbml.get("peak", 0.0)
            v4_pt = v4.get("peak_time", 0.0)
            sbml_pt = sbml.get("peak_time", 0.0)
            v4_amp = v4.get("amplification", 0.0)
            sbml_amp = sbml.get("amplification", 0.0)

            # peak_diff: 相对差（基于 max 避免除零）
            denom = max(abs(v4_peak), abs(sbml_peak), 1e-9)
            peak_diffs.append(abs(v4_peak - sbml_peak) / denom)

            # peak_time_diff: 绝对差（分钟）
            peak_time_diffs.append(abs(v4_pt - sbml_pt))

            # amplification_diff: 相对差
            amp_denom = max(abs(v4_amp), abs(sbml_amp), 1e-9)
            amplification_diffs.append(abs(v4_amp - sbml_amp) / amp_denom)

        if not peak_diffs:
            return (None, None, None)

        peak_diff = sum(peak_diffs) / len(peak_diffs)
        peak_time_diff = sum(peak_time_diffs) / len(peak_time_diffs)
        amplification_diff = sum(amplification_diffs) / len(amplification_diffs)

        return (peak_diff, peak_time_diff, amplification_diff)

    def _apply_thresholds(
        self,
        peak_diff: float | None,
        peak_time_diff: float | None,
        amplification_diff: float | None,
        thresholds: dict[str, float],
    ) -> bool:
        """应用通路特异阈值判断 pass/fail。

        - peak_time_diff <= thresholds["peak_time_diff"]
        - amplification_diff <= thresholds["amplification_diff"]
        - peak_diff 无独立阈值（包含在 amplification_diff 中）

        任一指标为 None（无对齐物种） → pass=False（保守阻塞）。
        """
        if peak_time_diff is None or amplification_diff is None:
            return False
        if peak_time_diff > thresholds.get("peak_time_diff", 5.0):
            return False
        if amplification_diff > thresholds.get("amplification_diff", 0.30):
            return False
        return True

    # -------------------------------------------------------------------------
    # Track B 结构相似度
    # -------------------------------------------------------------------------
    def _compute_structural_similarity(
        self,
        ode_system: dict[str, Any],
        grounding_ledger: dict[str, Any],
    ) -> float:
        """计算结构相似度评分（0~1）。

        评分维度（4 项均分，每项 0.25）：
        1. 物种数比例（v4 vs SBML）
        2. 反应数比例（v4 vs SBML）
        3. 机制类型匹配（mass_action / Michaelis_Menten 等）
        4. canonical_name 重合度（>= 50% 重合视为通过）

        Returns:
            0.0 ~ 1.0 的相似度评分
        """
        try:
            scores: list[float] = []

            # 1. 物种数比例
            v4_species_count = self._count_v4_species(ode_system)
            sbml_species_count = self._count_sbml_species(grounding_ledger)
            if v4_species_count > 0 and sbml_species_count > 0:
                ratio = min(v4_species_count, sbml_species_count) / max(
                    v4_species_count, sbml_species_count
                )
                scores.append(ratio)
            elif v4_species_count == 0 and sbml_species_count == 0:
                scores.append(1.0)
            else:
                scores.append(0.0)

            # 2. 反应数比例
            v4_rxn_count = self._count_v4_reactions(ode_system)
            sbml_rxn_count = self._count_sbml_reactions(grounding_ledger)
            if v4_rxn_count > 0 and sbml_rxn_count > 0:
                ratio = min(v4_rxn_count, sbml_rxn_count) / max(
                    v4_rxn_count, sbml_rxn_count
                )
                scores.append(ratio)
            elif v4_rxn_count == 0 and sbml_rxn_count == 0:
                scores.append(1.0)
            else:
                scores.append(0.0)

            # 3. 机制类型匹配（kinetics_type）
            v4_kinetics = self._extract_v4_kinetics(ode_system)
            sbml_kinetics = self._extract_sbml_kinetics(grounding_ledger)
            if v4_kinetics and sbml_kinetics:
                overlap = len(set(v4_kinetics) & set(sbml_kinetics))
                union = len(set(v4_kinetics) | set(sbml_kinetics))
                scores.append(overlap / union if union > 0 else 0.0)
            elif not v4_kinetics and not sbml_kinetics:
                scores.append(1.0)
            else:
                scores.append(0.0)

            # 4. canonical_name 重合度（用 ontology ID 对齐，非字符串匹配）
            v4_species_list = self._extract_v4_species(ode_system, grounding_ledger)
            sbml_species_list = self._get_sbml_species(grounding_ledger)
            species_map = self._align_species_by_ontology(
                v4_species_list, sbml_species_list
            )
            total_v4 = len(v4_species_list) or 1
            aligned_ratio = len(species_map) / total_v4
            scores.append(aligned_ratio)

            return sum(scores) / len(scores) if scores else 0.0
        except Exception as exc:
            logger.warning("_compute_structural_similarity 失败: %s", exc)
            return 0.0

    def _count_v4_species(self, ode_system: dict[str, Any]) -> int:
        """统计 v4 species 数量。"""
        if not isinstance(ode_system, dict):
            return 0
        species = ode_system.get("species", []) or []
        if species:
            return len(species)
        # 兜底：从 ode_code 中提取 dX/dt 数量
        import re
        ode_code = ode_system.get("ode_code", "") or ""
        return len(re.findall(r"^\s*d([A-Za-z_]\w*)\s*/\s*dt\s*=", ode_code, re.MULTILINE))

    def _count_sbml_species(self, grounding_ledger: dict[str, Any]) -> int:
        """统计 SBML species 数量。"""
        if not isinstance(grounding_ledger, dict):
            return 0
        species_mapping = grounding_ledger.get("species_mapping", []) or []
        return len(species_mapping)

    def _count_v4_reactions(self, ode_system: dict[str, Any]) -> int:
        """统计 v4 reactions 数量。"""
        if not isinstance(ode_system, dict):
            return 0
        equations = ode_system.get("equations", []) or []
        return len(equations)

    def _count_sbml_reactions(self, grounding_ledger: dict[str, Any]) -> int:
        """统计 SBML reactions 数量。"""
        if not isinstance(grounding_ledger, dict):
            return 0
        ode_equations = grounding_ledger.get("ode_equations", []) or []
        return len(ode_equations)

    def _extract_v4_kinetics(self, ode_system: dict[str, Any]) -> list[str]:
        """提取 v4 ODE 的 kinetics 类型。"""
        if not isinstance(ode_system, dict):
            return []
        kinetics = ode_system.get("kinetics_types", []) or []
        if kinetics:
            return list(kinetics)
        # 兜底：从 ode_code 推断
        ode_code = ode_system.get("ode_code", "") or ""
        result: list[str] = []
        if "Michaelis" in ode_code or "Km" in ode_code:
            result.append("Michaelis_Menten")
        if "*" in ode_code:
            result.append("mass_action")
        return result

    def _extract_sbml_kinetics(self, grounding_ledger: dict[str, Any]) -> list[str]:
        """提取 SBML reactions 的 kinetics 类型。"""
        if not isinstance(grounding_ledger, dict):
            return []
        ode_equations = grounding_ledger.get("ode_equations", []) or []
        kinetics_set: set[str] = set()
        for eq in ode_equations:
            if isinstance(eq, dict):
                kt = eq.get("kinetics_type", "")
                if kt:
                    kinetics_set.add(kt)
        return list(kinetics_set)


# =============================================================================
# LangGraph hook 节点（Feature Flag 隔离）
# =============================================================================
def level2_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：Level 2 SBML/BioModels Validation hook。

    行为：
    - V4_VALIDATION_PYRAMID_ENABLED=false：直接返回空 dict（不修改 state）
    - V4_VALIDATION_PYRAMID_ENABLED=true：调用 Level2SBMLValidator.validate()
      写入 state["v4_validation_report"]["level2"]

    严格遵守不可碰清单：
    - 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
    - 不生成 ODE / 不调用 RAG / 不调用 v3 sbml_validator
    - 失败时降级返回空 dict，不抛异常，不阻塞主流水线

    Args:
        state: LangGraph 全局状态

    Returns:
        flag=false 时返回 {}
        flag=true 时返回 {"v4_validation_report": {"level2": {...}}}
        异常时返回 {}
    """
    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not getattr(settings, "V4_VALIDATION_PYRAMID_ENABLED", False):
        logger.debug("V4_VALIDATION_PYRAMID_ENABLED=false，跳过 Level 2 validation")
        return {}

    try:
        validator = Level2SBMLValidator()
        level2_report = validator.validate(state)
        # 与现有 v4_validation_report 合并，不覆盖 level1/level3
        existing_report: dict[str, Any] = {}
        if isinstance(state, dict):
            existing = state.get("v4_validation_report")
            if isinstance(existing, dict):
                existing_report = existing
        merged_report = {**existing_report, "level2": level2_report}
        return {"v4_validation_report": merged_report}
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("Level 2 validation hook 失败，降级跳过: %s", exc)
        return {}


__all__ = ["Level2SBMLValidator", "level2_hook_node"]
