# BioDynamics Agent v4 - Experiment Planner（Phase 6 / Task 6.2）
#
# ExperimentDesigner 子组件：为每个假设设计实验验证方案。
#
# 职责（spec.md Part 5 第 367-370 行）：
# - 为每个假设设计实验验证方案
# - 输出 experiment_design：{perturbation, readout, time_points, controls,
#   cell_line, expected_result}
# - 依赖 P4 Specialist 的 Perturbation Module（药物 / KO / 突变）
#
# 设计原则（铁律）：
# 1. 不修改 v3 任何字段；仅消费 hypothesis dict + state.v4_pathway_class
# 2. 失败降级：任何异常返回最小可用 design，不阻塞 Hypothesis Agent
# 3. 不调用 LLM；不调用 RAG；纯规则匹配（基于 P4 Specialist 静态拓扑）
# 4. 不修改 P4 Specialist 注册表（只读 apply_perturbation 接口）
# 5. 输出 6 字段全部填充，缺字段时使用合理默认值
#
# 对应 spec.md Part 5 Experiment Planner（第 367-370 行）+ Part 6（第 440 行）

from __future__ import annotations

import logging
import re
from typing import Any

from app.hypothesis.hypothesis_generator import HypothesisStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# 通路 → 默认细胞系映射
# =============================================================================
# 选取各通路文献中常用的实验细胞系（便于 benchmark 比对）：
# - EGFR_RTK → A431（EGFR 高表达表皮癌细胞）
# - MAPK → A375（BRAF V600E 黑色素瘤）
# - PI3K_AKT_MTOR → MCF7（PTEN 缺失乳腺癌）
# - p53 → MCF7（野生型 p53 乳腺癌）
# - NF_KB → HEK293（NF-κB 报告基因常用）
# - WNT → HEK293T（TCF/LEF 报告基因常用）
# - TGF_BETA → HaCaT（角质形成细胞，TGF-β 应答敏感）
# - JAK_STAT → HELA（IFN 信号经典模型）
# - APOPTOSIS → HELA（凋亡广泛研究模型）
# - CELL_CYCLE → HELA（细胞周期同步化经典模型）
_PATHWAY_CELL_LINE_MAP: dict[str, str] = {
    "EGFR_RTK": "A431",
    "MAPK": "A375",
    "PI3K_AKT_MTOR": "MCF7",
    "P53": "MCF7",
    "NF_KB": "HEK293",
    "WNT": "HEK293T",
    "TGF_BETA": "HaCaT",
    "JAK_STAT": "HELA",
    "APOPTOSIS": "HELA",
    "CELL_CYCLE": "HELA",
}

_DEFAULT_CELL_LINE: str = "HEK293"

# 默认时间点采样方案（分钟）
# 通用 6 点采样：覆盖早期激活 + 中期响应 + 晚期衰减
_DEFAULT_TIME_POINTS: list[int] = [0, 5, 15, 30, 60, 120]

# 振荡假设的密集采样（捕捉多个周期）
_OSCILLATION_TIME_POINTS: list[int] = [0, 15, 30, 45, 60, 90, 120, 180, 240]

# 双稳态假设的长时采样（捕捉 ON/OFF 切换）
_BISTABILITY_TIME_POINTS: list[int] = [0, 30, 60, 120, 240, 480, 720, 1440]


# =============================================================================
# ExperimentDesigner 主类
# =============================================================================
class ExperimentDesigner:
    """为每个假设设计实验验证方案（spec.md Part 5 第 367-370 行）。

    主入口 design(hypothesis, state) -> dict 输出 6 字段 experiment_design：
    - perturbation: {type, agent, target, dose, duration, mechanism, description}
    - readout: {species, metric, threshold}
    - time_points: list[int]（分钟）
    - controls: list[str]
    - cell_line: str
    - expected_result: str

    依赖（spec.md 第 370 行 + 第 440 行）：
    - P4 Specialist 的 Perturbation Module：通过
      ``app.pathways.pathway_registry.get_specialist(pathway_class).apply_perturbation()``
      获取通路特异药物 / KO / 突变候选列表
    - Specialist 未注册或失败时降级到默认扰动（不阻塞）

    用法：
        designer = ExperimentDesigner()
        design = designer.design(hypothesis, state)
        # design = {perturbation, readout, time_points, controls, cell_line, expected_result}
    """

    def __init__(
        self,
        time_points: list[int] | None = None,
        cell_line_map: dict[str, str] | None = None,
    ) -> None:
        """初始化。

        Args:
            time_points: 默认时间点采样方案（分钟）。None → 使用 _DEFAULT_TIME_POINTS。
            cell_line_map: 通路 → 细胞系映射覆盖。None → 使用 _PATHWAY_CELL_LINE_MAP。
        """
        self._time_points: list[int] = (
            list(time_points) if time_points else list(_DEFAULT_TIME_POINTS)
        )
        # 复制并覆盖默认细胞系映射
        self._cell_line_map: dict[str, str] = dict(_PATHWAY_CELL_LINE_MAP)
        if cell_line_map:
            self._cell_line_map.update(cell_line_map)

    # =========================================================================
    # 主入口：design
    # =========================================================================
    def design(
        self,
        hypothesis: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """为单个假设设计实验验证方案。

        Args:
            hypothesis: 假设 dict，含 strategy / target_species / feedback_node /
                threshold_node / target_param / prediction / expected_result
            state: LangGraph 全局状态，含 v4_pathway_class

        Returns:
            6 字段 experiment_design dict（perturbation / readout / time_points /
            controls / cell_line / expected_result）。失败时返回最小可用 design。
        """
        try:
            if not isinstance(hypothesis, dict) or not isinstance(state, dict):
                return self._minimal_design(hypothesis)

            pathway_class = state.get("v4_pathway_class") or hypothesis.get(
                "pathway_class", ""
            )
            if not isinstance(pathway_class, str):
                pathway_class = ""

            # 1. 从 P4 Specialist 的 Perturbation Module 选择扰动方案
            perturbation = self._select_perturbation(hypothesis, pathway_class)

            # 2. 从 hypothesis 推断 readout（species + metric + threshold）
            readout = self._select_readout(hypothesis)

            # 3. 时间点（根据策略调整采样密度）
            time_points = self._select_time_points(hypothesis)

            # 4. 对照组（根据扰动类型选择）
            controls = self._select_controls(perturbation)

            # 5. 细胞系（通路特异默认）
            cell_line = self._select_cell_line(pathway_class)

            # 6. 预期结果（从 hypothesis 派生）
            expected_result = self._select_expected_result(hypothesis)

            design = {
                "perturbation": perturbation,
                "readout": readout,
                "time_points": time_points,
                "controls": controls,
                "cell_line": cell_line,
                "expected_result": expected_result,
            }
            logger.debug(
                "ExperimentDesigner: 假设 %s 实验设计完成 (pathway=%s, perturbation=%s)",
                hypothesis.get("id", "?"),
                pathway_class,
                perturbation.get("agent", "?"),
            )
            return design
        except Exception as exc:
            logger.warning(
                "ExperimentDesigner.design 失败，降级返回最小可用 design: %s",
                exc,
            )
            return self._minimal_design(hypothesis)

    # =========================================================================
    # SubTask 6.2.3：从 P4 Specialist 的 Perturbation Module 获取药物/KO 方案
    # =========================================================================
    def _select_perturbation(
        self,
        hypothesis: dict[str, Any],
        pathway_class: str,
    ) -> dict[str, Any]:
        """根据假设从 P4 Specialist 的 Perturbation Module 选择扰动方案。

        选择优先级：
        1. target_species / feedback_node / threshold_node 匹配 Specialist 输出的 target
        2. 按 strategy 优先级：bistability → KO；oscillation/sensitivity → drug
        3. 无匹配时使用 Specialist 第一个候选
        4. Specialist 不可用时返回默认扰动

        Args:
            hypothesis: 假设 dict
            pathway_class: 通路类别字符串

        Returns:
            规范化的 perturbation dict（含 type/agent/target/dose/duration/
            mechanism/description 字段）
        """
        candidates = self._fetch_perturbation_candidates(pathway_class)

        # 推断期望 target：优先 target_species，其次 feedback_node / threshold_node
        target_species = (
            hypothesis.get("target_species")
            or hypothesis.get("readout_species")
            or ""
        )
        feedback_node = hypothesis.get("feedback_node") or ""
        threshold_node = hypothesis.get("threshold_node") or ""
        target_param = hypothesis.get("target_param") or ""

        # 期望 target 列表（按优先级）
        expected_targets: list[str] = []
        if target_species:
            expected_targets.append(target_species)
        if feedback_node:
            expected_targets.append(feedback_node)
        if threshold_node:
            expected_targets.append(threshold_node)
        if target_param:
            # 灵敏度假设的 target_param 通常是参数名，提取物种名前缀
            # 如 "k_pEGFR_dephos" → "pEGFR"
            param_species = self._extract_species_from_param(target_param)
            if param_species:
                expected_targets.append(param_species)

        # 1. 匹配 expected_targets 中的候选
        for expected in expected_targets:
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                cand_target = str(cand.get("target", "")).upper()
                if cand_target and cand_target == str(expected).upper():
                    return self._normalize_perturbation(cand)

        # 2. 按 strategy 优先级选择
        strategy = hypothesis.get("strategy", "")
        if candidates:
            preferred = None
            if strategy == HypothesisStrategy.BISTABILITY:
                # 双稳态假设优先选 KO 扰动
                preferred = next(
                    (c for c in candidates if isinstance(c, dict) and c.get("ko_target")),
                    None,
                )
            if preferred is None:
                # 振荡/灵敏度假设优先选 drug 扰动
                preferred = next(
                    (c for c in candidates if isinstance(c, dict) and c.get("drug")),
                    None,
                )
            if preferred is None:
                preferred = candidates[0]
            return self._normalize_perturbation(preferred)

        # 3. Specialist 不可用 → 默认扰动
        return self._default_perturbation(hypothesis)

    def _fetch_perturbation_candidates(
        self, pathway_class: str
    ) -> list[dict[str, Any]]:
        """从 P4 Specialist 的 apply_perturbation 获取候选扰动列表。

        复用 P4 Specialist 注册表（只读调用，不修改其代码）：
        - ``app.pathways.pathway_registry.get_specialist(pathway_class)``
        - ``specialist.apply_perturbation(pathway_graph={}, perturbation_points=[])``

        失败降级：返回空列表，由 _select_perturbation 进一步降级为默认扰动。
        """
        if not pathway_class:
            return []
        try:
            from app.pathways.pathway_registry import get_specialist

            specialist = get_specialist(pathway_class)
            if specialist is None:
                return []
            result = specialist.apply_perturbation({}, [])
            if not isinstance(result, list):
                return []
            # 仅保留 dict 候选
            return [c for c in result if isinstance(c, dict)]
        except Exception as exc:
            logger.warning(
                "ExperimentDesigner: 从 P4 Specialist 获取扰动方案失败 "
                "(pathway_class=%s): %s",
                pathway_class,
                exc,
            )
            return []

    @staticmethod
    def _normalize_perturbation(candidate: dict[str, Any]) -> dict[str, Any]:
        """将 Specialist 输出的 perturbation 候选规范化为实验设计字段。

        Specialist 输出结构（P4 _EGFR_PERTURBATIONS 等）：
        - target: 扰动靶点（如 "EGFR"）
        - drug: 药物名（如 "Gefitinib"）
        - mechanism: 机制（如 "inhibition"）
        - ko_target: KO 靶点（如 "EGFR"），与 drug 互斥
        - description: 描述

        规范化输出：
        - type: "drug" | "knockout" | "mutation" | "inhibition"
        - agent: 药物名 / KO 靶点 / 突变名
        - target: 靶点
        - dose: 剂量（默认 IC50 / complete KO）
        - duration: 处理时间
        - mechanism: 机制
        - description: 描述
        """
        drug = candidate.get("drug") or ""
        ko_target = candidate.get("ko_target")
        target = candidate.get("target") or ""
        mechanism = candidate.get("mechanism") or "inhibition"

        if ko_target:
            perturbation_type = "knockout"
            agent = str(ko_target)
            dose = "complete KO"
            duration = "48-72h (CRISPR/Cas9)"
        elif drug:
            perturbation_type = "drug"
            agent = str(drug)
            dose = "IC50"
            duration = "1h pre-treatment + sustained"
        else:
            perturbation_type = "inhibition"
            agent = str(target or "unknown")
            dose = "10 uM"
            duration = "1h pre-treatment"

        return {
            "type": perturbation_type,
            "agent": agent,
            "target": str(target),
            "dose": dose,
            "duration": duration,
            "mechanism": str(mechanism),
            "description": str(candidate.get("description", "")),
        }

    @staticmethod
    def _default_perturbation(hypothesis: dict[str, Any]) -> dict[str, Any]:
        """无 P4 Specialist 数据时构造降级默认扰动。

        根据 hypothesis.strategy 选择合理的默认扰动类型：
        - bistability → knockout（双稳态假设通常需要 KO 验证阈值）
        - oscillation/sensitivity → drug（药物抑制更易实施）
        """
        strategy = hypothesis.get("strategy", "")
        target = (
            hypothesis.get("feedback_node")
            or hypothesis.get("threshold_node")
            or hypothesis.get("target_species")
            or hypothesis.get("target_param")
            or "unknown"
        )

        if strategy == HypothesisStrategy.BISTABILITY:
            return {
                "type": "knockout",
                "agent": f"siRNA-{target}",
                "target": str(target),
                "dose": "25 nM siRNA",
                "duration": "48h transfection",
                "mechanism": "knockdown",
                "description": f"Default siRNA knockdown of {target} (no P4 Specialist)",
            }

        return {
            "type": "drug",
            "agent": f"anti-{target} inhibitor",
            "target": str(target),
            "dose": "10 uM",
            "duration": "1h pre-treatment",
            "mechanism": "inhibition",
            "description": f"Default inhibitor targeting {target} (no P4 Specialist)",
        }

    # =========================================================================
    # readout / time_points / controls / cell_line / expected_result 选择
    # =========================================================================
    def _select_readout(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """从假设提取 readout（species + metric + threshold）。

        readout.species 优先级：
        1. hypothesis.target_species（振荡/双稳态 X 物种）
        2. hypothesis.readout_species（灵敏度假设的输出物种）
        3. hypothesis.feedback_node / threshold_node（被抑制的节点）

        readout.metric：根据策略选择
        - oscillation → "oscillation_amplitude"
        - bistability → "on_off_ratio"
        - sensitivity → "peak"

        readout.threshold：从 prediction 中提取 ">50%" 等百分比
        """
        species = (
            hypothesis.get("target_species")
            or hypothesis.get("readout_species")
            or hypothesis.get("feedback_node")
            or hypothesis.get("threshold_node")
            or ""
        )

        strategy = hypothesis.get("strategy", "")
        if strategy == HypothesisStrategy.OSCILLATION:
            metric = "oscillation_amplitude"
            default_threshold = 0.5  # 振幅下降 >50%
        elif strategy == HypothesisStrategy.BISTABILITY:
            metric = "on_off_ratio"
            default_threshold = 0.7  # ON/OFF 比 <30%（即下降 >70%）
        else:
            metric = "peak"
            default_threshold = 0.3  # 峰值下降 >30%

        # 从 prediction 提取百分比阈值
        threshold = self._extract_threshold_from_prediction(
            hypothesis.get("prediction", ""),
            default=default_threshold,
        )

        return {
            "species": str(species),
            "metric": metric,
            "threshold": threshold,
        }

    def _select_time_points(self, hypothesis: dict[str, Any]) -> list[int]:
        """根据假设策略选择时间点采样方案。"""
        strategy = hypothesis.get("strategy", "")
        if strategy == HypothesisStrategy.OSCILLATION:
            return list(_OSCILLATION_TIME_POINTS)
        if strategy == HypothesisStrategy.BISTABILITY:
            return list(_BISTABILITY_TIME_POINTS)
        return list(self._time_points)

    @staticmethod
    def _select_controls(perturbation: dict[str, Any]) -> list[str]:
        """根据扰动类型选择对照组。

        所有实验都包含：
        - vehicle：溶媒对照（DMSO / PBS）
        - untreated：未处理对照

        drug 扰动额外加：
        - DMSO（药物溶媒对照）

        knockout 扰动额外加：
        - scramble siRNA（非靶向 siRNA 对照）
        """
        controls: list[str] = ["vehicle", "untreated"]
        perturbation_type = perturbation.get("type", "")
        if perturbation_type == "drug":
            controls.append("DMSO")
        elif perturbation_type == "knockout":
            controls.append("scramble siRNA")

        # 去重保序
        seen: set[str] = set()
        unique: list[str] = []
        for c in controls:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def _select_cell_line(self, pathway_class: str) -> str:
        """根据通路类别选择默认细胞系。"""
        if not pathway_class:
            return _DEFAULT_CELL_LINE
        # 支持多通路 "MULTI:EGFR_RTK+PI3K_AKT_MTOR" 形式，取第一个匹配
        for key, value in self._cell_line_map.items():
            if key in pathway_class:
                return value
        return _DEFAULT_CELL_LINE

    @staticmethod
    def _select_expected_result(hypothesis: dict[str, Any]) -> str:
        """从 hypothesis 派生 expected_result。

        优先级：
        1. hypothesis.expected_result（已在 Generator 中生成）
        2. hypothesis.prediction
        3. 默认模板
        """
        expected = hypothesis.get("expected_result") or ""
        if expected and isinstance(expected, str):
            return expected
        prediction = hypothesis.get("prediction") or ""
        if prediction and isinstance(prediction, str):
            return prediction
        return "实验组与对照组相比，readout 出现显著差异（p<0.05）"

    # =========================================================================
    # 辅助函数
    # =========================================================================
    @staticmethod
    def _extract_threshold_from_prediction(
        prediction: str,
        default: float = 0.5,
    ) -> float:
        """从 prediction 文本中提取百分比阈值。

        示例：
        - "敲低 IKK 后，NFkB 的振荡振幅下降 >50%" → 0.5
        - "pEGFR 水平降低 >30%" → 0.3
        - "下降 70%" → 0.7

        Args:
            prediction: 假设 prediction 文本
            default: 未提取到时使用的默认阈值

        Returns:
            阈值（0-1 之间的小数）
        """
        if not isinstance(prediction, str) or not prediction:
            return default
        # 匹配 ">50%" / "50%" / "70 %" 等
        match = re.search(r">?\s*(\d+(?:\.\d+)?)\s*%", prediction)
        if match:
            try:
                percent = float(match.group(1))
                # 百分比转小数（50 → 0.5）
                if percent > 1.0:
                    return percent / 100.0
                return percent
            except (TypeError, ValueError):
                pass
        return default

    @staticmethod
    def _extract_species_from_param(param_name: str) -> str:
        """从参数名提取物种名前缀。

        示例：
        - "k_pEGFR_dephos" → "pEGFR"
        - "V_max_NFkB" → "NFkB"
        - "k1" → ""

        用于灵敏度假设的 target_param 匹配 Specialist 扰动靶点。
        """
        if not isinstance(param_name, str) or not param_name:
            return ""
        # 匹配 _<species>_ 形式（如 k_pEGFR_dephos）
        match = re.search(r"_([A-Za-z][A-Za-z0-9]{1,15})_(?:dephos|phos|deg|syn|bind|act|inact)", param_name)
        if match:
            return match.group(1)
        # 匹配 V_max_<species> 形式
        match = re.search(r"(?:V_max|k_cat|Kd|Km|Ki)_([A-Za-z][A-Za-z0-9]{1,15})", param_name)
        if match:
            return match.group(1)
        return ""

    def _minimal_design(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """失败降级时返回最小可用 experiment_design。"""
        perturbation = self._default_perturbation(hypothesis if isinstance(hypothesis, dict) else {})
        return {
            "perturbation": perturbation,
            "readout": {
                "species": "",
                "metric": "peak",
                "threshold": 0.5,
            },
            "time_points": list(_DEFAULT_TIME_POINTS),
            "controls": ["vehicle", "untreated"],
            "cell_line": _DEFAULT_CELL_LINE,
            "expected_result": (
                hypothesis.get("expected_result", "")
                if isinstance(hypothesis, dict)
                else ""
            ),
        }


__all__ = ["ExperimentDesigner"]
