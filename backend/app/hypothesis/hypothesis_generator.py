# BioDynamics Agent v4 - Hypothesis Generator（Phase 6 / Task 6.1）
#
# HypothesisGenerator 子组件：从仿真特征（振荡 / 双稳态 / 灵敏度）生成候选假设。
#
# 设计原则（铁律）：
# 1. 不修改 v3 任何字段；仅消费 metrics / feature_metadata / v4_sensitivity_report
# 2. 失败降级：任何异常都返回空列表，不阻塞 Hypothesis Agent
# 3. 输出候选假设列表（未过滤，由 Falsification Checker 过滤）
# 4. 每个候选假设含基础字段（id/statement/prediction/expected_result/falsifiable）；
#    experiment_design/supporting_pmids/contradicting_pmids 由后续子组件填充
#
# 对应 spec.md Part 5 Hypothesis Generator（第 359-365 行）：
# - 振荡特征 → "X 的振荡周期由 Y 反馈环决定，抑制 Y 将消除振荡"
# - 双稳态特征 → "X 的 ON/OFF 切换由 Z 阈值决定，Z 敲除将消除切换"
# - 灵敏度特征 → "参数 k1 对输出敏感，药物抑制 k1 将显著降低输出"

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# 假设策略枚举（spec.md 第 361-364 行）
# =============================================================================
class HypothesisStrategy:
    """假设生成策略枚举（对应 spec.md 第 361-364 行三种特征）。"""

    OSCILLATION = "oscillation"   # 振荡特征 → 反馈环假设
    BISTABILITY = "bistability"   # 双稳态特征 → 阈值假设
    SENSITIVITY = "sensitivity"   # 灵敏度特征 → 参数假设


# =============================================================================
# HypothesisGenerator 主类
# =============================================================================
class HypothesisGenerator:
    """从仿真特征生成候选假设。

    主入口 generate(metrics, feature_metadata, ...) 遍历三种特征策略：
    1. 振荡特征：从 metrics.species[sp].oscillation 检测振荡行为
       → 生成"X 的振荡周期由 Y 反馈环决定"假设
    2. 双稳态特征：从 metrics.species[sp].bistability 检测双稳态
       → 生成"X 的 ON/OFF 切换由 Z 阈值决定"假设
    3. 灵敏度特征：从 v4_sensitivity_report.local_sensitivity 提取高灵敏度参数
       → 生成"参数 k1 对输出敏感，药物抑制 k1 将显著降低输出"假设

    每个候选假设含基础字段（未过滤，falsifiable 默认 True）：
    - id: "H001" / "H002" 等
    - statement: 假设陈述
    - prediction: 可证伪预测
    - expected_result: 预期结果
    - validation_method: 验证方法
    - falsifiable: True（待 Falsification Checker 复核）
    - strategy: 来源策略（oscillation / bistability / sensitivity）

    用法：
        generator = HypothesisGenerator()
        candidates = generator.generate(metrics, feature_metadata,
                                         v4_sensitivity_report=...)
    """

    def __init__(self) -> None:
        # 候选 ID 计数器（每次 generate 重置）
        self._id_counter = 0

    # =========================================================================
    # 主入口：generate
    # =========================================================================
    def generate(
        self,
        metrics: dict[str, Any],
        feature_metadata: dict[str, Any] | None = None,
        v4_sensitivity_report: dict[str, Any] | None = None,
        pathway_class: str = "",
    ) -> list[dict[str, Any]]:
        """主入口：从仿真特征生成候选假设。

        Args:
            metrics: 仿真结果指标 dict（n8_scientific_features 输出），
                结构：{species: {sp_name: {peak, peak_time, oscillation, bistability, ...}}, ...}
            feature_metadata: 特征元数据 dict（含 method / confidence / warnings）
            v4_sensitivity_report: P5 Sensitivity 报告 dict（含 local_sensitivity），
                用于灵敏度特征假设生成
            pathway_class: 通路类别字符串（用于假设陈述上下文，如 "EGFR_RTK"）

        Returns:
            候选假设列表（每项为 dict），未过滤。失败时返回空列表。
        """
        try:
            if not isinstance(metrics, dict):
                return []

            self._id_counter = 0  # 重置 ID 计数器
            candidates: list[dict[str, Any]] = []

            # 1. 振荡特征 → 反馈环假设
            candidates.extend(
                self._generate_oscillation_hypotheses(metrics, pathway_class)
            )

            # 2. 双稳态特征 → 阈值假设
            candidates.extend(
                self._generate_bistability_hypotheses(metrics, pathway_class)
            )

            # 3. 灵敏度特征 → 参数假设
            candidates.extend(
                self._generate_sensitivity_hypotheses(
                    metrics, v4_sensitivity_report, pathway_class
                )
            )

            return candidates
        except Exception as exc:
            logger.warning("HypothesisGenerator.generate 失败，返回空列表: %s", exc)
            return []

    # =========================================================================
    # SubTask 6.1.2: 振荡特征 → 反馈环假设
    # =========================================================================
    def _generate_oscillation_hypotheses(
        self, metrics: dict[str, Any], pathway_class: str
    ) -> list[dict[str, Any]]:
        """振荡特征 → 反馈环假设。

        spec.md 第 362 行：
        - 振荡特征 → "X 的振荡周期由 Y 反馈环决定，抑制 Y 将消除振荡"

        检测规则：
        - 遍历 metrics.species，找到 oscillation.oscillatory=True 的物种
        - 为每个振荡物种生成 1 条反馈环假设
        - Y（反馈环节点）从 pathway_class 推断（默认 NF-κB → IKK；p53 → MDM2）

        Args:
            metrics: 仿真结果指标 dict
            pathway_class: 通路类别（用于推断反馈环节点 Y）

        Returns:
            振荡假设列表
        """
        hypotheses: list[dict[str, Any]] = []
        species_metrics = self._extract_species_metrics(metrics)
        if not species_metrics:
            return hypotheses

        for sp_name, sp_data in species_metrics.items():
            oscillation = self._extract_oscillation(sp_data)
            if not oscillation or not oscillation.get("oscillatory", False):
                continue

            # 推断反馈环节点 Y（基于 pathway_class）
            feedback_node = self._infer_feedback_node(pathway_class, sp_name)
            period = oscillation.get("period_minutes") or oscillation.get("period_hours")
            period_unit = "min" if oscillation.get("period_minutes") else "h"

            hyp_id = self._next_id()
            statement = (
                f"{sp_name} 的振荡周期由 {feedback_node} 反馈环决定，"
                f"抑制 {feedback_node} 将消除振荡"
            )
            prediction = (
                f"敲低 {feedback_node} 后，{sp_name} 的振荡振幅下降 >50%"
            )
            expected_result = (
                f"{sp_name} 振荡振幅下降 >50%，振荡周期消失（原周期 {period} {period_unit}）"
            )

            hypotheses.append(self._build_candidate(
                hyp_id=hyp_id,
                statement=statement,
                prediction=prediction,
                expected_result=expected_result,
                validation_method="Western blot time-course + siRNA knockdown",
                strategy=HypothesisStrategy.OSCILLATION,
                target_species=sp_name,
                feedback_node=feedback_node,
                pathway_class=pathway_class,
            ))

        return hypotheses

    # =========================================================================
    # SubTask 6.1.2: 双稳态特征 → 阈值假设
    # =========================================================================
    def _generate_bistability_hypotheses(
        self, metrics: dict[str, Any], pathway_class: str
    ) -> list[dict[str, Any]]:
        """双稳态特征 → 阈值假设。

        spec.md 第 363 行：
        - 双稳态特征 → "X 的 ON/OFF 切换由 Z 阈值决定，Z 敲除将消除切换"

        检测规则：
        - 遍历 metrics.species，找到 bistability.bistable=True 的物种
        - 为每个双稳态物种生成 1 条阈值假设
        - Z（阈值节点）从 pathway_class 推断（默认 Wnt → Axin；Apoptosis → Bax）

        Args:
            metrics: 仿真结果指标 dict
            pathway_class: 通路类别（用于推断阈值节点 Z）

        Returns:
            双稳态假设列表
        """
        hypotheses: list[dict[str, Any]] = []
        species_metrics = self._extract_species_metrics(metrics)
        if not species_metrics:
            return hypotheses

        for sp_name, sp_data in species_metrics.items():
            bistability = self._extract_bistability(sp_data)
            if not bistability or not bistability.get("bistable", False):
                continue

            # 推断阈值节点 Z（基于 pathway_class）
            threshold_node = self._infer_threshold_node(pathway_class, sp_name)

            hyp_id = self._next_id()
            statement = (
                f"{sp_name} 的 ON/OFF 切换由 {threshold_node} 阈值决定，"
                f"{threshold_node} 敲除将消除切换"
            )
            prediction = (
                f"{threshold_node} 基因敲除后，{sp_name} 失去双稳态切换能力"
            )
            expected_result = (
                f"{sp_name} 在刺激范围内无法实现 ON/OFF 切换，"
                f"响应变为单调递增"
            )

            hypotheses.append(self._build_candidate(
                hyp_id=hyp_id,
                statement=statement,
                prediction=prediction,
                expected_result=expected_result,
                validation_method="Dose-response curve + CRISPR knockout",
                strategy=HypothesisStrategy.BISTABILITY,
                target_species=sp_name,
                threshold_node=threshold_node,
                pathway_class=pathway_class,
            ))

        return hypotheses

    # =========================================================================
    # SubTask 6.1.2: 灵敏度特征 → 参数假设
    # =========================================================================
    def _generate_sensitivity_hypotheses(
        self,
        metrics: dict[str, Any],
        v4_sensitivity_report: dict[str, Any] | None,
        pathway_class: str,
    ) -> list[dict[str, Any]]:
        """灵敏度特征 → 参数假设。

        spec.md 第 364 行：
        - 灵敏度特征 → "参数 k1 对输出敏感，药物抑制 k1 将显著降低输出"

        检测规则：
        - 从 v4_sensitivity_report.local_sensitivity 提取 top-K 高灵敏度参数
        - 为每个高灵敏度参数生成 1 条参数假设
        - 输出物种从 metrics.species 中选 fold_change 最大的

        Args:
            metrics: 仿真结果指标 dict
            v4_sensitivity_report: P5 Sensitivity 报告 dict
            pathway_class: 通路类别

        Returns:
            参数灵敏度假设列表
        """
        hypotheses: list[dict[str, Any]] = []
        if not v4_sensitivity_report or not isinstance(v4_sensitivity_report, dict):
            return hypotheses

        local_sens = v4_sensitivity_report.get("local_sensitivity") or {}
        if not isinstance(local_sens, dict):
            return hypotheses

        # 提取 top-K 高灵敏度参数（按 |sensitivity| 排序）
        top_params = self._extract_top_sensitive_params(local_sens, top_k=3)
        if not top_params:
            return hypotheses

        # 选 fold_change 最大的物种作为输出 readout
        target_species = self._select_max_fold_species(metrics)

        for param_info in top_params:
            param_name = param_info.get("param", "")
            sensitivity = param_info.get("sensitivity", 0.0)
            if not param_name:
                continue

            hyp_id = self._next_id()
            readout = target_species or "下游输出"
            direction = "降低" if sensitivity > 0 else "升高"
            statement = (
                f"参数 {param_name} 对 {readout} 敏感"
                f"（灵敏度={sensitivity:.3f}），"
                f"药物抑制 {param_name} 将显著{direction} {readout}"
            )
            prediction = (
                f"{param_name} 抑制剂处理细胞后，{readout} 水平{direction} >30%"
            )
            expected_result = (
                f"{readout} {direction} >30%（{param_name} 灵敏度 |S|>0.1）"
            )

            hypotheses.append(self._build_candidate(
                hyp_id=hyp_id,
                statement=statement,
                prediction=prediction,
                expected_result=expected_result,
                validation_method="Dose-response inhibitor treatment + Western blot",
                strategy=HypothesisStrategy.SENSITIVITY,
                target_param=param_name,
                sensitivity=float(sensitivity),
                target_species=readout,
                pathway_class=pathway_class,
            ))

        return hypotheses

    # =========================================================================
    # 候选假设构造 + ID 生成
    # =========================================================================
    def _build_candidate(
        self,
        hyp_id: str,
        statement: str,
        prediction: str,
        expected_result: str,
        validation_method: str,
        strategy: str,
        **extra: Any,
    ) -> dict[str, Any]:
        """构造候选假设 dict（基础字段 + strategy 元数据）。

        基础字段（对应 spec.md 第 355 行 Hypothesis schema）：
        - id, statement, prediction, experiment_design (待 ExperimentPlanner 填充),
          validation_method, expected_result, falsifiable=True,
          supporting_pmids=[] (待文献检索填充), contradicting_pmids=[]

        额外字段（strategy 元数据，便于后续子组件消费）：
        - strategy: 来源策略
        - target_species / feedback_node / threshold_node / target_param / sensitivity
        """
        return {
            # spec.md 第 355 行基础字段
            "id": hyp_id,
            "hypothesis_id": hyp_id,  # alias for Level 5 compatibility
            "statement": statement,
            "prediction": prediction,
            "experiment_design": {},  # 待 Task 6.2 ExperimentPlanner 填充
            "validation_method": validation_method,
            "expected_result": expected_result,
            "falsifiable": True,  # 默认 True，待 Task 6.3 FalsificationChecker 复核
            "supporting_pmids": [],  # 待文献检索填充（Task 6.1.4）
            "contradicting_pmids": [],  # 待文献检索填充
            # Level 5 兼容字段（P5 已提交代码用 falsifying_pmids）
            "falsifying_pmids": [],
            # strategy 元数据
            "strategy": strategy,
            **extra,
        }

    def _next_id(self) -> str:
        """生成下一个假设 ID（H001, H002, ...）。"""
        self._id_counter += 1
        return f"H{self._id_counter:03d}"

    # =========================================================================
    # 辅助函数：提取 metrics 字段
    # =========================================================================
    @staticmethod
    def _extract_species_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        """从 metrics 提取 species 指标 dict。

        metrics 结构：{species: {sp_name: {...}}, overall: {...}, combo: {...}}
        """
        species = metrics.get("species")
        if isinstance(species, dict):
            return species
        return {}

    @staticmethod
    def _extract_oscillation(species_data: Any) -> dict[str, Any]:
        """从单个 species 指标提取 oscillation dict。

        兼容两种结构：
        - species_data.oscillation (dict)
        - species_data.get("oscillation_period_hours") 等扁平字段
        """
        if not isinstance(species_data, dict):
            return {}
        osc = species_data.get("oscillation")
        if isinstance(osc, dict):
            return osc
        # 兼容扁平字段：oscillation_period_hours / oscillatory
        if "oscillation_period_hours" in species_data or "oscillatory" in species_data:
            return {
                "oscillatory": species_data.get("oscillatory", False),
                "period_hours": species_data.get("oscillation_period_hours"),
                "period_minutes": species_data.get("oscillation_period_minutes"),
            }
        return {}

    @staticmethod
    def _extract_bistability(species_data: Any) -> dict[str, Any]:
        """从单个 species 指标提取 bistability dict。"""
        if not isinstance(species_data, dict):
            return {}
        bis = species_data.get("bistability")
        if isinstance(bis, dict):
            return bis
        # 兼容扁平字段
        if "bistable" in species_data:
            return {
                "bistable": species_data.get("bistable", False),
                "threshold": species_data.get("bistability_threshold"),
            }
        return {}

    @staticmethod
    def _extract_top_sensitive_params(
        local_sensitivity: dict[str, Any], top_k: int = 3
    ) -> list[dict[str, Any]]:
        """从 local_sensitivity 提取 top-K 高灵敏度参数。

        local_sensitivity 结构（P5 Sensitivity 输出）：
        - {param_name: sensitivity_value, ...}  (扁平 dict)
        - 或 {params: [{param, sensitivity, ...}, ...]}  (结构化)

        Returns:
            [{param: str, sensitivity: float}, ...] 按 |sensitivity| 降序
        """
        candidates: list[dict[str, Any]] = []

        # 兼容结构化形式
        if "params" in local_sensitivity and isinstance(local_sensitivity["params"], list):
            for item in local_sensitivity["params"]:
                if not isinstance(item, dict):
                    continue
                param = item.get("param") or item.get("name") or ""
                sens = item.get("sensitivity") or item.get("value") or 0.0
                if param:
                    try:
                        candidates.append({"param": param, "sensitivity": float(sens)})
                    except (TypeError, ValueError):
                        continue
        else:
            # 扁平 dict 形式：{param_name: sensitivity_value}
            for param, sens in local_sensitivity.items():
                if not isinstance(param, str):
                    continue
                try:
                    candidates.append({"param": param, "sensitivity": float(sens)})
                except (TypeError, ValueError):
                    continue

        # 按 |sensitivity| 降序排序，取 top-K
        candidates.sort(key=lambda x: abs(x.get("sensitivity", 0.0)), reverse=True)
        return candidates[:top_k]

    @staticmethod
    def _select_max_fold_species(metrics: dict[str, Any]) -> str:
        """从 metrics.species 选 fold_change 最大的物种名。

        用于灵敏度假设的 readout 物种选择。
        """
        species_metrics = metrics.get("species") if isinstance(metrics, dict) else None
        if not isinstance(species_metrics, dict):
            return ""

        max_fold = -1.0
        max_species = ""
        for sp_name, sp_data in species_metrics.items():
            if not isinstance(sp_data, dict):
                continue
            fold = sp_data.get("fold_change")
            try:
                fold_val = abs(float(fold)) if fold is not None else 0.0
                if fold_val > max_fold:
                    max_fold = fold_val
                    max_species = sp_name
            except (TypeError, ValueError):
                continue
        return max_species

    # =========================================================================
    # 辅助函数：推断反馈环/阈值节点（基于 pathway_class）
    # =========================================================================
    @staticmethod
    def _infer_feedback_node(pathway_class: str, sp_name: str) -> str:
        """推断振荡反馈环节点 Y（基于 pathway_class）。

        spec.md 第 362 行：振荡特征 → "X 的振荡周期由 Y 反馈环决定"

        通路特异映射：
        - NF_KB → IKK（NF-κB/IKK 负反馈环）
        - p53 → MDM2（p53/MDM2 负反馈环）
        - WNT → Axin（β-catenin/Axin 负反馈环）
        - 默认 → "上游激酶"
        """
        mapping = {
            "NF_KB": "IKK",
            "p53": "MDM2",
            "WNT": "Axin",
            "JAK_STAT": "SOCS",
            "EGFR_RTK": "SOS",  # EGFR/MAPK 反馈
        }
        # 支持 MULTI: 前缀（取第一个匹配）
        for key, value in mapping.items():
            if key in (pathway_class or ""):
                return value
        return "上游激酶"

    @staticmethod
    def _infer_threshold_node(pathway_class: str, sp_name: str) -> str:
        """推断双稳态阈值节点 Z（基于 pathway_class）。

        spec.md 第 363 行：双稳态特征 → "X 的 ON/OFF 切换由 Z 阈值决定"

        通路特异映射：
        - WNT → Axin（β-catenin 降解复合物阈值）
        - Apoptosis → Bax（线粒体外膜通透性阈值）
        - Cell_Cycle → Rb（E2F/Rb 切换阈值）
        - 默认 → "上游调控因子"
        """
        mapping = {
            "WNT": "Axin",
            "Apoptosis": "Bax",
            "Cell_Cycle": "Rb",
            "p53": "MDM2",
        }
        for key, value in mapping.items():
            if key in (pathway_class or ""):
                return value
        return "上游调控因子"


__all__ = ["HypothesisGenerator", "HypothesisStrategy"]
