# BioDynamics Agent v4 - Falsification Checker（Phase 6 / Task 6.3）
#
# FalsificationChecker 子组件：检查假设的可证伪性（Karl Popper 标准）。
#
# 职责（spec.md Part 5 第 372-379 行）：
# - 检查每个假设的可证伪性
# - 输出 falsifiable: bool + falsification_criteria: str
# - 不可证伪的假设被过滤掉（spec.md 第 379 行）
#
# 可证伪性规则（spec.md 第 374-377 行）：
# 1. 每个假设必须有明确的可证伪预测（如"抑制 X 后 Y 下降 >50%"）
# 2. 必须有对照组（vehicle / scramble siRNA）
# 3. 必须有定量阈值（不能是"会变化"）
#
# 设计原则（铁律）：
# 1. 不修改 v3 任何字段；仅消费 hypothesis dict + experiment_design
# 2. 失败降级：任何异常返回 falsifiable=True（保守不阻塞，避免误杀）
# 3. 不调用 LLM；不调用 RAG；纯规则匹配
# 4. 不修改假设 statement / prediction 字段（只读 + 标注）
# 5. 过滤策略：check() 单个假设返回结果；filter() 批量过滤（由 Agent 调用）
#
# 对应 spec.md Part 5 Falsification Checker（第 372-379 行）

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# 可证伪性规则常量
# =============================================================================

# 规则 1：必须有可证伪预测（含明确方向 + 定量阈值）
# 合法形式：含 "下降" / "升高" / "消除" / "消失" / "失去" / "出现" + 数字 + % 或倍数
# 非法形式："会变化" / "可能改变" / "有所反应" 等模糊表述
_FALSIFIABLE_PREDICTION_PATTERNS: list[re.Pattern[str]] = [
    # 含百分比阈值：">50%" / "降低 30%" / "下降 >2 倍"
    re.compile(r"(?:下降|降低|升高|增加|消除|消失|失去|出现|减少|增大)"
               r"[^0-9%]*>?\s*\d+(?:\.\d+)?\s*(?:%|倍|fold)", re.IGNORECASE),
    # 含倍数变化：">2 倍" / "<3 fold"
    re.compile(r">?\s*\d+(?:\.\d+)?\s*(?:倍|fold)", re.IGNORECASE),
    # 含明确状态变化：消除振荡 / 失去双稳态
    re.compile(r"消除\s*\w+\s*(?:振荡|切换|响应|信号)", re.IGNORECASE),
    re.compile(r"失去\s*\w+\s*(?:双稳态|切换能力|振荡)", re.IGNORECASE),
]

# 模糊预测关键词（命中即视为不可证伪）
_VAGUE_PREDICTION_KEYWORDS: list[str] = [
    "会变化",
    "可能改变",
    "有所反应",
    "出现一些",
    "可能影响",
    "有所变化",
    "发生改变",
    "产生反应",
    "存在差异",
]

# 规则 2：对照组必须存在
# experiment_design.controls 必须为 list 且非空，且含至少一个标准对照（vehicle / untreated / DMSO / scramble）
_REQUIRED_CONTROL_KEYWORDS: list[str] = [
    "vehicle",
    "untreated",
    "dmso",
    "scramble",
]

# 规则 3：定量阈值必须存在
# readout.threshold 必须为 > 0 的数值
# 或 prediction/experiment_design.expected_result 含 % / 倍 / fold 字样
_QUANTITATIVE_THRESHOLD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\d+(?:\.\d+)?\s*%", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?\s*(?:倍|fold)", re.IGNORECASE),
    re.compile(r">\s*\d+(?:\.\d+)?", re.IGNORECASE),
]


# =============================================================================
# FalsificationChecker 主类
# =============================================================================
class FalsificationChecker:
    """检查假设的可证伪性（spec.md Part 5 第 372-379 行）。

    主入口 check(hypothesis) -> dict 输出：
    - falsifiable: bool（是否可证伪）
    - falsification_criteria: str（可证伪标准描述）
    - failure_reasons: list[str]（不可证伪时给出原因，可证伪时为空）

    可证伪性三规则（spec.md 第 374-377 行）：
    1. 必须有明确的可证伪预测（含方向 + 定量阈值）
    2. 必须有对照组（vehicle / scramble siRNA）
    3. 必须有定量阈值（不能是"会变化"）

    用法：
        checker = FalsificationChecker()
        result = checker.check(hypothesis)
        # result = {falsifiable: bool, falsification_criteria: str, failure_reasons: list}
    """

    def __init__(self) -> None:
        """初始化。"""
        # 编译模式列表
        self._falsifiable_patterns: list[re.Pattern[str]] = list(
            _FALSIFIABLE_PREDICTION_PATTERNS
        )
        self._vague_keywords: list[str] = list(_VAGUE_PREDICTION_KEYWORDS)
        self._required_control_keywords: list[str] = list(
            _REQUIRED_CONTROL_KEYWORDS
        )
        self._quantitative_patterns: list[re.Pattern[str]] = list(
            _QUANTITATIVE_THRESHOLD_PATTERNS
        )

    # =========================================================================
    # 主入口：check
    # =========================================================================
    def check(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """检查单个假设的可证伪性。

        Args:
            hypothesis: 假设 dict，含 prediction / experiment_design /
                expected_result / statement 字段

        Returns:
            dict 含 3 字段：
            - falsifiable: bool
            - falsification_criteria: str
            - failure_reasons: list[str]
        """
        try:
            if not isinstance(hypothesis, dict):
                return self._failure_result(
                    ["假设对象无效（非 dict）"],
                    "无法评估（假设对象无效）",
                )

            failure_reasons: list[str] = []

            # 规则 1：可证伪预测
            rule1_pass, rule1_reason = self._check_falsifiable_prediction(hypothesis)
            if not rule1_pass:
                failure_reasons.append(rule1_reason)

            # 规则 2：对照组
            rule2_pass, rule2_reason = self._check_controls(hypothesis)
            if not rule2_pass:
                failure_reasons.append(rule2_reason)

            # 规则 3：定量阈值
            rule3_pass, rule3_reason = self._check_quantitative_threshold(hypothesis)
            if not rule3_pass:
                failure_reasons.append(rule3_reason)

            if failure_reasons:
                return self._failure_result(
                    failure_reasons,
                    self._build_criteria_from_failure(failure_reasons),
                )

            # 三规则全部通过 → 可证伪
            criteria = self._build_falsification_criteria(hypothesis)
            return {
                "falsifiable": True,
                "falsification_criteria": criteria,
                "failure_reasons": [],
            }
        except Exception as exc:
            # 铁律 #2：失败降级返回 falsifiable=True（保守不阻塞）
            logger.warning(
                "FalsificationChecker.check 失败，降级返回 falsifiable=True: %s",
                exc,
            )
            return {
                "falsifiable": True,
                "falsification_criteria": "降级评估（异常）：默认可证伪",
                "failure_reasons": [],
            }

    # =========================================================================
    # 批量过滤：filter（由 Agent 调用，过滤不可证伪假设）
    # =========================================================================
    def filter(
        self, hypotheses: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """批量过滤不可证伪的假设（spec.md 第 379 行）。

        不可证伪的假设被过滤掉。对每个假设：
        1. 调用 check() 评估可证伪性
        2. falsifiable=True → 保留，更新 falsifiable + falsification_criteria 字段
        3. falsifiable=False → 丢弃，记录 logger.info

        Args:
            hypotheses: 假设列表

        Returns:
            过滤后的可证伪假设列表
        """
        if not isinstance(hypotheses, list):
            return []

        filtered: list[dict[str, Any]] = []
        for hyp in hypotheses:
            if not isinstance(hyp, dict):
                continue
            result = self.check(hyp)
            if result.get("falsifiable", False):
                # 标注可证伪性字段 + 标准描述
                hyp["falsifiable"] = True
                hyp["falsification_criteria"] = result.get(
                    "falsification_criteria", ""
                )
                filtered.append(hyp)
            else:
                logger.info(
                    "FalsificationChecker: 假设 %s 不可证伪，已过滤（原因: %s）",
                    hyp.get("id", "?"),
                    "; ".join(result.get("failure_reasons", [])),
                )
        return filtered

    # =========================================================================
    # 规则 1：可证伪预测检查
    # =========================================================================
    def _check_falsifiable_prediction(
        self, hypothesis: dict[str, Any]
    ) -> tuple[bool, str]:
        """规则 1：必须有明确的可证伪预测（spec.md 第 375 行）。

        检查 prediction 文本：
        - 必须含方向词（下降/升高/消除/失去/出现 等）
        - 必须含定量阈值（百分比 / 倍数 / fold）
        - 不应含模糊表述（会变化 / 可能改变 等）

        Returns:
            (True, "") 表示通过
            (False, reason) 表示失败 + 原因
        """
        prediction = hypothesis.get("prediction") or ""
        if not isinstance(prediction, str) or not prediction.strip():
            return False, "规则 1 失败：prediction 缺失或为空"

        prediction_lower = prediction.lower()

        # 检查模糊表述
        for vague in self._vague_keywords:
            if vague in prediction_lower:
                return (
                    False,
                    f"规则 1 失败：prediction 含模糊表述 '{vague}'，"
                    f"应为明确方向 + 定量阈值（如'下降 >50%'）",
                )

        # 检查是否匹配可证伪模式
        for pattern in self._falsifiable_patterns:
            if pattern.search(prediction):
                return True, ""

        # 未匹配任何可证伪模式
        return (
            False,
            "规则 1 失败：prediction 缺乏明确的可证伪预测"
            "（需含方向词 + 定量阈值，如'抑制 X 后 Y 下降 >50%'）",
        )

    # =========================================================================
    # 规则 2：对照组检查
    # =========================================================================
    def _check_controls(
        self, hypothesis: dict[str, Any]
    ) -> tuple[bool, str]:
        """规则 2：必须有对照组（spec.md 第 376 行）。

        检查 experiment_design.controls：
        - 必须为 list 且非空
        - 必须含至少一个标准对照（vehicle / untreated / DMSO / scramble siRNA）

        Returns:
            (True, "") 表示通过
            (False, reason) 表示失败 + 原因
        """
        experiment_design = hypothesis.get("experiment_design")
        if not isinstance(experiment_design, dict):
            return (
                False,
                "规则 2 失败：experiment_design 缺失（非 dict），无法评估对照组",
            )

        controls = experiment_design.get("controls")
        if not isinstance(controls, list) or not controls:
            return (
                False,
                "规则 2 失败：experiment_design.controls 缺失或为空",
            )

        # 检查是否含至少一个标准对照
        controls_lower = " ".join(str(c).lower() for c in controls)
        has_standard_control = any(
            kw in controls_lower for kw in self._required_control_keywords
        )
        if not has_standard_control:
            return (
                False,
                f"规则 2 失败：controls 未含标准对照"
                f"（vehicle / untreated / DMSO / scramble siRNA），"
                f"当前 controls={controls}",
            )

        return True, ""

    # =========================================================================
    # 规则 3：定量阈值检查
    # =========================================================================
    def _check_quantitative_threshold(
        self, hypothesis: dict[str, Any]
    ) -> tuple[bool, str]:
        """规则 3：必须有定量阈值（spec.md 第 377 行）。

        检查途径（任一通过即视为有定量阈值）：
        1. experiment_design.readout.threshold 为 > 0 的数值
        2. prediction 含 % / 倍 / fold 字样
        3. expected_result 含 % / 倍 / fold 字样

        Returns:
            (True, "") 表示通过
            (False, reason) 表示失败 + 原因
        """
        # 途径 1：readout.threshold
        experiment_design = hypothesis.get("experiment_design")
        if isinstance(experiment_design, dict):
            readout = experiment_design.get("readout")
            if isinstance(readout, dict):
                threshold = readout.get("threshold")
                if isinstance(threshold, (int, float)) and threshold > 0:
                    return True, ""

        # 途径 2：prediction 含 % / 倍 / fold
        prediction = hypothesis.get("prediction") or ""
        if isinstance(prediction, str):
            for pattern in self._quantitative_patterns:
                if pattern.search(prediction):
                    return True, ""

        # 途径 3：expected_result 含 % / 倍 / fold
        expected_result = hypothesis.get("expected_result") or ""
        if isinstance(expected_result, str):
            for pattern in self._quantitative_patterns:
                if pattern.search(expected_result):
                    return True, ""

        return (
            False,
            "规则 3 失败：未找到定量阈值"
            "（readout.threshold > 0 / prediction 含 %/倍/fold / "
            "expected_result 含 %/倍/fold，任一即可）",
        )

    # =========================================================================
    # 辅助函数：构造可证伪标准描述
    # =========================================================================
    def _build_falsification_criteria(
        self, hypothesis: dict[str, Any]
    ) -> str:
        """根据假设构造可证伪标准描述（falsifiable=True 时使用）。

        模板：
        "假设 X 可证伪：prediction；若实验观察结果与 prediction 方向相反或
        未达到阈值（threshold），则假设被证伪。"
        """
        prediction = hypothesis.get("prediction") or "未指定预测"

        # 提取阈值描述
        threshold_desc = ""
        experiment_design = hypothesis.get("experiment_design")
        if isinstance(experiment_design, dict):
            readout = experiment_design.get("readout")
            if isinstance(readout, dict):
                threshold = readout.get("threshold")
                metric = readout.get("metric", "")
                species = readout.get("species", "")
                if isinstance(threshold, (int, float)) and threshold > 0:
                    threshold_pct = threshold * 100 if threshold <= 1.0 else threshold
                    threshold_desc = (
                        f"（{species} {metric} 变化阈值 ≥ {threshold_pct:.1f}%）"
                    )

        criteria = (
            f"假设可证伪：prediction='{prediction}'{threshold_desc}；"
            f"若实验组与对照组相比未达到该阈值或方向相反，则假设被证伪。"
        )
        return criteria

    def _build_criteria_from_failure(
        self, failure_reasons: list[str]
    ) -> str:
        """根据失败原因构造不可证伪的描述（falsifiable=False 时使用）。"""
        return (
            "假设不可证伪：" + "; ".join(failure_reasons)
            + "。需补充明确的可证伪预测 + 对照组 + 定量阈值后重新评估。"
        )

    @staticmethod
    def _failure_result(
        failure_reasons: list[str], criteria: str
    ) -> dict[str, Any]:
        """构造 falsifiable=False 的结果 dict。"""
        return {
            "falsifiable": False,
            "falsification_criteria": criteria,
            "failure_reasons": failure_reasons,
        }


__all__ = ["FalsificationChecker"]
