# BioDynamics Agent v4 - Cross-talk Time Scale Aligner (Phase 4 / Task 4.13.6)
# 多通路时间尺度对齐：统一 max_step，避免慢通路被快通路"拖垮"。
#
# 设计原则（铁律）：
# 1. 不修改 v3 任何字段；不生成 ODE；不做 SBML 验证（职责边界严格）
# 2. 失败降级：任何异常返回空 dict，不阻塞主流水线
# 3. 多通路场景取所有通路 max_step 最小值（保守策略，保证数值稳定性）
# 4. 单通路场景直接返回该通路的 max_step
#
# 参考：
# - spec.md Part 3 Cross-talk Coordinator Agent（第 262-272 行）
# - tasks.md SubTask 4.13.6

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# 默认 max_step（分钟），与 PathwayGraph TemporalAnnotation 对齐
# =============================================================================
# FAST: 0.1 min（磷酸化/结合）
# MEDIUM: 1.0 min（转录/翻译/ubiquitination）
# SLOW: 10.0 min（细胞周期/降解）
_DEFAULT_MAX_STEP: dict[str, float] = {
    "fast": 0.1,
    "medium": 1.0,
    "slow": 10.0,
}

# 兜底默认 max_step（无法识别 time_scale 时使用）
_FALLBACK_MAX_STEP: float = 0.1


class TimeScaleAligner:
    """多通路时间尺度对齐器。

    协调多通路场景下的 ODE 求解器 max_step 配置：
    - 多通路：取所有通路 max_step 的最小值（保守策略）
    - 单通路：直接返回该通路的 max_step

    职责边界：
    - 不修改 Specialist 内部 Reaction
    - 不生成 ODE
    - 不做 SBML 验证
    """

    def align_time_scales(self, specialist_outputs: list[dict]) -> dict:
        """对齐多通路时间尺度。

        Args:
            specialist_outputs: 多通路 Specialist 输出列表，每条 dict 可含：
                - ``pathway_class``: 通路类别键（如 "EGFR_RTK"）
                - ``max_step``: 该通路推荐 max_step（分钟，float）
                - ``time_scale``: 时间尺度标签（"fast"/"medium"/"slow"）
                - ``t_end``: 仿真总时长（分钟，可选）

        Returns:
            dict 含：
            - ``unified_max_step``: 统一最大步长（取所有通路最小值）
            - ``pathway_time_scales``: 每通路时间尺度信息列表
            - ``alignment_strategy``: 对齐策略（"min_of_all" 或 "single_pathway"）
            异常时返回空 dict。
        """
        try:
            if not specialist_outputs:
                return {
                    "unified_max_step": _FALLBACK_MAX_STEP,
                    "pathway_time_scales": [],
                    "alignment_strategy": "min_of_all",
                }

            pathway_time_scales: list[dict[str, Any]] = []
            max_steps: list[float] = []

            for output in specialist_outputs:
                pathway_class = output.get("pathway_class", "UNKNOWN")
                max_step = output.get("max_step")
                time_scale = output.get("time_scale", "")
                t_end = output.get("t_end")

                # 若未显式提供 max_step，根据 time_scale 推断
                if max_step is None:
                    max_step = _DEFAULT_MAX_STEP.get(
                        time_scale.lower() if time_scale else "",
                        _FALLBACK_MAX_STEP,
                    )

                # 确保是 float
                try:
                    max_step = float(max_step)
                except (TypeError, ValueError):
                    max_step = _FALLBACK_MAX_STEP

                max_steps.append(max_step)
                pathway_time_scales.append({
                    "pathway_class": pathway_class,
                    "max_step": max_step,
                    "time_scale": time_scale,
                    "t_end": t_end,
                })

            # 单通路：直接返回该通路的 max_step
            if len(specialist_outputs) == 1:
                strategy = "single_pathway"
            else:
                strategy = "min_of_all"

            unified = min(max_steps) if max_steps else _FALLBACK_MAX_STEP

            return {
                "unified_max_step": unified,
                "pathway_time_scales": pathway_time_scales,
                "alignment_strategy": strategy,
            }
        except Exception as exc:
            logger.warning("TimeScaleAligner.align_time_scales 失败: %s", exc)
            return {}


__all__ = ["TimeScaleAligner"]
