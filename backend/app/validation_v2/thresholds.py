# BioDynamics Agent v4 - Pathway-Specific Thresholds (Phase 5 / Task 5.3.6)
#
# 通路特异阈值：Level 2 SBML Validation 在判断 pass/fail 时使用。
#
# 设计原则（铁律）：
# 1. 不同通路对仿真误差的容忍度不同（生物变异 / 振荡相位 / 脉冲幅度等）
# 2. NF-κB 容忍更大 peak_time_diff（振荡相位难对齐）
# 3. p53 容忍更大 amplification_diff（脉冲幅度生物变异大）
# 4. 未识别通路使用 default 阈值
#
# 对应 spec.md Part 4 Level 2（第 291 行）：
#   NF-κB 容忍更大 peak_time_diff；p53 容忍更大 amplification_diff
#
# 依赖：无（纯数据 + 查找）

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# 通路特异阈值常量（spec.md 第 291 行 + 实施要求）
# =============================================================================
# 阈值含义：
# - peak_time_diff: 达峰时间允许偏差（分钟，绝对差）
# - amplification_diff: 振幅允许偏差（相对差，0~1 表示 0~100%）
#
# 来源：spec.md Part 4 Level 2 + Level 4 benchmark 经验值
_THRESHOLD_TABLE: dict[str, dict[str, float]] = {
    # EGFR 通路：受体激活快速（5-10 min 达峰），抖动小
    "EGFR_RTK": {
        "peak_time_diff": 2.0,        # 2 分钟
        "amplification_diff": 0.20,   # 20%
    },
    "EGFR": {
        "peak_time_diff": 2.0,
        "amplification_diff": 0.20,
    },
    # MAPK 通路：级联放大，振幅变化较大
    "MAPK": {
        "peak_time_diff": 2.0,
        "amplification_diff": 0.30,   # 30%
    },
    # NF-κB 通路：振荡相位难对齐，peak_time_diff 容忍 30 分钟
    "NF_KB": {
        "peak_time_diff": 30.0,       # 30 分钟（spec.md 第 291 行）
        "amplification_diff": 0.50,   # 50%
    },
    "NF_KAPPAB": {
        "peak_time_diff": 30.0,
        "amplification_diff": 0.50,
    },
    # p53 通路：脉冲幅度生物变异大，amplification_diff 容忍 100%
    "P53": {
        "peak_time_diff": 30.0,
        "amplification_diff": 1.00,   # 100%（spec.md 第 291 行）
    },
    # Wnt 通路：β-catenin 稳态慢动力学，peak_time_diff 容忍 60 分钟
    "WNT": {
        "peak_time_diff": 60.0,
        "amplification_diff": 0.50,
    },
    # PI3K-AKT-mTOR 通路
    "PI3K_AKT_mTOR": {
        "peak_time_diff": 5.0,
        "amplification_diff": 0.30,
    },
    # 凋亡通路
    "APOPTOSIS": {
        "peak_time_diff": 30.0,
        "amplification_diff": 0.50,
    },
    # 细胞周期
    "CELL_CYCLE": {
        "peak_time_diff": 60.0,
        "amplification_diff": 0.50,
    },
    # JAK-STAT
    "JAK_STAT": {
        "peak_time_diff": 5.0,
        "amplification_diff": 0.30,
    },
    # TGF-β
    "TGF_BETA": {
        "peak_time_diff": 30.0,
        "amplification_diff": 0.50,
    },
    # 多通路混合（取最宽松阈值）
    "MULTI": {
        "peak_time_diff": 30.0,
        "amplification_diff": 0.50,
    },
    # default
    "default": {
        "peak_time_diff": 5.0,        # 5 分钟
        "amplification_diff": 0.30,   # 30%
    },
}


# =============================================================================
# PathwayThresholds 主类
# =============================================================================
class PathwayThresholds:
    """通路特异阈值查找器。

    根据 v4_pathway_class 返回对应的 peak_time_diff / amplification_diff 阈值。
    未识别通路返回 default 阈值。

    用法：
        thresholds = PathwayThresholds()
        t = thresholds.get_thresholds("NF_KB")
        # t = {"peak_time_diff": 30.0, "amplification_diff": 0.50}
    """

    # 默认阈值（对应 spec.md default: peak_time_diff=5.0min, amplification_diff=30%）
    DEFAULT_PEAK_TIME_DIFF: float = 5.0
    DEFAULT_AMPLIFICATION_DIFF: float = 0.30

    def __init__(self, custom_table: dict[str, dict[str, float]] | None = None) -> None:
        """初始化。

        Args:
            custom_table: 可选，自定义通路阈值表（用于测试注入）。
                若提供则覆盖默认 _THRESHOLD_TABLE。
        """
        self._table = custom_table if custom_table is not None else _THRESHOLD_TABLE

    def get_thresholds(self, pathway_class: str | None) -> dict[str, float]:
        """返回通路特异阈值。

        Args:
            pathway_class: v4_pathway_class 字符串（如 "EGFR_RTK" / "NF_KB" / "P53"）。
                None / 空字符串 / 未识别通路返回 default 阈值。

        Returns:
            {peak_time_diff: float, amplification_diff: float}
            - peak_time_diff: 达峰时间允许偏差（分钟，绝对差）
            - amplification_diff: 振幅允许偏差（相对差，0~1）
        """
        if not pathway_class or not isinstance(pathway_class, str):
            return self._fallback_thresholds()

        # 直接匹配
        key = pathway_class.strip().upper()
        if key in self._table:
            entry = self._table[key]
            return self._normalize_entry(entry)

        # 模糊匹配优先级 1：prefix match（"MULTI:EGFR_RTK+..." → MULTI）
        # 这一步保证多通路场景返回 MULTI 的最宽松阈值，而非被首个子通路抢占
        for known_key, entry in self._table.items():
            if known_key == "default":
                continue
            if key.startswith(known_key + ":") or key.startswith(known_key + "+"):
                return self._normalize_entry(entry)

        # 模糊匹配优先级 2：substring match（"FOO_EGFR_RTK_BAR" → EGFR_RTK）
        for known_key, entry in self._table.items():
            if known_key == "default":
                continue
            if known_key in key:
                return self._normalize_entry(entry)

        # 未识别 → default
        logger.debug(
            "未识别 pathway_class=%s，使用 default 阈值", pathway_class
        )
        return self._fallback_thresholds()

    def _fallback_thresholds(self) -> dict[str, float]:
        """返回 default 阈值。"""
        default_entry = self._table.get("default")
        if default_entry:
            return self._normalize_entry(default_entry)
        return {
            "peak_time_diff": self.DEFAULT_PEAK_TIME_DIFF,
            "amplification_diff": self.DEFAULT_AMPLIFICATION_DIFF,
        }

    @staticmethod
    def _normalize_entry(entry: dict[str, Any]) -> dict[str, float]:
        """规范化阈值条目，确保返回 dict[str, float]。"""
        return {
            "peak_time_diff": float(entry.get("peak_time_diff", 5.0)),
            "amplification_diff": float(entry.get("amplification_diff", 0.30)),
        }


__all__ = ["PathwayThresholds"]
