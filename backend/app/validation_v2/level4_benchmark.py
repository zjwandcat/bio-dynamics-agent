# BioDynamics Agent v4 - Level 4 Benchmark Validation (Phase 5 / Task 5.5)
#
# Level4BenchmarkValidator 主类 + LangGraph hook 节点。
# 职责：通路特异 benchmark 验证（PubMed 检索的关键参数实验值对比）。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_VALIDATION_PYRAMID_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
# 3. 仅消费 P1/P2/P3 + P4 + 仿真 metrics（v4_ode_system / v4_pathway_class / metrics）
# 4. 失败降级：任何异常都返回 pass=False，但不抛异常
# 5. 输出写入 state["v4_validation_report"]["level4"]（新增 v4 字段）
# 6. 不修改 rag_client.py 代码，仅 import 调用其接口
#
# 对应 spec.md Part 4 Level 4（第 301-311 行）
# - 输入：仿真结果 + 通路特异 benchmark
# - 输出：v4_validation_report.level4: {pass, benchmarks: [{name, source_pmid,
#         expected, actual, diff, pass}]}
# - 失败策略：任一 benchmark 偏差 > 通路特异阈值 → pass=False
# - 回滚策略：V4_VALIDATION_PYRAMID_ENABLED=false，Level 4 跳过
#
# 依赖：
# - app.config.settings（Feature Flag）
# - app.pathways.pathway_planner.parse_pathway_class（解析 MULTI: 前缀）
# - app.rag_client.RagClient（仅 import 调用，不修改其代码）

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark 清单常量（spec.md Part 4 Level 4 第 304-309 行）
# =============================================================================
# TD-023 (IB-016) 修复：10 个通路特异 benchmark（原仅 5/10 通路覆盖，补齐缺失 5 个）。
# 每个含：
# - name: benchmark 名称
# - description: 描述
# - source_pmid: 来源 PubMed ID
# - metric: 仿真结果中对应的指标 key
# - expected_range: 期望范围 (min, max)，None 表示无下限/上限
# - tolerance: 允许偏差（超出 expected_range 但在 tolerance 内仍判 pass）
# - benchmark_source: "literature"（精确实验值）/ "estimated"（按文献综述范围估计）
BENCHMARK_REGISTRY: dict[str, dict[str, Any]] = {
    "EGFR_RTK": {
        "name": "pEGFR_peak_time",
        "description": "pEGFR 5-10 min 达峰（Schoeberl 2002）",
        "source_pmid": "PMID:12124381",  # Schoeberl 2002
        "metric": "peak_time_minutes",
        "expected_range": (5.0, 10.0),
        "tolerance": 2.0,  # 允许偏差 2 min
        "benchmark_source": "literature",
    },
    "MAPK_ERK": {
        "name": "MAPK_Hill_coefficient",
        "description": "零阶 ultrasensitivity Hill >2（Markevich 2004）",
        "source_pmid": "PMID:14757805",  # Markevich 2004
        "metric": "hill_coefficient",
        "expected_range": (2.0, None),  # > 2
        "tolerance": 0.5,
        "benchmark_source": "literature",
    },
    "NF_KB": {
        "name": "NF_kB_oscillation_period",
        "description": "振荡周期 1-2h（Nelson 2004）",
        "source_pmid": "PMID:14975635",  # Nelson 2004
        "metric": "oscillation_period_hours",
        "expected_range": (1.0, 2.0),
        "tolerance": 0.5,
        "benchmark_source": "literature",
    },
    "p53": {
        "name": "p53_pulse_period",
        "description": "脉冲周期 5-7h（Lev Bar-Or 2000）",
        "source_pmid": "PMID:10644694",  # Lev Bar-Or 2000
        "metric": "pulse_period_hours",
        "expected_range": (5.0, 7.0),
        "tolerance": 1.0,
        "benchmark_source": "literature",
    },
    "WNT": {
        "name": "beta_catenin_steady_state",
        "description": "β-catenin 稳态 <10 nM 无 Wnt（Lee 2003）",
        "source_pmid": "PMID:12906785",  # Lee 2003
        "metric": "steady_state_nM",
        "expected_range": (None, 10.0),  # < 10
        "tolerance": 2.0,
        "benchmark_source": "literature",
    },
    # TD-023 (IB-016) 修复：以下 5 个通路原无 benchmark 覆盖，现按文献综述范围补齐。
    # 阈值取自 backend/benchmarks/*.yaml 官方 benchmark 套件（含真实 PMID），
    # 因注册表为单指标摘要（YAML 含多指标），标记 benchmark_source="estimated"。
    "PI3K_AKT_mTOR": {
        "name": "pAKT_peak_time",
        "description": "pAKT 30-60 min 达峰（Mazzoletti 2009）",
        "source_pmid": "PMID:19211571",  # Mazzoletti 2009
        "metric": "pAKT_peak_time",
        "expected_range": (30.0, 60.0),
        "tolerance": 5.0,
        "benchmark_source": "estimated",  # 单指标摘要，按文献综述范围估计
    },
    "TGF_BETA": {
        "name": "pSmad2_peak_time",
        "description": "pSmad2 5-15 min 达峰（Clarke 2009 / Massagué 1998）",
        "source_pmid": "PMID:9674480",  # Massagué 1998
        "metric": "pSmad2_peak_time",
        "expected_range": (5.0, 15.0),
        "tolerance": 1.0,
        "benchmark_source": "estimated",
    },
    "APOPTOSIS": {
        "name": "Cyt_c_precedes_Casp3",
        "description": "Cyt c 释放早于 Caspase-3 激活 5-15 min（Rehm 2006 / Green & Kroemer 2004）",
        "source_pmid": "PMID:15241432",  # Green & Kroemer 2004
        "metric": "Cyt_c_precedes_Casp3",
        "expected_range": (5.0, 15.0),
        "tolerance": 1.0,
        "benchmark_source": "estimated",
    },
    "CELL_CYCLE": {
        "name": "CyclinB_APC_oscillation_period",
        "description": "CyclinB-APC/C 振荡周期 8-12h（Tyson 1991 / Pomerening 2005）",
        "source_pmid": "PMID:11389814",  # Pomerening 2005
        "metric": "CyclinB_APC_oscillation_period",
        "expected_range": (8.0, 12.0),
        "tolerance": 0.5,
        "benchmark_source": "estimated",
    },
    "JAK_STAT": {
        "name": "pSTAT5_peak_time",
        "description": "pSTAT5 5-15 min 达峰（Timm 2003 / Schwartz 2003）",
        "source_pmid": "PMID:15286703",  # Schwartz 2003
        "metric": "pSTAT5_peak_time",
        "expected_range": (5.0, 15.0),
        "tolerance": 1.0,
        "benchmark_source": "estimated",
    },
}


# =============================================================================
# Level4BenchmarkValidator 主类
# =============================================================================
class Level4BenchmarkValidator:
    """Level 4 Benchmark Validation 验证器。

    主入口 validate(state) 执行通路特异 benchmark 对比：
    1. 解析 v4_pathway_class（支持单通路 / MULTI: 多通路）
    2. 从 BENCHMARK_REGISTRY 匹配通路特异 benchmark
    3. 可选：通过 rag_client 检索 PubMed 文献补充 benchmark 参数
    4. 从仿真结果 metrics 中提取 actual 值
    5. 对比 expected_range + tolerance，判定每个 benchmark pass/fail
    6. 任一 benchmark 失败 → pass=False

    失败策略（对应 spec.md 第 310 行）：
    - 任一 benchmark 偏差 > 通路特异阈值 → pass=False

    用法：
        validator = Level4BenchmarkValidator()
        report = validator.validate(state)
        # report = {pass, benchmarks: [{name, source_pmid, expected, actual, diff, pass}]}
    """

    def __init__(
        self,
        rag_client: Any = None,
        benchmark_registry: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """初始化。

        Args:
            rag_client: 可选，RagClient 实例（用于文献检索，测试可注入 mock）。
                默认 None → 延迟创建，失败时降级到 BENCHMARK_REGISTRY 硬编码值。
            benchmark_registry: 可选，自定义 benchmark 注册表（测试可注入）。
                默认 None → 使用 BENCHMARK_REGISTRY 全局常量。
        """
        self._rag_client = rag_client
        self._benchmark_registry = (
            benchmark_registry if benchmark_registry is not None else BENCHMARK_REGISTRY
        )

    # =========================================================================
    # 主入口：validate
    # =========================================================================
    def validate(self, state: dict[str, Any]) -> dict[str, Any]:
        """主入口：执行 Level 4 Benchmark 验证。

        Args:
            state: LangGraph 全局状态，含：
                - v4_ode_system: ODE 系统
                - v4_pathway_class: 通路类别（如 "EGFR_RTK" / "MULTI:EGFR_RTK+PI3K_AKT_mTOR"）
                - metrics: 仿真结果指标 dict（含 peak_time_minutes / hill_coefficient 等）

        Returns:
            Level 4 报告 dict（对应 spec.md 第 303 行）：
            {
                pass: bool,
                benchmarks: [
                    {
                        name: str,             # benchmark 名称
                        source_pmid: str,      # 来源 PMID
                        expected: dict,        # 期望范围 + tolerance
                        actual: float | None,  # 仿真实际值（无数据时为 None）
                        diff: float | None,    # 偏差（无数据时为 None）
                        pass: bool             # 单个 benchmark 是否通过
                    },
                    ...
                ]
            }
            异常时返回 pass=False（阻塞流水线），不抛异常（铁律 #5）。
        """
        try:
            if not isinstance(state, dict):
                return self._fail_report("invalid_state_type")

            pathway_class = state.get("v4_pathway_class", "") or ""
            metrics = state.get("metrics", {}) or {}
            if not isinstance(metrics, dict):
                metrics = {}

            # 1. 匹配通路特异 benchmark
            benchmarks = self._match_benchmark_to_pathway(pathway_class)

            # 2. 评估每个 benchmark
            evaluated: list[dict[str, Any]] = []
            for bm in benchmarks:
                evaluated.append(self._evaluate_benchmark(bm, metrics))

            # 3. 任一 benchmark 失败 → pass=False（spec.md 第 310 行）
            #    TD-017 修复（硬门）：无 benchmark 匹配时 pass=False（不再 pass=True 软门放水）
            if not evaluated:
                return {
                    "pass": False,
                    "skipped": True,
                    "reason": "no_benchmark_matched",
                    "benchmarks": [],
                    "method": "no_benchmark_matched",
                }

            pass_flag = all(b.get("pass", False) for b in evaluated)
            return {
                "pass": pass_flag,
                "benchmarks": evaluated,
                "method": "benchmark_validation",
                "pathway_class": pathway_class,
            }
        except Exception as exc:
            # 铁律 #5：失败降级返回 pass=False，但不抛异常
            logger.warning(
                "Level4BenchmarkValidator.validate 失败，降级 pass=False: %s", exc
            )
            return self._fail_report(f"validation_exception: {exc}")

    # =========================================================================
    # SubTask 5.5.4: 通路特异阈值匹配
    # =========================================================================
    def _match_benchmark_to_pathway(self, pathway_class: str) -> list[dict[str, Any]]:
        """解析 pathway_class 并匹配通路特异 benchmark。

        支持 MULTI: 前缀多通路场景：每个识别的子通路都返回对应 benchmark。
        未识别通路（UNKNOWN / 空字符串 / 不在 BENCHMARK_REGISTRY 中）返回空列表。

        Args:
            pathway_class: "EGFR_RTK" 或 "MULTI:EGFR_RTK+PI3K_AKT_mTOR" 或 "UNKNOWN"

        Returns:
            benchmark dict 列表（每项含 BENCHMARK_REGISTRY 中的字段 +
            pathway_class 字段标识来源通路）
        """
        if not pathway_class or pathway_class == "UNKNOWN":
            return []

        # 解析多通路（复用 pathway_planner.parse_pathway_class 逻辑）
        pathways = self._parse_pathway_class(pathway_class)

        benchmarks: list[dict[str, Any]] = []
        seen_metric: set[str] = set()  # 避免重复 benchmark（如多通路含同名）

        for pathway in pathways:
            bm = self._benchmark_registry.get(pathway)
            if bm is None:
                continue
            # 用 metric 作为去重 key（同 metric 的 benchmark 只保留第一个）
            metric_key = bm.get("metric", "")
            if metric_key in seen_metric:
                continue
            seen_metric.add(metric_key)
            # 复制一份并附加 pathway_class 标识
            bm_copy = dict(bm)
            bm_copy["pathway_class"] = pathway
            benchmarks.append(bm_copy)

        return benchmarks

    # =========================================================================
    # SubTask 5.5.3: 文献 benchmark 检索（复用 rag_client）
    # =========================================================================
    def _fetch_benchmark_from_literature(
        self, pathway_class: str
    ) -> dict[str, Any] | None:
        """从 PubMed/RAG 文献检索通路特异 benchmark 参数。

        复用 rag_client.RagClient.search_params 接口（仅 import 调用，不修改其代码）。
        如果 rag_client 不可用或检索失败，返回 None（调用方降级到 BENCHMARK_REGISTRY 硬编码值）。

        Args:
            pathway_class: 通路类别（如 "EGFR_RTK"）

        Returns:
            benchmark dict（含 source_pmid / expected_range / tolerance）；
            不可用或检索失败返回 None。
        """
        try:
            client = self._get_rag_client()
            if client is None:
                return None

            # 不在 BENCHMARK_REGISTRY 中的通路不检索
            base_bm = self._benchmark_registry.get(pathway_class)
            if base_bm is None:
                return None

            # 构造检索 query（基于 benchmark 描述）
            query = (
                f"{pathway_class} {base_bm.get('name', '')} "
                f"{base_bm.get('description', '')}"
            )
            results = client.search_params(query=query, top_k=5)
            if not results:
                return None

            # 从检索结果中提取最相关的实验参数
            # 优先含 PMID 的记录（保证 source_pmid 可追溯）
            for rec in results:
                if not isinstance(rec, dict):
                    continue
                # 期望结果含 source 字段（如 "PMID:12124381"）和 value 字段
                source = rec.get("source", "") or rec.get("pmid", "")
                value = rec.get("value") or rec.get("parameter_value")
                if source and value is not None:
                    try:
                        # 复用 BENCHMARK_REGISTRY 的 expected_range + tolerance
                        # 但用文献检索到的 value 更新 expected（取单点）
                        return {
                            "name": base_bm.get("name", ""),
                            "description": base_bm.get("description", ""),
                            "source_pmid": source,
                            "metric": base_bm.get("metric", ""),
                            "expected_range": base_bm.get("expected_range"),
                            "tolerance": base_bm.get("tolerance", 0.0),
                            "literature_value": float(value),
                            "fetched_from_rag": True,
                        }
                    except (TypeError, ValueError):
                        continue

            return None
        except Exception as exc:
            logger.warning(
                "_fetch_benchmark_from_literature 检索失败，降级到硬编码值: %s", exc
            )
            return None

    # =========================================================================
    # Benchmark 评估
    # =========================================================================
    def _evaluate_benchmark(
        self, benchmark: dict[str, Any], metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """评估单个 benchmark 是否通过。

        判定逻辑：
        1. 从 metrics 中提取 actual 值（按 benchmark.metric 字段查找）
        2. 判断 actual 是否在 expected_range 内 → pass=True
        3. 若超出 expected_range 但在 tolerance 内 → pass=True（容忍偏差）
        4. 若超出 tolerance → pass=False
        5. 若 actual 缺失（None / 不在 metrics 中）→ pass=False（缺数据视为失败）

        Args:
            benchmark: benchmark dict（含 name / source_pmid / expected_range /
                tolerance / metric / pathway_class）
            metrics: 仿真结果指标 dict

        Returns:
            评估结果 dict：{name, source_pmid, expected, actual, diff, pass}
        """
        metric_key = benchmark.get("metric", "")
        expected_range = benchmark.get("expected_range", (None, None))
        tolerance = benchmark.get("tolerance", 0.0)
        source_pmid = benchmark.get("source_pmid", "")
        name = benchmark.get("name", "")

        # 1. 提取 actual 值
        actual = metrics.get(metric_key)
        if actual is not None:
            try:
                actual = float(actual)
            except (TypeError, ValueError):
                actual = None

        # 2. 计算 expected 与 diff
        expected_min, expected_max = self._unpack_range(expected_range)

        # 3. actual 缺失 → pass=False（缺数据视为失败）
        if actual is None:
            return {
                "name": name,
                "source_pmid": source_pmid,
                "expected": self._format_expected(expected_range, tolerance),
                "actual": None,
                "diff": None,
                "pass": False,
                "reason": f"metric '{metric_key}' not found in simulation metrics",
            }

        # 4. 判断是否在 expected_range 内
        in_range = self._is_in_range(actual, expected_min, expected_max)
        if in_range:
            diff = 0.0
            pass_flag = True
        else:
            # 5. 超出 range，检查是否在 tolerance 内
            diff = self._compute_out_of_range_diff(actual, expected_min, expected_max)
            pass_flag = diff <= tolerance

        return {
            "name": name,
            "source_pmid": source_pmid,
            "expected": self._format_expected(expected_range, tolerance),
            "actual": actual,
            "diff": diff,
            "pass": pass_flag,
            "pathway_class": benchmark.get("pathway_class", ""),
        }

    # =========================================================================
    # 辅助函数
    # =========================================================================
    def _parse_pathway_class(self, pathway_class: str) -> list[str]:
        """解析 pathway_class 字符串为 pathway 列表。

        复用 app.pathways.pathway_planner.parse_pathway_class 函数。
        若 import 失败则降级到本地实现（与 parse_pathway_class 保持一致）。

        Args:
            pathway_class: "EGFR_RTK" 或 "MULTI:EGFR_RTK+PI3K_AKT_mTOR" 或 "UNKNOWN"

        Returns:
            pathway 列表（UNKNOWN / 空字符串返回空列表）
        """
        try:
            from app.pathways.pathway_planner import parse_pathway_class

            return parse_pathway_class(pathway_class)
        except Exception:
            # 降级本地实现（与 parse_pathway_class 保持一致）
            if not pathway_class or pathway_class == "UNKNOWN":
                return []
            if pathway_class.startswith("MULTI:"):
                return pathway_class[6:].split("+")
            return [pathway_class]

    def _get_rag_client(self) -> Any:
        """获取 RagClient 实例（延迟创建，避免 import 时副作用）。

        失败返回 None（调用方降级到硬编码值）。
        对注入的 client 也会校验 available 属性，保证统一行为。
        """
        if self._rag_client is not None:
            # 注入的 client 必须显式 available 才返回
            if getattr(self._rag_client, "available", True):
                return self._rag_client
            return None
        try:
            from app.rag_client import RagClient

            client = RagClient()
            if not client.available:
                return None
            self._rag_client = client
            return client
        except Exception as exc:
            logger.debug("RagClient 不可用，降级到硬编码 benchmark: %s", exc)
            return None

    @staticmethod
    def _unpack_range(
        expected_range: Any,
    ) -> tuple[float | None, float | None]:
        """解包 expected_range 为 (min, max)，None 表示无下限/上限。"""
        if not expected_range or not isinstance(expected_range, (tuple, list)):
            return (None, None)
        if len(expected_range) != 2:
            return (None, None)
        return (expected_range[0], expected_range[1])

    @staticmethod
    def _is_in_range(
        value: float, expected_min: float | None, expected_max: float | None
    ) -> bool:
        """判断 value 是否在 [expected_min, expected_max] 范围内。

        None 边界表示无下限/上限约束。
        """
        if expected_min is not None and value < expected_min:
            return False
        if expected_max is not None and value > expected_max:
            return False
        return True

    @staticmethod
    def _compute_out_of_range_diff(
        value: float, expected_min: float | None, expected_max: float | None
    ) -> float:
        """计算 value 超出 expected_range 的绝对偏差。

        - value < expected_min → diff = expected_min - value
        - value > expected_max → diff = value - expected_max
        - 在范围内 → diff = 0
        - 双侧均无约束 → diff = 0
        """
        if expected_min is not None and value < expected_min:
            return float(expected_min - value)
        if expected_max is not None and value > expected_max:
            return float(value - expected_max)
        return 0.0

    @staticmethod
    def _format_expected(
        expected_range: Any, tolerance: float
    ) -> dict[str, Any]:
        """格式化 expected 字段输出到 benchmark 评估结果。"""
        expected_min, expected_max = Level4BenchmarkValidator._unpack_range(
            expected_range
        )
        return {
            "range": [expected_min, expected_max],
            "tolerance": float(tolerance),
        }

    def _fail_report(self, reason: str) -> dict[str, Any]:
        """构造失败降级报告。"""
        return {
            "pass": False,
            "benchmarks": [],
            "method": f"failed: {reason}",
        }


# =============================================================================
# LangGraph hook 节点（Feature Flag 隔离）
# =============================================================================
def level4_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：Level 4 Benchmark Validation hook。

    行为：
    - V4_VALIDATION_PYRAMID_ENABLED=false：直接返回空 dict（不修改 state）
    - V4_VALIDATION_PYRAMID_ENABLED=true：调用 Level4BenchmarkValidator.validate()
      写入 state["v4_validation_report"]["level4"]

    严格遵守不可碰清单：
    - 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
    - 不修改 rag_client.py 代码（仅 import 调用）
    - 不生成 ODE / 不做 SBML 验证
    - 失败时降级返回空 dict，不抛异常，不阻塞主流水线

    Args:
        state: LangGraph 全局状态

    Returns:
        flag=false 时返回 {}
        flag=true 时返回 {"v4_validation_report": {"level4": {...}}}
        异常时返回 {}
    """
    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not settings.effective_v4_validation_pyramid_enabled():
        logger.debug("V4_VALIDATION_PYRAMID_ENABLED effective=false，跳过 Level 4 validation")
        return {}

    try:
        validator = Level4BenchmarkValidator()
        level4_report = validator.validate(state)
        # 与现有 v4_validation_report 合并，不覆盖 level1/level2/level3
        existing_report: dict[str, Any] = {}
        if isinstance(state, dict):
            existing = state.get("v4_validation_report")
            if isinstance(existing, dict):
                existing_report = existing
        merged_report = {**existing_report, "level4": level4_report}
        return {"v4_validation_report": merged_report}
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("Level 4 validation hook 失败，降级跳过: %s", exc)
        return {}


__all__ = [
    "Level4BenchmarkValidator",
    "BENCHMARK_REGISTRY",
    "level4_hook_node",
]
