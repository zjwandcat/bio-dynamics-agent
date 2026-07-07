# BioDynamics Agent v4 - Level 4 Benchmark Validation 单元测试
# (Phase 5 / Task 5.5.5)
#
# 测试 Level4BenchmarkValidator 主类 + BENCHMARK_REGISTRY + 文献检索 +
# 通路匹配 + benchmark 评估 + LangGraph hook。
#
# 测试用例（≥25 个）：
#   - TestLevel4BenchmarkValidator: validate() 主入口 + 异常降级
#   - TestLevel4HookNode: Feature Flag + hook 行为
#   - TestBenchmarkRegistry: 5 个 benchmark 完整性
#   - TestLiteratureFetch: rag_client 可用 / 不可用 / 检索失败降级
#   - TestPathwayMatching: 单通路 / 多通路 / 未知通路 / NF-κB / p53 验证
#   - TestBenchmarkEvaluation: 通过 / 失败 / 多 benchmark / 无 benchmark
#
# 运行：cd backend && python -m pytest tests/test_level4_benchmark.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.validation_v2.level4_benchmark import (
    BENCHMARK_REGISTRY,
    Level4BenchmarkValidator,
    level4_hook_node,
)


# =============================================================================
# TestLevel4BenchmarkValidator: validate() 主入口 + 异常降级
# =============================================================================
class TestLevel4BenchmarkValidator(unittest.TestCase):
    """Level4BenchmarkValidator.validate 主入口测试。"""

    def test_validate_egfr_pass(self):
        """EGFR 通路 benchmark 通过场景（actual 在 range 内）。"""
        validator = Level4BenchmarkValidator()
        state = {
            "v4_pathway_class": "EGFR_RTK",
            "metrics": {"peak_time_minutes": 7.5},  # 在 5-10 min 内
        }
        report = validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertEqual(len(report["benchmarks"]), 1)
        bm = report["benchmarks"][0]
        self.assertEqual(bm["name"], "pEGFR_peak_time")
        self.assertTrue(bm["pass"])
        self.assertEqual(bm["actual"], 7.5)
        self.assertEqual(bm["diff"], 0.0)

    def test_validate_invalid_state_type(self):
        """state 非 dict 时降级 pass=False。"""
        validator = Level4BenchmarkValidator()
        report = validator.validate("not_a_dict")  # type: ignore[arg-type]
        self.assertFalse(report["pass"])
        self.assertEqual(report["benchmarks"], [])

    def test_validate_missing_pathway_class(self):
        """无 v4_pathway_class 时无 benchmark 匹配，pass=True。"""
        validator = Level4BenchmarkValidator()
        state = {"v4_pathway_class": "", "metrics": {}}
        report = validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertEqual(report["benchmarks"], [])
        self.assertEqual(report["method"], "no_benchmark_matched")

    def test_validate_unknown_pathway_class(self):
        """UNKNOWN pathway_class 时无 benchmark 匹配。"""
        validator = Level4BenchmarkValidator()
        state = {"v4_pathway_class": "UNKNOWN", "metrics": {}}
        report = validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertEqual(report["benchmarks"], [])

    def test_validate_exception_degradation(self):
        """异常时降级返回 pass=False，不抛异常。"""
        validator = Level4BenchmarkValidator()

        # 注入 _match_benchmark_to_pathway 抛异常
        with patch.object(
            validator,
            "_match_benchmark_to_pathway",
            side_effect=RuntimeError("mocked exception"),
        ):
            report = validator.validate(
                {"v4_pathway_class": "EGFR_RTK", "metrics": {}}
            )
        self.assertFalse(report["pass"])
        self.assertEqual(report["benchmarks"], [])
        self.assertIn("validation_exception", report["method"])

    def test_validate_metrics_non_dict_defaults_to_empty(self):
        """metrics 非 dict 时降级为空 dict。"""
        validator = Level4BenchmarkValidator()
        state = {
            "v4_pathway_class": "EGFR_RTK",
            "metrics": "not_a_dict",  # type: ignore[assignment]
        }
        report = validator.validate(state)
        # 无 actual 数据 → benchmark 失败
        self.assertFalse(report["pass"])
        self.assertEqual(len(report["benchmarks"]), 1)
        self.assertFalse(report["benchmarks"][0]["pass"])


# =============================================================================
# TestLevel4HookNode: Feature Flag + hook 行为
# =============================================================================
class TestLevel4HookNode(unittest.TestCase):
    """LangGraph hook 节点 Feature Flag 行为测试。"""

    def test_hook_flag_off_returns_empty(self):
        """V4_VALIDATION_PYRAMID_ENABLED=false 时 hook 返回 {}。"""
        with patch(
            "app.validation_v2.level4_benchmark.settings"
        ) as mock_settings:
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = False
            mock_settings.effective_v4_validation_pyramid_enabled.return_value = False
            result = level4_hook_node(
                {"v4_pathway_class": "EGFR_RTK", "metrics": {}}
            )
        self.assertEqual(result, {})

    def test_hook_flag_on_writes_level4(self):
        """V4_VALIDATION_PYRAMID_ENABLED=true 时 hook 写入 v4_validation_report.level4。"""
        with patch(
            "app.validation_v2.level4_benchmark.settings"
        ) as mock_settings:
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
            result = level4_hook_node(
                {
                    "v4_pathway_class": "EGFR_RTK",
                    "metrics": {"peak_time_minutes": 7.5},
                }
            )
        self.assertIn("v4_validation_report", result)
        self.assertIn("level4", result["v4_validation_report"])
        self.assertTrue(result["v4_validation_report"]["level4"]["pass"])

    def test_hook_preserves_existing_levels(self):
        """hook 不覆盖已存在的 level1/level2/level3。"""
        existing_report = {
            "level1": {"pass": True},
            "level2": {"pass": False, "track": "B"},
        }
        with patch(
            "app.validation_v2.level4_benchmark.settings"
        ) as mock_settings:
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
            result = level4_hook_node(
                {
                    "v4_pathway_class": "EGFR_RTK",
                    "metrics": {"peak_time_minutes": 7.5},
                    "v4_validation_report": existing_report,
                }
            )
        merged = result["v4_validation_report"]
        self.assertIn("level1", merged)
        self.assertIn("level2", merged)
        self.assertIn("level4", merged)
        self.assertEqual(merged["level1"]["pass"], True)
        self.assertEqual(merged["level2"]["track"], "B")

    def test_hook_exception_returns_empty(self):
        """hook 异常时返回 {}，不抛异常。"""
        with patch(
            "app.validation_v2.level4_benchmark.settings"
        ) as mock_settings:
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
            with patch(
                "app.validation_v2.level4_benchmark.Level4BenchmarkValidator.validate",
                side_effect=RuntimeError("mocked"),
            ):
                result = level4_hook_node(
                    {"v4_pathway_class": "EGFR_RTK", "metrics": {}}
                )
        self.assertEqual(result, {})


# =============================================================================
# TestBenchmarkRegistry: BENCHMARK_REGISTRY 5 个 benchmark 完整性
# =============================================================================
class TestBenchmarkRegistry(unittest.TestCase):
    """BENCHMARK_REGISTRY 5 个 benchmark 完整性测试。"""

    def test_registry_has_five_benchmarks(self):
        """BENCHMARK_REGISTRY 含 5 个通路 benchmark。"""
        expected_keys = {"EGFR_RTK", "MAPK_ERK", "NF_KB", "p53", "WNT"}
        self.assertEqual(set(BENCHMARK_REGISTRY.keys()), expected_keys)

    def test_egfr_benchmark_schoeberl_2002(self):
        """EGFR benchmark 来源 Schoeberl 2002 (PMID:12124381)。"""
        bm = BENCHMARK_REGISTRY["EGFR_RTK"]
        self.assertEqual(bm["source_pmid"], "PMID:12124381")
        self.assertEqual(bm["name"], "pEGFR_peak_time")
        self.assertEqual(bm["metric"], "peak_time_minutes")
        self.assertEqual(bm["expected_range"], (5.0, 10.0))
        self.assertEqual(bm["tolerance"], 2.0)

    def test_mapk_benchmark_markevich_2004(self):
        """MAPK benchmark 来源 Markevich 2004 (PMID:14757805)。"""
        bm = BENCHMARK_REGISTRY["MAPK_ERK"]
        self.assertEqual(bm["source_pmid"], "PMID:14757805")
        self.assertEqual(bm["name"], "MAPK_Hill_coefficient")
        self.assertEqual(bm["metric"], "hill_coefficient")
        self.assertEqual(bm["expected_range"], (2.0, None))
        self.assertEqual(bm["tolerance"], 0.5)

    def test_nf_kappa_b_benchmark_nelson_2004(self):
        """NF-κB benchmark 来源 Nelson 2004 (PMID:14975635)。"""
        bm = BENCHMARK_REGISTRY["NF_KB"]
        self.assertEqual(bm["source_pmid"], "PMID:14975635")
        self.assertEqual(bm["name"], "NF_kB_oscillation_period")
        self.assertEqual(bm["metric"], "oscillation_period_hours")
        self.assertEqual(bm["expected_range"], (1.0, 2.0))
        self.assertEqual(bm["tolerance"], 0.5)

    def test_p53_benchmark_lev_bar_or_2000(self):
        """p53 benchmark 来源 Lev Bar-Or 2000 (PMID:10644694)。"""
        bm = BENCHMARK_REGISTRY["p53"]
        self.assertEqual(bm["source_pmid"], "PMID:10644694")
        self.assertEqual(bm["name"], "p53_pulse_period")
        self.assertEqual(bm["metric"], "pulse_period_hours")
        self.assertEqual(bm["expected_range"], (5.0, 7.0))
        self.assertEqual(bm["tolerance"], 1.0)

    def test_wnt_benchmark_lee_2003(self):
        """Wnt benchmark 来源 Lee 2003 (PMID:12906785)。"""
        bm = BENCHMARK_REGISTRY["WNT"]
        self.assertEqual(bm["source_pmid"], "PMID:12906785")
        self.assertEqual(bm["name"], "beta_catenin_steady_state")
        self.assertEqual(bm["metric"], "steady_state_nM")
        self.assertEqual(bm["expected_range"], (None, 10.0))
        self.assertEqual(bm["tolerance"], 2.0)

    def test_all_benchmarks_have_source_pmid(self):
        """每个 benchmark 都有 source_pmid 字段。"""
        for key, bm in BENCHMARK_REGISTRY.items():
            self.assertIn("source_pmid", bm, f"{key} 缺 source_pmid")
            self.assertTrue(
                bm["source_pmid"].startswith("PMID:"),
                f"{key} source_pmid 不是 PMID 格式: {bm['source_pmid']}",
            )

    def test_all_benchmarks_have_required_fields(self):
        """每个 benchmark 都含 name / description / source_pmid / metric /
        expected_range / tolerance 字段。"""
        required = {
            "name",
            "description",
            "source_pmid",
            "metric",
            "expected_range",
            "tolerance",
        }
        for key, bm in BENCHMARK_REGISTRY.items():
            missing = required - set(bm.keys())
            self.assertEqual(missing, set(), f"{key} 缺字段: {missing}")


# =============================================================================
# TestLiteratureFetch: rag_client 可用 / 不可用 / 检索失败降级
# =============================================================================
class TestLiteratureFetch(unittest.TestCase):
    """_fetch_benchmark_from_literature 文献检索测试。"""

    def test_rag_client_available_returns_benchmark(self):
        """rag_client 可用时返回文献检索结果。"""
        mock_client = MagicMock()
        mock_client.available = True
        mock_client.search_params.return_value = [
            {
                "source": "PMID:12124381",
                "value": 7.5,
                "species": "EGFR",
            }
        ]
        validator = Level4BenchmarkValidator(rag_client=mock_client)
        result = validator._fetch_benchmark_from_literature("EGFR_RTK")
        self.assertIsNotNone(result)
        self.assertEqual(result["source_pmid"], "PMID:12124381")
        self.assertEqual(result["literature_value"], 7.5)
        self.assertTrue(result["fetched_from_rag"])

    def test_rag_client_unavailable_returns_none(self):
        """rag_client 不可用时返回 None，降级到硬编码值。"""
        mock_client = MagicMock()
        mock_client.available = False
        validator = Level4BenchmarkValidator(rag_client=mock_client)
        # _get_rag_client 检查 available=False 返回 None
        result = validator._fetch_benchmark_from_literature("EGFR_RTK")
        self.assertIsNone(result)

    def test_rag_client_search_returns_empty_returns_none(self):
        """rag_client 检索返回空列表时返回 None。"""
        mock_client = MagicMock()
        mock_client.available = True
        mock_client.search_params.return_value = []
        validator = Level4BenchmarkValidator(rag_client=mock_client)
        result = validator._fetch_benchmark_from_literature("EGFR_RTK")
        self.assertIsNone(result)

    def test_rag_client_search_raises_returns_none(self):
        """rag_client 检索抛异常时返回 None（不抛异常）。"""
        mock_client = MagicMock()
        mock_client.available = True
        mock_client.search_params.side_effect = RuntimeError("search failed")
        validator = Level4BenchmarkValidator(rag_client=mock_client)
        result = validator._fetch_benchmark_from_literature("EGFR_RTK")
        self.assertIsNone(result)

    def test_fetch_unknown_pathway_returns_none(self):
        """未在 BENCHMARK_REGISTRY 中的通路返回 None。"""
        mock_client = MagicMock()
        mock_client.available = True
        validator = Level4BenchmarkValidator(rag_client=mock_client)
        result = validator._fetch_benchmark_from_literature("UNKNOWN_PATHWAY")
        self.assertIsNone(result)

    def test_fetch_invalid_value_skipped(self):
        """检索结果 value 无法转为 float 时跳过该记录。"""
        mock_client = MagicMock()
        mock_client.available = True
        mock_client.search_params.return_value = [
            {"source": "PMID:12124381", "value": "not_a_number"},
        ]
        validator = Level4BenchmarkValidator(rag_client=mock_client)
        result = validator._fetch_benchmark_from_literature("EGFR_RTK")
        self.assertIsNone(result)

    def test_rag_client_none_defaults_to_lazy_create(self):
        """rag_client=None 时延迟创建，失败时返回 None。"""
        validator = Level4BenchmarkValidator(rag_client=None)
        # 通过 patch RagClient import 失败模拟
        with patch(
            "app.rag_client.RagClient",
            side_effect=ImportError("mocked unavailable"),
        ):
            result = validator._get_rag_client()
        self.assertIsNone(result)


# =============================================================================
# TestPathwayMatching: 单通路 / 多通路 / 未知通路 / NF-κB / p53
# =============================================================================
class TestPathwayMatching(unittest.TestCase):
    """_match_benchmark_to_pathway 通路匹配测试。"""

    def test_single_pathway_egfr_rtk(self):
        """单通路 EGFR_RTK 匹配 1 个 benchmark。"""
        validator = Level4BenchmarkValidator()
        benchmarks = validator._match_benchmark_to_pathway("EGFR_RTK")
        self.assertEqual(len(benchmarks), 1)
        self.assertEqual(benchmarks[0]["name"], "pEGFR_peak_time")
        self.assertEqual(benchmarks[0]["pathway_class"], "EGFR_RTK")

    def test_multi_pathway_returns_multiple(self):
        """MULTI: 多通路匹配多个 benchmark。"""
        validator = Level4BenchmarkValidator()
        benchmarks = validator._match_benchmark_to_pathway(
            "MULTI:EGFR_RTK+PI3K_AKT_mTOR"
        )
        # PI3K_AKT_mTOR 不在 BENCHMARK_REGISTRY 中，只有 EGFR_RTK 匹配
        self.assertEqual(len(benchmarks), 1)
        self.assertEqual(benchmarks[0]["pathway_class"], "EGFR_RTK")

    def test_multi_pathway_egfr_and_nfkb(self):
        """MULTI:EGFR_RTK+NF_KB 匹配 2 个 benchmark。"""
        validator = Level4BenchmarkValidator()
        benchmarks = validator._match_benchmark_to_pathway(
            "MULTI:EGFR_RTK+NF_KB"
        )
        self.assertEqual(len(benchmarks), 2)
        names = {b["name"] for b in benchmarks}
        self.assertEqual(
            names, {"pEGFR_peak_time", "NF_kB_oscillation_period"}
        )

    def test_unknown_pathway_returns_empty(self):
        """UNKNOWN pathway_class 返回空列表。"""
        validator = Level4BenchmarkValidator()
        benchmarks = validator._match_benchmark_to_pathway("UNKNOWN")
        self.assertEqual(benchmarks, [])

    def test_empty_pathway_returns_empty(self):
        """空字符串 pathway_class 返回空列表。"""
        validator = Level4BenchmarkValidator()
        benchmarks = validator._match_benchmark_to_pathway("")
        self.assertEqual(benchmarks, [])

    def test_nf_kappa_b_oscillation_period_1_to_2_hours(self):
        """验证 NF-κB 振荡周期 benchmark：1-2h（Nelson 2004）。"""
        validator = Level4BenchmarkValidator()
        state = {
            "v4_pathway_class": "NF_KB",
            "metrics": {"oscillation_period_hours": 1.5},  # 在 1-2h 内
        }
        report = validator.validate(state)
        self.assertTrue(report["pass"])
        bm = report["benchmarks"][0]
        self.assertEqual(bm["name"], "NF_kB_oscillation_period")
        self.assertEqual(bm["actual"], 1.5)
        self.assertTrue(bm["pass"])

    def test_p53_pulse_period_5_to_7_hours(self):
        """验证 p53 脉冲周期 benchmark：5-7h（Lev Bar-Or 2000）。"""
        validator = Level4BenchmarkValidator()
        state = {
            "v4_pathway_class": "p53",
            "metrics": {"pulse_period_hours": 6.0},  # 在 5-7h 内
        }
        report = validator.validate(state)
        self.assertTrue(report["pass"])
        bm = report["benchmarks"][0]
        self.assertEqual(bm["name"], "p53_pulse_period")
        self.assertEqual(bm["actual"], 6.0)
        self.assertTrue(bm["pass"])

    def test_pathway_class_with_pathway_planner(self):
        """复用 pathway_planner.parse_pathway_class 解析 MULTI: 前缀。"""
        validator = Level4BenchmarkValidator()
        # 验证 _parse_pathway_class 内部调用了 pathway_planner
        benchmarks = validator._match_benchmark_to_pathway(
            "MULTI:WNT+p53"
        )
        self.assertEqual(len(benchmarks), 2)
        names = {b["name"] for b in benchmarks}
        self.assertEqual(
            names, {"beta_catenin_steady_state", "p53_pulse_period"}
        )


# =============================================================================
# TestBenchmarkEvaluation: 通过 / 失败 / 多 benchmark / 无 benchmark
# =============================================================================
class TestBenchmarkEvaluation(unittest.TestCase):
    """_evaluate_benchmark 单个 benchmark 评估测试。"""

    def test_benchmark_in_range_passes(self):
        """actual 在 expected_range 内 → pass=True，diff=0。"""
        validator = Level4BenchmarkValidator()
        bm = dict(BENCHMARK_REGISTRY["EGFR_RTK"])
        bm["pathway_class"] = "EGFR_RTK"
        result = validator._evaluate_benchmark(
            bm, {"peak_time_minutes": 7.5}
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["diff"], 0.0)
        self.assertEqual(result["actual"], 7.5)

    def test_benchmark_out_of_tolerance_fails(self):
        """actual 超出 tolerance → pass=False。"""
        validator = Level4BenchmarkValidator()
        bm = dict(BENCHMARK_REGISTRY["EGFR_RTK"])
        # expected_range=(5, 10), tolerance=2.0
        # actual=15 → diff=15-10=5 > tolerance=2 → fail
        bm["pathway_class"] = "EGFR_RTK"
        result = validator._evaluate_benchmark(
            bm, {"peak_time_minutes": 15.0}
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["diff"], 5.0)
        self.assertEqual(result["actual"], 15.0)

    def test_benchmark_within_tolerance_passes(self):
        """actual 超出 range 但在 tolerance 内 → pass=True。"""
        validator = Level4BenchmarkValidator()
        bm = dict(BENCHMARK_REGISTRY["EGFR_RTK"])
        # expected_range=(5, 10), tolerance=2.0
        # actual=11.5 → diff=11.5-10=1.5 <= tolerance=2 → pass
        bm["pathway_class"] = "EGFR_RTK"
        result = validator._evaluate_benchmark(
            bm, {"peak_time_minutes": 11.5}
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["diff"], 1.5)

    def test_benchmark_missing_metric_fails(self):
        """actual 缺失（不在 metrics 中）→ pass=False。"""
        validator = Level4BenchmarkValidator()
        bm = dict(BENCHMARK_REGISTRY["EGFR_RTK"])
        bm["pathway_class"] = "EGFR_RTK"
        result = validator._evaluate_benchmark(bm, {})
        self.assertFalse(result["pass"])
        self.assertIsNone(result["actual"])
        self.assertIsNone(result["diff"])

    def test_benchmark_with_lower_bound_only(self):
        """MAPK benchmark 只有下限（>2）→ actual=3.0 通过。"""
        validator = Level4BenchmarkValidator()
        bm = dict(BENCHMARK_REGISTRY["MAPK_ERK"])
        bm["pathway_class"] = "MAPK_ERK"
        result = validator._evaluate_benchmark(
            bm, {"hill_coefficient": 3.0}
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["diff"], 0.0)

    def test_benchmark_with_upper_bound_only(self):
        """Wnt benchmark 只有上限（<10）→ actual=8.0 通过。"""
        validator = Level4BenchmarkValidator()
        bm = dict(BENCHMARK_REGISTRY["WNT"])
        bm["pathway_class"] = "WNT"
        result = validator._evaluate_benchmark(
            bm, {"steady_state_nM": 8.0}
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["diff"], 0.0)

    def test_benchmark_upper_bound_exceeded(self):
        """Wnt benchmark actual=15 超出上限 → fail（diff=5 > tolerance=2）。"""
        validator = Level4BenchmarkValidator()
        bm = dict(BENCHMARK_REGISTRY["WNT"])
        bm["pathway_class"] = "WNT"
        result = validator._evaluate_benchmark(
            bm, {"steady_state_nM": 15.0}
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["diff"], 5.0)

    def test_multiple_benchmarks_one_fails_overall_fails(self):
        """多 benchmark 场景：任一失败 → pass=False（spec.md 第 310 行）。"""
        validator = Level4BenchmarkValidator()
        state = {
            "v4_pathway_class": "MULTI:EGFR_RTK+NF_KB",
            "metrics": {
                "peak_time_minutes": 7.5,  # EGFR pass
                "oscillation_period_hours": 5.0,  # NF-κB fail (5-2=3 > tolerance=0.5)
            },
        }
        report = validator.validate(state)
        self.assertFalse(report["pass"])
        self.assertEqual(len(report["benchmarks"]), 2)
        bm_dict = {b["name"]: b for b in report["benchmarks"]}
        self.assertTrue(bm_dict["pEGFR_peak_time"]["pass"])
        self.assertFalse(bm_dict["NF_kB_oscillation_period"]["pass"])

    def test_multiple_benchmarks_all_pass_overall_passes(self):
        """多 benchmark 全部通过 → pass=True。"""
        validator = Level4BenchmarkValidator()
        state = {
            "v4_pathway_class": "MULTI:EGFR_RTK+NF_KB",
            "metrics": {
                "peak_time_minutes": 7.5,  # EGFR pass
                "oscillation_period_hours": 1.5,  # NF-κB pass
            },
        }
        report = validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertEqual(len(report["benchmarks"]), 2)

    def test_no_benchmark_passes(self):
        """无 benchmark 匹配时 pass=True（无验证规则）。"""
        validator = Level4BenchmarkValidator()
        state = {
            "v4_pathway_class": "UNKNOWN_PATHWAY",  # 不在 registry 中
            "metrics": {},
        }
        report = validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertEqual(report["benchmarks"], [])
        self.assertEqual(report["method"], "no_benchmark_matched")

    def test_benchmark_expected_field_format(self):
        """benchmark 评估结果 expected 字段格式正确。"""
        validator = Level4BenchmarkValidator()
        bm = dict(BENCHMARK_REGISTRY["EGFR_RTK"])
        bm["pathway_class"] = "EGFR_RTK"
        result = validator._evaluate_benchmark(
            bm, {"peak_time_minutes": 7.5}
        )
        expected = result["expected"]
        self.assertIn("range", expected)
        self.assertIn("tolerance", expected)
        self.assertEqual(expected["range"], [5.0, 10.0])
        self.assertEqual(expected["tolerance"], 2.0)

    def test_benchmark_with_custom_registry(self):
        """注入自定义 benchmark_registry 测试。"""
        custom_registry = {
            "CUSTOM_PATHWAY": {
                "name": "custom_benchmark",
                "description": "test custom benchmark",
                "source_pmid": "PMID:00000001",
                "metric": "custom_metric",
                "expected_range": (0.0, 1.0),
                "tolerance": 0.1,
            }
        }
        validator = Level4BenchmarkValidator(
            benchmark_registry=custom_registry
        )
        state = {
            "v4_pathway_class": "CUSTOM_PATHWAY",
            "metrics": {"custom_metric": 0.5},
        }
        report = validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertEqual(len(report["benchmarks"]), 1)
        self.assertEqual(report["benchmarks"][0]["name"], "custom_benchmark")


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    unittest.main()
