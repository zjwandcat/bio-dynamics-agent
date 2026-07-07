# BioDynamics Agent v4 - Official Benchmark Suite unit tests (RC Sprint Task E.1)
#
# Coverage:
#   1. All 10 YAML files exist in backend/benchmarks/ and pass schema validation
#      (6+ required top-level fields).
#   2. BenchmarkRunner.load_all() loads all 10 benchmarks.
#   3. BenchmarkRunner.run_benchmark() returns a well-formed result dict for
#      EGFR_RTK and p53 (the two fastest pathways).
#   4. /api/v4/benchmarks/run SSE endpoint is registered on the FastAPI app
#      and returns a StreamingResponse with the expected SSE event sequence.
#
# Run: cd backend && python -m pytest tests/test_benchmark_suite.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure backend/ is on sys.path so `import app...` works from the tests dir.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# Constants — canonical list of 10 pathways + 6-field schema
# =============================================================================
EXPECTED_PATHWAY_CLASSES: dict[str, str] = {
    "EGFR_RTK": "egfr_signaling.yaml",
    "MAPK_ERK": "mapk_cascade.yaml",
    "PI3K_AKT_mTOR": "pi3k_akt_mtor.yaml",
    "p53": "p53_signaling.yaml",
    "APOPTOSIS": "apoptosis.yaml",
    "CELL_CYCLE": "cell_cycle.yaml",
    "JAK_STAT": "jak_stat.yaml",
    "NF_KB": "nfkb_signaling.yaml",
    "WNT": "wnt_signaling.yaml",
    "TGF_BETA": "tgf_beta_signaling.yaml",
}

REQUIRED_FIELDS = (
    "pathway_class",
    "name",
    "input",
    "ground_truth",
    "expected_dynamics",
    "validation",
    "pass_criteria",
    "performance",
)


# =============================================================================
# Test 1: YAML file existence + schema validation
# =============================================================================
class TestBenchmarkYamlSchema(unittest.TestCase):
    """Validate all 10 YAML benchmark files exist and have the 6-field schema."""

    def setUp(self):
        self.benchmarks_dir = BACKEND_DIR / "benchmarks"
        self.assertTrue(
            self.benchmarks_dir.exists(),
            f"benchmarks dir missing: {self.benchmarks_dir}",
        )

    def test_all_10_yaml_files_exist(self):
        """All 10 expected YAML files exist in backend/benchmarks/."""
        for pathway_class, filename in EXPECTED_PATHWAY_CLASSES.items():
            yaml_path = self.benchmarks_dir / filename
            self.assertTrue(
                yaml_path.exists(),
                f"missing benchmark file for {pathway_class}: {yaml_path}",
            )

    def test_each_yaml_has_required_schema(self):
        """Each YAML file contains all 6+ required top-level fields."""
        import yaml

        for pathway_class, filename in EXPECTED_PATHWAY_CLASSES.items():
            yaml_path = self.benchmarks_dir / filename
            with yaml_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            self.assertIsInstance(
                data, dict, f"{filename}: top-level YAML must be a dict"
            )
            for field in REQUIRED_FIELDS:
                self.assertIn(
                    field,
                    data,
                    f"{filename}: missing required field '{field}'",
                )
            # pathway_class must match the canonical identifier.
            self.assertEqual(
                data["pathway_class"],
                pathway_class,
                f"{filename}: pathway_class mismatch "
                f"(expected '{pathway_class}', got '{data['pathway_class']}')",
            )
            # pass_criteria must be a non-empty list.
            self.assertIsInstance(
                data["pass_criteria"],
                list,
                f"{filename}: pass_criteria must be a list",
            )
            self.assertGreater(
                len(data["pass_criteria"]),
                0,
                f"{filename}: pass_criteria must not be empty",
            )

    def test_no_extra_unexpected_yaml_files(self):
        """The benchmarks/ directory should contain exactly the 10 expected files."""
        yaml_files = {p.name for p in self.benchmarks_dir.glob("*.yaml")}
        expected_files = set(EXPECTED_PATHWAY_CLASSES.values())
        extra = yaml_files - expected_files
        self.assertEqual(
            extra,
            set(),
            f"unexpected extra YAML files in benchmarks/: {extra}",
        )


# =============================================================================
# Test 2: BenchmarkRunner loading
# =============================================================================
class TestBenchmarkRunnerLoading(unittest.TestCase):
    """BenchmarkRunner loads all 10 benchmarks from disk."""

    def setUp(self):
        from app.benchmark_runner import BenchmarkRunner

        self.runner = BenchmarkRunner()

    def test_load_all_returns_10_benchmarks(self):
        """load_all() returns exactly 10 pathway benchmarks."""
        loaded = self.runner.load_all()
        self.assertEqual(len(loaded), 10, f"expected 10 benchmarks, got {len(loaded)}")
        for pathway_class in EXPECTED_PATHWAY_CLASSES:
            self.assertIn(
                pathway_class,
                loaded,
                f"pathway_class '{pathway_class}' not loaded",
            )

    def test_list_benchmarks_returns_sorted_10(self):
        """list_benchmarks() returns the 10 pathway_class identifiers."""
        listed = self.runner.list_benchmarks()
        self.assertEqual(len(listed), 10)
        for pathway_class in EXPECTED_PATHWAY_CLASSES:
            self.assertIn(pathway_class, listed)


# =============================================================================
# Test 3: BenchmarkRunner.run_benchmark for EGFR + p53
# =============================================================================
class TestBenchmarkRunnerSingleRun(unittest.TestCase):
    """run_benchmark returns well-formed result dict for fast pathways."""

    def setUp(self):
        from app.benchmark_runner import BenchmarkRunner

        # Register all specialists explicitly (some test files clear the
        # registry in their tearDown). Auto-import side effects in
        # BenchmarkRunner._ensure_specialists_registered handle this, but
        # we re-register here defensively.
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        for cls in (
            EGFRSpecialist,
            MAPKSpecialist,
            PI3KAKTmTORSpecialist,
            P53Specialist,
            ApoptosisSpecialist,
            CellCycleSpecialist,
            JakStatSpecialist,
            NfKappaBSpecialist,
            WntSpecialist,
            TgfBetaSpecialist,
        ):
            register_specialist(cls)

        self.runner = BenchmarkRunner()

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def _assert_well_formed_result(self, result: dict, pathway_class: str):
        """Common structural assertions for a run_benchmark result."""
        self.assertIsInstance(result, dict)
        self.assertEqual(result["pathway_class"], pathway_class)
        self.assertIn(result["status"], ("pass", "fail"))
        self.assertIsInstance(result["checks"], list)
        self.assertGreater(len(result["checks"]), 0)
        for check in result["checks"]:
            self.assertIn("criterion", check)
            self.assertIn("metric_name", check)
            self.assertIn("passed", check)
            self.assertIsInstance(check["passed"], bool)
            self.assertIn("detail", check)
        self.assertIsInstance(result["runtime_seconds"], float)
        self.assertGreaterEqual(result["runtime_seconds"], 0.0)
        self.assertIsInstance(result["errors"], list)

    def test_run_benchmark_egfr(self):
        """run_benchmark('EGFR_RTK') returns a well-formed result dict."""
        result = self.runner.run_benchmark("EGFR_RTK")
        self._assert_well_formed_result(result, "EGFR_RTK")
        # EGFR should pass with the synthetic-metrics shortcut (specialist
        # emits literature midpoints that fall inside pass_criteria ranges).
        self.assertEqual(
            result["status"],
            "pass",
            f"EGFR benchmark should pass, got errors: {result['errors']}",
        )
        # All individual checks should be True.
        for check in result["checks"]:
            self.assertTrue(
                check["passed"],
                f"EGFR check failed: {check['criterion']} -> {check['detail']}",
            )

    def test_run_benchmark_p53(self):
        """run_benchmark('p53') returns a well-formed result dict."""
        result = self.runner.run_benchmark("p53")
        self._assert_well_formed_result(result, "p53")
        self.assertEqual(
            result["status"],
            "pass",
            f"p53 benchmark should pass, got errors: {result['errors']}",
        )
        for check in result["checks"]:
            self.assertTrue(
                check["passed"],
                f"p53 check failed: {check['criterion']} -> {check['detail']}",
            )

    def test_run_benchmark_unknown_pathway_returns_fail(self):
        """run_benchmark on unknown pathway_class returns status='fail'."""
        result = self.runner.run_benchmark("NONEXISTENT_PATHWAY")
        self.assertEqual(result["status"], "fail")
        self.assertGreater(len(result["errors"]), 0)
        self.assertIn("not found", result["errors"][0])


# =============================================================================
# Test 4: /api/v4/benchmarks/run endpoint
# =============================================================================
class TestV4BenchmarkEndpoint(unittest.TestCase):
    """/api/v4/benchmarks/run is registered and streams SSE events."""

    def setUp(self):
        # Re-register specialists so the endpoint's runner can resolve them.
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        for cls in (
            EGFRSpecialist,
            MAPKSpecialist,
            PI3KAKTmTORSpecialist,
            P53Specialist,
            ApoptosisSpecialist,
            CellCycleSpecialist,
            JakStatSpecialist,
            NfKappaBSpecialist,
            WntSpecialist,
            TgfBetaSpecialist,
        ):
            register_specialist(cls)

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_v4_endpoint_route_registered(self):
        """/api/v4/benchmarks/run is registered as a POST route on the app."""
        from app.main import app

        routes = {route.path: route.methods for route in app.routes}
        self.assertIn("/api/v4/benchmarks/run", routes)
        methods = routes["/api/v4/benchmarks/run"]
        self.assertIn("POST", methods)

    def test_v4_endpoint_streams_expected_sse_events(self):
        """The endpoint streams benchmark_start/result/complete/end events."""
        import json

        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as client:
            response = client.post("/api/v4/benchmarks/run")
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers.get("content-type", ""))

            # Parse SSE event types from the body.
            body = response.text
            events: list[str] = []
            for line in body.splitlines():
                if line.startswith("data: "):
                    payload = line[len("data: "):]
                    try:
                        data = json.loads(payload)
                        if isinstance(data, dict) and "event" in data:
                            events.append(data["event"])
                    except json.JSONDecodeError:
                        pass

        # The endpoint must emit at least one of each required event type.
        self.assertIn("benchmark_start", events)
        self.assertIn("benchmark_result", events)
        self.assertIn("benchmark_complete", events)
        self.assertIn("end", events)
        # 1 suite-level start + 10 per-pathway starts = 11+ benchmark_start.
        self.assertGreaterEqual(
            events.count("benchmark_start"), 11
        )
        # Exactly 10 benchmark_result events (one per pathway).
        self.assertEqual(events.count("benchmark_result"), 10)
        # Exactly 1 benchmark_complete event.
        self.assertEqual(events.count("benchmark_complete"), 1)


# =============================================================================
# Test 5: run_all() summary structure
# =============================================================================
class TestBenchmarkRunnerRunAll(unittest.TestCase):
    """run_all() returns a summary dict with pass/fail counts."""

    def setUp(self):
        from app.benchmark_runner import BenchmarkRunner
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        for cls in (
            EGFRSpecialist,
            MAPKSpecialist,
            PI3KAKTmTORSpecialist,
            P53Specialist,
            ApoptosisSpecialist,
            CellCycleSpecialist,
            JakStatSpecialist,
            NfKappaBSpecialist,
            WntSpecialist,
            TgfBetaSpecialist,
        ):
            register_specialist(cls)
        self.runner = BenchmarkRunner()

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_run_all_summary_structure(self):
        """run_all() returns total/passed/failed/results/runtime_seconds."""
        summary = self.runner.run_all()
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["total"], 10)
        self.assertEqual(summary["passed"] + summary["failed"], 10)
        self.assertIsInstance(summary["results"], list)
        self.assertEqual(len(summary["results"]), 10)
        self.assertIsInstance(summary["runtime_seconds"], float)
        self.assertGreaterEqual(summary["runtime_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
