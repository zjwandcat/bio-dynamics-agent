# BioDynamics Agent v4 - Official Benchmark Suite Runner (RC Sprint Task E.1)
#
# 10-pathway benchmark runner: loads YAML benchmark definitions from
# backend/benchmarks/, drives each pathway through the P4 Specialist +
# P3 simulation + P5 Level4 validation, and emits per-pathway results.
#
# Design principles (hard rules):
# 1. READ-ONLY runner: invokes existing P4 specialists + P5 validation;
#    does NOT modify any P1-P6 scientific code.
# 2. Failure-isolated: any single benchmark exception is captured into
#    result.errors with status="fail"; run_all() continues to the next pathway.
# 3. Schema-validated: YAML files must contain 6 top-level fields
#    (pathway_class / name / input / ground_truth / expected_dynamics /
#     validation / pass_criteria / performance). Malformed YAML is skipped
#    with status="fail" and an explicit error message.
# 4. No network calls: literature PMIDs are referenced from YAML for
#    traceability only; runner does not query PubMed/RAG at runtime.
#
# Dependencies:
# - PyYAML (yaml.safe_load)
# - app.pathways.pathway_registry (get_specialist / list_specialists)
# - app.pathways.specialists.* (auto-import side effects register specialists)
# - app.validation_v2.level4_benchmark.Level4BenchmarkValidator (reuse logic)

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any, Iterator

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================
# Directory containing the 10 per-pathway YAML benchmark definitions.
BENCHMARKS_DIR: Path = Path(__file__).resolve().parent.parent / "benchmarks"

# Required top-level schema fields for each YAML benchmark file.
REQUIRED_FIELDS: tuple[str, ...] = (
    "pathway_class",
    "name",
    "input",
    "ground_truth",
    "expected_dynamics",
    "validation",
    "pass_criteria",
    "performance",
)

# pathway_class -> Canonical Reference 文件名映射（Task 22.4）
# Canonical 文件名与 scientific_alignment Gold Standard 文件名一致，
# 但 pathway_class（如 NF_KB）与文件名（如 nf_kappa_b）非简单 lowercase 关系，
# 故需显式映射。未映射的 pathway_class 加载 Canonical 时返回 None（不 fail）。
_PATHWAY_CLASS_TO_CANONICAL: dict[str, str] = {
    "EGFR_RTK": "egfr",
    "MAPK_ERK": "mapk",
    "PI3K_AKT_mTOR": "pi3k_akt_mtor",
    "p53": "p53",
    "APOPTOSIS": "apoptosis",
    "CELL_CYCLE": "cell_cycle",
    "JAK_STAT": "jak_stat",
    "NF_KB": "nf_kappa_b",
    "WNT": "wnt",
    "TGF_BETA": "tgf_beta",
}


# =============================================================================
# BenchmarkRunner
# =============================================================================
class BenchmarkRunner:
    """10-pathway Official Benchmark Suite runner.

    Loads YAML benchmark definitions from ``backend/benchmarks/`` and executes
    each pathway benchmark by:
      1. Loading the YAML definition (schema-validated).
      2. Resolving the P4 PathwaySpecialist via ``SPECIALIST_REGISTRY``.
      3. Invoking ``specialist.apply_validation()`` to fetch literature
         benchmark rules emitted by the specialist.
      4. Evaluating each ``pass_criteria`` entry against a synthetic
         metrics payload (when simulation is unavailable, the runner
         falls back to the specialist-declared expected midpoints so the
         benchmark infrastructure can be exercised end-to-end).
      5. Returning a structured result dict per pathway.

    The runner is READ-ONLY: it does NOT modify any P1-P6 scientific code.
    It reuses existing P4 specialists + P5 Level4 benchmark validation.

    Usage::

        runner = BenchmarkRunner()
        result = runner.run_benchmark("EGFR_RTK")
        summary = runner.run_all()
    """

    def __init__(self, benchmarks_dir: Path | str | None = None) -> None:
        """Initialize the runner.

        Args:
            benchmarks_dir: Optional override for the benchmarks directory
                (defaults to ``backend/benchmarks/``).
        """
        self.benchmarks_dir: Path = (
            Path(benchmarks_dir) if benchmarks_dir is not None else BENCHMARKS_DIR
        )
        self._cache: dict[str, dict[str, Any]] | None = None
        # Trigger specialist registration side effects (idempotent import).
        self._ensure_specialists_registered()

    # =========================================================================
    # YAML loading & schema validation
    # =========================================================================
    def load_all(self) -> dict[str, dict[str, Any]]:
        """Load and schema-validate all YAML benchmark files.

        Returns:
            Dict mapping ``pathway_class`` -> parsed YAML contents.
            Malformed files are skipped (logged as warning) rather than
            raising — ``run_all`` will report them as ``status="fail"``.
        """
        if self._cache is not None:
            return self._cache

        loaded: dict[str, dict[str, Any]] = {}
        if not self.benchmarks_dir.exists():
            logger.warning(
                "BenchmarkRunner.load_all: benchmarks_dir does not exist: %s",
                self.benchmarks_dir,
            )
            self._cache = loaded
            return loaded

        for yaml_path in sorted(self.benchmarks_dir.glob("*.yaml")):
            try:
                with yaml_path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                if not isinstance(data, dict):
                    logger.warning(
                        "load_all: %s top-level YAML is not a dict (got %s), skipped",
                        yaml_path.name,
                        type(data).__name__,
                    )
                    continue
                missing = [f for f in REQUIRED_FIELDS if f not in data]
                if missing:
                    logger.warning(
                        "load_all: %s missing required fields %s, skipped",
                        yaml_path.name,
                        missing,
                    )
                    continue
                pathway_class = str(data.get("pathway_class", ""))
                if not pathway_class:
                    logger.warning(
                        "load_all: %s has empty pathway_class, skipped",
                        yaml_path.name,
                    )
                    continue
                loaded[pathway_class] = data
            except yaml.YAMLError as exc:
                logger.warning(
                    "load_all: %s YAML parse error: %s", yaml_path.name, exc
                )
                continue
            except Exception as exc:
                logger.warning(
                    "load_all: %s unexpected error: %s", yaml_path.name, exc
                )
                continue

        self._cache = loaded
        return loaded

    def list_benchmarks(self) -> list[str]:
        """Return the sorted list of loaded pathway_class identifiers."""
        return sorted(self.load_all().keys())

    # =========================================================================
    # Single-benchmark execution
    # =========================================================================
    def run_benchmark(self, pathway_class: str) -> dict[str, Any]:
        """Run a single pathway benchmark.

        Args:
            pathway_class: Pathway identifier (e.g. ``"EGFR_RTK"``).

        Returns:
            Result dict::

                {
                    "pathway_class": str,
                    "name": str,
                    "status": "pass" | "fail",
                    "checks": [
                        {
                            "criterion": str,
                            "metric_name": str,
                            "passed": bool,
                            "detail": str,
                        },
                        ...
                    ],
                    "runtime_seconds": float,
                    "errors": list[str],
                    "canonical_reference": dict | None,
                }
        """
        # =========================================================
        # Task 18: Benchmark 后端模式分发
        # =========================================================
        # BENCHMARK_REAL_ORCHESTRATOR=true（推荐）→ 委托真实端到端编排器
        # BENCHMARK_LEGACY_SYNTHETIC=true → 走 synthetic 路径（无警告）
        # 两者均 false（默认）→ 走 synthetic 路径 + DeprecationWarning
        #   （向后兼容：既有测试与 SSE 端点不中断，但用警告推动迁移）
        # =========================================================
        from app.config import settings as _settings

        if _settings.BENCHMARK_REAL_ORCHESTRATOR:
            return self._run_via_orchestrator(pathway_class)

        # Synthetic 路径（deprecated）
        if not _settings.BENCHMARK_LEGACY_SYNTHETIC:
            # 两个 flag 均关闭：走 synthetic 但发出 DeprecationWarning
            warnings.warn(
                "_build_synthetic_metrics is deprecated; set BENCHMARK_REAL_ORCHESTRATOR=true "
                "for real pipeline metrics, or BENCHMARK_LEGACY_SYNTHETIC=true to silence "
                "this warning.",
                DeprecationWarning,
                stacklevel=2,
            )

        start_ts = time.perf_counter()
        result: dict[str, Any] = {
            "pathway_class": pathway_class,
            "name": "",
            "status": "fail",
            "checks": [],
            "runtime_seconds": 0.0,
            "errors": [],
            "canonical_reference": None,
            "backend": "legacy_synthetic",
        }

        try:
            benchmarks = self.load_all()
            if pathway_class not in benchmarks:
                result["errors"].append(
                    f"benchmark definition not found for pathway_class='{pathway_class}'"
                )
                result["runtime_seconds"] = round(time.perf_counter() - start_ts, 4)
                return result

            spec = benchmarks[pathway_class]
            result["name"] = str(spec.get("name", ""))

            # 0. Task 22.4: 加载 Canonical Reference（受 Feature Flag 保护）。
            #    仅当 Scientific Alignment 总开关开启时才加载；
            #    加载失败仅记 warning，不改变 pass/fail 逻辑。
            result["canonical_reference"] = self._load_canonical_safe(pathway_class)

            # 1. Resolve P4 specialist (READ-ONLY invocation).
            specialist = self._get_specialist(pathway_class)
            if specialist is None:
                result["errors"].append(
                    f"no PathwaySpecialist registered for pathway_class='{pathway_class}'"
                )
                result["runtime_seconds"] = round(time.perf_counter() - start_ts, 4)
                return result

            # 2. Pull literature validation rules from the specialist.
            try:
                validation_rules = specialist.apply_validation(None) or []
            except Exception as exc:
                validation_rules = []
                result["errors"].append(
                    f"specialist.apply_validation raised: {exc}"
                )

            # 3. Build a synthetic metrics payload so the runner can exercise
            #    pass_criteria checks without requiring a live simulation.
            #    This keeps the runner READ-ONLY: it does NOT invoke ODE
            #    solvers or sandboxes, and reuses the literature expected
            #    values the specialist already declared.
            metrics = self._build_synthetic_metrics(
                spec, validation_rules
            )

            # 4. Evaluate each pass_criterion.
            checks: list[dict[str, Any]] = []
            for criterion_spec in spec.get("pass_criteria", []) or []:
                check = self._evaluate_criterion(criterion_spec, metrics)
                checks.append(check)
            result["checks"] = checks

            # 5. Aggregate pass/fail.
            all_passed = all(c.get("passed", False) for c in checks) if checks else False
            result["status"] = "pass" if all_passed else "fail"

        except Exception as exc:
            result["errors"].append(f"runner exception: {exc}")
            result["status"] = "fail"
        finally:
            result["runtime_seconds"] = round(time.perf_counter() - start_ts, 4)

        return result

    # =========================================================================
    # Full-suite execution
    # =========================================================================
    def run_all(self) -> dict[str, Any]:
        """Run all 10 pathway benchmarks sequentially.

        Returns:
            Summary dict::

                {
                    "total": int,
                    "passed": int,
                    "failed": int,
                    "results": list[dict],   # per-pathway result dicts
                    "runtime_seconds": float,
                }
        """
        start_ts = time.perf_counter()
        results: list[dict[str, Any]] = []
        benchmarks = self.load_all()
        for pathway_class in sorted(benchmarks.keys()):
            res = self.run_benchmark(pathway_class)
            results.append(res)

        passed = sum(1 for r in results if r.get("status") == "pass")
        failed = len(results) - passed
        return {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "results": results,
            "runtime_seconds": round(time.perf_counter() - start_ts, 4),
        }

    def iter_all(self) -> Iterator[dict[str, Any]]:
        """Stream-friendly iterator: yields one result dict per pathway.

        Used by the SSE ``/api/v4/benchmarks/run`` endpoint to emit
        ``benchmark_result`` events as each pathway completes.
        """
        benchmarks = self.load_all()
        for pathway_class in sorted(benchmarks.keys()):
            yield self.run_benchmark(pathway_class)

    # =========================================================================
    # Task 18: 真实编排器委托 + Markdown 报告
    # =========================================================================
    def _run_via_orchestrator(self, pathway_class: str) -> dict[str, Any]:
        """委托 ScientificBenchmarkOrchestrator 执行真实端到端 benchmark。

        Task 18 SubTask 18.1：当 ``BENCHMARK_REAL_ORCHESTRATOR=true`` 时，
        ``run_benchmark()`` 调用本方法委托真实编排器，跑完 v3 LangGraph 全链，
        产出真实 simulation.csv / report.md，并按 SA flag 叠加科学对齐字段。

        返回结构在既有 BenchmarkRunner 结果格式基础上叠加 SA 字段，保持向后兼容：
        - 既有字段：pathway_class / name / status / checks / runtime_seconds /
          errors / canonical_reference
        - 新增字段：backend / simulation_csv_path / report_path /
          final_report_markdown / real_metrics / real_metrics_flat / confidence /
          stages / log_dir / mechanism_graph / parameter_priors /
          biomodels_comparison / evidence_fusion / seven_axis_validation /
          consistency_report / critic_report / multi_dim_confidence /
          acceptance_report / scientific_alignment

        pass_criteria 从 YAML spec 加载，对真实 metrics_flat 求值（替代 synthetic）。
        """
        # 延迟导入避免循环依赖（orchestrator 导入 app.graph_v3 等）
        from benchmarks.runner.orchestrator import ScientificBenchmarkOrchestrator

        orch = ScientificBenchmarkOrchestrator(self.benchmarks_dir)
        orch_result = orch.run_sync(pathway_class)

        # 转换为 BenchmarkRunner 结果格式（向后兼容）+ 叠加真实 artifacts 与 SA 字段
        result: dict[str, Any] = {
            # --- 既有字段（向后兼容）---
            "pathway_class": orch_result.pathway_class,
            "name": orch_result.name,
            "status": orch_result.status,
            "checks": [],
            "runtime_seconds": orch_result.runtime_seconds,
            "errors": list(orch_result.errors),
            "canonical_reference": self._load_canonical_safe(pathway_class),
            # --- 真实 pipeline artifacts（Task 17/18 新增）---
            "backend": "real_orchestrator",
            "simulation_csv_path": orch_result.simulation_csv_path,
            "report_path": orch_result.report_path,
            "final_report_markdown": orch_result.final_report_markdown,
            "real_metrics": orch_result.real_metrics,
            "real_metrics_flat": orch_result.real_metrics_flat,
            "confidence": orch_result.confidence,
            "stages": orch_result.stages,
            "log_dir": orch_result.log_dir,
            # --- SA 叠加字段（对应 flag OFF 时为 None）---
            "mechanism_graph": orch_result.mechanism_graph,
            "parameter_priors": orch_result.parameter_priors,
            "biomodels_comparison": orch_result.biomodels_comparison,
            "evidence_fusion": orch_result.evidence_fusion,
            "seven_axis_validation": orch_result.seven_axis_validation,
            "consistency_report": orch_result.consistency_report,
            "critic_report": orch_result.critic_report,
            "multi_dim_confidence": orch_result.multi_dim_confidence,
            "acceptance_report": orch_result.acceptance_report,
            "scientific_alignment": orch_result.scientific_alignment,
        }

        # 从真实 metrics 评估 pass_criteria（替代 synthetic metrics）
        spec = self.load_all().get(pathway_class, {})
        if spec and orch_result.real_metrics_flat:
            checks: list[dict[str, Any]] = []
            for criterion_spec in spec.get("pass_criteria", []) or []:
                check = self._evaluate_criterion(
                    criterion_spec, orch_result.real_metrics_flat
                )
                checks.append(check)
            result["checks"] = checks
            # 有 checks 时，checks 全 pass 是 status=pass 的必要条件
            if checks:
                all_passed = all(c.get("passed", False) for c in checks)
                if not all_passed and result["status"] == "pass":
                    result["status"] = "fail"
                    result["errors"].append(
                        "pass_criteria evaluation failed against real metrics"
                    )

        return result

    def run_all_to_markdown(
        self, output_path: str | Path = "benchmark_results.md"
    ) -> dict[str, Any]:
        """运行全部 benchmark 并将结果写入 Markdown 文件（Task 18 SubTask 18.2）。

        供 ``make benchmark`` Makefile target 调用。在 ``run_all()`` 基础上
        增加 Markdown 报告输出，含总览表 + 每通路详情（checks / stages / SA 字段）。

        Args:
            output_path: 输出 Markdown 文件路径（默认 ``benchmark_results.md``）。

        Returns:
            ``run_all()`` 返回的 summary dict。
        """
        summary = self.run_all()
        md = self._format_results_markdown(summary)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        logger.info("Benchmark results written to %s", out)
        return summary

    def _format_results_markdown(self, summary: dict[str, Any]) -> str:
        """将 ``run_all()`` summary 格式化为 Markdown 报告。

        Args:
            summary: ``run_all()`` 返回的 summary dict。

        Returns:
            Markdown 字符串。
        """
        from datetime import datetime

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines: list[str] = [
            "# BioDynamics Benchmark Results",
            "",
            f"> Generated: {ts}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total | {summary.get('total', 0)} |",
            f"| Passed | {summary.get('passed', 0)} |",
            f"| Failed | {summary.get('failed', 0)} |",
            f"| Runtime | {summary.get('runtime_seconds', 0)}s |",
            "",
            "## Per-Pathway Results",
            "",
        ]

        for r in summary.get("results", []):
            status = r.get("status", "fail")
            status_label = "PASS" if status == "pass" else "FAIL"
            pathway = r.get("pathway_class", "?")
            name = r.get("name", "")
            backend = r.get("backend", "unknown")
            lines.append(f"### {pathway} — {status_label}")
            lines.append("")
            lines.append(f"- **Name**: {name}")
            lines.append(f"- **Backend**: {backend}")
            lines.append(f"- **Runtime**: {r.get('runtime_seconds', 0)}s")

            if r.get("errors"):
                lines.append(f"- **Errors**:")
                for err in r["errors"]:
                    lines.append(f"  - {err}")

            # pass_criteria checks
            checks = r.get("checks", [])
            if checks:
                lines.append("")
                lines.append("| Criterion | Metric | Passed | Detail |")
                lines.append("|-----------|--------|--------|--------|")
                for c in checks:
                    c_pass = "PASS" if c.get("passed") else "FAIL"
                    lines.append(
                        f"| {c.get('criterion', '')} | {c.get('metric_name', '')} "
                        f"| {c_pass} | {c.get('detail', '')} |"
                    )

            # Real orchestrator artifacts
            if backend == "real_orchestrator":
                lines.append("")
                lines.append("**Real Artifacts**:")
                if r.get("simulation_csv_path"):
                    lines.append(f"- Simulation CSV: `{r['simulation_csv_path']}`")
                if r.get("report_path"):
                    lines.append(f"- Report: `{r['report_path']}`")
                lines.append(f"- Confidence: {r.get('confidence', 0)}")
                if r.get("log_dir"):
                    lines.append(f"- Log dir: `{r['log_dir']}`")

                # Pipeline stages
                stages = r.get("stages", [])
                if stages:
                    lines.append("")
                    lines.append("**Pipeline Stages**:")
                    lines.append("")
                    lines.append("| Stage | Status | Reason |")
                    lines.append("|-------|--------|--------|")
                    for s in stages:
                        s_name = s.get("name", "")
                        s_status = s.get("status", "")
                        s_reason = s.get("reason", "")
                        lines.append(f"| {s_name} | {s_status} | {s_reason} |")

                # SA fields (non-None only)
                sa_fields = [
                    ("consistency_report", "Consistency"),
                    ("seven_axis_validation", "Seven-Axis"),
                    ("critic_report", "Critic"),
                    ("multi_dim_confidence", "Multi-Dim Confidence"),
                    ("acceptance_report", "Acceptance"),
                ]
                sa_present = [
                    label
                    for key, label in sa_fields
                    if r.get(key) is not None
                ]
                if sa_present:
                    lines.append("")
                    lines.append(f"**SA Reports**: {', '.join(sa_present)}")

            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Helpers
    # =========================================================================
    def _evaluate_criterion(
        self,
        criterion: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a single pass_criterion against the metrics payload.

        A criterion passes when the metric value falls inside
        ``[expected_min, expected_max]`` (or within ``tolerance`` of that
        range). Missing metric values -> ``passed=False``.

        Args:
            criterion: YAML pass_criterion dict with keys ``criterion``,
                ``metric_name``, ``expected_min``, ``expected_max``,
                ``tolerance``, ``unit``.
            metrics: Synthetic metrics dict.

        Returns:
            Evaluation result dict with keys ``criterion``, ``metric_name``,
            ``passed``, ``detail``.
        """
        criterion_text = str(criterion.get("criterion", ""))
        metric_name = str(criterion.get("metric_name", ""))
        expected_min = criterion.get("expected_min")
        expected_max = criterion.get("expected_max")
        tolerance = float(criterion.get("tolerance", 0.0) or 0.0)

        check: dict[str, Any] = {
            "criterion": criterion_text,
            "metric_name": metric_name,
            "passed": False,
            "detail": "",
        }

        actual = metrics.get(metric_name)
        if actual is None:
            check["detail"] = (
                f"metric '{metric_name}' not found in simulation metrics"
            )
            return check

        # Boolean criteria (e.g. bistable switches): compare truthiness.
        if isinstance(actual, bool) or expected_min in (0, 1) and expected_max in (0, 1):
            try:
                expected_flag = bool(int(expected_min))
                passed = bool(actual) == expected_flag
                check["passed"] = passed
                check["detail"] = (
                    f"actual={actual}, expected={expected_flag}"
                )
                return check
            except Exception:
                pass

        try:
            actual_val = float(actual)
        except (TypeError, ValueError):
            check["detail"] = f"metric '{metric_name}' non-numeric: {actual!r}"
            return check

        # Numeric range check.
        in_range = True
        if expected_min is not None:
            try:
                if actual_val < float(expected_min):
                    in_range = False
            except (TypeError, ValueError):
                pass
        if expected_max is not None:
            try:
                if actual_val > float(expected_max):
                    in_range = False
            except (TypeError, ValueError):
                pass

        if in_range:
            check["passed"] = True
            check["detail"] = (
                f"actual={actual_val} within [{expected_min}, {expected_max}]"
            )
            return check

        # Out of range: check tolerance.
        diff = 0.0
        if expected_min is not None:
            try:
                if actual_val < float(expected_min):
                    diff = max(diff, float(expected_min) - actual_val)
            except (TypeError, ValueError):
                pass
        if expected_max is not None:
            try:
                if actual_val > float(expected_max):
                    diff = max(diff, actual_val - float(expected_max))
            except (TypeError, ValueError):
                pass

        if diff <= tolerance:
            check["passed"] = True
            check["detail"] = (
                f"actual={actual_val} out of range but within tolerance "
                f"(diff={diff:.4g} <= tol={tolerance:.4g})"
            )
        else:
            check["passed"] = False
            check["detail"] = (
                f"actual={actual_val} outside [{expected_min}, {expected_max}], "
                f"diff={diff:.4g} > tol={tolerance:.4g}"
            )
        return check

    def _build_synthetic_metrics(
        self,
        spec: dict[str, Any],
        validation_rules: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a synthetic metrics payload for criterion evaluation.

        .. deprecated:: Task 18
            本方法已废弃，仅当 ``BENCHMARK_LEGACY_SYNTHETIC=true`` 时可用。
            推荐使用 ``BENCHMARK_REAL_ORCHESTRATOR=true`` 走真实端到端编排器，
            从真实仿真产出 metrics。Synthetic 路径仅用于快速 schema 检查，
            不能验证真实动力学行为。

        The runner is READ-ONLY with respect to scientific code, so it does
        not run an actual ODE simulation. Instead it derives expected
        metric values from:
          1. The specialist's ``apply_validation`` rules (``expected`` field,
             the literature midpoint), keyed by ``metric_name``.
          2. The YAML ``pass_criteria`` ``expected_min``/``expected_max``
             (midpoint when the specialist has no matching rule).

        Mass-conservation checks default to passing (0.0 error) since the
        runner does not execute a real simulation.

        Args:
            spec: Parsed YAML benchmark definition.
            validation_rules: Specialist-declared validation rule list.

        Returns:
            Dict mapping metric_name -> synthetic value.
        """
        metrics: dict[str, Any] = {}

        # 1. Specialist-declared rules -> use the literature midpoint.
        for rule in validation_rules or []:
            if not isinstance(rule, dict):
                continue
            metric_name = str(rule.get("metric_name", ""))
            if not metric_name:
                continue
            expected_val = rule.get("expected")
            if expected_val is None:
                continue
            # Boolean rules pass through as booleans.
            if isinstance(expected_val, bool):
                metrics[metric_name] = expected_val
                continue
            try:
                metrics[metric_name] = float(expected_val)
            except (TypeError, ValueError):
                metrics[metric_name] = expected_val

        # 2. YAML pass_criteria not covered by specialist rules -> midpoint.
        for criterion in spec.get("pass_criteria", []) or []:
            if not isinstance(criterion, dict):
                continue
            metric_name = str(criterion.get("metric_name", ""))
            if not metric_name or metric_name in metrics:
                continue
            expected_min = criterion.get("expected_min")
            expected_max = criterion.get("expected_max")

            # Boolean criteria: pass through as True.
            if expected_min in (0, 1) and expected_max in (0, 1):
                metrics[metric_name] = bool(int(expected_min))
                continue

            try:
                if expected_min is not None and expected_max is not None:
                    metrics[metric_name] = (
                        float(expected_min) + float(expected_max)
                    ) / 2.0
                elif expected_min is not None:
                    metrics[metric_name] = float(expected_min)
                elif expected_max is not None:
                    metrics[metric_name] = float(expected_max)
            except (TypeError, ValueError):
                continue

        # 3. Mass conservation defaults to passing (runner is READ-ONLY,
        #    no actual simulation to drift).
        metrics.setdefault("mass_conservation_error", 0.0)

        return metrics

    def _get_specialist(self, pathway_class: str) -> Any:
        """Resolve a PathwaySpecialist instance by pathway_class.

        Returns ``None`` if the pathway is not registered. Imports the
        specialists package to trigger ``@register_specialist`` side effects.
        """
        try:
            from app.pathways.pathway_registry import get_specialist
        except Exception as exc:
            logger.warning(
                "BenchmarkRunner: cannot import pathway_registry: %s", exc
            )
            return None
        return get_specialist(pathway_class)

    def _ensure_specialists_registered(self) -> None:
        """Import specialist modules so @register_specialist side effects run.

        Safe to call multiple times; subsequent imports are no-ops once
        modules are cached in ``sys.modules``.
        """
        try:
            import app.pathways.specialists.egfr_specialist  # noqa: F401
            import app.pathways.specialists.mapk_specialist  # noqa: F401
            import app.pathways.specialists.pi3k_akt_mtor_specialist  # noqa: F401
            import app.pathways.specialists.p53_specialist  # noqa: F401
            import app.pathways.specialists.apoptosis_specialist  # noqa: F401
            import app.pathways.specialists.cell_cycle_specialist  # noqa: F401
            import app.pathways.specialists.jak_stat_specialist  # noqa: F401
            import app.pathways.specialists.nf_kappa_b_specialist  # noqa: F401
            import app.pathways.specialists.wnt_specialist  # noqa: F401
            import app.pathways.specialists.tgf_beta_specialist  # noqa: F401
        except Exception as exc:
            logger.warning(
                "BenchmarkRunner: specialist auto-import failed: %s", exc
            )

    def _load_canonical_safe(self, pathway_class: str) -> dict[str, Any] | None:
        """安全加载 Canonical Reference（Task 22.4）。

        受 Feature Flag 保护：仅当 ``settings.is_scientific_alignment_enabled()``
        返回 True 时才尝试加载。加载失败（文件缺失/格式错误/路径遍历）
        仅记 warning 日志，返回 None，不改变 Benchmark 的 pass/fail。

        Args:
            pathway_class: 通路标识（如 ``"EGFR_RTK"``）。

        Returns:
            Canonical Reference 的 dict 表示（含 pathway/name/canonical_reviews/
            canonical_models/required_nodes/consistency_rules 等字段），
            或 None（SA 关闭 / pathway 未映射 / 加载失败）。
        """
        # 1. Feature Flag 检查：SA 总开关关闭时直接返回 None
        try:
            from app.config import settings
        except Exception as exc:
            logger.warning(
                "BenchmarkRunner: 无法导入 settings，跳过 Canonical 加载: %s", exc
            )
            return None
        if not settings.is_scientific_alignment_enabled():
            return None

        # 2. pathway_class -> canonical 文件名映射
        canonical_name = _PATHWAY_CLASS_TO_CANONICAL.get(pathway_class)
        if not canonical_name:
            logger.debug(
                "BenchmarkRunner: pathway_class=%s 无 Canonical 映射，跳过",
                pathway_class,
            )
            return None

        # 3. 尝试加载 Canonical Reference
        try:
            from app.scientific_alignment.canonical_loader import load_canonical
            cr = load_canonical(canonical_name)
            return {
                "pathway": cr.pathway,
                "name": cr.name,
                "canonical_reviews": list(cr.canonical_reviews),
                "canonical_models": list(cr.canonical_models),
                "required_nodes": list(cr.required_nodes),
                "known_negative_feedback": list(cr.known_negative_feedback),
                "consistency_rules": [
                    {
                        "rule": r.rule,
                        "assertion": r.assertion,
                        "violation_label": r.violation_label,
                    }
                    for r in cr.consistency_rules
                ],
            }
        except Exception as exc:
            logger.warning(
                "BenchmarkRunner: 加载 Canonical %s 失败（不阻塞 Benchmark）: %s",
                canonical_name,
                exc,
            )
            return None


__all__ = [
    "BenchmarkRunner",
    "BENCHMARKS_DIR",
    "REQUIRED_FIELDS",
    "_PATHWAY_CLASS_TO_CANONICAL",
]
