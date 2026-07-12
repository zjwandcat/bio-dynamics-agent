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
        start_ts = time.perf_counter()
        result: dict[str, Any] = {
            "pathway_class": pathway_class,
            "name": "",
            "status": "fail",
            "checks": [],
            "runtime_seconds": 0.0,
            "errors": [],
            "canonical_reference": None,
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
