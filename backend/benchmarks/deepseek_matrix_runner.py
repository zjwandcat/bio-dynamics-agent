"""Run all Benchmark_QA_Collection cases through DeepSeek Flash and Pro.

The parent process launches one isolated Python worker per model because the
application constructs its global LLM at import time.  Each worker runs cases
concurrently, checkpoints every result, and can resume an interrupted matrix.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values


BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BACKEND_DIR / ".env"
DEFAULT_OUTPUT_ROOT = BACKEND_DIR / "data" / "benchmark_runs" / "deepseek_43_case"
PROFILES = {
    "flash": {"model": "deepseek-v4-flash", "default_concurrency": 4},
    "pro": {"model": "deepseek-v4-pro", "default_concurrency": 4},
}
PATHWAY_MAP = {
    "EGFR": "EGFR_RTK",
    "MAPK": "MAPK_ERK",
    "PI3K": "PI3K_AKT_mTOR",
    "p53": "p53_signaling",
    "CellCycle": "Cell_Cycle",
    "Apoptosis": "Apoptosis",
    "NFkB": "NF_kB",
    "JAK_STAT": "JAK_STAT",
    "Wnt": "Wnt",
    "TGF_beta": "TGF_beta",
    "CrossPathway": "CROSS_PATHWAY",
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def _profile_environment(profile: str, chroma_dir: Path | None = None) -> dict[str, str]:
    values = dotenv_values(ENV_PATH)
    # [r22] 优先使用 BACKUP2_*（DeepSeek 官方 API https://api.deepseek.com）凭据，
    #   避免误用 OpenRouter 端点调用 deepseek-v4-flash（OpenRouter 不支持该模型）。
    #   回退顺序：BACKUP2_API_KEY → BACKUP_API_KEY
    key = str(values.get("BACKUP2_API_KEY") or values.get("BACKUP_API_KEY") or "")
    base_url = str(values.get("BACKUP2_BASE_URL") or values.get("BACKUP_BASE_URL") or "https://api.deepseek.com")
    if not key:
        raise RuntimeError("BACKUP2_API_KEY/BACKUP_API_KEY is not configured in backend/.env")
    env = dict(os.environ)
    env.update(
        {
            "OPENAI_API_KEY": key,
            "OPENAI_BASE_URL": base_url,
            "OPENAI_MODEL": PROFILES[profile]["model"],
            "BACKUP_API_KEY": "",
            "BACKUP_BASE_URL": "",
            "BACKUP_MODEL": "",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if chroma_dir is not None:
        env["CHROMA_PERSIST_DIR"] = str(chroma_dir.resolve())
    return env


def _prepare_chroma_snapshot(profile: str, output: Path) -> Path:
    """Create a per-model Chroma snapshot before either worker opens SQLite."""
    values = dotenv_values(ENV_PATH)
    configured = str(values.get("CHROMA_PERSIST_DIR") or "").strip()
    source = Path(configured) if configured else BACKEND_DIR / "data" / "vector_db"
    if not source.is_absolute():
        source = (BACKEND_DIR / source).resolve()
    if not source.is_dir():
        raise RuntimeError(f"Chroma source directory does not exist: {source}")
    destination = output / "runtime" / profile / "vector_db"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def _question(case: dict[str, Any]) -> str:
    return (
        "Run a complete quantitative BioDynamics analysis for this benchmark task. "
        "Build the mechanism and ODE model, simulate the requested dynamics, validate "
        "against BioModels and literature, propose experiments, and produce a cited report. "
        "Do not invent citations or silently replace missing evidence.\n\n"
        f"Task: {case['title']}\n"
        f"Scientific objective: {case['scientific_objective']}"
    )


def _spec(case: dict[str, Any], pathway_class: str) -> dict[str, Any]:
    biomodels_ids = [
        str(item.get("id"))
        for item in case.get("expected_biomodels", [])
        if isinstance(item, dict) and item.get("id")
    ]
    # [P2-3 修复] 传递 canonical_required_pmids 到 spec（与 nvidia_matrix_runner 一致）
    canonical_pmids = [
        str(item.get("pmid", "")).strip()
        for item in case.get("expected_literature", {}).get("canonical_required", [])
        if isinstance(item, dict) and item.get("pmid")
    ]
    return {
        "name": f"QA {case['case_id']} {case['title']}",
        "pathway_class": pathway_class,
        "input": {
            "hypothesis": _question(case),
            "biomodels_ids": biomodels_ids,
            "track_a_semantics": case.get("track_a_semantics", "standard"),
            "canonical_required_pmids": canonical_pmids,  # [P2-3]
        },
        "pass_criteria": [],
    }


def _agent_payload(result: Any, profile: str, case_id: str) -> dict[str, Any]:
    raw = asdict(result)
    return {
        "case_id": case_id,
        "profile": profile,
        "model": PROFILES[profile]["model"],
        "pipeline_status": raw.get("status"),
        "runtime_seconds": raw.get("runtime_seconds"),
        "errors": raw.get("errors", []),
        "stages": raw.get("stages", []),
        "simulation_csv_path": raw.get("simulation_csv_path", ""),
        "report_path": raw.get("report_path", ""),
        "final_report_markdown": raw.get("final_report_markdown", ""),
        "real_metrics": raw.get("real_metrics", {}),
        "real_metrics_flat": raw.get("real_metrics_flat", {}),
        "confidence": raw.get("confidence", 0.0),
        "mechanism_graph": raw.get("mechanism_graph"),
        "canonical_reference": raw.get("canonical_reference"),
        "biomodels_comparison": raw.get("biomodels_comparison"),
        "evidence_fusion": raw.get("evidence_fusion"),
        "seven_axis_validation": raw.get("seven_axis_validation"),
        "acceptance_report": raw.get("acceptance_report"),
        "log_dir": raw.get("log_dir", ""),
    }


def _is_transient(message: str) -> bool:
    value = message.lower()
    return any(token in value for token in ("429", "rate limit", "timeout", "timed out", "connection", "temporar", "502", "503", "504"))


async def _run_one(
    case: dict[str, Any],
    profile: str,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    timeout_seconds: float,
    max_attempts: int,
) -> dict[str, Any]:
    from benchmarks.agent_case_evaluator import evaluate_agent_case
    from benchmarks.runner.orchestrator import ScientificBenchmarkOrchestrator

    case_id = str(case["case_id"])
    target = output_dir / "cases" / f"{case_id}.json"
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
            errors = " ".join(str(item) for item in existing.get("agent", {}).get("errors", []))
            if existing.get("completed") and "UnicodeDecodeError" not in errors:
                print(f"[{profile}] {case_id} resume: already complete", flush=True)
                return existing
        except (OSError, json.JSONDecodeError):
            pass

    pathway_class = PATHWAY_MAP[str(case["pathway"])]
    last_error = ""
    attempt = 0
    for attempt in range(1, max_attempts + 1):
        async with semaphore:
            started = time.perf_counter()
            case_spec_dir = output_dir / "specs" / case_id
            case_spec_dir.mkdir(parents=True, exist_ok=True)
            (case_spec_dir / "case.yaml").write_text(
                yaml.safe_dump(_spec(case, pathway_class), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            print(f"[{profile}] {case_id} attempt={attempt} start", flush=True)
            try:
                orchestrator = ScientificBenchmarkOrchestrator(benchmarks_dir=case_spec_dir)
                result = await asyncio.wait_for(orchestrator.run(pathway_class), timeout=timeout_seconds)
                agent = _agent_payload(result, profile, case_id)
                evaluation = evaluate_agent_case(case, agent)
                payload = {
                    "completed": True,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "attempt": attempt,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "agent": agent,
                    "evaluation": evaluation,
                }
                _atomic_json(target, payload)
                print(
                    f"[{profile}] {case_id} done operational={evaluation['operational']} "
                    f"scientific_pass={evaluation['scientific_pass']} elapsed={payload['elapsed_seconds']}s",
                    flush=True,
                )
                return payload
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                elapsed = round(time.perf_counter() - started, 3)
                print(f"[{profile}] {case_id} attempt={attempt} error={last_error} elapsed={elapsed}s", flush=True)
        if attempt < max_attempts and _is_transient(last_error):
            await asyncio.sleep(min(60.0, 5.0 * (2 ** (attempt - 1))))
            continue
        break

    payload = {
        "completed": True,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "elapsed_seconds": None,
        "agent": {"case_id": case_id, "profile": profile, "model": PROFILES[profile]["model"], "pipeline_status": "error", "errors": [last_error]},
        "evaluation": {
            "case_id": case_id,
            "pathway": case["pathway"],
            "difficulty": case["difficulty"],
            "weight": int(case["weight"]),
            "operational": False,
            "scientific_pass": False,
            "criteria_pass_rate": 0.0,
            "critical": [],
            "optional": [],
            "failed_critical": list(case["critical_criteria"]),
            "failed_optional": list(case["optional_criteria"]),
            "failure_diagnosis": case.get("failure_diagnosis", []),
            "simulation_species": [],
        },
    }
    _atomic_json(target, payload)
    return payload


def _summary(profile: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluations = [item["evaluation"] for item in results]
    total_weight = sum(item["weight"] for item in evaluations)
    passed_weight = sum(item["weight"] for item in evaluations if item["scientific_pass"])
    criterion_totals = {f"C{i}": {"passed": 0, "total": 0} for i in range(1, 13)}
    for item in evaluations:
        for criterion in item.get("critical", []) + item.get("optional", []):
            bucket = criterion_totals[criterion["criterion"]]
            bucket["total"] += 1
            bucket["passed"] += int(criterion["passed"])
    return {
        "profile": profile,
        "model": PROFILES[profile]["model"],
        "total": len(evaluations),
        "operational": sum(item["operational"] for item in evaluations),
        "scientific_passed": sum(item["scientific_pass"] for item in evaluations),
        "scientific_failed": sum(not item["scientific_pass"] for item in evaluations),
        "total_weight": total_weight,
        "passed_weight": passed_weight,
        "scientific_score": round(100.0 * passed_weight / total_weight, 2) if total_weight else 0.0,
        "criterion_totals": criterion_totals,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def _worker(args: argparse.Namespace) -> int:
    from benchmarks.qa_runner import load_cases

    cases = load_cases()
    selected = [cases[case_id] for case_id in args.case_ids] if args.case_ids else list(cases.values())
    output_dir = Path(args.output).resolve() / args.profile
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "profile": args.profile,
        "model": PROFILES[args.profile]["model"],
        "case_count": len(selected),
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output_dir / "metadata.json", metadata)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(_run_one(case, args.profile, output_dir, semaphore, args.timeout, args.max_attempts))
        for case in selected
    ]
    results = await asyncio.gather(*tasks)
    _atomic_json(output_dir / "summary.json", _summary(args.profile, results))
    return 0


def _parent(args: argparse.Namespace) -> int:
    profiles = args.profiles or ["flash", "pro"]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    for profile in profiles:
        concurrency = args.concurrency_flash if profile == "flash" else args.concurrency_pro
        chroma_dir = _prepare_chroma_snapshot(profile, output)
        command = [
            sys.executable,
            "-m",
            "benchmarks.deepseek_matrix_runner",
            "--worker",
            "--profile",
            profile,
            "--output",
            str(output),
            "--concurrency",
            str(concurrency),
            "--timeout",
            str(args.timeout),
            "--max-attempts",
            str(args.max_attempts),
        ]
        for case_id in args.case_ids:
            command.extend(("--case-id", case_id))
        process = subprocess.Popen(
            command,
            cwd=BACKEND_DIR,
            env=_profile_environment(profile, chroma_dir),
            text=True,
        )
        processes.append((profile, process))
    failed = 0
    for profile, process in processes:
        code = process.wait()
        print(f"[{profile}] worker exit={code}", flush=True)
        failed += int(code != 0)
    return 1 if failed else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", choices=sorted(PROFILES), help=argparse.SUPPRESS)
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--case-id", dest="case_ids", action="append", default=[])
    parser.add_argument("--concurrency-flash", type=int, default=4)
    parser.add_argument("--concurrency-pro", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        if not args.profile:
            raise SystemExit("--profile is required with --worker")
        return asyncio.run(_worker(args))
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
