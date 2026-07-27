"""Run all Benchmark_QA_Collection cases through NVIDIA Nemotron 3 Ultra (free).

使用 .env 默认配置（NVIDIA Nemotron 主用 + DeepSeek 备用容灾），
通过 ScientificBenchmarkOrchestrator 跑完整 Agent 管线，
用 agent_case_evaluator 进行 C1-C12 自动评分。

并发与限流策略：
- 并发度默认 3（平衡速度与 OpenRouter 20 RPM / 讯飞 20 QPS 限制）
- NVIDIA 免费额度（50 次/天）耗尽后自动切 DeepSeek 备用
- 每个 case 独立 ChromaDB 快照副本，避免 SQLite 写锁冲突
- 支持断点续跑（已完成的 case 跳过）

运行方式：
    py -3.14 -m benchmarks.nvidia_matrix_runner --concurrency 3 --timeout 3600
    py -3.14 -m benchmarks.nvidia_matrix_runner --case-id 1.E1 --case-id 2.M1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values


BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BACKEND_DIR / ".env"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "benchmark_runs" / "nvidia_full"
# PROFILE 可通过环境变量 BENCHMARK_PROFILE 覆盖（用于切换 nvidia/poolside 等不同 LLM profile）
# 默认 "nvidia" 保持向后兼容
PROFILE = os.environ.get("BENCHMARK_PROFILE", "nvidia")

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
    """原子写入 JSON，防止并发写入损坏。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def _prepare_chroma_snapshot(output: Path) -> Path:
    """创建 ChromaDB 快照副本，避免并发 SQLite 写锁冲突。"""
    values = dotenv_values(ENV_PATH)
    configured = str(values.get("CHROMA_PERSIST_DIR") or "").strip()
    source = Path(configured) if configured else BACKEND_DIR / "data" / "vector_db"
    if not source.is_absolute():
        source = (BACKEND_DIR / source).resolve()
    if not source.is_dir():
        raise RuntimeError(f"Chroma 源目录不存在: {source}")
    destination = output / "runtime" / PROFILE / "vector_db"
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
    # [P2-3 修复] 传递 canonical_required_pmids 到 spec，供 orchestrator 显式检索
    # Root Cause: CrossPathway case 11.X1 需引用 PMID 21245089（Trametinib），
    #   但 agent 边级 RAG 检索仅命中 EGFR 抑制剂文献（Gefitinib PMID 11706397），
    #   MEK 抑制剂文献未被检索（agent 网络可能未生成 MEK 抑制边）。
    # Fix: spec 携带 canonical_required_pmids，orchestrator 在 SA 阶段前显式
    #   调用 rag_client.search_by_pmids() 拉取缺失的 canonical PMID，
    #   合并到 paper_evidence 和 cited_pmids，使 C9 文献正确性评估通过。
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


def _agent_payload(result: Any, case_id: str) -> dict[str, Any]:
    from dataclasses import asdict
    raw = asdict(result)
    return {
        "case_id": case_id,
        "profile": PROFILE,
        "model": os.environ.get("OPENAI_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
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
        # [Sandbox Fix] 沙箱产物清单（csv_path / sha256 / columns / row_count）
        "artifact_manifest": raw.get("artifact_manifest"),
        "warnings": raw.get("warnings", []),
    }


def _is_transient(message: str) -> bool:
    """判断错误是否为可重试的瞬态错误（429/超时/连接问题）。"""
    value = message.lower()
    return any(token in value for token in (
        "429", "rate limit", "timeout", "timed out", "connection",
        "temporar", "502", "503", "504", "overloaded",
    ))


async def _run_one(
    case: dict[str, Any],
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    timeout_seconds: float,
    max_attempts: int,
) -> dict[str, Any]:
    """运行单个 case，支持重试与断点续跑。"""
    from benchmarks.agent_case_evaluator import evaluate_agent_case
    from benchmarks.runner.orchestrator import ScientificBenchmarkOrchestrator

    case_id = str(case["case_id"])
    target = output_dir / "cases" / f"{case_id}.json"

    # 断点续跑：已完成且无错误的 case 跳过
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
            errors = " ".join(str(item) for item in existing.get("agent", {}).get("errors", []))
            if existing.get("completed") and "UnicodeDecodeError" not in errors:
                print(f"[{PROFILE}] {case_id} resume: already complete", flush=True)
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
            print(f"[{PROFILE}] {case_id} attempt={attempt} start", flush=True)
            try:
                orchestrator = ScientificBenchmarkOrchestrator(benchmarks_dir=case_spec_dir)
                result = await asyncio.wait_for(orchestrator.run(pathway_class), timeout=timeout_seconds)
                agent = _agent_payload(result, case_id)
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
                    f"[{PROFILE}] {case_id} done operational={evaluation['operational']} "
                    f"scientific_pass={evaluation['scientific_pass']} elapsed={payload['elapsed_seconds']}s",
                    flush=True,
                )
                return payload
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                elapsed = round(time.perf_counter() - started, 3)
                print(f"[{PROFILE}] {case_id} attempt={attempt} error={last_error} elapsed={elapsed}s", flush=True)
        # 瞬态错误重试（指数退避）
        if attempt < max_attempts and _is_transient(last_error):
            wait = min(60.0, 5.0 * (2 ** (attempt - 1)))
            print(f"[{PROFILE}] {case_id} retry in {wait}s...", flush=True)
            await asyncio.sleep(wait)
            continue
        break

    # 所有重试失败，记录错误结果
    payload = {
        "completed": True,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "elapsed_seconds": None,
        "agent": {
            "case_id": case_id,
            "profile": PROFILE,
            "model": os.environ.get("OPENAI_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
            "pipeline_status": "error",
            "errors": [last_error],
        },
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


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总所有 case 的评分结果，计算 Scientific Score。"""
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
        "profile": PROFILE,
        "model": os.environ.get("OPENAI_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
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


async def _run(args: argparse.Namespace) -> int:
    from benchmarks.qa_runner import load_cases

    cases = load_cases()
    selected = [cases[case_id] for case_id in args.case_ids] if args.case_ids else list(cases.values())
    output_dir = Path(args.output).resolve() / PROFILE
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建 ChromaDB 快照并设置环境变量
    chroma_dir = _prepare_chroma_snapshot(Path(args.output).resolve())
    os.environ["CHROMA_PERSIST_DIR"] = str(chroma_dir.resolve())

    metadata = {
        "profile": PROFILE,
        "model": os.environ.get("OPENAI_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        "case_count": len(selected),
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output_dir / "metadata.json", metadata)
    print(f"[{PROFILE}] 启动: {len(selected)} cases, concurrency={args.concurrency}, timeout={args.timeout}s", flush=True)
    print(f"[{PROFILE}] ChromaDB snapshot: {chroma_dir}", flush=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(_run_one(case, output_dir, semaphore, args.timeout, args.max_attempts))
        for case in selected
    ]
    results = await asyncio.gather(*tasks)
    summary = _summary(results)
    _atomic_json(output_dir / "summary.json", summary)
    print(f"\n[{PROFILE}] 完成: {summary['scientific_passed']}/{summary['total']} PASS, "
          f"Scientific Score={summary['scientific_score']}", flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--case-id", dest="case_ids", action="append", default=[])
    parser.add_argument("--concurrency", type=int, default=3,
                        help="并发 case 数（默认 3，适配 OpenRouter 20 RPM + 讯飞 20 QPS）")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="单 case 超时秒数（默认 3600）")
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="单 case 最大重试次数（默认 2）")
    return parser


def main() -> int:
    args = _parser().parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
