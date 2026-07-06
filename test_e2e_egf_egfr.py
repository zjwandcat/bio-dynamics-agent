"""端到端测试：EGF-EGFR 信号通路仿真，捕获完整 SSE 事件流并保存。

测试问题：
  表皮生长因子（EGF）结合 EGFR 受体后诱导其二聚化和自磷酸化，
  激活下游 Shc-Grb2-SOS-Ras-MAPK 信号级联。
  请基于 BIOMD0000000205 模型的参数，仿真 EGF 刺激下 EGFR 磷酸化的动力学过程。
  初始条件：EGF=0.008 nM，EGFR=0.3 nM。

标准答案（Schoeberl et al. 2002 Nature Biotechnology, BIOMD0000000205）：
  - pEGFR 在 5-10 分钟达峰
  - 信号放大效应：MAPK 活化显著放大
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import requests

BACKEND = "http://localhost:8000"
USER_INPUT = (
    "表皮生长因子（EGF）结合 EGFR 受体后诱导其二聚化和自磷酸化，"
    "激活下游 Shc-Grb2-SOS-Ras-MAPK 信号级联。"
    "请基于 BIOMD0000000205 模型的参数，仿真 EGF 刺激下 EGFR 磷酸化的动力学过程。"
    "初始条件：EGF=0.008 nM，EGFR=0.3 nM。"
)
OUT_DIR = Path(__file__).resolve().parent / "test_outputs_egf"
OUT_DIR.mkdir(exist_ok=True)


def run() -> None:
    thread_id = f"e2e-egf-{uuid.uuid4().hex[:8]}"
    payload: dict[str, Any] = {
        "user_input": USER_INPUT,
        "thread_id": thread_id,
        "mode": "auto_standard",
        "manual_modules": [],
    }

    print(f"[Test] thread_id={thread_id}")
    print(f"[Test] mode=auto_standard")
    print(f"[Test] 用户问题：{USER_INPUT}")
    print("-" * 80)

    events: list[dict[str, Any]] = []
    t0 = time.time()

    try:
        with requests.post(
            f"{BACKEND}/api/chat",
            json=payload,
            stream=True,
            timeout=600,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("data: "):
                    data_str = raw[6:]
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        data = {"_raw": data_str}
                    events.append({"t": round(time.time() - t0, 2), **data})
                    _print_event(data)
    except Exception as exc:
        print(f"[ERROR] 流式请求失败：{exc}")
        events.append({"t": round(time.time() - t0, 2), "event": "client_error", "data": str(exc)})

    elapsed = round(time.time() - t0, 2)
    print("-" * 80)
    print(f"[Test] 完成，共 {len(events)} 个事件，耗时 {elapsed}s")

    # 保存原始事件流
    raw_path = OUT_DIR / "sse_events.json"
    raw_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Test] 原始事件流已保存：{raw_path}")

    # 提取关键信息
    summary = _summarize(events)
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Test] 摘要已保存：{summary_path}")

    # 提取最终报告
    final_report_md = ""
    for ev in reversed(events):
        if ev.get("event") == "report" and isinstance(ev.get("data"), dict):
            final_report_md = ev["data"].get("markdown", "")
            break
        if ev.get("event") == "report_ready" and isinstance(ev.get("data"), str):
            final_report_md = ev["data"]
            break
    if final_report_md:
        report_path = OUT_DIR / "final_report.md"
        report_path.write_text(final_report_md, encoding="utf-8")
        print(f"[Test] 最终报告已保存：{report_path}")

    # 提取生成的 ODE 代码
    ode_code = ""
    for ev in events:
        if ev.get("event") == "code_generated" and isinstance(ev.get("data"), str):
            ode_code = ev["data"]
            break
    if ode_code:
        ode_path = OUT_DIR / "ode_code.py"
        ode_path.write_text(ode_code, encoding="utf-8")
        print(f"[Test] ODE 代码已保存：{ode_path}")


def _print_event(data: dict[str, Any]) -> None:
    event = data.get("event", "?")
    payload = data.get("data")
    if event == "node_start":
        print(f"  [{data.get('t', 0):>6}s] NODE_START: {payload}")
    elif event == "workflow_v3_state":
        if isinstance(payload, dict):
            print(f"  [{data.get('t', 0):>6}s] STATE: {payload.get('current_node')} / {payload.get('status')}")
    elif event == "agent_registry":
        agents = payload if isinstance(payload, list) else []
        names = [a.get("name", "?") for a in agents]
        print(f"  [{data.get('t', 0):>6}s] REGISTRY: {names}")
    elif event == "agent_dispatch":
        if isinstance(payload, dict):
            print(
                f"  [{data.get('t', 0):>6}s] DISPATCH: "
                f"{payload.get('target_agent')} -> {payload.get('status')}"
            )
    elif event == "mcp_tool_call":
        if isinstance(payload, dict):
            print(f"  [{data.get('t', 0):>6}s] MCP_TOOL: {payload.get('tool_name')} status={payload.get('status')}")
    elif event == "mcp_term_definitions":
        if isinstance(payload, dict):
            defs = payload.get("definitions", [])
            print(f"  [{data.get('t', 0):>6}s] MCP_DEFS: {len(defs)} terms, tokens_saved={payload.get('tokens_saved', 0)}")
    elif event == "rag_insights":
        if isinstance(payload, dict):
            print(
                f"  [{data.get('t', 0):>6}s] RAG_INSIGHTS: hit_rate={payload.get('hit_rate')}, "
                f"top={len(payload.get('top_selections', []))}, drugs={len(payload.get('drug_candidates', []))}"
            )
    elif event == "rag_ready":
        if isinstance(payload, dict):
            print(f"  [{data.get('t', 0):>6}s] RAG_READY: hit_rate={payload.get('hit_rate')}, fallback={payload.get('fallback')}")
    elif event == "rag_online_fallback":
        if isinstance(payload, dict):
            print(f"  [{data.get('t', 0):>6}s] RAG_ONLINE: triggered={payload.get('triggered')}")
    elif event == "pkpd_profile":
        print(f"  [{data.get('t', 0):>6}s] PKPD_PROFILE: {json.dumps(payload, ensure_ascii=False)[:200]}")
    elif event == "drug_regimen":
        print(f"  [{data.get('t', 0):>6}s] DRUG_REGIMEN: {len(payload) if isinstance(payload, list) else 1} items")
    elif event == "code_generated":
        if isinstance(payload, str):
            print(f"  [{data.get('t', 0):>6}s] CODE_GENERATED: {len(payload)} chars")
    elif event == "execution_log":
        if isinstance(payload, str):
            print(f"  [{data.get('t', 0):>6}s] LOG: {payload[:200]}")
    elif event == "image_ready":
        if isinstance(payload, str):
            print(f"  [{data.get('t', 0):>6}s] IMAGE_READY: {len(payload)} chars (base64)")
    elif event == "simulation_csv":
        print(f"  [{data.get('t', 0):>6}s] CSV: {payload}")
    elif event == "dose_response":
        print(f"  [{data.get('t', 0):>6}s] DOSE_RESPONSE: {json.dumps(payload, ensure_ascii=False)[:200]}")
    elif event == "metrics":
        print(f"  [{data.get('t', 0):>6}s] METRICS: {json.dumps(payload, ensure_ascii=False)[:200]}")
    elif event == "experiment_protocols":
        print(f"  [{data.get('t', 0):>6}s] EXPERIMENT_PROTOCOLS: {len(payload) if isinstance(payload, list) else 1} items")
    elif event == "paper_evidence":
        print(f"  [{data.get('t', 0):>6}s] PAPER_EVIDENCE: {len(payload) if isinstance(payload, list) else 1} items")
    elif event == "report":
        if isinstance(payload, dict):
            print(f"  [{data.get('t', 0):>6}s] REPORT: markdown={len(payload.get('markdown', ''))} chars")
    elif event == "report_ready":
        if isinstance(payload, str):
            print(f"  [{data.get('t', 0):>6}s] REPORT_READY: {len(payload)} chars")
    elif event == "clarification_needed":
        print(f"  [{data.get('t', 0):>6}s] CLARIFICATION_NEEDED: {json.dumps(payload, ensure_ascii=False)[:300]}")
    elif event == "clarification_resolved":
        print(f"  [{data.get('t', 0):>6}s] CLARIFICATION_RESOLVED")
    elif event == "config":
        if isinstance(payload, dict):
            print(f"  [{data.get('t', 0):>6}s] CONFIG: model={payload.get('model_name')}")
    elif event == "token_usage":
        if isinstance(payload, dict):
            print(f"  [{data.get('t', 0):>6}s] TOKEN_USAGE: {json.dumps(payload, ensure_ascii=False)}")
    elif event == "rule_violations":
        print(f"  [{data.get('t', 0):>6}s] RULE_VIOLATIONS: {json.dumps(payload, ensure_ascii=False)[:200]}")
    elif event == "knowledge_graph":
        if isinstance(payload, dict):
            print(f"  [{data.get('t', 0):>6}s] KG: nodes={payload.get('node_count')}, edges={payload.get('edge_count')}, acyclic={payload.get('is_acyclic')}")
    elif event == "error":
        print(f"  [{data.get('t', 0):>6}s] ERROR: {payload}")
    elif event == "end":
        print(f"  [{data.get('t', 0):>6}s] END")
    else:
        print(f"  [{data.get('t', 0):>6}s] {event}: {str(payload)[:200]}")


def _summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_events": len(events),
        "events_by_type": {},
        "node_order": [],
        "agent_dispatches": [],
        "mcp_tool_calls": [],
        "mcp_term_definitions_count": 0,
        "mcp_tokens_saved": 0,
        "rag_hit_rate": None,
        "rag_online_fallback_triggered": False,
        "rag_top_selections_count": 0,
        "rag_drug_candidates_count": 0,
        "pkpd_profile_present": False,
        "drug_regimen_count": 0,
        "ode_code_chars": 0,
        "image_ready": False,
        "simulation_csv": "",
        "dose_response_present": False,
        "ic50": None,
        "ic90": None,
        "hed": None,
        "metrics": None,
        "experiment_protocols_count": 0,
        "paper_evidence_count": 0,
        "report_chars": 0,
        "clarification_needed": False,
        "clarification_questions": [],
        "token_usage": None,
        "errors": [],
        "rule_violations": None,
        "kg_node_count": None,
        "kg_edge_count": None,
        "kg_acyclic": None,
    }

    for ev in events:
        event = ev.get("event", "?")
        payload = ev.get("data")
        summary["events_by_type"][event] = summary["events_by_type"].get(event, 0) + 1

        if event == "workflow_v3_state" and isinstance(payload, dict):
            if payload.get("status") == "running":
                summary["node_order"].append(payload.get("current_node"))
        elif event == "agent_dispatch" and isinstance(payload, dict):
            summary["agent_dispatches"].append({
                "t": ev.get("t"),
                "target_agent": payload.get("target_agent"),
                "status": payload.get("status"),
                "node_name": payload.get("node_name"),
            })
        elif event == "mcp_tool_call" and isinstance(payload, dict):
            summary["mcp_tool_calls"].append({
                "tool_name": payload.get("tool_name"),
                "status": payload.get("status"),
            })
        elif event == "mcp_term_definitions" and isinstance(payload, dict):
            summary["mcp_term_definitions_count"] = len(payload.get("definitions", []))
            summary["mcp_tokens_saved"] = payload.get("tokens_saved", 0)
        elif event == "rag_insights" and isinstance(payload, dict):
            summary["rag_hit_rate"] = payload.get("hit_rate")
            summary["rag_top_selections_count"] = len(payload.get("top_selections", []))
            summary["rag_drug_candidates_count"] = len(payload.get("drug_candidates", []))
        elif event == "rag_ready" and isinstance(payload, dict):
            if summary["rag_hit_rate"] is None:
                summary["rag_hit_rate"] = payload.get("hit_rate")
        elif event == "rag_online_fallback" and isinstance(payload, dict):
            summary["rag_online_fallback_triggered"] = bool(payload.get("triggered"))
        elif event == "pkpd_profile":
            summary["pkpd_profile_present"] = True
        elif event == "drug_regimen":
            summary["drug_regimen_count"] = len(payload) if isinstance(payload, list) else 1
        elif event == "code_generated" and isinstance(payload, str):
            summary["ode_code_chars"] = len(payload)
        elif event == "image_ready":
            summary["image_ready"] = True
        elif event == "simulation_csv":
            summary["simulation_csv"] = payload
        elif event == "dose_response":
            summary["dose_response_present"] = True
            if isinstance(payload, dict):
                summary["ic50"] = payload.get("ic50")
                summary["ic90"] = payload.get("ic90")
                summary["hed"] = payload.get("hed")
        elif event == "metrics":
            summary["metrics"] = payload
        elif event == "experiment_protocols":
            summary["experiment_protocols_count"] = len(payload) if isinstance(payload, list) else 1
        elif event == "paper_evidence":
            summary["paper_evidence_count"] = len(payload) if isinstance(payload, list) else 1
        elif event == "report" and isinstance(payload, dict):
            summary["report_chars"] = len(payload.get("markdown", ""))
        elif event == "report_ready" and isinstance(payload, str):
            summary["report_chars"] = len(payload)
        elif event == "clarification_needed":
            summary["clarification_needed"] = True
            summary["clarification_questions"].append(payload)
        elif event == "token_usage":
            summary["token_usage"] = payload
        elif event == "error":
            summary["errors"].append(payload)
        elif event == "rule_violations":
            summary["rule_violations"] = payload
        elif event == "knowledge_graph" and isinstance(payload, dict):
            summary["kg_node_count"] = payload.get("node_count")
            summary["kg_edge_count"] = payload.get("edge_count")
            summary["kg_acyclic"] = payload.get("is_acyclic")

    return summary


if __name__ == "__main__":
    run()
