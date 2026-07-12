"""Task 17 编排器单元测试。

验证点（不调用真实 v3 pipeline，避免 LLM/网络依赖）：
1. benchmark YAML 加载（_load_benchmark_spec）
2. 12 阶段 post-hoc 校验 + fail-fast 逻辑（_validate_core_stages）
3. 真实 artifacts 提取（_extract_real_artifacts）
4. 扁平化指标 + 节点名提取 + PMID 提取（helper 函数）
5. SA 阶段 flag 守护（SA OFF → 全部 skipped）
6. SA 阶段 flag ON 签名正确性（调用真实 SA 模块，mock 输入）

运行：python -B benchmarks/runner/test_orchestrator.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# 确保从 backend/ 目录运行
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))


def _make_mock_final_state() -> dict:
    """构造 mock final_state（模拟 v3 pipeline 成功完成后的状态）。"""
    return {
        "mode": "auto_standard",
        "execution_plan": ["worker_mechanism", "worker_rag", "worker_ode", "worker_sandbox", "worker_report"],
        "mcp_term_definitions": [{"term": "EGFR", "definition": "Epidermal Growth Factor Receptor"}],
        "mechanism": {"pathway": "EGFR_RTK", "cell": "HEK293", "simulation_type": "ODE"},
        "knowledge_graph": {
            "nodes": [
                {"id": "n1", "name": "EGFR"},
                {"id": "n2", "name": "pEGFR"},
                {"id": "n3", "name": "RAS"},
                {"id": "n4", "name": "RAF"},
                {"id": "n5", "name": "MEK"},
                {"id": "n6", "name": "ERK"},
                {"id": "n7", "name": "DUSP"},
            ],
            "edges": [],
        },
        "network_json": {"nodes": [], "edges": []},
        "parameters": {
            "EGFR_EGFR_binding": {"value": 0.0021, "source": "BioModels", "confidence": "High"},
        },
        "pkpd_profile": {"drug": "Gefitinib", "ic50": 0.02},
        "ode_model": {"template": "cascade", "code": "# ODE code", "time_unit": "minute"},
        "execution_result": {"status": "success", "stdout_stderr": "Simulation completed"},
        "simulation_csv_path": "C:/tmp/mock_simulation.csv",
        "validation_report": {"error_diff": 0.01, "pass": True, "method": "sbml"},
        "report": {
            "markdown": "# EGFR Signaling Report\n\nERK peak at 15 min.\n",
            "persisted_path": "C:/tmp/mock_report.md",
        },
        "final_report": "# EGFR Signaling Report\n\nERK peak at 15 min.\n",
        "metrics": {
            "pEGFR": {"peak": 0.85, "peak_time": 3.0, "steady_state": 0.2},
            "ERK": {"peak": 0.72, "peak_time": 15.0, "steady_state": 0.1},
            "overall": {"amplification": 50.0, "mass_conservation_error": 0.01},
        },
        "confidence": 0.91,
        "paper_evidence": [
            {"pmid": "PMID:12451180", "title": "Schoeberl 2002"},
            {"pmid": "PMID:7657691", "title": "Marshall 1995"},
        ],
        "experiment_protocols": [{"name": "pEGFR Western blot", "target": "EGFR"}],
        "sbml_role": "validation_oracle",
        "sbml_model_id": "BIOMD0000000010",
        "rag_insights": {"hit_rate": 0.8},
        "rag_summary": "Retrieved 5 params from BioModels",
    }


def test_yaml_loading() -> None:
    """测试 1：benchmark YAML 加载。"""
    from benchmarks.runner.orchestrator import ScientificBenchmarkOrchestrator

    orch = ScientificBenchmarkOrchestrator()
    # 用 result 对象承载 errors
    result = orch.run.__wrapped__ if hasattr(orch.run, "__wrapped__") else None
    # 直接调用 _load_benchmark_spec
    from benchmarks.runner.orchestrator import OrchestratorResult

    res = OrchestratorResult(
        pathway_class="EGFR_RTK", name="", status="fail", stages=[],
        simulation_csv_path="", report_path="", final_report_markdown="",
        real_metrics={}, real_metrics_flat={}, confidence=0.0,
    )
    spec = orch._load_benchmark_spec("EGFR_RTK", res)
    assert spec is not None, f"EGFR_RTK spec not loaded, errors={res.errors}"
    assert spec.get("pathway_class") == "EGFR_RTK"
    assert "input" in spec
    assert "pass_criteria" in spec
    print(f"[PASS] test_yaml_loading: name={spec.get('name')!r}")

    # 测试未找到的 pathway
    res2 = OrchestratorResult(
        pathway_class="NONEXISTENT", name="", status="fail", stages=[],
        simulation_csv_path="", report_path="", final_report_markdown="",
        real_metrics={}, real_metrics_flat={}, confidence=0.0,
    )
    spec2 = orch._load_benchmark_spec("NONEXISTENT", res2)
    assert spec2 is None
    assert any("not found" in e for e in res2.errors)
    print("[PASS] test_yaml_loading: nonexistent pathway returns None")


def test_core_stages_pass() -> None:
    """测试 2：核心阶段全 pass（mock final_state 完整）。"""
    from benchmarks.runner.orchestrator import (
        OrchestratorResult,
        ScientificBenchmarkOrchestrator,
    )

    orch = ScientificBenchmarkOrchestrator()
    res = OrchestratorResult(
        pathway_class="EGFR_RTK", name="", status="fail", stages=[],
        simulation_csv_path="", report_path="", final_report_markdown="",
        real_metrics={}, real_metrics_flat={}, confidence=0.0,
    )
    mock_state = _make_mock_final_state()
    critical_failed = orch._validate_core_stages(mock_state, res)
    assert not critical_failed, f"critical_failed should be False, stages={res.stages}"
    # 核心阶段 0-8 共 9 个
    core_stages = [s for s in res.stages if not s["name"].startswith("stage_1") or s["name"] in ("stage_1_mcp",)]
    # 实际上阶段 0-8 是 9 个（stage_0 到 stage_8）
    assert len(res.stages) == 9, f"expected 9 core stages, got {len(res.stages)}"
    all_pass = all(s["status"] == "pass" for s in res.stages)
    assert all_pass, f"not all pass: {[(s['name'], s['status'], s.get('reason','')) for s in res.stages]}"
    print(f"[PASS] test_core_stages_pass: {len(res.stages)} stages all pass")


def test_core_stages_fail_fast() -> None:
    """测试 3：关键阶段缺失 → fail-fast。"""
    from benchmarks.runner.orchestrator import (
        OrchestratorResult,
        ScientificBenchmarkOrchestrator,
    )

    orch = ScientificBenchmarkOrchestrator()
    res = OrchestratorResult(
        pathway_class="EGFR_RTK", name="", status="fail", stages=[],
        simulation_csv_path="", report_path="", final_report_markdown="",
        real_metrics={}, real_metrics_flat={}, confidence=0.0,
    )
    # mock state 缺失 simulation_csv_path（stage_6 关键字段）
    mock_state = _make_mock_final_state()
    mock_state["simulation_csv_path"] = ""  # 清空 → stage_6 fail
    mock_state["execution_result"] = {}  # 清空 → stage_6 fail
    critical_failed = orch._validate_core_stages(mock_state, res)
    assert critical_failed, "critical_failed should be True when stage_6 missing"
    stage_6 = next(s for s in res.stages if s["name"] == "stage_6_sandbox")
    assert stage_6["status"] == "fail"
    assert "simulation_csv_path" in stage_6["reason"] or "execution_result" in stage_6["reason"]
    print(f"[PASS] test_core_stages_fail_fast: stage_6 fail, reason={stage_6['reason']!r}")

    # 测试 pipeline 异常（final_state=None）
    res2 = OrchestratorResult(
        pathway_class="EGFR_RTK", name="", status="fail", stages=[],
        simulation_csv_path="", report_path="", final_report_markdown="",
        real_metrics={}, real_metrics_flat={}, confidence=0.0,
    )
    critical_failed2 = orch._validate_core_stages(None, res2)
    assert critical_failed2, "critical_failed should be True when final_state=None"
    assert all(s["status"] == "fail" for s in res2.stages)
    print(f"[PASS] test_core_stages_fail_fast: None state → all {len(res2.stages)} stages fail")


def test_extract_real_artifacts() -> None:
    """测试 4：真实 artifacts 提取。"""
    from benchmarks.runner.orchestrator import (
        OrchestratorResult,
        ScientificBenchmarkOrchestrator,
    )

    orch = ScientificBenchmarkOrchestrator()
    res = OrchestratorResult(
        pathway_class="EGFR_RTK", name="", status="fail", stages=[],
        simulation_csv_path="", report_path="", final_report_markdown="",
        real_metrics={}, real_metrics_flat={}, confidence=0.0,
    )
    mock_state = _make_mock_final_state()
    orch._extract_real_artifacts(mock_state, res)
    assert res.simulation_csv_path == "C:/tmp/mock_simulation.csv"
    assert res.report_path == "C:/tmp/mock_report.md"
    assert "EGFR Signaling Report" in res.final_report_markdown
    assert res.confidence == 0.91
    assert "pEGFR" in res.real_metrics
    assert "ERK" in res.real_metrics
    # 扁平化：pEGFR_peak_time = 3.0
    assert res.real_metrics_flat.get("pEGFR_peak_time") == 3.0
    assert res.real_metrics_flat.get("ERK_peak_time") == 15.0
    assert res.real_metrics_flat.get("overall_amplification") == 50.0
    print(f"[PASS] test_extract_real_artifacts: csv={res.simulation_csv_path!r}")
    print(f"        metrics_flat keys={list(res.real_metrics_flat.keys())}")


def test_helper_functions() -> None:
    """测试 5：helper 函数。"""
    from benchmarks.runner.orchestrator import (
        _extract_cited_pmids,
        _extract_node_names,
        _flatten_metrics,
        _is_filled,
    )

    # _is_filled
    assert _is_filled(None) is False
    assert _is_filled("") is False
    assert _is_filled({}) is False
    assert _is_filled([]) is False
    assert _is_filled("abc") is True
    assert _is_filled({"k": 1}) is True
    assert _is_filled(0.0) is True  # 数值视为已填充
    print("[PASS] _is_filled: all cases correct")

    # _flatten_metrics
    flat = _flatten_metrics({
        "pEGFR": {"peak": 0.85, "peak_time": 3.0},
        "overall": {"amplification": 50.0},
    })
    assert flat == {"pEGFR_peak": 0.85, "pEGFR_peak_time": 3.0, "overall_amplification": 50.0}
    print("[PASS] _flatten_metrics: nested → flat correct")

    # _extract_node_names
    nodes = _extract_node_names(
        {"nodes": [{"name": "EGFR"}, {"name": "ERK"}, "DUSP"]},
        None,
    )
    assert nodes == ["EGFR", "ERK", "DUSP"]
    # 回退到 entities
    nodes2 = _extract_node_names({}, [{"id": "RAS"}, {"name": "RAF"}])
    assert nodes2 == ["RAS", "RAF"]
    print(f"[PASS] _extract_node_names: {nodes} / fallback {nodes2}")

    # _extract_cited_pmids
    pmids = _extract_cited_pmids([
        {"pmid": "PMID:12451180", "title": "Schoeberl"},
        {"pmid_id": "PMID:7657691"},
        "PMID:16246870",
        {"pmid": "PMID:12451180"},  # 重复
    ])
    assert pmids == ["PMID:12451180", "PMID:7657691", "PMID:16246870"]
    print(f"[PASS] _extract_cited_pmids: {pmids} (deduped)")


def test_sa_stages_flag_off() -> None:
    """测试 6：SA 阶段 flag OFF → 全部 skipped。

    确保 V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，编排器不触发任何 SA 计算。
    """
    # 确保 SA flags 全部 OFF
    os.environ.pop("V4_SCIENTIFIC_ALIGNMENT_ENABLED", None)
    os.environ.pop("SA_CONSISTENCY_CHECKER", None)
    os.environ.pop("SA_SEVEN_AXIS", None)
    os.environ.pop("SA_SCIENTIFIC_CRITIC", None)
    os.environ.pop("SA_LOOP_TERMINATION", None)

    from benchmarks.runner.orchestrator import (
        OrchestratorResult,
        ScientificBenchmarkOrchestrator,
    )

    orch = ScientificBenchmarkOrchestrator()
    mock_state = _make_mock_final_state()
    res = OrchestratorResult(
        pathway_class="EGFR_RTK", name="", status="fail", stages=[],
        simulation_csv_path="", report_path="", final_report_markdown="",
        real_metrics={}, real_metrics_flat={}, confidence=0.0,
    )
    # 先提取 artifacts（_run_sa_stages 依赖 result.real_metrics 等）
    orch._extract_real_artifacts(mock_state, res)

    # 运行 SA 阶段（SA OFF → 全部 skipped）
    asyncio.run(orch._run_sa_stages(mock_state, {}, res, Path("/tmp")))

    sa_stages = [s for s in res.stages if s["name"].startswith("stage_9") or s["name"].startswith("stage_1") and "sa_" in s["name"]]
    # 更准确的过滤：SA 阶段名含 "sa_"
    sa_stages = [s for s in res.stages if "sa_" in s["name"]]
    assert len(sa_stages) == 4, f"expected 4 SA stages, got {len(sa_stages)}: {[s['name'] for s in sa_stages]}"
    all_skipped = all(s["status"] == "skipped" for s in sa_stages)
    assert all_skipped, f"not all skipped: {[(s['name'], s['status']) for s in sa_stages]}"
    # SA 字段应为 None
    assert res.consistency_report is None
    assert res.seven_axis_validation is None
    assert res.critic_report is None
    assert res.acceptance_report is None
    print(f"[PASS] test_sa_stages_flag_off: {len(sa_stages)} SA stages all skipped")

    # 但 mechanism_graph / parameter_priors / biomodels_comparison / evidence_fusion
    # 这些是从 final_state 直接提取的，SA OFF 时也填充
    assert res.mechanism_graph is not None
    assert res.parameter_priors is not None
    assert res.biomodels_comparison is not None
    assert res.evidence_fusion is not None
    assert res.mechanism_graph["knowledge_graph"]["nodes"][0]["name"] == "EGFR"
    print("[PASS] test_sa_stages_flag_off: mechanism_graph/parameter_priors/biomodels_comparison/evidence_fusion still populated")


def test_sa_stages_flag_on() -> None:
    """测试 7：SA 阶段 flag ON → 调用真实 SA 模块（验证签名正确）。

    开启所有 SA flags，用 mock final_state 调用 SA 阶段。
    重点验证：函数签名正确、返回结构正确、不抛异常。

    注意：Settings 类属性在 import 时评估，setdefault os.environ 无效；
    必须直接 monkey-patch settings 实例属性。
    """
    from app.config import settings

    # 备份原值，测试后恢复
    orig = {
        "V4_SCIENTIFIC_ALIGNMENT_ENABLED": settings.V4_SCIENTIFIC_ALIGNMENT_ENABLED,
        "SA_CONSISTENCY_CHECKER": settings.SA_CONSISTENCY_CHECKER,
        "SA_SEVEN_AXIS": settings.SA_SEVEN_AXIS,
        "SA_SCIENTIFIC_CRITIC": settings.SA_SCIENTIFIC_CRITIC,
        "SA_LOOP_TERMINATION": settings.SA_LOOP_TERMINATION,
    }
    try:
        settings.V4_SCIENTIFIC_ALIGNMENT_ENABLED = True
        settings.SA_CONSISTENCY_CHECKER = True
        settings.SA_SEVEN_AXIS = True
        settings.SA_SCIENTIFIC_CRITIC = True
        settings.SA_LOOP_TERMINATION = True

        from benchmarks.runner.orchestrator import (
            OrchestratorResult,
            ScientificBenchmarkOrchestrator,
        )

        orch = ScientificBenchmarkOrchestrator()
        mock_state = _make_mock_final_state()
        res = OrchestratorResult(
            pathway_class="EGFR_RTK", name="", status="fail", stages=[],
            simulation_csv_path="", report_path="", final_report_markdown="",
            real_metrics={}, real_metrics_flat={}, confidence=0.0,
        )
        orch._extract_real_artifacts(mock_state, res)

        # 运行 SA 阶段（SA ON → 调用真实 SA 模块）
        asyncio.run(orch._run_sa_stages(mock_state, {}, res, Path("/tmp")))

        sa_stages = [s for s in res.stages if "sa_" in s["name"]]
        assert len(sa_stages) == 4, f"expected 4 SA stages, got {len(sa_stages)}"
        print(f"[PASS] test_sa_stages_flag_on: {len(sa_stages)} SA stages executed")
        for s in sa_stages:
            print(f"        {s['name']}: status={s['status']}, reason={s.get('reason','')!r}, dur={s['duration_seconds']}s")

        # 验证 SA 字段被填充（不要求全 pass，只要结构正确）
        assert res.consistency_report is not None, "consistency_report should be populated"
        print(f"        consistency: passed={res.consistency_report['passed']}, violations={res.consistency_report['violation_count']}, rules_evaluated={res.consistency_report['rules_evaluated']}")

        assert res.seven_axis_validation is not None, "seven_axis_validation should be populated"
        print(f"        seven_axis: overall_passed={res.seven_axis_validation['overall_passed']}, axes={len(res.seven_axis_validation['axes'])}")

        assert res.critic_report is not None, "critic_report should be populated"
        print(f"        critic: overall_status={res.critic_report['overall_status']}, retry_required={res.critic_report['retry_required']}")

        assert res.acceptance_report is not None, "acceptance_report should be populated"
        print(f"        acceptance: passed={res.acceptance_report['passed']}, failed={res.acceptance_report['failed_criteria']}")

        # scientific_alignment 汇总
        assert res.scientific_alignment is not None
        assert res.scientific_alignment["sa_enabled"] is True
        assert res.scientific_alignment["canonical_name"] == "egfr"
        print(f"[PASS] test_sa_stages_flag_on: all SA fields populated, canonical=egfr")
    finally:
        # 恢复原值，避免影响后续测试
        for k, v in orig.items():
            setattr(settings, k, v)


def test_run_sync_from_event_loop_error() -> None:
    """测试 8：run_sync 在事件循环中调用应抛 RuntimeError。"""
    from benchmarks.runner.orchestrator import ScientificBenchmarkOrchestrator

    orch = ScientificBenchmarkOrchestrator()

    async def _call_run_sync_in_loop():
        # 在事件循环中调用 run_sync 应抛 RuntimeError
        try:
            orch.run_sync("EGFR_RTK")
            return False  # 不应到达
        except RuntimeError as exc:
            return "running event loop" in str(exc)

    result = asyncio.run(_call_run_sync_in_loop())
    assert result, "run_sync should raise RuntimeError when called from event loop"
    print("[PASS] test_run_sync_from_event_loop_error: raises RuntimeError correctly")


def main() -> None:
    """运行所有测试。"""
    print("=" * 70)
    print("Task 17 ScientificBenchmarkOrchestrator 单元测试")
    print("=" * 70)
    test_yaml_loading()
    test_core_stages_pass()
    test_core_stages_fail_fast()
    test_extract_real_artifacts()
    test_helper_functions()
    test_sa_stages_flag_off()
    test_sa_stages_flag_on()
    test_run_sync_from_event_loop_error()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
