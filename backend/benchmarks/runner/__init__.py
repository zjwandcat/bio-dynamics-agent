# BioDynamics Benchmark Runner 包（Task 17）
#
# 真实端到端编排器入口：调用 v3 LangGraph 产出真实 simulation.csv / report.md，
# 替代 BenchmarkRunner 的 synthetic metrics 路径。
#
# 核心导出：
#   from benchmarks.runner import ScientificBenchmarkOrchestrator
#   orch = ScientificBenchmarkOrchestrator()
#   result = await orch.run("EGFR_RTK")          # 异步
#   result = orch.run_sync("EGFR_RTK")            # 同步包装

from benchmarks.runner.orchestrator import (
    PIPELINE_STAGES,
    StageSpec,
    StageTrace,
    ScientificBenchmarkOrchestrator,
    OrchestratorResult,
)

__all__ = [
    "ScientificBenchmarkOrchestrator",
    "PIPELINE_STAGES",
    "StageSpec",
    "StageTrace",
    "OrchestratorResult",
]
