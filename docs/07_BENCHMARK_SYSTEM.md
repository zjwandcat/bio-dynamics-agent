# Benchmark System

## Benchmark 层次

| 层 | 位置 | 数量/范围 | 作用 |
|---|---|---:|---|
| pathway configs | `backend/benchmarks/*.yaml` | 10 | 10 通路基础输入 |
| scientific alignment positive | `backend/benchmarks/scientific_alignment/*.yaml` | 10 | 机制、时间、证据、评分阈值 |
| scientific alignment negative | 同目录 `*_negative.yaml` | 10 | 负例/拒答/错误识别 |
| golden | `backend/benchmarks/golden/<pathway>/` | 10 | ground truth、expected markdown、metrics |
| regression | `backend/benchmarks/regression/` | 本地资产 | baseline/actual/regression report |
| runner tests | `backend/benchmarks/runner/test_orchestrator.py` | local | stage orchestration contract |

## 真实执行链

```text
BenchmarkRunner
  -> ScientificBenchmarkOrchestrator
  -> compiled_workflow_v3.ainvoke
  -> simulation.csv / report.md
  -> scientific alignment stages
  -> criteria / acceptance / score card
  -> benchmark result + trace
```

入口：`backend/app/benchmark_runner.py::BenchmarkRunner`；真实编排器：
`backend/benchmarks/runner/orchestrator.py::ScientificBenchmarkOrchestrator`。

`BENCHMARK_REAL_ORCHESTRATOR=true` 才启用真实端到端路径；legacy synthetic 需要
显式 `BENCHMARK_LEGACY_SYNTHETIC=true`，不应作为科学结果。

## 通路映射

| Pathway | Specialist | V4 模板 | Canonical | 基础 benchmark | 当前磁盘报告 |
|---|---|---|---|---|---|
| `EGFR_RTK` | `egfr_specialist.py` | `oscillatory_feedback.j2` | `knowledge/canonical/egfr.yaml` | `egfr_signaling.yaml` | fail: peak time |
| `MAPK_ERK` | `mapk_specialist.py` | `oscillatory_feedback.j2` | `canonical/mapk.yaml` | `mapk_cascade.yaml` | fail: peak time/order/adaptation |
| `PI3K_AKT_mTOR` | `pi3k_akt_mtor_specialist.py` | `oscillatory_feedback.j2` | `canonical/pi3k_akt_mtor.yaml` | `pi3k_akt_mtor.yaml` | fail: peak/order/adaptation/mass |
| `JAK_STAT` | `jak_stat_specialist.py` | `transcription_factor.j2` | `canonical/jak_stat.yaml` | `jak_stat.yaml` | fail: peak/order |
| `TGF_BETA` | `tgf_beta_specialist.py` | `transcription_factor.j2` | `canonical/tgf_beta.yaml` | `tgf_beta_signaling.yaml` | fail: peak/steady-state/explosion/BioModels |
| `WNT` | `wnt_specialist.py` | `destruction_complex.j2` | `canonical/wnt.yaml` | `wnt_signaling.yaml` | fail: peak/adaptation/steady-state/explosion/evidence |
| `p53` | `p53_specialist.py` | `transcriptional_delay.j2` | `canonical/p53.yaml` | `p53_signaling.yaml` | fail: peak/order/steady-state/oscillation |
| `NF_KB` | `nf_kappa_b_specialist.py` | `transcriptional_delay.j2` | `canonical/nf_kappa_b.yaml` | `nfkb_signaling.yaml` | fail: peak/order/steady-state/oscillation |
| `APOPTOSIS` | `apoptosis_specialist.py` | `caspase_cascade.j2` | `canonical/apoptosis.yaml` | `apoptosis.yaml` | fail: peak/order |
| `CELL_CYCLE` | `cell_cycle_specialist.py` | `cyclin_cdk_toggle.j2` | `canonical/cell_cycle.yaml` | `cell_cycle.yaml` | fail: peak/oscillation/explosion |

模板映射来源：`backend/app/ode_renderer_v2.py::_PATHWAY_TEMPLATE_MAP`；通路名转换还
经过 `v4_endpoints.py::_REGISTRY_TO_FRONTEND` 和 `main.py::_SA_PATHWAY_TO_CANONICAL`。

## 当前状态证据

检查时 `backend/data/sa_logs/all_10_pathways/<PATHWAY>/12_check_report.json` 的 10
个文件均为 `overall_passed=false`。这是磁盘上已有报告，不是本次重新运行的结果。
旧 `README.md` 的 1/10 “honest pass”是另一个时间点的快照；两者冲突时不要直接选一个，
应重新运行真实 benchmark 并记录 commit、flags、依赖和数据版本。

## Benchmark -> 代码定位

| 失败类型 | 优先检查 |
|---|---|
| peak time | pathway canonical、template selector、time scale、参数单位、solver |
| mass conservation | reaction IR、template source/sink、post validation |
| numerical explosion | initial conditions、rate units、solver step、template branch |
| flat-line | KG edge count、parameter fallback、ODE target mapping |
| BioModels mismatch | canonical model ID、SBML role、BioModels raw XML |
| evidence failure | N10、gold standard、evidence ranking/fusion |
| report/score mismatch | criteria、acceptance gate、validation matrix、renderer |

## 运行命令

窄测试优先：

```powershell
Set-Location backend
python -m pytest tests/test_benchmark_suite.py -q
python -m pytest tests/test_multi_pathway_e2e.py -q
```

真实 suite 需要外部 provider 和数据：

```powershell
$env:BENCHMARK_REAL_ORCHESTRATOR='true'
python -c "from app.benchmark_runner import BenchmarkRunner; print(BenchmarkRunner().run_all_to_markdown('benchmark_results.md'))"
```

不要用生成的 `benchmark_results.md` 标题判断通过；读取每个 pathway 的结构化检查和
CSV/metrics，确认是否是 flat-line、degraded 或真实通过。
