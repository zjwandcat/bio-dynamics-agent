# Scientific Regression Check

> Scientific Alignment Sprint — Sprint 0 Task 0.3
>
> 目的：检测科学指标退化，阻断导致科学正确性下降的 PR。所有结论基于真实运行结果，不接受 LLM 自述。

本目录实现 Scientific Regression CI 框架。CI workflow 见
`.github/workflows/scientific-regression.yml`，在 pull_request / push 到 `main`/`master`
时触发，运行 `run_regression.py`，FAIL 时退出码 1，从而阻断 Merge。

## 5 条回归规则

| # | 规则名 | 指标路径 | 基线 | 阈值 / 报警条件 | 比较方式 | 报警消息 |
|---|--------|----------|------|------------------|----------|----------|
| 1 | `erk_peak_time` | `simulation.erk_peak_min` | 20.0 min | 实际值 > 20 × (1 + 0.5) = 30 min（允许 50% 偏移，10–20 min 范围） | `max` | ERK Peak 时间退化：基线 20 min，实际超过允许范围（可能变为 40 min） |
| 2 | `egfr_peak_time` | `simulation.egfr_peak_min` | 5.0 min | 实际值 > 5 × (1 + 1.0) = 10 min | `max` | EGFR Peak 时间退化：基线 <5 min，实际超过 10 min |
| 3 | `discussion_topic_drift` | `discussion.topic_keywords` | — | 实际值包含任一禁用关键词：`癌症耐药` / `cancer drug resistance` / `EGFR qPCR` | `forbidden_contains` | Discussion 主题漂移：出现禁用关键词（如癌症耐药/qPCR） |
| 4 | `pmid_hallucination` | `discussion.cited_pmids` | — | 实际值包含任一已知幻觉 PMID：`18050474` / `39059397` / `40333694` | `not_contains` | PMID 幻觉复发：报告引用了审计已确认的幻觉 PMID |
| 5 | `confidence_inflation` | `validation.confidence` | — | 实际值 > 0.95 | `max_acceptable` | Confidence 虚高：超过 0.95 可能未真实执行 Consistency 检查 |

### 比较方式说明

- `max`：实际值不得超过 `baseline * (1 + max_deviation_pct)`。
- `max_acceptable`：实际值不得超过绝对上限 `max_acceptable`（无 baseline，用于 confidence 等绝对阈值）。
- `forbidden_contains`：实际值（转为字符串）不得包含任一禁用关键词。
- `not_contains`：实际值（PMID 列表）不得包含任一已知幻觉 PMID。

## 阻断逻辑

`run_regression.py` 的退出码：

- `0` — 报告状态为 `PASS` 或 `SKIPPED`（无 `actual.json` 时跳过，不阻断）。
- `1` — 报告状态为 `FAIL`，任一规则未通过。CI workflow 此步失败，**阻断 Merge**。

报告写入 `regression_report.json`，并由 workflow 以 artifact
`scientific-regression-report` 上传，便于事后审计。

## 文件清单

| 文件 | 用途 |
|------|------|
| `__init__.py` | Python 包标识（空） |
| `run_regression.py` | 回归检查主脚本 |
| `baseline.json` | 基线指标（Sprint 0 占位） |
| `actual.json.example` | 实际指标示例格式 |
| `regression_report.json` | 运行时生成（不入库），CI 上传为 artifact |
| `actual.json` | 运行时生成（不入库），各 Sprint 填充实际值后触发检查 |

## baseline.json 与 actual.json 填充时机

- **baseline.json（基线）**：Sprint 0 为占位值。**Sprint 1** 用真实 Benchmark 运行结果
  填充 `simulation.erk_peak_min`、`simulation.egfr_peak_min`、`validation.confidence` 等
  基线指标，作为后续 PR 比对的黄金基准。
- **actual.json（实际值）**：各 Sprint 运行 Benchmark / Discussion 生成后，将实际指标写入
  `actual.json`（复制 `actual.json.example` 后填充）。脚本检测到 `actual.json` 存在即触发
  规则检查；不存在则状态为 `SKIPPED`（首次运行不阻断）。
- `actual.json` 与 `regression_report.json` 为运行时产物，不应提交入库；如需纳入版本管理
  请通过 CI artifact 而非 commit。
