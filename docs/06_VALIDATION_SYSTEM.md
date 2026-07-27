# Validation System

## 两套验证层

项目同时包含：

1. v3 `worker_validator`：SBML role detection + BioModels/SBML 对比。
2. V4 Validation Pyramid：L1-L5 五层 state report。
3. Scientific Alignment 的 12-axis Validation Matrix、Consistency、Critic、Review。

这些层次不应混写为一个“passed”布尔值。

## L1-L5 Pyramid

| Level | 目标 | 实现 |
|---|---|---|
| L1 Internal | state/KG/ODE/仿真内部一致性、非负、守恒等 | `validation_v2/level1_internal.py` |
| L2 SBML | 与 loaded BioModels/SBML 对齐 | `validation_v2/level2_sbml.py`、`sbml_validator.py` |
| L3 Cross-pathway | shared species、cross-talk、time scale | `validation_v2/level3_crosstalk.py` |
| L4 Benchmark | pathway benchmark metrics/criteria | `validation_v2/level4_benchmark.py` |
| L5 Hypothesis | 假设可证伪性和实验支持 | `validation_v2/level5_hypothesis.py` |

编排入口是 `validation_v2/validation_agent.py::ValidationAgent`。结果写入
`state["v4_validation_report"]`，包含 `level1` 到 `level5`、`overall_pass`、
`failed_levels` 等字段。

## 12-axis Matrix

`scientific_alignment/validation_matrix.py` 定义 12 轴：

1. Mechanism
2. Ontology
3. Literature
4. BioModels
5. Parameter
6. Simulation
7. Dynamics
8. Experiment
9. Evidence Attribution
10. Scientific Writing
11. Reproducibility
12. Benchmark

每轴状态是 `PASS`、`PARTIAL` 或 `FAIL`；聚合逻辑在
`validation_matrix.py::_aggregate_status`。它与旧 seven-axis validator 不是同一接口。

## Scientific Alignment

主后处理入口：`backend/app/main.py::_run_scientific_alignment_postprocess`。
可包含：

- canonical consistency checker
- scientific critic
- multi-dimensional confidence
- evidence fusion / discussion renderer
- validation rule engine / scientific review
- parameter provenance / decision log
- BioModels calibration
- loop controller / regression monitor

总开关 `V4_SCIENTIFIC_ALIGNMENT_ENABLED` 关闭时子功能不生效；子 flag 用
`Settings.is_sa_feature_enabled()` 判断。

## 当前门控行为

- Validation Pyramid hook 位于 `worker_report` 之后，因为 N8 metrics 在 report worker 内生成。
- report-derived `v4_validation_report` 可能先发一次，完整 level1-level5 后再发一次。
- 当前 validation pyramid 是软门，异常会被 hook 捕获并可能返回空对象；它不会自动把失败重新路由到 simulator。
- 没有 SBML/CSV 时部分 level 会标记 skipped/pass，不能当作真实 BioModels 通过。
- `roadrunner`、`SALib`、`lmfit`、`lxml` 等缺失时采用降级路径。

## 验证输出最小检查

任何修复至少核对：

```json
{
  "pass": false,
  "method": "skipped|roadrunner|structural_similarity",
  "role": "none|calibration_reference|validation_oracle|primary_ground_truth",
  "error_diff": 0.0,
  "peak_time_diff": 0.0,
  "structural_confidence_score": 0.0
}
```

不要把 `skipped`、`degraded`、`passed` 三种语义压成一个“成功”。
