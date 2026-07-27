# Simulation Pipeline

## 目标与边界

simulation pipeline 将 KG/参数转为可执行 Python ODE，并在 sandbox 子进程中生成
time series。它有 v3 LLM-assisted relation path 与 V4 template path 两套来源，最终
都在 `worker_sandbox` 收口。

## 主流程

```text
knowledge_graph + parameters
  -> template_selector.select_template (v3)
  -> N6 network relations
  -> Jinja render (v3 templates)
  -> sandbox security + AST precheck
  -> scipy/JIT solver subprocess
  -> CSV + PNG + markers
  -> post_simulation_validation / retry
```

V4 开启时，`worker_ode` 还会：

```text
network_json
  -> ReactionBuilder -> ReactionIRv2
  -> PathwayGraphBuilder -> v4_pathway_graph
  -> ODERendererV2 -> v4_ode_system
  -> optional Mode B sandbox execution
```

## 模板 ownership

### v3 `backend/app/ode_templates/`

顶层模板包括：`Simple_Activation`、`Simple_Inhibition`、`Cascade_Activation`、
`Cascade_Inhibition`、`Signaling_Cascade_Phos`、`PKPD_OneCompartment`,
`PKPD_TwoCompartment`、`Combination`、`DoseSweep`；下划线文件是 include helpers。

模板选择入口是 `backend/app/template_selector.py::select_template`，不是 Prompt 中的
自由字符串。规则引擎在 `backend/app/rule_engine.py`。

### v4 `backend/app/ode_templates_v2/`

顶层 pathway/mechanism templates 包括：`oscillatory_feedback`、
`transcriptional_delay`、`transcription_factor`、`destruction_complex`、
`caspase_cascade`、`cyclin_cdk_toggle`、`nuclear_transport`、
`bistable_switch`、`ubiquitination_cascade`；下划线文件是 include helpers。

`backend/app/ode_renderer_v2.py::_select_template` 负责按 pathway/mechanism/DDE
选择模板。默认未知路径会回退 `oscillatory_feedback.j2`，因此新增通路必须补 explicit mapping。

## 求解与输出

`sandbox.py::execute_simulation_code_v2` 通过临时目录和子进程执行代码，记录：

- `simulation.csv` time points/species
- `simulation.png` image
- stdout/stderr markers
- `error_class`: `none`, `syntax_error`, `runtime_error`, `numerical_error`, `timeout`, `security_error`, `recursion_error`

`graph_v3.py::worker_sandbox` 默认按 mode 允许重试；失败时可复用 audit correction 和
N6 重生成。Mode B 使用 v4 ODE 时，重试不会回退到稀疏的 v3 LLM ODE。

## 数值安全

- AST/denylist 安全检查在 `sandbox.py::_check_code_security`、`_ast_precheck`。
- 确定性检查在 `_check_determinism`；默认不允许随机噪声。
- `post_simulation_validation` 检查非负性、finite、范围和生物有效性。
- optional `jitcdde` 支持延迟方程；缺失时会退化为 ODE 近似。
- 计算能力依赖 numpy/scipy/matplotlib，结果受 Python、依赖和 Feature Flag 影响。

## 失败排查顺序

1. 看 `worker_ode` 的 `template`、`rule_violations` 和生成代码，而不是只看图。
2. 检查 `parameters` 是否 fallback、单位是否已归一化、初始条件是否合理。
3. 在 `sandbox_logs` 找 `error_class`、solver message 和 CSV path。
4. 检查 CSV 是否 flat-line、负值、爆炸或 peak time 超出 canonical range。
5. 再判断是 template、parameter、solver、pathway topology 还是 benchmark threshold 问题。

## 不能推断的结论

生成 ODE、solver 返回 success、CSV 存在、PNG 能显示，都不等于机制、质量守恒或动态
时间尺度正确。科学通过必须由 `06_VALIDATION_SYSTEM.md` 和 `07_BENCHMARK_SYSTEM.md`
中的检查确认。
