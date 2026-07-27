# Architecture Snapshot

## 主链

```text
POST /api/chat
  -> main.py::chat
  -> main.py::_v3_event_stream
  -> graph_v3.py::compiled_workflow_v3
  -> START
  -> ontology_hook
  -> _dynamic_router_hook
  -> pre_router
  -> supervisor
  -> workers / clarification
  -> worker_report
  -> validation_pyramid hook
  -> END
  -> main.py::_emit_worker_outputs
  -> SSE
```

## 模块快照

| 模块 | 作用 | 输入 | 输出 | 文件位置 | 完成度（代码存在性） |
|---|---|---|---|---|---|
| API / SSE | 接收请求、运行图、翻译事件 | `ChatRequest` | SSE event stream | `backend/app/main.py` | complete |
| PreRouter | 按 mode 生成 worker plan | user input、mode、manual modules | `execution_plan` | `backend/app/graph_v3.py::pre_router` | complete |
| Supervisor | 调度下一个 worker、处理 HITL | plan、step、clarification state | `next_worker` 或 END | `graph_v3.py::supervisor` | complete |
| Terminology | MCP 术语标准化 | user input | definitions、rewritten query | `graph_v3.py::worker_mcp`, `mcp_client.py` | complete / 外部服务依赖 |
| Mechanism Parser | N0/N1/N2/N4 解析实体和机制 | input、SBML 可选 | entities、mechanism、KG、network JSON | `graph_v3.py::worker_mechanism`, `nodes_v2.py` | complete |
| Retriever | 机制、参数、实验、证据检索 | KG、edges、query | candidates、selected params、evidence | `rag_client.py`, `nodes_v2.py` | complete / 质量未证明 |
| Reaction IR | 统一反应机制模型 | `network_json` | `v4_reaction_ir` | `reaction_ir_v2/`, `graph_v3.py` | implemented / flag gated |
| Pathway Specialist | 通路特异拓扑和参数先验 | pathway class、IR | specialist output、feedback | `pathways/specialists/`, `specialist_hook.py` | implemented / flag gated |
| ODE Generator | 选择模板并渲染 Python | KG、parameters、template | `ode_model.code` | `nodes_v2.py::n6_ode_generator`, `template_selector.py` | complete |
| V4 ODE Renderer | 从 IR/PathwayGraph 渲染专用 ODE | Reaction IR、graph | `v4_ode_system` | `ode_renderer_v2.py`, `ode_templates_v2/` | implemented / flag gated |
| Simulator | 子进程执行 ODE、导出 CSV/PNG | generated Python | execution result、CSV、image | `sandbox.py`, `graph_v3.py::worker_sandbox` | complete / 非容器隔离 |
| SBML Validator | 与 BioModels/SBML 对比 | CSV、SBML role | validation report | `sbml_validator.py`, `validation_v2/level2_sbml.py` | implemented / optional RoadRunner |
| Validation Pyramid | L1-L5 验证 | state、metrics、SBML | `v4_validation_report` | `validation_v2/` | implemented / 当前软门 |
| Report | N8-N11 指标、实验、证据、报告 | simulation、metrics、evidence | markdown + JSON fields | `nodes_v2.py`, `report_renderer.py` | complete |
| Scientific Alignment | consistency、critic、matrix、review | report、metrics、evidence | SA SSE payloads | `main.py::_run_scientific_alignment_postprocess`, `scientific_alignment/` | implemented / flag gated |
| Frontend | 输入、进度、图、曲线、验证、报告 | REST + SSE | UI state | `frontend/app`, `frontend/components`, `frontend/lib` | implemented |

## 图组装与 hook 位置

唯一主图组装点是 `backend/app/graph_v3.py::build_workflow_v3`。当前边关系：

- `START -> ontology_hook -> _dynamic_router_hook -> pre_router -> supervisor`。
- `worker_mechanism -> pathway_planner -> specialist -> crosstalk -> supervisor`。
- `worker_validator -> sbml_grounder -> calibration -> sensitivity -> supervisor`。
- `supervisor` 选择 `worker_report` 时先进入 hypothesis hook。
- `worker_report -> validation_pyramid -> supervisor`。
- 其他 worker 完成后返回 supervisor；step 完成后进入 END。

Reaction IR、Pathway Graph、ODE Template v2 在 `worker_ode` 内部执行；Scientific
Alignment 在图结束后由 `main.py` 的后处理生成，不是 LangGraph 节点。

## 重要边界

1. `/api/chat` 是 v3 wire contract，不能仅改后端而不改前端 SSE ingestion。
2. `/api/v4/*` workspace REST 不调用聊天图，使用 pathway graph 的 deterministic simulation。
3. `backend/app/graph.py` 是 legacy bridge；`nodes.py` 仍被 v3 worker 复用，不能按文件名删除。
4. `WORKFLOW_VERSION` 在配置中存在，但当前 `main.py` 没有用它选择图版本。
5. V4 flat fields 和 `v4_state` 并存，必须通过 `state.py` helper 保持一致。
