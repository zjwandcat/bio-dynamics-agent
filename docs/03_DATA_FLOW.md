# Data Flow and State Contract

## 端到端数据流

| 阶段 | 输入 JSON / state | 输出 JSON / state | 主要代码 |
|---|---|---|---|
| Request | `{user_input, thread_id, mode, manual_modules}` | `ChatRequest` | `backend/app/schemas.py` |
| Initial state | `ChatRequest` | 清空后的 `BioDynamicsState` | `main.py::_v3_event_stream` |
| N0 / MCP | natural language、可选 `BIOMD*` | `sbml_model_id/text`、`mcp_term_definitions` | `graph_v3.py::worker_mechanism/worker_mcp`, `nodes_v2.py::n0_sbml_loader` |
| N1 | user input、MCP output | `entities[]`，含 name/type/aliases/canonical id | `nodes_v2.py::n1_ner_entity_normalize` |
| N2 | entities、user input | `mechanism`、`network_relations`、edges | `nodes_v2.py::n2_mechanistic_planner` |
| N3/N4 | mechanism、retrieved chunks | `knowledge_graph`、`network_json` | `nodes_v2.py::n3_mechanism_rag`, `n4_kg_builder` |
| N5 | KG edges、RAG candidates | `parameters[edge_key]`、fallback/provenance | `nodes_v2.py::n5_parameter_rag` |
| V4 hooks | network JSON、pathway class、IR | `v4_reaction_ir`、`v4_pathway_graph`、specialist outputs | `graph_v3.py`, `reaction_ir_v2`, `pathways/` |
| N6 | KG、parameters、template decision | `ode_model={template,code,parameters_used,rule_violations}` | `nodes_v2.py::n6_ode_generator` |
| Sandbox | `ode_model.code` 或 Mode B `v4_ode_system.ode_code` | `execution_result`、`simulation_csv_path`、image、error class | `sandbox.py`, `graph_v3.py::worker_sandbox` |
| Validator | CSV、SBML role/text/id | `validation_report`、`sbml_role` | `graph_v3.py::worker_validator`, `sbml_validator.py` |
| N8 | CSV、simulation output | nested `metrics`、`feature_metadata`、confidence | `nodes_v2.py::n8_scientific_features` |
| N9 | user input、pathway、metrics | `experiment_protocols[]` | `nodes_v2.py::n9_experiment_rag`, `scientific_alignment/experiment_planner.py` |
| N10 | report context、RAG/PubMed | `paper_evidence[]`、diagnostic | `nodes_v2.py::n10_evidence_rag` |
| N11 | metrics、protocols、evidence、KG | `report`, `final_report` | `nodes_v2.py::n11_scientific_report`, `report_renderer.py` |
| Pyramid | state + metrics + optional SBML/hypothesis | `v4_validation_report.level1..level5` | `validation_v2/validation_agent.py` |
| SA post-process | pathway、metrics、KG、evidence、report | SA SSE payloads | `main.py::_run_scientific_alignment_postprocess` |
| Frontend | SSE `{event,data}` | Zustand state + UI messages | `frontend/lib/sse.ts`, `frontend/lib/store.ts` |

## 核心 state 字段

| 领域 | 关键字段 |
|---|---|
| 请求/调度 | `user_input`, `thread_id`, `mode`, `execution_plan`, `current_step`, `completed_workers`, `next_worker` |
| HITL | `pending_clarification`, `clarification_request`, `clarification_response`, `stop_requested`, `llm_auto_decisions` |
| 语义 | `entities`, `mechanism`, `network_relations`, `knowledge_graph`, `network_json` |
| RAG | `rag_retrieved_params`, `rag_selected_params`, `rag_fallback`, `rag_hit_rate`, `rag_insights`, `paper_evidence` |
| ODE/仿真 | `parameters`, `ode_model`, `v4_reaction_ir`, `v4_pathway_graph`, `v4_ode_system`, `execution_result`, `simulation_csv_path`, `error_class` |
| 验证/报告 | `sbml_role`, `validation_report`, `v4_validation_report`, `metrics`, `experiment_protocols`, `report`, `final_report` |
| V4 grouped state | `v4_state` 与平铺 `v4_*` 字段并存 |

## 状态不变量

1. `BioDynamicsState` 是跨节点 contract；新增字段必须更新 `state.py`、initial state、producer 和 consumer。
2. `reset_if_empty_list` 使新请求传 `[]` 才能覆盖 checkpointer 中旧 list；新增 list 字段必须测试跨请求隔离。
3. V4 写入使用 `set_v4_state`，读取使用 `get_v4/get_v4_state`，必要时调用 `normalize_v4_state`。
4. state 输出和 SSE 输出不是同一 schema；`main.py::_emit_worker_outputs` 是二者之间的显式适配层。
5. 前端新一轮请求会清空上一轮 Results 和 SA fields；后端线程由 `thread_id` 隔离，但服务重启会清空 MemorySaver。

## SSE 事件到 UI

| 后端事件 | store 字段/消息 | 典型 UI |
|---|---|---|
| `knowledge_graph` | `knowledgeGraph` | Graph tab |
| `v4_pathway_graph` | `pathwayGraph` | PathwayGraph |
| `code_generated` | message type `code` | Logs/assistant |
| `v4_simulation_result` | `simulationResult` | Curves/SimulationPanel |
| `v4_validation_report` | `validationReport` | ValidationPyramid |
| `metrics` | message metrics | Report/metrics |
| `paper_evidence` | message evidence | Report/evidence |
| `report`/`report_ready` | `reportMarkdown` + report message | Report tab |
| `sa_*` | SA store fields | ScientificAlignmentPanel |

修改字段或事件时，必须沿 `producer -> state -> SSE -> store -> component -> test` 完整走一遍。
