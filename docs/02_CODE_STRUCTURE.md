# Code Structure and Ownership

## 顶层目录

| 路径 | ownership | 不应做什么 |
|---|---|---|
| `backend/app/` | 运行时 Python 包、FastAPI、LangGraph、科学模块 | 不把日志/数据写进源码目录 |
| `backend/app/agent_orchestration/` | V4 dynamic router/fail-safe/registry | 不误认为已经替代 v3 worker |
| `backend/app/agents_v4/` | 可被 dynamic router 调度的 Agent 类 | 不假设输出自动合并回主 state |
| `backend/app/pathways/` | 通路 planner、specialist、cross-talk hook | 新通路要同步 registry 和 YAML |
| `backend/app/reaction_ir_v2/` | 17 类机制的 IR schema/builder/validation | 不绕过 IR 直接写非声明机制 |
| `backend/app/ode_templates/` | v3 Jinja ODE templates | 不在模板里隐藏业务路由 |
| `backend/app/ode_templates_v2/` | V4 通路/机制模板及 include 片段 | include 片段不能当顶层模板 |
| `backend/app/scientific_alignment/` | SA、evidence、review、matrix、RCA | 不把报告通过当科学通过 |
| `backend/app/validation_v2/` | L1-L5 Validation Pyramid | 当前 hook 是软门，需看报告状态 |
| `backend/knowledge/` | canonical、experiment、literature YAML | 不把历史运行输出当 canonical |
| `backend/benchmarks/` | pathway、golden、SA、regression fixtures | 不在 runner 中硬编码期望结果 |
| `backend/data/` | raw SBML、vector DB、运行报告、日志 | 默认不参与源码搜索或提交 |
| `backend/scripts/` | 建库、导入、诊断和更新脚本 | 先确认脚本是否会改数据/向量库 |
| `frontend/app/` | Next App Router 路由页面 | 编辑前读 `frontend/AGENTS.md` |
| `frontend/components/` | UI 按域分组 | 不绕过 Zustand 复制 SSE 状态 |
| `frontend/lib/` | API client、SSE parser、Zustand store | 不破坏 `/api/chat` 旧 contract |
| `verification/` | 发布验证/benchmark/performance suite | 当前目录被 `.gitignore` 排除，先确认版本资产 |
| `docs/` | 设计、逆向和维护索引 | 数字与代码不一致时以代码为准 |

## 后端 ownership

| 文件/目录 | 负责行为 | 常用搜索符号 |
|---|---|---|
| `main.py` | FastAPI app、SSE、SA 后处理、v4 benchmark SSE | `chat`, `_v3_event_stream`, `_emit_worker_outputs` |
| `graph_v3.py` | 主图、worker wrapper、hook 链、HITL | `build_workflow_v3`, `supervisor`, `worker_*` |
| `state.py` | `BioDynamicsState`、reducers、V4 state merge | `reset_if_empty_list`, `set_v4_state`, `normalize_v4_state` |
| `config.py` | `.env`、LLM/embedding/rerank、feature flags | `Settings`, `_resolve_v4_flag`, `is_sa_feature_enabled` |
| `nodes_v2.py` | N0-N11 实际科学节点 | `n0_...` 至 `n11_...` |
| `nodes.py` | v1 compatibility nodes 被部分 v3 worker 复用 | `node0_mcp_term_lookup`, `node1_parse_network`, `node1_6_pkpd_inference` |
| `rag_client.py` | Chroma、hybrid search、rerank、online fallback | `RagClient`, `hybrid_search`, `search_params_hybrid` |
| `sandbox.py` | 安全扫描、AST precheck、subprocess、重试 | `execute_simulation_code_v2`, `execute_with_stability_retry` |
| `template_selector.py` | v3 模板选择、时间尺度 | `select_template`, `get_simulation_time_scale` |
| `ode_renderer_v2.py` | V4 pathway/template 选择和渲染 | `ODERendererV2`, `_select_template` |
| `v4_endpoints.py` | workspace deterministic REST | `list_pathways`, `_simulate_pathway`, `parameter_sweep` |
| `benchmark_runner.py` | 单 pathway/suite runner facade | `BenchmarkRunner` |
| `benchmarks/runner/orchestrator.py` | 真实端到端 scientific benchmark stages | `ScientificBenchmarkOrchestrator`, `StageSpec` |

## 前端 ownership

| 路径 | 负责行为 |
|---|---|
| `app/page.tsx` | `/` 主入口 |
| `components/minimal/` | 输入、7 步 workflow、Results tabs |
| `app/advanced/page.tsx` + `components/workspace/` | 四栏高级工作台 |
| `app/benchmarks/page.tsx` + `components/benchmark/` | benchmark suite UI |
| `lib/api.ts` | REST base URL、types、v3/v4 client |
| `lib/sse.ts` | `/api/chat` frame parser |
| `lib/benchmarkSse.ts` | benchmark SSE parser |
| `lib/store.ts` | Zustand state、`ingestSSEEvent`、send/stop/clear |

## 源码、数据、产物分界

默认可搜索：`backend/app`, `frontend/app`, `frontend/components`, `frontend/lib`,
`backend/benchmarks` YAML、`backend/knowledge` YAML、`verification` 测试。

默认跳过：`backend/data`, `backend/logs`, `backend/_*`, `test_outputs_*`,
`frontend/node_modules`, `frontend/.next`, `__pycache__`, CSV/PNG、向量库和历史审计报告。
