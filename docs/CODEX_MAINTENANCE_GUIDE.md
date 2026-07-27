# BioDynamics Agent: Codex 维护、升级与故障定位指南

> 首次进入项目时先读根目录 `PROJECT_STATE.json` 和
> `docs/00_PROJECT_OVERVIEW.md`。本文件保留为紧凑搜索导航；专题事实分别维护在
> `docs/01_ARCHITECTURE.md` 至 `docs/11_NEXT_STEPS.md`、
> `CONFIG_REFERENCE.md`、`PROMPT_INDEX.md`、`DEPENDENCY_GRAPH.md`、
> `DESIGN_DECISIONS.md` 和 `PROJECT_PHILOSOPHY.md`。

本文用于让后续 Codex 在尽量少搜索的情况下找到真实运行入口、核心调用链、
科学数据、前后端契约和验证方式。函数名比行号更耐升级，因此正文优先使用
`path::symbol` 作为定位锚点；需要行号时用文末的 `rg -n` 命令即时获取。

## 1. 范围与快照边界

核心仓库是：

```text
C:\Users\27553\Desktop\gzlab\bio-dynamics-agent
```

不要把同级的 `C:\Users\27553\Desktop\gzlab\backend` 当成应用后端；该目录当前
只有运行数据。`C:\Users\27553\Desktop\gzlab\mcp-servers` 也是仓库外的辅助目录。

本指南按 2026-07-20 的当前工作树编写。检查时的 Git 信息如下：

| 项目 | 检查值 | 含义 |
|---|---|---|
| branch | `main` | 当前本地分支 |
| HEAD | `b7a2820` | 仅是已提交基线，不代表当前文件内容 |
| tracking ref | `origin/main`, local 显示 ahead 10 / behind 0 | 未执行 fetch，不能据此断言远端最新状态 |
| working tree | 96 个 tracked 修改、163 个 untracked | 当前实现大量存在于未提交工作树中 |

后续 Codex 必须重新执行下面的命令，而不是沿用上述数字：

```powershell
Set-Location C:\Users\27553\Desktop\gzlab\bio-dynamics-agent
git status --short --branch
git diff --name-only
git log -3 --oneline --decorate
git branch -vv
```

没有用户明确授权时，不要用 `git reset`、`git checkout --`、`git clean`、切分支、
pull、rebase 或其他方式让工作树“回到 Git 版本”。当前工作树、HEAD 和远端可能是
三个不同版本；修复必须基于用户正在使用的文件。

## 2. 信息可信度顺序

发生冲突时按以下顺序判断：

1. 当前运行代码与实际路由/图组装。
2. 与目标模块直接对应的测试和 benchmark fixture。
3. `README.md`、`docs/ARCHITECTURE.md`、`docs/API.md` 等当前文档。
4. 历史审计报告、运行日志、生成报告和诊断脚本。

`ARCHITECTURE.md` 描述的是旧 v2 结构；可用于理解 N0-N11，但不能替代当前
`graph_v3.py::build_workflow_v3`。`README.md` 更接近当前工作树，但包含某次审计
快照和 benchmark 结论，升级后仍需用代码与测试复核。

## 3. 五分钟入口

先读以下文件，不要先全仓扫描：

| 想确认什么 | 第一入口 | 第二入口 |
|---|---|---|
| HTTP 启动、路由、SSE | `backend/app/main.py` | `backend/app/schemas.py` |
| Agent 调度与节点顺序 | `backend/app/graph_v3.py` | `backend/app/state.py` |
| N0-N11 科学处理 | `backend/app/nodes_v2.py` | `backend/app/nodes.py` |
| 配置、Provider、Feature Flag | `backend/app/config.py` | `backend/.env.example` |
| V4 工作台 REST | `backend/app/v4_endpoints.py` | `frontend/lib/api.ts` |
| 前端主产品 | `frontend/app/page.tsx` | `frontend/components/minimal/` |
| SSE 到 UI 状态 | `frontend/lib/sse.ts` | `frontend/lib/store.ts` |
| Benchmark 真链 | `backend/app/benchmark_runner.py` | `backend/benchmarks/runner/orchestrator.py` |
| 科学对齐 | `backend/app/scientific_alignment/` | `backend/knowledge/` |

推荐的窄搜索：

```powershell
# 找真实 FastAPI 路由
rg -n '^@(?:app|router)\.' backend/app/main.py backend/app/v4_endpoints.py

# 找 LangGraph 节点和边
rg -n 'add_node|add_edge|add_conditional_edges|compile' backend/app/graph_v3.py

# 找某个状态字段从哪里读写
rg -n '\bFIELD_NAME\b' backend/app frontend/lib frontend/components

# 找某个 SSE 事件的生产者和消费者
rg -n 'EVENT_NAME' backend/app/main.py frontend/lib frontend/components frontend/__tests__

# 只列源码，避开数据、日志与依赖
rg --files backend/app backend/tests frontend/app frontend/components frontend/lib verification
```

不要从仓库根对所有文件做无排除递归读取。`backend/data/sa_logs/`、向量库、
仿真 CSV/PNG、`frontend/node_modules/`、`.next/` 和大量 `_diag/_run/_result` 文件会
淹没真正的源码路径。

## 4. 真实运行架构

### 4.1 主聊天链路

```text
Browser
  -> POST /api/chat
  -> backend/app/main.py::chat
  -> backend/app/main.py::_v3_event_stream
  -> backend/app/graph_v3.py::compiled_workflow_v3.astream_events
  -> LangGraph workers/hooks
  -> backend/app/main.py::_emit_worker_outputs
  -> data: {"event": "...", "data": ...}
  -> frontend/lib/sse.ts::streamChat
  -> frontend/lib/store.ts::ingestSSEEvent
  -> minimal / advanced UI components
```

`main.py` 只导入 `compiled_workflow_v3`，`POST /api/chat` 当前无条件走 v3 事件流。
`config.py::Settings.WORKFLOW_VERSION` 虽然存在，但没有参与该入口的版本分派；修改
该环境变量不会切换主图。

### 4.2 LangGraph 骨架

图的唯一组装入口是 `backend/app/graph_v3.py::build_workflow_v3`：

```text
START
  -> ontology_hook
  -> _dynamic_router_hook
  -> pre_router
  -> supervisor
       -> clarification_node -> supervisor
       -> worker_mcp         -> supervisor
       -> worker_mechanism   -> pathway planner -> specialist -> crosstalk -> supervisor
       -> worker_rag         -> supervisor
       -> worker_pkpd        -> supervisor
       -> worker_ode         -> supervisor
       -> worker_sandbox     -> supervisor
       -> worker_validator   -> SBML grounder -> calibration -> sensitivity -> supervisor
       -> hypothesis hook -> worker_report -> validation pyramid -> supervisor
       -> END
```

V4 的 Reaction IR、Pathway Graph 和 ODE v2 renderer 不是独立外层节点，而是在
`graph_v3.py::worker_ode` 内作为可选 hook 执行。

### 4.3 运行模式

| 模式 | 计划来源 | 当前行为重点 |
|---|---|---|
| `auto_fast` | `graph_v3.py::pre_router` | 前端默认；跳过 MCP/PKPD，RAG 只生成低置信度参数占位，仍执行 mechanism/ODE/sandbox/validator/report |
| `auto_standard` | `graph_v3.py::_build_standard_plan` | 规则优先，无法确定时再让 LLM 判断 PKPD/证据需求 |
| `manual` | `graph_v3.py::_build_manual_plan` | 将 UI module key 映射为 worker，并自动补齐 ODE、sandbox、validator 等依赖 |

前端默认值在 `frontend/lib/store.ts` 的 `controlBarState`，当前为 `auto_fast`。
排查“为什么没有完整 RAG/MCP/PKPD”时先确认请求 mode，不要先改检索代码。

### 4.4 Worker 到内核节点映射

| Worker | 实际实现 | 主要输出 |
|---|---|---|
| `worker_mcp` | `nodes.py::node0_mcp_term_lookup` | 术语定义、工具调用、query rewrite |
| `worker_mechanism` | N0 `nodes_v2.py::n0_sbml_loader`; Standard/Manual 再走 N1/N2/N4；Fast 复用 `nodes.py::node1_parse_network` | entities、mechanism、knowledge_graph、network_json、SBML |
| `worker_rag` | N3 `n3_mechanism_rag` + N5 `n5_parameter_rag` | parameters、provenance、RAG insights、fallback 状态 |
| `worker_pkpd` | `nodes.py::node1_6_pkpd_inference` | PK/PD profile、drug regimen |
| `worker_ode` | V4 内部 hooks + N6 `nodes_v2.py::n6_ode_generator` | ode_model、Reaction IR、Pathway Graph、可选 v4 ODE |
| `worker_sandbox` | `graph_v3.py::worker_sandbox` -> `sandbox.py::execute_simulation_code_v2` | execution result、CSV、PNG、dose response、错误分类 |
| `worker_validator` | SBML role detection + `sbml_validator.py` | validation_report |
| `worker_report` | N8 features -> N9 experiment RAG -> N10 evidence RAG -> N11 report | metrics、protocols、evidence、report/final_report |

`backend/app/graph.py` 的图组装不是生产入口，但 `nodes.py` 仍被主 worker 复用。删除
legacy 文件前必须先检查 `graph_v3.py` 的 imports 和上述调用，不能按文件名直接判定废弃。

### 4.5 人在环路

进程内事件表在 `graph_v3.py` 顶部；触发判定在
`graph_v3.py::_check_clarification_triggers`；等待逻辑在
`graph_v3.py::clarification_node`。外部接口位于：

- `POST /api/chat/respond`
- `POST /api/chat/stop`
- `POST /api/chat/clear-memory`

当前 clarification timeout 是 120 秒。超时后会尝试由 LLM 自动选项，并写入
`llm_auto_decisions`。旧文档中更长的等待时间可能已过时。

## 5. 状态、线程与数据不变量

单一主状态定义在 `backend/app/state.py::BioDynamicsState`。数据大致按以下顺序演进：

```text
user_input
  -> entities / mechanism / network_json / knowledge_graph
  -> parameters / provenance / degradation markers
  -> ode_model + optional v4_reaction_ir / v4_pathway_graph / v4_ode_system
  -> execution_result / simulation_csv_path / image_base64
  -> validation_report / metrics
  -> experiment_protocols / paper_evidence
  -> report / final_report
  -> SSE hydration
```

修改状态时必须检查这些不变量：

1. 新字段加入 `BioDynamicsState`，类型与实际 producer 输出一致。
2. 容易跨请求残留的字段必须在 `main.py::_v3_event_stream` 的 `initial_state` 中重置。
3. list 字段受 `state.py::reset_if_empty_list` reducer 影响；新请求传空 list 才能清除旧值。
4. V4 同时保留平铺 `v4_*` 字段和 grouped `v4_state`。写入使用
   `set_v4_state`，读取使用 `get_v4/get_v4_state`，浅合并后需要时调用
   `normalize_v4_state`。
5. producer 之后检查 `main.py::_emit_worker_outputs` 是否发射了前端需要的事件。
6. 前端事件必须由 `frontend/lib/store.ts::ingestSSEEvent` hydration；新请求和
   `clearMemory` 还要清除对应 UI state。

LangGraph 使用 `MemorySaver`，按 `thread_id` 保存运行上下文。FastAPI lifespan 启动时
清空内存 checkpointer；前端 `sendMessage` 当前每次 query 创建新 thread id。因此进程
重启后不保留会话，“健康重启后恢复旧任务”不是现有能力。

## 6. Feature Flag 导航

所有 flag 的真相源是 `backend/app/config.py::Settings`，不是 `.env.example`。
Settings 和全局 provider client 在 import 时初始化；进程启动后修改 `.env` 通常需要
重启才能生效。

| 层 | 主开关 | 生效方式 |
|---|---|---|
| P1-P4 科学层 | `V4_SCIENTIFIC_LAYER_ENABLED` | `effective_v4_*` 聚合粗/细开关 |
| P5 验证层 | `V4_VALIDATION_ENABLED` | SBML grounder、validation、calibration |
| P6 假设层 | `V4_HYPOTHESIS_ENABLED` | hypothesis 与 dynamic routing |
| Scientific Alignment | `V4_SCIENTIFIC_ALIGNMENT_ENABLED` | 再由 `SA_*` 子开关控制 |
| Scientific Reviewer | `V4_SCIENTIFIC_REVIEWER_ENABLED` | reviewer/matrix/honesty 等新能力 |

细 flag 显式环境变量优先，其次粗 flag，再回退到属性默认值；逻辑集中在
`Settings._resolve_v4_flag`。Scientific Alignment 子能力必须通过
`Settings.is_sa_feature_enabled()` 判断。

两个容易误判的特例：

- `V4_SEQUENTIAL_RETRIEVER` 当前默认 `true`，并非所有 V4 行为都默认关闭。
- Specialist 影响旧主链还需要 KG feedback/writeback 配置。Mode A 将 Specialist
  拓扑写回 v3 KG；Mode B 让 sandbox 优先执行 `v4_ode_system.ode_code`。

Scientific Alignment 是主 LangGraph 完成后的 SSE 后处理，入口是
`main.py::_run_scientific_alignment_postprocess`，不是 `build_workflow_v3` 中的节点。

## 7. HTTP、SSE 与前端

### 7.1 后端接口表面

`backend/app/main.py`：

- `GET /`
- `POST /api/admin/update-vector-db`
- `GET /api/admin/rag-status`
- `GET /api/models/status`
- `POST /api/chat`
- `POST /api/chat/clear-memory`
- `POST /api/chat/respond`
- `POST /api/chat/stop`
- `POST /api/v4/benchmarks/run`（SSE suite）

`backend/app/v4_endpoints.py`，prefix `/api/v4`：

- `GET /pathways`
- `GET /pathways/{pathway_class}/graph`
- `POST /simulation/run`
- `POST /benchmark/{pathway_class}`
- `GET /reports/{report_id}`
- `POST /simulation/sweep`

这 6 个 V4 REST endpoint 是自包含的 deterministic workspace API，不等价于
`/api/chat` 的 LLM agent 流程。

### 7.2 SSE 契约

wire frame 由 `main.py::_sse_event` 生成：

```text
data: {"event":"EVENT_NAME","data":PAYLOAD}

```

事件映射集中在 `main.py::_emit_worker_outputs`，传输解析在
`frontend/lib/sse.ts::streamChat`，状态消费在
`frontend/lib/store.ts::ingestSSEEvent`。

修改或新增事件时同步检查：

1. 后端 payload shape。
2. `frontend/lib/sse.ts` 是否能解析该 frame。
3. `frontend/lib/store.ts` 的 case、类型、重置逻辑。
4. 最终展示组件。
5. `frontend/__tests__/store.test.ts` 和 `docs/API.md`。

当前已知契约热点：后端 workflow exception 分支发出 `{message, code}` object，而
前端 `error` case 主要按 string 处理；修错误展示时应先统一 shape 并补契约测试。
Validation Pyramid 在 report 后执行，因此后端可能先发 report-derived
`v4_validation_report`，再捕获完整 level1-level5 后重发；前端应以后一事件覆盖。

### 7.3 前端入口

| 路由 | 入口 | 用途 |
|---|---|---|
| `/` | `app/page.tsx` -> `components/minimal/MinimalApp.tsx` | 当前主产品，一句话到仿真/校验/报告 |
| `/workspace` | `app/workspace/page.tsx` | 当前重定向到 `/advanced` |
| `/advanced` | `app/advanced/page.tsx` -> `WorkbenchShell` | 归档保留的四栏工作台 |
| `/benchmarks` | `BenchmarkCenter` | 10 通路 SSE benchmark UI |
| `/report/[id]` | `ReportViewer` | report REST viewer |

REST base URL 定义在 `frontend/lib/api.ts`，chat SSE 在 `frontend/lib/sse.ts`，benchmark
SSE 在 `frontend/lib/benchmarkSse.ts`，全局状态在 `frontend/lib/store.ts`。

编辑 `frontend/` 前先读 `frontend/AGENTS.md`。当前是 Next.js 16；必须查本地
`frontend/node_modules/next/dist/docs/` 中与目标 API 对应的文档。

## 8. 科学模型、知识与命名

### 8.1 科学资产

| 数据/代码 | 路径 | 用途 |
|---|---|---|
| Canonical mechanisms/timelines | `backend/knowledge/canonical/*.yaml` | 机制和动态基线 |
| Experiment rules | `backend/knowledge/experiments/*.yaml` | rule-based experiment planner |
| Literature gold standard | `backend/knowledge/gold_standard/*.yaml` | PMID/证据基线 |
| Pathway benchmark inputs | `backend/benchmarks/*.yaml` | 10 通路基础 benchmark |
| Scientific Alignment fixtures | `backend/benchmarks/scientific_alignment/*.yaml` | 正/负样例和阈值 |
| Golden answers | `backend/benchmarks/golden/` | expected metrics/report |
| Raw SBML | `backend/data/raw/` | BioModels XML 输入，部分是本地运行资产 |
| Vector store | `backend/data/vector_db/` | Chroma 持久化，可重建，不是源码 |

### 8.2 10 通路注册与命名

通路命名存在多套形式：ontology registry 的大写 canonical key、pathway initializer
的混合 key、frontend 的小写枚举。转换点至少包括：

- `backend/app/ontology/pathway_registry.py`
- `backend/app/pathways/pathway_registry.py`
- `backend/app/pathway_graph/initializer.py`
- `backend/app/v4_endpoints.py::_REGISTRY_TO_FRONTEND`
- `backend/app/main.py::_SA_PATHWAY_TO_CANONICAL`
- `frontend/lib/api.ts::PathwayClass`

新增或重命名通路时，按下列顺序查漏：

1. Ontology keyword/alias registry。
2. Specialist class、specialist registry、specialist hook import。
3. Static PathwayGraph initializer。
4. Reaction IR/ODE template routing。
5. REST 和 SA name mapping。
6. Canonical、experiment、literature、benchmark YAML。
7. Frontend type、benchmark cards、示例输入。
8. Specialist、API、benchmark、E2E tests。

不要只改一个 mapping；当前不同模块的 `p53/NF_KB/WNT/TGF_BETA` 等大小写和后缀
并不完全一致。

## 9. 按故障症状找代码

| 症状/任务 | 第一搜索点 | 顺着检查 |
|---|---|---|
| 服务启动失败 | `main.py`, `config.py` | `logging_config.py`, provider 初始化, optional imports |
| `/api/chat` 无输出/中断 | `main.py::_v3_event_stream` | `graph_v3.py::build_workflow_v3`, `schemas.py` |
| Agent 顺序不对 | `graph_v3.py::pre_router/supervisor/build_workflow_v3` | `_advance_step`, clarification router |
| 新请求混入旧数据 | `main.py` initial_state | `state.py` reducers, frontend reset, thread id |
| 实体/机制/KG 错 | `nodes_v2.py` N1/N2/N4 | `species_ontology.py`, `kg_builder.py`, specialist writeback |
| 通路分类错 | `ontology/pathway_registry.py` | `pathways/pathway_planner.py`, name mappings |
| RAG 无命中/选错参数 | `rag_client.py` | `rag_collections.py`, N3/N5, `bio_db_client.py`, build scripts |
| Provider/rerank 异常 | `config.py` | `RerankManager`, model status endpoint, connectivity script |
| ODE 模板选错 | `template_selector.py` | `rule_engine.py`, `prompts_v2.py`, `ode_templates/` |
| V4 ODE/机制错 | `reaction_ir_v2/` | `ode_renderer_v2.py`, `ode_templates_v2/`, specialist hook |
| 仿真失败/负值/爆炸 | `sandbox.py` | `graph_v3.py::worker_sandbox`, solver modules, generated code |
| SBML 对比错 | `sbml_validator.py` | `biomodels_client.py`, `sbml_grounder/`, raw SBML |
| Validation 显示错 | `validation_v2/validation_agent.py` | `main.py` double emission, frontend store/panel |
| 报告缺证据/引用 | N9/N10/N11 in `nodes_v2.py` | `report_renderer.py`, discussion/evidence modules |
| SA 卡片/评分错 | `main.py::_run_scientific_alignment_postprocess` | `scientific_alignment/*`, SA flags, store/panel |
| Benchmark 假通过/不复现 | `benchmark_runner.py` | real orchestrator, validation fixtures, generated CSV/report |
| V4 REST 工作台错 | `v4_endpoints.py` | `frontend/lib/api.ts`, pathway initializer |
| UI 收到事件但不更新 | `frontend/lib/store.ts` | SSE parser, reset logic, target component |
| MCP 术语错 | `mcp_client.py` | endpoint env, `worker_mcp`, root auxiliary MCP server |

## 10. RAG、外部服务与沙箱边界

外部依赖入口：

- LLM/embedding/rerank：`backend/app/config.py`
- Chroma collections：`rag_client.py`, `rag_collections.py`
- KEGG/Reactome/UniProt/ChEMBL：`bio_db_client.py`
- BioModels：`biomodels_client.py`
- PubMed/MCP：`mcp_client.py` 与 N10 fallback
- ClinicalTrials：`rag_client.py::_query_clinical_trials`

`main.py` import 时设置 `NO_PROXY=*`，外部请求会绕过系统代理。企业网络、受限网络或
需要本地代理的环境中，provider/PubMed/RAG 全部失败时优先检查该副作用。

沙箱并非容器隔离。`sandbox.py` 使用代码安全扫描、AST 预检、临时目录、子进程和
timeout；它仍然是执行 LLM 生成 Python 的高风险边界。涉及允许 import、文件访问、
subprocess、timeout 或资源限制的修改必须运行 security、sandbox 和 E2E 测试。

可选科学依赖会改变算法路径：RoadRunner、lmfit、SALib、lxml、jitcdde 缺失时可能
降级而不报致命错误。比较两台机器结果时同时记录 Python 版本、optional dependency
availability、Feature Flag 和 benchmark fixture 版本。

## 11. 运行与部署

### 11.1 Windows 本地安装

```powershell
Set-Location C:\Users\27553\Desktop\gzlab\bio-dynamics-agent\backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env，填入实际 provider 配置；不要提交或打印该文件

Set-Location ..\frontend
npm ci
```

两个终端分别启动：

```powershell
Set-Location C:\Users\27553\Desktop\gzlab\bio-dynamics-agent\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```powershell
Set-Location C:\Users\27553\Desktop\gzlab\bio-dynamics-agent\frontend
npm run dev
```

访问 `http://localhost:3000`、`http://localhost:8000/` 和
`http://localhost:8000/docs`。Windows 一键入口是 `scripts/start-dev.bat`，它支持
`backend/.venv` 或 `backend/venv`。

### 11.2 Docker

`Dockerfile` 使用 Python 3.11 backend 与 Node 20 frontend；
`docker-compose.yml` 暴露 8000/3000，并把 `backend/data` 挂载为持久卷。

```powershell
Copy-Item backend/.env.example backend/.env
docker compose up --build
```

当前本机、CI、Docker 可能分别运行 Python 3.14/3.12/3.11，Node 24/20。升级依赖时
不能只验证本机版本。

## 12. 测试与发布门

### 12.1 按改动选测试

| 改动 | 优先测试 |
|---|---|
| graph/HITL/state | `test_supervisor_clarification.py`, `test_v4_state_consolidation.py`, `test_integration_flag_*.py`, `test_p*_hook_e2e.py` |
| RAG/ODE/sandbox | `test_rag_ode_integration.py`, `test_ode_template_routing.py`, `test_ode_templates_v2_complete.py`, `test_cascade_fixes.py`, `e2e_test.py` |
| specialist/crosstalk | `test_*_specialist.py`, `test_crosstalk_coordinator.py`, `test_adapter_v3_v4.py` |
| validation/calibration | `test_level*.py`, `test_validation_matrix.py`, `test_calibration_agent.py`, `test_sensitivity.py` |
| hypothesis/report/SA | `test_hypothesis_agent.py`, `test_scientific_*.py`, integration honesty tests |
| frontend SSE/store | `frontend/__tests__/store.test.ts`, component tests |
| API/benchmark | `test_benchmark_suite.py`, `test_multi_pathway_e2e.py`, `verification/benchmark/` |

### 12.2 常用命令

```powershell
# targeted backend
Set-Location backend
python -m pytest tests/test_v4_state_consolidation.py -q

# backend suite
python -m pytest tests -q

# root-configured verification
Set-Location ..\verification
python -m pytest benchmark -q
python -m pytest solver_validation -q

# frontend
Set-Location ..\frontend
npm test -- --run
npm run lint
npm run build
npm run test:e2e
```

`Makefile` 假设 Bash/GNU make，并使用 `/bin/bash -o pipefail` 和 POSIX 环境变量语法；
不要在原生 PowerShell 中照抄 `make benchmark` 的内部命令。

真实 10 通路 benchmark 会调用真实 orchestrator、LLM/RAG/外部数据并产生运行文件，
耗时且可能因环境不同失败。窄修复先跑单元/集成测试，再跑相关 pathway，最后才跑
全套。Pipeline 成功和报告生成不能替代数值 benchmark 判定。

### 12.3 测试资产风险

当前 `.gitignore` 排除了 `backend/tests/`、`verification/`、`frontend/__tests__/` 和
`frontend/e2e/`，但 Makefile/CI 又引用这些目录。检查时这些本地测试文件均未被 Git
跟踪。后续如果准备 clean clone、CI 或发布，必须先确认需要的测试/fixture 是否已经
纳入版本控制；不能用“本机测试存在”推断远端 CI 拥有它们。

## 13. 推荐升级/修复流程

1. 记录当前 branch、HEAD、tracking ref、tracked/untracked diff；不要清理用户改动。
2. 用第 9 节按症状定位第一入口，再沿 producer -> state -> transport -> consumer 搜索。
3. 先写或选择能复现问题的最窄测试，并记录 Feature Flag、mode 和 optional dependency。
4. 只改拥有该行为的模块；避免同时重构旧链、V4 hook、SA 和前端。
5. 改状态或 SSE 时按第 5、7 节逐项同步契约。
6. 跑 targeted test；再跑相邻 integration；科学输出变更再跑 pathway benchmark。
7. 检查新生成的 CSV、PNG、日志、向量库和报告没有混入源码 diff。
8. 更新与行为直接相关的文档；如果入口、图、状态或路径发生变化，更新本指南和
   根 `AGENTS.md`。

## 14. 当前已核实的维护陷阱

- `WORKFLOW_VERSION` 当前不切换主 `/api/chat` 流程。
- `graph.py` 的图是 legacy，但 `nodes.py` 仍有生产调用，不能整文件删除。
- `/api/chat` error payload 和前端 error case 存在 shape 同步热点。
- Validation Pyramid 在 report 后执行，是软门，并可能发两次 validation SSE。
- `scripts/demo_benchmark.sh` 检查 `/health`，但当前 FastAPI 没有该 route；它可能误判
  后端未启动并再次占用 8000。
- `scripts/demo.sh` 打开 `/workspace?pathway=...`，而 `/workspace` 当前直接重定向
  `/advanced`，旧 query 行为不可依赖。
- `SbmlUpload.tsx` 和 `BioModelsFetcher.tsx` 引用了 `/api/v4/sbml/import` 与
  `/api/v4/biomodels/{id}`，当前后端 router 没有对应 endpoint。
- `frontend/components/chat/ControlBar.tsx` 有硬编码 `http://localhost:8000`，绕开统一
  `NEXT_PUBLIC_API_BASE`。
- Next public environment 通常在 build 时内联；Compose 只在 runtime 注入
  `NEXT_PUBLIC_API_BASE`，远程部署时需要验证实际浏览器请求地址。
- CORS 只允许单个 `FRONTEND_URL`；当前 Compose 更接近本机研究环境，不是完整的
  多租户生产部署。
- `main.py` 强制 `NO_PROXY=*`，可能破坏必须经过代理的外呼。
- 沙箱是临时目录 + 子进程 + denylist/AST/timeout，不是容器安全边界。
- optional dependency 降级会让同一提交在不同机器给出不同科学结果。
- `backend/data/` 和根部大量审计报告是证据/产物，不是默认修改入口。
- `.github/workflows/scientific-regression.yml` 中存在带仓库目录前缀的路径；GitHub
  checkout 根通常已经是仓库根，修改 CI 前应复核该 workflow 的工作目录。
- Playwright config 可启动 frontend dev server，但不自动提供真实 backend；依赖
  API 的 E2E 需要另起 backend 或明确 mock。

## 15. 何时更新本指南

出现以下任一变化时，同一 PR/改动中更新本文件：

- FastAPI 入口、API prefix 或 SSE frame 改变。
- LangGraph 节点、边、worker/hook 插入点改变。
- `BioDynamicsState` 或 V4 state 双写规则改变。
- Feature Flag 层级或默认值改变。
- 新增/移除 pathway、specialist、template、benchmark fixture。
- 前端主路由、API base、store 或 SSE ingestion 改变。
- 测试目录重新纳入 Git，或 CI/Makefile 验证命令改变。

更新后至少运行以下快速自检：

```powershell
Test-Path AGENTS.md
Test-Path docs/CODEX_MAINTENANCE_GUIDE.md
rg -n 'def build_workflow_v3|compiled_workflow_v3' backend/app/graph_v3.py
rg -n 'def _v3_event_stream|def _emit_worker_outputs' backend/app/main.py
rg -n 'ingestSSEEvent' frontend/lib/store.ts
git status --short --branch
```
