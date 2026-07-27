# BioDynamics Agent 项目总览

## 项目定位

BioDynamics Agent 将自然语言生物医学假说转换为定量信号通路模型，生成 ODE，执行
确定性/半确定性仿真，再输出验证结果与科学报告。当前实现是一个以 v3
Supervisor-Worker 为主干、以 V4 hooks 和 Scientific Alignment 为增强层的研究型
原型。

核心仓库：`C:\Users\27553\Desktop\gzlab\bio-dynamics-agent`

## 当前入口

| 表面 | 入口 | 当前状态 |
|---|---|---|
| 主聊天 | `backend/app/main.py::chat` -> `POST /api/chat` | 当前实际线上主链 |
| LangGraph | `backend/app/graph_v3.py::compiled_workflow_v3` | 已编译，`MemorySaver` checkpointer |
| V4 workspace REST | `backend/app/v4_endpoints.py` | 6 个自包含 deterministic endpoint |
| Benchmark SSE | `backend/app/main.py::v4_benchmarks_run` | 10 通路 suite |
| 前端主入口 | `frontend/app/page.tsx` -> `MinimalApp` | `/`，一句话到报告 |
| 高级工作台 | `frontend/app/advanced/page.tsx` | `/advanced`，旧四栏 IDE |

## 架构快照

```text
Question
  -> Terminology / Ontology
  -> Mechanism Parser
  -> RAG Retriever
  -> Reaction / Knowledge Graph
  -> Parameter Selection
  -> ODE Generator + Jinja Templates
  -> Sandbox Solver
  -> SBML / Scientific Validation
  -> Experiment + Evidence Retrieval
  -> Report Renderer
  -> Scientific Alignment post-process
```

## 证据边界

本组文档按当前工作树逆向整理，不代表远端最新版本。检查快照为 `main@b7a2820`
附近的工作树；当前有大量 tracked 和 untracked 改动，且没有为本次文档重新运行
全量 benchmark。凡是带“当前结果”的内容，都应查看同段的证据路径和日期。

## 阅读顺序

1. `00_PROJECT_OVERVIEW.md`：范围、入口、证据边界。
2. `01_ARCHITECTURE.md`：Supervisor、workers、V4 hooks 和前后端边界。
3. `02_CODE_STRUCTURE.md`：文件 ownership 与修改入口。
4. `03_DATA_FLOW.md`：状态字段和 JSON 形状的演进。
5. `04_RAG_PIPELINE.md`、`05_SIMULATION_PIPELINE.md`、`06_VALIDATION_SYSTEM.md`：科学处理链。
6. `07_BENCHMARK_SYSTEM.md`、`08_KNOWN_PROBLEMS.md`：可验证性与现有缺陷。
7. `09_TECH_DEBT.md`、`10_DEVELOPMENT_RULES.md`、`11_NEXT_STEPS.md`：治理与后续维护。
8. `CONFIG_REFERENCE.md`、`PROMPT_INDEX.md`、`DEPENDENCY_GRAPH.md`、`DESIGN_DECISIONS.md`、`PROJECT_PHILOSOPHY.md`、`PROJECT_STATE.json`：机器和人共同使用的索引。

## 事实与愿景

“请求能返回报告”只证明工作流完成，不证明 ODE 科学正确。当前代码具备完整的
编排、检索、模板、仿真、验证和报告骨架；定量 benchmark 仍存在 peak time、质量
守恒、数值爆炸、证据和 BioModels 对齐等失败记录。后续 Codex 应优先保留可追溯性
和科学正确性，而不是增加表面功能数量。
