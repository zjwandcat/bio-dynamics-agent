# BioDynamics Agent

> 自然语言驱动的生物医学信号通路建模与仿真智能体。输入定性假设，输出可复现的 ODE 仿真、实验设计与多源证据链报告。

---

## 项目定位

BioDynamics Agent 是面向生物医学研究者的 AI 建模助手。它将一段自然语言描述的通路假设（如 *EGF 刺激后 pEGFR 与 ppERK 的时序响应*）转化为：

1. 标准化的反应网络与知识图谱
2. 基于真实文献参数的常微分方程（ODE）模型
3. 沙箱执行的数值仿真与曲线特征（Peak / Tpeak / AUC / Fold Change）
4. 多源证据链（PubMed / BioModels / 仿真 / 机制推理）渲染的科学报告
5. 机制驱动的实验设计建议（Western blot / ELISA / qPCR / Flow Cytometry）

---

## 已验证能力（每项均通过代码直接调用确认）

| 能力 | 验证证据 |
|---|---|
| 10 条信号通路 Specialist 注册 | `EGFR_RTK / MAPK_ERK / PI3K_AKT_mTOR / p53 / APOPTOSIS / CELL_CYCLE / JAK_STAT / NF_KB / WNT / TGF_BETA` |
| 10 通路确定性 mass-action ODE 仿真 | `_simulate_pathway()` 不依赖 LLM，`scipy.integrate.solve_ivp`，强制非负约束，每条通路绑定 BioModels 参考模型 |
| 14 个 REST API 端点 | OpenAPI 实测：4 个 `/api/chat` + 7 个 `/api/v4` + 2 个 `/api/admin` + 1 个 `/api/models` |
| LangGraph v3 工作流编译 | 22 个节点：11 个主流程 + 11 个 v4 hook chain |
| Reaction IR v2 中间表示 | 19 类机制枚举 + Pydantic v2 schema（PHOSPHORYLATION / UBIQUITINATION / BINDING / CLEAVAGE / GTP_GDP_EXCHANGE / TRANSCRIPTION / NUCLEAR_IMPORT / DEGRADATION …） |
| ODE Jinja2 模板库 | v3 12 个模板（含 PK/PD 一/二房室、Cascade、DoseSweep、Combination）+ v4 11 个机制专用模板（含振荡反馈、双稳态开关、Caspase 级联、Cyclin-CDK toggle、破坏复合体） |
| RAG 四路向量库 | ChromaDB 实测：mechanism=70 / parameter=4584 / experiment=20 / evidence=120 |
| 5 级验证金字塔 | `Level1InternalValidator / Level2SBMLValidator / Level3CrossPathwayValidator / Level4BenchmarkValidator / Level5HypothesisValidator` |
| Scientific Alignment 13 子模块 | Consistency Checker / Scientific Critic / Multi-dim Confidence / Validation Rule Engine / Scientific Review / Parameter Provenance / Decision Log / BioModels Calibration / Evidence Fuser / Discussion Renderer / Curve Metrics / Loop Controller / Regression Monitor |
| MCP 4 生物医学工具 | OpenBioMed 实体识别 / UMLS 同义词 / medical-terminologies（ICD-10/SNOMED CT）/ PubMed 检索 |
| 4 本体客户端 | HGNC / UniProt / ChEBI / GO（缓存优先 TTL 7 天） |
| BioModels Validation Oracle | Track A roadrunner 真实仿真对比，Track B 结构相似度降级 |
| Sandbox 执行保护 | AST 预检 + 静态安全扫描 + import guard + 子进程资源限制 + LSODA→BDF→Radau 阶梯重试 |
| Multi-provider LLM | `FallbackLLM` 0.5s 切换；Rerank 三提供商级联（讯飞 MaaS → OpenRouter → SiliconFlow） |

---

## 系统架构

```
┌──────────────── Frontend (Next.js 16.2.10 + React 19.2.4 + Zustand 5.0.14) ────────────────┐
│  /             Minimal Auto-Chat：自然语言输入 → 7 步工作流跟踪 → 4 标签结果（图/曲线/验证/报告）│
│  /advanced     Scientific IDE：4 栏（Project / Workspace / Validation / AI Assistant）       │
│  /benchmarks   Benchmark Center：10 通路金标准套件，SSE 流式进度                              │
│  /report/[id]  持久化仿真报告查看器                                                          │
└─────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                      │ SSE (text/event-stream, POST /api/chat)
                                      ▼
┌──────────────── Backend (FastAPI 0.139.0 + LangGraph 1.2.7 + Python 3.14) ─────────────────┐
│  /api/chat          v3 Supervisor-Worker 流式工作流（compiled_workflow_v3）                 │
│  /api/v4/*          7 个确定性 REST 端点（pathways / graph / simulation / benchmark / reports / sweep）│
│  /api/admin/*       向量库更新 + RAG 健康状态                                                │
│  /api/models/status LLM / Embedding / Rerank 提供商健康                                      │
│                                                                                            │
│  v3 主流程：pre_router → supervisor → 8 workers + clarification_node                       │
│  v4 hooks：ontology · pathway_planner · specialist · crosstalk · sbml_grounder ·            │
│            calibration · sensitivity · validation_pyramid · hypothesis（全部 Feature Flag 守护）│
│  SA 后处理：consistency · critic · multi_dim · validation · review · biomodels_calibration │
│            （master flag OFF 时系统行为同 v3）                                              │
│                                                                                            │
│  存储：ChromaDB 1.5.9（4 collections）· LangGraph MemorySaver · 本地 PMID 缓存              │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 后端工作流

### v3 Supervisor-Worker 主流程（`backend/app/graph_v3.py::compiled_workflow_v3`）

| 节点 | 角色 | 输出 |
|---|---|---|
| `pre_router` | 解析运行模式（manual / auto_fast / auto_standard） | `execution_plan` |
| `supervisor` | 调度下一个 worker 或触发 human-in-loop | `dispatch_record` |
| `clarification_node` | 等待用户澄清（10 min 超时） | `clarification_response` |
| `worker_mcp` | MCP 术语标准化（4 工具） | `mcp_term_definitions` |
| `worker_mechanism` | NER + 机制规划 + 知识图谱构建 | `entities / knowledge_graph / mechanism` |
| `worker_rag` | Mechanism RAG + Parameter RAG（每条边） | `parameter_provenance / rag_insights` |
| `worker_pkpd` | PK/PD 推断、剂量响应、联合用药协同 | `pkpd_profile / dose_response` |
| `worker_ode` | ODE 代码生成（Rule Engine + Jinja2） | `ode_code / reaction_graph` |
| `worker_sandbox` | AST 预检 + 数值稳定性阶梯重试 | `simulation_csv / metrics` |
| `worker_validator` | SBML 真值对比 + 可选动力学校准 | `validation_report` |
| `worker_report` | 特征提取 + 实验设计 + 证据检索 + 报告渲染 | `report_markdown / experiment_protocols / paper_evidence` |

### v4 hook chains（Feature Flag 默认 OFF，全部 OFF 时系统等价 v3）

- **ontology_hook** → **dynamic_router_hook**（worker_ode 后）
- **pathway_planner → specialist → crosstalk_coordinator**（worker_mechanism 后）
- **sbml_grounder → calibration → sensitivity**（worker_validator 后）
- **validation_pyramid**（worker_report 后）
- **hypothesis_agent**（worker_report 前）
- **Reaction IR v2 / Pathway Graph / ODE Template v2**（worker_ode 内部）

### SSE 事件契约

`/api/chat` 通过 `StreamingResponse(media_type="text/event-stream")` 推送 30+ 事件类型，包括 `agent_registry / agent_dispatch / node_start / code_generated / image_ready / rag_insights / rag_online_fallback / mcp_term_definitions / knowledge_graph / metrics / experiment_protocols / paper_evidence / v4_pathway_graph / v4_simulation_result / v4_validation_report / v4_hypothesis_list / sa_*（16 种）/ report / token_usage / clarification_needed / end`。

---

## 10 条信号通路 Specialist

每条通路提供 5 个统一接口：`apply_core / apply_feedback / apply_crosstalk / apply_perturbation / apply_validation`。

| 通路 | BioModels 参考 | 核心节点数 | 代表性扰动点 |
|---|---|---|---|
| EGFR_RTK | BIOMD0000000048 | 23 | Gefitinib / Erlotinib / Cetuximab |
| MAPK_ERK | BIOMD0000000010 | 7 | Trametinib / Vemurafenib |
| PI3K_AKT_mTOR | BIOMD0000000262 | 12 | Rapamycin / LY294002 |
| p53 | BIOMD0000000252 | 8 | Nutlin-3 |
| APOPTOSIS | BIOMD0000000102 | 9 | Venetoclax / ABT-199 |
| CELL_CYCLE | BIOMD0000000056 | 10 | Palbociclib |
| JAK_STAT | BIOMD0000000347 | 9 | Ruxolitinib |
| NF_KB | BIOMD0000000140 | 8 | Bortezomib |
| WNT | BIOMD0000000658 | 10 | ICG-001 |
| TGF_BETA | BIOMD0000000342 | 10 | Galunisertib |

`pathway_graph/initializer.py::PATHWAY_INITIALIZERS` 维护每条通路的 `core_nodes / core_edges / feedback_loops / cross_talk / perturbation_points`，节点带 `TimeScale`（FAST / MEDIUM / SLOW）与 `compartment`（extracellular / cytoplasm / nucleus / mitochondria …）。

---

## 反应中间表示（Reaction IR v2）

`backend/app/reaction_ir_v2/` 提供零 LLM 依赖的 Pydantic v2 中间表示：

- **19 类机制**：PHOSPHORYLATION / DEPHOSPHORYLATION / UBIQUITINATION / BINDING / DISSOCIATION / DIMERIZATION / COMPLEX_FORMATION / SEQUESTRATION / CLEAVAGE / GTP_GDP_EXCHANGE / TRANSCRIPTION / TRANSLATION / NUCLEAR_IMPORT / NUCLEAR_EXPORT / CYTOPLASM_TRANSLOCATION / DEGRADATION / PROTEASOMAL_DEGRADATION / INHIBITION / ACTIVATION
- **Schema**：`ReactionIRv2 / SpeciesV2 / ReactionV2 / CompositeReaction / StateMachine / Compartment / Constraint`
- **Builder**：`build_from_network_json` / `build_from_pathway_graph` / `StateMachineBuilder` / `CompositeReactionBuilder`
- **SBO 术语映射**：`ontology/sbo_terms.py` 维护 `MECHANISM_TO_SBO` / `DEFAULT_KINETICS`

---

## ODE 模板库

### v3 模板（`ode_templates/`，12 个）

`Simple_Activation` / `Simple_Inhibition` / `Cascade_Activation` / `Cascade_Inhibition` / `Signaling_Cascade_Phos` / `Combination`（联合用药）/ `DoseSweep`（剂量扫描）/ `PKPD_OneCompartment` / `PKPD_TwoCompartment` + 3 个 helper。

### v4 机制专用模板（`ode_templates_v2/`，11 个）

`_mechanism_phosphorylation_mm`（Michaelis-Menten 动力学）/ `oscillatory_feedback`（p53 / NF-κB / TGF-β / JAK-STAT 振荡，支持 DDE）/ `bistable_switch`（Apoptosis / Cell Cycle 双稳态）/ `transcriptional_delay` / `nuclear_transport` / `ubiquitination_cascade` / `destruction_complex`（Wnt β-catenin 五步耦合）/ `caspase_cascade`（Apoptosis + MOMP）/ `cyclin_cdk_toggle`（Cell Cycle + APC/C）/ `transcription_factor`（STAT / SMAD / β-catenin / NF-κB）。

模板选择由 `template_selector.py` 规则引擎完成：关键词匹配 → 机制类型投票 → SBML grounding → LLM 兜底。LLM 不直接挑选模板。

---

## RAG 知识库

`backend/app/rag_client.py::RagClient` + `rag_collections.py::RagCollections` 提供四路向量检索（ChromaDB 持久化）：

| 集合 | 用途 | 当前数据量 |
|---|---|---|
| `biodynamics_mechanism` | 通路/拓扑知识 | 70 条 |
| `biodynamics_parameter` | Kd / Km / Vmax / half-life / rate constant | 4 584 条 |
| `biodynamics_experiment` | Western / Flow / ELISA / qPCR 实验方案 | 20 条 |
| `biodynamics_evidence` | PMID / DOI / Figure / Cell Line 文献证据 | 120 条 |

检索管线：**查询重写** → **混合检索**（语义 + BM25） → **多提供商 Rerank**（讯飞 MaaS → OpenRouter → SiliconFlow） → **来源权重**（PMC > PubMed > Internal DB > Preprint）。

`drug_specific_retriever` 提供 PubMed LLM 抽取 + ChromaDB IC50/EC50 查询 + ClinicalTrials.gov v2 临床候选验证。本地命中率低于 `RAG_ONLINE_FALLBACK_THRESHOLD`（0.3）时自动触发 KEGG / Reactome / UniProt / ChEMBL 在线兜底（总预算 600 s）。

---

## MCP 生物医学术语集成

`backend/app/mcp_client.py::MCPBioClient` 注册 4 个 MCP 工具并编排 4 步术语查表：

| 工具 | 用途 |
|---|---|
| OpenBioMed Skills | 生物医学实体识别与关系抽取 |
| NIH UMLS MCP | 本体同义词与层级关系 |
| medical-terminologies-mcp | 临床术语标准化（ICD-10 / SNOMED CT） |
| pubmed-search-mcp | PubMed 增强版文献检索 |

`MCP_ENABLED=false` 或无端点 URL 时短路返回空定义并保留原 `user_input`；端点不可达时降级到 LLM 抽取 + NCBI E-utilities 直连（2 req/s 限速）。

---

## 本体层

`backend/app/ontology/` 提供 4 个本体客户端（缓存优先 TTL 7 天，3 次重试后降级）：

- `hgnc_client.query_hgnc(symbol)` → HGNC ID / UniProt ID / Entrez ID / Ensembl Gene ID
- `uniprot_client.query_uniprot(name)` → UniProt accession
- `chebi_client.query_chebi(name)` → ChEBI 化学实体 ID（配体/药物）
- `go_client.query_go(name)` → GO 功能术语列表

`OntologyAgent.annotate(user_input, v3_entities)` 编排实体抽取与多本体查表，输出 `entities`（含 `hgnc_id / uniprot_id / chebi_id / go_terms / sbo_term / verified`）与 `pathway_class`。

---

## 5 级验证金字塔

`backend/app/validation_v2/` 实现分层验证，由 `ValidationAgent` 串联：

| 级别 | 检查内容 |
|---|---|
| L1 Internal | 质量守恒（5% 阈值）/ 非负浓度 / 稳态可达性 / 数值稳定性（NaN/Inf/刚性/爆炸>1e6）/ 约束满足 |
| L2 SBML/BioModels | Track A roadrunner 真实仿真 + 通路特异阈值（peak_diff / peak_time_diff / amplification_diff）；Track B 结构相似度降级（≥0.6 通过） |
| L3 Cross-pathway | Cross-talk 一致性 / 共享物种守恒（≤10%）/ 时间尺度对齐（仅多通路触发） |
| L4 Benchmark | 通路特异 YAML 基准对比（peak_time / hill_coefficient / fold_change 等，含容差） |
| L5 Hypothesis | 假设验证（supporting PMIDs + experimental_data），统计 validated/falsified 计数 |

`thresholds.py::PathwayThresholds` 维护每通路的 L2/L4 阈值。

---

## Scientific Alignment 科学对齐模块

`backend/app/scientific_alignment/`（33 个模块，100% 规则驱动，无 LLM 调用）。设计原则：**LLM 组织与解释证据，规则引擎裁决科学正确性**。

| 子模块 | 功能 |
|---|---|
| `canonical_loader` | 10 通路 canonical 参考 YAML 加载（路径白名单 + `Path.resolve` + `relative_to` 三重防穿越） |
| `consistency_checker` | AST 安全求值 `consistency_rules`（如 EGFR 峰值时间必须早于 ERK） |
| `scientific_critic` | 6 类独立审计：机制覆盖 / 证据充分性（≥5 总、≥2 review）/ BioModels 状态 / 一致性 / 实验链 / 经典文献覆盖 |
| `multi_dim_confidence` | 6 维置信度：Mechanism / Simulation / Evidence / BioModels / Discussion / Experiment；总分 `min × 0.9` |
| `validation_rule_engine` | 4 条硬规则：质量守恒 / 峰值 vs canonical timeline / 峰值排序 / 证据数 ≥3 |
| `scientific_review` | 0–10 分 7 项评审 |
| `parameter_provenance` | 8 列溯源表：Edge · Parameter · Value · Unit · Source · Origin · Confidence · Status（来源排序 RAG > SBML > PubMed > KEGG > UniProt > ChEMBL > Inferred） |
| `explainability_log` | 8 维决策日志：Mechanism / Confidence / BioModels / Parameter / Discussion / Experiment / Validation / Cross-talk |
| `biomodels_calibration` | 与 BioModels SBML 参考做逐物种对比（fuzzy 物种匹配 + 时间归一化 + 重采样 + RMSE/Correlation） |
| `evidence_fuser` | 5 源证据融合：[A] PubMed · [B] BioModels · [C] Simulation · [D] Inference · [E] Hypothesis |
| `discussion_renderer` | 引用驱动渲染：每句 Discussion 携带单源 tag，违反时抛 `DiscussionRenderError` |
| `experiment_planner` | 10 通路 YAML 规则驱动的实验链规划，含 forbidden 模式检查（如 qPCR 不可作为磷酸化的主要验证手段） |
| `loop_controller` + `regression_monitor` | F5 修复回路控制器（最多 3 次迭代）+ 回归监控（REGRESSION / FIX / NO_CHANGE 分类 + Feature Flag 回滚建议） |

---

## BioModels Validation Oracle

`backend/app/biomodels_client.py` 提供 EBI BioModels REST API 客户端与 Validation Oracle：

- `BioModelsAPIClient.search(query)` / `download(model_id)` / `load_sbml_for_user_input(user_input)`：在线下载 + 本地缓存（SHA256 校验 + defusedxml schema 验证）
- `detect_sbml_role(...)`：4 角色 SBML 定位 — `primary_ground_truth`（建模阶段用户提供 ID）/ `calibration_reference`（自由建模校准）/ `validation_oracle`（仿真后金标准对比）/ `none`
- `run_biomodels_oracle(...)`：Track A 用 roadrunner 真实仿真，时间单位归一化（SBML 秒→分钟），逐物种峰值时间差 + per-species min-max 归一化 RMSE 对比，`overall_distance<0.3 且 max_relative_error<0.3` 判通过；Track B 在 roadrunner 不可用时降级为结构相似度评分

---

## Sandbox 沙箱执行保护

`backend/app/sandbox.py` 提供多层保护执行 LLM 生成的 Python 仿真代码：

- **AST 预检**：`ast.parse` 拦截语法错误
- **静态安全扫描**：拦截 `os/sys/subprocess/socket/pathlib/shutil/urllib/requests/http/pickle/ctypes/multiprocessing/threading/asyncio` 等模块导入，禁止 `eval/exec/compile/open/__import__`
- **import guard 注入**：子进程强制白名单导入，stderr 含 `[ImportBlocked]` 时分类为 `ERR_IMPORT_BLOCKED`
- **子进程资源限制**：Unix `RLIMIT_AS`(2GB) / `RLIMIT_CPU`(120s) / `RLIMIT_FSIZE`(100MB) / `RLIMIT_NPROC`(1)；Windows 降级为 import guard + 超时 kill
- **网络隔离**：`NO_PROXY=*` + 清空代理环境变量
- **生物学验证**：CSV 仿真后检测负浓度 / NaN / Inf / 爆炸（>1e6）/ 质量守恒
- **数值稳定性阶梯重试**：`execute_with_stability_retry` 在 LSODA 崩溃时按 收紧 max_step → BDF → Radau → QSSA 策略重试最多 4 次
- **审计日志**：`data/sandbox_logs/sandbox_YYYYMMDD.jsonl`（含 code_hash / error_class / duration）

---

## REST API

| 方法 | 端点 | 用途 |
|---|---|---|
| POST | `/api/chat` | 主 SSE 工作流（自然语言 → 报告） |
| POST | `/api/chat/respond` | 用户澄清后恢复 |
| POST | `/api/chat/stop` | 取消进行中的运行 |
| POST | `/api/chat/clear-memory` | 清空 LangGraph MemorySaver |
| GET | `/api/v4/pathways` | 列举 10 条通路（含 species_count / category / source_sbml / source_kegg） |
| GET | `/api/v4/pathways/{class}/graph` | 返回 PathwayGraphData（nodes / edges / modules） |
| POST | `/api/v4/simulation/run` | 确定性 mass-action ODE 仿真（不依赖 LLM） |
| POST | `/api/v4/simulation/sweep` | 参数扫描 |
| POST | `/api/v4/benchmark/{class}` | 单通路基准测试 |
| POST | `/api/v4/benchmarks/run` | 10 通路基准套件（SSE 流式） |
| GET | `/api/v4/reports/{id}` | 持久化报告查询 |
| GET | `/api/models/status` | LLM / Embedding / Rerank 提供商健康 |
| POST | `/api/admin/update-vector-db` | 刷新 ChromaDB 向量库 |
| GET | `/api/admin/rag-status` | RAG 集合数量 + 检索健康 |

---

## 前端

Next.js 16.2.10 / React 19.2.4，统一 Zustand store（`lib/store.ts::ingestSSEEvent` 派发 30+ 事件类型）。

| 路由 | 组件 | 功能 |
|---|---|---|
| `/` | `MinimalApp` | 自然语言输入 + 7 步工作流跟踪（Ontology → Pathway → Reaction Graph → ODE → Simulation → Validation → Report）+ 4 标签结果（Graph / Curves / Validation / Report） |
| `/advanced` | `WorkbenchShell` | 4 栏 Scientific IDE：Project 树 + Scientific Workspace（PathwayGraph / SimulationPanel / ParameterExplorer）+ Validation（ValidationPyramid + ScientificAlignmentPanel + HypothesisPanel）+ AI Assistant |
| `/benchmarks` | `BenchmarkCenter` | 10 通路金标准套件，Summary bar + Run All + Stop + 进度条，每张 `BenchmarkCard` 展示 pass criteria 清单 |
| `/report/[id]` | `ReportViewer` | 持久化仿真报告查看器 |

**主要面板**：
- `ValidationPyramid` — 5 级金字塔（L1 Network / L2 GitCompare / L3 Network / L4 BarChart3 / L5 Lightbulb），按 `level1`~`level5` 报告字段派生 pass/warning/fail/skipped
- `ScientificAlignmentPanel` — 9 张可折叠卡片（Consistency Gate / SA-1~SA-7 / SA-F BioModels Calibration），颜色编码 pass=emerald / warning=amber / fail=red / skipped=zinc
- `HypothesisPanel` — 假设卡（statement + strategy 徽章 + 置信度条 + falsifiable 徽章 + Evidence PMID 链表 + Suggested Experiments + Popper 3 规则 falsifiability 检查）

---

## 测试与 CI

`backend/tests/` 共 80+ pytest 文件，覆盖：10 个 specialist、Validation Pyramid L1–L5、Reaction IR v2、SBML grounder、calibration、sensitivity、hypothesis、pathway planner、dynamic router、feature-flag convergence、P0/P1 critical fixes、RAG-ODE 集成、sandbox 持久化等。

`verification/` 提供 8 个科学验证子套件：`pathway_regression / parameter_stress / ontology_validation / solver_validation / benchmark / hypothesis_validation / biomodels_regression / ui_workflow`（Playwright）。

Makefile：`make test` / `make test-integration` / `make test-benchmark` / `make ci`。

GitHub Actions：`.github/workflows/ci.yml` + `scientific-regression.yml`（含 `scientific-alignment-benchmark` 门禁 job）。

---

## 多提供商 LLM / Embedding / Rerank

`backend/app/config.py` 通过 `FallbackLLM`（0.5s 切换，`max_retries=0`）串联主备 LLM；Embedding 支持 OpenAI / local（sentence-transformers）/ OpenRouter / SiliconFlow / 讯飞 MaaS；Rerank 由 `RerankManager` 管理 3 提供商级联（默认优先级：讯飞 MaaS → OpenRouter → SiliconFlow）。具体 provider/model 值从 `backend/.env` 读取。

---

## 项目结构

```
bio-dynamics-agent/
├── backend/
│   ├── app/
│   │   ├── main.py                 FastAPI 入口 + /api/chat SSE + SA 后处理
│   │   ├── graph_v3.py             LangGraph v3 工作流 + v4 hook chains
│   │   ├── state.py                BioDynamicsState + v4_state 容器
│   │   ├── config.py               Settings + 3 粗 / 13 细 v4 flags + 18 SA flags
│   │   ├── supervisor.py           AGENT_REGISTRY (v1: 6) + AGENT_REGISTRY_V2 (v2: 10)
│   │   ├── nodes_v2.py             N0–N11 流水线节点
│   │   ├── v4_endpoints.py         7 个 v4 REST 端点
│   │   ├── rag_client.py           RagClient（混合检索 + rerank + ClinicalTrials.gov）
│   │   ├── rag_collections.py      四路 RAG 集合
│   │   ├── mcp_client.py           4 MCP 工具 + LLM/E-utilities 降级
│   │   ├── biomodels_client.py     EBI BioModels API + Validation Oracle
│   │   ├── sandbox.py              AST 预检 + 资源限制 + 阶梯重试
│   │   ├── ontology/               4 本体客户端 + SBO 术语
│   │   ├── reaction_ir_v2/         19 类机制 + Pydantic v2 schema
│   │   ├── pathway_graph/          PathwayGraph builder + 10 通路 initializer
│   │   ├── ode_templates(_v2)/     Jinja2 ODE 模板（v3: 12, v4: 11）
│   │   ├── pathways/specialists/   10 通路 specialist
│   │   ├── crosstalk/              Cross-talk 协调器 + 共享物种同步
│   │   ├── sbml_grounder/          5 级映射链 (ODE ↔ Reaction ↔ SBML ↔ Parameter ↔ PMID)
│   │   ├── validation_v2/          5 级验证金字塔
│   │   ├── calibration/            最小二乘 + 置信区间
│   │   ├── sensitivity/            local + Sobol + Morris
│   │   ├── hypothesis/             假设生成 + falsifiability 检查
│   │   ├── scientific_alignment/   SA 模块（33 文件，100% 规则驱动）
│   │   ├── solvers/                DDE / 振荡检测 / 双稳态检测
│   │   └── reliability/            熔断器 / 重试 / fail-safe / 结构化日志
│   ├── benchmarks/golden/          10 通路金标准 YAML + expected + metrics
│   ├── knowledge/canonical/        10 通路 canonical 参考
│   ├── knowledge/experiments/      10 通路实验链规则
│   ├── knowledge/gold_standard/    10 通路文献金标准
│   ├── tests/                      80+ pytest 文件
│   └── data/                       ChromaDB + BioModels SBML + PMID 缓存
├── frontend/
│   ├── app/                        4 个路由入口
│   ├── components/                 minimal / workspace / scientific_alignment /
│   │                               validation / hypothesis / benchmark
│   ├── lib/                        store.ts (Zustand) / api.ts / sse.ts
│   └── __tests__/ + e2e/           vitest + Playwright
├── verification/                   8 个科学验证子套件
├── .github/workflows/              ci.yml + scientific-regression.yml
├── Dockerfile                      多阶段构建（Python backend + Node frontend）
└── Makefile                        test / test-integration / test-benchmark / ci
```

---

## License

MIT. 见 [LICENSE](./LICENSE)。
