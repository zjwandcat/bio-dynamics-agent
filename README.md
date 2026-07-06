# BioDynamics Agent

> 把生物医学研究者的**定性假说**（Qualitative Hypothesis）一键转化为**定量 ODE 仿真**（Ordinary Differential Equation，常微分方程），并直接给出可视化预测与机理论证。**v3 架构**采用 **Supervisor-Worker 动态编排** + **三档运行模式**（Auto Fast / Auto Standard / Manual）+ **人在环路干预** + **上下文渐进式压缩** + 8 大 Worker 模块化流水线 + 高阶 RAG 知识注入 + MCP 术语标准化 + AST 校验沙箱 + Jinja2 模板驱动代码生成 + 四路 RAG + Rule Engine + 9 维科学特征提取 + 模板化 Markdown 报告。**P0-3 修复升级**新增 **TemplateSelectorSkill 规则引擎**（8 条优先级规则，强制模板选择）+ **Reaction IR 中间表示**（KG → Reaction Graph → Template → ODE 确定性管线）+ **DomainChecker 多维领域审查**（物理/生物/化学/医学）+ **BioModels REST API 客户端**（自动加载 SBML）+ **Signaling_Cascade_Phos 质量守恒模板**（mass-action binding + phosphorylation）+ **时间尺度分层**（磷酸级联 120 min / PKPD 48 h）+ **4-collection RAG 上下文富化**。**P0-4 修复升级**新增 **SBML 三角色定位**（Primary Ground Truth / Calibration Reference / Validation Oracle）+ **SBMLValidator 双轨验证器**（libroadrunner 真实仿真或参数对齐法兜底）+ **worker_validator 节点**（sandbox 后对比真实 SBML 仿真，输出 error_diff / peak_time_diff / amplification_diff）。PK/PD 模块支持药代动力学-药效学推断、靶点-药物真实世界知识图谱、联合用药协同/拮抗评估、剂量递增与治疗窗口分析。全栈 AI 智能体，覆盖 Web 前后端 + Supervisor-Worker 动态调度 + 人在环路 + MCP 术语标准化 + 高阶 RAG 知识注入 + 沙箱安全执行。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-green)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/MCP-integrated-purple)](https://modelcontextprotocol.io/)

---

## 目录

- [项目简介](#项目简介)
- [它能解决什么问题](#它能解决什么问题)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [前置环境](#前置环境)
  - [一键启动（Windows）](#一键启动windows)
  - [手动启动](#手动启动)
  - [环境变量配置](#环境变量配置)
- [使用指南](#使用指南)
- [v3 架构：Supervisor-Worker 动态编排](#v3-架构supervisor-worker-动态编排)
  - [三档运行模式](#三档运行模式)
  - [8 大 Worker 模块](#8-大-worker-模块)
  - [Supervisor 动态调度](#supervisor-动态调度)
  - [人在环路干预](#人在环路干预)
  - [上下文渐进式压缩](#上下文渐进式压缩)
- [P0-3 修复升级：Reaction-First Biomedical Simulation Compiler](#p0-3-修复升级reaction-first-biomedical-simulation-compiler)
- [P0-4 修复升级：SBML Validator 与三角色定位](#p0-4-修复升级sbml-validator-与三角色定位)
- [多智能体编排](#多智能体编排)
- [MCP 工具集成](#mcp-工具集成)
- [RAG 知识库与 SBML 复用](#rag-知识库与-sbml-复用)
- [PK/PD 建模与药物研发分析](#pkpd-建模与药物研发分析)
- [安全沙箱说明](#安全沙箱说明)
- [API 接口](#api-接口)
- [离线 RAG 知识库构建](#离线-rag-知识库构建)
- [端到端测试](#端到端测试)
- [常见问题](#常见问题)
- [开发规范](#开发规范)
- [许可证](#许可证)
- [致谢](#致谢)

---

## 项目简介

**BioDynamics Agent** 是一个面向**计算系统生物学**、**转化医学**与**药物研发**的端到端 AI 智能体平台。研究人员只需在 Web 聊天框中输入一段自然语言描述的**生物机制或药物假说**（例如 *"Galunisertib（TGF-β 抑制剂，IC50=51 nM）抑制 TGF-β 信号通路，并评估其与 PD-1 抗体联用对 CD8⁺ T 细胞恢复的协同效应"*），后端 **v3 Supervisor-Worker 动态编排引擎**会自动完成以下闭环：

**v3 流程（Supervisor-Worker 动态编排）**

1. **PreRouter** — 根据运行模式（Auto Fast / Auto Standard / Manual）生成 Worker 执行计划
2. **Supervisor** — 动态调度下一个 Worker，检查是否触发人在环路干预
3. **Worker: MCP 术语标准化** — 调用生物医学 MCP 工具标准化术语，注入定义上下文，重写查询
4. **Worker: 机制解析与图谱** — NER 实体归一化 → 机制规划 → 知识图谱构建（拓扑签名 / 环路检测）
5. **Worker: 知识检索 (RAG)** — Mechanism RAG + Parameter RAG（混合检索 + 重排序 + LLM 决策）；对 inhibition 边检索靶点药物候选
6. **Worker: PK/PD 推断** — 推断给药途径、房室模型、PK/PD 参数，输出药物方案
7. **Worker: ODE 方程生成** — LLM 输出 Network JSON → Jinja2 模板渲染 → Rule Engine 校验
8. **Worker: 沙箱仿真执行** — AST 预检 + 安全扫描 + subprocess 隔离执行 + 错误分类 + 审计重试（模式相关重试次数）
9. **Worker: 预测报告生成** — 9 维科学特征提取 + 实验方案 RAG + 文献证据 RAG + 模板化 Markdown 报告
10. **ClarificationNode**（条件触发）— 参数全缺失 / 知识图谱环路 / 建模方案分歧时，阻塞等待用户干预

整个过程通过 **SSE（Server-Sent Events）** 实时把**多智能体调度事件、节点执行状态、MCP 工具调用、RAG 洞察数据、PK/PD 卡片、剂量-反应曲线、联合用药协同结论、人在环路干预请求、Token 消耗、最终代码、图表与报告**推送到前端。v3 下发 `workflow_v3_state` / `clarification_needed` / `clarification_resolved` 等结构化事件，前端右侧控制栏实时展示运行模式、模块勾选、当前节点与干预对话框。

---

## 它能解决什么问题

生物医学研究长期存在 **"定性假说 ↔ 定量模型"鸿沟**：

| 痛点 | BioDynamics Agent 的解决方案 |
| --- | --- |
| 实验生物学家不懂 ODE / Hill 方程 / SciPy，无法把假说变成可执行代码 | 自然语言输入 → LLM 自动生成可执行的 `scipy.integrate.solve_ivp` 仿真代码 |
| 同一术语多种写法（TGF-β / TGF-beta / TGFB1），检索与建模口径不一 | Worker MCP 调用工具标准化术语，注入定义并重写查询 |
| 手动写 ODE 容易给出脱离文献的参数（Kd、Km、半衰期等） | 高阶 RAG：查询重写 + BM25 + 语义混合检索 + 来源权威性重排序，自动命中 PubMed 真实动力学参数 |
| 药物研发场景缺乏药物本身信息（IC50、EC50、临床剂量、CT.gov） | `drug_specific_retriever`：PubMed 提取 + ChromaDB 补充 + ClinicalTrials.gov 验证，输出靶点-药物知识图谱 |
| 传统 ODE 不包含药代动力学，无法做剂量-反应分析 | Worker PK/PD 推断房室模型与 Emax 参数，Worker Sandbox 默认执行 0.1×~100× EC50 剂量递增 |
| 联合用药协同/拮抗需要手动计算 | Worker Sandbox 自动基于 Chou-Talalay 中效方程计算 CI，输出"潜在协同 / 叠加效应 / 拮抗风险" |
| 已有 SBML 模型无法直接复用为新假说 | `SBML_PARSER_PROMPT` 解析 SBML 文本，提取可复用网络拓扑 |
| LLM 生成的代码可能存在 `os.system`、`eval`、网络访问等危险调用 | 沙箱静态黑名单扫描（`os/sys/subprocess/socket/...`）+ AST 预检 |
| 仿真失败（语法错、NaN、负浓度）需人工排查 | Worker Sandbox 内置审计重试（Auto Fast 1 次 / Auto Standard 3 次 / Manual 3 次） |
| 参数全部缺失时盲目仿真结果不可信 | Supervisor 触发人在环路干预，用户可选择继续估算 / 降低置信度 / 自定义方案 |
| 知识图谱存在反馈环路导致建模矛盾 | Supervisor 检测环路并触发干预，用户可选择稳态近似 / 重新解析 / 自定义 |
| 抑制关系建模方式有分歧（Emax vs 线性） | Supervisor 检测建模歧义并触发干预，用户可选择 Sigmoid Emax / 线性抑制 / 自定义 |
| 主 LLM 限流/过载导致整个流水线中断 | `FallbackLLM` 主备自动切换，主模型失败时无缝切到备用模型 |
| 仿真结果缺乏机理论证 | Worker Report 自动撰写 Markdown 预测报告 + 联合用药方案 + 剂量建议 + 湿实验验证建议 |
| 不同场景需要不同粒度的分析 | 三档运行模式：Auto Fast（极简直跑）/ Auto Standard（LLM 动态裁剪）/ Manual（用户勾选模块） |

### 典型应用场景

- **免疫治疗机制探索**：TGF-β / PD-1 / CD8⁺ T 细胞相互作用建模
- **肿瘤药物研发与 PK/PD 仿真**：抑制剂 IC50/EC50、房室模型、剂量递增、治疗窗口、人用等效剂量 HED
- **联合用药方案评估**：双药 Chou-Talalay CI 协同/拮抗分析，给出组合策略建议
- **肿瘤生长与药物响应**：信号通路 + 给药动力学联合仿真
- **传染病动力学**：宿主-病原体耦合 ODE 预测
- **教学与科普**：把教科书上的通路图变成可交互的时间序列图 + 剂量-反应曲线

---

## 技术栈

### 前端（[frontend/](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend)）

| 类别 | 技术 | 用途 |
| --- | --- | --- |
| 框架 | **Next.js 16**（App Router + Turbopack） | React 19 全栈框架 |
| UI 库 | **Shadcn UI** + **Tailwind CSS 4** + **Radix UI** | 暗色全屏聊天界面 |
| 动效 | **Framer Motion** | 消息进入 / 状态切换动画 |
| 渲染 | **react-markdown** + **react-syntax-highlighter** + **remark-gfm** | Markdown / 代码高亮 / 表格 |
| 图表 | **recharts 3** | RAG 来源分布、剂量-反应曲线等数据可视化 |
| 图标 | **lucide-react** | UI 图标 |
| 类型 | **TypeScript 5** | 全量静态类型 |

前端布局：**70% 聊天区域 + 30% 右侧控制栏**（最小宽度 360px）

前端可视化组件（[frontend/components/chat/](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/components/chat)）：

| 组件 | 作用 |
| --- | --- |
| `ControlBar` | **v3 核心**：右侧控制栏，三档模式切换、8 个模块勾选、当前节点展示、干预对话框宿主 |
| `ClarificationDialog` | **v3 核心**：人在环路干预面板，ABC 选项 + 自定义文本 + 停止按钮 |
| `AgentWorkflowTracker` | 多智能体工作流追踪器，按 `agent_dispatch` 事件点亮各 Agent 状态与延迟 |
| `RAGInsightPanel` | RAG 洞察面板：重写查询、来源分布、Top 候选参数、**药物候选（知识图谱）** |
| `MCPToolPanel` | MCP 工具调用状态面板：每次工具调用的动作 / 状态 / 延迟 / Token 节省 |
| `TermDefinitionCard` | 术语定义卡片：标准化名、定义、同义词、相关通路 |
| `TokenPerformanceBadge` | 性能徽标：RAG 命中率、延迟、Token 消耗、MCP Token 节省 |
| `DoseResponseCurve` | 剂量-反应曲线（Sigmoid Emax）：标出 IC50 / IC90 / HED，支持 log 浓度轴 |
| `ChatMessage` | 消息气泡，渲染 Markdown / 图表 / PK/PD 卡片 / 剂量曲线 / 协同评估 / Token 性能徽标 / KG 摘要 / Metrics 表格 / 实验方案 / 文献证据 / Markdown 报告 |
| `ChatInput` | 输入框（Enter 发送 / Shift+Enter 换行） |
| `WorkflowVisualization` | 流水线可视化 stepper |

### 后端（[backend/](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend)）

| 类别 | 技术 | 用途 |
| --- | --- | --- |
| Web 框架 | **FastAPI 0.139** + **Uvicorn 0.49** | REST + SSE 流式接口 |
| Agent 编排 | **LangGraph 1.2** + **LangChain 1.3** | Supervisor-Worker 动态状态机工作流 |
| 模板引擎 | **Jinja2 3.1** | ODE 与报告模板（LLM 禁止写代码，模板渲染统一生成可执行 Python） |
| 多智能体 | **app/supervisor.py**（`AGENT_REGISTRY` / `AGENT_REGISTRY_V2` + `BioDynamicsOrchestrator`） | 智能体注册表 + 调度事件生成 |
| MCP 协议 | **langchain-protocol 0.0** | 连接生物医学 MCP 工具（端点缺失时 LLM 降级） |
| 检查点 | **MemorySaver**（in-memory checkpoint） | 多轮对话短期记忆 |
| LLM | **OpenAI 兼容 API** + **FallbackLLM 主备切换** | 实体归一化、机制规划、ODE 关系抽取、报告字段填充 |
| Embedding | **5 个 Provider 可切换**：讯飞 xop3qwen8bembedding（768d，默认）/ SiliconFlow BAAI/bge-m3（1024d）/ OpenRouter nvidia/llama-nemotron-embed（2048d）/ OpenAI text-embedding-3-small / sentence-transformers 本地模型 | RAG 向量化（生物医学跨语言检索） |
| Rerank | **3 个 Provider 可切换**：讯飞 xop3qwen8breranker / SiliconFlow BAAI/bge-reranker-v2-m3 / OpenRouter cohere/rerank-4-pro | 混合检索后重排序（auto/llm 两种选择模式） |
| 向量库 | **ChromaDB 1.5**（本地持久化，无需 Docker） | 4 路 collection（mechanism / parameter / experiment / evidence） |
| 数值计算 | **NumPy 2.5** + **SciPy 1.18**（`solve_ivp`）+ **Matplotlib 3.11** | ODE 求解、可视化、9 维科学特征提取 |
| 模板引擎 | **Jinja2 3.1**（8 个 ODE 模板 + 1 个报告模板） | LLM 禁止写代码，模板渲染统一生成可执行 Python |
| 规则引擎 | **TemplateSelectorSkill**（8 条优先级规则） | 强制模板选择：关键词匹配 > mechanism 投票 > SBML grounding > LLM 兜底 |
| 领域审查 | **DomainChecker**（物理/生物/化学/医学 4 维） | 质量守恒 / 负浓度 / 信号放大 / 受体守恒 / 时间尺度 |
| BioModels | **EBI BioModels REST API** | 自动加载 SBML 模型（BIOMD* ID 识别） |
| 代码安全 | **ast.parse** + 静态黑名单 + 错误分类 | 沙箱前置 AST 预检，区分 syntax / runtime / numerical / timeout |
| HTTP | **requests** + **httpx** | PubMed E-utilities / OpenAI 客户端 / ClinicalTrials.gov v2 |
| 沙箱 | **subprocess + tempfile + 静态黑名单** | 隔离执行 LLM 生成的代码 |
| 配置 | **pydantic-settings** + **python-dotenv** | `.env` 环境变量加载 |

> 离线 RAG 知识库构建脚本默认使用 `requests` + `xml.etree.ElementTree` 解析 PubMed XML，**不依赖 biopython**（Python 3.14 兼容性更稳）。BM25 在 [app/rag_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/rag_client.py) 内联实现（`_BM25` 类），避免外部 rank-bm25 依赖。

---

## 项目结构

```text
bio-dynamics-agent/
├── README.md                     # 本文件
├── LICENSE                       # MIT 开源协议
├── ARCHITECTURE.md               # 架构详解（12 节点 / 状态字段 / 错误分类 / 模板 / Rule / 报告渲染）
├── TEMPLATES.md                  # Jinja2 模板参考（8 个 ODE 模板 + 1 个报告模板）
├── start-dev.bat                 # Windows 一键启动脚本
│
├── backend/                      # Python 后端
│   ├── .env.example              # 环境变量示例（拷贝为 .env 后填值）
│   ├── requirements.txt          # 依赖清单（含 jinja2==3.1.4）
│   ├── debug_e2e.py              # 端到端调试脚本
│   ├── app/                      # 应用代码
│   │   ├── main.py               # FastAPI 入口（SSE /api/chat + /api/chat/respond + /api/chat/stop）
│   │   ├── graph_v3.py           # **v3 核心** Supervisor-Worker 动态编排图（PreRouter / Supervisor / 7 Worker / ClarificationNode）
│   │   ├── context_compressor.py # **v3 核心** 上下文渐进式压缩模块（防止超大结构化数据撑爆 LLM 上下文）
│   │   ├── state.py              # BioDynamicsState 状态定义（含 v3 字段 + P0-3 字段：sbml_model_id / reaction_graph / template_selection）
│   │   ├── nodes.py              # v1 节点实现（复用为 v3 Worker 内部逻辑：MCP / 解析 / PK/PD / 生成 / 审计）
│   │   ├── nodes_v2.py           # v2 节点实现（N0 SBML Loader + N1 NER → N11 报告；含 _unique_species_from_edges 修复）
│   │   ├── supervisor.py         # 多智能体编排（AGENT_REGISTRY + AGENT_REGISTRY_V2 + 调度事件）
│   │   ├── schemas.py            # Pydantic 请求 / 响应模型（含 ClarificationResponseRequest / StopRequest）
│   │   ├── prompts.py            # System Prompt
│   │   ├── prompts_v2.py         # 11 套 Prompt（Few-shot / Bad / Good / Negative Constraints + Template 白名单 + 硬规则）
│   │   ├── config.py             # 全局配置 + 4 个 Chroma collection 名 + 5 个 Embedding provider
│   │   ├── mcp_client.py         # MCP 工具封装（4 个工具 + LLM 降级 + ToolCallRecord）
│   │   ├── rag_client.py         # ChromaDB 单库（查询重写 + BM25 + 语义 + 混合 + 重排序 + drug_specific_retriever + 4-collection 上下文富化）
│   │   ├── rag_collections.py    # 四路 RAG 集合（mechanism / parameter / experiment / evidence）
│   │   ├── sbml_parser.py        # SBML → 网络拓扑结构化抽取
│   │   ├── biomodels_client.py   # **P0-3 新增** EBI BioModels REST API 客户端（BIOMD* ID 自动加载 SBML + 本地缓存）
│   │   ├── template_selector.py # **P0-3 新增** TemplateSelectorSkill 规则引擎（8 条优先级规则 + 时间尺度分层）
│   │   ├── reaction_ir.py       # **P0-3 新增** Reaction IR 中间表示（KG → Reaction Graph → Template → ODE 确定性管线）
│   │   ├── domain_checker.py    # **P0-3 新增** DomainChecker 多维领域审查（物理 / 生物 / 化学 / 医学）
│   │   ├── sbml_validator.py   # **P0-4 新增** SBMLValidator 双轨验证器（libroadrunner / 参数对齐法 + 三角色定位）
│   │   ├── sandbox.py            # 沙箱（AST 预检 / 错误分类 / CSV 收集 / 剂量递增 / 联合用药解析）
│   │   ├── kg_builder.py         # 知识图谱构建器（节点 / 边 / 拓扑签名 / 环路检测）
│   │   ├── rule_engine.py        # Rule Engine（6 条 Rule：模板 / 参数范围 / 单位 / 方向 / Hill / 初值）
│   │   ├── feature_extractor.py  # 科学特征提取器（9 维指标，纯 NumPy）
│   │   ├── report_renderer.py    # 报告渲染器（LLM JSON Fill + 禁止词校验 + Jinja2 拼装）
│   │   ├── ode_templates/        # 8 个 Jinja2 ODE 模板（含 **Signaling_Cascade_Phos** 质量守恒模板）
│   │   ├── report_templates/     # 1 个报告模板（standard.md.j2）
│   │   ├── local_embeddings.py   # 离线 Embedding 兜底（sentence-transformers）
│   │   └── token_usage.py        # Token 累加与归一化
│   ├── scripts/                  # 离线工具脚本
│   │   ├── fetch_rag_data.py     # 从 PubMed 抓取文献片段
│   │   ├── embed_data.py         # 文献 → 参数结构化抽取
│   │   ├── build_rag_db.py       # 一键构建 RAG 知识库（需 --queries 参数）
│   │   ├── seed_collections.py   # 把历史数据迁移到 4 路 collection
│   │   └── update_vector_db.py   # 增量更新 ChromaDB
│   ├── tests/                    # 端到端与单元测试
│   │   ├── e2e_test.py           # 端到端
│   │   ├── debug_retry.py        # 审计重试调试
│   │   ├── test_upgrade.py       # 端到端测试（TGF-β / SMAD / CD8 / AST 重试 / RAG 缺参 fallback）
│   │   ├── test_rule_engine.py   # 6 条 Rule 单元测试
│   │   ├── test_feature_extractor.py  # 9 维指标单元测试
│   │   ├── test_report_renderer.py    # 禁止词 + Jinja2 渲染
│   │   ├── test_ode_templates.py      # 8 模板渲染断言
│   │   ├── test_rag_collections.py    # 四路 RAG 高层 API
│   │   └── test_sandbox_v2.py         # AST / 错误分类 / NaN 检测
│   └── data/                     # （首次运行后生成）原始文献 / ChromaDB 持久化目录
│
└── frontend/                     # Next.js 前端
    ├── package.json
    ├── tsconfig.json
    ├── components.json           # Shadcn UI 配置
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx              # 主页（70% 聊天 + 30% 控制栏；SSE 事件分发中枢）
    │   └── globals.css
    └── components/
        ├── chat/
        │   ├── ChatMessage.tsx              # 消息气泡（Markdown / 图表 / PK/PD 卡片 / 剂量曲线 / 协同评估 / KG / Metrics / 实验 / 证据 / 报告）
        │   ├── ChatInput.tsx                # 输入框（Enter 发送 / Shift+Enter 换行）
        │   ├── ControlBar.tsx               # **v3** 右侧控制栏（模式切换 / 模块勾选 / 当前节点 / 干预对话框宿主）
        │   ├── ClarificationDialog.tsx      # **v3** 人在环路干预面板（ABC 选项 + 自定义文本 + 停止）
        │   ├── AgentWorkflowTracker.tsx     # 多智能体工作流追踪器
        │   ├── WorkflowVisualization.tsx    # 流水线可视化
        │   ├── RAGInsightPanel.tsx          # RAG 洞察面板（含药物候选知识图谱）
        │   ├── MCPToolPanel.tsx             # MCP 工具调用状态面板
        │   ├── TermDefinitionCard.tsx       # 术语定义卡片
        │   ├── TokenPerformanceBadge.tsx    # Token 与性能徽标
        │   └── DoseResponseCurve.tsx        # 剂量-反应曲线（IC50 / IC90 / HED）
        └── ui/                   # Shadcn UI 原子组件
```

---

## 快速开始

### 前置环境

- **Python** ≥ 3.11（推荐 3.12，已在 3.14 验证可跑）
- **Node.js** ≥ 20
- **npm** ≥ 10（或 pnpm / yarn / bun）
- 一份 **OpenAI 兼容的 API Key**（官方 OpenAI、智谱 BigModel、DeepSeek、月之暗面等任一）
- （可选）**本地 Embedding**：`EMBEDDING_PROVIDER=local` 时首次运行会自动下载 `sentence-transformers/all-MiniLM-L6-v2`
- 本项目默认走 **ChromaDB 本地持久化**，**无需 Docker**

### 一键启动（Windows）

```powershell
# 1. 进入项目根目录
cd c:\Users\27553\Desktop\gzlab\bio-dynamics-agent

# 2. 拷贝并填写环境变量
copy backend\.env.example backend\.env
# 用任意编辑器打开 backend\.env，填入 OPENAI_API_KEY 等

# 3. 双击或命令行执行
.\start-dev.bat
```

脚本会自动：检测 `backend\venv\`、后台拉起后端 Uvicorn（端口 8000）和前端 Next.js（端口 3000）、等待后自动打开浏览器。

### 手动启动

#### 1. 启动后端

```powershell
cd backend

# 创建虚拟环境（首次）
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt

# 启动 FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

后端启动后访问 [http://localhost:8000](http://localhost:8000) 应返回：

```json
{"status":"ok","service":"BioDynamics Agent"}
```

#### 2. 启动前端

```powershell
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端访问 [http://localhost:3000](http://localhost:3000)。

### 环境变量配置

复制 [backend/.env.example](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/.env.example) 为 `backend/.env` 并填入真实值。所有变量均由 [app/config.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/config.py) 读取：

```ini
# —— LLM（OpenAI 兼容）——
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1     # 或 https://open.bigmodel.cn/api/paas/v4
OPENAI_MODEL=gpt-4o                            # 或 glm-4.7-flash、deepseek-chat 等

# —— 备用 LLM（主模型失败时自动切换，留空则不启用）——
BACKUP_API_KEY=
BACKUP_BASE_URL=
BACKUP_MODEL=

# —— 服务 ——
HOST=0.0.0.0
PORT=8000
FRONTEND_URL=http://localhost:3000

# —— ChromaDB（本地持久化，无需 Docker）——
CHROMA_PERSIST_DIR=./data/vector_db
CHROMA_COLLECTION_NAME=biodynamics_params

# 四路 RAG collection
CHROMA_COLLECTION_MECHANISM=biodynamics_mechanism
CHROMA_COLLECTION_PARAMETER=biodynamics_parameter
CHROMA_COLLECTION_EXPERIMENT=biodynamics_experiment
CHROMA_COLLECTION_EVIDENCE=biodynamics_evidence

# —— Embedding（5 个 Provider 可切换）——
# openai（text-embedding-3-small）| local（sentence-transformers）| openrouter（nvidia/llama-nemotron-embed）| siliconflow（BAAI/bge-m3）| xfyun（xop3qwen8bembedding，默认推荐）
EMBEDDING_PROVIDER=xfyun
EMBEDDING_MODEL=xop3qwen8bembedding
EMBEDDING_BASE_URL=                            # 留空则复用 OPENAI_*
EMBEDDING_API_KEY=                             # 留空则复用 OPENAI_*

# —— OpenRouter 专用配置（Embedding + Rerank）——
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
OPENROUTER_RERANK_MODELS=cohere/rerank-4-pro

# —— SiliconFlow 专用配置（Embedding + Rerank）——
SILICONFLOW_API_KEY=
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_EMBEDDING_MODEL=BAAI/bge-m3
SILICONFLOW_RERANK_MODELS=BAAI/bge-reranker-v2-m3

# —— 讯飞 MaaS 专用配置（Embedding + Rerank，默认推荐）——
XFYUN_MAAS_API_KEY=                            # 格式：appid:api_secret
XFYUN_MAAS_EMBEDDING_BASE_URL=https://maas-api.cn-huabei-1.xf-yun.com/v2
XFYUN_MAAS_RERANK_BASE_URL=https://maas-api.cn-huabei-1.xf-yun.com/v2
XFYUN_MAAS_EMBEDDING_MODEL=xop3qwen8bembedding
XFYUN_MAAS_RERANK_MODELS=xop3qwen8breranker

# —— Rerank 策略（rule 启发式 | model API 调用 | hybrid 混合）——
RERANK_PROVIDER=model
RERANK_PROVIDERS=xfyun,openrouter,siliconflow  # 逗号分隔，按优先级尝试
RERANK_SELECTION_MODE=auto                     # auto 按优先级选 | llm 让 LLM 选

# —— RAG 在线数据库自动补充 ——
RAG_ONLINE_FALLBACK=true                       # 本地 ChromaDB 命中不足时查询 KEGG/Reactome/UniProt/ChEMBL
RAG_ONLINE_FALLBACK_THRESHOLD=0.3              # 语义相似度低于此阈值时触发在线补充

# —— PubMed E-utilities 联系邮箱（NCBI 要求，用于离线建库）——
NCBI_EMAIL=your-email@example.com

# —— MCP 工具端点（留空则自动降级为 LLM 内部知识）——
MCP_ENABLED=true
MCP_OPENBIOMED_URL=
MCP_MEDTERM_URL=
MCP_PUBMED_URL=
MCP_UMLS_URL=
```

---

## 使用指南

1. 打开 [http://localhost:3000](http://localhost:3000)
2. **右侧控制栏**选择运行模式：
   - **Auto Fast**：极简链路（机制 → ODE → 沙箱 → 报告），单智能体直跑，跳过 MCP/RAG/PKPD
   - **Auto Standard**（默认）：LLM 动态判断是否需要 PK/PD 与实验证据，自动裁剪执行计划
   - **Manual**：手动勾选所需模块（术语标准化 / 机制解析 / RAG / PK/PD / 沙箱 / 剂量分析 / 实验文献 / 报告）
3. 在底部输入框用自然语言描述生物机制或药物假说，例如：
   > *"TGF-β 抑制 CD8⁺ T 细胞活性，抑制强度随时间累积；当 TGF-β 浓度达到 5 nM 时，T 细胞完全失活。"*

   或 PK/PD / 联合用药场景：
   > *"Galunisertib（TGF-β 抑制剂，IC50=51 nM）抑制 TGF-β 信号通路，评估其与 PD-1 抗体联用对 CD8⁺ T 细胞恢复的协同效应。"*
4. 点击 **发送**（或按 Enter；Shift+Enter 换行）
5. 顶部**智能体工作流追踪器**会按 `agent_dispatch` 事件依次点亮各 Agent 状态与延迟
6. 消息流会依次推送：
   - **MCP 工具调用面板**（每个工具的动作 / 状态 / 延迟 / Token 节省）
   - **术语定义卡片**（标准化名、定义、同义词、相关通路）
   - **RAG 洞察面板**（重写查询、来源分布、Top 候选参数、**药物候选知识图谱**、命中率）
   - **PK/PD 模型卡片**（给药途径、房室模型、PK/PD 参数）
   - **ODE 代码块**（可一键复制）
   - **仿真结果图表**（Base64 PNG，可右键保存）
   - **剂量-反应曲线**（Sigmoid Emax，标出 IC50 / IC90 / HED）
   - **联合用药协同评估**（Chou-Talalay CI、协同/拮抗结论）
   - **Markdown 预测报告**（趋势、机制、联合方案、剂量建议、实验验证建议，附带 Token 性能徽标）
7. **人在环路干预**（Auto Standard / Manual 模式下）：
   - 当参数全部缺失、知识图谱存在环路、或建模方案有分歧时，控制栏会弹出**干预对话框**
   - 选择 A / B / C 选项（或自定义文本），点击提交，工作流继续执行
   - 也可点击**停止**终止当前工作流
8. 顶部按钮：
   - **更新知识库**：后台触发 ChromaDB 增量更新
   - **清除当前对话**：清空当前 `thread_id` 的短期记忆

---

## v3 架构：Supervisor-Worker 动态编排

v3 取代了 v1/v2 的硬编码线性图，采用 **Supervisor-Worker 动态编排**模式。核心文件：[app/graph_v3.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/graph_v3.py)。

### 三档运行模式

由 [PreRouter](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/graph_v3.py) 在 `pre_router` 节点根据 `state.mode` 生成 `execution_plan`：

| 模式 | 执行计划 | 特点 |
| --- | --- | --- |
| **Auto Fast** | `mechanism → ode → sandbox → report` | 极简链路，跳过 MCP/RAG/PKPD/证据检索；沙箱最多重试 1 次 |
| **Auto Standard** | `mcp → mechanism → rag → [pkpd] → ode → sandbox → report` | 默认模式；先用规则检测是否需要 PK/PD，规则无法判定时调用 LLM；沙箱最多重试 3 次 |
| **Manual** | 按用户勾选模块生成，自动补全依赖 | 用户在前端控制栏勾选所需模块；依赖自动补全（如选报告 → 自动加沙箱+机制）；沙箱最多重试 3 次 |

**Auto Standard PK/PD 检测规则**（[_rule_based_pkpd_check](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/graph_v3.py)）：

- 命中强药物关键词（`inhibitor` / `dose` / `EC50` / `给药` / `药物` 等）→ 需要 PK/PD
- 仅含纯机制关键词（`信号通路` / `pathway` / `调控` 等）且无药物痕迹 → 跳过 PK/PD
- 规则无法判定时调用 LLM 做结构化输出决策；LLM 异常时保留规则结果

### 8 大 Worker 模块

每个 Worker 复用 v1/v2 的节点逻辑，但由 Supervisor 统一调度，并支持上下文压缩：

| Worker | 内部节点 | 职责 | Auto Fast |
| --- | --- | --- | --- |
| `worker_mcp` | node0 | MCP 术语标准化 + 查询重写 | 跳过 |
| `worker_mechanism` | N1 + N2 + N4 | NER 实体归一化 → 机制规划 → 知识图谱构建 | 仅 N1（单 LLM 抽取） |
| `worker_rag` | N3 + N5 | Mechanism RAG + Parameter RAG + 药物候选检索 | 跳过（估算参数占位） |
| `worker_pkpd` | node1_6 | PK/PD 推断：给药途径 / 房室模型 / PK/PD 参数 | 跳过 |
| `worker_ode` | N6 | LLM 输出 Network JSON → Jinja2 模板渲染 → Rule Engine 校验 | 执行 |
| `worker_sandbox` | N7 + node4 + node2 | AST 预检 → 沙箱执行 → 审计重试 → 剂量/联合用药解析 | 执行（最多 1 次重试） |
| `worker_validator` | **P0-4 新增** | SBML Validator：检测角色 → 双轨验证（libroadrunner / 参数对齐法）→ 输出 `validation_report` | 执行（有 BIOMD* ID 时） |
| `worker_report` | N8 + N9 + N10 + N11 | 特征提取 + 实验 RAG + 证据 RAG + 报告渲染 | 跳过 N9/N10 |

### Supervisor 动态调度

[Supervisor](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/graph_v3.py) 节点是 v3 的核心调度器：

```text
START → pre_router → supervisor ⇄ worker_* → supervisor → ... → END
                              ↕
                     clarification_node
```

- **严格按 `execution_plan[current_step]` 路由**：不依赖 `state.next_worker`，防止 clarification 后残留 stale 路由
- **每个 Worker 完成后回到 Supervisor**：由 `_advance_step` 推进 `current_step` 并记录 `completed_workers`
- **条件边 `_route_from_supervisor`**：判断路由到 Worker / ClarificationNode / END

### 人在环路干预

当 Supervisor 检测到特定条件时，触发 [ClarificationNode](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/graph_v3.py)（`clarification_node`），通过 `asyncio.Event` 阻塞等待前端回灌答案：

**三类触发条件**（仅 Auto Standard / Manual 模式触发，Auto Fast 不触发）：

| 触发场景 | 触发时机 | 选项 |
| --- | --- | --- |
| **参数严重缺失** | `worker_ode` 之前，所有边参数均来自估算 | A: 继续基于默认值运行 / B: 降低置信度要求 / C: 自定义方案 |
| **知识图谱环路** | `worker_ode` 之前，KG 存在反馈环路 | A: 按现有拓扑继续（稳态近似）/ B: 重新解析，忽略反馈边 / C: 自定义方案 |
| **建模方案分歧** | `worker_ode` 之前，有抑制边但无 PK/PD 模型 | A: Sigmoid Emax 模型 / B: 线性抑制项 / C: 自定义方案 |

**交互流程**：

1. Supervisor 设置 `pending_clarification` → 条件边路由到 `clarification_node`
2. `clarification_node` 发射 SSE `clarification_needed` 事件 → 前端弹出 `ClarificationDialog`
3. 用户选择选项并提交 → `POST /api/chat/respond` → `set_clarification_response` 唤醒 `asyncio.Event`
4. Supervisor 消费 `clarification_response`，清除 `pending_clarification`，继续调度下一个 Worker
5. 用户也可点击停止 → `POST /api/chat/stop` → `set_clarification_stop` 终止工作流
6. 超时保护：10 分钟无响应自动取消（`_CLARIFICATION_TIMEOUT_SECONDS = 600`）

**防串扰机制**：每次触发 intervention 前，清理该 `thread_id` 残留的旧 `clarification_response`，避免新旧问题串扰。

### 上下文渐进式压缩

[context_compressor.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/context_compressor.py) 防止超大结构化数据撑爆 LLM 上下文窗口：

- **Token 估算**：按 `json.dumps` 字符数 / 4 估算 Token 数
- **阈值触发**：超过 1000 Token 的字段自动压缩为摘要
- **零 LLM 摘要**：内置摘要函数对列表/字典/知识图谱等做轻量摘要（如 `[rag_retrieved_params] 共 42 条记录，已压缩。Top 3 示例：...`）
- **原始数据保留**：压缩后的摘要传入 LLM prompt，原始数据存入 `state.raw_cache` 供后续节点直接读取
- **每个 Worker 出口自动调用** `compress_worker_output`

---

## P0-3 修复升级：Reaction-First Biomedical Simulation Compiler

P0-3 修复解决了 EGF-EGFR 信号级联仿真中的根因问题：LLM 单点决策选错模板、KG→ODE 缺乏中间表示、模板缺质量守恒、时间尺度不一致。修复方案引入确定性管线 **KG → Reaction Graph → Template Engine → ODE**，由 4 个新模块保障。

### 1. BioModels REST API 客户端（[app/biomodels_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/biomodels_client.py)）

N0 SBML Loader 节点（[app/nodes_v2.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/nodes_v2.py) `n0_sbml_loader`）自动检测用户输入中的 `BIOMD*` ID，调用 EBI BioModels REST API 下载 SBML 文本，写入 `state.sbml_model_text` 与 `state.sbml_model_id`。本地缓存避免重复下载，网络失败时优雅降级。

### 2. TemplateSelectorSkill 规则引擎（[app/template_selector.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/template_selector.py)）

8 条优先级规则，强制模板选择不依赖 LLM 单点决策：

| 优先级 | 规则 | 触发条件 | 输出模板 |
|--------|------|----------|----------|
| 1 | keyword | 用户输入含 EGF/EGFR/pEGFR/Shc/Grb2/SOS/Ras/MAPK/磷酸化/级联 | Signaling_Cascade_Phos |
| 2 | keyword | 用户输入含 BIOMD0000000205/BIOMD0000000010/BIOMD0000000056 | Signaling_Cascade_Phos |
| 3 | keyword | 用户输入含 inhibitor/抑制剂/IC50/EC50 + 单药物单靶点 | Simple_Inhibition |
| 4 | keyword | 用户输入含 PK/PD/药代/房室/absorption | PKPD_OneCompartment 或 PKPD_TwoCompartment |
| 5 | mechanism | KG 含 binding + phosphorylation 边且边数 ≥3 | Signaling_Cascade_Phos |
| 6 | mechanism | KG 仅含 inhibition 边且边数=1 | Simple_Inhibition |
| 7 | sbml | sbml_model_id 非空且为 BIOMD* 信号通路模型 | Signaling_Cascade_Phos |
| 8 | llm_fallback | 以上均未命中 | 使用 LLM 输出（兜底） |

**关键特性**：`override_llm=True` 时规则引擎覆盖 LLM 的错误选择，并记录 `reason` / `rule_source` / `confidence` 供报告与可观测性使用。

### 3. Reaction IR 中间表示（[app/reaction_ir.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir.py)）

KG → Reaction Graph → Template 的确定性管线：

```text
KG (nodes + edges)
  ↓ build_reaction_graph(kg)
Reaction Graph (species + reactions, 含 reaction_equation / mechanism / enzyme_state)
  ↓ validate_reaction_graph(graph)
[校验：物种引用完整 / 反应方程合法 / mechanism 一致]
  ↓ Template Engine 渲染
ODE 代码（mass-action binding + Michaelis-Menten phosphorylation）
```

每条 edge 绑定 `reaction_equation` / `mechanism` / `enzyme_state` / `parameter_type`，模板引擎据此生成正确的 ODE 项。

### 4. DomainChecker 多维领域审查（[app/domain_checker.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/domain_checker.py)）

4 维领域知识审查，拦截不合理仿真结果：

| 维度 | 检查项 | 严重性 |
|------|--------|--------|
| physical | 负浓度 / NaN / Inf / 受体守恒（初始 vs 当前偏离 >10%） | high |
| biological | 信号放大效应（pMAPK / MAPK 初始 ≥10x）/ 通路完整性 | medium |
| chemical | 浓度单位一致性 / 时间单位一致性 | medium |
| medical | 剂量范围合理性 / EC50 量级 | low |

### 5. Signaling_Cascade_Phos 质量守恒模板（[app/ode_templates/Signaling_Cascade_Phos.j2](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates/Signaling_Cascade_Phos.j2)）

专为 EGF-EGFR 信号级联设计的模板，使用 mass-action kinetics（而非 Hill function）：

- **binding**：`d[A-B]/dt = k_on * [A] * [B] - k_off * [A-B]`（严格质量守恒，产物生成到反应产物索引）
- **phosphorylation**：`d[pX]/dt = k_phos * [enzyme] * [substrate] - k_dephos * [pX]`（酶不消耗，底物消耗）
- **exchange**：`d[RasGTP]/dt = k_exchange * [RasGDP]`（SOS 催化）
- 显式建模 pEGFR / pShc / pMEK / pMAPK 等磷酸化中间体
- 信号放大通过级联磷酸化实现（每级有激酶催化）
- pEGFR 达峰时间由 `k_phos/k_dephos` 比值决定（5-10 min 量级）

### 6. 时间尺度分层

[template_selector.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/template_selector.py) 的 `get_simulation_time_scale`：

| 模板 | t_end | 单位 | 适用场景 |
|------|-------|------|----------|
| Signaling_Cascade_Phos | 120 min | min | 受体信号级联（EGF-EGFR） |
| Simple_Inhibition / Simple_Activation | 48 h | h | 单药物抑制 |
| PKPD_OneCompartment / PKPD_TwoCompartment | 48 h | h | 药代动力学 |
| Cascade_Activation / Cascade_Inhibition | 48 h | h | 通用级联 |

### 7. 4-collection RAG 上下文富化

[rag_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/rag_client.py) 的 `search_params_hybrid_with_context` 在参数检索基础上富化机制与证据上下文：

1. **参数检索**：`search_params_hybrid`（查询重写 + BM25 + 语义 + 重排序）
2. **机制上下文**：`rag_collections.search_mechanism(query, top_k=3)`
3. **文献证据**：`rag_collections.search_evidence(query, top_k=3)`
4. **合并洞察**：`insights.collection_coverage = {parameter, mechanism, evidence}`

---

## P0-4 修复升级：SBML Validator 与三角色定位

P0-4 修复补齐了修复提示词1.md §二.6 与 §六正确架构中的关键一环——**SBML Validator 节点**。修复后端到端链路完整为：

```
User Input → LLM (Mechanism) → Reaction IR → Template Compiler → ODE
        → Sandbox Simulation → SBML Validator → Feature Extractor
        → LLM (Explanation only)
```

### 1. SBML 三角色定位（[app/biomodels_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/biomodels_client.py)）

修复提示词1.md §一评论 #1（必须采纳）指出原提示词的逻辑冲突：SBML 既被说成不是数据源，又被当成解析输入。正确观点是 SBML 有 3 种角色：

| 角色 | 触发条件 | 系统行为 |
|------|---------|---------|
| **Primary Ground Truth** | 用户输入含 `BIOMD*`/`MODEL*` ID 且仿真未跑 | 加载 SBML 注入参数与初值 |
| **Calibration Reference** | 无 ID 但含通路关键词（EGF/EGFR/MAPK 等） | 用 SBML 参数校准模板 ODE |
| **Validation Oracle** | 用户输入含 ID 且仿真已跑完 | 用 SBML 真实仿真对比模板仿真 |
| **None** | 无 SBML 信号 | 跳过 SBML 集成，纯 LLM 推断 |

`detect_sbml_role(user_input, has_simulation_run)` 函数实现角色判定，结果写入 `state.sbml_role`，供下游 worker_validator 决策。

### 2. SBMLValidator 双轨验证器（[app/sbml_validator.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/sbml_validator.py)）

修复提示词1.md §二.6 要求输出 `validation_report`：

```json
{
  "error_diff": 0.05,         // 峰值浓度差异
  "peak_time_diff": 2,        // 峰值时间差异（分钟）
  "amplification_diff": 0.1,  // 放大效应差异
  "sbml_sim_available": true,  // 是否成功跑通 SBML 真实仿真
  "method": "libroadrunner",   // libroadrunner | param_aligned | skipped
  "role": "validation_oracle",
  "pass": true
}
```

**双轨策略**（避免 libroadrunner 在 Windows 上 SWIG 编译困难）：

- **Track A（首选）**：`import roadrunner` 可用时跑真实 SBML 仿真（CVODE），用模糊匹配对齐物种名，计算峰值/peak_time/放大倍数差异
- **Track B（兜底）**：libroadrunner 不可用时从 SBML XML 提取参数（`<parameter id=".." value=".."/>`），评估参数覆盖度（≥0.5 视为通过）
- **跳过策略**：无 SBML 信号或 CSV 缺失时不阻塞流水线，`pass=true`

### 3. worker_validator 节点接入（[app/graph_v3.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/graph_v3.py)）

新增 `worker_validator` Worker，在 `worker_sandbox` 后执行：

| 修改点 | 内容 |
|--------|------|
| `WORKER_NAMES` | 追加 `"worker_validator"` |
| `_FULL_PLAN` | 在 `worker_sandbox` 后插入 `worker_validator` |
| `_MODULE_TO_WORKER` | 新增 `sbml_validation → worker_validator` 映射 |
| `worker_validator(state)` | 新函数：检测角色 → 调用 SBMLValidator → 写 `validation_report` |
| `_run_worker_validator` | 包装器，含 `_advance_step` |
| `build_workflow_v3` | 注册 `worker_validator` 节点与 conditional edges |
| Auto Fast 计划 | 新增 `worker_validator` |
| Auto Standard 计划 | 在 `worker_sandbox` 后追加 `worker_validator` |
| Manual 模式依赖补全 | `worker_report` 依赖 `worker_validator`；`worker_validator` 依赖 `worker_sandbox` |

### 4. SBML Validator Agent 注册（[app/supervisor.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/supervisor.py)）

`AGENT_REGISTRY_V2` 新增第 10 个智能体：

```python
AgentSpec(
    name="SBML Validator Agent",
    cn_label="SBML验证",
    description="对比模板仿真与 SBML 真实仿真，输出 error_diff / peak_time_diff / amplification_diff",
    mapped_node="worker_validator",
    icon="shield-check",
)
```

### 5. State 字段（[app/state.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/state.py)）

新增字段：

```python
sbml_role: str                      # "primary_ground_truth" | "calibration_reference" | "validation_oracle" | "none"
validation_report: dict             # {error_diff, peak_time_diff, amplification_diff, sbml_sim_available, method, role, pass, details}
```

### EGF-EGFR 端到端基准测试结果

[tests/test_egf_egfr_benchmark.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/tests/test_egf_egfr_benchmark.py) 9 步基准测试：

| 步骤 | 检查项 | 结果 |
|------|--------|------|
| 1 | BIOMD ID 提取（BIOMD0000000205） | ✓ |
| 2 | 模板选择（强制 Signaling_Cascade_Phos，覆盖 LLM 的 Cascade_Activation） | ✓ |
| 3 | Reaction Graph 构建（21 物种 / 9 反应 / 0 违规） | ✓ |
| 4 | 时间尺度分层（120 min） | ✓ |
| 5 | ODE 模板渲染（含质量守恒净反应速率） | ✓ |
| 6 | 领域常识审查（无 high 违规） | ✓ |
| 7 | 沙箱仿真执行（成功） | ✓ |
| 8 | pEGFR 达峰时间 10.03 min（目标 1-15 min） | ✓ |
| 9 | RAG 链路（参数命中 3 / 文献证据命中 3） | ✓ |

### P0-4 SBML Validator 端到端测试结果

[tests/test_sbml_validator.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/tests/test_sbml_validator.py) + [tests/test_validator_e2e.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/tests/test_validator_e2e.py) 共 30 个测试全部通过：

| 测试类别 | 测试数 | 通过 | 关键场景 |
|---------|-------|------|---------|
| SBMLRole 三角色检测 | 5 | ✓ | Primary Ground Truth / Validation Oracle / Calibration Reference / None |
| SBML XML 参数提取 | 2 | ✓ | k_on/k_off/k_phos 提取 / 空输入兜底 |
| 物种模糊匹配 | 4 | ✓ | 完全匹配 / 子串匹配 / 无匹配 / 空输入 |
| 仿真 CSV 读取 | 2 | ✓ | 标准 CSV / 不存在路径 |
| 指标提取 | 3 | ✓ | peak/peak_time/AUC / 放大倍数 / 零上游 |
| SBMLValidator 跳过场景 | 3 | ✓ | 无 SBML / 无 CSV / 本地 SBML 真实验证 |
| graph_v3 集成 | 5 | ✓ | WORKER_NAMES / _FULL_PLAN / 函数存在 / 跳过场景 / 异常场景 |
| Supervisor 注册 | 2 | ✓ | AGENT_REGISTRY_V2 / get_agent_by_node_v2 |
| State 字段 | 1 | ✓ | sbml_role / validation_report 字段存在 |
| 端到端集成 | 3 | ✓ | BIOMD* 触发验证 / 无 BIOMD* 优雅跳过 / dispatch 事件生成 |

**双轨策略验证**：
- Track A（libroadrunner 真实仿真）：环境未装 libroadrunner 时自动降级
- Track B（参数对齐法）：从 SBML XML 提取真实参数，评估参数覆盖度（≥0.5 视为通过）
- 跳过策略：无 SBML 信号或 CSV 缺失时不阻塞流水线，pass=True

### Embedding 模型对比测试结果

[tests/test_embedding_comparison.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/tests/test_embedding_comparison.py) 对比 3 款 Embedding 模型：

| 模型 | 维度 | 延迟 | 中文查询相似度 | 推荐场景 |
|------|------|------|----------------|----------|
| 讯飞 xop3qwen8bembedding | 768 | 0.24s | 0.85（最佳） | 生产环境（快 + 跨语言） |
| SiliconFlow BAAI/bge-m3 | 1024 | 0.28s | 0.71 | 备选（平衡维度与速度） |
| OpenRouter nvidia/llama-nemotron-embed | 2048 | 1.62s | 0.61 | 不推荐（延迟过高） |

---

## 多智能体编排

[app/supervisor.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/supervisor.py) 在 LangGraph 工作流之上叠加一层"编排语义"，将 LangGraph 节点映射为专业智能体：

### v3 — 智能体注册表

v3 复用 v2 的 10 大智能体注册表（P0-4 新增 SBML Validator Agent），并追加 Supervisor / PreRouter 元信息：

| 智能体 | 中文标签 | 映射 Worker / 节点 | 职责 |
| --- | --- | --- | --- |
| Terminology Agent | 术语查询 | worker_mcp / node0 | 调用 MCP 工具标准化生物医学术语，注入定义上下文 |
| Mechanism Analysis Agent | 机制解析 | worker_mechanism / N1 | 解析自然语言，提取生物实体与相互作用 |
| Entity & Planning Agent | 实体与规划 | worker_mechanism / N1 + N2 | NER 实体识别 + 机制规划器 |
| Knowledge Graph Engineer | 知识图谱 | worker_mechanism / N4 | 构建纯 Python 知识图谱（拓扑签名 / 环路检测） |
| Knowledge Retrieval Agent | 知识检索 | worker_rag / N3 + N5 | 高阶 RAG：混合检索 + 重排序；检索靶点-药物知识图谱 |
| Parameter Retrieval Agent | 参数检索 | worker_rag / N5 | 为每条边查询真实动力学参数，程序注入禁止 LLM 修改 |
| PK/PD Modeling Agent | PK/PD建模 | worker_pkpd / node1_6 | 推断给药途径、房室模型与 PK/PD 参数 |
| Simulation Engineer Agent | 仿真工程 | worker_ode + worker_sandbox / N6 + N7 | 生成 ODE 代码并执行仿真（含剂量递增） |
| Biology Validator Agent | 生物审计 | worker_sandbox / node4 | 审计仿真结果生物学合理性，失败则触发重试 |
| Scientific Analytics Agent | 特征提取 | worker_report / N8 | 从 simulation.csv 纯 NumPy 提取 9 维指标 |
| Experimental Design Agent | 实验设计 | worker_report / N9 | 推荐 Western/ELISA/qPCR/Flow 等验证手段 |
| Evidence Synthesis Agent | 证据综合 | worker_report / N10 | 检索 PMID/DOI/Figure 支撑报告结论 |
| Scientific Report Agent | 报告生成 | worker_report / N11 | LLM 输出 JSON 字段 + Python Markdown 模板渲染 |

`BioDynamicsOrchestrator` 在每个 Worker 入口/出口生成 `agent_dispatch` 事件（含 `target_agent` / `reasoning` / `status` / `latency_ms`），写入 `state.agent_dispatches` 并由 SSE 推送给前端工作流追踪器。开局还会下发 `agent_registry` 事件，前端据此初始化追踪器图标与标签。

---

## MCP 工具集成

[app/mcp_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/mcp_client.py) 通过 `langchain-protocol` 的 `MCPClient` 统一封装 4 个生物医学 MCP 工具：

| 工具 key | 名称 | 职责 | 端点环境变量 |
| --- | --- | --- | --- |
| `openbiomed` | OpenBioMed Skills | 生物医学实体识别与关系抽取（清华AIR×水木分子） | `MCP_OPENBIOMED_URL` |
| `medical_terminologies` | medical-terminologies-mcp | 临床术语标准化（ICD-10、SNOMED CT） | `MCP_MEDTERM_URL` |
| `pubmed_search` | pubmed-search-mcp | PubMed 增强版文献检索 | `MCP_PUBMED_URL` |
| `umls` | NIH UMLS MCP | 本体术语同义词与层级关系 | `MCP_UMLS_URL` |

**调用流程**（worker_mcp）：

1. OpenBioMed Skills 提取生物医学实体 → 失败则 LLM 降级提取术语
2. NIH UMLS MCP 查询同义词与层级关系 → 失败则降级
3. medical-terminologies-mcp 标准化临床术语 → 失败则降级
4. 合并 UMLS 同义词后生成术语定义卡片（估算每个定义节省 40 Token）
5. 基于标准化结果重写查询（如 `TGF-β` → `TGFB1`），供 worker_rag 提升检索精准度

每次工具调用均产出 `ToolCallRecord`（含 `action` / `status` / `latency_ms` / `tokens_saved`），由 SSE `mcp_tool_call` 事件推送给前端 MCP 工具面板渲染。

**降级策略**：

- `MCP_ENABLED=false` 完全跳过 worker_mcp，不阻塞主流程
- 端点 URL 留空 → 自动降级为 LLM 内部知识完成术语查询（非错误）
- 端点连接失败 → 同样降级，记录 `status: "fallback"`

---

## RAG 知识库与 SBML 复用

### 高阶 RAG（ChromaDB）

[app/rag_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/rag_client.py) 实现完整的高阶 RAG 流水线，由 worker_rag 调用 `search_params_hybrid` 串起四步：

1. **查询重写**（`QUERY_REWRITING_PROMPT`）：同义词映射（TGF-β → TGFB1）、拼写纠错、单位标准化、查询扩展（补充 `Kd Km Vmax half-life`）
2. **混合检索**：语义检索（cosine）+ BM25 关键词检索各取 Top 10，按文档内容去重合并，双命中标记为 `hybrid`
3. **重排序**：按来源权威性（PMC 1.0 > PubMed 0.85 > Internal DB 0.6 > Preprint 0.4）+ 物种特异性 + 参数完整性 + 语义相关度加权打分
4. **LLM 决策**（`RAG_DECISION_PROMPT`）：对每条边的检索结果做参数选择，输出 `param_found` / `selected_params` / `reasoning` / `fallback_to_estimation`

**离线建库**：`python scripts/build_rag_db.py --queries ...` 会依次：

1. 用 `requests` 调用 PubMed E-utilities 抓取文献摘要（不依赖 biopython）
2. 用 `RAG_EXTRACTION_PROMPT` 从摘要中抽取动力学参数（Kd / Km / Vmax / 半衰期等）
3. 自动做**单位归一化**（时间统一 h，浓度统一 nM）
4. Embedding 后写入本地 ChromaDB

**运行时检索**：worker_rag 把每条边编码为查询（含 MCP 标准化术语名），从 ChromaDB 拉重排序后的 Top-K 参数，再交 LLM 决策。

**靶点-药物知识图谱**：对每条 `inhibition` 边，worker_rag 调用 `mcp_client.search_pubmed` 预检索靶点抑制剂文献，再调用 `rag_client.drug_specific_retriever` 整合三源信息：
1. **PubMed 文献解析**：LLM 从摘要中提取药物名、IC50/EC50、临床剂量、是否进入临床
2. **ChromaDB 本地补充**：对每个候选药物执行混合检索，补充真实动力学参数
3. **ClinicalTrials.gov 验证**：查询是否已注册临床试验，输出 `{nct_id, phase, condition, status}`
最终按 `drug_name` 去重，写入 `state.drug_candidates` 与 `rag_insights.drug_candidates`，前端 RAG 面板高亮"临床候选药物"。

**参数优先级**：worker_ode 优先使用 RAG/PK/PD 真实参数（注释 `# 来源：RAG`），缺失才估算（注释 `# 估算值`）。

**洞察可视化**：检索过程聚合 `rewritten_query` / `source_distribution` / `top_selections` / `hit_rate` / `drug_candidates` 为 `rag_insights`，由 SSE `rag_insights` 事件推送给前端 RAG 洞察面板。

### SBML 复用

[app/sbml_parser.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/sbml_parser.py) 把已有 SBML 模型文本通过 `SBML_PARSER_PROMPT` 抽取为网络节点 / 边 JSON，worker_ode 优先基于该网络生成方程，避免从零构建。

---

## PK/PD 建模与药物研发分析

本章节对应四大医学升级的核心能力：PK/PD 推断、靶点-药物知识图谱、联合用药协同评估、剂量递增与治疗窗口。

### PK/PD 推断（worker_pkpd）

[app/nodes.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/nodes.py) 中的 `node1_6_pkpd_inference` 在 RAG 检索之后、代码生成之前执行：

- 输入：`network_json`、`drug_candidates`（来自 worker_rag）、`rag_selected_params`、`species_context`
- 输出：
  - `pkpd_profile`：`{drug_name, drug_target, route, compartment, pk_params, pd_params}`
  - `drug_regimen`：`[{drug_name, dose, ec50, emax, gamma, target}]`
- 推断规则：
  - 无药物候选 → 返回空 `pkpd_profile`，后续回退到纯 Hill 方程
  - 单药 → 推断给药途径（IV / oral）与房室模型（1-compartment / 2-compartment），生成 Emax 参数
  - 多药 → 为每个药物生成独立方案，供 worker_sandbox 评估协同/拮抗

worker_ode 的 `NODE2_PKPD_SECTION` 会强制要求生成的 ODE 包含：
- `drug_conc` 药物浓度变量
- 房室模型（如一室模型 `dC/dt = -k10*C` 或二室模型 `dC_central/dt = -k10*C - k12*C + k21*P`）
- Emax 药效模型：`effect = Emax * C^γ / (EC50^γ + C^γ)`
- 药物效应必须绑定到网络中的具体靶点

### 联合用药协同/拮抗评估（worker_sandbox）

当 `drug_regimen` 包含 ≥2 个药物时，[app/nodes.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/nodes.py) 的 `_compute_chou_talalay_ci` 基于中效方程计算各效应分数（fa = 0.5, 0.75, 0.9）下的联合用药指数：

```text
D_alone(fa) = EC50 * (fa / (1 - fa))^(1 / gamma)
CI = D_combo_A / D_alone_A(fa) + D_combo_B / D_alone_B(fa)
```

判定标准：

| CI 范围 | 评估结论 | 业务含义 |
| --- | --- | --- |
| CI < 0.8 | **潜在协同** | 联合效应强于单药叠加，值得进一步实验验证 |
| 0.8 ≤ CI ≤ 1.2 | **叠加效应** | 近似独立作用，可按药效简单相加设计剂量 |
| CI > 1.2 | **拮抗风险** | 联合效果弱于预期，需警惕竞争或毒性叠加 |

结论写入 `state.combination_index` 与 `state.synergy_assessment`，并通过 SSE `combination_synergy` 事件推送，前端以徽章 + CI 卡片形式展示。

### 剂量递增与治疗窗口（worker_sandbox）

[app/sandbox.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/sandbox.py) 在单次仿真基础上，默认执行剂量递增批处理：

- 浓度范围：0.1× ~ 100× EC50（对数均匀采样）
- 输出：
  - `DOSE_RESPONSE:` 完整浓度-效应曲线
  - `IC50:` 抑制率达到 50% 的浓度
  - `IC90:` 抑制率达到 90% 的浓度
  - `HED:` 基于异速生长缩放的人体等效剂量（Human Equivalent Dose）

前端 `DoseResponseCurve.tsx` 以 log 浓度轴绘制 Sigmoid Emax 曲线，并用不同颜色标出 IC50（绿）、IC90（蓝）、HED（琥珀色）。

---

## 安全沙箱说明

[app/sandbox.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/sandbox.py) 用 `tempfile` + `subprocess` 在隔离目录执行 LLM 生成的代码，并把 stdout/stderr 捕获回来。**两道防线**：

1. **AST 预检 + 静态黑名单扫描**（在执行前）
   - `ast.parse()` 拦截 SyntaxError，避免无谓启动子进程
   - 禁止导入：`os, sys, subprocess, socket, pathlib, shutil, urllib, requests, http, ftplib, smtplib, email, pickle, ctypes, multiprocessing, threading, asyncio`
   - 禁止内建：`eval, exec, compile, open, input, __import__`
2. **生物学常识校验**（在执行后）
   - 解析代码输出的 `BIO_CHECK: <species> = <value>`
   - 若出现 `负值 / NaN / Inf` 立即判定为失败，触发审计重试

**错误分类**：

| error_class | 触发条件 | 路由 |
| --- | --- | --- |
| `none` | returncode=0 且 CSV 无 NaN/Inf | 继续 |
| `syntax_error` | `ast.parse()` 失败 或 stderr 含 `SyntaxError`/`IndentationError` | 审计重试 |
| `runtime_error` | returncode != 0 且 stderr 无 NaN/ZeroDivision | 审计重试 |
| `numerical_error` | returncode=0 但 CSV 有 NaN/Inf，或 stderr 含 `ZeroDivisionError`/`OverflowError` | 审计重试 |
| `timeout` | 子进程 ≥ 60s 未结束 | 审计重试 |

**模式相关重试上限**：

| 模式 | 最大重试次数 |
| --- | --- |
| Auto Fast | 1 |
| Auto Standard | 3 |
| Manual | 3 |

**PK/PD 与剂量递增输出解析**：
- `DOSE_RESPONSE:` 单行 JSON → `dose_response_data`（concentrations / effects）
- `IC50:` / `IC90:` / `HED:` 标量标记 → 预测治疗窗口与人用等效剂量
- `COMBO_CI:` 多行标记 → 双药 Chou-Talalay 联合用药指数
- 超时时间已延长至 60 秒，以容纳剂量递增批处理

---

## API 接口

| Method | Path | 说明 |
| --- | --- | --- |
| `GET`  | `/` | 健康检查 |
| `POST` | `/api/chat` | **核心**：SSE 流式对话（`ChatRequest`：`user_input`, `thread_id`, `mode`, `manual_modules`） |
| `POST` | `/api/chat/respond` | **v3 人在环路**：接收用户干预答案（`ClarificationResponseRequest`：`thread_id`, `clarification_response`） |
| `POST` | `/api/chat/stop` | **v3 人在环路**：终止当前工作流（`StopRequest`：`thread_id`） |
| `POST` | `/api/chat/clear-memory` | 清空 `thread_id` 短期记忆（`ClearMemoryRequest`：`thread_id`） |
| `POST` | `/api/admin/update-vector-db` | 后台触发 ChromaDB 增量更新 |

`/api/chat` 的 SSE 事件（`data:` 行 JSON 解码后 `event` + `data` 字段）。

| `event` | 含义 |
| --- | --- |
| `agent_registry` | 开局下发智能体注册表，前端据此初始化工作流追踪器 |
| `agent_dispatch` | 单次智能体调度记录（`target_agent` / `reasoning` / `status` / `latency_ms`） |
| `node_start` | 节点状态变更（含中文状态文案） |
| `workflow_v3_state` | **v3** 流水线进度（`current_node` / `status` / `mode`） |
| `mcp_tool_call` | 单次 MCP 工具调用记录（动作 / 状态 / 延迟 / Token 节省） |
| `mcp_term_definitions` | MCP 术语定义列表 + Token 节省量 + 重写查询 |
| `rag_ready` | RAG 检索就绪（`summary` / `fallback` / `hit_rate`） |
| `rag_insights` | RAG 洞察数据（重写查询 / 来源分布 / Top 候选参数 / **药物候选**） |
| `pkpd_profile` | PK/PD 推断结果（给药途径、房室模型、PK/PD 参数） |
| `drug_regimen` | 药物方案列表（供后续联合用药评估使用） |
| `code_generated` | 生成的 Python 代码 |
| `execution_log` | 沙箱 stdout/stderr 或重试提示 |
| `image_ready` | 仿真输出 PNG（Base64） |
| `dose_response` | 剂量-反应曲线数据 + IC50 + IC90 + HED |
| `combination_synergy` | 联合用药协同评估（CI、结论、药物方案） |
| `knowledge_graph` | KG 摘要（节点数 / 边数 / 拓扑签名 / 是否无环） |
| `metrics` | 9 维指标（per-species + overall + combo）+ `confidence` |
| `experiment_protocols` | 实验方案列表（Western / Flow / ELISA / qPCR） |
| `paper_evidence` | 文献证据列表（PMID / DOI / Figure / Cell Line） |
| `report_ready` | Markdown 报告 |
| `clarification_needed` | **v3** 人在环路干预请求（`question` / `options` / `context`） |
| `clarification_resolved` | **v3** 人在环路干预已被消费，前端关闭对话框 |
| `token_usage` | 累计 Token 消耗（含 `mcp_tokens_saved`） |
| `error` | 失败 / 终态 |
| `end` | 工作流结束 |

---

## 离线 RAG 知识库构建

```powershell
cd backend
.\venv\Scripts\Activate.ps1

# 一键：抓文献 + 抽参数 + 入库
# --queries 必填，指定 PubMed 检索关键词；--max-results 控制每个关键词返回文章数
python scripts/build_rag_db.py --queries "TGF-beta CD8 T cell" "PD-1 PD-L1" --max-results 50

# 若关注特定靶点的药物候选，可加入抑制剂/抗体关键词
python scripts/build_rag_db.py --queries "TGF-beta inhibitor IC50" "TGF-beta receptor antagonist clinical" --max-results 30

# 重建 collection（删除旧库重新创建）
python scripts/build_rag_db.py --queries "TGF-beta" --recreate

# 增量更新已有库（读取 backend/data/raw 下的 .txt/.md/.json/.xml/.sbml 文件）
python scripts/update_vector_db.py
# 或通过 API 触发后台更新
curl -X POST http://localhost:8000/api/admin/update-vector-db
```

前置：`.env` 中 `OPENAI_API_KEY` 与 `NCBI_EMAIL` 已填写。

> 若云端 Embedding API 余额不足，可在 `.env` 中设置 `EMBEDDING_PROVIDER=local`，改用 `sentence-transformers/all-MiniLM-L6-v2` 本地模型，首次运行会自动下载。

### 四路 RAG 数据迁移（可选）

建议把历史数据按启发式归类迁移到 4 个 collection：

```powershell
cd backend
.\venv\Scripts\Activate.ps1

# 先做 dry-run 看会如何分类
python scripts/seed_collections.py --dry-run

# 确认无误后正式灌库
python scripts/seed_collections.py

# 重置重新灌
python scripts/seed_collections.py --recreate
```

脚本逻辑（[backend/scripts/seed_collections.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/scripts/seed_collections.py)）：

| 启发式信号 | 归类 |
| --- | --- |
| context/source 含 `western blot` / `flow cytometry` / `elisa` / `qpcr` | `experiment` |
| context/source 含 `pmid` / `doi` / `figure` / `cell line` | `evidence` |
| `param_name ∈ {Kd, Km, Vmax, half_life, k_on, k_off}` 且 `value` 非空 | `parameter` |
| context 含 `pathway` / `cascade` / `signaling` 且无具体数值 | `mechanism` |
| 其余（兜底） | 轮询分配到 4 路 |

幂等：脚本会把已灌 ID 写入 `data/seed_log.json`，重复运行只补缺。

---

## 端到端测试

后端测试位于 `backend/tests/`。

### 编译与导入验证

```powershell
cd backend
.\venv\Scripts\Activate.ps1

# v3 编译验证
python -c "from app.graph_v3 import compiled_workflow_v3; print('v3 OK')"
```

### 单元测试与端到端

```powershell
cd backend
.\venv\Scripts\Activate.ps1

# 端到端 + 审计重试
python -m unittest tests.e2e_test -v
python -m unittest tests.debug_retry -v

# 升级套件：E2E + Rule / Feature / Renderer / Templates / RAG / Sandbox 单元测试
python -m unittest tests.test_upgrade -v
python -m unittest tests.test_rule_engine -v
python -m unittest tests.test_feature_extractor -v
python -m unittest tests.test_report_renderer -v
python -m unittest tests.test_ode_templates -v
python -m unittest tests.test_rag_collections -v
python -m unittest tests.test_sandbox_v2 -v

# P0-3 修复升级测试：TemplateSelector / ReactionIR / DomainChecker / BioModels / Signaling_Cascade_Phos / 4-collection RAG / EGF-EGFR 端到端
python tests/test_p0_repairs.py
python tests/test_egf_egfr_benchmark.py

# P0-4 修复升级测试：SBML Validator + 三角色定位 + worker_validator 接入 + 端到端集成
python tests/test_sbml_validator.py
python tests/test_validator_e2e.py

# Embedding 模型对比测试（讯飞 / SiliconFlow / OpenRouter）
python tests/test_embedding_comparison.py
```

### 手动 E2E（启动前后端 + 浏览器）

```powershell
# 1) 后端
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000

# 2) 前端（新窗口）
cd ..\frontend
npm run dev

# 3) 浏览器打开 http://localhost:3000
# 右侧控制栏选择模式（Auto Standard 默认）
# 输入假说，例如：
#    "TGF-β inhibits CD8 T cell through SMAD"
#    "Galunisertib (TGF-β inhibitor) + PD-1 antibody combination on CD8"
# 期望：
#    - 顶部工作流追踪器依次点亮各 Agent
#    - Auto Standard 模式：TGF-β 问题跳过 PK/PD；PD-1 抑制剂问题包含 PK/PD
#    - 报告含 SMAD / CD8 / IFNG / Western blot / PMID
#    - 若参数全部缺失，控制栏弹出干预对话框
```

---

## 常见问题

**Q1. 前端能连上后端吗？**
默认前端通过 `http://localhost:8000` 访问后端；若后端端口变更，请同步修改 [frontend/app/page.tsx](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/app/page.tsx) 顶部的 `API_BASE`。

**Q2. 报错 `ModuleNotFoundError: No module named 'app'`？**
在 `backend/` 目录下启动 Uvicorn，确保 `app/` 是 `cwd` 的子目录。

**Q3. biopython 装不上？**
本项目已**完全避开 biopython**。离线建库脚本使用 `requests` + `xml.etree.ElementTree`，兼容 Python 3.14。BM25 也在 `rag_client.py` 内联实现，无需 `rank-bm25`。

**Q4. MCP 端点都没配，会影响主流程吗？**
不会。MCP 端点留空时自动降级为 LLM 内部知识完成术语查询；`MCP_ENABLED=false` 可完全跳过 worker_mcp。两种情况都不阻塞主流程。

**Q5. ChromaDB 与 Qdrant 怎么选？**
本项目当前实现使用 **ChromaDB**（本地持久化、无需 Docker、开箱即用），配置项为 `CHROMA_PERSIST_DIR` / `CHROMA_COLLECTION_NAME`。若需切换到 Qdrant，需自行改造 [app/rag_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/rag_client.py) 的客户端封装。

**Q6. 怎样替换 LLM 供应商？**
`.env` 中把 `OPENAI_BASE_URL` 改成目标供应商的 OpenAI 兼容端点（智谱、DeepSeek、火山、Azure OpenAI 等），`OPENAI_MODEL` 改成对应模型名即可。配置了 `BACKUP_*` 三项后，主模型失败会自动切到备用模型（`FallbackLLM`）。本地模型可用 Ollama（base_url 指向 `http://localhost:11434/v1`）。

**Q7. 云端 Embedding 余额不足怎么办？**
本项目支持 5 个 Embedding Provider 可切换：
- `EMBEDDING_PROVIDER=xfyun`（默认推荐）：讯飞 MaaS xop3qwen8bembedding，768 维，跨语言最佳，延迟 0.24s
- `EMBEDDING_PROVIDER=siliconflow`：SiliconFlow BAAI/bge-m3，1024 维，中文友好
- `EMBEDDING_PROVIDER=openrouter`：OpenRouter nvidia/llama-nemotron-embed（免费层），2048 维，延迟较高
- `EMBEDDING_PROVIDER=openai`：OpenAI text-embedding-3-small（需付费）
- `EMBEDDING_PROVIDER=local`：sentence-transformers/all-MiniLM-L6-v2（本地模型，无需 API）

切换后需重建 ChromaDB（向量维度可能不同）：
```powershell
cd backend
python scripts/rebuild_params_collection.py
```

Embedding 模型对比测试详见 `python tests/test_embedding_comparison.py`。

**Q8. 三种运行模式有什么区别？**
- **Auto Fast**：极简链路 `mechanism → ode → sandbox → report`，跳过 MCP/RAG/PKPD/证据检索，沙箱最多重试 1 次。适合快速验证想法。
- **Auto Standard**（默认）：先用规则检测是否需要 PK/PD，规则无法判定时调用 LLM，自动裁剪执行计划。沙箱最多重试 3 次。适合大多数场景。
- **Manual**：用户在前端控制栏手动勾选所需模块，依赖自动补全。沙箱最多重试 3 次。适合需要精确控制流程的高级用户。

**Q9. 人在环路干预对话框弹出来了，怎么办？**
当参数全部缺失、知识图谱存在环路、或建模方案有分歧时，Auto Standard / Manual 模式会弹出干预对话框。选择 A / B / C 选项（或选 C 输入自定义方案），点击提交即可继续。也可以点击**停止**终止工作流。Auto Fast 模式不会触发干预。

**Q10. 报告里出现 "某疾病" / "T1" / "T2" / "炎症因子" / "TGF-betta" 等占位词怎么办？**
这些是 ReportRenderer 的 `FORBIDDEN_TERMS` 黑名单。N11 命中后会用 `NODE11_REPORT_RETRY_PROMPT` 重新调用 LLM 一次；若再次命中直接走降级路径并在 `report.forbidden_terms_violations` 列出违例。前端 `ChatMessage` 会显示违例明细。

**Q11. 沙箱提示 `error_class=timeout` 怎么办？**
默认 60s 超时。检查 N6 生成的 ODE 是否设置了过大的 `t_end` 或过密的 `n_eval`；或临时调高 `sandbox.execute_simulation_code_v2(code, timeout=120)`。

---

## 开发规范

- **Python**：[PEP 8](https://peps.python.org/pep-0008/) + [PEP 20](https://peps.python.org/pep-0020/)，中文详细注释优先
- **TypeScript**：ESLint + Next.js 默认规范
- **提交规范**：建议使用 Conventional Commits（`feat:`, `fix:`, `docs:`, `chore:`）
- **v3 架构约定**：
  - v3 核心编排逻辑位于 [backend/app/graph_v3.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/graph_v3.py)
  - 上下文压缩位于 [backend/app/context_compressor.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/context_compressor.py)
  - Worker 复用 v1/v2 节点逻辑：v1 节点在 [backend/app/nodes.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/nodes.py)；v2 节点在 [backend/app/nodes_v2.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/nodes_v2.py)
  - v1 prompts 在 [backend/app/prompts.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/prompts.py)；v2 prompts 在 [backend/app/prompts_v2.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/prompts_v2.py)
  - 共享工具（如 `orchestrator` / `token_usage`）不区分版本复用
- **目录约定**：
  - RAG 相关模块位于 [backend/app/rag_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/rag_client.py)（单库 + 混合检索 + 重排序 + drug_specific_retriever）与 [backend/app/sbml_parser.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/sbml_parser.py)
  - 四路 RAG 位于 [backend/app/rag_collections.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/rag_collections.py)
  - MCP 工具集成位于 [backend/app/mcp_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/mcp_client.py)
  - 多智能体编排位于 [backend/app/supervisor.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/supervisor.py)
  - 知识图谱 / Rule / 特征 / 报告 / 模板分别位于 `backend/app/{kg_builder, rule_engine, feature_extractor, report_renderer}.py` 与 `backend/app/{ode_templates, report_templates}/`
  - 离线脚本位于 [backend/scripts/](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/scripts)
  - LangGraph Prompt 集中于 [backend/app/prompts.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/prompts.py) 与 [backend/app/prompts_v2.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/prompts_v2.py)
  - 前端可视化组件位于 [frontend/components/chat/](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/components/chat)
  - v3 前端核心组件：[ControlBar.tsx](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/components/chat/ControlBar.tsx) 与 [ClarificationDialog.tsx](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/components/chat/ClarificationDialog.tsx)

---

## 许可证

本项目以 **MIT License** 开源，详见 [LICENSE](./LICENSE) 文件。

```text
MIT License

Copyright (c) 2026 BioDynamics Agent Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**第三方依赖**（详见各 `package.json` 与 `requirements.txt`）：所有前端 / 后端依赖均沿用各自上游的开源协议（MIT / Apache-2.0 / BSD 等），本项目不修改其许可。

---

## 致谢

- [LangGraph](https://langchain-ai.github.io/langgraph/) / [LangChain](https://python.langchain.com/) — Agent 编排基石
- [Model Context Protocol](https://modelcontextprotocol.io/) / [langchain-protocol](https://github.com/langchain-ai/langchain-protocol) — MCP 工具协议
- [FastAPI](https://fastapi.tiangolo.com/) — 高性能 Python Web 框架
- [Next.js](https://nextjs.org/) + [Shadcn UI](https://ui.shadcn.com/) — 现代前端栈
- [ChromaDB](https://www.trychroma.com/) — 本地优先的向量数据库
- [sentence-transformers](https://www.sbert.net/) — 本地 Embedding 兜底
- [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) — 公开生物医学文献接口
- [OpenAI](https://openai.com/) / [智谱 BigModel](https://open.bigmodel.cn/) — LLM 与 Embedding 服务

---

**⭐ 如果这个项目对你有帮助，欢迎 Star / Fork / 提 PR！**
