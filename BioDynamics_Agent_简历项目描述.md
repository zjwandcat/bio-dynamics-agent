# 项目经历 · 简历版

> 用于应聘广州生物医疗行业「大模型应用开发 / 智能体构建」岗位
> 基于真实项目 BioDynamics Agent 整理，遵循 STAR 法则，未知量化数据以 `[待补充]` 标注

---

### 项目名称：基于 LangGraph 的生物医学信号通路仿真智能体（BioDynamics Agent）

**技术栈**：Python 3.11+、LangGraph 1.2、LangChain 1.3、FastAPI、Next.js 16、ChromaDB、Jinja2、SciPy / NumPy、OpenAI 兼容 LLM（GLM-4 / DeepSeek，主备 FallbackLLM 切换）、Embedding（讯飞 xop3qwen8b / BAAI bge-m3 / 本地 sentence-transformers）、Rerank（讯飞 / Cohere / bge-reranker）、MCP 协议、PubMed E-utilities、BioModels REST API、ClinicalTrials.gov API

---

**【S 项目背景】**

计算系统生物学与转化医学研究中，实验生物学家普遍不擅长编写 ODE 方程与 SciPy 仿真代码，而现有 LLM 直接生成的代码又常脱离文献真实动力学参数（Kd、Km、IC50 等），导致「定性假说 ↔ 定量仿真」之间存在难以跨越的工程鸿沟。本项目的目标是用一个多智能体系统，让研究者用自然语言描述生物机制或药物假说，即可自动产出可执行仿真、剂量-反应曲线与机理论证报告。

**【T 核心任务】**

我担任该项目的主程与架构设计者，负责：① 设计基于 LangGraph 的 Supervisor-Worker 多智能体编排；② 构建面向生物医学的高阶 RAG 系统（参数 / 机制 / 证据 / 实验四路 collection）；③ 封装生物医学分析工具（PubMed 检索、SBML 解析、PK/PD 推断、沙箱仿真）为可被 LLM 自动调用的工具节点；④ 保证全链路在 LLM 限流、参数缺失、知识图谱环路等异常下的健壮性。

**【A 技术行动与系统设计】**

- **Agent 架构设计（为何选 LangGraph 而非纯 LangChain）**
  纯 LangChain 的 Chain 是线性 DAG，难以表达「Supervisor 动态调度 + 人在环路阻塞 + 模式相关重试」这类带条件边与阻塞节点的状态机。我选用 LangGraph 构建显式状态图 `graph_v3.py`：以 `BioDynamicsState` 为中心状态对象，PreRouter 按运行模式（Auto Fast / Auto Standard / Manual）生成 `execution_plan`，Supervisor 节点严格按 `plan[current_step]` 路由到 8 个 Worker（术语标准化 / 机制解析 / RAG / PK-PD / ODE 生成 / 沙箱 / SBML 验证 / 报告），ClarificationNode 通过 `asyncio.Event` 阻塞等待用户干预。关键设计是**不依赖 `state.next_worker`**，避免人在环路后残留 stale 路由导致死循环——这是踩过坑后的修正。

- **RAG 系统优化**
  针对生物医学术语多写法（TGF-β / TGF-beta / TGFB1）导致召回不稳的问题，引入「查询重写 → BM25 + 语义混合检索 → 来源权威性加权重排序 → LLM 决策」四步流水线：① 查询重写由 `QUERY_REWRITING_PROMPT` 做同义词映射与单位归一化；② BM25 内联实现（避开 rank-bm25 依赖），与向量检索各取 Top-10 去重合并，双命中标 `hybrid`；③ 重排序按来源权威性（PMC 1.0 > PubMed 0.85 > Internal 0.6 > Preprint 0.4）+ 物种特异性 + 参数完整性加权；④ LLM 用 `RAG_DECISION_PROMPT` 对每条边的候选参数做结构化决策，输出 `param_found / selected_params / fallback_to_estimation`。Embedding 与 Rerank 均设计为多 Provider 可切换，避免单一供应商余额不足阻塞生产。

- **工具封装与自动调用**
  通过 MCP 协议（langchain-protocol）统一封装 4 个生物医学工具（OpenBioMed 实体识别、UMLS 同义词、medical-terminologies 临床术语标准化、pubmed-search 文献检索），每个工具调用产出 `ToolCallRecord`（动作 / 状态 / 延迟 / Token 节省）由 SSE 推送前端可观测。降级策略：端点 URL 留空 → 自动降级为 LLM 内部知识；`MCP_ENABLED=false` 完全跳过，不阻塞主流程。沙箱工具用 `ast.parse` + 静态黑名单（os/sys/subprocess/socket 等）做 AST 预检，按 syntax / runtime / numerical / timeout 四类错误分类，模式相关重试上限（Auto Fast=1，Standard/Manual=3）。

- **技术难点攻克**
  ① **结构化输出清洗**：BigModel（GLM）的 `with_structured_output` 会返回 ```` ```json ... ``` ```` markdown 包裹的 JSON，直接 `json.loads` 必崩。编写 `strip_markdown_json` 清洗函数统一处理，所有节点走 `_safe_json_parse` 兜底。② **State 字段污染**：LangGraph 中使用 `operator.add` reducer 的字段（messages / agent_dispatches / mcp_tool_calls）无法被 `initial_state` 清空，导致多轮对话串扰。改为每次请求生成新 `thread_id`，并在 `/api/chat/clear-memory` 显式删除。③ **Python 3.14 兼容**：biopython 在 3.14 无预编译 wheel，MSVC 编译困难。将离线建库脚本改为 `requests` + `xml.etree.ElementTree` 直接解析 PubMed XML，彻底移除 biopython 依赖。④ **Few-shot 污染**：含 TGF-β / SMAD / CD8 关键词的 few-shot 示例会让 LLM 在无关问题上凭空生成这些蛋白，替换为通用示例（EGFR / MAPK / Protein_A）后稳定。

**【R 项目成果】**

- **端到端基准**：EGF-EGFR 信号通路 9 步基准测试全部通过，pEGFR 达峰时间 10.03 min（文献目标 5-10 min 区间内），RAG 参数命中 3 条 / 文献证据命中 3 条。
- **测试覆盖**：后端累计 `[待补充]` 个单元 + 集成测试通过，覆盖 Rule Engine、Feature Extractor、ODE 模板渲染、四路 RAG、沙箱错误分类、SBML Validator 等模块。
- **Embedding 选型**：对比 3 款模型，讯飞 xop3qwen8bembedding 中文查询相似度 0.85、延迟 0.24s，胜出作为默认 Provider。
- **架构演进**：在 v3 基础上完成 v4 科学架构设计（八层科学栈 / 10 条癌症通路 × 5 模块 / 17 类生化机制 / 五层 Validation Pyramid / 13 Agent 动态编排），并按 7-Phase Migration Plan 以 Feature Flag 共存模式推进，保证迁移期系统不崩、可秒级回滚。

**💡 架构反思**

当前系统的局限在于：① SBML Validator 在 Windows 上依赖 libroadrunner 的 SWIG 编译，缺失时只能降级为参数对齐法，差异指标无法计算；② RAG 通路隔离仍靠 `pathway_tag` 软约束，跨通路 shared species（如 Ras / AKT）的参数污染尚未彻底解决。下一步计划引入 ontology ID 硬对齐（HGNC / UniProt）与五级溯源链（ODE ↔ Reaction ↔ SBML ↔ Parameter ↔ PMID），并向 v5 的贝叶斯参数估计与患者特异性仿真演进。
