# BioDynamics Agent

> **把一段自然语言假设，变成可复现的 ODE 仿真、实验设计与引用链报告。**
> 面向信号通路研究的全栈 AI 建模助手——输入 *EGF 刺激后 ppERK 的时序响应*，输出机制图、动力学曲线、BioModels 对比验证、PubMed 引用报告与 Western blot 实验方案，全程可追溯、零幻觉引用。

---

## 为什么需要这个 Agent

癌症信号通路研究长期受困于三件事：**建模门槛高**（需手写 ODE、查 Kd/Km、调参）、**参数散落**（散布在 PubMed、BioModels、ChEMBL、KEGG）、**结果不可复现**（LLM 直接生成数字、引用伪造）。BioDynamics Agent 把"读文献 → 建网络 → 查参数 → 写方程 → 跑仿真 → 对真值 → 出报告"这条传统需数天的人工流程，压缩到一次对话。

**核心科学原则**：LLM 仅负责组织与解释已有证据，**绝不创造科学事实**。每一个 `k_cat`、`K_m`、初始浓度都带溯源四元组 `(value, source, confidence, origin)`，每一个 Discussion 句子都带单源标签 `[A]PubMed / [B]BioModels / [C]Simulation / [D]Inference / [E]Hypothesis`。

---

## 与通用编码 Agent（Claude Code / Codex）的差异

市场上并不缺生物医学领域的 LLM，但绝大多数停留在"问答式文献摘要"。BioDynamics Agent 的差异化在于**把 LLM 关进科学规则的笼子**：

| 维度 | Claude Code / Codex | 通用生物医学 LLM | **BioDynamics Agent** |
|---|---|---|---|
| 知识来源 | 训练记忆，可能幻觉 | 训练记忆 + 浅层 RAG | **10 通路 Specialist 硬编码拓扑 + SBML 真值 + 四路 RAG + 4 MCP 本体** |
| 参数来源 | LLM 直接报数字 | LLM 直接报数字 | **三层查询链：SBML(0.95) → RAG(0.8) → LLM 推理(0.4) → 默认(0.2)** |
| 数值验证 | 无 | 无 | **5 级验证金字塔 + BioModels Track A roadrunner 真实仿真对比** |
| 引用真实性 | 易伪造 PMID | 易伪造 PMID | **C9 文献正确性强制 canonical PMID 全命中，C12 讨论句句带源标签** |
| 实验设计 | 文字建议 | 文字建议 | **规则引擎按机制匹配 assay（EGFR 激活 → Western blot pEGFR/ppERK，0-60 min 时间序列）** |
| 可复现性 | 每次不同 | 每次不同 | **同一通路生物学等价输入产出一致科学结论（确定性 Specialist + 规则驱动）** |

**Claude Code / Codex 做不到或极难做到的**：
1. 把 LLM 生成的 Python 代码放进 AST 预检 + 资源限制沙箱执行，失败时按 LSODA→BDF→Radau 阶梯重试
2. 用 roadrunner 真实仿真 EBI BioModels SBML 参考模型，逐物种对比峰值时间与振幅
3. 强制 Discussion 每句话携带证据源标签，违反时抛 `DiscussionRenderError` 阻断输出
4. 12 轴 Benchmark（C1-C12）机器可读评分，43 case 金标准回归测试

**对研究效率的提升**：传统建模流程（读 5-10 篇综述 → 查 20+ 参数 → 写 ODE → 调参 → 验证 → 报告）通常需 2-5 个工作日；本 Agent 在单次对话（10-20 分钟）内完成全链路，研究员只需**审阅与决策**，而非从零搭建。

---

## 系统能力一览

| 能力 | 实现 |
|---|---|
| 10 条信号通路 Specialist | EGFR / MAPK / PI3K-AKT-mTOR / p53 / Apoptosis / Cell Cycle / JAK-STAT / NF-κB / Wnt / TGF-β，每条绑定 BioModels 参考模型 |
| LangGraph v3 工作流 | pre_router → supervisor → 8 workers（mcp/mechanism/rag/pkpd/ode/sandbox/validator/report） |
| 四路 RAG 向量库 | ChromaDB：mechanism 70 / parameter 4584 / experiment 20 / evidence 120 |
| 19 类机制中间表示 | PHOSPHORYLATION / UBIQUITINATION / GTP_GDP_EXCHANGE / TRANSCRIPTION …（Pydantic v2 schema，零 LLM 依赖） |
| 23 个 ODE Jinja2 模板 | v3 12 个（含 PK/PD 一/二房室、DoseSweep）+ v4 11 个机制专用（振荡反馈、双稳态开关、Caspase 级联、Cyclin-CDK toggle、Wnt 破坏复合体） |
| 5 级验证金字塔 | L1 质量守恒 → L2 SBML 真值对比 → L3 跨通路一致性 → L4 基准对比 → L5 假设验证 |
| Scientific Alignment 13 子模块 | Consistency Checker / Critic / Multi-dim Confidence / Validation Rule Engine / Evidence Fuser / Discussion Renderer …（100% 规则驱动） |
| 4 本体客户端 | HGNC / UniProt / ChEBI / GO（缓存 TTL 7 天） |
| 4 MCP 工具 | OpenBioMed 实体识别 / UMLS 同义词 / ICD-10-SNOMED CT / PubMed 检索 |
| 沙箱执行保护 | AST 预检 + 静态安全扫描 + import guard + 子进程 RLIMIT + 数值稳定性阶梯重试 |
| 三链路 LLM 容灾 | DeepSeek v4-flash（主）→ NVIDIA Nemotron（备）→ DeepSeek 官网（兜底），0.5s 切换 |
| 多提供商 Rerank | 讯飞 MaaS Qwen3-Reranker-8B → OpenRouter → SiliconFlow BAAI/bge-reranker-v2-m3 |

---

## Scientific Benchmark 成绩

43 case 金标准套件覆盖 10 通路 × 4 难度（L1 Canonical / L2 Feedback / L3 Perturbation / L5 BioModels）+ 3 Cross-pathway。每个 case 需通过 12 项评估（C1 机制 → C12 讨论）方可判定 Scientific PASS。

**Batch 6（V3 + LLM 参数推理增强，2026-07-30）双 LLM 全量测试**：

| LLM | Scientific PASS | Scientific Score | Operational |
|---|---|---|---|
| poolside/laguna-s-2.1:free | 10/43 (23.3%) | 22.04% | 38/43 (88.4%) |
| deepseek-v4-flash | 10/43 (23.3%) | 22.04% | 43/43 (100%) |

**11 个 PASS 案例**（9 个双 LLM 一致通过 ★，2 个单 LLM 通过）：

| Case | 通路 | 难度 | 生物学场景 | 双验证 |
|---|---|---|---|---|
| 2.M4 | MAPK | L5 | EGF → RasGTP → pMEK → ppERK 级联 | ★ |
| 4.P2 | p53 | L2 | DNA 损伤 → pATM → p53-Mdm2 负反馈振荡 | ★ |
| 4.P4 | p53 | L3 | Nutlin-3 扰动 p53-Mdm2 反馈 | ★ |
| 5.C4 | CellCycle | L5 | Cyclin-CDK 周期振荡 + APC/C 降解 | deepseek |
| 6.A2 | Apoptosis | L2 | 死亡受体 → Caspase 级联 → MOMP | ★ |
| 7.N3 | NF-κB | L3 | TNF → IKK → IκBα 降解 → NF-κB 核转位 | ★ |
| 7.N4 | NF-κB | L5 | Bortezomib 扰动 NF-κB 通路 | ★ |
| 8.J3 | JAK-STAT | L3 | IFN-γ → JAK → STAT1 → SOCS3 负反馈 | ★ |
| 10.T3 | TGF-β | L3 | TGF-β → pSmad2 → SMAD7 反馈 | ★ |
| 11.X2 | CrossPathway | L4 | EGFR-MAPK 与 PI3K crosstalk | poolside |
| 11.X3 | CrossPathway | L4 | p53 与 Apoptosis 通路耦合 | ★ |

**主要瓶颈**：C5 峰值时间（34.9%）、C6 峰值振幅（32.6%）——这两个是动力学定量正确性的硬指标，也是后续优化方向。

---

## 工作流程

```
用户自然语言输入
    │
    ▼
┌─ pre_router ──────────────────────────────────────────────────┐
│  规则优先 + LLM 兜底，生成 execution_plan（如 7-8 个 worker）   │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ supervisor（纯状态机，无 LLM）─────────────────────────────┐
│  按 plan[current_step] 顺序调度 worker，通过 LangGraph state  │
│  传递数据，worker 间无对话，完全通过共享 state 读写            │
└──────────────────────────────────────────────────────────────┘
    │
    ├─ worker_mcp ────── MCP 术语标准化（4 工具：OpenBioMed/UMLS/ICD-10/PubMed）
    │                   输出：mcp_term_definitions（注入下游 prompt）
    │
    ├─ worker_mechanism ─ NER 实体标准化 → 机制规划 → 知识图谱构建
    │                   [Specialist Hook：10 通路专家注入核心拓扑/反馈环/crosstalk]
    │                   输出：network_json / knowledge_graph / entities
    │
    ├─ worker_rag ────── 机制 RAG + 按边参数 RAG（每条边查询重写 + BM25 + 语义 + Rerank）
    │                   [耗时最长：单通路 ~5min，双通路 ~15min]
    │                   输出：parameters（溯源四元组）/ rag_hit_rate
    │
    ├─ worker_pkpd ───── PK/PD 推断（仅药物相关问题触发）
    │                   输出：pkpd_profile / dose_response
    │
    ├─ worker_ode ────── LLM 输出 Network JSON + Jinja2 模板 + Rule Engine 选择
    │                   [SBML 初始浓度提取：上游激活源从 BioModels SBML 提取]
    │                   输出：ode_code（Python scipy.integrate.solve_ivp）
    │
    ├─ worker_sandbox ── AST 预检 + 子进程执行 + 数值稳定性阶梯重试
    │                   （LSODA → 收紧 max_step → BDF → Radau，最多 4 次）
    │                   输出：simulation_csv / curve_image / metrics
    │
    ├─ worker_validator ─ SBML Validator：Track A roadrunner 真实仿真对比
    │                     Track B 结构相似度降级
    │                     输出：validation_report（peak_diff / RMSE / correlation）
    │
    └─ worker_report ── 特征提取 → 实验设计 RAG → 证据检索 → 报告渲染
                        [Scientific Alignment 后处理：Consistency/Critic/MultiDim]
                        输出：report_markdown / experiment_protocols / paper_evidence
    │
    ▼
  最终 SSE 事件：report + token_usage + end
```

**参数三层查询链**（`_get_param` 解析每个动力学参数）：

```
1. LLM_INFERRED_PARAMS (N5 推理产物, confidence >= 0.4)
   └─ 命中 → 使用 LLM 推理值 (origin=llm_inferred)
2. _PATHWAY_KINETICS[PATHWAY_CLASS] (通路特异性默认值表)
   └─ 命中 → 使用通路默认值 (origin=default)
3. SBML/RAG 真实参数 (confidence 0.8-0.95)  ← 优先级最高，覆盖上述
   └─ 命中 → 使用真实文献参数 (origin=sbml/rag)
```

---

## 全栈技术架构

### 后端（FastAPI + LangGraph + Python 3.14）

```
backend/
├── app/
│   ├── main.py                  FastAPI 入口 + /api/chat SSE 流式接口
│   ├── graph_v3.py              LangGraph v3 工作流 + v4 hook chains
│   ├── config.py                Settings + 三链路 LLM 容灾 + 多提供商 Rerank
│   ├── supervisor.py            Orchestrator 状态机调度
│   ├── nodes_v2.py              N0-N11 流水线节点（V3 worker 内部调用）
│   ├── sandbox.py                AST 预检 + 资源限制 + 阶梯重试
│   ├── rag_client.py             混合检索（BM25 + 语义）+ Rerank + ClinicalTrials.gov
│   ├── biomodels_client.py       EBI BioModels API + Validation Oracle
│   ├── mcp_client.py             4 MCP 工具 + LLM/E-utilities 降级
│   ├── ontology/                 HGNC / UniProt / ChEBI / GO 客户端
│   ├── reaction_ir_v2/           19 类机制 Pydantic v2 中间表示
│   ├── ode_templates(_v2)/       23 个 Jinja2 ODE 模板
│   ├── pathways/specialists/     10 通路 Specialist
│   ├── validation_v2/             5 级验证金字塔
│   └── scientific_alignment/      13 子模块（100% 规则驱动）
├── benchmarks/golden/             10 通路金标准 YAML
└── data/                          ChromaDB + BioModels SBML + PMID 缓存
```

### 前端（Vite + React 18 + TypeScript + TailwindCSS）

```
frontend/
├── src/
│   ├── App.tsx                  主应用（三栏布局）
│   ├── hooks/
│   │   └── useChatStream.ts     SSE 客户端 Hook（@microsoft/fetch-event-source）
│   ├── components/
│   │   ├── TopBar.tsx           顶部状态栏（LLM/Agent/阶段/SSE 连接灯/知识库刷新）
│   │   ├── StageTimeline.tsx    左侧 8 阶段时间轴（Framer Motion 动画）
│   │   ├── MessageList.tsx      中间聊天消息区
│   │   ├── ChatInput.tsx        输入框
│   │   ├── ClarificationDialog.tsx  人工干预对话框（120s 超时）
│   │   └── ReportPanel.tsx      右侧 7 Tab 报告面板
│   ├── types/sse.ts             SSE 事件类型定义 + 节点→Agent 映射
│   └── lib/api.ts               REST API 封装
└── package.json                 Vite 5.3 + React 18.3 + Tailwind 3.4 + Framer Motion 11
```

**前端关键技术问题与解决**：

1. **SSE 连接保活**：RAG 阶段单次 LLM 调用 15-20s，多边累计 5-15min 无事件，浏览器 TCP 空闲超时断开。后端 `asyncio.wait(timeout=15)` 周期发送 `heartbeat` 事件，task 不被取消，工作流完整推进
2. **节点名→Agent 名映射**：后端 `target_agent` 为中文 label，前端 `NODE_NAME_TO_AGENT` 映射表转为真实 Agent 英文名（如 `worker_mechanism` → "Mechanism Analysis Agent"）
3. **7 Tab 自动切换**：报告面板在数据首次到达时自动切到对应 Tab（机制图/ODE 代码/仿真曲线/验证报告/实验方案/最终报告/文献证据）
4. **8 阶段状态机**：`StageTimeline` 按 `STAGE_NODES` 渲染 pending/running/done/failed 四态，Framer Motion 脉冲动画
5. **连接状态灯**：TopBar 四色指示（灰=未连接/黄=连接中/绿=已连接/红=错误），实时反映 SSE 健康度

### 一键启动

```bash
# 双击 start.bat（Windows）自动启动后端 + 前端 + 打开 Edge
# 或手动：
cd backend && py -3.14 -m uvicorn app.main:app --port 8000
cd frontend && npm install && npm run dev   # http://localhost:3000
```

---

## REST API

| 方法 | 端点 | 用途 |
|---|---|---|
| POST | `/api/chat` | 主 SSE 工作流（自然语言 → 报告） |
| POST | `/api/chat/respond` | 人工干预后恢复 |
| POST | `/api/chat/stop` | 取消运行 |
| GET | `/api/v4/pathways` | 列举 10 通路 |
| GET | `/api/v4/pathways/{class}/graph` | 通路图数据 |
| POST | `/api/v4/simulation/run` | 确定性 mass-action ODE 仿真 |
| POST | `/api/v4/benchmark/{class}` | 单通路基准测试 |
| GET | `/api/models/status` | LLM / Embedding / Rerank 健康检查 |

---

## 使用示例

**输入**（自然语言）：
> 分析 DNA 损伤后 p53-Mdm2 负反馈环路产生的脉冲式振荡动力学，关注 ATM 介导的 p53 磷酸化与 Mdm2 介导的 p53 泛素化降解之间的反馈调控，验证 p53 脉冲周期 (5-7h) 与阻尼衰减特性

**输出**：
1. **机制图**：ATM → p53（磷酸化）→ Mdm2（转录）→ p53（泛素化降解）负反馈环
2. **ODE 代码**：`dp53/dt = k_phos * ATM - k_deg * Mdm2 * p53 - k_nat * p53`（scipy solve_ivp）
3. **仿真曲线**：p53 呈 5-7h 周期脉冲振荡，振幅阻尼衰减
4. **验证报告**：与 BioModels BIOMD0000000252 对比，peak_time_diff < 0.5h
5. **实验方案**：Western blot p53/pMdm2，时间序列 0/1/3/5/7/12h
6. **文献证据**：PMID 10959078（canonical）、PMID 12451180（supporting）
7. **最终报告**：Markdown，每句 Discussion 带源标签 `[A]PubMed / [B]BioModels / [C]Simulation`

---

## 中国 Agent 的特色与优势

1. **国产 LLM 全链路适配**：DeepSeek v4-flash 作为主 LLM，讯飞 MaaS Qwen3 提供 Embedding 与 Rerank，SiliconFlow BAAI/bge 作为兜底——全栈国产模型容灾，不依赖 OpenAI
2. **多供应商 Rerank 级联**：讯飞 MaaS → OpenRouter → SiliconFlow 三级，按延迟与可用性自动切换
3. **离线 RAG 建库**：Python 脚本直接调用 NCBI E-utilities + xml.etree.ElementTree（兼容 Python 3.14），不依赖 biopython
4. **本地向量库优先**：ChromaDB 本地持久化（4 collections / 4794 条），网络不可用时降级到 LLM 推理
5. **生物医学本体本地缓存**：HGNC / UniProt / ChEBI / GO 客户端 TTL 7 天缓存，减少对外部 API 依赖

---

## 测试与 CI

- `backend/tests/`：80+ pytest 文件，覆盖 10 specialist、验证金字塔 L1-L5、Reaction IR v2、SBML grounder、Feature Flag 收敛
- `Benchmark_QA_Collection.md`：43 case 可执行回归规范，12 项评估准则（C1-C12）
- GitHub Actions：`ci.yml` + `scientific-regression.yml`（含 Scientific Alignment 门禁 job）
- Makefile：`make test` / `make test-integration` / `make test-benchmark`

---

## License

MIT. 见 [LICENSE](./LICENSE)。
