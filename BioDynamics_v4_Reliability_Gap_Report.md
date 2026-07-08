# BioDynamics v4 — Reliability Gap Report

> **审查角色**：AI for Science Reliability Engineer / Systems Biology QA Lead
> **审查基准**：Martin Fowler《Building Reliable Agentic AI Systems》+《Benchmarking LLM-based agents for single-cell omics analysis》
> **审查对象**：`bio-dynamics-agent/` 仓库（v4 RC，commit `9f6bfbd` 之后）
> **审查性质**：只读审计，未修改任何源代码
> **报告日期**：2026-07-08

---

## 执行摘要（Executive Summary）

BioDynamics v4 在**确定性渲染层**（`ode_renderer_v2.py` + `reaction_ir_v2/` + Jinja2 模板）做得相当扎实——这是值得肯定的设计。但在**可靠性**与**精准度**两个维度上存在 6 类致命缺陷，其中 3 项为 P0 级（可直接导致生物学错误结果被静默放行）：

| # | 缺陷 | 等级 | 后果 |
|---|------|------|------|
| D1 | 求解器发散/NaN 无参数级恢复，重试生成相同代码 | **P0** | 数值错误被静默通过，前端展示无效结果 |
| D2 | 基因/蛋白实体零外部验证（ontology 默认 OFF） | **P0** | LLM 幻觉基因直接进入 ODE，产出生物学谬误 |
| D3 | `fallback_used` 是无人消费的死标志，主图 worker 无 fail-safe | **P0** | v4 "降级到 v3" 的承诺在代码层面未实现 |
| D4 | PMID 仅用于检索而非验证 LLM 引用 | P1 | 虚构文献无法被发现 |
| D5 | v3 路径 `_parse_reaction_equation` 用 split 解析 LLM 自由文本 | P1 | 格式漂移导致化学计量静默出错 |
| D6 | worker_validator 异常时 `pass=True`（放行） | P1 | 验证器失败反而通过 |

---

## 1. 【精准度风险清单】5 个最可能导致生物学错误的幻觉路径

### 路径 1：基因/蛋白实体幻觉（UNPROTECTED）— P0

**链路**：`N1 NER`（`nodes_v2.py`）→ 输出 `entities[].canonical_id` → `KG` → `Reaction IR` → `ODE`

**证据**：
- `nodes_v2.py:84`：`canonical_id: str = ""` 仅是 Pydantic 字段，无任何外部 DB 校验调用。
- `prompts_v2.py:71-76` 的【Negative Constraints】仅写"禁止编造未在用户输入中提及的实体"——这是**建议性指令**，无强制力。
- 本体客户端（`ontology/hgnc_client.py`、`uniprot_client.py`、`chebi_client.py`、`go_client.py`）**存在但被 feature flag 默认关闭**：
  - `ontology_agent.py:389`：`if not settings.effective_v4_ontology_enabled(): return {}`
  - `.env.example`：`V4_SCIENTIFIC_LAYER_ENABLED=false` → `V4_ONTOLOGY_AGENT_ENABLED` 解析为 false。
- 若 LLM 输出 `{"name": "FAKE_GENE_1", "canonical_id": "HGNC:99999"}`，**没有任何代码会向 HGNC 验证该 ID 是否存在**。

**生物学后果**：幻觉基因进入 ODE 后，模板会为其生成 `d[FAKE_GENE_1]/dt = ...` 方程，仿真产出看似合理但完全虚构的曲线，被前端渲染为"科学结论"。

### 路径 2：PubMed ID 虚构（UNPROTECTED）— P1

**链路**：`N9/N10 Evidence RAG`（`nodes_v2.py`）→ `report_renderer` → 最终报告

**证据**：
- `mcp_client.py:437-587` 的 `search_pubmed` 调用 NCBI E-utilities（`esearch` + `efetch`），但用途是**主动检索参数/文献**，PMID 由 E-utilities 响应生成（`:571-572`）。
- **没有任何代码**接收 LLM 在报告中引用的 PMID 并反向验证其存在性或内容匹配性。
- `rag_client.py` 中无 `verify_pmid` / `fetch_pmid` 函数（grep 零命中）。

**生物学后果**：LLM 可在报告"文献对比"章节虚构 `PMID:12345678`，用户无法察觉，损害科学可信度。

### 路径 3：动力学参数幻觉（WEAK）— P1

**链路**：`N5 Parameter RAG`（`nodes_v2.py`）→ `parameters` → `ODE 模板`

**证据**：
- `rag_client.py:752` `search_params_hybrid` 确实实现了 BM25 + 语义 + rerank（符合 project memory 约束）。
- 但 **RAG 决策 prompt 仅决定"用检索值还是估计"**，无物理可行性硬门：
  - 检索返回的 `k_cat=5000 s⁻¹`（远超扩散极限 ~1000 s⁻¹）会被直接采用。
  - 负值 `Km` 不会被拒绝——`rule_engine.py:116` 仅 `logger.warning`，不阻塞。
- `RAG_DECISION_PROMPT` 未要求 LLM 校验参数量级（如 `k_on` 应在 10⁴–10⁷ M⁻¹s⁻¹）。

**生物学后果**：物理上不可能的参数进入 ODE，导致仿真时间尺度错误（如毫秒级反应被仿真为小时）。

### 路径 4：反应机制幻觉（WEAK）— P1

**链路**：`N2 Mechanistic Planner` → `edges[].mechanism` → `Reaction IR._parse_reaction_equation`

**证据**：
- `prompts_v2.py:90-119` 给出了 mechanism 白名单（binding/phosphorylation/...），但 N2 输出仅做 `with_structured_output` 的 schema 校验，**不校验 mechanism 与 reaction_equation 的一致性**。
- `reaction_ir.py:205-225` `_parse_reaction_equation` 用 `split("→",1)` + `split("+")` 解析 `reaction_equation`（LLM 自由文本）。若 LLM 写 `"FAKE_KINASE + ATP → pFAKE_KINASE + ADP"`，解析器会信任并生成对应反应。
- `pre_validate_reaction_graph`（`reaction_ir.py:270-413`）仅做 token boundary 与冲突检测，**不验证物种是否真实存在**。

**生物学后果**：虚构的酶催化反应进入 ODE，质量作用方程看似平衡实则描述不存在的生物学过程。

### 路径 5：SBML 来源参数与当前模型错配（WEAK）— P2

**链路**：`rag_client.py` 检索 BioModels 参数 → 注入当前 pathway ODE

**证据**：
- RAG 从 BIOMD0000000010（EGF-EGFR）检索的 `k_phos` 可能被注入到完全不同的 pathway（如 Wnt）的 ODE 中。
- `RAG_DECISION_PROMPT` 虽要求判断参数适用性，但**无物种/通路匹配硬校验**——若检索结果物种集合与当前模型无交集，仍可能被采用。
- `rag_client.py` 无 `source_model_id` ↔ `target_model_id` 一致性检查。

**生物学后果**：跨通路参数迁移导致动力学行为失真，仿真结果无法复现。

---

## 2. 【确定性违规清单】LLM 与确定性逻辑耦合不当

### 2.1 v4 路径（高确定性，推荐保留）

| 阶段 | 分类 | 文件:行号 | 评价 |
|------|------|-----------|------|
| PathwayGraph → ReactionIRv2 | **DETERMINISTIC** | `reaction_ir_v2/reaction_builder.py:58-251` | 纯规则映射，文件头声明"不调用 LLM" |
| 模板选择 | **DETERMINISTIC** | `ode_renderer_v2.py:198-238` `_select_template` | 静态路由表 `_PATHWAY_TEMPLATE_MAP` |
| ODE 渲染 | **DETERMINISTIC** | `ode_renderer_v2.py:176-187` | Jinja2 `template.render()`，无 LLM |
| 质量守恒自动生成 | **DETERMINISTIC** | `reaction_ir_v2/reaction_builder.py:222` | `auto_generate_mass_conservation()` |

### 2.2 v3 路径（含伪确定性违规，需整改）

| # | 文件:行号 | 违规类型 | 说明 |
|---|-----------|----------|------|
| V1 | `reaction_ir.py:205-225` `_parse_reaction_equation` | **PSEUDO-DETERMINISTIC** | 用 `split("→",1)` + `replace("+"," ").split()` 解析 LLM 生成的 `reaction_equation` 自由文本。若 LLM 输出 `"->"` / `"⇒"` / 物种名含空格，静默出错 |
| V2 | `reaction_ir.py:346-348` `pre_validate_reaction_graph` | **PSEUDO-DETERMINISTIC** | 同上 split 逻辑用于冲突检测，依赖 V1 的解析结果 |
| V3 | `template_selector.py:329-337` | **LLM 兜底** | 规则 8：无规则命中时采用 `llm_template`（置信度 0.5），确定性路由降级为 LLM 决策 |
| V4 | `template_selector.py:429-431` `_matches_any` | **PSEUDO-DETERMINISTIC** | 对 `user_input` 自由文本做 `kw.lower() in text_lower` 子串匹配——`"mapk"` 会误匹配 `"non-mapk"` |
| V5 | `level1_internal.py:237-386` | **PSEUDO-DETERMINISTIC** | 用 regex 扫描 ODE 源码字符串做数值稳定性检查（`re.finditer(r"/\s*(\w+)")` 除零检测），静态分析误报漏报并存 |
| V6 | `sandbox.py:186` `_check_biological_validity` | **PSEUDO-DETERMINISTIC** | `re.search(r"BIO_CHECK:\s*(\S+)\s*=")` 解析仿真代码 stdout——但 `ode_templates_v2/*.j2` 从不输出 `BIO_CHECK` 行，此检查实际永不触发 |
| V7 | `nodes_v2.py:1379` N6 ODE Generator | **FORMAT COUPLING** | `_safe_json_parse` 字符串解析 LLM 输出，无 Pydantic schema 强制（仅 N5 用 `with_structured_output`） |

### 2.3 Fowler 原则违规总结

Fowler 主张"LLM 只规划，确定性代码执行"。当前系统的核心违规：
1. **N6 ODE Generator** 让 LLM 同时规划 + 生成代码结构（虽渲染用 Jinja2，但 KG 边的 `reaction_equation` 自由文本由 LLM 生成，确定性代码再解析它）。
2. **template_selector 规则 8** 把确定性路由的兜底交给 LLM（`llm_template`），违反"规则引擎应穷尽"原则。
3. **Level 1 数值稳定性**用 regex 静态扫描代码字符串，而非运行时检测——这是"伪确定性"的典型：看似是代码分析，实则依赖 LLM 生成的代码格式。

---

## 3. 【MCP 引入建议】3 个必需 MCP 规格说明

当前系统用直接 HTTP 调用 BioModels / HGNC / UniProt / ChEBI / PubMed（`biomodels_client.py`、`ontology/*.py`、`mcp_client.py`），存在三大问题：无统一鉴权、无缓存复用、无 schema 校验、测试时需 mock 整个 HTTP 层。引入 MCP 可标准化外部数据访问边界。

### MCP-1：`biomodels-mcp`（BioModels 数据库 MCP）

**动机**：`biomodels_client.py` 直接 HTTP 调用 `https://www.ebi.ac.uk/biomodels/`，无重试、无缓存、无 schema。RAG 离线建库脚本也重复实现 BioModels 拉取逻辑。

**规格**：
- **工具集**：
  - `get_model(model_id) -> SBMLContent`：按 BIOMD ID 拉取 SBML，返回结构化 `{"sbml": str, "name": str, "publication_id": str, "parameters": [...], "species": [...]}`。
  - `search_models(query, filters) -> list[ModelSummary]`：按通路/物种/疾病检索。
  - `get_parameters(model_id) -> list[Parameter]`：直接返回结构化参数（含 `k_cat`/`Km`/`Vmax` + 单位 + 出处 PMID），替代当前 XML 解析。
  - `validate_model_id(model_id) -> bool`：**硬校验**——解决路径 5 的参数错配问题。
- **缓存策略**：本地 SQLite 缓存 7 天（BioModels 模型版本稳定）。
- **鉴权**：EBI API key（可选，提升速率限制）。
- **接入点**：替换 `biomodels_client.py` + `biomodels_reactions.py` + RAG 建库脚本中的 `requests.get(biomodels_url)`。
- **期望收益**：参数检索 schema 化，消除 `xml.etree` 解析脆弱性；`validate_model_id` 可在 RAG 决策前硬校验来源模型。

### MCP-2：`hgnc-validator-mcp`（HGNC 基因符号验证 MCP）

**动机**：路径 1（基因幻觉）的根因是 N1 NER 不验证 `canonical_id`。`ontology/hgnc_client.py` 已存在但被 feature flag 关闭，且是直接 HTTP，无标准化工具接口。升级为 MCP 后可作为 Node 0 的强制前置。

**规格**：
- **工具集**：
  - `validate_symbol(symbol) -> HGNCRecord | null`：验证基因符号存在，返回 `{"hgnc_id": "HGNC:XXXX", "symbol": "EGFR", "name": "...", "uniprot_id": "P00533", "status": "Approved"}`。不存在的符号返回 `null`。
  - `validate_id(canonical_id) -> HGNCRecord | null`：按 `HGNC:XXXX` / `UniProt:XXXXX` 反查。
  - `suggest_synonyms(symbol) -> list[str]`：返回官方别名，用于 N1 NER 的 aliases 校验。
  - `batch_validate(symbols: list[str]) -> dict[str, HGNCRecord | null]`：批量校验 N1 输出的所有实体。
- **缓存策略**：HGNC 数据季度更新，本地缓存 30 天 + 模糊匹配。
- **接入点**：Node 0（MCP term lookup）之后、Node 1 之前插入 `node0_5_gene_validate`，对 N1 输出的所有 Protein/Gene 类型实体调用 `batch_validate`。`null` 结果触发硬阻塞 + 用户澄清，而非静默通过。
- **期望收益**：直接消除路径 1（基因幻觉），使 `FAKE_GENE_1` 在进入 KG 前被拦截。

### MCP-3：`pubmed-verifier-mcp`（PubMed 文献验证 MCP）

**动机**：路径 2（PMID 虚构）。当前 `mcp_client.search_pubmed` 仅用于检索，不验证 LLM 在报告中引用的 PMID。需要反向验证能力。

**规格**：
- **工具集**：
  - `verify_pmid(pmid: str) -> PubmedRecord | null`：验证 PMID 存在，返回 `{"pmid": "12345678", "title": "...", "authors": [...], "journal": "...", "year": 2023, "abstract": "..."}`。不存在返回 `null`。
  - `verify_citation(pmid: str, claim: str) -> {relevance: float, contradiction: bool}`：用 NLP 验证引用是否支撑文中论断（可选，用小模型）。
  - `batch_verify_pmids(pmids: list[str]) -> dict[str, PubmedRecord | null]`：批量校验报告所有引用。
- **接入点**：`worker_report`（`nodes_v2.py` N11）之后插入 `report_citation_verify` 节点，扫描报告全文 `PMID:\d+` 模式，对每个 PMID 调用 `verify_pmid`。`null` 触发报告标记 `[UNVERIFIED CITATION]`。
- **期望收益**：消除路径 2，使虚构 PMID 在报告渲染前被发现。

---

## 4. 【硬校验 Skill 设计】2 个必需 Skill

当前系统缺的不是"软警告"（已有大量 `logger.warning`），而是会**阻塞流水线**的硬门。以下两个 Skill 应作为 LangGraph 节点插入，失败时返回 `rule_violations` 并阻断渲染。

### Skill-1：`stoichiometry-guard`（化学计量守卫）

**触发时机**：`worker_ode` 渲染 ODE 代码后、`worker_sandbox` 执行前。即 Jinja2 模板渲染产出 Python 代码字符串后立即运行。

**接口规格**：
```python
class StoichiometryGuard:
    """化学计量守卫：动态质量守恒硬校验。"""

    def check(
        self,
        ode_code: str,
        reaction_ir: dict,
        species: list[str],
        initial_concentrations: dict[str, float],
    ) -> GuardResult:
        """
        1. 静态分析 ODE 代码：解析每个 d[X]/dt 方程，提取所有物种的系数矩阵。
        2. 构造化学计量矩阵 S（reactions × species），验证 rank(S) 满足守恒约束。
        3. 数值验证：用 initial_concentrations 在 t=0 计算 Σ(d[X]/dt) 是否为 0
           （对每个守恒量，如 EGFR_total = EGFR + pEGFR + EGF-EGFR）。
        4. 对所有 reaction_ir.constraints 中的 mass_conservation 约束，
           验证 ODE 代码确实包含对应的守恒量导数求和为零。
        
        Returns:
            GuardResult(passed: bool, violations: list[str], 
                       conservation_matrix: np.ndarray)
            passed=False 时 worker_sandbox 拒绝执行，返回 rule_violations。
        """
```

**关键设计**：
- **动态而非静态**：当前 `level1_internal.py:127-208` 仅检查 t=0 初始浓度，本 Skill 通过解析 ODE rhs 在任意 t 验证。
- **硬门**：`passed=False` → `worker_sandbox` 跳过执行，直接进入 `worker_validator` 标记 `stoichiometry_violation`。
- **解决缺陷**：D5（v3 路径 split 解析）+ 4.1（Level 1 静态检查）。

### Skill-2：`numerical-stability-retry`（数值稳定性重试机制）

**触发时机**：`worker_sandbox` 检测到 `ERR_NUMERICAL`（NaN/Inf/overflow，`sandbox.py:382`）时触发，而非直接重试相同代码。

**接口规格**：
```python
class NumericalStabilityRetry:
    """数值稳定性重试：阶梯式参数降级，而非重新生成相同代码。"""

    STRATEGIES = [
        # 策略 1：收紧步长
        {"max_step": lambda v: v * 0.1, "method": "LSODA", "rtol": 1e-6, "atol": 1e-9},
        # 策略 2：切换刚性求解器
        {"max_step": "auto", "method": "BDF", "rtol": 1e-8, "atol": 1e-10},
        # 策略 3：放宽容差（牺牲精度换稳定）
        {"max_step": "auto", "method": "Radau", "rtol": 1e-4, "atol": 1e-6},
        # 策略 4：降阶——移除最快反应尺度（如绑定步骤用准稳态近似）
        {"model_reduction": "QSSA", "method": "LSODA"},
    ]

    def retry(
        self,
        ode_code: str,
        failure_reason: str,  # "nan" | "inf" | "overflow" | "divergence"
        attempt: int,         # 0-indexed
        current_params: dict,
    ) -> RetryResult:
        """
        1. 解析 ode_code 中的 solver 配置（method/max_step/rtol/atol）。
        2. 按 attempt 选择 STRATEGIES[attempt] 生成新配置。
        3. 用 Jinja2 重新渲染 ode_code（不调用 LLM，纯模板替换 solver 段）。
        4. 返回新代码 + 降级说明。
        
        Returns:
            RetryResult(ode_code: str, strategy: str, 
                       degraded: bool, max_attempts: int = 4)
            超过 max_attempts 返回 degraded=True，由 worker_sandbox 标记
            sandbox_failure_reason="numerical_unstable_after_retry"。
        """
```

**关键设计**：
- **不重新调用 LLM**：当前 `graph_v3.py:1158` 重试调用 `n6_ode_generator`，但 N6 不读 `retry_count`（grep 零命中），大概率生成相同代码。本 Skill 纯确定性参数降级。
- **阶梯策略**：步长收紧 → 刚性求解器 → 放宽容差 → 模型降阶，逐级牺牲。
- **解决缺陷**：D1（求解器发散无恢复）+ 2.2（NaN 静默通过，模板无 `sol.success` 检查）。

---

## 5. 【提示词工程优化建议】防幻觉 System Prompt 修改

### 5.1 N1 NER Prompt（`prompts_v2.py:25-77`）— 增加 ID 强制校验

**当前问题**：`【Negative Constraints】` 仅写"禁止编造"，无强制力。`canonical_id` 字段允许空字符串。

**建议修改**：
```
【Negative Constraints】
- 禁止编造未在用户输入中提及的实体
- 禁止输出除 JSON 外的任何文字
- 仅提取用户输入中实际出现的实体，不得添加示例中的占位实体

【Canonical ID 硬规则（新增）】
- canonical_id 字段：仅当能从用户输入或上下文中明确识别标准 ID 时填写，
  否则必须留空字符串 ""。
- 严禁凭记忆填写 UniProt:/HGNC: 前缀 ID —— 若无明确出处，留空。
- 留空的 canonical_id 将由后续 Node 0.5 (HGNC 验证) 补全，
  你不需要也不应该猜测。
- 若实体是你推断的磷酸化形式（如 pEGFR），canonical_id 必须留空 
  —— 磷酸化形式无独立 HGNC ID。
```

### 5.2 N2 Planner Prompt（`prompts_v2.py:83-...`）— 增加 mechanism-reaction 一致性

**当前问题**：mechanism 字段与 reaction_equation 字段可不一致，`reaction_ir.py` 用 split 解析 reaction_equation 但不校验与 mechanism 的匹配。

**建议修改**：
```
【Mechanism-Reaction 一致性硬规则（新增）】
- mechanism="binding" 时，reaction_equation 必须形如 "A + B → A-B"（含 "+" 与复合物名）
- mechanism="phosphorylation" 时，reaction_equation 必须形如 "Xxx → pXxx" 或 "Kinase + Xxx → Kinase + pXxx"
- mechanism="inhibition" 时，reaction_equation 必须形如 "Drug + Target → Drug-Target" 或显式标注抑制
- 禁止 mechanism 与 reaction_equation 矛盾（如 mechanism="binding" 但方程无 "+"）
- reaction_equation 中分隔符必须使用 "→"（U+2192），禁止使用 "->" 或 "=>" 
```

### 5.3 N5 Parameter RAG Prompt — 增加物理可行性校验

**当前问题**：`RAG_DECISION_PROMPT` 仅决定用检索值还是估计，不校验量级。

**建议修改**（追加到 RAG_DECISION_PROMPT）：
```
【物理可行性硬门（新增）】
采用任何检索参数前，必须校验其量级是否在生物学合理范围：
- k_on（结合速率）：1e3 ~ 1e7 M^-1·s^-1，超出则标记 [SUSPECT]
- k_off（解离速率）：1e-4 ~ 1e1 s^-1，超出则标记 [SUSPECT]
- k_cat（催化常数）：1e-2 ~ 1e3 s^-1，超出 1e3 标记 [SUSPECT]
- Km（米氏常数）：1e-7 ~ 1e-2 M，超出则标记 [SUSPECT]
- 任何负值参数：直接拒绝，回退到 estimation
被标记 [SUSPECT] 的参数：在输出 JSON 中添加 "confidence": "low"，
并附 "reason": "value_outside_typical_range"。
```

### 5.4 N11 Report Prompt — 增加 PMID 自校验指令

**当前问题**：报告可自由引用 PMID，无验证。

**建议修改**（追加到 N11 Report Prompt）：
```
【文献引用硬规则（新增）】
- 报告中所有 PMID 引用必须来自前序 N9/N10 RAG 节点检索结果，
  严禁凭记忆生成 PMID。
- 若需引用文献但 RAG 未返回，使用占位符 [CITATION_NEEDED]，
  后续验证节点会补全或标记。
- 严禁在报告中出现形如 "PMID:12345678" 但未在 N9/N10 outputs 中出现的引用。
```

### 5.5 全局：所有 LLM 节点输出强制 Pydantic schema

**当前问题**：仅 N5 用 `with_structured_output`，其余 5 个节点（N1/N2/N3/N6/N11）用 `_safe_json_parse` 字符串解析。

**建议**：所有 LLM 节点输出强制 `with_structured_output(PydanticModel)`，消除 V7（`nodes_v2.py:1379`）的格式耦合。Pydantic schema 既是输出约束，也是输入校验门——若 LLM 输出不符合 schema，LangChain 直接抛 `OutputParserException`，触发重试而非静默错误。

---

## 附录 A：fail_safe.py 健壮性评估

### A.1 覆盖范围缺口

`fail_safe.py:130-274` `FailSafeDispatcher.dispatch` **仅被 `dynamic_router.py:124-130` 调用**，包装的是 Dynamic Router 的 13 个旁路 Agent。主图核心 worker **无保护**：

| 节点 | 是否被 fail_safe 包裹 | 证据 |
|------|----------------------|------|
| `worker_ode`（含 N6 + LLM 调用） | **否** | `graph_v3.py:1417-1431` `_run_worker_ode` 直接调用 |
| `worker_sandbox`（子进程） | 部分（仅 subprocess timeout） | `sandbox.py:474` `subprocess.run(timeout=...)` |
| `worker_validator` | **否** | 同上裸调用 |
| `worker_report`（含 N9/N10 RAG + LLM） | **否** | 同上裸调用 |

### A.2 线程超时无法杀死挂起 LLM 调用

`fail_safe.py:219-228` 使用 `threading.Thread(daemon=True)` + `join(timeout)`。Python threading 无 kill API，daemon 线程在进程存活期间持续运行。若 LLM HTTP 请求阻塞在 socket read，超时后该线程持续占用连接，多次超时累积资源泄漏。

### A.3 `fallback_used` 是无人消费的死标志

`fail_safe.py:252` 注释自认"异常不强制回退 v3，由调用方决定"。`dynamic_router.py:133-135` 注释"调用方可根据 fallback_used 标记决定"——但 `dynamic_router.py:144` 仅 `return {"v4_agent_dispatches": dispatches}`，**不检查 fallback_used，不调用 v3 流水线**。`dynamic_router_hook_node`（`:418-467`）同样仅写 dispatches 到 state 后返回。

**结论**：v4 "降级到 v3" 的承诺在代码层面未实现。v4 Agent 是旁路增强，`fallback_used=True` 仅是日志记录，无实际降级动作。

### A.4 worker_validator 异常放行

`graph_v3.py:1318-1330`：`worker_validator` 异常时 `pass=True`（`:1328` 注释"异常不阻塞"）。验证器失败反而通过，与"验证金字塔"设计意图相悖。

---

## 附录 B：验证金字塔严格性汇总

| 检查项 | 文件:行号 | 类型 | 缺陷 |
|--------|-----------|------|------|
| Level 1 质量守恒 | `level1_internal.py:127-208` | 硬门（静态 t=0） | 仅初始浓度；无约束即通过（:148-151） |
| Level 1 数值稳定性 | `level1_internal.py:322-386` | 硬门（静态 regex） | regex 扫描源码，误报漏报并存 |
| Level 1 非负性 | `level1_internal.py:213-265` | 软警告 | 不影响 pass |
| 动态 CSV 守恒 | `nodes_v2.py:1980-1988` | 软警告 | `metadata["warnings"].extend(...)` |
| overall_pass=False | `validation_agent.py:333-340` | 软门（人机环路） | 不阻塞主流程，可选 continue |
| worker_validator 异常 | `graph_v3.py:1318-1330` | **放行** | 异常时 `pass=True` |
| Reaction IR 预校验 | `reaction_ir.py:1217-1242` | 硬门（渲染前） | 唯一真正的硬门，但仅 token 级 |

---

## 附录 C：建议的整改优先级

| 优先级 | 缺陷 | 建议动作 | 涉及模块 |
|--------|------|----------|----------|
| **P0** | D1 求解器发散无恢复 | 实现 Skill-2（`numerical-stability-retry`） | `graph_v3.py` worker_sandbox 重试逻辑 + 新 Skill |
| **P0** | D2 基因零验证 | 引入 MCP-2（`hgnc-validator-mcp`）+ Node 0.5 强制前置 | `nodes_v2.py` N1 之后 + 新 MCP |
| **P0** | D3 fallback 死标志 | 消费 `fallback_used`，主图 worker 接入 FailSafeDispatcher | `graph_v3.py` `_run_worker_*` + `dynamic_router.py` |
| P1 | D4 PMID 虚构 | 引入 MCP-3（`pubmed-verifier-mcp`）+ report 后置验证节点 | `nodes_v2.py` N11 之后 + 新 MCP |
| P1 | D5 split 解析 LLM 文本 | 迁移到 v4 `reaction_ir_v2`（已确定性），废弃 v3 `reaction_ir._parse_reaction_equation` | `nodes_v2.py` N6 路由 |
| P1 | D6 validator 放行 | `worker_validator` 异常时 `pass=False` 而非 `True` | `graph_v3.py:1328` |
| P2 | 参数量级 | N5 prompt 追加物理可行性硬门 | `prompts_v2.py` |
| P2 | 全节点 schema | 6 个 LLM 节点全部 `with_structured_output` | `nodes_v2.py` |

---

*审查完毕。等待用户确认后续整改方向。*
