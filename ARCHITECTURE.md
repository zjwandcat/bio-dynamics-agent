# BioDynamics Agent v2 — Architecture Reference

> 本文件配套 [README.md](README.md) 与 [TEMPLATES.md](TEMPLATES.md)，是 v2 12 节点流水线的权威定义。
> 升级计划见 [.trae/documents/biodynamics-v2-upgrade-plan.md](.trae/documents/biodynamics-v2-upgrade-plan.md)。

---

## 目录

1. [Overview](#1-overview)
2. [State Schema](#2-state-schema)
3. [12 节点详细定义](#3-12-节点详细定义)
4. [错误分类与重试策略](#4-错误分类与重试策略)
5. [RAG 四路分类](#5-rag-四路分类)
6. [模板与 Rule 引擎](#6-模板与-rule-引擎)
7. [报告渲染流程](#7-报告渲染流程)
8. [多智能体映射](#8-多智能体映射)
9. [SSE 事件清单](#9-sse-事件清单)
10. [关键不变量与硬约束](#10-关键不变量与硬约束)

---

## 1. Overview

v2 流水线由 `app.graph.build_workflow_v2()` 组装，编译产物为 `compiled_workflow_v2`；通过 `.env` 中 `WORKFLOW_VERSION=v2` 启用。v1 / v2 同时编译，重启服务即切换。

**12 节点链路**：

```
START
  → N1 NER              (LLM, 1 次调用)
  → N2 Planner           (LLM, 1 次调用)
  → N3 Mechanism RAG     (Chroma, 0 LLM)
  → N4 KG Builder        (纯 Python, 0 LLM)
  → N5 Parameter RAG     (Chroma, 0 LLM)
  → N6 ODE Generator     (LLM 1 次调用 + Jinja2 模板 + Rule Engine)
  → N7 Sandbox Execute   (ast.parse → subprocess → 错误分类, 0 LLM)
  ↳ 失败且 retry_count < 3 → 回 N6
  → N8 Features          (纯 NumPy, 0 LLM)
  → N9 Experiment RAG    (Chroma, 0 LLM)
  → N10 Evidence RAG     (Chroma, 0 LLM)
  → N11 Report           (LLM 1 次调用 + 禁止词校验 + Jinja2 模板)
  → END
```

**LLM 调用次数**：N1 (1) + N2 (1) + N6 (1, 失败重试时 2) + N11 (1, 触发重试时 2) = **典型 4 次 / 极端 6 次**。

**总执行时间目标**：单次成功 < 60 秒（不含 LLM 网络延迟）。

---

## 2. State Schema

`BioDynamicsState` (TypedDict) 兼容 v1 字段（标 `# DEPRECATED`），v2 在 `state.py` 中以追加方式声明。

### 2.1 v1 字段（保留 / 标记 DEPRECATED）

| 字段 | 类型 | 状态 |
| --- | --- | --- |
| `mcp_term_definitions` | `list[dict]` | v1 Node 0 写入；v2 由 N1 复用 |
| `mcp_rewritten_query` | `str` | v1 Node 0 写入；v2 由 N1 复用 |
| `network_json` | `dict` | v1 Node 1 写入；v2 N6 写入等价字段 `network_relations` |
| `rag_selected_params` | `dict` | v1 Node 1.5 写入；v2 N5 写入等价字段 `parameters` |
| `drug_candidates` | `list[dict]` | v1 Node 1.5 写入；v2 N5 合并到 `parameters` 内的 `drug_candidates` 子键 |
| `pkpd_profile` | `dict` | v1 Node 1.6 写入；v2 N5 + N6 模板共同产出 |
| `drug_regimen` | `list[dict]` | v1 Node 1.6 写入；v2 N5 产出 |
| `combination_index` | `dict` | v1 Node 4 写入；v2 N11 从 `metrics.combo` 派生 |
| `synergy_assessment` | `str` | 同上 |
| `python_code` | `str` | v1 Node 2 写入；v2 N6 等价于 `ode_model.code` |
| `simulation_image` | `str` | v1 Node 3 写入；v2 N7 写入 `image_base64` |
| `dose_response_data` | `dict` | v1 Node 3 写入；v2 N7 写入 `execution_result.dose_response_data` |
| `final_report` | `str` | v1 Node 5 写入；v2 N11 等价于 `report.markdown` |
| `agent_dispatches` | `list[dict]` | v1 / v2 共用 |

### 2.2 v2 新增字段

```python
# N1 / N2
entities: list[dict]                # [{entity_id, name, type, aliases, canonical_id}]
mechanism: dict                     # {pathway, cell, simulation_type, template, required_outputs, exemplars}

# N4
knowledge_graph: dict               # {nodes, edges, adjacency, topology_signature, is_acyclic, node_count, edge_count, dropped_edges}

# N5
parameters: dict                    # {edge_key: {param_name, value, unit, source, is_fallback, ...}, drug_candidates: [...]}
rag_fallback: bool                  # 至少有一条边 RAG 缺失

# N6
ode_model: dict                     # {template, code, parameters_used, rule_violations}
network_relations: dict             # LLM 输出的定性关系（不写代码、不给数值）
rule_violations: list[dict]         # Rule Engine 校验结果

# N7
execution_result: dict              # 镜像 N3 输出 + 新增 simulation_csv_path / error_class
simulation_csv_path: str            # 沙箱代码生成的 simulation.csv 绝对路径
error_class: str                    # "none" | "syntax_error" | "runtime_error" | "numerical_error" | "timeout"
retry_count: int                    # 沙箱失败回 N6 的次数，最多 3

# N8
metrics: dict                       # {species: {...}, overall: {...}, combo: {...}}
feature_metadata: dict              # {method, version, confidence, warnings}
confidence: float                   # 0..1 整体置信度

# N9 / N10
experiment_protocols: list[dict]    # [{name, target, detection_method, cell_line, pmid, ...}]
paper_evidence: list[dict]          # [{pmid, doi, title, figure_ref, cell_line, species}]

# N11
report: dict                        # {markdown, llm_filled_json, forbidden_terms_violations}
```

### 2.3 State 字段读写矩阵

| 字段 | 写入节点 | 读取节点 |
| --- | --- | --- |
| `user_input` | 入口 | N1, N2 |
| `entities` | N1 | N2, N3, N4, N5, N11 |
| `mechanism` | N2 | N3, N4, N6, N9, N10, N11 |
| `mechanism.exemplars` | N3 | N6（仅作 decoration） |
| `knowledge_graph` | N4 | N5, N6, N11 |
| `parameters` | N5 | N6（模板渲染）, N11（报告展示） |
| `rag_fallback` | N5 | N11（报告加 fallback 提示） |
| `network_relations` | N6 (LLM) | N6 (Rule Engine), N6 (Jinja2 模板) |
| `ode_model` | N6 | N7 |
| `rule_violations` | N6 (Rule) | 路由回 N6 时用 |
| `execution_result` | N7 | N8 |
| `simulation_csv_path` | N7 | N8, 前端下载 |
| `error_class` | N7 | 路由判断 + N11（报告降级路径） |
| `retry_count` | N7 | 路由判断 |
| `metrics` | N8 | N9, N10, N11 |
| `feature_metadata` | N8 | N11 |
| `confidence` | N8 | N11 |
| `experiment_protocols` | N9 | N11 |
| `paper_evidence` | N10 | N11 |
| `report` | N11 | 前端渲染 + 下载 |

---

## 3. 12 节点详细定义

### N1 — `n1_ner_entity_normalize`

| 维度 | 内容 |
| --- | --- |
| LLM | 1 次（`NODE1_NER_PROMPT`） |
| 输入 | `user_input`, `mcp_term_definitions`（v1 Node 0 注入）, `mcp_rewritten_query` |
| 输出 | `entities: list[Entity]` |
| 失败回退 | LLM 解析失败 → `entities=[]`, 写 `mechanism.need_human_review=True` |
| 文件 | `app/nodes_v2.py::n1_ner_entity_normalize` |

### N2 — `n2_mechanistic_planner`

| 维度 | 内容 |
| --- | --- |
| LLM | 1 次（`NODE2_PLANNER_PROMPT`） |
| 输入 | `entities`, `mcp_term_definitions` |
| 输出 | `mechanism: {pathway, cell, simulation_type, template, required_outputs, need_human_review, review_question}` |
| 失败回退 | 解析失败 → `template="Simple_Inhibition"`, `simulation_type="unknown"`, warning 日志，继续 |
| 文件 | `app/nodes_v2.py::n2_mechanistic_planner` |

`simulation_type` ∈ {`cascade_inhibition`, `cascade_activation`, `simple_inhibition`, `simple_activation`, `pkpd_onecomp`, `pkpd_twocomp`, `dose_sweep`, `combination`}
`template` ∈ {`Cascade_Inhibition`, `Cascade_Activation`, `Simple_Inhibition`, `Simple_Activation`, `PKPD_OneCompartment`, `PKPD_TwoCompartment`, `DoseSweep`, `Combination`}

### N3 — `n3_mechanism_rag`

| 维度 | 内容 |
| --- | --- |
| LLM | 0 次 |
| 输入 | `entities`, `mechanism.pathway` |
| 输出 | `mechanism.exemplars: list[{pathway, snippet, pmid}]` |
| 实现 | `RagCollections.search_mechanism(query, top_k=3)` |
| 文件 | `app/nodes_v2.py::n3_mechanism_rag` |

### N4 — `n4_kg_builder`

| 维度 | 内容 |
| --- | --- |
| LLM | 0 次 |
| 输入 | `entities`, `mechanism`, 旧版 `network_json`（兼容） |
| 输出 | `knowledge_graph: {nodes, edges, adjacency, topology_signature, is_acyclic, node_count, edge_count, dropped_edges}` |
| 实现 | `app/kg_builder.py::KGBuilder.build()` |
| 文件 | `app/nodes_v2.py::n4_kg_builder` |

**关键不变量**：
- 拓扑签名 = `NODES[id1,id2,...];EDGES[src|tgt|int,...]` 排序后字符串
- 出现环 → DFS 检测 → 丢弃"最弱"边（inhibition > conversion > activation），记入 `dropped_edges`

### N5 — `n5_parameter_rag`

| 维度 | 内容 |
| --- | --- |
| LLM | 0 次 |
| 输入 | `knowledge_graph`, `entities`, `species_context` |
| 输出 | `parameters: {edge_key: {param_name, value, unit, source, is_fallback, pmid}}`, `rag_fallback: bool` |
| 实现 | 对每条边调用 `RagCollections.search_parameter(edge_key, species)` |
| 硬约束 | LLM 永远看不见 `parameters`；N6 prompt 不包含此字段；模板渲染时由程序注入 |
| 文件 | `app/nodes_v2.py::n5_parameter_rag` |

**Fallback 标记**：当 ChromaDB 不可用 / 无命中时，每条边写入 `{is_fallback: true, source: "ESTIMATED", value: <heuristic_default>}`。

### N6 — `n6_ode_generator`

| 维度 | 内容 |
| --- | --- |
| LLM | 1 次（`NODE6_RELATION_PROMPT`），失败重试时 2 次（`NODE6_RETRY_PROMPT`） |
| 输入 | `knowledge_graph`, `mechanism.template`（不包含 `parameters`） |
| 输出 | `ode_model: {template, code, parameters_used, rule_violations}`, `network_relations: dict` |
| 管线 | (1) LLM 输出 `network_relations` → (2) `RuleEngine.check` → (3) 失败时回 N6 一次（带 `rule_violations`） → (4) `render_template(mechanism.template, {network_relations, parameters, ...})` |
| 文件 | `app/nodes_v2.py::n6_ode_generator` |

**LLM 禁令**：
- 禁止写 Python 代码
- 禁止给具体数值（Kd、Vmax、half-life）
- 禁止使用 markdown 代码块

### N7 — `n7_sandbox_execute`

| 维度 | 内容 |
| --- | --- |
| LLM | 0 次 |
| 输入 | `ode_model.code` |
| 输出 | `execution_result`, `simulation_csv_path`, `error_class`, `image_base64` |
| 管线 | (1) `ast.parse()` 预检 → (2) `_check_code_security` 静态黑名单 → (3) `subprocess` 执行（cwd=tempfile, timeout=60s） → (4) `_classify_error` → (5) 收集 CSV / PNG 路径 |
| 重试路由 | `error_class in {syntax, runtime, numerical, timeout}` 且 `retry_count < 3` → 回 N6；否则继续 N8 |
| 文件 | `app/nodes_v2.py::n7_sandbox_execute`, `app/sandbox.py::execute_simulation_code_v2` |

### N8 — `n8_scientific_features`

| 维度 | 内容 |
| --- | --- |
| LLM | 0 次 |
| 输入 | `execution_result`, `simulation_csv_path`, `knowledge_graph` |
| 输出 | `metrics: {species, overall, combo}`, `feature_metadata`, `confidence` |
| 实现 | `app/feature_extractor.py::ScientificFeatureExtractor.extract(csv_path, kg)` |
| 9 维指标 | peak / peak_time / half_life / steady_state / fold_change / auc / rise_time / decay_time / max_slope |
| 文件 | `app/nodes_v2.py::n8_scientific_features` |

**confidence 计算**：
```
base = 1.0
for species in species_list:
    if any metric is None: base -= 0.1
for warning in metadata.warnings: base -= 0.05
return max(base, 0.0)
```

### N9 — `n9_experiment_rag`

| 维度 | 内容 |
| --- | --- |
| LLM | 0 次 |
| 输入 | `knowledge_graph`, `mechanism`, `metrics` |
| 输出 | `experiment_protocols: list[{name, target, detection_method, cell_line, pmid, steps}]` |
| 实现 | `RagCollections.search_experiment(mechanism.pathway, output_species, top_k=5)` |
| 协议名 | ∈ {Western blot, Flow Cytometry, ELISA, qPCR, Immunohistochemistry} |
| 文件 | `app/nodes_v2.py::n9_experiment_rag` |

### N10 — `n10_evidence_rag`

| 维度 | 内容 |
| --- | --- |
| LLM | 0 次 |
| 输入 | `knowledge_graph`, `mechanism`, `metrics` |
| 输出 | `paper_evidence: list[{pmid, doi, title, figure_ref, cell_line, species, snippet}]` |
| 实现 | `RagCollections.search_evidence(mechanism, output_species, top_k=8)` |
| 硬约束 | 不编造 PMID / DOI；列表 = ChromaDB 真实返回 |
| 文件 | `app/nodes_v2.py::n10_evidence_rag` |

### N11 — `n11_scientific_report`

| 维度 | 内容 |
| --- | --- |
| LLM | 1 次（`NODE11_REPORT_FILL_PROMPT`），触发重试时 2 次（`NODE11_REPORT_RETRY_PROMPT`） |
| 输入 | `metrics`, `feature_metadata`, `experiment_protocols`, `paper_evidence`, `knowledge_graph`, `entities`, `mechanism`, `confidence` |
| 输出 | `report: {markdown, llm_filled_json, forbidden_terms_violations}` |
| 管线 | (1) LLM 填充 4 字段 JSON → (2) `ReportRenderer.check_forbidden_terms` → (3) 命中时用 retry prompt 再调 LLM 一次 → (4) `ReportRenderer.render(llm_filled, metrics, evidence, experiments, knowledge_graph, confidence)` |
| 文件 | `app/nodes_v2.py::n11_scientific_report`, `app/report_renderer.py::ReportRenderer` |

**LLM 字段**：
```json
{
  "mechanism_analysis": "TGF-beta 通过 SMAD3 抑制 CD8+ T 细胞激活。{metric.SMAD.peak:.2f} nM...",
  "simulation_interpretation": "在 {metric.CD8.peak_time:.1f} h 达到峰值，fold change {metric.CD8.fold_change:.1f}。",
  "discussion": "本结果与文献 PMID:{evidence[0].pmid} 报道一致。建议 Western blot 检测 pSMAD2。",
  "limitations": "参数来自估计；未考虑时间延迟。"
}
```

---

## 4. 错误分类与重试策略

### 4.1 error_class 分类器（`app/sandbox.py::_classify_error`）

```python
def _classify_error(returncode: int, stderr: str, was_timeout: bool, csv_path: str | None) -> str:
    if was_timeout: return "timeout"
    if returncode == 0:
        if csv_path and _csv_has_nan_inf(csv_path): return "numerical_error"
        return "none"
    if "SyntaxError" in stderr or "IndentationError" in stderr: return "syntax_error"
    if "ZeroDivisionError" in stderr or "OverflowError" in stderr or "NaN" in stderr: return "numerical_error"
    return "runtime_error"
```

### 4.2 路由表（`app/graph.py::_route_v2_after_sandbox`）

| error_class | retry_count | 路由目标 |
| --- | --- | --- |
| `none` | 任意 | N8（继续） |
| `syntax_error` / `runtime_error` / `numerical_error` | < 3 | N6（带 `error_class` + `stdout_stderr` + `rule_violations`） |
| `syntax_error` / `runtime_error` / `numerical_error` | ≥ 3 | N8（继续；`error_class` 保留在 state） |
| `timeout` | 任意 | N6（`stdout_stderr` 包含 "execution exceeded N seconds"） |

### 4.3 重试 Prompt 变体（`app/prompts_v2.py::NODE6_RETRY_PROMPT`）

按 `error_class` 注入针对性说明：

| error_class | 解释文本 |
| --- | --- |
| `syntax_error` | Python 语法错误。请检查括号、缩进、关键字拼写。 |
| `runtime_error` | 运行时异常（如 NameError/TypeError）。请检查变量名与函数调用。 |
| `numerical_error` | 数值异常（NaN/Inf/Overflow）。请检查参数范围、初值、单位是否匹配（h, nM）。 |
| `timeout` | 代码运行超过 60 秒。请减小 t_end 或降低 n_eval 精度。 |

---

## 5. RAG 四路分类

`RagCollections`（`app/rag_collections.py`）封装 4 个 Chroma collection。每个 collection 有独立的 metadata 字段、embedding 文档、查询接口。

| 角色 | Collection 名 | Metadata 关键字段 | Embeddable 文档 | 谁读 |
| --- | --- | --- | --- | --- |
| mechanism | `biodynamics_mechanism` | pathway, cell_type, species, key_genes, snippet | `f"{pathway} {cell_type} {snippet[:500]}"` | N3 |
| parameter | `biodynamics_parameter` | edge_key, source, target, interaction, param_name, value, unit, species, pmid, confidence | `f"{source} {interaction} {target} {param_name} {context[:300]}"` | N5 |
| experiment | `biodynamics_experiment` | name, target, cell_line, species, pmid, protocol_steps | `f"{name} {target} {cell_line} {species}"` | N9 |
| evidence | `biodynamics_evidence` | pmid, doi, title, figure_ref, cell_line, species, snippet | `f"{title} {figure_ref} {cell_line} {species} {snippet[:300]}"` | N10 |

### 5.1 写入权限

- N3 读 mechanism；N3 不写
- N5 读 parameter；N5 不写（建库时通过 `scripts/build_rag_db.py` 写入）
- N9 读 experiment；N9 不写
- N10 读 evidence；N10 不写
- **N6 / N7 / N8 / N11 完全不接触任何 RAG 集合**（硬约束）

### 5.2 离线灌库

详见 README [§离线 RAG 知识库构建](README.md#离线-rag-知识库构建) 与 [§v2 四路 RAG 数据迁移](README.md#v2-四路-rag-数据迁移可选)。

---

## 6. 模板与 Rule 引擎

### 6.1 8 个 ODE 模板

| 模板 | simulation_type | 必填变量 | 输出图形 |
| --- | --- | --- | --- |
| `Cascade_Inhibition.j2` | cascade_inhibition | species_names, edges[], parameters{}, t_end, n_eval, y0 | 多线时间序列 |
| `Cascade_Activation.j2` | cascade_activation | 同上 | 多线时间序列 |
| `Simple_Inhibition.j2` | simple_inhibition | species_names, inhibitor, target, kd, n_hill, degradation, production, t_end, n_eval, y0 | 双线时间序列 |
| `Simple_Activation.j2` | simple_activation | 同上 | 双线时间序列 |
| `PKPD_OneCompartment.j2` | pkpd_onecomp | drug_name, dose, k10, target, ec50, emax, gamma, t_end, n_eval, y0 | 2 子图（PK + PD） |
| `PKPD_TwoCompartment.j2` | pkpd_twocomp | drug_name, dose, k10, k12, k21, target, ec50, emax, gamma, t_end, n_eval, y0 | 2 子图（PK + PD） |
| `DoseSweep.j2` | dose_sweep | drug_name, target, ec50, emax, gamma, conc_min_factor, conc_max_factor, n_points, t_end, y0 | 浓度-效应曲线 + 打印 IC50/IC90/HED |
| `Combination.j2` | combination | drugs[{name, dose, ec50, emax, gamma, target}], t_end, n_eval, y0 | 3 子图（A / B / combo）+ 打印 Chou-Talalay CI |

详细变量与示例见 [TEMPLATES.md](TEMPLATES.md)。

### 6.2 6 条 Rule（`app/rule_engine.py`）

| Rule | 校验项 | severity |
| --- | --- | --- |
| `TemplateRule` | 每条边都有 template 槽位 | error |
| `ParameterRangeRule` | Kd ∈ [1e-5, 1e6] nM；half_life ∈ [0.01, 1000] h | error |
| `UnitRule` | unit ∈ {"nM", "h", "M", "s"} | error |
| `ActivationDirectionRule` | inhibition 边必须有 Kd；activation 边必须有 Vmax | error |
| `HillCoefficientRule` | n ∈ [0.5, 6.0] | error |
| `InitialValueRule` | y0[i] ∈ [0, 1e6] | error |

### 6.3 校验失败行为

```python
result: RuleResult = rule_engine.check(network_relations, parameters)
if not result.ok and state["retry_count"] < 1:
    return {
        "rule_violations": [v.__dict__ for v in result.violations],
        "retry_count": state["retry_count"] + 1,
    }  # 路由回 N6
```

---

## 7. 报告渲染流程

### 7.1 4 字段 JSON Fill（`NODE11_REPORT_FILL_PROMPT`）

LLM 接收 4 个 prompt slot + 上下文（entities / mechanism / metrics / evidence / experiments），输出：

```json
{
  "mechanism_analysis": "TGF-β 通过 SMAD3 抑制 CD8+ T 细胞激活...",
  "simulation_interpretation": "在 12.3h 达到峰值，fold change 2.5...",
  "discussion": "本结果与文献 PMID:12345678 报道一致...",
  "limitations": "参数来自估计；未考虑时间延迟。"
}
```

LLM 严禁：
- 写 Python 代码
- 编造 PMID（必须从 `paper_evidence` 选取）
- 使用 `["某疾病", "T1", "T2", "炎症因子", "等等", "TGF-betta"]` 之一

### 7.2 禁止词校验（`ReportRenderer.check_forbidden_terms`）

```python
violations = renderer.check_forbidden_terms(llm_filled)
# violations: list[str], e.g. ["mechanism_analysis 包含禁止词：'T1'"]
```

### 7.3 Retry / 降级

```python
if violations and report_retry_count < 1:
    # 用 NODE11_REPORT_RETRY_PROMPT（含显式禁止词列表）重叫一次 LLM
    llm_filled = retry_chain.invoke({...})
    violations = renderer.check_forbidden_terms(llm_filled)
    report_retry_count += 1
if violations:
    # 仍违例：写入 report.forbidden_terms_violations，渲染仍执行但显示违例
    pass
```

### 7.4 Jinja2 渲染（`ReportRenderer.render`）

```python
md = renderer.render(
    llm_filled=llm_filled,
    metrics=metrics,
    evidence=paper_evidence,
    experiments=experiment_protocols,
    knowledge_graph=knowledge_graph,
    confidence=confidence,
)
```

模板：`backend/app/report_templates/standard.md.j2`，详见 [TEMPLATES.md](TEMPLATES.md#report-template)。

---

## 8. 多智能体映射

详见 [README.md#多智能体编排](README.md#多智能体编排)。v2 共 9 个 Agent 映射：

```
N1 + N2   → Entity & Planning Agent
N3        → Mechanism Retrieval Agent
N4        → Knowledge Graph Engineer
N5        → Parameter Retrieval Agent
N6 + N7   → Simulation Engineer Agent
N8        → Scientific Analytics Agent
N9        → Experimental Design Agent
N10       → Evidence Synthesis Agent
N11       → Scientific Report Agent
```

注册表：[`app/supervisor.py::AGENT_REGISTRY_V2`](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/supervisor.py)。

---

## 9. SSE 事件清单

详见 [README.md#api-接口](README.md#api-接口)。v2 额外下发 6 个事件：

| event | 触发节点 | payload |
| --- | --- | --- |
| `workflow_v2_state` | 入口 + 每节点 start | `{current_node, step_index, total_steps, pipeline, status}` |
| `knowledge_graph` | N4 | `{node_count, edge_count, is_acyclic, topology_signature, dropped_edges}` |
| `rule_violations` | N6 | `{violations: [{rule, edge_key, message, severity}]}` |
| `metrics` | N8 | `{species, overall, combo, confidence}` |
| `experiment_protocols` | N9 | `{protocols: [...]}` |
| `paper_evidence` | N10 | `{evidence: [...]}` |
| `report` | N11 | `{markdown, llm_filled_json, forbidden_terms_violations}` |

---

## 10. 关键不变量与硬约束

按重要性排序：

1. **LLM 不写代码**：N6 LLM 仅输出 `network_relations` JSON；ODE Python 由 `app/ode_templates/*.j2` 渲染生成
2. **LLM 不编数值**：N5 `parameters` 永不进入 N6 LLM 的 prompt；模板渲染时由程序注入
3. **LLM 不编文献**：N10 `paper_evidence` 是 ChromaDB 真实返回；N11 报告仅引用这些 evidence
4. **Rule Engine 在 LLM 之后**：先 LLM 输出，再 Python 跑 6 条 Rule；不通过 → 自动回 N6 一次
5. **AST 预检在 subprocess 之前**：N7 `ast.parse()` 拦 SyntaxError，避免无谓 60s 启动
6. **错误分类驱动重试**：`error_class` ∈ {`none`/`syntax`/`runtime`/`numerical`/`timeout`}；前 4 类且 `retry_count < 3` 路由回 N6
7. **特征提取零 LLM**：N8 纯 NumPy 算 9 维指标
8. **报告禁止词强校验**：N11 渲染前扫黑名单，命中则重试一次
9. **拓扑签名稳定可比**：N4 `topology_signature` 排序后字符串
10. **环路检测 → 弱边打破**：N4 出现环时丢弃 `inhibition` 优先的边，warning 写入 `dropped_edges`
11. **共享工具不区分 v1/v2**：`orchestrator` / `token_usage` / `mcp_client` 两套工作流共用
12. **新增功能默认 v2**：v1 仅做兼容性维护

---

**反馈与修订**：如发现定义与代码不一致，请优先以代码为准并提交 PR；本文档由 [.trae/documents/biodynamics-v2-upgrade-plan.md](.trae/documents/biodynamics-v2-upgrade-plan.md) 派生。
