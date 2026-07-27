# Prompt and Template Index

本索引区分 active、compatibility、conditional 和 template。精确 Prompt 文本以源码为准；
本文记录 owner、consumer 和用途，避免仅按名称误删。

## Active v2 prompts

| Prompt | 文件 | Consumer | 输出 | 状态 |
|---|---|---|---|---|
| `N1_NER_PROMPT` | `backend/app/prompts_v2.py` | `nodes_v2.py::n1_ner_entity_normalize` | entities JSON | active Standard/Manual |
| `N2_PLANNER_PROMPT` | 同上 | `n2_mechanistic_planner` | mechanism/template/edges JSON | active |
| `N3_MECHANISM_RAG_PROMPT` | 同上 | `n3_mechanism_rag` | mechanism description JSON | active |
| `N5_PARAMETER_DECISION_PROMPT` | 同上 | `n5_parameter_rag` | selected parameter JSON | active |
| `N6_ODE_PROMPT` | 同上 | `n6_ode_generator` | qualitative relations JSON，禁止 Python | active |
| `N11_REPORT_FILL_PROMPT` | 同上 | `n11_scientific_report` | report fields JSON | active |

## Legacy / compatibility prompts

`backend/app/prompts.py` 仍不能整体删除，因为 v3 worker 复用 `nodes.py`：

| Prompt | Consumer / relevance | 状态 |
|---|---|---|
| `NODE1_PARSER_PROMPT` | Fast mechanism `nodes.py::node1_parse_network` | active in auto_fast |
| `NODE1_6_PKPD_PROMPT` | `nodes.py::node1_6_pkpd_inference` | active Standard when PKPD selected |
| `NODE2_BASE_PROMPT` | legacy direct ODE/code paths and retry compatibility | compatibility，确认调用后再删 |
| `NODE4_AUDITOR_PROMPT` | sandbox audit/correction path | active retry compatibility |
| `NODE6_REPORT_PROMPT` | legacy report path | legacy/compatibility |
| `RAG_EXTRACTION_PROMPT` | data extraction/build scripts | offline/data pipeline |
| `RAG_DECISION_PROMPT` | legacy parameter selection | compatibility |
| `SBML_PARSER_PROMPT` | legacy SBML parsing | compatibility |
| `ORCHESTRATOR_PROMPT` | v1 supervisor concept | legacy; current supervisor is rule/state based |
| `RAG_SPECIALIST_PROMPT` | legacy specialist | compatibility |
| `QUERY_REWRITING_PROMPT` | query normalization | active through RAG clients where imported |

## MCP / RAG prompts

| Prompt | 文件 | 用途 |
|---|---|---|
| `TERM_EXTRACTION_PROMPT` | `backend/app/mcp_client.py` | 从用户文本抽取 biomedical terms |
| `TERM_DEFINITION_PROMPT` | 同上 | 无 MCP/补充时由 LLM 标准化定义 |
| `DRUG_EXTRACTION_PROMPT` | `backend/app/rag_client.py` | PubMed snippet -> drug candidates |
| inline rewrite prompts | `rag_client.py` | query rewrite / relevance processing |
| inline PubMed parameter prompt | `nodes_v2.py` | 文献参数抽取/决策 |

## LangGraph inline prompts

| 位置 | 用途 | 状态 |
|---|---|---|
| `graph_v3.py::_build_standard_plan` | 无法规则判断时决定 PKPD/evidence | active Standard |
| `graph_v3.py::_llm_auto_decide_clarification` | 120s HITL 超时后选已有选项 | active fallback |
| `nodes_v2.py::n0/n1/n2/n3/n5/n6/n11` 附近 | structured calls 的 system/human wrappers | active |
| `nodes.py` 多处 `ChatPromptTemplate.from_messages` | Fast/PKPD/audit/legacy paths | mixed，逐调用确认 |

## Scientific Alignment prompts

| Prompt/Builder | 文件 | Gate / purpose |
|---|---|---|
| `get_scientific_discussion_prompt` | `scientific_alignment/discussion_prompt.py` | `SA_SEVEN_AXIS`; 10 questions + evidence tags |
| Discussion renderer | `scientific_alignment/discussion_renderer.py` | rule/template evidence-first rendering |
| Scientific critic concerns | `scientific_alignment/scientific_critic.py` | concerns 可注入 report retry |
| Reviewer logic | `scientific_alignment/scientific_reviewer.py` | 主要结构化/rule，不应混同 free prompt |
| Experiment planner | `scientific_alignment/experiment_planner.py` | sprint4 可使用 YAML rule engine，非 LLM |

Scientific Alignment 的治理顺序在 `loop_controller.py` 明确把 prompt 放最后：先修
mechanism/evidence/BioModels/parameter/simulation/validation/discussion，最后才优化 prompt。

## Jinja ODE templates

### v3 active top-level

`Simple_Activation`, `Simple_Inhibition`, `Cascade_Activation`,
`Cascade_Inhibition`, `Signaling_Cascade_Phos`, `PKPD_OneCompartment`,
`PKPD_TwoCompartment`, `Combination`, `DoseSweep`。

Helpers：`_cascade_helpers`, `_mechanism_binding`, `_mechanism_phosphorylation`。
Loader：`backend/app/ode_templates/__init__.py::render_template`；selector：
`backend/app/template_selector.py`。

### v4 active top-level

`bistable_switch`, `caspase_cascade`, `cyclin_cdk_toggle`, `destruction_complex`,
`nuclear_transport`, `oscillatory_feedback`, `transcription_factor`,
`transcriptional_delay`, `ubiquitination_cascade`。

Helpers：`_dde_helpers`, `_mechanism_phosphorylation_mm`。Loader/selector：
`backend/app/ode_renderer_v2.py::ODERendererV2`。

## Report template

`backend/app/report_templates/standard.md.j2` 由
`backend/app/report_renderer.py::ReportRenderer.render` 消费。N11 先让 LLM 输出结构化
fields，再由 Jinja 渲染 Markdown；不要让 LLM 自由生成最终结构替代模板。

## Prompt 修改规则

1. 先确认 consumer 是否在当前主链和目标 mode 中执行。
2. Prompt 不能替代 schema、rule、template、validation 或 benchmark。
3. 修改 structured output 时同步 Pydantic model/parser/tests。
4. 修改 N2/N6 会影响所有通路，必须跑多通路 regression。
5. 不得把 benchmark expected values 塞入一般运行 Prompt 来“提高通过率”。
