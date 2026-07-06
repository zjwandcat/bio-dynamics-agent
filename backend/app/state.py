# BioDynamics Agent - LangGraph 状态定义
# 描述整个仿真工作流在节点之间传递的数据结构。

import operator
from typing import Annotated, TypedDict


def reset_if_empty_list(existing: list, new: list) -> list:
    """自定义 reducer：新请求传入空列表时重置，同请求内累加。

    替代 operator.add，解决 initial_state 设 [] 被 reducer 计算为
    [] + existing = existing 而无法清空的跨请求数据污染问题。
    """
    if new == []:
        return []
    return (existing or []) + new


class BioDynamicsState(TypedDict, total=False):
    """LangGraph 全局状态。所有字段均为可选，节点返回时只需提供发生变化的键值对。"""

    user_input: str
    messages: Annotated[list[str], reset_if_empty_list]
    network_json: dict
    need_human_review: bool
    review_question: str
    python_code: str
    execution_status: str
    stdout_stderr: str
    image_base64: str
    retry_count: int
    auditor_status: str
    correction_suggestion: str
    failure_report: str
    final_report: str
    token_usage: dict

    # RAG 与知识注入相关字段
    species_context: str
    sbml_model_text: str
    # P0-3 修复：SBML 模型 ID（用于 BioModels API 加载与模板选择 grounding）
    sbml_model_id: str
    sbml_parsed_network: dict
    rag_retrieved_params: list[dict]
    rag_selected_params: dict[str, dict]
    rag_fallback: bool
    rag_summary: str

    # 多智能体编排相关字段（对应 1233.md 升级规范）
    # agent_dispatches 记录每次智能体调度的历史，供前端工作流追踪器渲染
    agent_dispatches: Annotated[list[dict], reset_if_empty_list]
    # rag_insights 存储高阶 RAG 的可视化数据：rewritten_query / rewrites / source_distribution / top_selections
    rag_insights: dict
    # rag_hit_rate 表示本次检索的命中率（0.0-1.0），供前端性能监控标签展示
    rag_hit_rate: float

    # MCP 工具集成相关字段
    # mcp_term_definitions 存储术语标准化定义列表，供前端术语卡片渲染
    mcp_term_definitions: list[dict]
    # mcp_tool_calls 记录每次 MCP 工具调用的状态记录，供前端工具调用状态面板渲染
    mcp_tool_calls: Annotated[list[dict], reset_if_empty_list]
    # mcp_tokens_saved 估算通过 MCP 术语标准化节省的 Token 总数
    mcp_tokens_saved: int
    # mcp_rewritten_query 经术语标准化重写后的查询，供 RAG 节点提升检索精准度
    mcp_rewritten_query: str

    # =============================================================================
    # 医学升级：PK/PD、药物知识图谱、联合用药、剂量递增
    # =============================================================================
    # Upgrade 1: PK/PD 推断结果（Node 1.6 产出）
    # {drug_name, drug_target, route, compartment, pk_params, pd_params}
    pkpd_profile: dict
    # Upgrade 2: 靶点-药物真实世界知识图谱
    # [{drug_name, ic50, ec50, clinical_dose, source, is_clinical_candidate, target_name}]
    drug_candidates: list[dict]
    # ClinicalTrials.gov 临床试验信息（在线时填充）
    # [{nct_id, phase, condition, status}]
    clinical_trial_info: list[dict]
    # Upgrade 3: 联合用药方案（Node 1.6 一次性产出，后续只读）
    # [{drug_name, dose, ec50, emax, gamma, target}]
    drug_regimen: list[dict]
    # Chou-Talalay 联合用药指数 {fa_0.5: CI, fa_0.75: CI, fa_0.9: CI}
    combination_index: dict
    # 协同评估结论："潜在协同" | "叠加效应" | "拮抗风险" | ""
    synergy_assessment: str
    # Upgrade 4: 剂量递增与治疗窗口
    # {concentrations: [...], effects: [...], drug_name}
    dose_response_data: dict
    # 仿真得出的 IC50 / IC90 / HED（人体等效剂量）
    ic50: float
    ic90: float
    hed: float
    # v1 node3 writes this key without declaring it; declare explicitly to silence TypedDict
    # misuse warnings and document the v1→v2 migration path.
    simulation_ci: dict  # DEPRECATED: v2 uses combination_index instead

    # =============================================================================
    # v2 升级：12 节点流水线、模板 + 规则、四路 RAG、科学特征提取、模板报告
    # 详见 .trae/documents/biodynamics-v2-upgrade-plan.md
    # =============================================================================

    # v3 升级：Supervisor-Worker 动态编排、人在环路、上下文压缩
    # =============================================================================
    mode: str                            # auto_fast | auto_standard | manual
    manual_modules: list[str]           # Manual 模式下用户勾选的模块键
    execution_plan: list[str]           # PreRouter 生成的 Worker 执行计划
    current_step: int                   # 当前执行到 execution_plan 的索引
    completed_workers: list[str]        # 已完成的 Worker 列表
    pending_clarification: dict | None  # 待人工干预的问题与选项
    clarification_response: dict | None # 用户回灌的干预答案
    clarification_request: dict | None  # 供 SSE 发射的 clarification 事件数据
    clarification_resolved: bool        # 人工干预是否已被消费并继续执行
    stop_requested: bool                # 用户是否点击停止生成
    raw_cache: dict                     # 超大原始数据的缓存（不传入 LLM prompt）
    next_worker: str                    # Supervisor 决定的下一个 Worker

    # N1 / N2 — NER + 机制规划
    entities: list[dict]                # [{entity_id, name, type, aliases, canonical_id}]
    mechanism: dict                     # {pathway, cell, simulation_type, template, required_outputs, exemplars}

    # N4 — 知识图谱（pure Python，零 LLM）
    knowledge_graph: dict               # {nodes, edges, adjacency, topology_signature, is_acyclic, ...}

    # N5 — 参数 RAG（程序注入，LLM 禁止修改）
    # 参数对象必须包含溯源四元组：value / source / confidence / origin
    #   - value: float | None（None 表示 missing_parameter）
    #   - source: "RAG" | "SBML" | "PubMed" | "KEGG" | "UniProt" | "ChEMBL" | "Inferred"
    #   - confidence: float in [0.0, 1.0]（数值化置信度，0.3 以下触发 online fallback）
    #   - origin: str（具体来源标识，如 "PMID:12345" / "BIOMD0000000205" / "KEGG:hsa04010"）
    # 缺失策略：当 confidence < 0.3 时必须标记 missing_parameter=True 并触发在线回退，
    # 禁止 LLM 估算默认值；is_fallback=True 表示已使用估算兜底（仅 Template-only 模式允许）。
    parameters: dict                    # {edge_key: {param_name, value, unit, source, confidence, origin, is_fallback, missing_parameter}}

    # N6 — ODE 生成（Template + Rule）
    ode_model: dict                     # {template, code, parameters_used, rule_violations, template_selection, reaction_graph, domain_check_summary, time_unit}
    network_relations: dict             # LLM 输出的定性关系（不写代码、不给数值）
    # P0-3 修复：Reaction Graph 中间表示（KG → Reaction Graph → Template → ODE）
    reaction_graph: dict
    # P0-3 修复：模板选择元信息（规则引擎输出，供报告与可观测性使用）
    template_selection: dict

    # N7 — 沙箱执行（v2 含 AST 预检 + 错误分类）
    execution_result: dict              # 镜像 N3 输出，并新增 simulation_csv_path / error_class
    simulation_csv_path: str            # 沙箱代码生成的 simulation.csv 绝对路径
    error_class: str                    # "none" | "syntax_error" | "runtime_error" | "numerical_error" | "timeout"
    sandbox_failure_reason: str         # 重试耗尽时的失败原因描述，供报告模板使用

    # P0-4 升级：SBML Validator 节点输出（对应修复提示词1.md §二.6）
    # SBML 三角色定位 + 验证报告，量化模板仿真 vs SBML 真实仿真的差异
    sbml_role: str                      # "primary_ground_truth" | "calibration_reference" | "validation_oracle" | "none"
    validation_report: dict             # {error_diff, peak_time_diff, amplification_diff, sbml_sim_available, method, structural_confidence_score}

    # 系统降级模式（Full / RAG-only / Template-only）
    # - Full Mode: RAG + SBML + Sandbox 全部可用
    # - RAG-only Mode: SBML 失败时，仅使用 RAG 参数
    # - Template-only Mode: RAG 严重缺失时，使用带警告标签的模板默认参数
    # 报告中必须根据此字段标注 confidence: low 或 parameters: estimated
    degradation_mode: str               # "full" | "rag_only" | "template_only"
    # 缺失参数清单（confidence < 0.3 或 source=Inferred），供报告标注与人工干预
    missing_parameters: list[str]       # ["edge_key:param_name", ...]

    # N8 — 科学特征提取（pure NumPy，零 LLM）
    metrics: dict                       # {species: {peak, peak_time, ...}, overall: {...}}
    feature_metadata: dict              # {method, version, confidence, warnings}
    confidence: float                   # 0..1 整体置信度

    # N9 / N10 — 实验方案 RAG + 文献证据 RAG
    experiment_protocols: list[dict]    # [{name, target, detection_method, cell_line, pmid, ...}]
    paper_evidence: list[dict]          # [{pmid, doi, title, figure_ref, cell_line, species}]

    # N11 — 报告（Python Markdown 模板 + LLM JSON 填充）
    report: dict                        # {markdown, llm_filled_json, forbidden_terms_violations}

    # =============================================================================
    # v4 迁移字段（Phase 1 新增）
    # 详见 BioDynamics_v4_Migration_Plan.md
    # 命名规则：v4_ 前缀，与 v3 字段共存，P7 阶段统一清理
    # =============================================================================
    # v4 Ontology Agent 输出（Phase 1）
    # 结构：{entities: [{name, hgnc_id, uniprot_id, chebi_id, go_terms, sbo_term,
    #                    species_type, verified, source}], pathway_class, warnings}
    # V4_ONTOLOGY_AGENT_ENABLED=false 时保持 None，不影响 v3 流程
    v4_ontology_entities: dict
