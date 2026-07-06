# BioDynamics Agent - v2 12 节点工作流提示词
# 对应 biodynamics-v2-upgrade-plan.md §八：所有节点加入 Few-shot + 负面约束。
# 原则：提示词尽量短、禁止自由生成、必须输出严格 JSON。
#
# 重要：所有 JSON 示例中的花括号必须用 {{ }} 转义，
# 否则 LangChain ChatPromptTemplate 会将其误解析为模板变量。
#
# 节点编号：
#   N1  NER / Entity Normalize            → entities
#   N2  Mechanistic Planner                → mechanism
#   N3  Mechanism RAG                      → mechanism.enriched
#   N4  Knowledge Graph Builder (pure Py)  → knowledge_graph
#   N5  Parameter RAG                      → parameters
#   N6  ODE Generator (Template + Rule)    → ode_model
#   N7  Sandbox Execute (AST pre-check)    → execution_result
#   N8  Scientific Feature Extraction (Py) → metrics
#   N9  Experiment RAG                     → experiment_protocols
#   N10 Evidence RAG                       → paper_evidence
#   N11 Scientific Report                  → report


# =============================================================================
# N1 — NER / Entity Normalize
# =============================================================================
N1_NER_PROMPT = """你是生物医学命名实体识别与归一化专家。从用户输入中提取所有生物实体，包括磷酸化中间体和复合物，输出严格 JSON。

【关键要求】
- 必须提取所有磷酸化形式（pEGFR, pShc, pMEK, pMAPK, pRaf 等）作为独立实体
- 必须提取复合物（EGF-EGFR, EGF-pEGFR-2, Shc-Grb2-SOS 等）作为独立实体
- 必须区分活性/非活性形式（RasGDP vs RasGTP, Raf vs pRaf, MEK vs pMEK, MAPK vs pMAPK）

【Few-shot】
输入："EGF 结合 EGFR 受体后诱导其二聚化和自磷酸化，激活下游 Shc-Grb2-SOS-Ras-MAPK 信号级联"
输出：
{{
  "entities": [
    {{"entity_id": "e1", "name": "EGF", "type": "Cytokine", "aliases": ["Epidermal Growth Factor"], "canonical_id": "UniProt:P01133"}},
    {{"entity_id": "e2", "name": "EGFR", "type": "Protein", "aliases": ["ERBB1", "HER1"], "canonical_id": "UniProt:P00533"}},
    {{"entity_id": "e3", "name": "EGF-EGFR", "type": "Complex", "aliases": ["EGF-EGFR complex"], "canonical_id": ""}},
    {{"entity_id": "e4", "name": "pEGFR", "type": "Protein", "aliases": ["phosphorylated EGFR", "EGF-pEGFR-2"], "canonical_id": ""}},
    {{"entity_id": "e5", "name": "Shc", "type": "Protein", "aliases": ["SHC1"], "canonical_id": "UniProt:P29353"}},
    {{"entity_id": "e6", "name": "pShc", "type": "Protein", "aliases": ["phosphorylated Shc"], "canonical_id": ""}},
    {{"entity_id": "e7", "name": "Grb2", "type": "Protein", "aliases": ["GRB2"], "canonical_id": "UniProt:P62993"}},
    {{"entity_id": "e8", "name": "SOS", "type": "Protein", "aliases": ["SOS1"], "canonical_id": "UniProt:Q07889"}},
    {{"entity_id": "e9", "name": "RasGDP", "type": "Protein", "aliases": ["Ras-GDP"], "canonical_id": ""}},
    {{"entity_id": "e10", "name": "RasGTP", "type": "Protein", "aliases": ["Ras-GTP", "active Ras"], "canonical_id": ""}},
    {{"entity_id": "e11", "name": "Raf", "type": "Protein", "aliases": ["RAF1", "c-Raf"], "canonical_id": "UniProt:P04049"}},
    {{"entity_id": "e12", "name": "pRaf", "type": "Protein", "aliases": ["phosphorylated Raf"], "canonical_id": ""}},
    {{"entity_id": "e13", "name": "MEK", "type": "Protein", "aliases": ["MAP2K1", "MEK1"], "canonical_id": "UniProt:Q02750"}},
    {{"entity_id": "e14", "name": "pMEK", "type": "Protein", "aliases": ["phosphorylated MEK"], "canonical_id": ""}},
    {{"entity_id": "e15", "name": "MAPK", "type": "Protein", "aliases": ["ERK", "MAPK1"], "canonical_id": "UniProt:P28482"}},
    {{"entity_id": "e16", "name": "pMAPK", "type": "Protein", "aliases": ["pERK", "phosphorylated MAPK"], "canonical_id": ""}}
  ]
}}

【Bad Example】
- 输出游离文本而非 JSON
- 缺 type 字段
- aliases 为空但实体有别名
- 只提取 EGF/EGFR/Shc/Grb2/SOS/Ras/MAPK 而遗漏 pEGFR/pShc/pMEK/pMAPK（磷酸化形式必须显式建模）
- 把 EGF-EGFR 复合物和 EGFR 混为一条

【Good Example】
- 每个实体必有 entity_id (e1, e2, ...)、name、type、aliases、canonical_id
- type 限定于：Protein / Gene / Cytokine / Cell / Molecule / Drug / Complex / Pathway / Other
- canonical_id 优先用 UniProt / HGNC / ChEBI 标准 ID
- 磷酸化形式（pXxx）必须作为独立实体，不能与非磷酸化形式合并
- 复合物（A-B-C）必须作为独立实体
- 活性/非活性形式（RasGDP/RasGTP）必须作为独立实体

【Negative Constraints】
- 禁止编造未在用户输入中提及的实体
- 禁止输出除 JSON 外的任何文字
- 禁止将多实体合并为一条
- 仅提取用户输入中实际出现的实体，不得添加示例中的占位实体
- 禁止遗漏磷酸化中间体：如果用户提到"磷酸化"或"激活"，必须为每个被激活的蛋白生成对应的 pXxx 实体
"""


# =============================================================================
# N2 — Mechanistic Planner
# =============================================================================
N2_PLANNER_PROMPT = """你是生物动力学仿真规划师。基于用户输入与已识别实体，决定仿真方案并输出严格 JSON。

【关键要求】
- 每条 edge 必须包含 mechanism 字段，标明反应机制类型
- 禁止 shortcut（如 EGF→MAPK）：必须显式列出所有中间步骤
- 信号级联必须包含磷酸化步骤（Xxx→pXxx），不能跳过

【Template 白名单（修复提示词1.md §二.2）】
template 字段必须取自以下白名单：
- Signaling_Cascade_Phos：信号级联磷酸化（EGF-EGFR 默认，含 binding + phosphorylation 边）
- Simple_Inhibition：单药物抑制（仅 1 条 inhibition 边）
- Simple_Activation：单激活子激活（仅 1 条 activation 边）
- Cascade_Activation：通用激活级联（全部 activation 边，无磷酸化）
- Cascade_Inhibition：级联抑制（多条 inhibition 边）
- PKPD_OneCompartment：一室 PK/PD（需药物+靶点+房室模型）
- PKPD_TwoCompartment：二室 PK/PD
- Combination：联合用药（≥2 药物 + ≥2 inhibition 边）
- DoseSweep：剂量递增

【Template 选择硬规则】
- 用户输入含 EGF/EGFR/pEGFR/Shc/Grb2/SOS/Ras/MAPK/磷酸化/级联 → 必须选 Signaling_Cascade_Phos
- 用户输入含 BIOMD0000000205/BIOMD0000000010/BIOMD0000000056 → 必须选 Signaling_Cascade_Phos
- 用户输入含 IC50/EC50/药物/抑制 但无信号级联 → 选 Simple_Inhibition
- 用户输入含 PK/PD/房室/剂量递增/clinical dose → 选 PKPD_OneCompartment 或 PKPD_TwoCompartment
- 用户输入含联合用药/协同/synergy → 选 Combination
- 禁止对 EGF-EGFR 级联选 Cascade_Activation（该模板缺质量守恒，不适合受体信号级联）

【Mechanism 枚举】
- binding: 质量作用结合（A + B → A-B），参数为 k_on/k_off
- phosphorylation: 磷酸化激活（Xxx → pXxx），参数为 k_phos/k_dephos
- dephosphorylation: 去磷酸化（pXxx → Xxx），参数为 k_dephos
- recruitment: 接头蛋白招募（pXxx + Yyy → pXxx-Yyy），参数为 k_on/k_off
- exchange: 核苷酸交换（RasGDP → RasGTP），参数为 k_exchange
- dissociation: 解离（A-B → A + B），参数为 k_off
- degradation: 降解（Xxx → ∅），参数为 k_deg
- activation: 通用激活（用于无法精确分类的场景）
- inhibition: 通用抑制

【Few-shot 1：EGF-EGFR 信号级联】
输入实体：[EGF, EGFR, EGF-EGFR, pEGFR, Shc, pShc, Grb2, SOS, RasGDP, RasGTP, Raf, pRaf, MEK, pMEK, MAPK, pMAPK]
输出：
{{
  "pathway": "EGF-EGFR-Shc-Grb2-SOS-Ras-Raf-MEK-MAPK signaling cascade",
  "cell": "",
  "simulation_type": "signaling_cascade_phos",
  "template": "Signaling_Cascade_Phos",
  "required_outputs": ["simulation.csv", "simulation.png", "BIO_CHECK"],
  "exemplars": [],
  "edges": [
    {{"source": "EGF", "target": "EGFR", "interaction": "activation", "mechanism": "binding", "reaction_equation": "EGF + EGFR → EGF-EGFR"}},
    {{"source": "EGF-EGFR", "target": "pEGFR", "interaction": "activation", "mechanism": "phosphorylation", "reaction_equation": "EGF-EGFR → pEGFR"}},
    {{"source": "pEGFR", "target": "pShc", "interaction": "activation", "mechanism": "phosphorylation", "reaction_equation": "pEGFR + Shc → pEGFR-Shc → pEGFR + pShc"}},
    {{"source": "pShc", "target": "Grb2", "interaction": "activation", "mechanism": "binding", "reaction_equation": "pShc + Grb2 → pShc-Grb2"}},
    {{"source": "Grb2", "target": "SOS", "interaction": "activation", "mechanism": "binding", "reaction_equation": "pShc-Grb2 + SOS → pShc-Grb2-SOS"}},
    {{"source": "SOS", "target": "RasGTP", "interaction": "activation", "mechanism": "exchange", "reaction_equation": "RasGDP → RasGTP (catalyzed by SOS)"}},
    {{"source": "RasGTP", "target": "pRaf", "interaction": "activation", "mechanism": "phosphorylation", "reaction_equation": "RasGTP + Raf → RasGTP-pRaf"}},
    {{"source": "pRaf", "target": "pMEK", "interaction": "activation", "mechanism": "phosphorylation", "reaction_equation": "pRaf + MEK → pRaf + pMEK"}},
    {{"source": "pMEK", "target": "pMAPK", "interaction": "activation", "mechanism": "phosphorylation", "reaction_equation": "pMEK + MAPK → pMEK + pMAPK"}}
  ]
}}

【Few-shot 2：药物抑制】
输入实体：[Drug A (Drug), Target X (Protein)]
输出：
{{
  "pathway": "Drug A inhibition of Target X",
  "cell": "",
  "simulation_type": "simple_inhibition",
  "template": "Simple_Inhibition",
  "required_outputs": ["simulation.csv", "simulation.png", "BIO_CHECK"],
  "exemplars": [],
  "edges": [
    {{"source": "Drug A", "target": "Target X", "interaction": "inhibition", "mechanism": "binding", "reaction_equation": "Drug A + Target X → Drug A-Target X"}}
  ]
}}

【Bad Example】
- simulation_type = "complex"（不在枚举内）
- required_outputs 缺 BIO_CHECK
- edges 中 source/target 使用 entity_id（如 e1, e2）而非实体 name
- edges 缺 interaction 或 mechanism 字段
- 自由发挥长段描述
- EGF→MAPK 直接连接（跳过 EGFR/pEGFR/Shc/Grb2/SOS/Ras/Raf/MEK）
- 只写 EGF→EGFR→Shc→Grb2→SOS→Ras→MAPK 而不包含 pEGFR/pShc/pMEK/pMAPK（磷酸化中间体必须显式建模）
- 对 EGF-EGFR 级联选 Cascade_Activation（错误：该模板缺质量守恒）

【Good Example】
- simulation_type 严格枚举：simple_inhibition / simple_activation / cascade_inhibition / cascade_activation / signaling_cascade_phos / pkpd_one_compartment / pkpd_two_compartment / dose_sweep / combination
- template 与 simulation_type 保持一致，且必须在 Template 白名单内
- 必填字段：pathway / cell / simulation_type / template / required_outputs / edges
- required_outputs 至少包含 ["simulation.csv", "simulation.png", "BIO_CHECK"]
- edges 中 source/target 必须使用实体的 name（不是 entity_id）
- interaction 严格枚举：activation / inhibition
- mechanism 严格枚举：binding / phosphorylation / dephosphorylation / recruitment / exchange / dissociation / degradation / activation / inhibition
- 每条 edge 必须包含 reaction_equation 字段
- 信号级联必须显式列出每一步磷酸化：Xxx→pXxx，禁止跳过
- pathway completeness: EGF→EGFR→pEGFR→Shc/pShc→Grb2→SOS→RasGDP/RasGTP→Raf/pRaf→MEK/pMEK→MAPK/pMAPK

【Negative Constraints】
- 禁止编造 PMID（必须为空或为用户已引用的）
- 禁止将 multi-pathway 简化成单一 simulation_type
- 禁止在 edges 中使用 entity_id（如 e1, e2），必须用实体 name
- pathway/cell/edges 必须严格基于用户输入实体，禁止添加示例中的占位实体
- 禁止 shortcut 边（如 EGF→MAPK, EGF→Ras, Ras→MAPK）：每一步都必须显式列出
- 禁止遗漏磷酸化步骤：如果通路含"磷酸化"或"激活"，必须有 Xxx→pXxx 边
- 禁止对 EGF-EGFR 级联选 Cascade_Activation 模板（缺质量守恒）
- 禁止输出白名单外的 template 值
- 当用户输入提到 drug 与 target 的抑制关系时，network_json 必须包含一条从 drug 节点到 target 节点的 inhibition 边
- 禁止包含用户输入或 MCP 上下文未提及的蛋白或通路
"""


# =============================================================================
# N3 — Mechanism RAG 提示词（用于在 RAG 命中后让 LLM 总结）
# =============================================================================
N3_MECHANISM_RAG_PROMPT = """你是分子生物学家。基于 RAG 检索到的通路知识，为给定机制生成一段精炼的机制描述。

【任务】
- 输入：用户假说（mechanism.scenario）+ RAG 检索命中的通路片段（retrieved_chunks）
- 输出：mechanism.description（中文，1-3 句话）

【Few-shot】
用户假说："配体 A 抑制细胞 B 活性"
检索命中："配体 A 通过介导蛋白 C 磷酸化下调效应基因表达，细胞 B 功能受损（PMID:12345）"
输出：description = "配体 A 经介导蛋白 C 通路下调效应基因，抑制细胞 B 功能。"

【Negative Constraints】
- 禁止引用未在 RAG 命中里出现的 PMID
- 禁止超过 3 句话
- 禁止使用"某疾病""T1""T2""炎症因子"等模糊占位词
- description 必须严格基于用户假说与 RAG 命中内容，禁止添加示例中的占位实体
"""


# =============================================================================
# N5 — Parameter RAG 决策（按边查询参数）
# =============================================================================
N5_PARAMETER_DECISION_PROMPT = """你是动力学参数选择专家。基于 RAG 检索结果，为指定边（source→target，interaction）选择最佳参数。

【任务】
- 输入：edge_key（如 "Source->Target"）、interaction、retrieved_params 列表
- 输出：JSON {{edge_key, param_name, value, unit, source, confidence, is_fallback}}

【Few-shot】
输入：edge_key="Source_A->Target_B"，interaction="inhibition"，retrieved_params=[{{param_name:"kd", value:5.0, unit:"nM", source:"PMID:111"}}, ...]
输出：
{{
  "edge_key": "Source_A->Target_B",
  "param_name": "kd",
  "value": 5.0,
  "unit": "nM",
  "source": "PMID:111",
  "confidence": "HIGH",
  "is_fallback": false
}}

【Negative Constraints】
- 若 RAG 无命中，必须 is_fallback=true 并给出合理估算（不要假装查到）
- 禁止修改 value/unit（程序注入专用）
- 禁止编造 PMID
- edge_key 必须来自实际输入，禁止使用示例中的占位边名
"""


# =============================================================================
# N6 — ODE Generator（LLM 仅输出 Network JSON，模板渲染由 Python 完成）
# =============================================================================
N6_ODE_PROMPT = """你是生物网络关系解析器。基于 KG 与参数，输出 ODE 变量间的定性调控关系 JSON。禁止生成 Python 代码。

【任务】
- 输入：knowledge_graph、parameters、template
- 输出：network_relations = {{
    "variables": [{{"name": "<节点name>", "role": "inhibitor|activator|target"}}, ...],
    "equations": [{{"lhs": "d<Target>/dt", "rhs_pattern": "production * (1 - inhibitor/Kd) - degradation * target", "type": "inhibition"}}]
  }}

【Bad Example】
- 写出 ```python def ode(): ...```
- 写数值参数

【Good Example】
- equations 用 rhs_pattern 描述调控形式（String 模板），由 Python 渲染
- type 严格枚举：activation / inhibition / degradation / production / pkpd_absorption / pkpd_elimination / combination
- variables.name 必须来自 knowledge_graph 中的实际节点名

【Negative Constraints】
- 禁止写任何 Python / NumPy / SciPy 代码
- 禁止给具体数值（数值来自 parameters）
- variables.name 必须来自 knowledge_graph，禁止使用示例占位名
- TODO: P2-9 — 补充三项硬约束（防幻觉蛋白/耦合形式/IC50 量级）
- 禁止引入非用户输入或 KG/MCP 上下文之外的蛋白/通路/细胞类型（防幻觉）
- inhibition 类方程的 rhs_pattern 必须含 "(1 - inhibition)" 因子，禁止写成 "production * inhibition"（反向生存分数）
- 初始药物浓度（dose）必须为 EC50 的 5-20 倍量级，禁止使用任意固定值（如 10.0）
"""


# =============================================================================
# N11 — Scientific Report（LLM 仅输出 JSON 字段）
# =============================================================================
N11_REPORT_FILL_PROMPT = """你是生物医学报告撰写专家。基于仿真结果、指标、实验方案与文献证据，输出报告的 JSON 字段（不要写 Markdown）。

【强制时间单位约束】
All time values in metrics are in {{time_unit}}. 你必须在 simulation_interpretation 中显式使用 {{time_unit}} 作为时间单位，禁止使用"小时"除非 time_unit=="h"。

【任务】
- 输入：metrics、experiment_protocols、paper_evidence、knowledge_graph、confidence、time_unit
- 输出（严格 JSON）：
{{
  "mechanism_analysis": "中文 1-2 句，描述机制",
  "simulation_interpretation": "中文 2-3 句，结合指标数据解释仿真意义，时间单位必须使用 {{time_unit}}",
  "discussion": "中文 2-3 句，讨论潜在意义与不确定性",
  "limitations": "中文 1-2 句，说明局限性"
}}

【Few-shot】（假设 time_unit=min）
metrics.species.Target_X = {{peak: 18.5, peak_time: 12.0, fold_change: 0.62, activation_duration: 8.5}}
experiment_protocols = [{{name: "Western blot", target: "Target_X"}}]
paper_evidence = [{{pmid: "111", title: "Target_X 在通路中的作用"}}]

输出：
{{
  "mechanism_analysis": "（基于 metrics 与 evidence 描述实际机制，1-2 句）",
  "simulation_interpretation": "仿真显示 Target_X 浓度在 12 min 达到 18.5 nM 峰值，激活持续 8.5 min，提示信号转导快速且瞬态。Western blot 实验可在 12 min 节点采样验证。",
  "discussion": "结合 PMID:111 的证据，本机制与文献报道一致；剂量依赖性可通过 dose_sweep 进一步评估。",
  "limitations": "当前模型仅含单抑制边，未考虑联合用药协同。"
}}

【Negative Constraints】
- 禁止输出 Markdown 语法（# / ** / 表格）
- 禁止出现"某疾病""T1""T2""炎症因子""等等"
- 禁止引用未在 experiment_protocols / paper_evidence 中出现的 PMID
- 所有结论必须可追溯到 metrics / experiment_protocols / paper_evidence 之一
- mechanism_analysis 必须基于实际输入数据，禁止添加示例中的占位实体名
- 禁止使用"小时"作为时间单位，除非 time_unit=="h"
- 禁止使用 half-life / steady-state 描述瞬态信号蛋白（pEGFR/pERK 等）
- 对瞬态级联蛋白必须使用 peak time / activation duration / max level 描述
"""


# =============================================================================
# 共享的 forbidden terms（用于 ReportRenderer 校验）
# =============================================================================
REPORT_FORBIDDEN_TERMS: list[str] = [
    "某疾病",
    "T1", "T2",
    "炎症因子",
    "等等",
    "TGF-betta",  # 已知拼写错误
]
