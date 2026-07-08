# BioDynamics v4 — Final Verification Report

> **审计身份**：Principal Investigator (Systems Biology) + AI for Science Reviewer + Scientific Software QA Lead + Computational Systems Biology Architect + Senior Code Reviewer
> **审计方法**：代码级静态审查 + 架构一致性比对 + 科学语义验证 + 失败模式推演 + Verification Suite 设计
> **Ground Truth**：`docs/archive/BioDynamics_v4_P1_P2_P3_Integration_Report.md` + `docs/archive/BioDynamics_v4_P4_P5_P6_Integration_Report.md` + `BioDynamics_v4_Release_Report.md` + 全代码仓库
> **审计日期**：2026-07-08
> **审计约束**：只读审计，不修改代码；Verification Suite 仅新增测试骨架

---

## 目录

1. [Architecture Audit](#1-architecture-audit)
2. [Scientific Audit — 10 Pathway Coverage](#2-scientific-audit)
3. [Simulation Audit](#3-simulation-audit)
4. [Validation Audit](#4-validation-audit)
5. [Frontend Audit](#5-frontend-audit)
6. [Verification Suite Design](#6-verification-suite-design)
7. [Benchmark Coverage](#7-benchmark-coverage)
8. [Performance Report](#8-performance-report)
9. [Threat Model — Top 100 Failure Modes](#9-threat-model)
10. [Research Gap Analysis](#10-research-gap)
11. [Top 100 Technical Debt](#11-top-100-technical-debt)
12. [Top 100 Future TODO](#12-top-100-future-todo)
13. [Final Readiness Assessment](#13-final-readiness-assessment)

---

## 1. Architecture Audit

### 1.1 架构评分

| 维度 | 评分 | 扣分依据 | 证据 |
|------|------|---------|------|
| Ontology Layer | **55/100** | SBO 反向映射丢失 3 个机制（-20）；EGF 双重身份导致 ChEBI 查询失败（-10）；GO 客户端 `geneProductSymbol` 参数无效（-10）；species_type 降级（-5） | [sbo_terms.py:79](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/sbo_terms.py) ; [ontology_agent.py:33-61](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/ontology_agent.py) ; [go_client.py:63](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/go_client.py) |
| Pathway Graph | **75/100** | Hook 读取 `state["entities"]` 时 N1 NER 尚未执行（-15）；normalize_v4_state 原地修改（-5）；pathway_tag 跨通路混用（-5） | [graph_v3.py:1497](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/graph_v3.py) ; [state.py:481](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/state.py) |
| Reaction IR v2 | **35/100** | INHIBITION 反应物=产物（-20）；else: 分支将 8 种机制错误归类（-25）；Modifier schema 无 Ki/Kact/n_hill（-10）；Provenance 永远 None（-5）；stoichiometry 仅 int（-5） | [reaction_builder.py:166-193](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/reaction_builder.py) ; [schema.py:83-96](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/schema.py) |
| ODE Template | **15/100** | `_extract_species_names` 读取不存在的 `name` 字段（-30）；`_extract_params` 读取不存在的 `parameters`（-25）；`_extract_edges` 无法提取 source/target（-20）；磷酸化 MM 公式错误（-5）；binding 一级反应（-5） | [ode_renderer_v2.py:244-330](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_renderer_v2.py) |
| Solver | **20/100** | DDE 求解器完全不可用（-40）；无事件检测（-15）；max_step 硬编码（-10）；bistability 无滞后验证（-10）；oscillation 无 FFT（-5） | [dde_solver.py:81-96](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/solvers/dde_solver.py) |
| Validation Pyramid | **25/100** | L2 MOCK 仿真（-30）；L4 仅 5/10 通路（-20）；L1 稳态检查不充分（-10）；L3 计数非通量（-10）；L5 从不阻断（-5） | [level2_sbml.py:601](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/validation_v2/level2_sbml.py) ; [level4_benchmark.py:180](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/validation_v2/level4_benchmark.py) |
| Hypothesis Layer | **40/100** | Hook 在 worker_report 之前执行 metrics 为空（-20）；experimental_data orphan（-15）；无 refinement 循环（-15）；无迭代验证（-10） | [hypothesis_agent.py:119](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/hypothesis/hypothesis_agent.py) |
| Pathway Specialists | **55/100** | 全部 10 条通路无动力学参数（-25）；p53 substrate=product（-5）；Apoptosis 未定义 procaspase（-5）；Cell Cycle PMID 错误（-5）；Wnt/TGF-β 引用不存在模板（-5） | [pathways/specialists/](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathways/specialists/) |
| Frontend Workflow | **60/100** | 无 provenance 追踪（-10）；无 SBML Compare（-10）；无参数 provenance 链（-10）；无实验设计导出（-5）；无 PDF 导出（-5） | [store.ts](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/lib/store.ts) |

**Architecture 总分：42/100**

### 1.2 Architecture Drift

| ID | Drift | 证据 | 严重度 |
|----|-------|------|--------|
| AD-1 | ODE Renderer 读取的 `name`/`parameters`/`source`/`target` 字段在 Reaction IR v2 schema 中不存在 | [ode_renderer_v2.py:244-330](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_renderer_v2.py) vs [schema.py:37-55](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/schema.py) | P0 |
| AD-2 | Calibration Hook 在 Validation Hook 之后执行，违反科学方法论 | [graph_v3.py:1540-1547](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/graph_v3.py) | P1 |
| AD-3 | Calibration 输出不回写仿真参数 | [calibration_agent.py:207](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/calibration/calibration_agent.py) | P1 |
| AD-4 | DDE 模板声称支持延迟但始终降级为 ODE | [dde_solver.py:81-96](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/solvers/dde_solver.py) | P0 |
| AD-5 | `experimental_data` 被 calibration/hypothesis 读取但未在 state 中声明 | [calibration_agent.py:280](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/calibration/calibration_agent.py) | P1 |

### 1.3 Hidden Coupling

| ID | 耦合 | 证据 | 严重度 |
|----|------|------|--------|
| HC-1 | validation_pyramid_hook_node 返回 `pending_clarification`（v3 路由字段），静默劫持路由 | [validation_agent.py:333-340](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/validation_v2/validation_agent.py) | P1 |
| HC-2 | `_reaction_ir_v2_hook` 条件覆盖 v3 `network_json` | [graph_v3.py:924-933](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/graph_v3.py) | P2 |
| HC-3 | `_ontology_agent` 单例 `self.warnings` 非线程安全 | [ontology_agent.py:358](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/ontology_agent.py) | P1 |
| HC-4 | `_pubmed_cache` 无 TTL、无驱逐、跨请求共享 | [nodes_v2.py:486](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/nodes_v2.py) | P1 |
| HC-5 | `_clarification_events` 进程全局 dict，多 worker 失效 | [graph_v3.py:86-88](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/graph_v3.py) | P1 |
| HC-6 | V4_CALIBRATION_AGENT_ENABLED 与 V4_ODE_TEMPLATE_V2_ENABLED 隐式跨 flag 耦合 | [calibration_agent.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/calibration/calibration_agent.py) | P2 |

### 1.4 Scientific Anti-patterns

| ID | Anti-pattern | 证据 | 严重度 |
|----|-------------|------|--------|
| SA-1 | 磷酸化 MM 公式 `k_cat * src^2 / (Km + src)` 在全部 9 个模板中错误 | [oscillatory_feedback.j2:105-114](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates_v2/oscillatory_feedback.j2) | P0 |
| SA-2 | Binding 渲染为一级反应 `k_bind * src` 而非双分子 `k_on * [A] * [B]` | [destruction_complex.j2:79-84](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates_v2/destruction_complex.j2) | P0 |
| SA-3 | Dimerization 化学计量 1→1 而非 2→1 | [reaction_builder.py:190-193](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/reaction_builder.py) | P0 |
| SA-4 | Nuclear transport 无 compartment 体积缩放 | [nuclear_transport.j2:76-88](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates_v2/nuclear_transport.j2) | P1 |
| SA-5 | "DNA" 作为 transcription 的 substrate | [nf_kappa_b_specialist.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathways/specialists/nf_kappa_b_specialist.py) | P2 |
| SA-6 | Calibration 占位 `_default_model` 返回固定列表 | [least_squares_fitter.py:261-278](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/calibration/least_squares_fitter.py) | P0 |
| SA-7 | Hash-based RNG seed 破坏 bootstrap 可重复性 | [confidence_interval.py:219](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/calibration/confidence_interval.py) | P1 |
| SA-8 | L2 验证 `_simulate_v4_ode` 是 MOCK 线性衰减 | [level2_sbml.py:601-656](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/validation_v2/level2_sbml.py) | P0 |
| SA-9 | 全部 10 条 specialist 无动力学参数 | [pathways/specialists/](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathways/specialists/) | P0 |
| SA-10 | Sandbox 生物检查仅扫描 `BIO_CHECK:` 标记但无模板输出 | [sandbox.py:178-203](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/sandbox.py) | P0 |

---

## 2. Scientific Audit — 10 Pathway Coverage

### Coverage Matrix（0–5 分）

| # | 通路 | Ontology | Mechanism | Reaction IR | Sim Ready | Validation | Cross-talk | 总分/30 |
|---|------|----------|-----------|-------------|-----------|------------|------------|---------|
| 1 | EGFR/RTK | 4 | 4 | 4 | 1 | 5 | 4 | **22** |
| 2 | MAPK/ERK | 4 | 4 | 4 | 1 | 5 | 4 | **22** |
| 3 | PI3K-AKT-mTOR | 5 | 4 | 3 | 1 | 4 | 5 | **22** |
| 4 | p53 | 4 | 5 | 2 | 2 | 5 | 4 | **22** |
| 5 | Apoptosis | 4 | 5 | 2 | 1 | 4 | 5 | **21** |
| 6 | Cell Cycle | 5 | 5 | 2 | 2 | 4 | 5 | **23** |
| 7 | JAK-STAT | 4 | 4 | 2 | 2 | 5 | 5 | **22** |
| 8 | NF-κB | 5 | 5 | 3 | 2 | 5 | 5 | **25** |
| 9 | Wnt/β-catenin | 5 | 5 | 3 | 2 | 4 | 5 | **24** |
| 10 | TGF-β/SMAD | 5 | 5 | 3 | 2 | 5 | 4 | **24** |
| | **平均** | **4.5** | **4.6** | **2.8** | **1.6** | **4.7** | **4.6** | **22.7** |

### Missing Mechanisms

| 通路 | 缺失机制 | 证据 |
|------|---------|------|
| EGFR/RTK | EGFR 内吞/泛素化（Cbl 介导）；ERK 核转位 | [egfr_specialist.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathways/specialists/egfr_specialist.py) |
| MAPK/ERK | ERK→SOS 核质分区；RasGAP 介导 GTP→GDP | [mapk_specialist.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathways/specialists/mapk_specialist.py) |
| PI3K-AKT-mTOR | mTORC2→AKT Ser473；TSC1/2 复合物；Rheb GTPase | [pi3k_akt_mtor_specialist.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathways/specialists/pi3k_akt_mtor_specialist.py) |
| p53 | p53 乙酰化（p300/CBP）；MDM4/MDMX 抑制 | [p53_specialist.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathways/specialists/p53_specialist.py) |
| Apoptosis | Procaspase 物种未定义；线粒体 PTP | [apoptosis_specialist.py:225,257,303,318](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathways/specialists/apoptosis_specialist.py) |
| Cell Cycle | APC/C-Cdh1（G1 期）；DNA damage checkpoint | [cell_cycle_specialist.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathways/specialists/cell_cycle_specialist.py) |
| JAK-STAT | STAT3 物种定义但无 reaction 产生（orphaned） | [jak_stat_specialist.py:122](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathways/specialists/jak_stat_specialist.py) |
| NF-κB | A20 蛋白酶抑制；non-canonical（p100→p52） | [nf_kappa_b_specialist.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathways/specialists/nf_kappa_b_specialist.py) |
| Wnt | LRP6 内吞；Wnt/Ca2+ 非经典通路 | [wnt_specialist.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathways/specialists/wnt_specialist.py) |
| TGF-β | Smad2 vs Smad3 差异；Smad linker 磷酸化 | [tgf_beta_specialist.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathways/specialists/tgf_beta_specialist.py) |

---

## 3. Simulation Audit

### Scientific Bug 清单

| ID | Bug 类型 | 描述 | 证据 | 严重度 |
|----|---------|------|------|--------|
| SB-1 | 零通量 | ODE Renderer 读取不存在的 `name`/`parameters`/`source`/`target`，仿真产出零通量 ODE | [ode_renderer_v2.py:244-330](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_renderer_v2.py) | P0 |
| SB-2 | 负浓度无检测 | Sandbox 生物检查仅扫描 `BIO_CHECK:` 标记但无模板输出 | [sandbox.py:178-203](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/sandbox.py) | P0 |
| SB-3 | 质量守恒无后验证 | 无代码在仿真后加载 CSV 验证守恒约束 | [sandbox.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/sandbox.py) | P0 |
| SB-4 | 无限增长风险 | 转录反应 TF 净通量为零，转录产物可能无限增长 | [reaction_builder.py:190-193](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/reaction_builder.py) | P1 |
| SB-5 | DDE 延迟失效 | `y(t-τ) ≈ y(t)` 消除全部延迟效应 | [dde_solver.py:81-96](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/solvers/dde_solver.py) | P0 |
| SB-6 | MM 公式错误 | `v = k_cat * src^2 / (Km + src)` 在全部 9 个模板中 | [oscillatory_feedback.j2:105-114](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates_v2/oscillatory_feedback.j2) | P0 |
| SB-7 | 单位不一致 | nM 与 molecule_per_cell 混用无转换因子 | [schema.py:50-51](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/schema.py) | P1 |
| SB-8 | Compartment 无缩放 | Nuclear transport 无 V_cyto/V_nuc 体积比 | [nuclear_transport.j2:76-88](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates_v2/nuclear_transport.j2) | P1 |
| SB-9 | Binding 一级反应 | `v = k_bind * src` 应为 `k_on * [A] * [B]` | [destruction_complex.j2:79-84](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates_v2/destruction_complex.j2) | P0 |
| SB-10 | Dimerization 化学计量错误 | 1→1 而非 2→1 | [reaction_builder.py:190](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/reaction_builder.py) | P0 |
| SB-11 | Cleavage 净零通量 | `dy[t_idx] -= _rate` 后 `dy[t_idx] += _rate` | [caspase_cascade.j2:74-88](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates_v2/caspase_cascade.j2) | P1 |
| SB-12 | Hill 函数负值崩溃 | `src ** n_hill` 当 src < 0 且 n_hill 非整数 | [oscillatory_feedback.j2:122-124](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates_v2/oscillatory_feedback.j2) | P1 |
| SB-13 | Solver 无刚度自适应 | `max_step=0.5` 硬编码 | [dde_solver.py:50](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/solvers/dde_solver.py) | P1 |
| SB-14 | Ubiquitination 符号错误 | `dy[t_idx] -= _rate` 应为 `+=` | [oscillatory_feedback.j2:136-141](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates_v2/oscillatory_feedback.j2) | P1 |
| SB-15 | DDE 双重计数 | `_dde_rhs` 返回 `dy_ode + dy` 转录边计算两次 | [oscillatory_feedback.j2:230](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates_v2/oscillatory_feedback.j2) | P1 |

---

## 4. Validation Audit

### Validation Pyramid 5 级审计

| Level | 检查内容 | False Positive | False Negative | Blind Spot | 严重度 |
|-------|---------|---------------|---------------|------------|--------|
| **L1 Internal** | 质量守恒 + 非负 + 稳态 + 数值稳定 + 约束 | **HIGH**：稳态检查仅验证 `-k*X` 存在，有降解无合成的模型也 pass | **MEDIUM**：质量守恒仅验证初始浓度 | 无实际 ODE 积分；无参数范围验证 | P1 |
| **L2 SBML** | Track A (roadrunner) / Track B (结构相似度) / skipped | **CRITICAL**：`_simulate_v4_ode` 是 MOCK 线性衰减；Track B 结构相似度 ≥ 0.6 可在动态错误时通过 | **MEDIUM**：ontology_ref 缺失时 species 对齐返回空 → pass=False | 仅检查 peak/peak_time，忽略振荡频率、稳态、剂量响应 | P0 |
| **L3 Crosstalk** | Cross-talk 一致性 + 共享物种守恒 + 时间尺度对齐 | **HIGH**：共享物种守恒基于计数非通量；cross-talk 仅检查物种名存在 | **LOW**：时间尺度仅检查 `step > 0` | 无多通路联合仿真 | P1 |
| **L4 Benchmark** | 5 通路 benchmark 对比 | **CRITICAL**：仅 5/10 通路有 benchmark，其余 `pass=True, method="no_benchmark_matched"` | **MEDIUM**：PMID 不一致（NF-κB, p53） | 单一 PMID 来源；tolerance 过宽 | P0 |
| **L5 Hypothesis** | 假说验证/证伪计数 | **HIGH**：从不阻断（pass=True 即使全部证伪） | N/A | P6 默认关闭 → auto-skipped；experimental_data orphan | P1 |

---

## 5. Frontend Audit

### UX Anti-patterns

| ID | Anti-pattern | 证据 | 严重度 |
|----|-------------|------|--------|
| UX-1 | 无 Provenance 追踪 | [store.ts](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/lib/store.ts) | P1 |
| UX-2 | 无 SBML Compare 视图 | [ValidationPyramid.tsx](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/components/validation/ValidationPyramid.tsx) | P1 |
| UX-3 | Parameter Explorer 无单位标注 | [ParameterExplorer.tsx](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/components/parameter/ParameterExplorer.tsx) | P1 |
| UX-4 | Hypothesis Panel 无实验设计导出 | [HypothesisPanel.tsx](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/components/hypothesis/HypothesisPanel.tsx) | P2 |
| UX-5 | Simulation Panel 无时序标记 | [SimulationPanel.tsx](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/components/simulation/SimulationPanel.tsx) | P2 |
| UX-6 | 无 Evidence Navigation | [AIAssistantPanel.tsx](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/components/ai_assistant/AIAssistantPanel.tsx) | P1 |
| UX-7 | 无 Report 导出 | [WorkbenchShell.tsx](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/components/workspace/WorkbenchShell.tsx) | P1 |
| UX-8 | Validation Pyramid 无 drill-down | [ValidationPyramid.tsx](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/components/validation/ValidationPyramid.tsx) | P1 |
| UX-9 | Pathway Graph 无 SBML overlay | [PathwayGraph.tsx](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend/components/pathway/PathwayGraph.tsx) | P2 |
| UX-10 | Benchmark Center 无统计聚合 | 无对应组件 | P2 |

---

## 6. Verification Suite Design

### 目录结构

```
verification/
├── biomodels_regression/      # Part 8: 30 BioModels 回归测试
│   ├── biomodels_config.yaml  # 30 条 BioModels 配置
│   └── test_biomodels_regression.py  # pytest 参数化测试
├── pathway_regression/        # Part 9: 50 通路回归测试
│   ├── pathway_config.yaml    # 50 条通路配置（5×10）
│   └── test_pathway_regression.py  # pytest 参数化测试
├── hypothesis_validation/     # Part 10: 100 假说验证
│   ├── hypotheses_dataset.json  # 100 条假说数据集
│   └── test_hypothesis_validation.py  # pytest + 成功率统计
├── parameter_stress/          # Part 11: 参数压力测试
│   └── test_parameter_stress.py  # ±10%/30%/50% 扰动 × 10 通路
├── solver_validation/         # Part 12: Solver 验证
│   └── test_solver_validation.py  # 解析解 + 刚度 + 振荡 + 双稳态
├── ontology_validation/       # Part 13: 本体验证
│   └── test_ontology_validation.py  # HGNC/UniProt/GO/ChEBI + SBO
├── ui_workflow/               # Part 14: 前端工作流测试
│   ├── playwright.config.ts   # Playwright 配置
│   └── tests/
│       └── scientific_workflow.spec.ts  # 完整科研流程 E2E
├── benchmark/                 # Part 15: 性能基准
│   └── test_performance.py    # 速度/内存/延迟/启动时间
├── reports/                   # 报告输出目录
└── dashboard/                 # 性能仪表盘
    └── generate_dashboard.py  # HTML 仪表盘生成器
```

### 测试统计

| 模块 | 测试用例数 | 已实现 | Skip（已知限制） | 文件 |
|------|-----------|--------|-----------------|------|
| BioModels Regression | 30 | 0 | 30 | test_biomodels_regression.py |
| Pathway Regression | 50 | 0 | 50 | test_pathway_regression.py |
| Hypothesis Validation | 102 | 0 | 102 | test_hypothesis_validation.py |
| Parameter Stress | 32 | 0 | 32 | test_parameter_stress.py |
| Solver Validation | 19 | 13 | 6 | test_solver_validation.py |
| Ontology Validation | 18 | 1 | 17 | test_ontology_validation.py |
| Frontend Workflow | 5 | 5 | 0 | scientific_workflow.spec.ts |
| Performance Benchmark | 17 | 0 | 17 | test_performance.py |
| **总计** | **273** | **19** | **254** | |

**已实现的 19 个测试**：Solver Validation 的 13 个解析解测试（线性衰减/刚度/振荡/双稳态）+ Ontology Validation 的 1 个 species_type 覆盖测试 + Frontend Workflow 的 5 个 Playwright E2E 测试。

### 自动执行方式

```bash
# 全部验证套件
cd verification && python -m pytest . -v --tb=short

# 分模块执行
python -m pytest biomodels_regression/ -v
python -m pytest pathway_regression/ -v
python -m pytest hypothesis_validation/ -v
python -m pytest parameter_stress/ -v
python -m pytest solver_validation/ -v
python -m pytest ontology_validation/ -v

# 前端 E2E
cd ui_workflow && npx playwright test

# 性能仪表盘
cd dashboard && python generate_dashboard.py
```

---

## 7. Benchmark Coverage

### BioModels Regression Coverage（Part 8）

| 指标 | 值 |
|------|-----|
| 配置总数 | 30 |
| 覆盖通路 | 10/10 |
| 比较指标 | RMSE, Pearson, Peak Time, Peak Amplitude, Steady State, AUC |
| RMSE 阈值 | 15.0（默认） |
| Pearson 阈值 | 0.85（默认） |
| 当前可执行 | 0/30（全部 skip — 依赖 ODE Renderer 修复） |

### Pathway Regression Coverage（Part 9）

| 指标 | 值 |
|------|-----|
| 配置总数 | 50 |
| 每通路用例 | 5 |
| 覆盖机制 | binding, phosphorylation, dephosphorylation, ubiquitination, degradation, transcription, translation, nuclear_import, complex_formation, cleavage, gtp_gdp_exchange, dimerization, proteasomal_degradation |
| 当前可执行 | 0/50（全部 skip — 依赖 ODE Renderer 修复） |

### Hypothesis Validation Coverage（Part 10）

| 指标 | 值 |
|------|-----|
| 假说总数 | 100 |
| 每通路假说 | 10 |
| 预期 pass | 92 |
| 预期 fail | 8（边界测试用例） |
| 成功率目标 | > 60% |
| 当前可执行 | 0/100（全部 skip — 依赖 LLM 集成） |

---

## 8. Performance Report

### 性能指标（设计规格）

| 指标 | 阈值 | 当前状态 | 测试文件 |
|------|------|---------|---------|
| Backend Startup Time | < 5s | 未测量 | test_performance.py |
| Single Pathway Simulation | < 30s | 未测量 | test_performance.py |
| RAG Retrieval Latency | < 2s | 未测量 | test_performance.py |
| Pathway Recognition Time | < 5s | 未测量 | test_performance.py |
| SBML Parsing Time | < 3s | 未测量 | test_performance.py |
| Peak Memory | < 2 GB | 未测量 | test_performance.py |
| Per-Pathway Simulation | < 60s | 未测量 | test_performance.py |

### 已知性能风险

| 风险 | 证据 | 影响 |
|------|------|------|
| `_pubmed_cache` 无界增长 | [nodes_v2.py:486](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/nodes_v2.py) | 长时间运行内存泄漏 |
| `simulation_csv_path` 目录泄漏 | [sandbox.py:519](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/sandbox.py) | 磁盘空间耗尽 |
| LLM 单次调用无并发 | [config.py:1239](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/config.py) | 假说生成串行阻塞 |

---

## 9. Threat Model — Top 100 Failure Modes

| ID | 类别 | 问题描述 | 触发条件 | 严重度 | 所在文件/模块 |
|----|------|---------|---------|--------|-------------|
| FM-001 | Scientific | ODE Renderer 读取不存在的 `name` 字段 | 任何 v4 仿真 | P0 | ode_renderer_v2.py:244 |
| FM-002 | Scientific | ODE Renderer 读取不存在的 `parameters` 字段 | 任何 v4 仿真 | P0 | ode_renderer_v2.py:271 |
| FM-003 | Scientific | ODE Renderer 读取不存在的 `source`/`target` | 任何 v4 仿真 | P0 | ode_renderer_v2.py:308 |
| FM-004 | Scientific | 磷酸化 MM 公式错误 | 含磷酸化的仿真 | P0 | oscillatory_feedback.j2:105 |
| FM-005 | Scientific | DDE 延迟完全失效 | p53/NF-κB 仿真 | P0 | dde_solver.py:81 |
| FM-006 | Scientific | Binding 渲染为一级反应 | 含 binding 的仿真 | P0 | destruction_complex.j2:79 |
| FM-007 | Scientific | Dimerization 化学计量错误 | 含 dimerization 的仿真 | P0 | reaction_builder.py:190 |
| FM-008 | Scientific | INHIBITION 反应物=产物 | 含 inhibition 的仿真 | P0 | reaction_builder.py:166 |
| FM-009 | Scientific | else: 分支将 8 种机制错误归类 | 多种机制 | P0 | reaction_builder.py:190 |
| FM-010 | Scientific | L2 验证使用 MOCK 仿真 | 任何 L2 验证 | P0 | level2_sbml.py:601 |
| FM-011 | Scientific | L4 benchmark 仅覆盖 5/10 通路 | 5 条通路 | P0 | level4_benchmark.py:180 |
| FM-012 | Scientific | Sandbox 生物检查不触发 | 任何 v4 仿真 | P0 | sandbox.py:178 |
| FM-013 | Scientific | 全部 specialist 无动力学参数 | 任何 specialist 仿真 | P0 | pathways/specialists/ |
| FM-014 | Scientific | Calibration 占位 model_func | 任何 calibration | P0 | least_squares_fitter.py:261 |
| FM-015 | Scientific | Calibration 输出不回写参数 | 任何 calibration | P0 | calibration_agent.py:207 |
| FM-016 | Scientific | SBO 反向映射丢失 3 个机制 | SBML grounding | P1 | sbo_terms.py:79 |
| FM-017 | Scientific | EGF 双重身份导致 ChEBI 失败 | EGFR ontology | P1 | ontology_agent.py:33 |
| FM-018 | Scientific | GO 客户端参数无效 | 任何 GO 查询 | P1 | go_client.py:63 |
| FM-019 | Scientific | GO evidence 字段映射错误 | 任何 GO 查询 | P2 | go_client.py:85 |
| FM-020 | Scientific | Cleavage 净零通量 | Apoptosis 仿真 | P1 | caspase_cascade.j2:74 |
| FM-021 | Scientific | Ubiquitination 符号错误 | NF-κB/Wnt 仿真 | P1 | oscillatory_feedback.j2:136 |
| FM-022 | Scientific | Hill 函数负值崩溃 | stiff 仿真 | P1 | oscillatory_feedback.j2:122 |
| FM-023 | Scientific | Nuclear transport 无体积缩放 | 任何 nuclear transport | P1 | nuclear_transport.j2:76 |
| FM-024 | Scientific | 单位不一致 | 混合单位模型 | P1 | schema.py:50 |
| FM-025 | Scientific | Constraint 表达式解析错误 | 含复合物名约束 | P1 | constraints.py:53 |
| FM-026 | Scientific | auto_generate_mass_conservation 仅磷酸化 | binding/dimerization | P1 | constraints.py:74 |
| FM-027 | Scientific | is_enzymatic_mechanism 遗漏 ubiquitination | ubiquitination 验证 | P1 | mechanism_types.py:179 |
| FM-028 | Scientific | p53 substrate=product | p53 仿真 | P1 | p53_specialist.py:153 |
| FM-029 | Scientific | Apoptosis 未定义 procaspase | Apoptosis 仿真 | P1 | apoptosis_specialist.py:225 |
| FM-030 | Scientific | Cell Cycle PMID 错误 | Cell Cycle benchmark | P1 | cell_cycle_specialist.py:70 |
| FM-031 | Scientific | L1 稳态检查不充分 | 任何 L1 验证 | P1 | level1_internal.py:270 |
| FM-032 | Scientific | L3 共享物种守恒基于计数 | 多通路仿真 | P1 | level3_crosstalk.py:302 |
| FM-033 | Scientific | L5 从不阻断 | 任何 L5 验证 | P1 | level5_hypothesis.py |
| FM-034 | Scientific | EGFR KEGG ID 错误 | EGFR ontology | P2 | pathway_registry.py:64 |
| FM-035 | Scientific | Hash-based RNG 不可重复 | calibration CI | P1 | confidence_interval.py:219 |
| FM-036 | Scientific | Sensitivity 占位 model_func | sensitivity 分析 | P0 | sensitivity_analyzer.py:293 |
| FM-037 | Scientific | Wnt 引用不存在的模板 | Wnt specialist | P1 | wnt_specialist.py:518 |
| FM-038 | Scientific | TGF-β 引用不存在的模板 | TGF-β specialist | P1 | tgf_beta_specialist.py:393 |
| FM-039 | Scientific | DDE 双重计数 | 若 DDE 启用 | P1 | oscillatory_feedback.j2:230 |
| FM-040 | Scientific | Compartment 仅 5 种 | 分泌通路 | P2 | schema.py:49 |
| FM-041 | Scientific | stoichiometry 仅 int | lumped model | P2 | schema.py:71 |
| FM-042 | Scientific | Modifier 无 allosteric 参数 | 别构调节 | P0 | schema.py:83 |
| FM-043 | Scientific | transport 统一映射到 nuclear_import | cytoplasm_translocation | P1 | mechanism_types.py:130 |
| FM-044 | Scientific | first_order 折叠为 mass_action | kinetics 验证 | P1 | mechanism_types.py:91 |
| FM-045 | Scientific | DDE delay 不从 reaction_ir 提取 | 纯 IR 输入 | P1 | ode_renderer_v2.py:157 |
| FM-046 | Scientific | Solver 无事件检测 | bistability 检测 | P1 | dde_solver.py:116 |
| FM-047 | Scientific | max_step 硬编码 | 快/慢混合系统 | P1 | dde_solver.py:50 |
| FM-048 | Scientific | bistability 无滞后验证 | bistability 判定 | P1 | bistability_detector.py:73 |
| FM-049 | Scientific | oscillation 无 FFT | 振荡周期估计 | P2 | oscillation_detector.py:84 |
| FM-050 | Scientific | Provenance 永远 None | 追溯性 | P1 | reaction_builder.py:214 |
| FM-051 | Numerical | NaN/Inf 在 CSV 中不被检测 | 仿真后 CSV | P1 | sandbox.py:382 |
| FM-052 | Numerical | 负浓度无后验检查 | 任何仿真 | P0 | sandbox.py:178 |
| FM-053 | Numerical | 质量守恒无后验验证 | 任何仿真 | P0 | sandbox.py 全文 |
| FM-054 | Numerical | CSV 目录泄漏 | 多次仿真 | P2 | sandbox.py:519 |
| FM-055 | Numerical | _BLOCKED_BUILTINS 阻止 open() | 用户自定义 ODE | P2 | sandbox.py:62 |
| FM-056 | Numerical | _STOCHASTIC_PATTERNS 误匹配 | noise 变量 | P2 | sandbox.py:90 |
| FM-057 | Numerical | 生物检查在无图像时跳过 | matplotlib 后端 | P1 | sandbox.py:532 |
| FM-058 | Engineering | _clarification_events 进程全局 | 多 worker | P1 | graph_v3.py:86 |
| FM-059 | Engineering | _ontology_agent 单例非线程安全 | 并发请求 | P1 | ontology_agent.py:358 |
| FM-060 | Engineering | _pubmed_cache 无界泄漏 | 长时间运行 | P1 | nodes_v2.py:486 |
| FM-061 | Engineering | experimental_data orphan state | calibration | P1 | state.py |
| FM-062 | Engineering | next_worker 死状态 | supervisor 循环 | P2 | state.py:135 |
| FM-063 | Engineering | need_human_review 遗留死状态 | schema | P2 | state.py:45 |
| FM-064 | Engineering | simulation_ci 已废弃仍写入 | sandbox | P2 | state.py:115 |
| FM-065 | Engineering | validation hook 副作用修改 v3 路由 | 验证失败 | P1 | validation_agent.py:333 |
| FM-066 | Engineering | _reaction_ir_v2_hook 覆盖 v3 network_json | flag 开启 | P2 | graph_v3.py:924 |
| FM-067 | Engineering | Calibration 在 Validation 之后 | v4 验证 | P1 | graph_v3.py:1540 |
| FM-068 | Engineering | ontology_hook 在 worker_mechanism 之前 | v4 运行 | P1 | graph_v3.py:1497 |
| FM-069 | Engineering | hypothesis_hook 在 worker_report 之前 | hypothesis 生成 | P1 | graph_v3.py:1514 |
| FM-070 | Engineering | pre_router 清除 20+ 状态字段 | 每次请求 | P2 | graph_v3.py:251 |
| FM-071 | Engineering | worker_ode 原地修改 state | ODE 渲染 | P2 | graph_v3.py:796 |
| FM-072 | Engineering | normalize_v4_state 原地修改 | worker_ode | P2 | state.py:481 |
| FM-073 | Engineering | Jinja2 无 StrictUndefined | 模板变量缺失 | P2 | ode_renderer_v2.py:109 |
| FM-074 | Engineering | EGFR KEGG ID 错误 | EGFR ontology | P2 | pathway_registry.py:64 |
| FM-075 | Engineering | Reactome ID 范围不一致 | JAK-STAT/Apoptosis | P2 | pathway_registry.py |
| FM-076 | Engineering | pathway_registry 无 BioModels ID | calibration | P1 | pathway_registry.py |
| FM-077 | Engineering | EGFR/MAPK 路由到振荡模板 | EGFR/MAPK 仿真 | P1 | ode_renderer_v2.py:61 |
| FM-078 | Engineering | _select_template 忽略 requires_dde | DDE 模板选择 | P2 | ode_renderer_v2.py:198 |
| FM-079 | UX | 无 Provenance 追踪 | 参数来源 | P1 | store.ts |
| FM-080 | UX | 无 SBML Compare 视图 | L2 验证失败 | P1 | ValidationPyramid.tsx |
| FM-081 | UX | Parameter Explorer 无单位 | 参数量级 | P1 | ParameterExplorer.tsx |
| FM-082 | UX | 无 Report 导出 | 工作流末端 | P1 | WorkbenchShell.tsx |
| FM-083 | UX | Validation Pyramid 无 drill-down | 验证失败 | P1 | ValidationPyramid.tsx |
| FM-084 | UX | 无 Evidence Navigation | 文献关联 | P1 | AIAssistantPanel.tsx |
| FM-085 | UX | Simulation Panel 无时序标记 | 动态特征 | P2 | SimulationPanel.tsx |
| FM-086 | UX | Pathway Graph 无 SBML overlay | 视觉对比 | P2 | PathwayGraph.tsx |
| FM-087 | UX | Hypothesis Panel 无实验导出 | 假说闭环 | P2 | HypothesisPanel.tsx |
| FM-088 | UX | Benchmark Center 无统计聚合 | 整体质量 | P2 | 无 |
| FM-089 | Security | _BLOCKED_BUILTINS 阻止 open 但允许 import | 沙箱逃逸 | P2 | sandbox.py:62 |
| FM-090 | Security | _pubmed_cache 跨请求共享 | 数据隔离 | P1 | nodes_v2.py:486 |
| FM-091 | Security | 无 API rate limiting | API 滥用 | P2 | main.py |
| FM-092 | Security | LLM API key 硬编码单例 | 密钥轮换 | P2 | config.py:1239 |
| FM-093 | Data Integrity | SBO 反向映射数据丢失 | SBML grounding | P1 | sbo_terms.py:79 |
| FM-094 | Data Integrity | Provenance 永远 None | 追溯链 | P1 | reaction_builder.py:214 |
| FM-095 | Data Integrity | Hash-based RNG 不可重复 | CI/CD | P1 | confidence_interval.py:219 |
| FM-096 | Data Integrity | v4_state reducer 非原子 | 并发写入 | P2 | state.py:19 |
| FM-097 | Data Integrity | Apoptosis ABT-199 重复 | 药物计算 | P2 | apoptosis_specialist.py:501 |
| FM-098 | Data Integrity | JAK-STAT Tofacitinib 重复 | 药物计算 | P2 | jak_stat_specialist.py:498 |
| FM-099 | Data Integrity | L4 PMID 与 specialist 不一致 | benchmark | P1 | level4_benchmark.py:66 |
| FM-100 | Data Integrity | Reactome ID 范围不一致 | 参考数据 | P2 | pathway_registry.py |

### 严重度分布

| 严重度 | 数量 | 占比 |
|--------|------|------|
| P0 | 16 | 16% |
| P1 | 36 | 36% |
| P2 | 48 | 48% |
| **总计** | **100** | 100% |

---

## 10. Research Gap Analysis

### 如果投稿 AI for Science，缺失方向

| # | 方向 | 优先级 | 影响 | 建议 |
|---|------|--------|------|------|
| 1 | Stochastic Simulation (SSA/Gillespie) | P0 | 低拷贝数物种需随机仿真 | 集成 gillespy2 |
| 2 | Rule-based Modeling (BioNetGen/Kappa) | P1 | 组合复杂性无法表达 | 集成 BioNetGen |
| 3 | Parameter Inference / Bayesian Calibration | P0 | 无后验分布、无不确定性量化 | 集成 PyMC/emcee |
| 4 | Spatial Modeling / PDE | P1 | 无法建模 morphogen gradient | 集成 FEniCS/py-pde |
| 5 | Multi-cell / Tissue Simulation | P2 | 无法建模肿瘤微环境 | 集成 PhysiCell |
| 6 | COMBINE Archive | P1 | 无法与其他工具互操作 | 集成 libCombine |
| 7 | CellML Support | P2 | CellML 模型无法导入 | 集成 cellmlmanip |
| 8 | SED-ML Support | P1 | 无法复现标准仿真协议 | 集成 tellurium |
| 9 | Digital Cell / Whole-Cell | P2 | 超出当前架构范围 | 需完整代谢网络 |
| 10 | Sensitivity → Experimental Design | P1 | 灵敏度不反馈实验设计 | sensitivity_guided_design |
| 11 | Model Reduction / Lumping | P2 | 大通路仿真慢 | 集成 py-subnets |
| 12 | Bifurcation Analysis | P1 | 无分岔图 | 集成 PyDSTool |
| 13 | Uncertainty Quantification (UQ) | P0 | 无置信区间 | 集成 chaospy |
| 14 | Causal Inference | P2 | 无因果推理 | 集成 causal-learn |
| 15 | Active Learning | P1 | 无主动实验选择 | 集成 modAL |

---

## 11. Top 100 Technical Debt

| ID | 类别 | 问题描述 | 触发条件 | 严重度 | 所在文件/模块 |
|----|------|---------|---------|--------|-------------|
| TD-001 | ODE Renderer | 读取不存在的 `name` 字段 | 任何 v4 仿真 | P0 | ode_renderer_v2.py:244 |
| TD-002 | ODE Renderer | 读取不存在的 `parameters` 字段 | 任何 v4 仿真 | P0 | ode_renderer_v2.py:271 |
| TD-003 | ODE Renderer | 读取不存在的 `source`/`target` | 任何 v4 仿真 | P0 | ode_renderer_v2.py:308 |
| TD-004 | ODE Template | 磷酸化 MM 公式错误（全部 9 模板） | 含磷酸化仿真 | P0 | oscillatory_feedback.j2:105 |
| TD-005 | Solver | DDE 求解器空壳（jitcdde 未接线） | DDE 通路仿真 | P0 | dde_solver.py:81 |
| TD-006 | Calibration | 占位 model_func 无生物学意义 | 任何 calibration | P0 | least_squares_fitter.py:261 |
| TD-007 | Validation L2 | MOCK 线性衰减而非真实仿真 | L2 SBML 验证 | P0 | level2_sbml.py:601 |
| TD-008 | Specialists | 全部 10 specialist 无动力学参数 | 任何 specialist 仿真 | P0 | pathways/specialists/ |
| TD-009 | Ontology | SBO 反向映射丢失 3 个机制 | SBML grounding | P1 | sbo_terms.py:79 |
| TD-010 | Reaction IR | INHIBITION 反应物=产物 | 含 inhibition 仿真 | P0 | reaction_builder.py:166 |
| TD-011 | Reaction IR | else: 分支 8 种机制错误归类 | 多种机制 | P0 | reaction_builder.py:190 |
| TD-012 | State | experimental_data orphan state | calibration/hypothesis | P1 | state.py |
| TD-013 | Graph | _pubmed_cache 无界内存泄漏 | 长时间运行 | P1 | nodes_v2.py:486 |
| TD-014 | Calibration | 输出不回写仿真参数 | 任何 calibration | P0 | calibration_agent.py:207 |
| TD-015 | Validation | validation hook 副作用修改 v3 路由 | 验证失败 | P1 | validation_agent.py:333 |
| TD-016 | ODE Template | Binding 渲染为一级反应 | 含 binding 仿真 | P0 | destruction_complex.j2:79 |
| TD-017 | ODE Template | Complex formation 一级反应 | 含 complex 仿真 | P0 | destruction_complex.j2:72 |
| TD-018 | Reaction IR | Dimerization 化学计量 1→1 | 含 dimerization | P0 | reaction_builder.py:190 |
| TD-019 | Reaction IR | Transcription TF 作为 substrate | 含 transcription | P0 | reaction_builder.py:190 |
| TD-020 | Reaction IR | Translation mRNA 作为 substrate | 含 translation | P0 | reaction_builder.py:190 |
| TD-021 | Reaction IR | GTP_GDP exchange 缺 GEF/GAP modifier | 含 GTP_GDP | P0 | reaction_builder.py:190 |
| TD-022 | Reaction IR | Cleavage 缺 enzyme modifier | 含 cleavage | P0 | reaction_builder.py:190 |
| TD-023 | Reaction IR | Dissociation 单 product | 含 dissociation | P0 | reaction_builder.py:190 |
| TD-024 | Reaction IR | Sequestration 语义反转 | 含 sequestration | P0 | reaction_builder.py:190 |
| TD-025 | Reaction IR | Activation 默认 source→target | 含 activation | P0 | reaction_builder.py:190 |
| TD-026 | Schema | Modifier 无 Ki/Kact/n_hill/inhibition_type | 别构调节 | P0 | schema.py:83 |
| TD-027 | Schema | kinetics_type 5 值枚举折叠细节 | kinetics 验证 | P1 | schema.py:141 |
| TD-028 | Schema | Modifier.site 单 string 无法多位点 | 多位点修饰 | P0 | schema.py:88 |
| TD-029 | Schema | Constraint.expression 无结构字符串 | 约束验证 | P1 | schema.py:111 |
| TD-030 | Schema | Constraint.type 静默降级 | 未知约束类型 | P1 | schema.py:120 |
| TD-031 | Schema | compartments 无序丢失方向 | transport | P2 | reaction_builder.py:195 |
| TD-032 | Schema | stoichiometry 仅 int | lumped model | P2 | schema.py:71 |
| TD-033 | Schema | 仅 5 compartment 无 ER/Golgi | 分泌通路 | P2 | schema.py:49 |
| TD-034 | Schema | SpeciesV2 无单位转换 | 混合单位 | P1 | schema.py:50 |
| TD-035 | Schema | Provenance 永远 None | 追溯性 | P1 | reaction_builder.py:214 |
| TD-036 | Mechanism | is_enzymatic_mechanism 遗漏 ubiquitination | ubiquitination 验证 | P1 | mechanism_types.py:179 |
| TD-037 | Mechanism | _normalize_kinetics_name 折叠 first_order | kinetics 验证 | P1 | mechanism_types.py:91 |
| TD-038 | Mechanism | transport 统一映射 nuclear_import | cytoplasm_translocation | P1 | mechanism_types.py:130 |
| TD-039 | Mechanism | 19 枚举值声称 17 | 文档/合约 | P1 | mechanism_types.py:24 |
| TD-040 | Constraints | check_mass_conservation 仅 token 检查 | 质量守恒验证 | P0 | constraints.py:45 |
| TD-041 | Constraints | auto_generate 仅 phosphorylation pair | binding/dimerization | P1 | constraints.py:74 |
| TD-042 | Constraints | check_enzymatic 仅 catalytic modifier | 别构/激活 | P1 | constraints.py:161 |
| TD-043 | Constraints | check_thermodynamic 仅关键词 | 热力学验证 | P1 | constraints.py:199 |
| TD-044 | Constraints | check_non_negative 仅初始浓度 | 动态负浓度 | P1 | constraints.py:146 |
| TD-045 | Constraints | check_steady_state 为 no-op | 稳态验证 | P2 | constraints.py:122 |
| TD-046 | Constraints | 无 moiety conservation | 磷酸/泛素/GTP | P1 | constraints.py 全文 |
| TD-047 | ODE Renderer | _select_template 忽略 requires_dde | DDE 模板选择 | P2 | ode_renderer_v2.py:198 |
| TD-048 | ODE Renderer | EGFR/MAPK 路由到振荡模板 | EGFR/MAPK 仿真 | P1 | ode_renderer_v2.py:61 |
| TD-049 | ODE Renderer | Jinja2 无 StrictUndefined | 模板变量缺失 | P2 | ode_renderer_v2.py:109 |
| TD-050 | ODE Renderer | DDE delay 不从 reaction_ir 提取 | 纯 IR 输入 | P1 | ode_renderer_v2.py:157 |
| TD-051 | ODE Template | degradation 仅 1/9 模板实现 | 含 degradation | P1 | oscillatory_feedback.j2 |
| TD-052 | ODE Template | dissociation 仅 2/9 模板实现 | 含 dissociation | P1 | bistable_switch.j2 |
| TD-053 | ODE Template | dimerization 仅 2/9 模板正确 | 含 dimerization | P1 | transcription_factor.j2 |
| TD-054 | ODE Template | ubiquitination 符号错误 | 含 ubiquitination | P1 | oscillatory_feedback.j2:136 |
| TD-055 | ODE Template | Hill 函数无负值保护 | stiff 仿真 | P1 | oscillatory_feedback.j2:122 |
| TD-056 | ODE Template | nuclear_transport 无体积缩放 | nuclear transport | P1 | nuclear_transport.j2:76 |
| TD-057 | ODE Template | caspase cleavage 净零通量 | Apoptosis | P1 | caspase_cascade.j2:74 |
| TD-058 | ODE Template | bistable cleavage 净零通量 | Apoptosis | P1 | bistable_switch.j2:78 |
| TD-059 | ODE Template | cyclin_cdk max_step 条件错误 | Cell Cycle | P2 | cyclin_cdk_toggle.j2:233 |
| TD-060 | Solver | 无事件检测 | bistability | P1 | dde_solver.py:116 |
| TD-061 | Solver | max_step 硬编码无刚度自适应 | 快/慢混合 | P1 | dde_solver.py:50 |
| TD-062 | Solver | bistability 无滞后验证 | bistability 判定 | P1 | bistability_detector.py:73 |
| TD-063 | Solver | oscillation 无 FFT | 振荡周期 | P2 | oscillation_detector.py:84 |
| TD-064 | Solver | oscillation 阈值任意 | 振荡分类 | P2 | oscillation_detector.py:96 |
| TD-065 | Solver | bistability _find_key_species 首匹配 | Apoptosis | P2 | bistability_detector.py:104 |
| TD-066 | Solver | 无 stiff MM 专用 solver | MM 饱和 | P1 | solvers/__init__.py |
| TD-067 | Sandbox | BIO_CHECK 标记无模板输出 | 任何 v4 仿真 | P0 | sandbox.py:178 |
| TD-068 | Sandbox | 无质量守恒后验证 | 任何仿真 | P0 | sandbox.py 全文 |
| TD-069 | Sandbox | NaN/Inf CSV 不检测 | 仿真后 | P1 | sandbox.py:382 |
| TD-070 | Sandbox | CSV 目录泄漏 | 多次仿真 | P1 | sandbox.py:519 |
| TD-071 | Sandbox | _BLOCKED_BUILTINS 阻止 open | 用户 ODE | P1 | sandbox.py:62 |
| TD-072 | Sandbox | 生物检查在无图像时跳过 | matplotlib | P1 | sandbox.py:532 |
| TD-073 | Graph | _clarification_events 进程全局 | 多 worker | P1 | graph_v3.py:86 |
| TD-074 | Graph | _ontology_agent 单例非线程安全 | 并发 | P1 | ontology_agent.py:358 |
| TD-075 | Graph | pre_router 清除 20+ 字段 | 每次请求 | P2 | graph_v3.py:251 |
| TD-076 | Graph | worker_ode 原地修改 state | ODE 渲染 | P2 | graph_v3.py:796 |
| TD-077 | Graph | calibration 在 validation 之后 | v4 验证 | P1 | graph_v3.py:1540 |
| TD-078 | Graph | ontology_hook 在 worker_mechanism 之前 | v4 运行 | P1 | graph_v3.py:1497 |
| TD-079 | Graph | hypothesis_hook 在 worker_report 之前 | hypothesis | P1 | graph_v3.py:1514 |
| TD-080 | State | next_worker 死状态 | supervisor | P2 | state.py:135 |
| TD-081 | State | need_human_review 遗留 | schema | P2 | state.py:45 |
| TD-082 | State | simulation_ci 已废弃 | schema | P2 | state.py:115 |
| TD-083 | Ontology | EGF 双重身份 | EGFR | P1 | ontology_agent.py:33 |
| TD-084 | Ontology | GO geneProductSymbol 无效 | GO 查询 | P1 | go_client.py:63 |
| TD-085 | Ontology | GO evidence 字段错误 | GO 查询 | P2 | go_client.py:85 |
| TD-086 | Ontology | pathway_registry 无 BioModels ID | calibration | P1 | pathway_registry.py |
| TD-087 | Ontology | EGFR KEGG ID 错误 | EGFR | P2 | pathway_registry.py:64 |
| TD-088 | Specialist | p53 substrate=product | p53 仿真 | P1 | p53_specialist.py:153 |
| TD-089 | Specialist | Apoptosis 未定义 procaspase | Apoptosis | P1 | apoptosis_specialist.py:225 |
| TD-090 | Specialist | Cell Cycle PMID 错误 | Cell Cycle | P1 | cell_cycle_specialist.py:70 |
| TD-091 | Specialist | Apoptosis ABT-199 重复 | 药物计算 | P2 | apoptosis_specialist.py:501 |
| TD-092 | Specialist | JAK-STAT Tofacitinib 重复 | 药物计算 | P2 | jak_stat_specialist.py:498 |
| TD-093 | Specialist | JAK-STAT STAT3 orphaned | JAK-STAT | P2 | jak_stat_specialist.py:122 |
| TD-094 | Specialist | Wnt 引用不存在模板 | Wnt | P1 | wnt_specialist.py:518 |
| TD-095 | Specialist | TGF-β 引用不存在模板 | TGF-β | P1 | tgf_beta_specialist.py:393 |
| TD-096 | Calibration | Hash-based RNG 不可重复 | CI/CD | P1 | confidence_interval.py:219 |
| TD-097 | Sensitivity | 占位 model_func 无意义 | sensitivity | P0 | sensitivity_analyzer.py:293 |
| TD-098 | Validation | L4 PMID 不一致 | benchmark | P1 | level4_benchmark.py:66 |
| TD-099 | Validation | L3 守恒基于计数非通量 | crosstalk | P1 | level3_crosstalk.py:302 |
| TD-100 | Frontend | 无 Provenance/SBML Compare/Report 导出 | UX | P1 | store.ts / WorkbenchShell.tsx |

---

## 12. Top 100 Future TODO

| ID | 类别 | 任务 | 优先级 | 关联 TD |
|----|------|------|--------|---------|
| FT-001 | ODE Renderer | 修复 `_extract_species_names` 读取 `canonical_name` 而非 `name` | P0 | TD-001 |
| FT-002 | ODE Renderer | 修复 `_extract_params` 从 ReactionV2/Specialist 读取参数 | P0 | TD-002 |
| FT-003 | ODE Renderer | 修复 `_extract_edges` 从 SpeciesRef.species_id 提取 source/target | P0 | TD-003 |
| FT-004 | ODE Template | 修复磷酸化 MM 公式 `v = k_cat * [E] * [S] / (Km + [S])` | P0 | TD-004 |
| FT-005 | Solver | 完成 jitcdde DDE 求解器接线 | P0 | TD-005 |
| FT-006 | Calibration | 实现真实 model_func（调用 ODE solver） | P0 | TD-006 |
| FT-007 | Validation L2 | 用真实 ODE 仿真替换 MOCK 线性衰减 | P0 | TD-007 |
| FT-008 | Specialists | 为全部 10 条通路添加文献动力学参数 | P0 | TD-008 |
| FT-009 | Ontology | 修复 SBO 反向映射（分配唯一 SBO ID） | P1 | TD-009 |
| FT-010 | Reaction IR | 修复 INHIBITION 语义（inhibitor 为 modifier） | P0 | TD-010 |
| FT-011 | Reaction IR | 为 8 种机制实现正确的 reactant/product/modifier | P0 | TD-011 |
| FT-012 | State | 声明 `experimental_data` 并添加 writer node | P1 | TD-012 |
| FT-013 | Graph | 为 `_pubmed_cache` 添加 TTL + LRU 驱逐 | P1 | TD-013 |
| FT-014 | Calibration | 将 calibrated_params 回写 `state.parameters` | P0 | TD-014 |
| FT-015 | Validation | 移除 validation hook 对 `pending_clarification` 的副作用 | P1 | TD-015 |
| FT-016 | ODE Template | 修复 binding 为双分子反应 | P0 | TD-016 |
| FT-017 | ODE Template | 修复 complex_formation 为多 substrate | P0 | TD-017 |
| FT-018 | Reaction IR | 修复 dimerization 化学计量为 2→1 | P0 | TD-018 |
| FT-019 | Reaction IR | 修复 transcription（TF 为 modifier） | P0 | TD-019 |
| FT-020 | Reaction IR | 修复 translation（mRNA 为 modifier） | P0 | TD-020 |
| FT-021 | Reaction IR | 添加 GEF/GAP 为 GTP_GDP exchange modifier | P0 | TD-021 |
| FT-022 | Reaction IR | 添加 enzyme 为 cleavage modifier | P0 | TD-022 |
| FT-023 | Reaction IR | 修复 dissociation 为多 product | P0 | TD-023 |
| FT-024 | Reaction IR | 修复 sequestration 语义 | P0 | TD-024 |
| FT-025 | Reaction IR | 修复 activation 为 modifier-based | P0 | TD-025 |
| FT-026 | Schema | 添加 Modifier 的 Ki/Kact/n_hill/inhibition_type/alpha | P0 | TD-026 |
| FT-027 | Schema | 扩展 kinetics_type 枚举 | P1 | TD-027 |
| FT-028 | Schema | Modifier.site 改为 list[str] | P0 | TD-028 |
| FT-029 | Schema | Constraint.expression 改为 AST/symbolic | P1 | TD-029 |
| FT-030 | Schema | Constraint.type 拒绝未知类型而非降级 | P1 | TD-030 |
| FT-031 | Schema | 添加 from_compartment/to_compartment | P2 | TD-031 |
| FT-032 | Schema | stoichiometry 改为 float | P2 | TD-032 |
| FT-033 | Schema | 添加 ER/Golgi/lysosome compartment | P2 | TD-033 |
| FT-034 | Schema | 添加单位转换因子 | P1 | TD-034 |
| FT-035 | Reaction IR | 传播 sbml_model_id 到 Provenance | P1 | TD-035 |
| FT-036 | Mechanism | 将 ubiquitination 加入 is_enzymatic_mechanism | P1 | TD-036 |
| FT-037 | Mechanism | 区分 first_order 与 mass_action | P1 | TD-037 |
| FT-038 | Mechanism | 区分 transport 子类型 | P1 | TD-038 |
| FT-039 | Mechanism | 修正文档声称 17→19 | P1 | TD-039 |
| FT-040 | Constraints | 实现数值质量守恒检查 | P0 | TD-040 |
| FT-041 | Constraints | 扩展 auto_generate 覆盖 binding/dimerization/complex | P1 | TD-041 |
| FT-042 | Constraints | 扩展 check_enzymatic 覆盖 allosteric/activating | P1 | TD-042 |
| FT-043 | Constraints | 实现 K_eq = k_forward/k_reverse 验证 | P1 | TD-043 |
| FT-044 | Constraints | 添加动态负浓度检查 | P1 | TD-044 |
| FT-045 | Constraints | 实现真实稳态验证 | P2 | TD-045 |
| FT-046 | Constraints | 添加 moiety conservation 类型 | P1 | TD-046 |
| FT-047 | ODE Renderer | _select_template 根据 requires_dde 选择 | P2 | TD-047 |
| FT-048 | ODE Renderer | 为 EGFR/MAPK 创建 transient_cascade 模板 | P1 | TD-048 |
| FT-049 | ODE Renderer | 启用 Jinja2 StrictUndefined | P2 | TD-049 |
| FT-050 | ODE Renderer | 从 reaction_ir 提取 DDE delay | P1 | TD-050 |
| FT-051 | ODE Template | 在全部模板添加 degradation 分支 | P1 | TD-051 |
| FT-052 | ODE Template | 在全部模板添加 dissociation 分支 | P1 | TD-052 |
| FT-053 | ODE Template | 在全部模板统一 dimerization 2nd-order | P1 | TD-053 |
| FT-054 | ODE Template | 修复 ubiquitination 符号 | P1 | TD-054 |
| FT-055 | ODE Template | 添加 Hill 函数负值保护 | P1 | TD-055 |
| FT-056 | ODE Template | 添加 compartment 体积缩放 | P1 | TD-056 |
| FT-057 | ODE Template | 修复 caspase cleavage 符号 | P1 | TD-057 |
| FT-058 | ODE Template | 修复 bistable cleavage 符号 | P1 | TD-058 |
| FT-059 | ODE Template | 条件化 max_step | P2 | TD-059 |
| FT-060 | Solver | 添加 solve_ivp events 参数 | P1 | TD-060 |
| FT-061 | Solver | 实现刚度自适应 max_step | P1 | TD-061 |
| FT-062 | Solver | 实现滞后验证 bistability | P1 | TD-062 |
| FT-063 | Solver | 实现 FFT 振荡周期估计 | P2 | TD-063 |
| FT-064 | Solver | 基于实验数据校准振荡阈值 | P2 | TD-064 |
| FT-065 | Solver | 传递 key_species_hint | P2 | TD-065 |
| FT-066 | Solver | 添加 BDF/Radau fallback | P1 | TD-066 |
| FT-067 | Sandbox | 在 ODE 模板输出 BIO_CHECK 标记 | P0 | TD-067 |
| FT-068 | Sandbox | 实现仿真后质量守恒验证 | P0 | TD-068 |
| FT-069 | Sandbox | 实现 CSV NaN/Inf 检测 | P1 | TD-069 |
| FT-070 | Sandbox | 清理 persistent_dir | P1 | TD-070 |
| FT-071 | Sandbox | 允许 np.savetxt 的 open 调用 | P1 | TD-071 |
| FT-072 | Sandbox | 独立执行生物检查（不依赖图像） | P1 | TD-072 |
| FT-073 | Graph | 将 clarification_events 迁移到 Redis/数据库 | P1 | TD-073 |
| FT-074 | Graph | 将 ontology_agent 改为请求级实例 | P1 | TD-074 |
| FT-075 | Graph | 精简 pre_router 清除列表 | P2 | TD-075 |
| FT-076 | Graph | 修复 worker_ode 不原地修改 state | P2 | TD-076 |
| FT-077 | Graph | 调换 calibration 和 validation 顺序 | P1 | TD-077 |
| FT-078 | Graph | 移动 ontology_hook 到 worker_mechanism 之后 | P1 | TD-078 |
| FT-079 | Graph | 移动 hypothesis_hook 到 worker_report 之后 | P1 | TD-079 |
| FT-080 | State | 移除 next_worker 死状态 | P2 | TD-080 |
| FT-081 | State | 移除 need_human_review 遗留 | P2 | TD-081 |
| FT-082 | State | 移除 simulation_ci 废弃字段 | P2 | TD-082 |
| FT-083 | Ontology | 修复 EGF 双重身份（优先 chemical） | P1 | TD-083 |
| FT-084 | Ontology | 修复 GO 客户端参数 | P1 | TD-084 |
| FT-085 | Ontology | 修复 GO evidence 字段 | P2 | TD-085 |
| FT-086 | Ontology | 添加 BioModels ID 到 pathway_registry | P1 | TD-086 |
| FT-087 | Ontology | 修复 EGFR KEGG ID 为 hsa04012 | P2 | TD-087 |
| FT-088 | Specialist | 修复 p53 substrate/product 命名 | P1 | TD-088 |
| FT-089 | Specialist | 添加 procaspase 物种定义 | P1 | TD-089 |
| FT-090 | Specialist | 修复 Cell Cycle PMID | P1 | TD-090 |
| FT-091 | Specialist | 去除 Apoptosis ABT-199 重复 | P2 | TD-091 |
| FT-092 | Specialist | 去除 JAK-STAT Tofacitinib 重复 | P2 | TD-092 |
| FT-093 | Specialist | 为 STAT3 添加 core reaction 或移除 | P2 | TD-093 |
| FT-094 | Specialist | 实现 destruction_complex.j2 模板 | P1 | TD-094 |
| FT-095 | Specialist | 实现 transcription_factor.j2 模板 | P1 | TD-095 |
| FT-096 | Calibration | 用固定 seed 替换 hash-based RNG | P1 | TD-096 |
| FT-097 | Sensitivity | 实现真实 model_func | P0 | TD-097 |
| FT-098 | Validation | 统一 L4 与 specialist PMID | P1 | TD-098 |
| FT-099 | Validation | L3 改为通量守恒 | P1 | TD-099 |
| FT-100 | Frontend | 添加 Provenance + SBML Compare + Report 导出 | P1 | TD-100 |

---

## 13. Final Readiness Assessment

### 评分汇总

| 维度 | 评分 | 阻塞项 |
|------|------|--------|
| Architecture | 42/100 | ODE Renderer 与 Schema 脱节；DDE 空壳；Calibration 装饰性 |
| Scientific Coverage | 75.7% (22.7/30) | 全部 specialist 无动力学参数；Reaction IR 语义错误 |
| Simulation Correctness | 15/100 | 零通量 ODE；MM 公式错误；负浓度无检测 |
| Validation Rigor | 25/100 | L2 MOCK 仿真；L4 仅 50% 覆盖；L5 从不阻断 |
| Frontend Scientific UX | 60/100 | 无 provenance/SBML Compare/Report 导出 |
| Verification Suite | 90/100 | 273 测试用例设计完成，19 个可执行 |
| Documentation | 85/100 | 9 篇文档 + Release Report + Verification Report |
| Code Quality | 40/100 | 100 个 Failure Modes（16 P0 + 36 P1） |
| Reproducibility | 20/100 | Hash-based RNG 不可重复；无 COMBINE/SED-ML |

---

### READY FOR OPEN SOURCE ?

## **NO**

**评分：35/100**

**阻塞项**：

1. **P0-FM-001/002/003**：ODE Renderer 读取不存在的字段，所有 v4 仿真为零通量空壳。开源后任何用户运行 v4 仿真都会得到空白结果。
2. **P0-FM-004**：磷酸化 MM 公式在全部 9 个模板中错误（`k_cat * src^2`），系统性 copy-paste 错误。
3. **P0-FM-005**：DDE 求解器完全不可用，p53/NF-κB 振荡机制失效。
4. **P0-FM-013**：全部 10 条 specialist 无动力学参数，仿真使用无意义默认值。
5. **P0-FM-010**：L2 验证使用 MOCK 线性衰减，产生 False Positive。
6. **P0-FM-012**：Sandbox 生物检查不触发，负浓度/NaN 不被检测。
7. **P0-FM-008/009**：Reaction IR Builder 的 INHIBITION 和 else: 分支语义错误，影响 12/16 种机制。
8. **HC-1/HC-3/HC-4/HC-5**：多个进程全局状态导致多 worker 部署不安全。
9. **TD-073**：`_clarification_events` 进程全局 dict 在多 worker 部署下失效。
10. **TD-013**：`_pubmed_cache` 无界内存泄漏。

**开源前必须修复的 16 个 P0 项**，估计需要修复 TD-001 至 TD-068 中的全部 P0 项。

---

### READY FOR AI FOR SCIENCE INTERVIEW ?

## **NO (但有展示价值)**

**评分：55/100**

**可以展示的部分**：
- 架构设计思路（Supervisor-Worker LangGraph + 6 个 v4 Hook 链 + Feature Flag 隔离）
- 10 通路 specialist 的生物学覆盖（机制完整性 4.6/5）
- Validation Pyramid 5 级设计理念
- 4-pane Scientific IDE 前端设计
- Verification Suite 273 测试用例架构
- 3 粗粒度 Feature Flag 收敛设计

**面试中的风险**：
- 如果面试官要求现场运行 v4 仿真，将产出零通量结果
- 如果面试官检查磷酸化 MM 公式，会发现 `k_cat * src^2` 错误
- 如果面试官检查 DDE 实现，会发现 jitcdde 未接线
- 如果面试官检查 calibration，会发现占位 model_func
- 如果面试官检查 L2 验证，会发现 MOCK 线性衰减

**建议**：在面试中展示架构设计、生物学覆盖、Verification Suite 设计，但避免现场运行 v4 仿真。将 v4 定位为"架构完成、科学验证进行中"的阶段。

---

### READY FOR RESEARCH COLLABORATION ?

## **NO**

**评分：30/100**

**阻塞项**：
1. 仿真结果不可信（零通量 + 错误 MM 公式 + DDE 失效）
2. 无动力学参数（合作者无法运行有意义的仿真）
3. 无参数不确定性量化（无 Bayesian calibration）
4. 无 COMBINE/SED-ML 支持（无法与其他工具互操作）
5. 无随机仿真（低拷贝数物种无法建模）
6. Calibration 装饰性（输出不回写仿真）
7. Validation 不可信（L2 MOCK + L4 50% 覆盖）
8. 无可重复性（Hash-based RNG）

**合作者需要的最低功能**：修复 P0 后 + 添加 Bayesian calibration + 至少 1 条通路完整动力学参数。

---

### READY FOR PAPER FOUNDATION ?

## **NO**

**评分：25/100**

**阻塞项**：
1. 全部 P0 bug 未修复 — 仿真结果不可发表
2. 无 Uncertainty Quantification — 结果无置信区间
3. 无 BioModels 回归验证通过 — 无法与参考文献对比
4. 无 Stochastic Simulation — 低拷贝数系统无法建模
5. 无 Bifurcation Analysis — 无法分析双稳态/分岔
6. 无 COMBINE Archive — 无法满足期刊可重复性要求
7. Hypothesis Layer 无 refinement 循环 — 假说生成无迭代验证
8. Calibration 无后验分布 — 参数辨识不可发表

**投稿前必须完成**：
1. 修复全部 16 个 P0 bug
2. 实现 Bayesian Calibration（PyMC/emcee）
3. 实现 Uncertainty Quantification（chaospy）
4. 通过至少 30 个 BioModels 回归测试
5. 实现 Stochastic Simulation（gillespy2）
6. 实现 COMBINE Archive 导入/导出
7. 完成 100 条假说验证（成功率 > 60%）

**估计修复路径**：先修复 16 个 P0 → 补充动力学参数 → 通过 BioModels 回归 → 添加 UQ/Bayesian → 投稿。

---

### 总结

| 问题 | 结论 | 评分 |
|------|------|------|
| READY FOR OPEN SOURCE ? | **NO** | 35/100 |
| READY FOR AI FOR SCIENCE INTERVIEW ? | **NO (但有展示价值)** | 55/100 |
| READY FOR RESEARCH COLLABORATION ? | **NO** | 30/100 |
| READY FOR PAPER FOUNDATION ? | **NO** | 25/100 |

**核心结论**：BioDynamics v4 的架构设计具有前瞻性（LangGraph + 10-pathway specialist + 5-level Validation Pyramid + 4-pane Scientific IDE），但实现层存在 16 个 P0 级科学错误，导致 v4 仿真管线当前不可用。系统在 v3 模式下（所有 flag OFF）仍可运行，但 v4 科学层需要系统性修复后才能达到开源、合作或投稿标准。

**最高杠杆修复顺序**：
1. **TD-001/002/003**（ODE Renderer 字段修复）→ 解锁全部 v4 仿真
2. **TD-004**（MM 公式修复）→ 修复系统性 copy-paste 错误
3. **TD-005**（DDE 接线）→ 解锁 p53/NF-κB 振荡
4. **TD-008**（动力学参数）→ 使仿真有生物学意义
5. **TD-010/011**（Reaction IR Builder 修复）→ 修复 12/16 机制语义
6. **TD-067/068**（Sandbox 生物检查）→ 使验证可信

---

*End of BioDynamics v4 Final Verification Report.*
