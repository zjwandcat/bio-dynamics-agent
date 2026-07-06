# BioDynamics Agent — 严格结构审计报告 v1.0

> 审计对象：bio-dynamics-agent（早期科研级癌症信号通路仿真系统）
> 审计范围：生物学真实性 / 数学建模 / Reaction IR / ODE 模板 / 路由与 Agent / DomainChecker / SBML 对齐 / RAG / Agent 架构 / 可扩展性
> 审计方法：基于代码事实的逐行核查，所有结论均可在 cited file 中验证
> 审计约束：只指出"错误"与"结构问题"，不写代码改动清单，不展开工程实现

---

## 1. Executive Summary

该系统声称覆盖 10 条癌症信号通路并具备早期科研级仿真能力，但代码事实表明其**实际能力远低于声明**：

**核心断言（基于代码证据）**：

1. **通路覆盖实际为 1/10，非 10/10**。[biomodels_reactions.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/biomodels_reactions.py) 中 `ALLOWED_PATHWAY_SET = frozenset({"EGF_EGFR_MAPK"})` 硬编码仅允许 EGF-EGFR-MAPK 一条通路；`FORBIDDEN_PATHWAY_TERMS` 显式禁止 PI3K、AKT、mTOR、NF-κB、JAK、STAT、crosstalk 等关键词。系统对其他 9 条通路（Apoptosis / p53 / Cell cycle / JAK-STAT / NF-κB / Wnt / TGF-β / PI3K-AKT-mTOR / MAPK 独立模型）的反应会被静默丢弃。

2. **ODE 模板覆盖率仅 1/10 通路**。9 个 Jinja2 模板中仅 `Signaling_Cascade_Phos.j2` 支持真实生化机制（binding + phosphorylation），其余 8 个为 Hill/Emax 表型黑盒。6 类关键模板完全缺失：bistable switch、oscillatory feedback、transcriptional delay、nuclear transport、ubiquitination、destruction complex。

3. **数学建模严重退化**。`_mechanism_phosphorylation.j2` 第 3 行明确写有"TASK 4 修复：移除 Michaelis-Menten"，使 MAPK 级联的 ultrasensitivity 与 bistability 无法建模。所有模板均为常微分初值问题（solve_ivp / LSODA），无 DDE（延迟微分方程）、无 SDE（随机微分方程）、无 rule-based modeling。

4. **RAG 四层分层名存实亡**。`experiment` collection 在自动建库流程中始终为空（`extend_rag_db.py` 第 433 行 `experiment_records: list[dict[str, Any]] = []` 从不追加）；`parameter` collection **无 `pathway` 字段**，cross-pathway contamination（如 Ras/MAPK/AKT 共享分子）无法防范；`evidence` collection 的 `doi`、`figure_ref`、`cell_line` 永远为空字符串。

5. **Agent 架构名不副实**。号称 supervisor-worker 动态编排，实际是固定线性流水线（`_FULL_PLAN` 硬编码 8 步顺序），Supervisor 仅做步进计数。三类关键 Agent 缺失：Pathway Classifier、Pathway Specialist、Cross-talk Coordinator。Cross-talk 更被显式列为禁止词。

6. **Validation Oracle 形同虚设**。[sbml_validator.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/sbml_validator.py) 的 Track B 输出 `error_diff=0.0`（实际未仿真），误导下游认为仿真准确；`_skipped_report` 默认 `pass=True`，任何 SBML 不可用情况均静默通过。

7. **标准本体完全未对齐**。[species_ontology.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/species_ontology.py) 全文无 HGNC / UniProt / ChEBI / GO / SBO 引用，所有类型推断依赖硬编码字符串列表，仅识别 EGF-EGFR-MAPK 链的 8 个蛋白池。

**总评**：系统当前不具备"10 条通路仿真"能力，仅是 EGF-EGFR-MAPK 单通路演示原型。要从原型升级到声明目标，需重建模板系统、Agent 架构、RAG 分层与本体对齐，而非局部修补。

---

## 2. Biological Failure Points（生物学失败点）

### 2.1 通路覆盖：声明 10 条，实际 1 条

`biomodels_reactions.py` L35 与 L38-42 的硬编码是生物学层面的根本性失败：

```
ALLOWED_PATHWAY_SET = frozenset({"EGF_EGFR_MAPK"})
FORBIDDEN_PATHWAY_TERMS = ("pi3k", "akt", "mtor", "nf-kappa", "nf-kb",
                           "jak", "stat", "stat3", "stat5",
                           "feedback", "crosstalk", "cross-talk", "emergent")
```

这意味着：

- 用户问 PI3K-AKT 通路 → 反应图构建阶段被静默丢弃
- 用户问 NF-κB 振荡 → "feedback" 被禁止词过滤
- 用户问 Wnt/β-catenin → 无 destruction complex 机制可表达
- 用户问 TGF-β/SMAD → 无 nuclear transport 机制可表达
- 用户问 apoptosis → 无 caspase 级联模板可用
- 用户问 p53 → 无 transcription + ubiquitination 机制可表达
- 用户问 cell cycle → 无 bistable + oscillation 机制可表达
- 用户问 JAK-STAT → 无 transcription + nuclear transport 可表达

更严重的是：**`biomodels_client.py` L103-110 的 `_PATHWAY_KEYWORDS` 包含了 PD1/PDL1/CTLA4/Wnt/Notch/Hedgehog/p53 等 20+ 关键词**，能识别这些通路为 `calibration_reference` 角色。但下游 `biomodels_reactions.py` 直接拒绝这些通路的反应。这种"客户端识别 + 下游拒绝"的设计形成**功能性死锁**：系统能"看出"用户问的是哪条通路，但"无法处理"任何非 EGF-EGFR-MAPK 的反应。

### 2.2 Cross-talk 完全缺失且被禁止

`biomodels_reactions.py` L41 显式将 `crosstalk`、`cross-talk` 列入 `FORBIDDEN_NON_MODEL_TERMS`。这是一个**生物学错误的禁令**：癌症信号网络的核心特征就是通路间 cross-talk（如 PI3K→MAPK 交叉激活、AKT→Raf Ser259 抑制、Ras→PI3K 直接激活）。

代码中无任何 Cross-talk Coordinator Agent、无 cross-talk edge 类型、无 cross-talk 专用 RAG 检索策略。`Signaling_Cascade_Phos.j2` 的 `PLOT_CANONICAL_SET`（L209-217）硬编码 7 个 EGFR 通路物种，跨通路 shared species（如 Ras/AKT/MEK）的非 canonical 角色会被静默丢弃。

### 2.3 关键机制缺失清单

下表汇总 10 条通路必需但代码中完全缺失的机制：

| 通路 | 缺失机制 | 代码证据 |
|------|---------|---------|
| EGFR / MAPK | Michaelis-Menten 酶饱和（ultrasensitivity 必需） | `_mechanism_phosphorylation.j2` L3 主动移除 |
| PI3K-AKT-mTOR | 负反馈（S6K→IRS1 抑制） | `Signaling_Cascade_Phos.j2` mechanism 列表无 inhibition 分支 |
| p53 | transcription + ubiquitination + negative feedback + bistability | 全栈无 transcription/ubiquitination 模板；`Signaling_Cascade_Phos.j2` L178 主动 `禁止 gene-expression-like` |
| Apoptosis | caspase cascade + bistable switch + MOMP | 无 caspase 切割机制；caspase 是蛋白酶非激酶，误用 phosphorylation 模板会引入伪可逆性 |
| Cell cycle | bistable switch + oscillation + delayed feedback | 全栈无 bistable/oscillatory/delay 模板；LSODA 收敛定态 |
| JAK-STAT | transcription + nuclear translocation + negative feedback (SOCS/CIS) | 全栈无 transcription/nuclear transport 模板 |
| NF-κB | oscillatory feedback + IκBα degradation + nuclear transport | 仅 degradation 存在；振荡完全缺失；IκBα 转录延迟无法表达 |
| Wnt/β-catenin | destruction complex + transcription + ubiquitination | 全栈无 destruction complex 机制（需 binding+phosphorylation+ubiquitination 三步耦合） |
| TGF-β/SMAD | transcription + nuclear translocation + SMAD 复合体 | 同 JAK-STAT 缺口 |

### 2.4 DomainChecker 通路盲化

[domain_checker.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/domain_checker.py) 中无任何 p53 / caspase / Bcl-2 / Wnt / β-catenin / SMAD / NF-κB / Cyclin / CDK / STAT / JAK 关键词（grep 全文 0 命中）。所有通路特异规则缺失：

- p53 → Mdm2 negative feedback loop：未实现
- Wnt → β-catenin destruction complex：未实现
- NF-κB → IκBα oscillation：未实现
- Cell cycle → Cyclin-CDK switch：未实现
- Apoptosis → caspase bistable switch：未实现

`_check_pathway_completeness`（L378-398）仅检查 4 个 EGF-EGFR-MAPK shortcut（`("EGF","MAPK")`、`("EGF","Ras")`、`("Ras","MAPK")`、`("EGF","pMAPK")`），对其他 9 条通路无任何完整性约束。

### 2.5 KG 环路打破策略违反生物学

[kg_builder.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/kg_builder.py) L17-21 的 `_INTERACTION_STRENGTH` 仅 3 类边，且 L105-118 的 `_break_cycle` **优先丢弃 inhibition 边**（strength=1 最低）。

这是**生物学错误**：负反馈环路（由 inhibition 构成的环）是信号稳态的核心机制，不应被丢弃。NF-κB-IκBα 互抑制、p53-Mdm2 互抑制、CDK-Cyclin 反馈环等关键生物学结构会被这一策略破坏。

### 2.6 物种本体未对齐

[species_ontology.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/species_ontology.py) 全文 298 行，无 HGNC / UniProt / ChEBI / GO / SBO 引用。`SPECIES_TYPES`（L33-45）是自定义 11 类，与标准本体不互通。`_get_pool_name`（L216-243）仅识别 EGFR/Shc/Grb2/SOS/Ras/Raf/MEK/ERK 8 个蛋白池，对 STAT1/3、SMAD2/3/4、β-catenin、Cyclin D/E、CDK4/6、Caspase-3/8/9 等其他通路核心蛋白无守恒分组支持。

---

## 3. Mathematical Modeling Issues（数学建模问题）

### 3.1 动力学类型分布严重失衡

基于对 12 个模板文件的逐行核查：

| 动力学类型 | 出现位置 | 是否合理 |
|-----------|---------|---------|
| mass-action | `Signaling_Cascade_Phos.j2` + 2 个子模块 | 合理但不足以覆盖所有通路 |
| Michaelis-Menten | **完全缺失**（被 TASK 4 主动移除） | **错误**：MAPK ultrasensitivity 必需 |
| Hill | 8 个通用模板 | 过度依赖，掩盖机制细节 |
| Emax | PKPD 系列 + Combination + DoseSweep | 仅适合作表型建模 |
| first-order degradation | 所有模板 | 合理 |
| zero-order production | Simple/PKPD/Combination 系列 | 部分合理但硬编码速率 |

**最严重问题**：`_mechanism_phosphorylation.j2` L3 的注释 `TASK 4 修复：mass-action ONLY，移除 Michaelis-Menten 酶饱和项`。这是一次"修复"造成的退化——原本可能有 MM 实现，被改为纯 mass-action。后果：

- MAPK 级联的零阶 ultrasensitivity（Markevich 2004）无法再现
- bistability 无法涌现
- 信号开关行为丢失

### 3.2 缺失的数学结构

| 数学结构 | 用途 | 是否实现 |
|---------|------|---------|
| Bistable switch | Apoptosis all-or-none / Cell cycle toggle | **完全缺失** |
| Oscillatory feedback (delay ODE / DDE) | NF-κB 振荡 / p53 脉冲 / Cell cycle 振子 | **完全缺失** |
| Transcriptional delay | p53-Mdm2 / JAK-STAT-SOCS / NF-κB-IκBα / TGF-β-SMAD7 | **完全缺失** |
| Stochastic fluctuation | 低拷贝数物种（如 mRNA） | **完全缺失** |
| Rule-based modeling | 多状态蛋白（如多位点磷酸化）组合爆炸 | **完全缺失** |
| Spatial / compartmental | 核-质转运 | **完全缺失**（无 compartment 概念） |
| Bifurcation analysis | 通路决策点参数扫描 | **完全缺失** |

### 3.3 ODE 模板硬编码参数问题

- `Simple_Activation.j2` L27：`d_activator = -0.1 * activator` —— 0.1 不可配置
- `Simple_Inhibition.j2` L32：`d_inhibitor = -0.1 * inhibitor` —— 0.1 不可配置
- `PKPD_OneCompartment.j2` L29：`0.5 * (1.0 - effect) - 0.05 * target` —— 量产速率 0.5 与降解速率 0.05 硬编码
- `PKPD_TwoCompartment.j2` L32、`DoseSweep.j2` L31、`Combination.j2` L29：同上硬编码

这意味着不同通路的 target protein turnover 差异（如 p53 半衰期 ~20 min vs Cyclin B1 半衰期 ~60 min）无法建模。

### 3.4 _cascade_helpers.j2 静默修改用户参数

`_cascade_helpers.j2` L17-32 的 `_enforce_phos_dephos_ratio` 在模板加载时**静默修改用户传入的 `k_dephos`**（L27 `eparams["k_dephos"] = new_k_dephos`），强制比例为 10:1。

数学后果：

- 用户对 phosphatase 活性的真实设定被覆盖
- 对 PI3K-AKT 通路中 PTEN（PI3K 的关键负调控因子）的活性调节产生系统性偏差
- 无法研究 phosphatase 抑制剂（如 okadaic acid）的剂量响应

### 3.5 Cascade_Inhibition.j2 逻辑反转风险

`Cascade_Inhibition.j2` L21-24 的 `_hill_inh(x, kd, n) = kd^n / (kd^n + x^n)` 返回"未抑制分数"（递减函数）。但 L42-43：

```
if e["interaction"] == "inhibition":
    factor = 1.0 - _hill_inh(...)   # → x^n/(kd^n+x^n) 递增
```

这意味着**抑制剂浓度越高，靶点生成越多**——与"抑制"语义完全相反。L44-45 的 activation 分支同样反转。这是一个数学层面的逻辑错误，会导致所有使用 Cascade_Inhibition 模板的仿真结果失真。

### 3.6 求解器局限

`Signaling_Cascade_Phos.j2` L191-201 使用 `solve_ivp(method="LSODA", max_step=T_END/100)`：

- LSODA 适合 stiff 系统但收敛到定态
- 无 DDE 求解器（如 dde23 / JiTCDDE）
- 无随机求解器（如 SSA / Gillespie）
- max_step 限制过粗，对快速瞬态（如 caspase 切割）可能漏掉峰值

### 3.7 9 维特征提取的数学盲点

[feature_extractor.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/feature_extractor.py) L20-32 的 9 维特征（peak / peak_time / half_life / steady_state / fold_change / auc / rise_time / decay_time / max_slope）全部是通用时序统计量，缺失：

- 振荡周期 / 振荡频率 / 振幅衰减率（NF-κB / p53 / cell cycle 必需）
- 双稳态检测（apoptosis / cell cycle 必需）
- Hill 系数（ultrasensitivity 必需）
- Adaptation 精度（EGF-EGFR 信号自适应必需）
- 核质比（JAK-STAT / NF-κB / TGF-β / Wnt 必需）
- 分岔参数（决策点分析必需）

且 L318-330 中 `is_transient=True` 时 `half_life` 与 `steady_state` 被强制设为 None，瞬态系统实际只有 7 个有效特征 + 1 个别名（`max_level` = `peak`），并非 9 维。

---

## 4. Reaction IR Issues（Reaction IR 问题）

### 4.1 反应类型覆盖不完整

[reaction_ir.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir.py) L33-44 的 `REACTION_TYPES` 包含 10 类：binding、phosphorylation、dephosphorylation、transcription、translation、degradation、exchange、recruitment、dissociation、transport。

**缺失的关键反应类型**：

- ubiquitination（泛素化）—— p53 / Wnt / NF-κB 必需
- proteasomal_degradation（蛋白酶体降解）—— 与 spontaneous degradation 区分
- nuclear_import / nuclear_export（核输入/输出）—— STAT / NF-κB / SMAD / β-catenin 必需
- complex_formation（复合物形成）—— destruction complex 必需
- cleavage（蛋白切割）—— caspase / Notch 必需
- allostery（变构调控）—— Wnt-Dvl 调控 destruction complex 必需

虽然 `transport` 存在，但未区分核膜运输与膜运输，无法支持核-质两 compartment 建模。

### 4.2 mechanism → reaction 映射存在语义错误

`reaction_ir.py` L55-65 的 `_MECHANISM_TO_REACTION` 映射：

- `"activation": ("phosphorylation", "Michaelis_Menten")` —— 错误。激活不等于磷酸化，可能是变构激活、表达上调、招募等多种形式。强制映射为 phosphorylation 会让所有"activation"边被错误地用磷酸化模板渲染。
- `"inhibition": ("binding", "mass_action")` —— 错误。抑制可能是竞争性抑制、变构抑制、降解加速、转录抑制等，强制映射为 binding + mass_action 会丢失机制语义。

### 4.3 信息损失：graph → equation mapping

`_parse_reaction_equation`（L205-225）解析反应方程，但：

- 仅支持 `"A + B → C"` 简单格式，无法解析化学计量（如 `2 A → A2`）
- 无法标注酶催化（如 `A + E → A + E + B` 中 E 是酶）
- 无 compartment 标注（如 `A_cyto → A_nucleus`）
- 无修饰状态（如磷酸化位点 Ser/Thr/Tyr）

模板渲染时（`Signaling_Cascade_Phos.j2` L92-100 的 `_is_enzyme` 判定）只能基于单条 `reaction_eq` 字符串判断酶角色，无法表达"同一酶在不同通路中催化不同底物"的 context-dependent 活性。

### 4.4 预校验层的局限

`pre_validate_reaction_graph`（L270-413）实现了 Token Boundary Check 与 Conflict Detection，但：

- L325-330 的子串匹配检测（`sp in s and sp != s`）会误报：`"Ras"` 在 `"KRas"` 中是子串，但 `"KRas"` 是不同基因，应被识别为不同物种而非误匹配
- L382-387 的自环检测：`source == target` 时若 mechanism 不是 phosphorylation/dephosphorylation 则报冲突——但自磷酸化（autophosphorylation，如 EGFR 二聚体自磷酸化）的 mechanism 应是 phosphorylation，规则看似合理，但若 LLM 输出 `mechanism="activation"` 描述自磷酸化会被错误判为冲突
- L390-405 的命名碰撞检测：用 `\b` word boundary 正则，但 Python `re` 的 `\b` 不识别 Unicode 生物名（如 `"p65"` vs `"p65_NFkB"`），会产生大量误报或漏报

### 4.5 state_transitions 字段为空扩展点

`ReactionIR.__init__` 的 `state_transitions` 字段（L82）声明为"留作未来扩展"，但全栈无任何代码写入该字段。这导致：

- 蛋白质状态转换（如 EGFR 单体 → 二聚体 → 自磷酸化 → 招募 Grb2）无法表达
- 多步骤级联（如 TGF-β: 受体结合 → SMAD 磷酸化 → SMAD4 异源复合 → 入核）被压扁为平面 reaction 列表

---

## 5. Template System Issues（模板系统问题）

### 5.1 模板-通路覆盖率：1/10

详见 §2.1 与子审计报告。9 个模板对 10 条通路的覆盖情况：

| 通路 | 覆盖状态 | 主模板 |
|------|---------|--------|
| EGFR / RTK | 完整覆盖 | `Signaling_Cascade_Phos.j2` |
| MAPK/ERK | 部分（缺酶饱和） | `Signaling_Cascade_Phos.j2` |
| PI3K-AKT-mTOR | 部分（缺反馈） | `Signaling_Cascade_Phos.j2`（误用） |
| p53 | 不覆盖 | 无 |
| Apoptosis | 不覆盖 | 无 |
| Cell cycle | 不覆盖 | 无 |
| JAK-STAT | 不覆盖 | 无 |
| NF-κB | 部分（仅降解） | `Signaling_Cascade_Phos.j2`（误用） |
| Wnt/β-catenin | 不覆盖 | 无 |
| TGF-β/SMAD | 不覆盖 | 无 |

**覆盖率：1/10（10%）**。声称的 10 条通路覆盖在模板层完全落空。

### 5.2 严重模板-通路不匹配

| 风险 | 后果 |
|------|------|
| Apoptosis 误用 `Signaling_Cascade_Phos.j2` | caspase 是蛋白酶（cleavage 不可逆），模板引入伪可逆性（去磷酸化项），削弱 all-or-none 决策 |
| p53 误用 `PKPD_OneCompartment.j2` | 丢失 Mdm2-p53 反馈环、泛素化、转录延迟；无法产生 p53 脉冲振荡 |
| NF-κB 误用 `Cascade_Activation.j2` | L39-42 静默丢弃 inhibition 边，NF-κB-IκBα 互抑制关系丢失，模型退化为单向级联 |
| Cell cycle 误用 `Simple_Activation.j2` | 单调饱和函数无法产生振子行为；CDK1 自激活 + APC/C 延迟负反馈无法建模 |
| Wnt 误用 `_mechanism_binding.j2` | 仅 A+B↔AB 二元结合，无法表达 destruction complex 内 GSK3β 磷酸化 β-catenin（需 binding+phosphorylation 耦合） |

### 5.3 6 类关键模板完全缺失

详见 §3.2 与子审计报告。重申缺失清单：

1. **Bistable switch template** —— apoptosis / cell cycle / p53 决策点必需
2. **Oscillatory feedback template** —— NF-κB / cell cycle / p53 振荡必需
3. **Transcriptional delay template** —— p53-Mdm2 / JAK-STAT-SOCS / NF-κB-IκBα / TGF-β-SMAD7 必需
4. **Nuclear transport template** —— JAK-STAT / NF-κB / TGF-β / Wnt 必需
5. **Ubiquitination template** —— p53 / Wnt / NF-κB 必需
6. **Destruction complex template** —— Wnt 必需

### 5.4 文档-代码不一致

`Signaling_Cascade_Phos.j2` L58 docstring 仍写：

```
phosphorylation: d[pX]/dt = Vmax * [enzyme] * [X]/(Km+[X]) - k_dephos * [pX]
```

但 `_mechanism_phosphorylation.j2` L3 明确 `TASK 4 修复：移除 Michaelis-Menten`，L29 实际代码为 `_phos_rate = k_phos * src * _sub_conc`（mass-action）。docstring 与实现矛盾，会误导维护者。

### 5.5 Tier 3 组合能力受限

`Signaling_Cascade_Phos.j2` 通过 `{% include %}` 组合 `_cascade_helpers.j2` / `_mechanism_binding.j2` / `_mechanism_phosphorylation.j2`，但：

- 子模块仅限 Tier 3 内部组合，不能被其他 8 个模板复用
- PKPD 系列、Combination、Simple、Cascade 系列之间无组合接口
- "EGFR 信号 + 药物 PK"组合无法实现（PKPD 模板的 target 动力学硬编码 `0.5*(1-effect) - 0.05*target`，不能注入到 Signaling_Cascade_Phos 的 ODE 右端）

### 5.6 模板降级映射风险

`template_selector.py` L51-54 的 `TEMPLATE_FALLBACK_MAP`：

```
"Signaling_Cascade_Phos": "Cascade_Activation",  # 需补质量守恒
```

注释自己承认"需补质量守恒"，但 `Cascade_Activation.j2` 第 39-42 行**静默丢弃非 activation 边**。这意味着降级时会丢失 inhibition 与 binding 边，模型静默失真。

---

## 6. Routing / Agent Issues（路由与 Agent 问题）

### 6.1 架构名不副实

[graph_v3.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/graph_v3.py) L115-124 的 `_FULL_PLAN` 硬编码 8 步顺序：

```
worker_mcp → worker_mechanism → worker_rag → worker_pkpd →
worker_ode → worker_sandbox → worker_validator → worker_report
```

`supervisor`（L385-438）的调度逻辑就是 `next_worker = plan[current_step]`，纯顺序执行。`_route_from_supervisor`（L510-529）也是 `return plan[current_step]`。**Supervisor 不检查 state 内容、不判断上游产出质量、不做条件分支**（除了 clarification 触发）。号称的"动态编排"实际是固定流水线。

### 6.2 三类关键 Agent 缺失

| 缺失 Agent | 影响 |
|-----------|------|
| **Pathway Classifier Agent** | 无 agent 输出结构化通路类别；`n2_mechanistic_planner` 的 `pathway` 字段是自由文本描述，无分类决策 |
| **Pathway Specialist Agents**（按 10 条通路分） | 全部 8 个 worker 是 generalist；apoptosis/caspase 级联、SMAD 核转位、β-catenin 破坏复合物、NICD 切割等通路特异机制无专门处理 |
| **Cross-talk Coordinator** | crosstalk 被列为禁止词；无 cross-talk edge 类型；无 shared species 协调机制 |

### 6.3 PreRouter 决策维度过窄

`graph_v3.py` L130-133 的 `_PreRouterOutput` 只有 `needs_pkpd` / `needs_experiment_evidence` / `reasoning` 三个字段。**PreRouter 只做 PK/PD 二分类，不做通路分类**。

后果：

- 用户问 apoptosis → PreRouter 只判断"是否需要 PK/PD"，不影响后续 worker 的执行
- 所有通路都走同一条 `worker_mechanism → worker_rag → worker_ode → ...` 路径
- 无 pathway-aware routing

### 6.4 误路由场景

| 场景 | 误路由后果 |
|------|-----------|
| 用户问"细胞凋亡通路调控" | `pure_mechanism_keywords`（L307-311）含"通路"、"调控"，会被判为纯机制问题，跳过 PK/PD——对药物诱导凋亡场景错误 |
| 用户问 PI3K-AKT 通路 | `worker_mechanism` 用通用 NER prompt，`template_selector` 无 PI3K 关键词，回退到 LLM 兜底，最终可能选 `Signaling_Cascade_Phos`（误用） |
| 用户问 NF-κB 振荡 | `worker_ode` 无 oscillatory 模板，强制用 `Cascade_Activation`，inhibition 边被静默丢弃 |
| `worker_mechanism` 产出空 `network_json` | Supervisor 仍调度 `worker_rag → worker_ode`；`worker_ode` 在 `edges=[]` 时走 `nodes_v2.py` L1608-1611 无边回退分支，硬编码 `y0=10.0` |
| `auto_fast` 模式下任何药物问题 | `worker_pkpd` L737-749 直接返回空 `pkpd_profile`；`n6_ode_generator` L1252-1256 判 `pkpd_active=False`，不会走 PKPD 模板——与 PreRouter 的 PK/PD 判断无关 |

### 6.5 职责混乱与重叠

**单 agent 承担过多职责**：

- `worker_mechanism`：SBML 加载 + NER + 机制规划 + KG 构建（4 节点合一），无法独立重试或独立触发人在环路
- `worker_report`：特征提取 + 实验 RAG + 文献 RAG + 报告渲染（4 节点合一），特征提取（数值计算）与报告生成（自然语言）混淆
- `worker_sandbox`：仿真执行 + 审计纠错 + 代码重写（`graph_v3.py` L813-851），审计本应是独立 Agent

**职责重叠**：验证职责四处分散——`worker_sandbox` 内的 `node4_audit_and_correct`、`worker_validator`（SBML 验证）、`n6_ode_generator` 内的 `domain_checker.check_ode_code`、`n6_ode_generator` 内的 `pre_validate_reaction_graph`。四套校验机制无统一协调，失败处理路径各不相同（阻断 / warning / 重试）。

**RAG 检索职责分散**：`worker_rag` 做机制 + 参数 RAG，`worker_report` 做实验 + 证据 RAG——四路 RAG 被切分到两个 worker，无法统一缓存或统一过滤。

### 6.6 LangGraph 状态流风险

**Clarification 阻塞**：单个 clarification 最长阻塞 600 秒（`graph_v3.py` L560），期间整个 thread 占用。三个 clarification 触发点全部集中在 `worker_ode` 前，最耗时的 `worker_rag`（单边可达 10s）完全没有干预点。

**全局字典无自动清理**：`_clarification_events` / `_clarification_responses` / `_clarification_stop_events`（L50-52）定义了 `cleanup_clarification_events`（L82-86）但**在 graph_v3.py 内部从未被调用**。thread_id 残留会导致旧事件干扰新请求。

**Sandbox 重试无失败短路**：`graph_v3.py` L805 `while retry_count <= max_retries`，重试耗尽后 `execution_result["status"] = "error"`，但 `worker_sandbox` 仍返回正常 update，supervisor 继续推进到 `worker_validator` → `worker_report`。

**PreRouter 显式清空 30+ 字段**（L216-241）：注释表明存在跨请求数据污染历史。这种"清空"治标不治本，新增字段忘记加入清空列表又会污染。

### 6.7 AGENT_REGISTRY 中残留的孤立 Agent

`supervisor.py` L65-71 的旧 `AGENT_REGISTRY` 包含 `Biology Validator Agent`，但 v3 图中已被 `worker_validator`（SBML 验证）取代。生物学合理性审计退化为 sandbox 内子步骤，独立的 Biology Validator Agent 失踪。

`prompts.py` L307 的 `RAG_SPECIALIST_PROMPT` 在 v3 图中未被引用（`graph_v3.py` import 列表 L20-43 未引入）。孤立 prompt 表明架构演进未清理旧定义。

---

## 7. Missing System Capabilities（缺失的系统能力）

### 7.1 完全缺失的能力

| 能力 | 影响 |
|------|------|
| **Pathway Classification** | 无结构化通路类别输出，所有通路一视同仁处理 |
| **Pathway Specialist Agents** | 9 条非 EGF-EGFR 通路无专门处理 |
| **Cross-talk Coordinator** | 通路间交互完全缺失且被禁止 |
| **Calibration Agent** | `calibration_reference` 角色被检测但无任何下游消费 |
| **Independent Biology Validator** | 生物学合理性审计退化为 sandbox 子步骤 |
| **Sensitivity Analysis** | 全栈无灵敏度分析 |
| **Uncertainty Quantification** | 全栈无不确定性量化 |
| **Parameter Estimation** | 无拟合 / 优化 / MCMC 流程 |
| **Bifurcation Analysis** | 无分岔分析 |
| **Cancer Subtype Modeling** | 无 subtype 分类器、无 mutation profile、无 KRAS G12D / TP53 R175H 等突变建模 |
| **Perturbation Simulation** | 仅 `combo_` 列名前缀支持联合用药；无 KO / CRISPR / dose-response 梯度 |
| **Plugin Architecture** | 5 个硬编码常量（`ALLOWED_PATHWAY_SET` / `CANONICAL_REDUCTION_MAP` / `CORE_SPECIES_SET` / `_REQUIRED_MAPK_CHAIN` / `_INTERACTION_STRENGTH`）阻塞扩展，违反开闭原则 |
| **DDE / SDE Solvers** | 仅 LSODA / solve_ivp，无延迟 / 随机求解器 |
| **Compartmental Modeling** | 无核-质 / 胞内-胞外 / 线粒体 compartment 概念 |
| **Standard Ontology Alignment** | 无 HGNC / UniProt / ChEBI / GO / SBO / Reactome / KEGG ID 映射 |
| **Multi-state Protein Modeling** | 无多位点磷酸化（如 ERK1 双磷酸化 T202/Y204）组合爆炸处理 |

### 7.2 严重退化的能力

| 能力 | 当前状态 | 退化点 |
|------|---------|--------|
| Michaelis-Menten kinetics | 完全移除 | TASK 4 主动改为 mass-action，丢失酶饱和 |
| RAG evidence layer | 字段定义存在 | `doi` / `figure_ref` / `cell_line` 永远为空 |
| RAG experiment layer | collection 存在 | 自动建库中始终为空 |
| Validation Oracle | Track A 可用 | Track B 输出 `error_diff=0` 误导；skipped 默认 `pass=True` |
| DomainChecker | 4 维审查框架 | 仅 EGF-EGFR-MAPK 关键词，9 条通路无约束 |
| KG edge types | 3 类 | 缺 catalysis / binding / transcription / ubiquitination / cleavage |

### 7.3 数据源对齐缺口

| 数据源 | 在线客户端 | 实际灌库 |
|--------|-----------|---------|
| PubMed | 已覆盖 | 已灌库 |
| BioModels | 已覆盖 | 已灌库 |
| KEGG | 已覆盖（`bio_db_client.py` L29-109） | **未灌库** |
| Reactome | 已覆盖（L116-187） | **未灌库** |
| UniProt | 已覆盖（L194-270） | **未灌库** |
| ChEMBL | 已覆盖（L277-341） | **未灌库** |
| ClinicalTrials.gov | 运行时查询 | 不入库 |

`bio_db_client.py` L475-559 提供了 `search_kegg_mechanism` 和 `search_chembl_parameters` 适配方法，但 `build_rag_db.py` 和 `extend_rag_db.py` 均未调用。这意味着 KEGG / Reactome / UniProt / ChEMBL 四个权威源在 RAG 离线库中**完全无数据**，仅在运行时 Node 1.5 在线补充时才会被查询——而在线补充的延迟（单源 1-3s）会显著拖慢 worker_rag。

### 7.4 Reference grounding 缺失层级

- **Model 级 grounding**：存在（`biomodels_reactions.py` L624 `source_model`）
- **ODE 级 grounding**：缺失（生成 ODE 与 SBML `<reaction>` 元素无显式映射）
- **Parameter 级 grounding**：缺失（`k_forward` / `k_reverse` 未关联到 SBML `<parameter id="...">` 原始 ID）
- **Species 级 grounding**：缺失（`collapse_species` 合并多个 SBML 物种为 canonical 节点，未保留原始 species id 列表）
- **PMID 级 grounding**：缺失（parameter schema 中 `pmid` 非必填，BioModels/ChEMBL 来源参数无文献支撑）

---

## 8. Recommended Target Architecture（目标架构，仅结构）

> 仅描述目标架构的角色与职责分层，不展开工程实现。

### 8.1 三层架构

**Layer 1: Intake & Classification Layer（入口与分类层）**

- **Pathway Classifier Agent**：基于用户输入 + MCP 术语 + RAG mechanism 命中，输出结构化 `pathway_class`（10 条通路之一 + cross-talk 标记）。规则优先（关键词 + BIOMD ID 映射），LLM 兜底。
- **Perturbation Interpreter**：解析用户输入中的 drug / KO / mutation / dose 信息，输出结构化 `perturbation_profile`。
- **PreRouter**：基于 `pathway_class` + `perturbation_profile` 选择执行 plan（哪些 worker 参与、顺序、是否需要 cross-talk 协调）。

**Layer 2: Pathway Specialist Layer（通路专家层）**

- **10 个 Pathway Specialist Agents**（按通路分）：
  - EGFR Specialist（已具备，基于 `Signaling_Cascade_Phos.j2`）
  - MAPK Specialist（需补 MM 酶饱和与 ultrasensitivity）
  - PI3K-AKT Specialist（需补 PIP2/PIP3 + 反馈环）
  - p53 Specialist（需 transcription + ubiquitination + delay feedback + bistability）
  - Apoptosis Specialist（需 caspase cascade + MOMP + bistable switch）
  - Cell Cycle Specialist（需 CDK-Cyclin toggle + oscillation + delay）
  - JAK-STAT Specialist（需 transcription + nuclear transport + SOCS feedback）
  - NF-κB Specialist（需 IκBα degradation + oscillation + nuclear transport）
  - Wnt Specialist（需 destruction complex + transcription）
  - TGF-β Specialist（需 SMAD complex + nuclear transport + transcription）
- **Cross-talk Coordinator Agent**：当 `pathway_class` 包含多个通路或 cross-talk 标记时激活，负责：
  - 识别 shared species（如 Ras / AKT / MEK 同时在多通路）
  - 协调多个 Specialist 的 ODE 耦合（cross-inhibition / cross-activation edges）
  - 防止 cross-pathway parameter contamination

**Layer 3: Validation & Calibration Layer（验证与校准层）**

- **Calibration Agent**：用 BioModels reference 对参数做最小二乘 / MCMC 拟合，输出 `calibrated_params` + `confidence_intervals`。
- **Independent Biology Validator**：检查通路特异规则（如 p53-Mdm2 反馈、NF-κB 振荡、cell cycle toggle），独立于 sandbox。
- **Validation Oracle**：严格双轨——Track A 必须运行真实 SBML 仿真，Track B 不得输出 `error_diff=0`；skipped 状态必须 `pass=False` 阻塞流水线。
- **Sensitivity & Uncertainty Agent**：参数扫描 + 全局灵敏度分析（Sobol / Morris）+ 不确定性量化。

### 8.2 横切关注点

- **Standard Ontology Service**：HGNC / UniProt / ChEBI / GO / SBO / Reactome / KEGG ID 解析，作为 species_ontology 的底层。
- **Pathway Tagging Service**：所有 RAG 记录强制 `pathway` 字段；检索时支持通路过滤。
- **Plugin Registry**：通路定义、模板、DomainChecker 规则、Specialist Agent 通过注册表加载，违反开闭原则的硬编码全部移除。
- **Reference Grounding Ledger**：维护 ODE ↔ SBML reaction ↔ parameter ↔ species ↔ PMID 五级映射链。

### 8.3 数据流原则

- 所有参数必须有 PMID / DOI / BioModels ID 来源（reference grounding 硬约束）
- 所有物种必须有 HGNC / UniProt ID（standard ontology 硬约束）
- 所有通路必须有 KEGG / Reactome pathway ID（pathway tagging 硬约束）
- Cross-talk 必须显式建模为 edge，不得静默丢弃

---

## 9. Upgrade Roadmap（升级路线图）

### Phase 1: 单通路正确性修复（去除"伪覆盖"）

**目标**：让 EGF-EGFR-MAPK 通路仿真真正达到科研级，移除虚假能力声明。

- 移除 `biomodels_reactions.py` 的 `ALLOWED_PATHWAY_SET` 硬编码与 `FORBIDDEN_PATHWAY_TERMS` 禁止词列表，改为按通路白名单动态加载
- 修复 `_mechanism_phosphorylation.j2` 的 Michaelis-Menten 退化为 mass-action 的问题（恢复 `Vmax * [E] * [S] / (Km + [S])` 形式）
- 修复 `Cascade_Inhibition.j2` L42-45 的 inhibition/activation 分支逻辑反转
- 修复 `_cascade_helpers.j2` L17-32 静默修改用户参数的问题
- 修复 `Signaling_Cascade_Phos.j2` L58 docstring 与 `_mechanism_phosphorylation.j2` L3 实现矛盾
- 修复 `sbml_validator.py` Track B 输出 `error_diff=0` 误导 + `_skipped_report` 默认 `pass=True` 失效
- 完整覆盖 RAG evidence layer 的 `doi` / `figure_ref` / `cell_line` 字段提取

### Phase 2: 10 通路真实覆盖（重建模板与 Agent）

**目标**：让 10 条通路都具备至少 minimal viable 模板与 DomainChecker 规则。

- 新增 6 类关键模板：bistable switch、oscillatory feedback、transcriptional delay、nuclear transport、ubiquitination、destruction complex
- 引入 Pathway Classifier Agent，输出结构化 `pathway_class`
- 引入 10 个 Pathway Specialist Agents（每个通路一个），替换当前 generalist workers
- DomainChecker 增加通路特异规则：p53-Mdm2 feedback、Wnt destruction complex、NF-κB oscillation、Cell cycle CDK toggle、Apoptosis caspase bistability
- RAG parameter collection 增加 `pathway` 字段，支持按通路过滤检索
- species_ontology 对齐 HGNC / UniProt / ChEBI / GO
- 引入 DDE 求解器（如 JiTCDDE）支持延迟反馈

### Phase 3: 系统能力扩展（cross-talk + 校准 + 不确定性）

**目标**：达到早期科研级声明的能力。

- 引入 Cross-talk Coordinator Agent，支持 shared species 与 cross-inhibition edges
- 引入 Calibration Agent，用 BioModels reference 做参数拟合
- 引入 Sensitivity & Uncertainty Agent（Sobol / Morris 全局灵敏度）
- 引入 Plugin Registry，支持通路 / 模板 / 规则 / Agent 通过配置加载
- 引入 Compartmental Modeling（核-质 / 胞内-胞外 / 线粒体）
- 引入 Perturbation Simulation 完整支持（drug + KO + CRISPR + dose-response 梯度）
- 引入 Cancer Subtype Modeling（mutation profile + subtype 特异参数集）
- RAG 离线库灌入 KEGG / Reactome / UniProt / ChEMBL 数据
- Validation Oracle 升级为严格双轨 + skipped 必阻塞

---

## 10. Risk Analysis（风险分析）

### 10.1 科学风险

| 风险 | 严重度 | 后果 |
|------|--------|------|
| **虚假能力声明** | 极高 | 系统声称覆盖 10 条通路，实际仅 1 条。若用于科研决策，会误导用户认为某通路已被仿真验证 |
| **Cross-talk 被禁止** | 极高 | 癌症信号网络的核心特征被显式过滤，仿真结果无法反映真实生物学 |
| **MM 动力学退化** | 高 | MAPK ultrasensitivity 与 bistability 无法涌现，所有激酶级联被压扁为线性响应 |
| **Cascade_Inhibition 逻辑反转** | 高 | 所有使用该模板的仿真结果方向相反（抑制剂浓度越高靶点生成越多） |
| **inhibition 边被环路打破优先丢弃** | 高 | 负反馈环路（NF-κB-IκBα / p53-Mdm2 / CDK-Cyclin）会被破坏 |
| **Validation Oracle 默认 pass=True** | 高 | 任何 SBML 不可用情况静默通过，仿真结果无质量保证 |
| **evidence 字段永远为空** | 高 | "证据优先"原则在数据层落空，报告中的文献溯源不可信 |
| **parameter 无 pathway 字段** | 高 | cross-pathway contamination 不可防范，参数可能错误注入其他通路模型 |

### 10.2 工程风险

| 风险 | 严重度 | 后果 |
|------|--------|------|
| **5 个硬编码常量阻塞扩展** | 高 | 添加新通路需修改 5 处常量，违反开闭原则，易引入回归 |
| **clarification 全局字典无自动清理** | 中 | thread_id 残留导致旧事件干扰新请求 |
| **Sandbox 重试无失败短路** | 中 | 仿真失败仍继续推进到报告生成，浪费计算资源 |
| **PreRouter 清空 30+ 字段治标不治本** | 中 | 新增字段忘记清空又会污染 |
| **worker_mechanism 承担 4 节点** | 中 | 单点失败影响大，无法独立重试 |
| **四套校验机制无统一协调** | 中 | 失败处理路径不一致，难调试 |
| **BM25 索引每次重建** | 中 | 数据量增大时性能显著下降 |

### 10.3 信任风险

| 风险 | 严重度 | 后果 |
|------|--------|------|
| **Track B 输出 error_diff=0** | 高 | 下游消费者无法区分"真实仿真差异为 0"与"未做仿真" |
| **species_ontology 字符串匹配** | 中 | "Ras" 会匹配 "KRas" / "NRas"，物种覆盖度被高估 |
| **RAG species 字段硬编码 "Human"** | 中 | 与文献实际物种无关，跨物种参数选择失真 |
| **SBML 参数 species="Unknown"** | 中 | species 过滤下永远无法被检索，造成死数据 |
| **docstring 与实现矛盾** | 低 | 误导维护者，但已通过审计暴露 |

### 10.4 升级风险

| 风险 | 严重度 | 缓解策略 |
|------|--------|---------|
| **Phase 2 引入 10 Specialist 成本高** | 高 | 可按通路优先级分批引入（EGFR 已完成 → MAPK → PI3K → p53 → Apoptosis → 其他） |
| **DDE / SDE 求解器集成复杂** | 中 | 仅在 Phase 2 引入 DDE，SDE 推迟到 Phase 3 |
| **Plugin Registry 重构影响面大** | 中 | 必须在 Phase 3 完成，Phase 1/2 可继续硬编码 |
| **RAG 重建（加 pathway 字段）需重灌库** | 中 | 写迁移脚本，旧记录按 mechanism/species 推断 pathway |
| **Specialist Agent 职责边界难划分** | 中 | cross-talk 场景下多个 Specialist 需协调，Coordinator 必须先就位 |

### 10.5 风险优先级矩阵

```
极高：虚假能力声明 / Cross-talk 被禁止
高  ：MM 退化 / Cascade_Inhibition 反转 / inhibition 被丢弃 / Oracle 默认 pass / evidence 空 / parameter 无 pathway
中  ：硬编码阻塞 / clarification 残留 / 无失败短路 / Track B 误导 / species 字符串匹配
低  ：docstring 矛盾
```

**建议**：Phase 1 必须在系统对外宣称"10 通路仿真"之前完成；Phase 2 必须在引入任何癌症决策支持功能之前完成；Phase 3 必须在对外提供科研服务之前完成。

---

## 附录：审计证据索引

| 审计维度 | 关键文件 | 关键行号 |
|---------|---------|---------|
| 通路硬编码 | `biomodels_reactions.py` | L35, L38-42 |
| MM 退化 | `_mechanism_phosphorylation.j2` | L3, L29 |
| Cascade_Inhibition 反转 | `Cascade_Inhibition.j2` | L21-24, L42-45 |
| 静默参数修改 | `_cascade_helpers.j2` | L17-32 |
| Oracle 默认 pass | `sbml_validator.py` | L548-550, L581-594 |
| Evidence 字段为空 | `extend_rag_db.py` | L450-459 |
| Parameter 无 pathway | `rag_collections.py` | L322-332 |
| 环路打破丢 inhibition | `kg_builder.py` | L17-21, L105-118 |
| 本体未对齐 | `species_ontology.py` | L33-45, L94-157 |
| 固定流水线 | `graph_v3.py` | L115-124, L422 |
| PreRouter 二分类 | `graph_v3.py` | L130-133 |
| Agent 职责混乱 | `graph_v3.py` | L599-673, L1012-1046 |
| KG 边类型不足 | `kg_builder.py` | L17-21 |
| 9 维特征通用 | `feature_extractor.py` | L20-32 |
| 数据源未灌库 | `bio_db_client.py` | L475-559 |

---

**审计声明**：本报告基于对 bio-dynamics-agent 系统代码的逐行核查，所有结论均可在 cited file 与 line number 处验证。审计未发现任何"虚假正面"——所有声称的能力缺失均有代码证据支撑。审计未涉及运行时行为验证（如实际跑一次仿真），建议在 Phase 1 修复后补做端到端验证。

报告版本：v1.0
审计日期：2026-07-05
审计范围：backend/app/ 全部模块 + ode_templates/ 全部模板 + scripts/ 建库脚本
