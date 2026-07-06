# BioDynamics Agent v4 — Scientific Architecture Design Document

> **文档性质**：科学架构设计文档（非工程实现文档）
> **基于**：BioDynamics_Agent_严格结构审计报告_v1.0.md
> **目标**：重新定义 BioDynamics Agent 为真正的 Cancer Signaling Network Simulation Platform
> **约束**：只讨论 Scientific Architecture，不涉及代码 / FastAPI / LangGraph / Python 实现
> **版本**：v4.0
> **日期**：2026-07-05

---

## 设计原则（Design Principles）

### 第一原则：Scientific Fidelity 优先于 Engineering Convenience

任何为了简化代码而破坏真实生物机制的设计全部废弃。当科学真实性与工程便利冲突时，以科学真实性为准。**审计报告 §3.1 中 `_mechanism_phosphorylation.j2` 为"修复"而主动移除 Michaelis-Menten 是典型的工程便利破坏科学真实性，v4 严格禁止此类退化。**

### 第二原则：Pathway First，而非 ODE First

整个系统围绕 **Pathway** 组织，不是围绕模板组织。Pathway 是一等公民，ODE 是 Pathway 的派生产物。**审计报告 §2.1 中 `ALLOWED_PATHWAY_SET = frozenset({"EGF_EGFR_MAPK"})` 是 ODE-First 思维的极端表现——把通路当成模板的附属品，v4 反转这一关系。**

### 第三原则：Mechanism First，Reaction 不是字符串

Reaction 是真实生物机制，必须表达 phosphorylation / dephosphorylation / binding / dissociation / dimerization / ubiquitination / degradation / transcription / translation / nuclear translocation / cytoplasm translocation / complex formation / cleavage / GTP/GDP exchange / inhibition / activation / sequestration 等 17 类机制。**审计报告 §4.1 中 Reaction IR 缺失 ubiquitination / nuclear transport / cleavage / destruction complex 等 6 类关键反应类型，v4 必须完整覆盖。**

### 第四原则：SBML 是唯一 Ground Truth

任何模板、任何 Reaction、任何 ODE 都必须可以映射回 SBML。SBML 不是附件，而是系统的唯一真源。**审计报告 §7.4 中 Reference grounding 五级缺失（ODE 级 / Parameter 级 / Species 级 / PMID 级均缺失），v4 必须建立五级映射链。**

### 第五原则：Pathway 不是单一路径，必须支持 Network

未来必须支持 EGFR → MAPK → PI3K → mTOR → NF-κB → p53 → Cell Cycle → Apoptosis 之间的 Cross-talk。**审计报告 §2.2 中 crosstalk 被显式列为禁止词是生物学错误，v4 必须 explicit 支持 cross-talk。**

### 第六原则（新增）：Time-Space Dimension Principle

癌症信号网络不仅是拓扑网络，还是**时间分层**（磷酸化 min / 转录 h / 细胞周期 h）和**空间区室**（胞质-核-线粒体）的网络。v4 必须在设计层面强制多时间尺度与空间区室的真实性。**审计报告 §3.2 中无 DDE、无 Compartmental Modeling、无核-质转运是数学与空间维度的双重缺失，v4 必须从架构层面内置这两维。**

### 第七原则（追加）：组合机制与状态机

复杂生物学结构（如 Wnt 的 destruction complex 需要 binding+phosphorylation+ubiquitination 三步耦合）必须通过组合机制表达，不能压扁为单一 reaction。蛋白质状态转换（如 EGFR 单体→二聚体→磷酸化→招募 Grb2）必须通过状态机表达，不能压扁为平面 reaction 列表。**审计报告 §4.3 中 graph→equation mapping 信息损失、§4.5 中 state_transitions 字段为空扩展点，v4 必须在 Reaction IR 层面解决。**

### 第八原则（追加）：溯源与通路隔离

任何参数、文献、实验数据必须携带 `pathway_tag` 和 `provenance_id`，RAG 检索必须支持严格的通路隔离防污染。**审计报告 §5（RAG）中 parameter 无 pathway 字段、evidence 字段为空、cross-pathway contamination 不可防范，v4 必须在设计层面强制规定。**

### 第九原则（追加）：每个设计必须回应 v1.0 审计报告

为确保不是凭空构想，v4 每个 Part 必须简述如何解决了审计报告中的具体问题（如 MM 退化、inhibition 被丢弃、Oracle 默认 pass 等）。**这是元原则，约束设计过程本身。**

---

## Part 1: Scientific Vision

### 1.1 这个系统到底是什么

BioDynamics v4 是一个 **Cancer Signaling Network Simulation Platform**——专为早期科研设计的癌症信号网络仿真平台。它不是一个通用 ODE Agent，不是一个模板引擎，不是一个 RAG 问答系统。

它的核心身份是：**一个能用机制真实、参数可溯、多时间尺度、空间区室化、可验证的方式，对癌症信号网络进行假设驱动（hypothesis-driven）仿真的科研工具。**

### 1.2 解决什么科研问题

v4 解决以下五类科研问题：

1. **通路动力学重建**：给定一条通路（如 p53-Mdm2 反馈），能否重建其脉冲振荡行为？
2. **Cross-talk 影响**：当 EGFR 信号与 PI3K-AKT 信号交叉时，AKT 对 Raf Ser259 的抑制如何改变 MAPK 输出？
3. **Perturbation 预测**：给定一个药物（如 MEK 抑制剂），能否预测其对下游 ERK 磷酸化与细胞周期进程的影响？
4. **Mutation Phenotype**：给定一个突变（如 KRAS G12D），能否预测其对 RAS-MAPK 通路持续激活的贡献？
5. **Hypothesis Generation**：基于仿真结果，能否生成可验证的实验假设（如"抑制 Wnt destruction complex 会导致 β-catenin 累积，进而激活 Cyclin D1 转录"）？

### 1.3 哪些问题不解决

明确边界：

- **不解决临床决策**：v4 不提供临床诊断或治疗建议，仅提供科研级仿真
- **不解决单细胞分辨率**：v4 是群体平均（population-level）模型，不做单细胞随机建模（推迟到 v6）
- **不解决全基因组规模**：v4 聚焦 10 条核心通路，不做 whole-genome scale modeling
- **不解决空间异质性**：v4 是 compartmental（核-质-线粒体），不做 tissue-scale spatial PDE
- **不解决药物发现**：v4 不做 virtual screening，不做 docking，不做 ADMET
- **不解决实时数据整合**：v4 不接 live sequencing data，不做 patient-specific modeling（推迟到 v6）

### 1.4 系统边界

```
┌─────────────────────────────────────────────────────────┐
│  v4 系统边界                                              │
├─────────────────────────────────────────────────────────┤
│  输入：                                                  │
│  - 用户科研问题（自然语言）                                │
│  - 可选：SBML 模型 ID / 突变 / 药物 / 剂量                │
│  - 可选：实验观测数据（用于校准）                          │
├─────────────────────────────────────────────────────────┤
│  输出：                                                  │
│  - 仿真结果（时间序列 + 通路状态）                        │
│  - 科学特征（峰值/振荡/双稳态/灵敏度）                    │
│  - 假设列表（可验证的实验预测）                            │
│  - 溯源链（ODE ← Reaction ← SBML ← PMID）                │
│  - 验证报告（与 BioModels 对齐程度）                      │
├─────────────────────────────────────────────────────────┤
│  不输出：                                                │
│  - 临床诊断                                              │
│  - 治疗方案                                              │
│  - 药物推荐                                              │
└─────────────────────────────────────────────────────────┘
```

### 1.5 回应 v1.0 审计报告

v4 的 Vision 直接回应审计报告 §1 Executive Summary 的核心矛盾：**系统声称 10 通路实际仅 1 通路**。v4 通过明确系统身份（Cancer Signaling Network Simulation Platform，非通用 ODE Agent）与系统边界（10 通路 + cross-talk，不做临床决策），从根本上消除"虚假能力声明"的根源。审计报告指出的"通路覆盖 1/10"在 v4 中不再是"声明与实现不符"的问题，而是"v4 必须实现 10 通路"的硬约束。

---

## Part 2: Scientific Stack

### 2.1 八层科学栈

v4 的科学栈从下到上分为八层，每层有明确的输入、输出、边界与依赖：

```
┌─────────────────────────────────────────────────────────┐
│  Layer 8: Hypothesis Layer（假设层）                      │
│  ↑ 输入：仿真结果 + 科学特征 + 验证报告                    │
│  ↓ 输出：可验证假设列表 + 实验设计建议                     │
├─────────────────────────────────────────────────────────┤
│  Layer 7: Validation Layer（验证层）                      │
│  ↑ 输入：仿真结果 + SBML reference                        │
│  ↓ 输出：验证报告 + 置信度评分                            │
├─────────────────────────────────────────────────────────┤
│  Layer 6: Simulation Layer（仿真层）                      │
│  ↑ 输入：ODE 系统 + 参数 + 扰动                           │
│  ↓ 输出：时间序列 + 稳态 + 振荡 + 双稳态                  │
├─────────────────────────────────────────────────────────┤
│  Layer 5: ODE Generation Layer（ODE 生成层）              │
│  ↑ 输入：Reaction Graph + 动力学类型 + 参数               │
│  ↓ 输出：可执行 ODE 系统（含 DDE / SDE）                  │
├─────────────────────────────────────────────────────────┤
│  Layer 4: Mechanism Layer（机制层）                       │
│  ↑ 输入：Reaction IR + 通路上下文                         │
│  ↓ 输出：动力学类型选择 + 约束检查                         │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Reaction IR Layer（反应中间表示层）              │
│  ↑ 输入：Pathway Graph + SBML reference                   │
│  ↓ 输出：结构化反应图（含状态机）                          │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Pathway Graph Layer（通路图谱层）                │
│  ↑ 输入：用户问题 + Biological Knowledge                   │
│  ↓ 输出：通路拓扑 + cross-talk edges                      │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Biological Knowledge Layer（生物知识层）          │
│  ↑ 输入：用户问题 + 标准本体查询                          │
│  ↓ 输出：标准化的物种 / 通路 / 机制定义                    │
└─────────────────────────────────────────────────────────┘
```

### 2.2 各层职责

#### Layer 1: Biological Knowledge Layer（生物知识层）

- **职责**：提供标准化的物种、通路、机制定义。所有物种必须对齐 HGNC（基因）/ UniProt（蛋白）/ ChEBI（化学实体）/ GO（功能）/ SBO（系统生物学本体）。所有通路必须对齐 KEGG / Reactome pathway ID。所有机制必须对齐 SBO term（如 SBO:0000216 = phosphorylation）。
- **输入**：用户问题中的实体名 + 标准 API 查询
- **输出**：标准化实体定义（含 HGNC ID / UniProt accession / KEGG pathway ID / SBO term）
- **边界**：不生成 ODE，不选择模板，仅提供本体对齐
- **依赖**：HGNC / UniProt / ChEBI / GO / SBO / KEGG / Reactome 在线 API + 离线缓存

#### Layer 2: Pathway Graph Layer（通路图谱层）

- **职责**：构建通路拓扑图（nodes + edges + cross-talk edges）。每个 node 必须携带 Layer 1 的标准化 ID；每条 edge 必须携带 mechanism type + pathway tag + provenance。
- **输入**：用户问题 + Layer 1 标准化实体 + RAG mechanism 检索
- **输出**：Pathway Graph（含 cross-talk edges，标记 shared species）
- **边界**：不决定动力学类型，不生成 Reaction IR
- **依赖**：Layer 1 + RAG mechanism collection

#### Layer 3: Reaction IR Layer（反应中间表示层）

- **职责**：将 Pathway Graph 转换为结构化 Reaction IR。支持 17 类机制（见 Part 4）。支持组合机制（如 destruction complex = binding + phosphorylation + ubiquitination）。支持蛋白质状态机（如 EGFR 单体→二聚体→磷酸化）。
- **输入**：Pathway Graph + SBML reference（如有）
- **输出**：Reaction IR（含 species / reactions / state_machines / compartments / constraints）
- **边界**：不选择动力学类型，不生成 ODE
- **依赖**：Layer 2 + SBML parser

#### Layer 4: Mechanism Layer（机制层）

- **职责**：为每条 Reaction 选择合适的动力学类型（mass-action / Michaelis-Menten / Hill / Boolean / hybrid）。基于机制语义与参数可用性决策。强制保留 Michaelis-Menten 用于酶催化反应。
- **输入**：Reaction IR + RAG parameter 检索
- **输出**：带动力学类型的 Reaction IR + 约束检查报告
- **边界**：不生成 ODE 代码
- **依赖**：Layer 3 + RAG parameter collection

#### Layer 5: ODE Generation Layer（ODE 生成层）

- **职责**：将带动力学类型的 Reaction IR 渲染为可执行 ODE 系统。支持 ODE / DDE / SDE / hybrid。支持多 compartment（核-质-线粒体）。支持参数注入（含 provenance）。
- **输入**：带动力学类型的 Reaction IR + 参数 + 初始条件
- **输出**：可执行 ODE 系统（数学对象，非代码字符串）
- **边界**：不求解，仅生成
- **依赖**：Layer 4 + 参数库

#### Layer 6: Simulation Layer（仿真层）

- **职责**：求解 ODE 系统。支持 time-scale 分层（磷酸化 min / 转录 h / 细胞周期 h）。支持 steady state / oscillation / bistability / dose response / parameter scan / global sensitivity / local sensitivity / virtual knockout / drug perturbation。
- **输入**：ODE 系统 + 仿真配置（时长 / 求解器 / 扰动）
- **输出**：时间序列 + 科学特征 + 数值稳定性报告
- **边界**：不验证生物学合理性，仅求数值解
- **依赖**：Layer 5 + 数值求解器（LSODA / DDE / SSA）

#### Layer 7: Validation Layer（验证层）

- **职责**：验证仿真结果的科学合理性。建立 Validation Pyramid（见 Part 7）。严格双轨：Track A 必须运行真实 SBML 仿真对比，Track B 不得输出 error_diff=0；skipped 状态必须 pass=False 阻塞流水线。
- **输入**：仿真结果 + SBML reference + 通路特异规则
- **输出**：验证报告 + 置信度评分 + 失败短路信号
- **边界**：不生成假设，仅验证
- **依赖**：Layer 6 + SBML reference + DomainChecker 规则

#### Layer 8: Hypothesis Layer（假设层）

- **职责**：基于仿真结果 + 科学特征 + 验证报告，生成可验证的实验假设。每个假设必须包含：预测 / 实验设计 / 验证方式 / 预期结果。
- **输入**：仿真结果 + 科学特征 + 验证报告
- **输出**：假设列表（每个含预测 + 实验设计 + 验证方式）
- **边界**：不做实验，仅生成假设
- **依赖**：Layer 7 + 文献检索

### 2.3 回应 v1.0 审计报告

v4 八层栈直接回应审计报告的多项缺陷：

- **Layer 1 回应 §2.6 本体未对齐**：强制 HGNC / UniProt / ChEBI / GO / SBO 对齐
- **Layer 3 回应 §4 Reaction IR 问题**：支持 17 类机制 + 组合机制 + 状态机
- **Layer 4 回应 §3.1 MM 退化**：强制保留 Michaelis-Menten 用于酶催化
- **Layer 5 回应 §3.2 无 DDE/SDE + §3.6 无 compartment**：内置 DDE / SDE / multi-compartment
- **Layer 6 回应 §3.7 9 维特征通用**：扩展科学特征（振荡 / 双稳态 / 灵敏度）
- **Layer 7 回应 §7.2 Validation Oracle 退化**：严格双轨 + skipped 必阻塞
- **Layer 8 回应 §7.1 缺失 Hypothesis Agent**：新增假设生成层

审计报告 §6.1 的"固定流水线"问题在 v4 中被八层栈的明确职责分离取代——每层有独立输入输出与边界，不再是单 worker 承担 4 节点的混乱结构。

---

## Part 3: Pathway Architecture

### 3.1 通路模块化设计原则

v4 中每条通路不是单一模板，而是由 **5 个模块** 组成：

1. **Core Module**：通路核心拓扑与机制（如 EGFR 的 ligand-binding → dimerization → autophosphorylation → Shc/Grb2/SOS 招募 → Ras-Raf-MEK-ERK 级联）
2. **Feedback Module**：通路内负反馈 / 正反馈（如 MAPK 的 SOS 反馈抑制、p53-Mdm2 互反馈）
3. **Cross-talk Module**：与其他通路的交叉点（如 AKT→Raf Ser259 抑制、Ras→PI3K 直接激活）
4. **Perturbation Module**：药物 / KO / 突变对通路的影响（如 Gefitinib 抑制 EGFR、KRAS G12D 持续激活）
5. **Validation Module**：通路特异的验证规则与 benchmark（如 pEGFR 5-10 min 达峰、MAPK >10x 放大、NF-κB 振荡周期 1-2h）

**为什么这样拆**：

- Core 保证通路主干完整性，避免审计报告 §2.4 的"通路盲化"
- Feedback 独立模块化，避免审计报告 §3.2 的"无 feedback loop 动态"
- Cross-talk 独立模块化，避免审计报告 §2.2 的"cross-talk 被禁止"
- Perturbation 独立模块化，避免审计报告 §7.1 的"无 Perturbation Simulation 完整支持"
- Validation 独立模块化，避免审计报告 §7.2 的"Validation Oracle 形同虚设"

### 3.2 十条通路架构

#### 通路 1: EGFR / RTK Signaling

- **Core**：EGF + EGFR → EGF-EGFR → dimerization → autophosphorylation → pEGFR → Shc/Grb2/SOS 招募 → Ras-GTP → Raf-MEK-ERK 级联
- **Feedback**：ERK 磷酸化 SOS Ser/Thr 导致 SOS 失活（负反馈）；EGFR 内吞与降解（负反馈）
- **Cross-talk**：pEGFR → PI3K 激活（指向通路 3）；pERK → ELK1 → Fos 转录（指向通路 6）；AKT → Raf Ser259 抑制（来自通路 3）
- **Perturbation**：Gefitinib / Erlotinib（EGFR TKI）；Cetuximab（抗 EGFR 抗体）；EGFR vIII 突变
- **Validation**：pEGFR 5-10 min 达峰（Schoeberl 2002）；MAPK >10x 放大；EGFR 内吞半衰期 10-15 min

#### 通路 2: MAPK / ERK Pathway

- **Core**：Ras-GTP → Raf (MKKK) → MEK (MKK) → ERK (MAPK) 三级级联，含双磷酸化（MEK Ser218/222，ERK Thr202/Tyr204）
- **Feedback**：ERK → SOS 负反馈；ERK → Raf Ser259 反馈抑制（与通路 1 共享）
- **Cross-talk**：ERK → RSK → TSC2 抑制（指向 mTOR，通路 3）；ERK → Bim 磷酸化导致降解（指向 Apoptosis，通路 5）；ERK → Cyclin D 转录（指向 Cell Cycle，通路 6）
- **Perturbation**：Vemurafenib / Dabrafenib（BRAF 抑制剂）；Trametinib（MEK 抑制剂）；BRAF V600E 突变
- **Validation**：零阶 ultrasensitivity（Markevich 2004）；Hill 系数 >2；稳态传递信号放大

#### 通路 3: PI3K-AKT-mTOR Pathway

- **Core**：pRTK → PI3K → PIP2→PIP3 → PDK1 + mTORC2 → AKT phosphorylation → TSC2 抑制 → Rheb-GTP → mTORC1 → S6K1 / 4E-BP1
- **Feedback**：S6K1 → IRS1 Ser 磷酸化抑制（负反馈环）；mTORC1 → ULK1 抑制（自噬抑制）
- **Cross-talk**：AKT → Raf Ser259 抑制（指向通路 2）；AKT → Bad 磷酸化抑制（指向通路 5）；mTORC1 → HIF-1α 翻译增强
- **Perturbation**：Rapamycin / Everolimus（mTOR 抑制剂）；BKM120 / Idelalisib（PI3K 抑制剂）；PTEN loss-of-function
- **Validation**：pAKT 30-60 min 达峰；PIP3 与 PIP2 质量守恒；S6K1 磷酸化晚于 AKT 30 min

#### 通路 4: p53 Tumor Suppressor Pathway

- **Core**：DNA damage → ATM/ATR → p53 phosphorylation → p53 tetramerization → nuclear translocation → transcription（Mdm2 / p21 / Bax / PUMA / Noxa）
- **Feedback**：p53 → Mdm2 转录 → Mdm2 E3 泛素连接酶 → p53 ubiquitination → proteasomal degradation（负反馈环）；ATM → p53 与 Mdm2 双向调控
- **Cross-talk**：p53 → Bax / PUMA（指向 Apoptosis，通路 5）；p53 → p21（指向 Cell Cycle，通路 6，p21 抑制 Cyclin-CDK）；AKT → Mdm2 磷酸化促进 p53 降解（来自通路 3）
- **Perturbation**：Nutlin-3（Mdm2 抑制剂）；PRIMA-1（p53 重激活剂）；TP53 R175H / R273H 突变
- **Validation**：p53 脉冲振荡（Lev Bar-Or 2000），周期 5-7h；Mdm2 转录延迟 1-2h；p21 转录延迟 2-4h

#### 通路 5: Apoptosis (Intrinsic / Extrinsic)

- **Core**：
  - Intrinsic：DNA damage / stress → BH3-only (Bim/PUMA/Noxa) → Bax/Bak 寡聚 → MOMP → Cytochrome c 释放 → Apaf-1 → Caspase-9 → Caspase-3 → PARP cleavage
  - Extrinsic：FasL/Fas → DISC → Caspase-8 → Caspase-3（直接） + Bid → Bax（指向 intrinsic）
- **Feedback**：Caspase-3 → Caspase-6 → Caspase-8 正反馈（executioner→initiator amplification，bistable switch）；XIAP 抑制 Caspase-3/9（负反馈）
- **Cross-talk**：p53 → Bax/PUMA/Noxa（来自通路 4）；AKT → Bad 磷酸化抑制（来自通路 3）；ERK → Bim 磷酸化降解（来自通路 2）；NF-κB → anti-apoptotic genes（Bcl-2 / Bcl-xL，来自通路 8）
- **Perturbation**：ABT-199 / Venetoclax（Bcl-2 抑制剂）；Z-VAD-FMK（pan-caspase 抑制剂）；BCL2 overexpression
- **Validation**：Caspase-3 activation 是 bistable（all-or-none）；MOMP 是 point-of-no-return；Cytochrome c 释放应在 Caspase-3 activation 前 5-15 min

#### 通路 6: Cell Cycle (CDK Network)

- **Core**：Cyclin D-CDK4/6（G1 早期） → Cyclin E-CDK2（G1/S） → Cyclin A-CDK2（S） → Cyclin A-CDK1（G2） → Cyclin B-CDK1（M） → APC/C-Cdc20（M 后期） → Cyclin B 降解
- **Feedback**：Cyclin B-CDK1 → APC/C-Cdc20 激活 → Cyclin B 降解（延迟负反馈振子）；p53 → p21 → Cyclin-CDK 抑制（来自通路 4）；Rb → E2F 抑制（G1/S toggle）
- **Cross-talk**：ERK → Cyclin D 转录（来自通路 2）；p53 → p21（来自通路 4）；AKT → p21 cytoplasmic sequestration（来自通路 3）；MYC → Cyclin D/E 转录
- **Perturbation**：Palbociclib（CDK4/6 抑制剂）；Roscovitine（CDK1/2 抑制剂）；CDKN2A loss（p16 缺失）
- **Validation**：Cyclin-CDK toggle 是 bistable；Cyclin B-APC/C 振荡周期 8-12h（in vitro）；Rb-E2F 是 G1/S switch

#### 通路 7: JAK-STAT Signaling

- **Core**：Cytokine（IL-6 / IFN-γ / EPO）+ 受体 → JAK 磷酸化 → STAT 招募 → STAT 酪氨酸磷酸化 → STAT 二聚化 → nuclear import → 转录靶基因（SOCS / CIS / Bcl-xL / IRF）
- **Feedback**：SOCS / CIS → JAK 或受体抑制（转录延迟负反馈）；STAT → PIAS → STAT nuclear export
- **Cross-talk**：STAT3 → Bcl-xL / Mcl-1（指向 Apoptosis，通路 5）；STAT3 → Cyclin D1 / MYC（指向 Cell Cycle，通路 6）；EGFR → STAT3（指向通路 1）
- **Perturbation**：Ruxolitinib（JAK1/2 抑制剂）；Tofacitinib（JAK 抑制剂）；STAT3 S727F 突变
- **Validation**：pSTAT5 5-15 min 达峰；SOCS mRNA 30-60 min 延迟；STAT5 核质比振荡（单脉冲）

#### 通路 8: NF-κB Pathway

- **Core**：TNF / IL-1 / LPS → 受体 → IKK 复合物激活 → IκBα phosphorylation → IκBα ubiquitination → IκBα proteasomal degradation → NF-κB (p65/p50) 释放 → nuclear import → 转录靶基因（IκBα / A20 / TNF / Bcl-2）
- **Feedback**：NF-κB → IκBα 转录 → IκBα 蛋白合成 → NF-κB nuclear export（延迟负反馈振荡）；NF-κB → A20 → IKK 抑制（双负反馈）
- **Cross-talk**：NF-κB → Bcl-2 / Bcl-xL（指向 Apoptosis，通路 5）；NF-κB → Cyclin D1（指向 Cell Cycle，通路 6）；AKT → IKK 激活（来自通路 3）；p53 → NF-κB 抑制（pro-apoptotic vs anti-apoptotic）
- **Perturbation**：Bortezomib（蛋白酶体抑制剂）；IKK-16（IKK 抑制剂）；NFKBIA loss
- **Validation**：NF-κB 核振荡周期 1-2h（Nelson 2004）；IκBα 转录延迟 30-60 min；振荡持续 6-20h

#### 通路 9: Wnt / β-catenin Pathway

- **Core**：
  - Off 状态：Axin-APC-GSK3β-CK1 destruction complex → β-catenin phosphorylation → β-catenin ubiquitination → proteasomal degradation
  - On 状态：Wnt + Frizzled + LRP5/6 → Dvl 招募 → LRP6 phosphorylation → Axin 招募到信号体 → destruction complex 解离 → β-catenin 累积 → nuclear import → TCF/LEF 转录（Cyclin D1 / MYC / Axin2）
- **Feedback**：β-catenin → Axin2 转录 → destruction complex 重建（负反馈）；β-catenin → negative regulators（DKK / sFRP）分泌
- **Cross-talk**：β-catenin → Cyclin D1 / MYC（指向 Cell Cycle，通路 6）；β-catenin → survivin（指向 Apoptosis，通路 5）；PI3K-AKT → GSK3β Ser9 抑制（来自通路 3，间接稳定 β-catenin）
- **Perturbation**：XAV939（tankyrase 抑制剂，稳定 Axin）；LGK974（PORCN 抑制剂，阻断 Wnt 分泌）；CTNNB1 S45F 突变
- **Validation**：β-catenin 稳态水平在无 Wnt 时 <10 nM；Axin2 mRNA 在 Wnt 刺激后 1-2h 达峰；destruction complex 三步耦合（binding→phosphorylation→ubiquitination）

#### 通路 10: TGF-β / SMAD Pathway

- **Core**：TGF-β + TβRII → TβRI 招募 → TβRI 磷酸化 → R-SMAD (Smad2/3) 磷酸化 → Co-SMAD (Smad4) 异源复合 → nuclear import → 转录靶基因（PAI-1 / p15 / p21 / SMAD7）
- **Feedback**：SMAD7 → TβRI 抑制（负反馈）；SMAD → SMURF → R-SMAD ubiquitination
- **Cross-talk**：TGF-β → p15 / p21（指向 Cell Cycle，通路 6，生长抑制）；TGF-β → Bim / PUMA（指向 Apoptosis，通路 5）；TGF-β → PI3K-AKT（EMT 时双向）；ERK → Smad linker region 磷酸化（来自通路 2）
- **Perturbation**：Galunisertib（TβRI 抑制剂）；SB431542（ALK5 抑制剂）；SMAD4 loss
- **Validation**：pSmad2 5-15 min 达峰；pSmad2-Smad4 复合物核累积 15-30 min；SMAD7 mRNA 30-60 min 延迟

### 3.3 回应 v1.0 审计报告

v4 的 5 模块 × 10 通路设计直接回应审计报告 §2.3 的"关键机制缺失清单"：

| 审计缺失 | v4 解决方式 |
|---------|------------|
| MM 酶饱和（MAPK ultrasensitivity） | 通路 2 Core 明确双磷酸化 + MM 动力学 |
| p53 缺 transcription + ubiquitination + bistability | 通路 4 Feedback 明确 p53-Mdm2 转录延迟负反馈 + ubiquitination |
| Apoptosis 缺 caspase cascade + bistable + MOMP | 通路 5 Core 明确 MOMP + Caspase 级联，Feedback 明确 bistable switch |
| Cell cycle 缺 bistable + oscillation + delay | 通路 6 Feedback 明确 Cyclin B-APC/C 延迟负反馈振子 + Rb-E2F toggle |
| JAK-STAT 缺 transcription + nuclear transport | 通路 7 Core 明确 nuclear import + SOCS 转录延迟 |
| NF-κB 缺 oscillation + IκBα degradation + nuclear transport | 通路 8 Core + Feedback 完整覆盖 |
| Wnt 缺 destruction complex | 通路 9 Core 明确三步耦合 |
| TGF-β 缺 transcription + nuclear transport | 通路 10 Core 明确 R-SMAD-CoSMAD 复合 + nuclear import |
| Cross-talk 被禁止 | 每条通路均有 Cross-talk Module，显式列出交叉点 |

审计报告 §6.2 的"Pathway Specialist 缺失"在 v4 中通过 10 个通路各自的 5 模块化设计解决——每条通路有自己的 Core / Feedback / Cross-talk / Perturbation / Validation，不再是 generalist worker 一视同仁处理。

---

## Part 4: Reaction IR v2

### 4.1 为什么当前 IR 不适合癌症网络

v1.0 审计报告 §4 指出当前 Reaction IR 的核心缺陷：

1. **机制类型不全**（§4.1）：缺 ubiquitination / nuclear transport / cleavage / destruction complex 等 6 类关键反应类型
2. **语义错误映射**（§4.2）：`activation → phosphorylation` 强制映射丢失激活的多样性；`inhibition → binding` 强制映射丢失抑制的多样性
3. **信息损失**（§4.3）：graph→equation mapping 无法表达化学计量、酶催化、compartment、修饰状态
4. **预校验局限**（§4.4）：子串匹配误报、自环检测过严、Unicode 命名碰撞漏报
5. **state_transitions 空扩展点**（§4.5）：蛋白质状态转换无法表达，多步骤级联被压扁

这些缺陷使当前 IR 无法表达癌症网络的核心结构：destruction complex 三步耦合、p53 状态机（磷酸化→四聚化→入核→转录）、Caspase 级联（initiator→executor amplification）、NF-κB 振荡（IκBα 转录延迟负反馈）。

### 4.2 Reaction IR v2 Schema

Reaction IR v2 由 6 个核心组件构成：

#### 4.2.1 Species Schema

```
Species:
  id: str                          # 全局唯一 ID（如 "SP_001"）
  canonical_name: str              # 规范名（如 "EGFR"）
  display_name: str                # 显示名（如 "EGF Receptor"）
  ontology:
    hgnc_id: str                   # HGNC 基因 ID（如 "HGNC:3236"）
    uniprot_id: str                # UniProt 蛋白 accession（如 "P00533"）
    chebi_id: str | None           # ChEBI 化学实体 ID（配体/药物用）
    go_terms: list[str]            # GO 功能术语（如 ["GO:0004672"]）
    sbo_term: str                  # SBO 生物学本体（如 "SBO:0000252" for protein）
  species_type: str                # 11 类之一（ligand/receptor/kinase/...）
  state_machine: str | None        # 关联的状态机 ID（如 "EGFR_STATE_MACHINE"）
  compartment: str                 # "extracellular" / "membrane" / "cytoplasm" / "nucleus" / "mitochondria"
  initial_concentration: float     # 初始浓度（nM）
  concentration_unit: str          # "nM" / "molecule_per_cell"
  provenance:
    source_sbml: str | None        # SBML model ID（如 "BIOMD0000000205"）
    source_pmid: str | None        # PMID
    source_uniprot: str | None     # UniProt entry
```

#### 4.2.2 Reaction Schema（含组合机制）

```
Reaction:
  id: str                          # 全局唯一 ID（如 "RXN_001"）
  reaction_type: str               # 17 类之一（见 §4.3）
  kinetics_type: str               # mass_action / Michaelis_Menten / Hill / Boolean / hybrid
  reactants: list[SpeciesRef]      # 反应物（含化学计量 + 角色）
  products: list[SpeciesRef]       # 产物（含化学计量 + 角色）
  modifiers: list[Modifier]        # 调控因子（酶 / 催化剂 / 变构调节子）
  compartments: list[str]          # 涉及的 compartment（用于跨区室反应）
  parameter_context: str           # 参数上下文（如 "EGF-EGFR binding kon/koff"）
  pathway_tag: str                 # 通路标签（如 "EGFR_EGFR_MAPK"）
  provenance:
    source_sbml_reaction: str | None  # SBML reaction ID
    source_pmid: str | None
    source_kegg: str | None           # KEGG reaction ID
  constraints: list[Constraint]    # 反应级约束（见 §4.4）

SpeciesRef:
  species_id: str
  stoichiometry: int = 1
  role: str                        # "substrate" / "product" / "enzyme" / "cofactor"

Modifier:
  species_id: str
  modifier_type: str               # "catalytic" / "allosteric" / "inhibitory" / "activating"
  site: str | None                 # 修饰位点（如 "Ser259" / "Tyr1068"）
```

#### 4.2.3 Composite Reaction（组合反应）

```
CompositeReaction:
  id: str
  name: str                        # 如 "Wnt destruction complex"
  sub_reactions: list[Reaction]    # 有序子反应列表
  coupling_type: str               # "sequential" / "branched" / "cyclic"
  intermediate_species: list[Species]  # 中间产物
  net_reaction: str                # 净反应方程（如 "β-catenin → β-catenin-Ub")
```

**Wnt destruction complex 示例**：
- Sub-reaction 1: Axin + APC + GSK3β + CK1 → destruction complex（complex_formation）
- Sub-reaction 2: destruction complex + β-catenin → destruction complex-β-catenin（binding）
- Sub-reaction 3: destruction complex-β-catenin → destruction complex + p-β-catenin（phosphorylation，GSK3β 为 enzyme）
- Sub-reaction 4: p-β-catenin + β-TrCP → p-β-catenin-Ub（ubiquitination）
- Sub-reaction 5: p-β-catenin-Ub → ∅（proteasomal_degradation）
- Net: β-catenin → ∅（依赖 Wnt off 状态）

#### 4.2.4 State Machine（蛋白质状态转换）

```
StateMachine:
  id: str                          # 如 "EGFR_STATE_MACHINE"
  species: str                     # 关联的蛋白（如 "EGFR"）
  states: list[State]
  transitions: list[Transition]

State:
  name: str                        # 如 "monomer" / "dimer" / "phosphorylated_dimer"
  species_id: str                  # 对应的 Species ID
  is_initial: bool

Transition:
  from_state: str
  to_state: str
  reaction_id: str                 # 关联的 Reaction（触发状态转换的反应）
  trigger: str                     # "ligand_binding" / "phosphorylation" / "internalization"
```

**EGFR 状态机示例**：
- States: monomer → EGF-bound monomer → dimer → phosphorylated dimer → Grb2-bound → internalized → degraded
- Transitions: 每个状态转换关联一个 Reaction（binding / dimerization / autophosphorylation / recruitment / endocytosis / degradation）

#### 4.2.5 Compartment Schema

```
Compartment:
  name: str                        # "extracellular" / "membrane" / "cytoplasm" / "nucleus" / "mitochondria"
  size: float                      # 体积比（如 cytoplasm=0.5, nucleus=0.1）
  transport_reactions: list[str]   # 跨该区室的运输反应 ID
```

#### 4.2.6 Constraint Schema

```
Constraint:
  type: str                        # "mass_conservation" / "steady_state" / "non_negative" / "enzymatic" / "thermodynamic"
  scope: str                       # "species" / "reaction" / "pathway" / "global"
  expression: str                  # 约束表达式（如 "EGFR + pEGFR + EGF-EGFR = EGFR_total"）
  tolerance: float                 # 容差（如 0.05 表示 5%）
  provenance: str                  # 约束来源（如 "Schoeberl 2002"）
```

### 4.3 Mechanism Types（17 类）

v4 Reaction IR 必须支持以下 17 类机制，每类关联 SBO term 与默认动力学：

| 机制类型 | SBO Term | 默认动力学 | 适用场景 |
|---------|----------|-----------|---------|
| phosphorylation | SBO:0000216 | Michaelis-Menten | 激酶-底物（强制 MM，禁止降级为 mass-action） |
| dephosphorylation | SBO:0000330 | Michaelis-Menten | 磷酸酶-底物 |
| binding | SBO:0000177 | mass-action (reversible) | 配体-受体、蛋白-蛋白结合 |
| dissociation | SBO:0000180 | first-order | 复合物解离 |
| dimerization | SBO:0000434 | mass-action (2A → A2) | 受体二聚化 |
| ubiquitination | SBO:0000218 | mass-action (E3 依赖) | p53-Mdm2、IκBα-β-TrCP |
| degradation | SBO:0000179 | first-order | 蛋白降解 |
| proteasomal_degradation | SBO:0000218 | mass-action (26S proteasome) | ubiquitin-dependent |
| transcription | SBO:0000183 | Hill (n=1-4) | p53→Mdm2、NF-κB→IκBα |
| translation | SBO:0000184 | first-order (mRNA-dependent) | mRNA→蛋白 |
| nuclear_import | SBO:0000186 | first-order + cargo | NF-κB、STAT、SMAD 入核 |
| nuclear_export | SBO:0000187 | first-order + cargo | NF-κB-IκBα 出核 |
| cytoplasm_translocation | SBO:0000186 | first-order | 胞质内转运 |
| complex_formation | SBO:0000526 | mass-action (multi-component) | destruction complex |
| cleavage | SBO:0000213 | Michaelis-Menten (irreversible) | Caspase、Notch NICD |
| GTP_GDP_exchange | SBO:0000174 | mass-action (GEF/GAP) | Ras-RasGTP / Ras-RasGDP |
| inhibition | SBO:0000169 | mixed (competitive/allosteric) | 药物抑制、负反馈 |
| activation | SBO:0000170 | mixed (multiple forms) | 多种激活形式 |
| sequestration | SBO:0000169 | mass-action | Bad-Bcl-2 sequestration |

### 4.4 Constraint System

v4 Reaction IR 内置 5 类约束：

1. **Mass Conservation**：受体总量守恒（EGFR + pEGFR + EGF-EGFR = EGFR_total）、蛋白池守恒（Ras-GDP + Ras-GTP = Ras_total）
2. **Steady State**：未刺激状态下稳态约束（如 pEGFR 基线 <1% EGFR total）
3. **Non-negative**：所有浓度非负（数值保护）
4. **Enzymatic**：酶不被消耗（酶同时出现在 reactants 与 products）
5. **Thermodynamic**：可逆反应的热力学一致性（K_eq = k_forward / k_reverse）

### 4.5 Validation Rules

Reaction IR v2 的校验规则（取代 v3 的 pre_validate_reaction_graph）：

1. **Ontology Alignment**：所有 species 必须有 HGNC / UniProt ID（药物除外，用 ChEBI）
2. **Pathway Tag**：所有 reaction 必须有 pathway_tag
3. **Provenance**：所有 reaction 必须有 source_sbml_reaction 或 source_pmid
4. **Compartment Consistency**：跨区室反应必须有 transport 类型
5. **State Machine Closure**：状态机的所有 transition 必须关联到存在的 Reaction
6. **Constraint Satisfaction**：所有约束在初始条件下满足
7. **Composite Reaction Order**：sequential 类型的 sub_reactions 必须有明确顺序
8. **Enzyme Role**：标记为 enzyme 的 species 必须同时出现在 reactants 与 products
9. **Kinetics-Mechanism Match**：phosphorylation 必须用 MM（强制，禁止降级）
10. **Cross-talk Edge Validation**：cross-talk edge 必须标记两个 pathway_tag

### 4.6 Reaction Ontology

v4 建立机制本体层级（基于 SBO）：

```
Reaction
├── Covalent Modification
│   ├── Phosphorylation (SBO:0000216)
│   ├── Dephosphorylation (SBO:0000330)
│   ├── Ubiquitination (SBO:0000218)
│   └── Acetylation / Methylation (future)
├── Complex Formation
│   ├── Binding (SBO:0000177)
│   ├── Dimerization (SBO:0000434)
│   ├── Complex Assembly (SBO:0000526)
│   └── Sequestration (SBO:0000169)
├── Conversion
│   ├── Cleavage (SBO:0000213)
│   ├── GTP/GDP Exchange (SBO:0000174)
│   └── Conformational Change (future)
├── Gene Expression
│   ├── Transcription (SBO:0000183)
│   └── Translation (SBO:0000184)
├── Transport
│   ├── Nuclear Import (SBO:0000186)
│   ├── Nuclear Export (SBO:0000187)
│   └── Cytoplasmic Translocation
├── Degradation
│   ├── Spontaneous Degradation (SBO:0000179)
│   └── Proteasomal Degradation (SBO:0000218)
└── Regulation
    ├── Inhibition (SBO:0000169)
    └── Activation (SBO:0000170)
```

### 4.7 回应 v1.0 审计报告

Reaction IR v2 直接回应审计报告 §4 全部问题：

| 审计问题 | v4 解决方式 |
|---------|------------|
| §4.1 缺 ubiquitination/nuclear transport/cleavage 等 | 17 类机制完整覆盖（含 SBO term） |
| §4.2 activation→phosphorylation 语义错误 | activation 独立机制类型，不强制映射 |
| §4.3 graph→equation 信息损失 | SpeciesRef 含 stoichiometry + role，Modifier 含 site |
| §4.3 无法表达 context-dependent 酶活性 | State Machine 表达多状态蛋白 |
| §4.3 无法表达 destruction complex | CompositeReaction 表达三步耦合 |
| §4.3 无 compartment | Compartment schema + 跨区室反应校验 |
| §4.4 预校验局限 | 10 条 Validation Rules 替代子串匹配 |
| §4.5 state_transitions 空 | State Machine 一等公民，全栈支持 |
| §2.5 环路打破丢 inhibition | Constraint System 强制 mass conservation，不靠丢边打破环路 |
| §3.1 MM 退化 | Validation Rule 9：phosphorylation 必须用 MM（强制） |

---

## Part 5: ODE Framework v2

### 5.1 模板体系重新分类

v4 的 ODE 框架不再是"9 个模板覆盖 10 通路"，而是按机制-动力学二维分类：

#### 5.1.1 必须保留的模板（升级版）

| 模板 | 用途 | 升级点 |
|------|------|--------|
| Signaling_Cascade_Phos | EGFR 信号级联 | 恢复 MM 动力学；支持状态机；多 compartment |
| Simple_Inhibition | 单药物抑制 | 移除硬编码 0.1；支持 competitive/allosteric |
| Simple_Activation | 单激活子 | 移除硬编码 0.1；支持多种激活形式 |
| PKPD_OneCompartment | 一室 PK/PD | 移除硬编码 0.5/0.05；target 动力学可配置 |
| PKPD_TwoCompartment | 二室 PK/PD | 同上 |
| Combination | 联合用药 | 保留 Chou-Talalay CI |
| DoseSweep | 剂量递增 | 保留 |

#### 5.1.2 必须新增的模板

| 模板 | 用途 | 关键机制 |
|------|------|---------|
| **Bistable_Switch** | Apoptosis / Cell Cycle / p53 决策 | 正反馈 + double-well potential |
| **Oscillatory_Feedback** | NF-κB / p53 / Cell Cycle 振荡 | 延迟负反馈 + DDE |
| **Transcriptional_Delay** | p53-Mdm2 / JAK-STAT-SOCS / NF-κB-IκBα / TGF-β-SMAD7 | mRNA 延迟 + DDE |
| **Nuclear_Transport** | STAT / NF-κB / SMAD / β-catenin 入核 | 双 compartment + nuclear-cytoplasmic ratio |
| **Ubiquitination_Cascade** | p53-Mdm2 / IκBα-β-TrCP / β-catenin | phosphorylation-dependent ubiquitination 三步 |
| **Destruction_Complex** | Wnt β-catenin | CompositeReaction 渲染 |
| **Caspase_Cascade** | Apoptosis initiator→executor | cleavage（不可逆 MM）+ bistable |
| **Cyclin_CDK_Toggle** | Cell Cycle G1/S + G2/M switch | bistable + delayed negative feedback |
| **Transcription_Factor** | 通用转录因子激活 | Hill + nuclear import + target gene transcription |

#### 5.1.3 必须废弃的模板

| 模板 | 废弃原因 |
|------|---------|
| Cascade_Activation | L39-42 静默丢弃 inhibition 边；逻辑反转风险（审计 §3.5） |
| Cascade_Inhibition | L42-45 inhibition/activation 分支逻辑反转（审计 §3.5） |

#### 5.1.4 改成组合模板的

- "EGFR 信号 + 药物 PK"组合：通过 CompositeReaction + ODE composition（多个 ODE 右端项相加）
- "p53 信号 + 药物 Mdm2 抑制剂"组合：通过 CompositeReaction + Perturbation Module

#### 5.1.5 改成动态图生成的

复杂通路（如 destruction complex、Caspase 级联）不使用静态模板，而是从 Reaction IR v2 动态生成 ODE。生成规则由 Mechanism Layer 决定。

### 5.2 动力学类型适用范围

| 动力学 | 适用场景 | 不适用场景 |
|--------|---------|-----------|
| **Mass-Action** | 双分子结合（A+B→AB）、一阶降解、二聚化 | 酶催化（应 MM）、转录（应 Hill） |
| **Michaelis-Menten** | 酶催化磷酸化/去磷酸化/cleavage | 化学计量结合（应 mass-action） |
| **Hill** | 转录因子-启动子（n=1-4）、cooperativity | 单分子反应 |
| **Boolean** | 早期决策网络（如 p53 on/off）、定性分析 | 定量仿真 |
| **Hybrid** | 复杂通路（部分 mass-action + 部分 MM + 部分 Hill） | 简单反应（过度复杂） |
| **Rule-based** | 多状态蛋白组合爆炸（如多位点磷酸化 ERK T202/Y204） | 状态数 < 4 的蛋白 |

### 5.3 ODE 生成原则

1. **机制优先于模板**：动力学类型由 Mechanism Layer 决定，不是由模板硬编码
2. **MM 不可降级**：phosphorylation / dephosphorylation / cleavage 必须用 MM，禁止降级为 mass-action（审计报告 §3.1 致命错误）
3. **Hill 默认用于转录**：transcription 默认 Hill n=2，除非有 cooperativity 证据
4. **DDE 用于延迟反馈**：transcriptional delay 必须用 DDE，不用 ODE 近似（审计报告 §3.2 致命错误）
5. **Compartment 显式**：跨区室反应必须建模为 transport 项，不能压扁为单 compartment
6. **参数注入带 provenance**：每个参数必须携带 source_sbml / source_pmid

### 5.4 回应 v1.0 审计报告

| 审计问题 | v4 解决方式 |
|---------|------------|
| §3.1 MM 退化（TASK 4 主动移除） | 原则 2：MM 不可降级，phosphorylation 强制 MM |
| §3.2 无 DDE/SDE | 新增 Oscillatory_Feedback + Transcriptional_Delay 模板，用 DDE |
| §3.2 无 bistable | 新增 Bistable_Switch + Caspase_Cascade + Cyclin_CDK_Toggle |
| §3.2 无 compartment | 新增 Nuclear_Transport 模板 + Compartment schema |
| §3.3 硬编码参数 | 所有模板移除硬编码，参数必须从 Parameter Agent 注入 |
| §3.4 静默修改用户参数 | 移除 _cascade_helpers 的 _enforce_phos_dephos_ratio |
| §3.5 Cascade_Inhibition 逻辑反转 | 废弃 Cascade_Activation 与 Cascade_Inhibition |
| §5.1 模板覆盖率 1/10 | 9 个新模板覆盖 9 条缺失通路 |
| §5.2 模板-通路不匹配 | 每条通路有专属模板（如 Apoptosis 用 Caspase_Cascade，不用 phosphorylation） |
| §5.6 模板降级映射风险 | Cascade_Activation 废弃，不再有降级映射 |

---

## Part 6: Simulation Engine

### 6.1 仿真能力矩阵

v4 Simulation Layer 必须支持以下 11 类仿真：

| 仿真类型 | 用途 | 求解器要求 |
|---------|------|-----------|
| **Time-scale simulation** | 单次时间序列仿真 | ODE: LSODA / DDE: dde23 |
| **Steady state** | 求解稳态 | Newton + continuation |
| **Oscillation detection** | NF-κB / p53 / Cell Cycle 振荡 | 频谱分析 + limit cycle 检测 |
| **Bistability detection** | Apoptosis / Cell Cycle toggle | 双稳态检测 + basins of attraction |
| **Dose response** | 药物剂量-效应曲线 | 参数扫描 + Hill fit |
| **Parameter scan** | 单参数扫描 | 1D sweep |
| **Global sensitivity** | Sobol / Morris 全局灵敏度 | Saltelli sampling |
| **Local sensitivity** | 单参数灵敏度 | forward/backward difference |
| **Virtual knockout** | 基因敲除仿真 | species 设为 0 + 反应移除 |
| **Drug perturbation** | 药物扰动 | PK/PD 耦合 + IC50/Emax |
| **Multi-pathway simulation** | 通路网络仿真 | 多通路 ODE 耦合求解 |

### 6.2 多时间尺度支持

v4 必须支持三层时间尺度：

| 时间尺度 | 范围 | 典型过程 | 求解策略 |
|---------|------|---------|---------|
| Fast (seconds-minutes) | 0-30 min | 磷酸化、结合、cleavage | 小步长（max_step=0.1 min） |
| Medium (minutes-hours) | 30 min - 6h | 转录、翻译、ubiquitination | 中步长（max_step=1 min） |
| Slow (hours-days) | 6h - 48h | 细胞周期、降解、稳态 | 大步长（max_step=10 min） |

**多尺度耦合策略**：
- 同一 ODE 系统内不同 species 用不同时间尺度参数
- 用 implicit method 处理 stiff 系统（fast-slow 耦合）
- transcriptional delay 必须用 DDE，不用 ODE 近似

### 6.3 Oscillation Detection

NF-κB / p53 / Cell Cycle 的振荡检测：
- **周期检测**：FFT 主频 + autocorrelation
- **振幅衰减率**：(peak_n - peak_n+1) / peak_n
- **振荡持续**：振幅衰减 < 50% 视为 sustained oscillation
- **极限环**：phase portrait 在 (x, dx/dt) 平面闭合

### 6.4 Bistability Detection

Apoptosis / Cell Cycle 的双稳态检测：
- **多初值扫描**：从不同初始条件求解稳态
- **basin of attraction**：每个稳态的吸引域
- **bifurcation parameter**：扫描控制参数（如 Caspase-3 总量）
- **hysteresis**：增加 vs 减少参数的稳态曲线是否重合

### 6.5 Sensitivity Analysis

| 类型 | 方法 | 用途 |
|------|------|------|
| Local | forward difference | 单参数影响 |
| Global - Sobol | Saltelli sampling + variance decomposition | 全局参数重要性排名 |
| Global - Morris | elementary effects | 筛选重要参数 |
| Global - PRCC | partial rank correlation | 相关性 |

### 6.6 Virtual Knockout

| 扰动类型 | 实现 |
|---------|------|
| Gene knockout | species 初始浓度设为 0 + 相关 reaction 移除 |
| Knockdown | species 初始浓度降低 50% / 90% |
| Overexpression | species 初始浓度升高 5x / 10x |
| Mutation (constitutive active) | 修改参数（如 Ras GTP→GDP 速率设为 0） |
| Mutation (dominant negative) | 添加竞争性抑制 species |
| Drug inhibition | 添加 drug species + inhibition reaction（含 IC50） |

### 6.7 Multi-Pathway Simulation

跨通路仿真策略：
1. **Shared species 同步**：多个通路共享的 species（如 Ras / AKT / MEK）在 ODE 系统中是同一变量
2. **Cross-talk edges**：cross-talk 作为 cross-pathway reaction 注入 ODE 右端
3. **时间尺度对齐**：不同通路的时间尺度通过 max_step 统一
4. **Compartment 一致性**：跨通路共享的 compartment（如 nucleus）保持一致

### 6.8 回应 v1.0 审计报告

| 审计问题 | v4 解决方式 |
|---------|------------|
| §3.6 无 DDE/SDE 求解器 | 6.1 必须支持 DDE（dde23 类）用于延迟反馈 |
| §3.7 9 维特征通用 | 6.3 / 6.4 新增振荡检测 + 双稳态检测 |
| §7.1 无 Sensitivity Analysis | 6.5 Sobol / Morris / PRCC |
| §7.1 无 Perturbation Simulation | 6.6 完整 KO / knockdown / overexpression / mutation / drug |
| §3.6 max_step 过粗 | 6.2 多时间尺度分层 + 自适应步长 |
| §2.2 cross-talk 被禁止 | 6.7 Multi-Pathway Simulation 支持 shared species + cross-talk edges |

---

## Part 7: Scientific Validation

### 7.1 Validation Pyramid

v4 建立五层 Validation Pyramid，从底到上严格性递增：

```
                    ┌─────────────────┐
                    │  Level 5:       │
                    │  Hypothesis     │
                    │  Validation     │
                    │  (实验数据)      │
                    └─────────────────┘
                  ┌─────────────────────┐
                  │  Level 4:           │
                  │  Benchmark          │
                  │  Validation         │
                  │  (社区 benchmark)    │
                  └─────────────────────┘
                ┌─────────────────────────┐
                │  Level 3:               │
                │  Cross-Pathway          │
                │  Validation             │
                │  (通路间一致性)          │
                └─────────────────────────┘
              ┌─────────────────────────────┐
              │  Level 2:                   │
              │  SBML/BioModels             │
              │  Validation                 │
              │  (canonical model 对齐)      │
              └─────────────────────────────┘
            ┌─────────────────────────────────┐
            │  Level 1:                       │
            │  Internal Consistency           │
            │  Validation                     │
            │  (质量守恒/非负/稳态)            │
            └─────────────────────────────────┘
```

#### Level 1: Internal Consistency

- Mass conservation（受体守恒、蛋白池守恒）
- Non-negative concentration
- Steady state 检查（未刺激状态）
- Numerical stability（无 NaN / Inf）
- Constraint satisfaction（所有 Reaction IR v2 约束）

#### Level 2: SBML / BioModels Validation

- **Track A（强制）**：libroadrunner 跑真实 SBML 仿真，对比 peak / peak_time / amplification
- **Track B（fallback，不得输出 error_diff=0）**：结构相似度评分，但差异指标标记为 `null` 而非 `0`
- **skipped 状态 = pass=False**（强制阻塞流水线，审计报告 §7.2 致命错误修复）
- **物种对齐用 ontology ID**（HGNC/UniProt），不用字符串匹配（审计报告 §10.3 修复）
- **阈值通路特异**：不同通路用不同阈值（如 NF-κB peak_time_diff 容忍更大）

#### Level 3: Cross-Pathway Validation

- Cross-talk consistency：cross-talk edges 在两个通路中行为一致
- Shared species conservation：跨通路 shared species 总量守恒
- Time-scale consistency：跨通路时间尺度对齐

#### Level 4: Benchmark Validation

- 社区 benchmark 对齐：
  - MAPK: Markevich 2004 ultrasensitivity
  - NF-κB: Nelson 2004 oscillation period
  - p53: Lev Bar-Or 2000 pulse dynamics
  - EGFR: Schoeberl 2002 pEGFR peak time
  - Wnt: Lee 2003 β-catenin steady state
- 文献 benchmark：从 PubMed 检索关键参数的实验值

#### Level 5: Hypothesis Validation

- 实验数据对比：用户提供实验数据，仿真结果对比
- 假设可证伪性：每个生成的假设必须有可验证的预测

### 7.2 数据源对齐

| 数据源 | 用途 | 必须灌库 |
|--------|------|---------|
| SBML / BioModels | Level 2 validation ground truth | 是 |
| KEGG | 通路拓扑 reference | 是（审计 §7.3 修复） |
| Reactome | 通路反应 reference | 是（审计 §7.3 修复） |
| UniProt | 蛋白本体 + 修饰位点 | 是（审计 §7.3 修复） |
| ChEBI | 化学实体（配体/药物） | 是（审计 §7.3 修复） |
| PubMed | 文献证据 + 参数溯源 | 是 |
| ClinicalTrials.gov | 药物剂量 reference | 运行时查询 |
| BRENDA | 酶动力学参数 | 是（v5） |
| SABIO-RK | 反应动力学参数 | 是（v5） |

### 7.3 Reference Grounding 五级映射

v4 必须建立 ODE ↔ SBML reaction ↔ parameter ↔ species ↔ PMID 五级映射链（审计报告 §7.4 修复）：

```
ODE Equation
  ↓ generated_from
Reaction IR
  ↓ aligned_to
SBML Reaction (reaction_id in BioModels)
  ↓ uses_parameter
Parameter (k_forward, k_reverse, Km, Vmax)
  ↓ sourced_from
Literature (PMID + figure_ref)
  ↓ describes
Species (HGNC ID + UniProt accession)
```

每个 ODE 方程必须能溯源到：哪个 Reaction → 哪个 SBML reaction → 哪个参数 → 哪篇文献 → 哪个物种的标准 ID。

### 7.4 回应 v1.0 审计报告

| 审计问题 | v4 解决方式 |
|---------|------------|
| §7.2 Oracle 默认 pass=True | Level 2 skipped = pass=False 强制阻塞 |
| §7.2 Track B 输出 error_diff=0 | Level 2 Track B 差异指标标记 null |
| §10.3 species_ontology 字符串匹配 | Level 2 用 ontology ID 对齐 |
| §7.2 阈值硬编码 | Level 2 阈值通路特异 |
| §7.4 Reference grounding 五级缺失 | 7.3 五级映射链强制 |
| §7.3 数据源未灌库 | 7.2 KEGG/Reactome/UniProt/ChEBB 强制灌库 |
| §2.4 DomainChecker 通路盲化 | Level 1 + 通路特异 Validation Module |
| §3.7 9 维特征通用 | Level 4 benchmark 含振荡/双稳态/ultrasensitivity |

---

## Part 8: Knowledge Architecture

### 8.1 知识架构组件

v4 Knowledge Architecture 由 6 个组件构成：

```
┌─────────────────────────────────────────────────────────┐
│  Knowledge Graph (KG)                                    │
│  通路拓扑 + cross-talk + 物种关系                         │
├─────────────────────────────────────────────────────────┤
│  Ontology Service                                        │
│  HGNC / UniProt / ChEBI / GO / SBO / KEGG / Reactome    │
├─────────────────────────────────────────────────────────┤
│  Parameter DB                                            │
│  动力学参数（含 pathway_tag + provenance_id）             │
├─────────────────────────────────────────────────────────┤
│  Evidence DB                                             │
│  文献证据（含 doi + figure_ref + cell_line）              │
├─────────────────────────────────────────────────────────┤
│  Mechanism DB                                            │
│  机制模板 + 反应方程 + SBO term                           │
├─────────────────────────────────────────────────────────┤
│  Experiment DB                                           │
│  实验协议 + 观测数据 + 细胞系                              │
└─────────────────────────────────────────────────────────┘
```

### 8.2 Knowledge Graph (KG)

KG 节点：
- Species（含 HGNC/UniProt ID）
- Pathway（含 KEGG/Reactome ID）
- Disease（cancer subtype）
- Drug（含 ChEBI ID）
- Mutation

KG 边：
- activates / inhibits
- phosphorylates / dephosphorylates
- binds / dissociates
- ubiquitinates / degrades
- transcribes / translates
- translocates_to
- cross_talks_with（显式 cross-talk edge）
- mutation_affects

**审计报告 §2.5 环路打破丢 inhibition 的修复**：KG 不再做环路打破。负反馈环路是生物学核心结构，必须保留。如果检测到环路，标记为 "feedback_loop" 并要求 Validation Layer 验证其稳定性。

### 8.3 Ontology Service

强制对齐标准本体（审计报告 §2.6 修复）：

| 本体 | 用途 | 强制字段 |
|------|------|---------|
| HGNC | 基因符号 | 所有 gene product 必须有 HGNC ID |
| UniProt | 蛋白 accession | 所有 protein 必须有 UniProt ID |
| ChEBI | 化学实体 | 所有 ligand / drug 必须有 ChEBI ID |
| GO | 功能术语 | 所有 species 必须有 ≥1 GO term |
| SBO | 系统生物学本体 | 所有 reaction 必须有 SBO term |
| KEGG | 通路 ID | 所有 pathway 必须有 KEGG pathway ID |
| Reactome | 通路反应 ID | 所有 reaction 优先对齐 Reactome reaction ID |

### 8.4 Parameter DB（强制溯源与通路隔离）

#### 8.4.1 Parameter Schema

```
Parameter:
  id: str                          # 全局唯一 ID（provenance_id）
  param_name: str                  # 如 "k_phos_EGFR"
  value: float
  unit: str                        # "nM^-1 min^-1" / "min^-1" / "nM" / "min"
  type: str                        # "kon" / "koff" / "kcat" / "Km" / "Kd" / "Vmax" / "n_hill" / "initial_conc"
  pathway_tag: str                 # 强制：通路标签（如 "EGFR_EGFR_MAPK"）
  species: str                     # 物种（如 "Human" / "Mouse"），从文献提取，不硬编码
  cell_line: str                   # 细胞系（如 "HeLa" / "MCF7" / "A431"）
  reaction_context: str            # 反应上下文（如 "EGF-EGFR binding"）
  provenance:
    source_type: str               # "PubMed" / "BioModels" / "ChEMBL" / "BRENDA" / "SABIO-RK"
    source_id: str                 # PMID / BIOMD ID / ChEMBL ID
    source_pmid: str               # 强制：PMID（无 PMID 则 source_pmid="unverified"）
    figure_ref: str                # figure / table 引用
    year: int
    journal: str
  confidence: float                # 0.0-1.0
  conditions:
    ph: float | None
    temperature: float | None
    assay_type: str                # "in vitro" / "in vivo" / "in cellulo"
```

#### 8.4.2 通路隔离强制规则

1. **pathway_tag 必填**：无 pathway_tag 的参数拒绝入库（审计报告 §5.1 修复）
2. **检索必须按通路过滤**：查询参数时必须传 pathway_tag，否则拒绝返回
3. **shared species 特殊处理**：Ras / AKT / MEK 等跨通路分子的参数，每条记录的 pathway_tag 必须明确标注来源通路
4. **cross-talk 参数独立标记**：cross-talk 反应的参数 pathway_tag 为 "CROSSTALK_A_B"

#### 8.4.3 Provenance 强制规则

1. **source_pmid 必填**：无 PMID 的参数标记为 `source_pmid="unverified"`，confidence ≤ 0.3
2. **BioModels / ChEMBL 参数必须有 source_pmid**：若 SBML 模型无 PMID，从模型描述文本提取并验证
3. **figure_ref 必填**：从文献提取参数时必须标注 figure / table（审计报告 §5.6 修复）

### 8.5 Evidence DB

#### 8.5.1 Evidence Schema

```
Evidence:
  pmid: str                        # 强制
  doi: str                         # 强制（从 PubMed efetch 解析）
  title: str
  abstract: str
  figure_ref: str                  # 强制（如 "Figure 3B"）
  table_ref: str                   # 如 "Table 2"
  cell_line: str                   # 强制（从 abstract 提取，不硬编码 "Human"）
  species: str                     # 强制（从 abstract 提取，不硬编码）
  year: int
  journal: str
  pathway_tags: list[str]          # 强制：可属于多条通路
  experiment_type: str             # "Western blot" / "ELISA" / "FACS" / "live imaging"
  key_findings: str                # 关键发现摘要
  parameters_extracted: list[str]  # 关联的 Parameter ID
```

#### 8.5.2 Evidence-Parameter 关联

evidence 与 parameter 必须 1:1 或 1:N 关联（审计报告 §6.3 修复）：
- 每个 parameter 的 provenance.source_pmid 必须能在 Evidence DB 找到对应记录
- 检索 parameter 时，自动关联对应 evidence
- 不再用同一 query 独立检索 evidence（避免错配）

### 8.6 Mechanism DB

```
Mechanism:
  mechanism_type: str              # 17 类之一
  sbo_term: str                    # SBO ID
  default_kinetics: str            # 默认动力学
  template_reaction: str           # 反应方程模板（如 "A + B → A-B"）
  required_parameters: list[str]   # 必需参数（如 binding 需 kon + koff）
  validation_rules: list[str]      # 机制级验证规则
  pathway_examples: list[str]      # 适用通路示例
```

### 8.7 Experiment DB

```
Experiment:
  experiment_id: str
  name: str
  target_pathway: str              # 通路标签
  cell_line: str
  species: str
  perturbation: str                # "EGF 100 ng/mL" / "TNF-α 10 ng/mL"
  time_points: list[float]         # 观测时间点
  observed_species: list[str]      # 观测物种
  observations: dict               # {species: [(time, concentration)]}
  source_pmid: str
  source_figure: str
```

### 8.8 RAG 检索策略

| 检索类型 | 过滤维度 | 排序 |
|---------|---------|------|
| Mechanism RAG | pathway_tag + mechanism_type | rerank (model) + BM25 |
| Parameter RAG | pathway_tag + species + cell_line + reaction_context | rerank + BM25 + threshold |
| Evidence RAG | pathway_tags + experiment_type | rerank + BM25 |
| Experiment RAG | target_pathway + cell_line + perturbation | rerank + BM25 |

#### 8.8.1 Threshold 强制规则

- 相似度 < 0.3 的结果不返回（审计报告 §4.3 修复）
- 相似度 0.3-0.5 的结果标记为 "low_confidence"
- 相似度 > 0.5 的结果正常返回

#### 8.8.2 Species 过滤强制

- BM25 检索必须传 species_filter（审计报告 §4.4 修复）
- species 字段从文献提取，不硬编码 "Human"（审计报告 §5.5 修复）
- SBML 参数 species="Unknown" 时用 BioModels 元数据补全（审计报告 §5.3 修复）

### 8.9 回应 v1.0 审计报告

| 审计问题 | v4 解决方式 |
|---------|------------|
| §5.1 parameter 无 pathway 字段 | Parameter DB pathway_tag 强制必填 |
| §5.1 experiment collection 为空 | Experiment DB 独立组件，强制灌库 |
| §5.2 数据源未灌库 | KEGG/Reactome/UniProt/ChEMBL 强制灌库 |
| §5.2 reference grounding 缺失 | 7.3 五级映射 + Parameter provenance.source_pmid 强制 |
| §5.3 species 硬编码 "Human" | 从文献提取，不硬编码 |
| §5.3 SBML species="Unknown" | 用 BioModels 元数据补全 |
| §5.4 BM25 不带 species_filter | 8.8.2 BM25 强制传 species_filter |
| §5.4 species 精确字符串匹配 | 用 Ontology Service 做 ID 对齐 |
| §5.4 无 threshold | 8.8.1 相似度 < 0.3 不返回 |
| §5.5 cross-pathway contamination | 通路隔离强制规则 + shared species 特殊处理 |
| §5.6 evidence 字段为空 | Evidence DB doi/figure_ref/cell_line 强制必填 |
| §5.6 evidence-parameter 错配 | 8.5.2 evidence-parameter 1:1 关联 |
| §2.5 环路打破丢 inhibition | KG 不做环路打破，标记 feedback_loop |

---

## Part 9: Agent Architecture v4

### 9.1 Agent 重构原则

v4 Agent 架构遵循 Scientific Stack 八层映射：

| Scientific Layer | 对应 Agent |
|----------------|-----------|
| Layer 1: Biological Knowledge | Ontology Agent |
| Layer 2: Pathway Graph | Pathway Planner Agent |
| Layer 3: Reaction IR | Reaction Builder Agent |
| Layer 4: Mechanism | Mechanism Builder Agent |
| Layer 5: ODE Generation | ODE Builder Agent |
| Layer 6: Simulation | Simulation Planner Agent |
| Layer 7: Validation | SBML Grounder + Validation Agent |
| Layer 8: Hypothesis | Hypothesis Agent |

### 9.2 Agent 清单（保留 / 删除 / 新增）

#### 9.2.1 保留（升级）

| Agent | v3 职责 | v4 升级点 |
|-------|--------|---------|
| MCP Term Lookup | 术语查询 | 升级为 Ontology Agent，强制 HGNC/UniProt 对齐 |
| NER + Mechanism Planner | 实体识别 + 机制规划 | 拆分为 Pathway Planner + Reaction Builder |
| KG Builder | 知识图谱构建 | 保留，但边类型扩展到 17 类，环路不打破 |
| RAG (mechanism + parameter) | 知识检索 | 拆分为 Parameter Agent + Mechanism Agent |
| PK/PD Inference | PK/PD 推断 | 保留，移除 auto_fast 跳过逻辑 |
| Sandbox | 仿真执行 | 拆分为 Simulation Planner + Validation Agent |
| Report Renderer | 报告渲染 | 保留，但增加 Hypothesis Agent |

#### 9.2.2 删除

| Agent | 删除原因 |
|-------|---------|
| LLM Template Selector | 机制优先原则下，动力学类型由 Mechanism Layer 决定，不需 LLM 选模板 |
| DomainChecker（v3 版本） | 升级为 Validation Agent，含通路特异规则 |
| node4_audit_and_correct | 审计嵌入 sandbox 是职责混乱，拆分为独立 Validation Agent |
| Cascade_Activation / Cascade_Inhibition 模板选择逻辑 | 模板已废弃 |

#### 9.2.3 新增

| Agent | 职责 |
|-------|------|
| **Ontology Agent** | HGNC/UniProt/ChEBI/GO/SBO 对齐，所有 species 必须有标准 ID |
| **Pathway Planner Agent** | 通路分类 + 通路图谱构建 + cross-talk 识别 |
| **Pathway Specialist Agents (×10)** | 每条通路一个 Specialist，负责该通路的 Core/Feedback/Cross-talk/Perturbation/Validation 模块 |
| **Cross-talk Coordinator Agent** | 协调多通路 shared species + cross-talk edges |
| **Reaction Builder Agent** | Pathway Graph → Reaction IR v2（含状态机 + 组合反应） |
| **Mechanism Builder Agent** | Reaction IR + 动力学类型选择（强制 MM 用于磷酸化） |
| **ODE Builder Agent** | Reaction IR + 动力学 → 可执行 ODE 系统（含 DDE/SDE） |
| **SBML Grounder Agent** | ODE ↔ SBML 五级映射链建立 |
| **Calibration Agent** | 用 BioModels reference 做参数拟合 |
| **Validation Agent** | 五层 Validation Pyramid 执行 |
| **Hypothesis Agent** | 基于仿真结果生成可验证假设 |
| **Simulation Planner Agent** | 仿真配置（time-scale / sensitivity / perturbation） |
| **Parameter Agent** | 参数检索 + 通路隔离 + provenance 校验 |

### 9.3 Agent 职责详述

#### 9.3.1 Ontology Agent

- **输入**：用户问题中的实体名
- **输出**：标准化实体定义（HGNC/UniProt/ChEBI/GO/SBO ID）
- **职责**：
  - 查询 HGNC API 验证基因符号
  - 查询 UniProt API 获取蛋白 accession
  - 查询 ChEBI API 获取化学实体 ID
  - 查询 GO API 获取功能术语
  - 查询 SBO 获取机制本体
  - 失败时阻塞流水线（不静默降级）

#### 9.3.2 Pathway Planner Agent

- **输入**：用户问题 + Ontology Agent 输出
- **输出**：`pathway_class`（10 条通路之一 + cross-talk 标记）+ Pathway Graph
- **职责**：
  - 通路分类（规则优先：关键词 + BIOMD ID 映射；LLM 兜底）
  - 识别用户涉及的通路（可能多条）
  - 识别 cross-talk edges
  - 构建 Pathway Graph（含 shared species 标记）

#### 9.3.3 Pathway Specialist Agents (×10)

每条通路一个 Specialist，**这是 v4 与 v3 的根本区别**：

- **输入**：Pathway Graph（仅该通路的部分）+ pathway-specific 知识
- **输出**：该通路的 Core / Feedback / Cross-talk / Perturbation / Validation 模块
- **职责**：
  - 加载该通路的 5 模块定义
  - 选择该通路的专属模板（如 Apoptosis Specialist 选 Caspase_Cascade）
  - 应用该通路的 Validation 规则（如 NF-κB Specialist 检测振荡周期）
  - 与其他 Specialist 协调 cross-talk（通过 Cross-talk Coordinator）

#### 9.3.4 Cross-talk Coordinator Agent

- **输入**：多个 Pathway Specialist 的输出
- **输出**：cross-talk edges + shared species 同步策略
- **职责**：
  - 识别跨通路 shared species（如 Ras / AKT / MEK）
  - 协调 cross-talk edges 的注入（如 AKT→Raf Ser259）
  - 防止 cross-pathway parameter contamination（强制 pathway_tag 隔离）
  - 处理 cross-talk 时间尺度对齐

#### 9.3.5 Reaction Builder Agent

- **输入**：Pathway Graph + SBML reference
- **输出**：Reaction IR v2（含 species / reactions / state_machines / compartments / constraints）
- **职责**：
  - 将 Pathway Graph 转换为 Reaction IR v2
  - 构建 CompositeReaction（如 destruction complex）
  - 构建 State Machine（如 EGFR 状态机）
  - 执行 10 条 Validation Rules

#### 9.3.6 Mechanism Builder Agent

- **输入**：Reaction IR v2 + Parameter Agent 输出
- **输出**：带动力学类型的 Reaction IR v2
- **职责**：
  - 为每条 Reaction 选择动力学类型
  - **强制 MM 用于 phosphorylation / dephosphorylation / cleavage**（审计报告 §3.1 修复）
  - 强制 Hill 用于 transcription
  - 强制 DDE 用于延迟反馈
  - 执行 Constraint System 检查

#### 9.3.7 ODE Builder Agent

- **输入**：带动力学类型的 Reaction IR v2 + 参数 + 初始条件
- **输出**：可执行 ODE 系统（数学对象）
- **职责**：
  - 从 Reaction IR 渲染 ODE（可能用模板或动态生成）
  - 支持 DDE / SDE / hybrid
  - 支持 multi-compartment
  - 注入参数（含 provenance）

#### 9.3.8 SBML Grounder Agent

- **输入**：ODE 系统 + SBML reference
- **输出**：五级映射链（ODE ↔ Reaction ↔ SBML ↔ Parameter ↔ PMID）
- **职责**：
  - 建立 ODE 方程与 SBML reaction 的映射
  - 建立参数与 SBML parameter ID 的映射
  - 建立物种与 SBML species ID 的映射
  - 建立 PMID 关联

#### 9.3.9 Calibration Agent

- **输入**：ODE 系统 + BioModels reference + 实验数据
- **输出**：校准后参数 + 置信区间
- **职责**：
  - 用最小二乘 / MCMC 拟合参数
  - 输出参数置信区间
  - 标记无法校准的参数

#### 9.3.10 Validation Agent

- **输入**：仿真结果 + SBML reference + 通路特异规则
- **输出**：验证报告 + 置信度评分 + 失败短路信号
- **职责**：
  - 执行五层 Validation Pyramid
  - **skipped = pass=False**（审计报告 §7.2 修复）
  - **Track B 差异指标 = null**（审计报告 §7.2 修复）
  - 通路特异阈值（如 NF-κB 容忍更大 peak_time_diff）
  - 失败时阻塞流水线

#### 9.3.11 Hypothesis Agent

- **输入**：仿真结果 + 科学特征 + 验证报告
- **输出**：假设列表
- **职责**：
  - 基于仿真结果生成可验证假设
  - 每个假设含预测 + 实验设计 + 验证方式
  - 检索相关文献支持/反驳假设

#### 9.3.12 Simulation Planner Agent

- **输入**：ODE 系统 + 用户需求
- **输出**：仿真配置（time-scale / sensitivity / perturbation）
- **职责**：
  - 选择仿真类型（time-scale / steady state / oscillation / bistability / dose response / parameter scan / sensitivity / KO / drug）
  - 配置求解器（LSODA / DDE / SDE）
  - 配置多时间尺度

#### 9.3.13 Parameter Agent

- **输入**：Reaction IR v2 + pathway_tag
- **输出**：参数集（含 provenance）
- **职责**：
  - 检索 Parameter DB（强制 pathway_tag 过滤）
  - 强制 species / cell_line 过滤
  - 强制 threshold（< 0.3 不返回）
  - 关联 Evidence DB
  - 标记 unverified 参数

### 9.4 Agent 编排

v4 不再是固定线性流水线（审计报告 §6.1 修复），而是基于 `pathway_class` 的动态编排：

```
User Question
  ↓
Ontology Agent
  ↓
Pathway Planner (输出 pathway_class)
  ↓
[动态分支]
  ├── 单通路：1 个 Pathway Specialist
  └── 多通路：N 个 Pathway Specialist + Cross-talk Coordinator
  ↓
Reaction Builder
  ↓
Mechanism Builder (强制 MM)
  ↓
Parameter Agent (强制 pathway 隔离)
  ↓
ODE Builder
  ↓
SBML Grounder
  ↓
Calibration Agent (optional)
  ↓
Simulation Planner
  ↓
Simulation Engine
  ↓
Validation Agent (skipped = pass=False)
  ↓
Hypothesis Agent
  ↓
Report
```

**关键改动**：
- 不再有 `auto_fast` 模式跳过 PK/PD（审计报告 §6.4 修复）
- Validation 失败必须阻塞（审计报告 §6.6 修复）
- 每个 Agent 独立可重试（审计报告 §6.5 职责混乱修复）

### 9.5 回应 v1.0 审计报告

| 审计问题 | v4 解决方式 |
|---------|------------|
| §6.1 固定流水线 | 动态编排，基于 pathway_class 分支 |
| §6.2 Pathway Classifier 缺失 | Pathway Planner Agent |
| §6.2 Pathway Specialist 缺失 | 10 个 Pathway Specialist Agents |
| §6.2 Cross-talk Coordinator 缺失 | Cross-talk Coordinator Agent |
| §6.3 PreRouter 二分类 | Pathway Planner 做完整通路分类 |
| §6.4 误路由 | pathway_class 驱动 Specialist 选择 |
| §6.5 职责混乱 | 13 个 Agent 单一职责 |
| §6.5 验证职责重叠 | 统一到 Validation Agent |
| §6.5 RAG 检索分散 | 统一到 Parameter Agent + Mechanism Agent |
| §6.6 clarification 阻塞 | 仿真前多阶段 clarification |
| §6.6 全局字典无清理 | Agent 编排无全局状态，每次新会话 |
| §6.6 sandbox 重试无失败短路 | Validation Agent 失败短路 |
| §7.1 Calibration 缺失 | Calibration Agent |
| §7.1 Sensitivity 缺失 | Simulation Planner + Simulation Engine 支持 |
| §7.1 Hypothesis 缺失 | Hypothesis Agent |
| §7.1 Independent Biology Validator 缺失 | Validation Agent 独立于 sandbox |

---

## Part 10: Future Research Roadmap

### 10.1 v4（当前目标）

**主题**：Scientific Fidelity Foundation

- 10 通路完整覆盖
- Reaction IR v2（17 类机制 + 状态机 + 组合反应）
- ODE Framework v2（含 DDE / bistable / oscillatory / compartment）
- Validation Pyramid 五层
- 13 Agent 动态编排
- 强制 HGNC/UniProt/SBO 对齐
- 强制 pathway 隔离 + provenance

### 10.2 v5（中期目标）

**主题**：Quantitative Precision

- **Cancer Subtype Modeling**：mutation profile + subtype 特异参数集（如 KRAS G12D / TP53 R175H / PTEN loss）
- **Parameter Estimation Pipeline**：Bayesian inference + MCMC + profile likelihood
- **BRENDA / SABIO-RK 灌库**：酶动力学参数扩充
- **Patient-Derived Data**：cell line specific parameters（CCLE / TCGA）
- **Multi-Omics Integration**：转录组 + 蛋白组 + 磷酸化组数据整合
- **Uncertainty Quantification**：parameter uncertainty → prediction uncertainty
- **Bifurcation Analysis**：分岔图 + 续算方法
- **COMBINE Archive 支持**：OMEX 标准打包

### 10.3 v6（远期目标）

**主题**：Digital Cell

- **CellML 支持**：cellular model exchange standard
- **Rule-based Modeling**：BioNetGen / NFsim 多状态蛋白组合爆炸
- **Spatial Simulation**：VCell / CompuCell3D tissue-scale PDE
- **Stochastic Simulation**：Gillespie SSA + tau-leaping + hybrid
- **Single-Cell Modeling**：cell-to-cell variability + stochastic
- **Multi-Scale Modeling**：molecular + cellular + tissue
- **Digital Cell**：whole-cell simulation（如 Karr 2012 风格）
- **Patient-Specific Modeling**：digital twin

### 10.4 最终目标

建立 **Cancer Digital Cell**——一个能整合多组学数据、支持多尺度建模、可做患者特异性仿真的癌症数字细胞模型。这个目标超出 v4 范围，但 v4 的 Scientific Fidelity Foundation 是其不可绕过的基石。

### 10.5 回应 v1.0 审计报告

| 审计问题 | v4 解决 | v5/v6 解决 |
|---------|---------|-----------|
| §7.1 Cancer Subtype Modeling | v4 不解决（边界外） | v5 mutation profile + subtype |
| §7.1 Parameter Estimation | v4 不解决 | v5 Bayesian + MCMC |
| §7.1 Uncertainty Quantification | v4 不解决 | v5 UQ pipeline |
| §7.1 Stochastic Simulation | v4 不解决 | v6 Gillespie SSA |
| §7.1 Spatial Simulation | v4 仅 compartmental | v6 tissue-scale PDE |
| §7.1 Multi-state Protein | v4 状态机部分解决 | v5 rule-based 完整解决 |
| §7.1 Whole-genome scale | v4 不解决 | v6 digital cell |

---

## 附录：v4 设计与 v1.0 审计报告问题对照表

| 审计报告问题 | 审计章节 | v4 解决 Part | v4 解决方式 |
|------------|---------|-------------|------------|
| 通路覆盖 1/10 | §2.1 | Part 3 | 10 通路 × 5 模块化设计 |
| Cross-talk 被禁止 | §2.2 | Part 3 + Part 9 | Cross-talk Module + Cross-talk Coordinator Agent |
| 关键机制缺失 | §2.3 | Part 4 + Part 5 | 17 类机制 + 9 个新模板 |
| DomainChecker 通路盲化 | §2.4 | Part 3 + Part 7 | 通路特异 Validation Module + Level 1 检查 |
| 环路打破丢 inhibition | §2.5 | Part 8 | KG 不做环路打破，标记 feedback_loop |
| 本体未对齐 | §2.6 | Part 2 + Part 8 + Part 9 | Layer 1 + Ontology Service + Ontology Agent |
| MM 退化 | §3.1 | Part 5 + Part 9 | ODE Framework 原则 2 + Mechanism Builder 强制 MM |
| 无 bistable/oscillatory/DDE | §3.2 | Part 5 + Part 6 | 9 个新模板 + DDE 求解器 + 振荡/双稳态检测 |
| 硬编码参数 | §3.3 | Part 5 | 所有模板移除硬编码，Parameter Agent 注入 |
| 静默修改用户参数 | §3.4 | Part 5 | 废弃 _cascade_helpers |
| Cascade_Inhibition 反转 | §3.5 | Part 5 | 废弃 Cascade_Activation / Cascade_Inhibition |
| 求解器局限 | §3.6 | Part 6 | DDE / SDE / multi-compartment + 多时间尺度 |
| 9 维特征通用 | §3.7 | Part 6 | 振荡检测 + 双稳态检测 + 灵敏度 |
| Reaction IR 机制不全 | §4.1 | Part 4 | 17 类机制 + SBO term |
| 语义错误映射 | §4.2 | Part 4 | activation/inhibition 独立机制类型 |
| 信息损失 | §4.3 | Part 4 | SpeciesRef + Modifier + State Machine + CompositeReaction |
| 预校验局限 | §4.4 | Part 4 | 10 条 Validation Rules |
| state_transitions 空 | §4.5 | Part 4 | State Machine 一等公民 |
| 模板覆盖率 1/10 | §5.1 | Part 5 | 9 个新模板 + 动态生成 |
| 模板-通路不匹配 | §5.2 | Part 5 | 每通路专属模板 |
| 6 类关键模板缺失 | §5.3 | Part 5 | 全部新增 |
| 文档-代码矛盾 | §5.4 | Part 5 | 废弃矛盾模板 |
| 组合能力受限 | §5.5 | Part 4 + Part 5 | CompositeReaction + ODE composition |
| 降级映射风险 | §5.6 | Part 5 | 废弃降级映射 |
| 固定流水线 | §6.1 | Part 9 | 动态编排 |
| 三类 Agent 缺失 | §6.2 | Part 9 | Pathway Planner + 10 Specialist + Cross-talk Coordinator |
| PreRouter 二分类 | §6.3 | Part 9 | Pathway Planner 完整分类 |
| 误路由 | §6.4 | Part 9 | pathway_class 驱动 |
| 职责混乱 | §6.5 | Part 9 | 13 Agent 单一职责 |
| 状态流死锁 | §6.6 | Part 9 | 无全局状态 + 失败短路 |
| 孤立 Agent | §6.7 | Part 9 | 删除孤立定义 |
| Calibration 缺失 | §7.1 | Part 9 | Calibration Agent |
| Sensitivity 缺失 | §7.1 | Part 6 + Part 9 | Simulation Engine + Simulation Planner |
| Perturbation 缺失 | §7.1 | Part 6 | 11 类仿真含 KO / drug / mutation |
| Plugin 缺失 | §7.1 | Part 9 | Pathway Specialist 模块化 |
| DDE/SDE 缺失 | §7.1 | Part 5 + Part 6 | ODE Framework + Simulation Engine |
| Compartment 缺失 | §7.1 | Part 4 + Part 6 | Compartment schema + multi-compartment 求解 |
| Ontology 缺失 | §7.1 | Part 8 + Part 9 | Ontology Service + Ontology Agent |
| MM 退化（重申） | §7.2 | Part 5 + Part 9 | 强制 MM |
| evidence 字段空 | §7.2 | Part 8 | Evidence DB 强制 doi/figure_ref/cell_line |
| experiment 空 | §7.2 | Part 8 | Experiment DB 独立组件 |
| Oracle 默认 pass | §7.2 | Part 7 | skipped = pass=False |
| DomainChecker 偏科 | §7.2 | Part 3 + Part 7 | 通路特异 Validation Module |
| KG 边类型不足 | §7.2 | Part 8 | KG 边扩展到 17 类 |
| 数据源未灌库 | §7.3 | Part 7 + Part 8 | 强制灌库 KEGG/Reactome/UniProt/ChEMBL |
| Reference grounding 缺失 | §7.4 | Part 7 | 五级映射链 |
| Track B 误导 | §10.3 | Part 7 | Track B 差异 = null |
| species 字符串匹配 | §10.3 | Part 7 + Part 9 | Ontology ID 对齐 + Ontology Agent |
| species 硬编码 | §10.3 | Part 8 | 从文献提取 |
| SBML species="Unknown" | §10.3 | Part 8 | BioModels 元数据补全 |
| parameter 无 pathway | §5.1 | Part 8 | pathway_tag 强制 |

---

## 设计声明

本 v4 Scientific Architecture Design Document 基于 v1.0 审计报告的 26 项致命错误清单，逐一提出架构级解决方案。所有设计遵循 9 条原则（含用户追加的 4 条）：

1. Scientific Fidelity > Engineering Convenience
2. Pathway First, not ODE First
3. Mechanism First, Reaction is real biology
4. SBML is Ground Truth
5. Pathway is Network (supports Cross-talk)
6. Time-Space Dimension Principle
7. 组合机制与状态机
8. 溯源与通路隔离
9. 每个设计回应 v1.0 审计报告

文档不涉及代码实现、FastAPI、LangGraph、Python 等工程细节。所有设计满足可扩展、符合 Systems Biology、符合 SBML/BioModels、符合 Cancer Signaling、未来能支持真实科研的要求。

**文档版本**：v4.0
**完成日期**：2026-07-05
**下一步**：基于本架构文档启动 v4 工程实现规划（另立工程文档）
