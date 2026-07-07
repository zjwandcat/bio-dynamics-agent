# BioDynamics Agent v4 — P1 / P2 / P3 Integration Report

> **文档性质**：P1/P2 实现审计 + P3 Pathway Graph 设计与实施总结
> **审计对象**：`backend/app/ontology/`（P1）+ `backend/app/reaction_ir_v2/`（P2）+ `backend/app/adapters/`（P2 Adapter）+ `backend/app/pathway_graph/` + `backend/app/ode_templates_v2/` + `backend/app/solvers/` + `backend/app/ode_renderer_v2.py`（P3）
> **参考文档**：[BioDynamics_v4_Scientific_Architecture.md](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/BioDynamics_v4_Scientific_Architecture.md) / [BioDynamics_v4_Migration_Plan.md](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/BioDynamics_v4_Migration_Plan.md)
> **审计方法**：代码级静态审查 + 架构一致性比对 + 失败模式推演 + 端到端数据流追踪
> **硬约束遵守**：P1/P2 不可碰清单（sandbox.py / ode_templates/ / nodes_v2.py 核心 / rag_client.py / 前端）完整保留；P3 仅新增文件 + state.py/config.py/graph_v3.py 钩子注入
> **版本**：v1.1-integration-fixed
> **日期**：2026-07-06（v1.0 审计 → v1.1 P2→P3 集成修复）

---

## 执行摘要（Executive Summary）

| 维度 | 评分 | 判定 |
|------|------|------|
| P1 Ontology Agent 正确性 | 6.5 / 10 | 有结构缺陷 |
| P1 Ontology Agent 覆盖度 | 7 / 10 | 基本覆盖 |
| P2 Reaction IR v2 完整性 | 8 / 10 | 超额覆盖 |
| P2 Reaction IR v2 正确性 | 5 / 10 | 存在 P0 语义错误 |
| Adapter 无损性 | 4 / 10 | 有损（部分有意，部分错误） |
| P3 Pathway Graph 实现完整度 | 9 / 10 | 单元 + 集成测试均通过 |
| P3 ODE Template v2 实现完整度 | 8 / 10 | MM 恢复 + DDE + 双稳态齐全 |
| **P1→P2→P3 端到端就绪度** | **6.5 / 10** | **P2→P3 集成阻断已修复** |

**最终判定**：**PARTIAL GO — P3 可端到端启用，但 P2 内部 P0 阻断（PHOSPHORYLATION / mechanism 反向映射）未修复，需走 PathwayInitializer 硬编码通路规避**

P1 与 P2 在结构骨架上对齐 v4 架构文档，Feature Flag / State 共存 / 不可碰清单遵守良好。P3 已按 Migration Plan §Phase 3 完整实现 pathway_graph / ode_templates_v2 / solvers / ode_renderer_v2 四大模块。P2→P3 集成层 4 项 P0 阻断（B1/B2/S2/S3）已于 2026-07-06 修复，11 个真实 `ReactionIRv2.model_dump()` 驱动的集成测试全部通过，PathwayGraph.edges 不再为空。**剩余 P0 阻断**集中在 P2 内部（B3 PHOSPHORYLATION 反应物构建错误 + B4 build_from_pathway_graph mechanism 反向映射丢失），可通过 PathwayInitializer 硬编码通路规避，Phase 4 前必须修复以支持从 P1→P2→P3 真实用户输入链路。

---

## 1. P1 Ontology Audit Result

### 1.1 实现概览

| 组件 | 文件 | 行数 | 实现状态 |
|------|------|------|---------|
| Ontology Agent 主流程 | [ontology_agent.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/ontology_agent.py) | 408 | 实体抽取/合并/查询/Hook 节点齐全 |
| HGNC 客户端 | [hgnc_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/hgnc_client.py) | 96 | 重试3次 + 缓存7天 |
| UniProt 客户端 | [uniprot_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/uniprot_client.py) | 137 | gene_exact + organism_id:9606 |
| ChEBI 客户端 | [chebi_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/chebi_client.py) | 104 | stars=3 三星级过滤 |
| GO 客户端 | [go_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/go_client.py) | 108 | QuickGO + taxonId=9606 |
| 通路注册表 | [pathway_registry.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/pathway_registry.py) | 206 | 10 条通路 + KEGG/BioModels ID |
| SBO 常量 | [sbo_terms.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/sbo_terms.py) | 143 | 17 类机制 SBO term |
| 缓存层 | [_cache.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/_cache.py) | 96 | md5 key + TTL 7天 + JSON 文件 |
| 单元测试 | [test_ontology_agent.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/tests/test_ontology_agent.py) | 424 | 覆盖 5 个 P1 标准 |

### 1.2 正确性评分

| 维度 | 评分 | 依据 |
|------|------|------|
| API 调用正确性 | 6/10 | HGNC/UniProt/ChEBI 基本正确；[go_client.py:85](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/go_client.py) `evidence` 字段误取 `qualifier`（应取 `evidenceType`）；[uniprot_client.py:65](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/uniprot_client.py) organism 参数失效（硬编码 9606） |
| 缓存策略 | 8/10 | md5 key + TTL 7天 + 原子写入，实现合理 |
| 容错降级 | 9/10 | [ontology_agent.py:236-252](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/ontology_agent.py) 单实体失败降级 + Hook 全局 fail-safe，遵守"不抛异常"铁律 |
| Feature Flag 隔离 | 9/10 | Hook 仅返回 `{"v4_ontology_entities": result}`，测试显式校验 v3 字段黑名单 |

**综合正确性：7.4/10** | **覆盖度：7/10**（ChEBI/GO 无独立测试）

### 1.3 关键缺陷

**【P1-1】【sbo_terms.py:40,44,47】SBO ID 复用导致反向映射丢失**
`CYTOPLASM_TRANSLOCATION = "SBO:0000186"` 复用 `NUCLEAR_IMPORT`，`PROTEASOMAL_DEGRADATION = "SBO:0000218"` 复用 `UBIQUITINATION`，`INHIBITION = "SBO:0000169"` 复用 `SEQUESTRATION`。第 79 行 `SBO_TO_MECHANISM = {v: k for k, v in MECHANISM_TO_SBO.items()}` 反向映射时后写者覆盖前写者，导致 `nuclear_import` / `ubiquitination` / `sequestration` 反查丢失。

**【P1-2】【ontology_agent.py:32-60】EGF 双重身份导致本体查询路径错误**
`_KNOWN_PROTEINS` 与 `_KNOWN_CHEMICALS` 均包含 "EGF"。`_extract_entities_from_text` 先匹配 protein 集合，EGF 命中后加入 `seen`，导致 chemical 匹配被跳过。EGF 永远走 protein 路径，与 [chebi_client.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/chebi_client.py) 文档注释 "EGF → CHEBI:132945" 矛盾。

**【P1-3】【ontology_agent.py:63,150-153】species_type 类型集合不全**
`_SPECIES_TYPES = {"gene", "protein", "chemical", "pathway", "unknown"}`，但 v3 NER 实体 type 可能为 "ligand"/"receptor"/"kinase"/"drug"。`_merge_with_v3_entities` 第 151-153 行将这些类型一律降级为 "unknown"，导致 ligand/drug 类实体无法获得 ChEBI ID。

**【P1-4】【uniprot_client.py:56,65】organism 参数失效**
函数签名 `organism: str = "Homo sapiens"`，但第 65 行硬编码 `organism_id:9606`，第 56 行又把 organism 作为缓存键，导致不同 organism 共享同一查询结果但缓存键不同。

**【P1-5】【go_client.py:85】evidence 字段映射错误**
`evidence = ann.get("qualifier", "")` 注释为 evidence，但 QuickGO API 中 `qualifier` 是限定符（NOT, contributes_to），`evidenceType` 才是证据类型（EXP/IDA/IEA）。

### 1.4 失败用例

**用例 1**：输入 `"EGF binds EGFR"` — EGF 在 `_KNOWN_PROTEINS` 命中，`species_type="protein"`，`_annotate_single` 第 303 行 `if species_type == "chemical"` 不触发，EGF 不会查 ChEBI，`chebi_id=""`。

**用例 2**：v3 NER 返回 `[{"name": "EGF", "type": "ligand"}]` — `_merge_with_v3_entities` 第 151-153 行 `species_type = "ligand"` 不在 `_SPECIES_TYPES`，降级为 "unknown"，化学配体被误识别为基因。

**用例 3**：`sbo_term = "SBO:0000186"` — `SBO_TO_MECHANISM["SBO:0000186"] = "cytoplasm_translocation"`（后写者覆盖），`get_mechanism_name` 无法反查 `nuclear_import`。

---

## 2. P2 Reaction IR v2 Audit Result

### 2.1 实现概览

| 组件 | 文件 | 行数 | 实现状态 |
|------|------|------|---------|
| 核心 Schema | [schema.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/schema.py) | 304 | 6 个 Pydantic 模型 |
| 机制类型 | [mechanism_types.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/mechanism_types.py) | 208 | 17 类枚举 + 反向映射 |
| 构建器 | [reaction_builder.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/reaction_builder.py) | 327 | network_json → IR + pathway_graph → IR |
| 组合反应 | [composite_reaction.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/composite_reaction.py) | 275 | 5 条 Wnt 通路组合 |
| 状态机 | [state_machine.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/state_machine.py) | 210 | EGFR 状态机模板 |
| 约束检查 | [constraints.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/constraints.py) | 271 | 5 类守恒/酶/化学计量 |
| 校验规则 | [validation_rules.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/validation_rules.py) | 294 | 10 条规则 |
| 单元测试 | [test_reaction_ir_v2.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/tests/test_reaction_ir_v2.py) | 478 | Schema/规则/组合/状态机/约束 |

### 2.2 17 种 mechanism 类型支持矩阵

| # | MechanismType | Schema | CompositeReaction | StateMachine | reaction_builder 构建 |
|---|---|---|---|---|---|
| 1 | PHOSPHORYLATION | ✓ | ✓ | ✓ | **错误**（source 同时作 substrate+modifier） |
| 2 | DEPHOSPHORYLATION | ✓ | ✓ | ✓ | 默认 source→target |
| 3 | UBIQUITINATION | ✓ | ✓（Wnt step4） | ✓ | 默认 source→target |
| 4 | BINDING | ✓ | ✓（Wnt step2） | ✓ | 默认 source→target |
| 5 | DISSOCIATION | ✓ | ✓ | ✓ | 默认 source→target |
| 6 | DIMERIZATION | ✓ | ✓ | ✓ | 默认 source→target |
| 7 | COMPLEX_FORMATION | ✓ | ✓（Wnt step1） | ✓ | 默认 source→target |
| 8 | SEQUESTRATION | ✓ | ✓ | ✓ | 默认 source→target |
| 9 | CLEAVAGE | ✓ | ✓ | ✓ | 默认 source→target |
| 10 | GTP_GDP_EXCHANGE | ✓ | ✓ | ✓ | 默认 source→target |
| 11 | TRANSCRIPTION | ✓ | ✓ | ✓ | 默认 source→target |
| 12 | TRANSLATION | ✓ | ✓ | ✓ | 默认 source→target |
| 13 | NUCLEAR_IMPORT | ✓ | ✓ | ✓ | 默认 source→target |
| 14 | NUCLEAR_EXPORT | ✓ | ✓ | ✓ | 默认 source→target |
| 15 | CYTOPLASM_TRANSLOCATION | ✓ | ✓ | ✓ | 默认 source→target |
| 16 | DEGRADATION | ✓ | ✓ | ✓ | source 单 reactant |
| 17 | PROTEASOMAL_DEGRADATION | ✓ | ✓（Wnt step5） | ✓ | source 单 reactant |
| + | INHIBITION / ACTIVATION | ✓ | ✓ | ✓ | inhibition: target 作 substrate，source 作 inhibitor modifier |

### 2.3 正确性评分

| 维度 | 评分 | 依据 |
|------|------|------|
| Schema 设计 | 7/10 | Pydantic v2 + field_validator 完善区室/动力学/角色校验；但 [schema.py:47,140](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/schema.py) `species_type`/`reaction_type` 无验证器 |
| 17 类机制覆盖 | 6/10 | 枚举完整，但 [reaction_builder.py:285-321](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/reaction_builder.py) `build_from_pathway_graph` 把 v4 mechanism 当 v3 interaction 传入反向映射，8 种机制被 fallback 到 ACTIVATION |
| Validation Rules | 7/10 | 10 条全实现；Rule 9 transcription 仅"建议"非"必须"；Rule 10 cross-talk 触发条件不实际 |
| 约束检查 | 6/10 | [constraints.py:104](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/constraints.py) `pname.startswith("p"+rname)` 误命中 p53/p21；`check_enzymatic` 集合运算不严谨 |

**综合正确性：6.5/10**

### 2.4 关键缺陷

**【P0-1】【reaction_builder.py:174-181】PHOSPHORYLATION 反应物构建逻辑错误**
```python
elif mechanism == MechanismType.PHOSPHORYLATION:
    reactants.append(SpeciesRef(species_id=source_id, role="substrate"))
    products.append(SpeciesRef(species_id=target_id, role="product"))
    if source_id != target_id:
        modifiers.append(_make_modifier(source_id, "catalytic"))
```
source 同时作为 reactant(substrate) 和 modifier(catalytic)。对于 v3 edge `EGFR → pEGFR (interaction=phosphorylation)`：EGFR 既是底物又是催化激酶，化学计量与生物学语义均错误。下游 `check_enzymatic` 会因 enzyme 在 reactants 而不在 products 判定酶被消耗。

**【P0-2】【reaction_builder.py:285-321】build_from_pathway_graph 反向映射丢失机制**
第 290-296 行把 pathway_graph edge 的 `mechanism` 字段直接当作 `interaction` 传入 `build_from_network_json`。`_V3_INTERACTION_TO_V4_MECHANISM`（[mechanism_types.py:130-143](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/mechanism_types.py)）不包含 `dimerization`/`complex_formation`/`sequestration`/`gtp_gdp_exchange`/`nuclear_import`/`nuclear_export`/`cytoplasm_translocation`/`proteasomal_degradation` 等 8 种机制，全部 fallback 到 `ACTIVATION`。P3 Pathway Graph 输出的丰富机制语义在 P2 入口处被压扁。

**【P1-1】【composite_reaction.py:150-167】GSK3β modifier 注释与代码不一致**
注释"GSK3β 作为 catalytic modifier（不被消耗）"，但第 162 行 `modifiers=[]` 实际为空。

**【P1-2】【composite_reaction.py:170-185 vs pathway_graph/builder.py:49】ubiquitination 默认动力学不一致**
`composite_reaction.py` 第 173 行 `kinetics_type="mass_action"`，`pathway_graph/builder.py` 第 49 行 `ubiquitination: "Michaelis_Menten"`，两个模块对同一机制默认动力学不一致。

**【P1-3】【constraints.py:99-115】auto_generate_mass_conservation 误命中**
`pname.lower().startswith("p" + rname.lower())` 会误命中 rname="53"/pname="p53"（p53 是肿瘤抑制蛋白，非磷酸化形式）。

**【P1-4】【validation_rules.py:203-208】Rule 9 transcription 强度不足**
message 为"建议用 Hill"而非"必须用 Hill"，与架构 §4.3 强制 Hill 动力学不一致。

### 2.5 失败用例

**用例 1**：`EGFR → pEGFR (interaction=phosphorylation)` — source="EGFR" 作 substrate，target="pEGFR" 作 product，source≠target 时 EGFR 同时作 catalytic modifier。EGFR 既是底物又是酶，化学计量错误。

**用例 2**：`pathway_graph.edges[0].mechanism = "dimerization"` — `build_from_pathway_graph` 把 "dimerization" 当 interaction 传入 `v3_interaction_to_mechanism`，不在映射表，fallback 到 `ACTIVATION`，机制语义丢失。

**用例 3**：species 名 "53" 和 "p53" — `auto_generate_mass_conservation` 生成 `53 + p53 = 53_total` 守恒约束，表达式无意义。

---

## 3. Adapter Integrity Analysis

### 3.1 实现概览

| 组件 | 文件 | 行数 | 实现状态 |
|------|------|------|---------|
| v3→v4 适配器 | [v3_v4_adapter.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/adapters/v3_v4_adapter.py) | 107 | 复用 build_from_network_json |
| v4→v3 适配器 | [v4_v3_adapter.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/adapters/v4_v3_adapter.py) | 207 | 机制降级 + species.id 丢失 |
| 注册表 | [adapter_registry.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/adapters/adapter_registry.py) | 224 | 单例 + fail-safe + 指标 |
| 单元测试 | [test_adapter_v3_v4.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/tests/test_adapter_v3_v4.py) | 583 | round-trip + fail-safe |

### 3.2 v3→v4 转换正确性

字段映射基本完整：v3 `nodes{id,name,type}` → v4 `SpeciesV2{id, canonical_name, species_type}`，v3 `edges{source,target,interaction}` → v4 `ReactionV2{reactants, products, modifiers, reaction_type}`。

**问题**：[v3_v4_adapter.py:88](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/adapters/v3_v4_adapter.py) `ir.source = "v3_downgraded"` 强制覆盖 `build_from_network_json` 的 source 判断。即使传入 ontology_entities（build_from_network_json 标记 "v4_native"），adapter 又强制改为 "v3_downgraded"，矛盾。

### 3.3 v4→v3 转换正确性

**机制降级语义武断**（[v4_v3_adapter.py:39-41,49,55-57](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/adapters/v4_v3_adapter.py)）：
- `PHOSPHORYLATION → "activation"`（磷酸化也可能是抑制，如 p53 磷酸化 MDM2）
- `DEPHOSPHORYLATION → "inhibition"`（武断）
- `UBIQUITINATION → "inhibition"`（泛素化不必然导致降解/抑制）
- 默认 fallback "activation"（未知机制全部降级为 activation）

**species.id 丢失**（[v4_v3_adapter.py:102](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/adapters/v4_v3_adapter.py)）：`"id": name` 用 name 覆盖 id，丢失 v4 species.id（如 SP_001）。若 v4 有多个 species 共享 canonical_name（如 EGFR 在 membrane+cytoplasm 两个区室），v3 nodes 会出现 id 冲突。

### 3.4 双向一致性（round-trip）

测试覆盖：`test_roundtrip_preserves_node_names`、`test_roundtrip_node_count_diff_below_5_percent`、`test_roundtrip_edge_count_diff_below_5_percent` 均通过。

**实际 round-trip 一致性问题**：
- **机制语义不可逆丢失**：v3 `phosphorylation` → v4 `PHOSPHORYLATION` → v3 `activation`，round-trip 后 phosphorylation 变成 activation
- **species.id 不参与 round-trip**：v3 node.id="EGFR" → v4 species.id="SP_001" → v3 node.id="EGFR"（name 覆盖）

### 3.5 缺陷清单

**【P1-1】机制降级语义武断** — round-trip 后 phosphorylation 边变成 activation 边，不可逆。
**【P1-2】species.id 丢失** — 多 species 同名时 v3 node id 冲突。
**【P1-3】source 标记矛盾** — v3_v4_adapter 强制覆盖 build_from_network_json 的 source 判断。
**【P1-4】降解反应生成自环边** — [v4_v3_adapter.py:194-198](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/adapters/v4_v3_adapter.py) 降解反应 target 取 reactants 中的 substrate，即 source=target，生成自环边。

---

## 4. System Readiness Score

### 4.1 模块级评分

| 模块 | 评分 | 说明 |
|------|------|------|
| P1 Ontology Agent | 6.5/10 | API 客户端基本可用，降级策略完善，但字段映射缺陷（go_terms 类型、species_type 集合、SBO 反向映射）影响下游 |
| P2 Reaction IR v2 | 5/10 | Schema 设计合理，但 reaction_builder 存在 PHOSPHORYLATION 逻辑错误与 build_from_pathway_graph 反向映射丢失两个 P0 缺陷 |
| Adapter | 6/10 | fail-safe 设计优秀，但 v4→v3 机制降级语义丢失、species.id 丢失 |
| P3 Pathway Graph | 8/10 | 单元级完整（17/17 测试通过），集成级有字段名缺口 |
| P3 ODE Template v2 | 8/10 | MM 恢复 + DDE + 双稳态齐全，模板选择逻辑清晰 |

### 4.2 数据流级评分

| 数据流 | 评分 | 关键问题 |
|--------|------|---------|
| P1 → P2 | 5/10 | go_terms 类型不匹配（list[dict] vs list[str]）、species_type 不传递、pathway_class 需手动桥接 |
| P2 → P3 | **2/10 → 8/10**（v1.1 已修复） | **评审时**：字段名严重不匹配，P3 实际无法消费 P2 输出（详见 §6.2）。**修复后**：B1/B2/S2/S3 全部修复，11/11 集成测试通过 |
| P3 → Sandbox | 8/10 | ODE 渲染产物调用 sandbox.py 执行，沙盒不变，链路完整 |

### 4.3 综合就绪度

**综合评分（评审时刻 v1.0）：4.5/10** → **修复后 v1.1：6.5/10**（详见 §10）

当前 P1/P2/Adapter/P3 四模块**单元测试均可通过**，但**集成层面阻断性问题严重**。P2→P3 数据流存在两个 P0 级字段名不匹配，导致 P3 PathwayGraphBuilder 实际无法从 P2 `ReactionIRv2.model_dump()` 输出构建有效 edges。P3 单元测试通过仅因 fixture 直接提供了 `name` 字段，未覆盖真实 P2 序列化输出路径。

**v1.1 修复后状态**（2026-07-06）：
- P2→P3 集成层 4 项 P0 阻断（B1/B2/S2/S3）已修复，11 个真实 `ReactionIRv2.model_dump()` 驱动的集成测试全部通过
- 端到端就绪度从 4.5 提升至 6.5/10
- 剩余 2 项 P0 阻断（B3/B4）在 P2 内部，可通过 PathwayInitializer 硬编码通路规避，Phase 4 前必须修复

---

## 5. P3 Pathway Graph Design（已实施）

### 5.1 实施清单

| 子任务 | 文件 | 行数 | 状态 |
|--------|------|------|------|
| P3-1 Pathway Graph schema | [pathway_graph/schema.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathway_graph/schema.py) | 254 | ✓ PathwayNode/Edge/FeedbackLoop/CrossTalk/Temporal |
| P3-1 Pathway Graph builder | [pathway_graph/builder.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathway_graph/builder.py) | 327 | ✓ 纯规则，无 LLM |
| P3-1 10 通路初始化器 | [pathway_graph/initializer.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathway_graph/initializer.py) | 485 | ✓ EGFR/MAPK/PI3K/p53/Apoptosis/CellCycle/JAK-STAT/NF-κB/Wnt/TGF-β |
| P3-2 ODE Template v2 | [ode_templates_v2/](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates_v2/) | 4 文件 | ✓ MM 恢复 + oscillatory + bistable + DDE helpers |
| P3-3 Solvers | [solvers/](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/solvers/) | 3 文件 | ✓ DDE try-import + oscillation/bistability detector |
| P3-4 ODE Renderer v2 | [ode_renderer_v2.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_renderer_v2.py) | 287 | ✓ 模板选择 + 渲染 |
| P3-5 State + Config | [state.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/state.py) / [config.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/config.py) | - | ✓ v4_pathway_graph + v4_ode_system + V4_ODE_TEMPLATE_V2_ENABLED |
| P3-6 graph_v3 hooks | [graph_v3.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/graph_v3.py) | - | ✓ _pathway_graph_hook + _ode_template_v2_hook（无路由修改） |
| P3-7 单元测试 | [test_pathway_graph.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/tests/test_pathway_graph.py) | 17 用例 | ✓ 17/17 通过 |

### 5.2 核心设计决策

**5.2.1 Michaelis-Menten 恢复（审计 §3.1 致命错误修复）**

v3 [ode_templates/_mechanism_phosphorylation.j2](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates/_mechanism_phosphorylation.j2) 第 3-4 行注释 "TASK 4 修复：mass-action ONLY，移除 Michaelis-Menten 酶饱和项"，但架构 §4.3 明确 phosphorylation 强制 Michaelis-Menten。v4 [ode_templates_v2/_mechanism_phosphorylation_mm.j2](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates_v2/_mechanism_phosphorylation_mm.j2) 恢复 MM：

```python
k_cat = _get_param(tgt_name, "k_cat", _get_param(tgt_name, "k_phos", 0.1))
Km = _get_param(tgt_name, "Km", _get_param(tgt_name, "km", 0.1))
_Vmax = k_cat * src
_phos_rate = _Vmax * _sub_conc / (Km + _sub_conc) if (Km + _sub_conc) > 0 else 0.0
```

**5.2.2 DDE 支持（审计 §3.2）**

[solvers/dde_solver.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/solvers/dde_solver.py) 实现 jitcdde try-import 降级：

```python
try:
    from jitcdde import jitcdde, y as jitcdde_y, t as jitcdde_t
    _JITCDDE_AVAILABLE = True
except ImportError:
    _JITCDDE_AVAILABLE = False
    warnings.warn("jitcdde 不可用，DDE 将降级为 ODE 求解...")
```

p53_signaling / NF_kB / TGF_beta / JAK_STAT 四条通路 `requires_dde=True`，转录延迟分别为 60/30/60/30 分钟。

**5.2.3 多时间尺度（审计 §3.6）**

[ode_templates_v2/oscillatory_feedback.j2](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates_v2/oscillatory_feedback.j2) 实现 FAST/MEDIUM/SLOW 三档 max_step：
- FAST（磷酸化）：0.1 min
- MEDIUM（转录）：0.5 min
- SLOW（细胞周期）：1.0 min

**5.2.4 双稳态检测（审计 §3.4）**

[solvers/bistability_detector.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/solvers/bistability_detector.py) 实现 ON/OFF/TRANSIENT/UNKNOWN 四态判定，包含幅度检查（`amplitude < 1e-6` → UNKNOWN）防止平坦信号误判。

**5.2.5 Feature Flag 隔离**

`V4_PATHWAY_GRAPH_ENABLED`（Phase 1 已定义）+ `V4_ODE_TEMPLATE_V2_ENABLED`（Phase 3 新增），均默认 false。flag=false 时 `_pathway_graph_hook` / `_ode_template_v2_hook` 返回 None，v3 行为零侵入。Smoke test 验证：flag=false 时 P3 hooks 完全跳过。

**5.2.6 State 共存策略**

[state.py:201-213](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/state.py) 新增 `v4_pathway_graph` / `v4_ode_system` 字段，与 v3 `network_json` / `ode_model` 共存。`_ode_template_v2_hook` 写入 `v4_ode_system`，**不覆盖** `ode_model`，保证 v3 ODE 链路不受影响。

### 5.3 端到端验证

**Smoke Test 1（flag=false 隔离）**：✓ 通过
- Config flags 默认 false 验证
- graph_v3 hooks 返回 None，v3 行为不变

**Smoke Test 2（flag=true 端到端）**：✓ 通过
- p53_signaling PathwayGraph 构建：3 nodes, 1 edge, 1 feedback, 2 crosstalks, requires_dde=True, delay=60min
- ODE 渲染：8617 chars 代码，含 DDE_DELAY 和 Hill 函数

---

## 6. Data Flow Architecture (P1 → P2 → P3)

### 6.1 P1 → P2 数据流

**P1 输出**（[ontology_agent.py:257-261](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/ontology_agent.py)）：
```
{entities: [{name, hgnc_id, uniprot_id, chebi_id, go_terms, sbo_term, 
              species_type, verified, source}], pathway_class, warnings}
```

**P2 消费**（[reaction_builder.py:96-101](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/reaction_builder.py)）：
```python
for ent in ontology_entities.get("entities", []) or []:
    name = ent.get("name", "")
    if name:
        ont_by_name[name] = ent
```

**字段对齐**：

| P1 字段 | P2 消费位置 | 对齐状态 |
|---------|-------------|---------|
| `name` | reaction_builder.py:100 | ✓ |
| `hgnc_id` | reaction_builder.py:119 | ✓ |
| `uniprot_id` | reaction_builder.py:120 | ✓ |
| `chebi_id` | reaction_builder.py:121 | ✓ |
| `go_terms` | reaction_builder.py:122 | **✗ 类型不匹配**：P1 输出 `list[dict]`（[go_client.py:80-91](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ontology/go_client.py)），P2 [OntologyRef.go_terms](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/reaction_ir_v2/schema.py) 期望 `list[str]` |
| `sbo_term` | reaction_builder.py:123 | ✓ |
| `species_type` | 不消费 | ✗ P2 直接用 v3 `node.type`，忽略 P1 的 species_type 修正 |
| `pathway_class` | 不直接消费 | ✗ 需调用方手动桥接为 `pathway_tag` 参数 |

**就绪度：5/10** — 字段名对齐，但 go_terms 类型不一致、species_type 不传递、pathway_class 需手动桥接。

### 6.2 P2 → P3 数据流（**✅ 已修复**）

**修复前问题**（2026-07-06 早期版本）：
P3 builder 实际无法从 P2 `ReactionIRv2.model_dump()` 输出构建有效 edges。P3 单元测试通过仅因 fixture 直接提供了 `name` 字段，未覆盖真实 P2 序列化输出。

**修复方案**（在 P3 builder 端做 P2 schema 适配，保持 P2 不可碰）：
1. **B1 修复**：`_extract_nodes_from_ir` 优先读 `canonical_name`，fallback 到 `name`/`id`
2. **B2 修复**：`_extract_edges_from_ir` 接收 `id_to_name` 反查表；`_get_first_reactant/_product` 优先查 `species_id` → 反查表 → dict 字段（name/species/id）→ string fallback
3. **S2 修复**：sbo_term 从 P2 `MechanismType.sbo_term` 反查（避免硬编码 SBO 映射表）
4. **S3 修复**：`provenance` 嵌套字段路径（先读 `provenance.*` 嵌套，fallback 到平铺字段）

**修复后字段对齐**：

| P2 输出字段 | P3 期望字段 | 对齐状态 |
|-------------|-------------|---------|
| `species[i].canonical_name` | `species[i].canonical_name` (优先) | ✅ B1 已修复 |
| `reactions[i].reactants[0].species_id` | `PN_<id_to_name[species_id]>` | ✅ B2 已修复 |
| `reactions[i].reaction_type` | `reaction_type` | ✅ |
| `reactions[i].sbo_term` (P2 无) | 从 `MechanismType.sbo_term` 反查 | ✅ S2 已修复 |
| `reactions[i].provenance.source_sbml_reaction` | `edge.source_sbml_reaction` | ✅ S3 已修复 |
| `reactions[i].provenance.source_pmid` | `edge.source_pmid` | ✅ S3 已修复 |

**就绪度：8/10**（B1/B2/S2/S3 已修复，剩余 B3/B4 在 P2 内部，不阻断 P3 端到端）

**关键变更文件**：[pathway_graph/builder.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathway_graph/builder.py)（新增 `_build_species_id_index` / 改写 `_get_first_reactant/_product` / 改造 `_extract_nodes_from_ir` / 嵌套 provenance 读取 + MechanismType sbo_term 反查）

### 6.3 P3 → Sandbox 数据流

**P3 输出**：`v4_ode_system.ode_code`（渲染后的 Python 代码字符串）

**Sandbox 消费**：[sandbox.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/sandbox.py) `execute_simulation_code_v2(code)` 直接执行代码字符串，沙盒不变。

**就绪度：8/10** — 链路完整，但 v4 ODE 代码引用了 `jitcdde`（DDE 通路），sandbox 需确认该依赖可用或降级路径生效。

---

## 7. Blocking Issues Before P3 Full Activation

### 7.1 P0 阻断问题

| # | 问题 | 模块 | 影响 | 状态 |
|---|------|------|------|------|
| ~~**B1**~~ | ~~P2→P3 `species.canonical_name` vs `name` 字段名不匹配~~ | P2/P3 接口 | ~~P3 node.canonical_name 错误~~ | ✅ **已修复**（2026-07-06） |
| ~~**B2**~~ | ~~P2→P3 `reactants[].species_id` vs `name/species/id` 不匹配~~ | P2/P3 接口 | ~~**PathwayGraph.edges 为空**~~ | ✅ **已修复**（2026-07-06） |
| **B3** | reaction_builder PHOSPHORYLATION 反应物构建逻辑错误 | P2 内部 | 磷酸化反应化学计量错误，下游约束检查失效 | ⏳ **待修复**（Phase 4 前必须） |
| **B4** | build_from_pathway_graph mechanism 反向映射丢失 | P2 内部 | dimerization/complex_formation 等 8 种机制被压扁为 activation | ⏳ **待修复**（Phase 4 前必须） |

**修复方式汇总**：
- **B1** 修复：[pathway_graph/builder.py:228-233](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/pathway_graph/builder.py) `_extract_nodes_from_ir` 改为 `sp.get("canonical_name") or sp.get("name") or sp_id`
- **B2** 修复：新增 `_build_species_id_index` 在 `build()` 主入口构造 `id → canonical_name` 反查表；`_get_first_reactant/_product` 优先查 `species_id` 走反查表
- **S2 同步修复**：sbo_term 从 P2 `MechanismType.sbo_term` 反查（导入 `app.reaction_ir_v2.mechanism_types`）
- **S3 同步修复**：`provenance` 嵌套读取（先读 `provenance.*`，fallback 平铺）
- **B3 规避**：当前 P3 走 PathwayInitializer 硬编码通路，PHOSPHORYLATION 由 builder.py:47 直接映射到 Michaelis_Menten，与 P2 错误无关
- **B4 规避**：当前 P3 走 PathwayInitializer 硬编码通路，17 类 mechanism 语义在 initializer 内完整保留

### 7.2 P1 严重问题（建议修复）

| # | 问题 | 模块 | 影响 |
|---|------|------|------|
| S1 | P1→P2 `go_terms` 类型不匹配（list[dict] vs list[str]） | P1/P2 接口 | Pydantic 校验失败或下游崩溃 |
| S2 | P2 `ReactionV2` 缺 `sbo_term` 字段 | P2/P3 接口 | P3 拿不到 SBO term |
| S3 | P2 `provenance` 嵌套 vs P3 平铺字段路径 | P2/P3 接口 | P3 拿不到 SBML/PMID 溯源 |
| S4 | SBO ID 复用导致反向映射丢失 | P1 | nuclear_import/sequestration/ubiquitination 反查丢失 |
| S5 | Adapter v4→v3 机制降级语义武断 | Adapter | round-trip 后 phosphorylation 变 activation |
| S6 | Adapter v4→v3 species.id 丢失 | Adapter | 多 species 同名时 id 冲突 |
| S7 | EGF 双重身份导致本体查询路径错误 | P1 | EGF 永远走 protein 路径，不查 ChEBI |
| S8 | species_type 类型集合不全 | P1 | ligand/drug 类实体被降级为 unknown |

---

## 8. Recommended Implementation Plan for P3（已实施总结）

### 8.1 实施顺序与完成情况

按 [BioDynamics_v4_Migration_Plan.md](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/BioDynamics_v4_Migration_Plan.md) §Phase 3 推荐顺序执行：

| 阶段 | 任务 | 状态 | 备注 |
|------|------|------|------|
| 8.1 | pathway_graph/ 模块（schema + builder + initializer） | ✓ 完成 | 254 + 327 + 485 行 |
| 8.2 | ode_templates_v2/ 模板（MM 恢复 + oscillatory + bistable） | ✓ 完成 | 4 个 j2 文件 |
| 8.3 | solvers/ 模块（DDE try-import + detectors） | ✓ 完成 | 3 个 py 文件 |
| 8.4 | ode_renderer_v2.py | ✓ 完成 | 287 行，模板选择 + 渲染 |
| 8.5 | state.py + config.py 修改 | ✓ 完成 | v4_pathway_graph + v4_ode_system + V4_ODE_TEMPLATE_V2_ENABLED |
| 8.6 | graph_v3.py P3 hooks（无路由修改） | ✓ 完成 | _pathway_graph_hook + _ode_template_v2_hook |
| 8.7 | 单元测试 + 语法验证 | ✓ 完成 | 18/18 通过 |
| 8.8 | P1-P3 综合报告 | ✓ 完成 | 本文档 |
| **8.9** | **P2→P3 集成阻断修复（B1/B2/S2/S3）** | **✓ 完成** | **builder.py 适配 P2 schema + 11 个集成测试** |
| **8.10** | **P2→P3 集成测试** | **✓ 完成** | **[test_p2_to_p3_integration.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/tests/test_p2_to_p3_integration.py)，11/11 通过** |

### 8.2 不可碰清单遵守情况

| 不可碰文件 | 是否触碰 | 证据 |
|------------|---------|------|
| sandbox.py | 否 | P3 ODE 代码渲染后仍调用 `execute_simulation_code_v2`，沙盒不变 |
| ode_templates/ | 否 | P3 新建 `ode_templates_v2/` 目录，v3 模板完整保留 |
| nodes_v2.py 核心 | 否 | P3 仅在 graph_v3.py worker_ode 中插入 hook，未修改 nodes_v2 |
| rag_client.py | 否 | P3 不涉及 RAG |
| 前端代码 | 否 | P3 仅后端，前端 SSE 事件处理留待 Phase 4+ |

### 8.3 Feature Flag 矩阵

| Flag | 默认值 | 作用 | 隔离验证 |
|------|--------|------|---------|
| V4_ONTOLOGY_AGENT_ENABLED | false | P1 Ontology Agent hook | ✓ hook 返回 None |
| V4_REACTION_IR_ENABLED | false | P2 Reaction IR v2 hook | ✓ hook 返回 None |
| V4_REACTION_IR_ADAPTER_ENABLED | false | P2 Adapter 双向同步 | ✓ 不同步 network_json |
| V4_PATHWAY_GRAPH_ENABLED | false | P3 Pathway Graph hook | ✓ hook 返回 None |
| V4_ODE_TEMPLATE_V2_ENABLED | false | P3 ODE Template v2 hook | ✓ hook 返回 None，仍走 v3 ode_templates/ |

---

## 9. Risk Assessment (P3 Failure Modes)

### 9.1 已识别风险

| 风险 | 严重级 | 概率 | 影响 | 缓解措施 |
|------|--------|------|------|---------|
| ~~P2→P3 字段名不匹配导致 PathwayGraph.edges 为空~~ | ~~P0~~ | ~~100%~~ | ~~P3 完全失效~~ | ✅ 已修复（B1/B2/S2/S3，11/11 集成测试通过） |
| PHOSPHORYLATION 反应物构建错误传播到 ODE | P0 | 高（仅 P2→P3 路径） | 磷酸化动力学错误 | ⏳ 待修复 B3；当前走 PathwayInitializer 硬编码通路规避 |
| 8 种机制在 build_from_pathway_graph 被压扁 | P0 | 高（仅 P2→P3 路径） | 机制语义丢失 | ⏳ 待修复 B4；当前走 PathwayInitializer 硬编码通路规避 |
| jitcdde 不可用时 DDE 降级为 ODE 但语义可能偏差 | P1 | 中 | 振荡通路频率/相位失真 | 降级时在 v4_ode_system.warnings 标注 |
| 双稳态检测器对噪声敏感 | P1 | 中 | 误判 ON/OFF 态 | 已加 amplitude < 1e-6 → UNKNOWN 兜底 |
| Adapter round-trip 机制语义丢失 | P1 | 100% | v3↔v4 往返后 phosphorylation 变 activation | 扩展 v3 interaction 词表或保留 mechanism 扩展字段 |
| ~~P3 单元测试 fixture 不反映真实 P2 输出~~ | ~~P1~~ | ~~100%~~ | ~~集成时才发现字段不匹配~~ | ✅ 已修复（新增 11 个 P2→P3 集成测试用真实 `ReactionIRv2.model_dump()`） |

### 9.2 潜在风险

| 风险 | 严重级 | 触发条件 |
|------|--------|---------|
| 10 通路 initializer 硬编码与 P1 pathway_registry 不一致 | P2 | 人工维护两份通路清单 |
| Multi-timescale max_step 在 solve_ivp 中触发步长警告 | P2 | t_end 跨度大时 |
| bistable_switch.j2 detect_bistability 阈值硬编码 | P2 | 不同通路阈值差异大 |
| 当前 P3 走 PathwayInitializer，绕开 P2→P3 真实用户输入链路 | P1 | 用户输入经 P1/P2 后 P3 仍可能因 B3/B4 失败 |

---

## 10. Next Step Decision

### 10.1 判定

**PARTIAL GO — P3 可端到端启用（P2→P3 集成阻断已修复），但 P2 内部 2 项 P0 阻断（PHOSPHORYLATION / mechanism 反向映射）需走 PathwayInitializer 硬编码通路规避**

P3 实施完整度达标（9/10），29/29 测试通过（11 个 P2→P3 集成测试 + 18 个 P3 单元测试），Feature Flag 隔离零侵入。P2→P3 集成层 4 项 P0 阻断（B1/B2/S2/S3）已于 2026-07-06 全部修复，PathwayGraphBuilder 现在可以正确消费真实 `ReactionIRv2.model_dump()` 输出，PathwayGraph.edges 不再为空。剩余 2 项 P0 阻断（B3/B4）在 P2 内部，不影响 P3 自身。

**当前可启用状态**：
- ✅ `V4_PATHWAY_GRAPH_ENABLED=true` + `V4_ODE_TEMPLATE_V2_ENABLED=true` + 走 PathwayInitializer 硬编码通路 → P3 完全可用
- ⚠️ `V4_PATHWAY_GRAPH_ENABLED=true` + P2 真实用户输入（走 `build_from_network_json` / `build_from_pathway_graph`）→ B3/B4 仍会触发磷酸化化学计量错误和 8 种机制被压扁

### 10.2 已执行的修复（2026-07-06）

1. ✅ 修复 B1：P3 builder `_extract_nodes_from_ir` 优先 `canonical_name`
2. ✅ 修复 B2：P3 builder 新增 `_build_species_id_index` 反查表 + `_get_first_reactant/_product` 优先 `species_id` 解析
3. ✅ 同步 S2：sbo_term 从 P2 `MechanismType.sbo_term` 反查（导入 `app.reaction_ir_v2.mechanism_types`）
4. ✅ 同步 S3：`provenance` 嵌套字段路径（先读 `provenance.*`，fallback 平铺）
5. ✅ 新增 11 个 P2→P3 集成测试（用真实 `ReactionIRv2.model_dump()` 驱动 PathwayGraphBuilder）
6. ✅ Smoke test 验证：flag=false 时 P3 hooks 完全跳过，v3 行为不受影响

### 10.3 后续阶段前置条件

进入 Phase 4（Cross-talk Engine + Pathway Specialist）前必须完成：
- ⏳ 修复 B3：reaction_builder PHOSPHORYLATION 区分自/异磷酸化
- ⏳ 修复 B4：build_from_pathway_graph 直接构造 MechanismType 枚举
- ⏳ 修复 S1：P1→P2 `go_terms` 类型不匹配（list[dict] vs list[str]）
- ⏳ 修复 S5：Adapter v4→v3 机制降级语义武断（round-trip 后 phosphorylation 变 activation）
- ⏳ 修复 S7：P1 EGF 双重身份导致本体查询路径错误
- ⏳ 修复 S8：P1 species_type 类型集合不全

进入 Phase 4 前的最低条件是修复 B3 + B4，让真实用户输入链路（P1→P2→P3 端到端）可用。

---

## 附录 A：文件清单

### P3 新增文件（12 个）

```
backend/app/pathway_graph/
├── __init__.py
├── schema.py          (254 行)
├── builder.py         (327 → 380 行，P2→P3 适配：+id_to_name 反查表 + provenance 嵌套读取)
└── initializer.py     (485 行)

backend/app/ode_templates_v2/
├── __init__.py
├── _mechanism_phosphorylation_mm.j2
├── oscillatory_feedback.j2
├── bistable_switch.j2
└── _dde_helpers.j2

backend/app/solvers/
├── __init__.py
├── dde_solver.py
├── oscillation_detector.py
└── bistability_detector.py

backend/app/ode_renderer_v2.py    (287 行)
backend/tests/test_pathway_graph.py (18 用例)
backend/tests/test_p2_to_p3_integration.py (11 用例，P2→P3 集成测试)
```

### P3 修改文件（3 个）

```
backend/app/state.py        (+15 行：v4_pathway_graph + v4_ode_system)
backend/app/config.py       (+5 行：V4_ODE_TEMPLATE_V2_ENABLED)
backend/app/graph_v3.py     (+50 行：_pathway_graph_hook + _ode_template_v2_hook + worker_ode 集成)
```

---

## 附录 B：测试结果

| 测试套件 | 用例数 | 通过 | 失败 | 备注 |
|---------|--------|------|------|------|
| test_ontology_agent.py | - | - | - | P1 阶段已验证 |
| test_reaction_ir_v2.py | - | - | - | P2 阶段已验证 |
| test_adapter_v3_v4.py | - | - | - | P2 阶段已验证 |
| test_pathway_graph.py | 18 | 18 | 0 | P3 单元级全通过 |
| **test_p2_to_p3_integration.py** | **11** | **11** | **0** | **P2→P3 集成测试全通过**（真实 `ReactionIRv2.model_dump()` 驱动） |
| Smoke Test (flag=false) | 1 | 1 | 0 | v3 行为零侵入 |
| Smoke Test (flag=true, p53) | 1 | 1 | 0 | PathwayGraph + ODE 渲染成功 |
| **合计** | **31** | **31** | **0** | **全通过** |

### P2→P3 集成测试覆盖矩阵

| 阻断 | 测试用例 | 覆盖场景 |
|------|---------|---------|
| B1 (canonical_name vs name) | test_b1_canonical_name_priority / fallback_to_name / fallback_to_id | 真实 P2 schema 字段名 |
| B2 (species_id vs name) | test_b2_species_id_resolved_via_index / fallback_to_dict_keys / string_list_compat | SpeciesRef.species_id 解析 + v3 字符串列表兼容 |
| S2 (sbo_term 缺失) | test_s2_sbo_term_inferred_from_mechanism | 从 P2 MechanismType 反查 |
| S3 (provenance 嵌套) | test_s3_provenance_nested_path / flat_path_fallback | 嵌套路径 + 平铺 fallback |
| 端到端 | test_p2_to_p3_integration_end_to_end | 真实 `ReactionIRv2.model_dump()` → PathwayGraphBuilder |
| 容错 | test_unresolvable_edge_warns_but_does_not_block | dangling species_id 跳过 + warning |

---

**报告结束**
