# BioDynamics Agent v4 — Migration Plan

> **文档性质**：迁移计划文档（Architecture-driven Development, ADD）
> **前置文档**：BioDynamics_v4_Scientific_Architecture.md（架构设计）/ BioDynamics_Agent_严格结构审计报告_v1.0.md（现状审计）
> **方法论**：Architecture → Migration Plan → Module Refactor → Feature Refactor → Validation → Next Module（不允许跳序）
> **核心约束**：每个 Phase 必须保证 git commit → 测试 → 继续；几乎不会崩
> **版本**：v4.0-migration
> **日期**：2026-07-05

---

## 总纲（General Principles）

### 1. Architecture-driven Development（ADD）方法论

v4 迁移严格遵循 ADD 顺序，禁止跳序：

```
Architecture (已完成)
    ↓
Migration Plan (本文档)
    ↓
Module Refactor (按 Phase 顺序)
    ↓
Feature Refactor (Module 内部)
    ↓
Validation (冒烟测试 + 回归测试)
    ↓
Next Module (下一 Phase)
```

**铁律**：
1. 每个 Phase 完成前不得启动下一 Phase
2. 每个 Phase 必须 git commit + 测试通过才推进
3. 每个 Phase 必须保证系统可运行（不能崩）
4. 每个 Phase 必须可独立回滚（不依赖后续 Phase）
5. 每个 Phase 必须明确"不可碰"清单

### 2. Phase 概览

| Phase | 主题 | 范围 | 预计复杂度 | Breaking Change |
|-------|------|------|-----------|----------------|
| P1 | Scientific Layer 建立与 Ontology Agent | 新增 layer / app/ontology/ | 低 | 否（仅新增） |
| P2 | Reaction IR v2 + Adapter | 新增 app/reaction_ir_v2/ + Adapter | 中 | 否（v3 IR 保留） |
| P3 | ODE Template v2 + 9 新模板 | 新增 ode_templates_v2/ | 中 | 否（旧模板保留） |
| P4 | Cross-talk Engine + Pathway Specialist | 新增 app/pathways/ | 高 | 否（新模块） |
| P5 | SBML Grounding + Validation Pyramid | 新增 app/validation_v2/ | 中 | 否（旧 validator 保留） |
| P6 | Hypothesis Layer + Agent 完整编排 | 新增 app/hypothesis/ + 路由切换 | 高 | 否（旧路由保留） |
| P7 | v3 弃用标记 + 清理 | 标记 deprecated | 低 | 是（最终清理） |

**总原则**：P1-P6 全部为"新增共存"模式，不删旧代码；P7 才做最终清理。任何 Phase 出问题可立即回滚到上一 Phase。

---

## Compatibility Strategy（兼容性策略）

> **元原则**：每设计一个新模块，都必须回答以下 6 个问题：
> 1. 旧模块怎么办？
> 2. 保留？Adapter？删除？弃用？
> 3. 什么时候删？
> 4. 是否影响 API？
> 5. 是否影响 SSE 事件？
> 6. 是否影响数据库？

### 约束 1：Feature Flag 机制

每个 Phase 上线时，用环境变量控制开关，可在 v3/v4 之间随时切换，出问题立即回滚到 v3 路径，不需要 git revert。

**Feature Flag 清单**（在 `.env` 中管理）：

```env
# Phase 1 flags
V4_ONTOLOGY_AGENT_ENABLED=false
V4_PATHWAY_GRAPH_ENABLED=false

# Phase 2 flags
V4_REACTION_IR_ENABLED=false
V4_REACTION_IR_ADAPTER_ENABLED=false

# Phase 3 flags
V4_ODE_TEMPLATE_V2_ENABLED=false
V4_DDE_SOLVER_ENABLED=false
V4_BISTABLE_TEMPLATE_ENABLED=false
V4_OSCILLATORY_TEMPLATE_ENABLED=false

# Phase 4 flags
V4_PATHWAY_PLANNER_ENABLED=false
V4_PATHWAY_SPECIALIST_ENABLED=false
V4_CROSSTALK_COORDINATOR_ENABLED=false

# Phase 5 flags
V4_SBML_GROUNDER_ENABLED=false
V4_VALIDATION_PYRAMID_ENABLED=false
V4_CALIBRATION_AGENT_ENABLED=false

# Phase 6 flags
V4_HYPOTHESIS_AGENT_ENABLED=false
V4_DYNAMIC_ROUTING_ENABLED=false

# Phase 7 flags
V3_DEPRECATED_CLEANUP=false
```

**默认值**：所有 flag 默认 `false`。Phase 完成验证后逐步切 `true`。生产环境通过环境变量注入，不修改代码。

**回滚策略**：将对应 flag 切回 `false`，系统自动走 v3 路径。无需 git revert，无需重新部署。

### 约束 2：Adapter Pattern 显式定义

v3 的 `network_json` 与 v4 的 `ReactionIRv2` 之间建立双向 Adapter：

```
v3 network_json  ──→  [V3V4Adapter]  ──→  ReactionIRv2 (降级模式)
ReactionIRv2     ──→  [V4V3Adapter]  ──→  v3 network_json (兼容旧模板)
```

**Adapter 职责**：
- `v3_to_v4(network_json: dict) -> ReactionIRv2`：将 v3 的 network_json 转换为 v4 ReactionIRv2 的降级形式（无状态机、无组合反应、无 compartment）
- `v4_to_v3(reaction_ir: ReactionIRv2) -> dict`：将 v4 ReactionIRv2 转换为 v3 network_json 格式，供旧模板渲染

**过渡期策略**：
- P2 阶段：新 IR 可驱动旧模板（通过 v4_to_v3 Adapter）
- P2 阶段：旧 IR 也能喂给新模板（通过 v3_to_v4 Adapter）
- 迁移完成后（P7）：Adapter 标记为 deprecated，但保留 90 天

**Adapter 失败处理**：
- 转换失败时记录 warning + 降级到 v3 路径
- 不阻塞流水线
- 失败次数 > 阈值时自动禁用 v4 路径（fail-safe）

### 约束 3：State 字段共存策略

`BioDynamicsState` 在迁移期间同时携带 v3 和 v4 字段，**不替换，共存**。

**字段命名规则**：
- v3 字段保持原名（如 `network_json`、`parameters`、`rag_retrieved_params`）
- v4 字段以 `v4_` 前缀共存（如 `v4_reaction_ir`、`v4_pathway_class`、`v4_ontology_entities`、`v4_validation_report`）

**读写规则**：
- 旧代码读 v3 字段（如 `state["network_json"]`）
- 新代码读 v4 字段（如 `state["v4_reaction_ir"]`）
- 新代码写入 v4 字段时，同时通过 Adapter 同步到 v3 字段（保证旧代码可读）
- 旧代码写入 v3 字段时，**不同步到 v4 字段**（v4 字段保持 None，新代码检测到 None 时走 v3 路径）

**清理时机**：P7 阶段统一删除 v3 字段，State 精简为纯 v4 结构。

### 约束 4：SSE 事件契约冻结

**前端在迁移期间不动**。所有新增的 v4 内部数据结构不得改变 SSE 事件格式。

**冻结清单**：
- `chat_message` 事件：字段名、字段类型、字段顺序不变
- `workflow_v3_state` 事件：保留事件名（即使内部是 v4，事件名仍为 v3_state）
- `clarification_needed` / `clarification_resolved` 事件：格式不变
- `pkpd_profile` / `drug_regimen` / `dose_response` / `combination_synergy` 事件：格式不变

**追加规则**：
- v4 产生的新数据（如 `validation_report` 的新字段、`hypothesis_list`）以**追加字段**方式注入现有事件
- 不得删除或重命名现有事件字段
- 新字段必须 nullable，前端不识别时忽略

**新增事件规则**：
- 仅当 v4 产生全新功能（如 Hypothesis Layer）才允许新增事件类型
- 新增事件必须以 `v4_` 前缀（如 `v4_hypothesis_generated`）
- 前端可不订阅，不影响现有功能

### 约束 5：ChromaDB 不可逆操作禁令

迁移期间**禁止** `delete_collection()` / `recreate()` / `drop_collection()`。

**新建 v4 collection 命名**：
- `biodynamics_mechanism_v4`（v3: `biodynamics_mechanism`）
- `biodynamics_parameter_v4`（v3: `biodynamics_params`）
- `biodynamics_evidence_v4`（v3: `biodynamics_evidence`）
- `biodynamics_experiment_v4`（v3: 空，新建）
- `biodynamics_pathway_graph_v4`（新增）

**数据迁移策略**：
1. P1-P6：v4 collection 独立灌库，v3 collection 不动
2. 每个 Phase 验证 v4 检索正常后才推进
3. P7 阶段：v3 collection 保留 30 天（只读），30 天后删除
4. 期间提供 `migrate_v3_to_v4.py` 脚本（带 dry-run 模式）

**回滚策略**：
- v4 collection 数据问题：禁用对应 v4 flag，回到 v3 collection
- v3 collection 数据问题：从备份恢复（迁移前必须备份）

### 约束 6：每阶段的"冒烟测试"定义

不能只说"跑现有测试"。每个 Phase 必须定义最小冒烟用例：

| Phase | 冒烟用例输入 | 期望行为 |
|-------|------------|---------|
| P1 | "EGF activates EGFR" | v3 行为不变；v4 ontology 字段填充但不生效（flag=false） |
| P2 | "EGF activates EGFR" | ReactionIRv2 生成成功；Adapter 转 v3 格式；仿真结果与 v3 一致（diff < 5%） |
| P3 | "p53 Mdm2 feedback" | 新模板（Oscillatory_Feedback）渲染成功；旧模板（Signaling_Cascade_Phos）仍可用于 EGF |
| P4 | "EGF + PI3K crosstalk" | Pathway Planner 识别双通路；Cross-talk Coordinator 激活；shared species（Ras/AKT）正确同步 |
| P5 | 输入含 BIOMD ID 的假设（如 "BIOMD0000000205"） | Validation Agent 阻塞不合格仿真；skipped 状态 pass=False |
| P6 | 任意仿真完成 | Hypothesis Agent 输出假设列表（≥1 条）；假设含预测 + 实验设计 + 验证方式 |
| P7 | 任意 v3 输入 | v3 字段被弃用标记；运行时 warning 但不阻塞 |

**冒烟测试执行**：
- 每个 Phase 完成后必须跑对应冒烟用例
- 冒烟用例失败 = Phase 未完成
- 冒烟用例通过才允许 git commit + 推进下一 Phase

### 约束 7：依赖隔离原则

新引入的第三方库必须做 try-import 降级：

```python
# 依赖隔离模板（伪代码，仅示意）
try:
    from jitcdde import jitcdde as _DDESolver
    DDE_AVAILABLE = True
except ImportError:
    DDE_AVAILABLE = False
    # 降级为 ODE 近似 + warning

try:
    from SALib import Sample, Analyze
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    # 降级为 local sensitivity only

try:
    import roadrunner
    ROADRUNNER_AVAILABLE = True
except ImportError:
    ROADRUNNER_AVAILABLE = False
    # 降级到 Track B
```

**新依赖清单**（按 Phase）：

| 依赖 | 用途 | Phase | 降级策略 |
|------|------|-------|---------|
| `jitcdde` | DDE 求解器（延迟反馈） | P3 | 降级为 ODE 近似 + warning |
| `SALib` | Sobol/Morris 全局灵敏度 | P5 | 降级为 local sensitivity only |
| `pyviabilit`（或类似） | 双稳态检测 | P5 | 降级为多初值扫描 |
| `lmfit` | 参数校准 | P5 | 降级为简单最小二乘 |
| `emcee` | MCMC（v5 预留） | P5+ | 降级为 grid search |
| `bioservices` | KEGG/Reactome API | P1 | 降级为本地缓存 |

**迁移期间不强制安装新依赖**，缺少依赖时降级运行而非崩溃。P7 阶段统一升级 requirements.txt。

### 约束 8："不可碰"清单

每个 Phase 明确列出本阶段绝对不许动的文件，防止手痒重构：

| Phase | 不可碰文件 |
|-------|-----------|
| P1 | sandbox.py / ode_templates/ / rag_client.py / graph_v3.py 核心路由 |
| P2 | sandbox.py / ode_templates/ / graph_v3.py 核心路由 |
| P3 | graph_v3.py 的核心路由逻辑 / rag_client.py / sandbox.py |
| P4 | ode_templates/（只加新模板，不改旧的）/ rag_client.py |
| P5 | ode_templates/ / report_renderer.py / 前端代码 |
| P6 | 全部 v3 文件（只新增 v4 文件） |
| P7 | 全部 v4 文件（只清理 v3 deprecated） |

**违反规则**：任何 Phase 修改了"不可碰"清单中的文件 = Phase 失败，必须 git revert 重做。

---

## Phase 1: Scientific Layer 建立与 Ontology Agent

### 1.1 目标

建立 v4 Scientific Stack 的 Layer 1（Biological Knowledge Layer）与对应 Agent（Ontology Agent），为后续 Phase 提供标准化的物种/通路/机制定义。**不替换任何 v3 模块**，仅新增。

### 1.2 修改目录

```
backend/app/
├── ontology/                    【新增目录】
│   ├── __init__.py
│   ├── hgnc_client.py           # HGNC API 客户端
│   ├── uniprot_client.py        # UniProt API 客户端
│   ├── chebi_client.py          # ChEBI API 客户端
│   ├── go_client.py             # GO API 客户端
│   ├── sbo_terms.py             # SBO term 常量定义
│   ├── pathway_registry.py      # 10 通路 + KEGG/Reactome ID 映射
│   ├── ontology_agent.py        # Ontology Agent 主逻辑
│   └── cache/                   # 本地缓存（JSON）
└── state.py                     【修改：新增 v4_ontology_entities 字段】
```

### 1.3 删除

无（P1 仅新增，不删任何 v3 代码）。

### 1.4 新增

- `backend/app/ontology/` 完整目录
- `state.py` 新增字段：`v4_ontology_entities: dict`（默认 None）
- `.env.example` 新增 `V4_ONTOLOGY_AGENT_ENABLED=false`
- `backend/tests/test_ontology_agent.py`（冒烟测试）

### 1.5 是否 Breaking Change

**否**。

- v3 代码完全不受影响（flag=false 时 Ontology Agent 不执行）
- API 接口不变
- SSE 事件不变
- 数据库不变

### 1.6 如何验证

**冒烟测试**（P1）：

| 用例 | 输入 | 期望 |
|------|------|------|
| 1 | "EGF activates EGFR" | v3 行为不变；仿真结果与 P0 基线 diff < 1% |
| 2 | "EGF activates EGFR"（flag=true） | v4_ontology_entities 填充：EGF → HGNC:3229, EGFR → HGNC:3236, UniProt P00533 |
| 3 | "EGF activates EGFR"（flag=false） | v4_ontology_entities 为 None；v3 行为完全不变 |

**回归测试**：
- 跑全量 v3 测试套件，确保 0 失败
- 跑 EGF-EGFR 端到端测试，pEGFR 达峰时间 5-10 min 仍满足

### 1.7 是否可回滚

**是，完全可回滚**。

- 回滚方式：`V4_ONTOLOGY_AGENT_ENABLED=false`
- 回滚后：v4_ontology_entities 字段保留（无害），Ontology Agent 不执行
- 数据库：无影响（Ontology Agent 不写库，仅读 API + 缓存）
- Git：`git revert` P1 commit 即可彻底回滚

### 1.8 不可碰清单（P1）

- `sandbox.py`
- `ode_templates/`（全部）
- `rag_client.py`
- `graph_v3.py` 的核心路由逻辑（仅允许在 pre_router 前加 hook，不改路由）
- `nodes_v2.py`（不改）
- 前端代码

### 1.9 Ontology Agent 职责（简要）

- 输入：用户问题中的实体名
- 输出：标准化实体定义（HGNC/UniProt/ChEBI/GO/SBO ID）
- 失败处理：API 失败时降级为本地缓存，缓存 miss 时标记 `unverified`，不阻塞流水线
- 性能：缓存优先（TTL 7 天），API 失败重试 3 次后降级

### 1.10 完成标准

- [ ] `backend/app/ontology/` 目录建立
- [ ] HGNC/UniProt/ChEBI/GO 客户端实现 + 缓存
- [ ] 10 通路 + KEGG/Reactome ID 映射表
- [ ] SBO term 常量定义（17 类机制）
- [ ] Ontology Agent 主逻辑
- [ ] state.py 新增 v4_ontology_entities 字段
- [ ] Feature Flag 接入
- [ ] 冒烟测试 3 个用例全部通过
- [ ] 回归测试 0 失败
- [ ] git commit

---

## Phase 2: Reaction IR v2 + Adapter Pattern

### 2.1 目标

建立 v4 Scientific Stack 的 Layer 3（Reaction IR Layer），实现 17 类机制 + CompositeReaction + State Machine + Compartment + Constraint。通过 Adapter 与 v3 network_json 双向兼容。

### 2.2 修改目录

```
backend/app/
├── reaction_ir_v2/              【新增目录】
│   ├── __init__.py
│   ├── schema.py                # Species/Reaction/CompositeReaction/StateMachine/Compartment/Constraint schema
│   ├── mechanism_types.py       # 17 类机制 + SBO term
│   ├── validation_rules.py      # 10 条 Validation Rules
│   ├── reaction_builder.py      # Pathway Graph → Reaction IR v2
│   ├── state_machine.py         # 蛋白质状态机
│   ├── composite_reaction.py    # 组合反应（如 destruction complex）
│   └── constraints.py           # 5 类约束（mass conservation / steady state / ...）
├── adapters/                    【新增目录】
│   ├── __init__.py
│   ├── v3_v4_adapter.py         # v3 network_json → ReactionIRv2（降级模式）
│   ├── v4_v3_adapter.py         # ReactionIRv2 → v3 network_json（兼容旧模板）
│   └── adapter_registry.py      # Adapter 注册 + fail-safe
├── graph_v3.py                  【修改：在 n2_mechanistic_planner 后加 v4 hook】
└── state.py                     【修改：新增 v4_reaction_ir 字段】
```

### 2.3 删除

无（v3 的 `reaction_ir.py` 保留，P7 才清理）。

### 2.4 新增

- `backend/app/reaction_ir_v2/` 完整目录
- `backend/app/adapters/` 完整目录
- `state.py` 新增字段：`v4_reaction_ir: dict`（默认 None）
- `.env.example` 新增 `V4_REACTION_IR_ENABLED=false` + `V4_REACTION_IR_ADAPTER_ENABLED=false`
- `backend/tests/test_reaction_ir_v2.py`
- `backend/tests/test_adapter_v3_v4.py`

### 2.5 是否 Breaking Change

**否**。

- v3 的 `reaction_ir.py` 不动
- v3 的 `network_json` 流程不变
- Adapter 失败时降级到 v3 路径（fail-safe）

### 2.6 如何验证

**冒烟测试**（P2）：

| 用例 | 输入 | 期望 |
|------|------|------|
| 1 | "EGF activates EGFR" | v3 行为不变（flag=false） |
| 2 | "EGF activates EGFR"（IR flag=true, Adapter flag=false） | v4_reaction_ir 生成成功；仿真仍走 v3 路径 |
| 3 | "EGF activates EGFR"（IR flag=true, Adapter flag=true） | v4_reaction_ir 生成；v4_to_v3 Adapter 转 network_json；仿真结果与 v3 diff < 5% |
| 4 | "Wnt destruction complex"（IR flag=true） | v4_reaction_ir 含 CompositeReaction（三步耦合） |
| 5 | "EGFR phosphorylation"（IR flag=true） | v4_reaction_ir 含 phosphorylation 机制 + MM 动力学（强制） |

**回归测试**：
- 全量 v3 测试套件
- EGF-EGFR 端到端：pEGFR 达峰时间 5-10 min
- Adapter 转换一致性：v3 → v4 → v3 往返 diff < 5%

### 2.7 是否可回滚

**是**。

- 回滚方式：`V4_REACTION_IR_ENABLED=false`
- 回滚后：v4_reaction_ir 字段保留（无害），系统完全走 v3 路径
- 数据库：无影响
- Git：`git revert` P2 commit

### 2.8 不可碰清单（P2）

- `sandbox.py`
- `ode_templates/`（全部）
- `nodes_v2.py` 的核心仿真逻辑
- `rag_client.py`
- 前端代码

### 2.9 完成标准

- [ ] Reaction IR v2 schema（6 个组件）
- [ ] 17 类机制 + SBO term 定义
- [ ] 10 条 Validation Rules
- [ ] CompositeReaction（含 destruction complex 示例）
- [ ] State Machine（含 EGFR 状态机示例）
- [ ] v3_v4_adapter + v4_v3_adapter
- [ ] Adapter fail-safe 机制
- [ ] state.py 新增 v4_reaction_ir 字段
- [ ] Feature Flag 接入
- [ ] 冒烟测试 5 个用例全部通过
- [ ] 回归测试 0 失败
- [ ] git commit

---

## Phase 3: ODE Template v2 + 9 新模板

### 3.1 目标

建立 v4 ODE Framework 的 9 个新模板（Bistable_Switch / Oscillatory_Feedback / Transcriptional_Delay / Nuclear_Transport / Ubiquitination_Cascade / Destruction_Complex / Caspase_Cascade / Cyclin_CDK_Toggle / Transcription_Factor），引入 DDE 求解器（依赖隔离）。

### 3.2 修改目录

```
backend/app/
├── ode_templates_v2/            【新增目录】
│   ├── __init__.py
│   ├── bistable_switch.j2
│   ├── oscillatory_feedback.j2
│   ├── transcriptional_delay.j2
│   ├── nuclear_transport.j2
│   ├── ubiquitination_cascade.j2
│   ├── destruction_complex.j2
│   ├── caspase_cascade.j2
│   ├── cyclin_cdk_toggle.j2
│   ├── transcription_factor.j2
│   ├── _mechanism_phosphorylation_mm.j2   # 恢复 MM 动力学
│   └── _dde_helpers.j2                     # DDE 求解器辅助
├── solvers/                     【新增目录】
│   ├── __init__.py
│   ├── dde_solver.py            # jitcdde 封装（try-import 降级）
│   ├── bistability_detector.py  # 双稳态检测
│   └── oscillation_detector.py  # 振荡检测
├── ode_renderer_v2.py           【新增：从 ReactionIRv2 渲染 ODE】
└── state.py                     【修改：新增 v4_ode_system 字段】
```

### 3.3 删除

无（v3 的 `ode_templates/` 保留，v3 模板仍可用）。

**关键约束**：v3 的 `Cascade_Activation.j2` 与 `Cascade_Inhibition.j2` **不删除**，仅标记 deprecated（添加注释），P7 才真正删除。

### 3.4 新增

- `backend/app/ode_templates_v2/` 完整目录（9 个新模板 + 2 个辅助）
- `backend/app/solvers/` 完整目录（DDE 求解器 + 检测器）
- `backend/app/ode_renderer_v2.py`（从 ReactionIRv2 渲染 ODE）
- `state.py` 新增字段：`v4_ode_system: dict`（默认 None）
- `.env.example` 新增多个 flag
- `requirements_v4.txt`（新增 `jitcdde` 等，但不强制安装）

### 3.5 是否 Breaking Change

**否**。

- v3 的 `ode_templates/` 不动
- v3 的 `n6_ode_generator` 不动
- flag=false 时完全走 v3 模板渲染

### 3.6 如何验证

**冒烟测试**（P3）：

| 用例 | 输入 | 期望 |
|------|------|------|
| 1 | "EGF activates EGFR" | v3 模板渲染；仿真结果与 v3 基线 diff < 1% |
| 2 | "p53 Mdm2 feedback"（flag=true） | Oscillatory_Feedback 模板渲染；DDE 求解（jitcdde 可用时） |
| 3 | "p53 Mdm2 feedback"（flag=true, jitcdde 不可用） | 降级为 ODE 近似 + warning；仿真不崩 |
| 4 | "EGF activates EGFR"（flag=true） | 新 MM 模板（_mechanism_phosphorylation_mm）渲染；pEGFR 达峰 5-10 min |
| 5 | "apoptosis caspase cascade"（flag=true） | Caspase_Cascade 模板渲染；bistability 检测 |
| 6 | "NF-κB IκBα oscillation"（flag=true） | Oscillatory_Feedback 模板；振荡周期 1-2h 检测 |

**回归测试**：
- 全量 v3 测试套件
- EGF-EGFR 端到端（v3 路径）
- DDE 依赖隔离测试（卸载 jitcdde 后系统不崩）

### 3.7 是否可回滚

**是**。

- 回滚方式：`V4_ODE_TEMPLATE_V2_ENABLED=false`
- 回滚后：完全走 v3 模板
- DDE 降级：`V4_DDE_SOLVER_ENABLED=false` 单独控制
- Git：`git revert` P3 commit

### 3.8 不可碰清单（P3）

- `graph_v3.py` 的核心路由逻辑（仅允许在 n6_ode_generator 后加 hook）
- `rag_client.py`
- `sandbox.py` 的核心仿真逻辑
- `nodes_v2.py` 的核心节点逻辑
- 前端代码

### 3.9 完成标准

- [ ] 9 个新模板实现
- [ ] _mechanism_phosphorylation_mm.j2 恢复 MM 动力学
- [ ] DDE 求解器封装（try-import 降级）
- [ ] 双稳态检测器
- [ ] 振荡检测器
- [ ] ode_renderer_v2.py（从 ReactionIRv2 渲染）
- [ ] state.py 新增 v4_ode_system 字段
- [ ] Feature Flag 接入
- [ ] 冒烟测试 6 个用例全部通过
- [ ] 回归测试 0 失败
- [ ] git commit

---

## Phase 4: Cross-talk Engine + Pathway Specialist

### 4.1 目标

建立 v4 的 Pathway Architecture：Pathway Planner Agent + 10 Pathway Specialist Agents + Cross-talk Coordinator Agent。实现 10 通路 × 5 模块化设计。

### 4.2 修改目录

```
backend/app/
├── pathways/                    【新增目录】
│   ├── __init__.py
│   ├── pathway_planner.py       # Pathway Planner Agent
│   ├── pathway_specialist_base.py  # Specialist 基类
│   ├── specialists/             【10 个 Specialist 子目录】
│   │   ├── egfr_specialist.py
│   │   ├── mapk_specialist.py
│   │   ├── pi3k_akt_mtor_specialist.py
│   │   ├── p53_specialist.py
│   │   ├── apoptosis_specialist.py
│   │   ├── cell_cycle_specialist.py
│   │   ├── jak_stat_specialist.py
│   │   ├── nf_kappa_b_specialist.py
│   │   ├── wnt_specialist.py
│   │   └── tgf_beta_specialist.py
│   ├── pathway_modules/         【5 模块定义】
│   │   ├── core/                # 10 通路的 Core 模块
│   │   ├── feedback/            # 10 通路的 Feedback 模块
│   │   ├── crosstalk/           # 10 通路的 Cross-talk 模块
│   │   ├── perturbation/        # 10 通路的 Perturbation 模块
│   │   └── validation/          # 10 通路的 Validation 模块
│   └── pathway_registry.py      # 通路注册表（plugin pattern）
├── crosstalk/                   【新增目录】
│   ├── __init__.py
│   ├── coordinator.py           # Cross-talk Coordinator Agent
│   ├── shared_species_sync.py   # shared species 同步策略
│   └── crosstalk_edges.py       # cross-talk edge 注入
└── state.py                     【修改：新增 v4_pathway_class, v4_crosstalk_edges 字段】
```

### 4.3 删除

无。

**关键约束**：v3 的 `biomodels_reactions.py` 中 `ALLOWED_PATHWAY_SET = frozenset({"EGF_EGFR_MAPK"})` **不删除**，仅添加 v4 的并行路径。flag=false 时仍走 v3 的硬编码白名单。

### 4.4 新增

- `backend/app/pathways/` 完整目录
- `backend/app/crosstalk/` 完整目录
- `state.py` 新增字段：`v4_pathway_class: str`、`v4_crosstalk_edges: list`、`v4_shared_species: list`
- `.env.example` 新增 `V4_PATHWAY_PLANNER_ENABLED=false` 等
- `backend/tests/test_pathway_planner.py`
- `backend/tests/test_crosstalk_coordinator.py`

### 4.5 是否 Breaking Change

**否**。

- v3 的 `biomodels_reactions.py` 不动（flag=false 时走 v3 白名单）
- v3 的 `domain_checker.py` 不动
- v3 的 `kg_builder.py` 不动

### 4.6 如何验证

**冒烟测试**（P4）：

| 用例 | 输入 | 期望 |
|------|------|------|
| 1 | "EGF activates EGFR" | v3 行为不变（flag=false） |
| 2 | "EGF activates EGFR"（flag=true） | Pathway Planner 输出 pathway_class="EGFR_EGFR_MAPK" |
| 3 | "EGF + PI3K crosstalk"（flag=true） | Pathway Planner 识别双通路；Cross-talk Coordinator 激活；shared species（Ras/AKT）正确同步 |
| 4 | "apoptosis caspase"（flag=true） | Pathway Planner 输出 pathway_class="APOPTOSIS"；Apoptosis Specialist 激活 |
| 5 | "NF-κB oscillation"（flag=true） | Pathway Planner 输出 pathway_class="NF_KB"；NF-κB Specialist 激活；不再被 FORBIDDEN_PATHWAY_TERMS 阻塞 |
| 6 | "Wnt β-catenin"（flag=true） | Pathway Planner 输出 pathway_class="WNT"；Wnt Specialist 激活；Destruction complex 模块加载 |

**回归测试**：
- 全量 v3 测试套件
- EGF-EGFR 端到端（v3 路径）仍满足 pEGFR 5-10 min 达峰

### 4.7 是否可回滚

**是**。

- 回滚方式：`V4_PATHWAY_PLANNER_ENABLED=false` + `V4_CROSSTALK_COORDINATOR_ENABLED=false`
- 回滚后：完全走 v3 的硬编码白名单
- 数据库：无影响（Pathway Planner 不写库）
- Git：`git revert` P4 commit

### 4.8 不可碰清单（P4）

- `ode_templates/`（只加新模板，不改旧的）—— 已在 P3 完成
- `rag_client.py`
- `sbml_validator.py`
- `report_renderer.py`
- 前端代码

### 4.9 完成标准

- [ ] Pathway Planner Agent（含规则优先 + LLM 兜底）
- [ ] 10 个 Pathway Specialist Agent
- [ ] 10 通路 × 5 模块定义（Core/Feedback/Cross-talk/Perturbation/Validation）
- [ ] Cross-talk Coordinator Agent
- [ ] shared species 同步策略
- [ ] pathway_registry.py（plugin pattern）
- [ ] state.py 新增 3 个 v4 字段
- [ ] Feature Flag 接入
- [ ] 冒烟测试 6 个用例全部通过
- [ ] 回归测试 0 失败
- [ ] git commit

---

## Phase 5: SBML Grounding + Validation Pyramid

### 5.1 目标

建立 v4 的 Validation Layer：五层 Validation Pyramid（Internal Consistency / SBML BioModels / Cross-Pathway / Benchmark / Hypothesis）+ SBML Grounder Agent + Calibration Agent。**修复审计报告 §7.2 的 Oracle 默认 pass=True 致命错误**。

### 5.2 修改目录

```
backend/app/
├── validation_v2/               【新增目录】
│   ├── __init__.py
│   ├── validation_pyramid.py    # 五层 Pyramid 编排
│   ├── level1_internal.py       # Level 1: Internal Consistency
│   ├── level2_sbml.py           # Level 2: SBML/BioModels（修复 Oracle）
│   ├── level3_crosstalk.py      # Level 3: Cross-Pathway
│   ├── level4_benchmark.py      # Level 4: Benchmark
│   ├── level5_hypothesis.py     # Level 5: Hypothesis Validation
│   ├── validation_agent.py      # Validation Agent 主逻辑
│   └── thresholds.py            # 通路特异阈值
├── sbml_grounder/               【新增目录】
│   ├── __init__.py
│   ├── grounder_agent.py        # SBML Grounder Agent
│   ├── five_level_mapping.py    # ODE ↔ Reaction ↔ SBML ↔ Parameter ↔ PMID 五级映射
│   └── sbml_parser_v2.py        # 真正的 XML 解析（替代 v3 的 LLM 解析）
├── calibration/                 【新增目录】
│   ├── __init__.py
│   ├── calibration_agent.py     # Calibration Agent
│   ├── least_squares_fitter.py  # 最小二乘（lmfit try-import 降级）
│   └── confidence_interval.py   # 置信区间
├── sensitivity/                 【新增目录】
│   ├── __init__.py
│   ├── sobol_analyzer.py        # Sobol 全局灵敏度（SALib try-import 降级）
│   ├── morris_analyzer.py       # Morris elementary effects
│   └── local_sensitivity.py     # 局部灵敏度（默认可用）
└── state.py                     【修改：新增 v4_validation_report, v4_grounding_ledger 字段】
```

### 5.3 删除

无。

**关键约束**：v3 的 `sbml_validator.py` **不删除**，仅标记 deprecated。v3 的 `_skipped_report` 默认 `pass=True` 的问题**不修改 v3 代码**，而是在 v4 的 `level2_sbml.py` 中实现正确的 `skipped = pass=False` 逻辑。

### 5.4 新增

- `backend/app/validation_v2/` 完整目录
- `backend/app/sbml_grounder/` 完整目录
- `backend/app/calibration/` 完整目录
- `backend/app/sensitivity/` 完整目录
- `state.py` 新增字段：`v4_validation_report: dict`、`v4_grounding_ledger: dict`、`v4_calibration_result: dict`
- `.env.example` 新增多个 flag
- `requirements_v4.txt` 追加 `lmfit`、`SALib`

### 5.5 是否 Breaking Change

**否**。

- v3 的 `sbml_validator.py` 不动
- v3 的 `model_consistency_validator.py` 不动
- v3 的 `conservation_checker.py` 不动
- flag=false 时完全走 v3 验证路径

### 5.6 如何验证

**冒烟测试**（P5）：

| 用例 | 输入 | 期望 |
|------|------|------|
| 1 | "EGF activates EGFR"（含 BIOMD0000000205） | v3 行为不变（flag=false） |
| 2 | "EGF activates EGFR"（含 BIOMD0000000205，flag=true） | Level 2 Track A 真实 SBML 仿真对比；grounding_ledger 建立 |
| 3 | 仿真失败（人为破坏参数）（flag=true） | Validation Agent 阻塞流水线；v4_validation_report.pass=False |
| 4 | SBML 不可用（人为断网）（flag=true） | Level 2 skipped 状态 pass=False（修复审计 §7.2）；阻塞流水线 |
| 5 | "NF-κB oscillation"（flag=true） | Level 4 benchmark 验证振荡周期 1-2h（Nelson 2004） |
| 6 | "p53 pulse"（flag=true） | Level 4 benchmark 验证 p53 脉冲周期 5-7h（Lev Bar-Or 2000） |
| 7 | 任意仿真（flag=true） | Level 1 mass conservation 检查通过（< 5% 误差） |

**回归测试**：
- 全量 v3 测试套件
- EGF-EGFR 端到端（v3 路径）
- lmfit/SALib 依赖隔离测试（卸载后系统不崩）

### 5.7 是否可回滚

**是**。

- 回滚方式：`V4_VALIDATION_PYRAMID_ENABLED=false` + `V4_SBML_GROUNDER_ENABLED=false` + `V4_CALIBRATION_AGENT_ENABLED=false`
- 回滚后：完全走 v3 验证路径
- 数据库：无影响
- Git：`git revert` P5 commit

### 5.8 不可碰清单（P5）

- `ode_templates/`（v3 模板不动）
- `report_renderer.py`
- `rag_client.py`
- 前端代码
- `biomodels_reactions.py`（v3 的硬编码白名单不动）

### 5.9 完成标准

- [ ] 五层 Validation Pyramid
- [ ] Level 2 skipped = pass=False（修复审计 §7.2）
- [ ] Level 2 Track B 差异指标 = null（修复审计 §10.3）
- [ ] 通路特异阈值（NF-κB / p53 / cell cycle 等）
- [ ] SBML Grounder Agent + 五级映射链
- [ ] sbml_parser_v2.py（真正 XML 解析，替代 v3 LLM 解析）
- [ ] Calibration Agent（lmfit try-import 降级）
- [ ] Sobol/Morris 灵敏度（SALib try-import 降级）
- [ ] Local sensitivity（默认可用）
- [ ] state.py 新增 3 个 v4 字段
- [ ] Feature Flag 接入
- [ ] 冒烟测试 7 个用例全部通过
- [ ] 回归测试 0 失败
- [ ] git commit

---

## Phase 6: Hypothesis Layer + Agent 完整编排

### 6.1 目标

建立 v4 的 Hypothesis Layer（Layer 8）+ 完整 Agent 动态编排（替换 v3 的固定流水线）。实现 13 个 Agent 的协同工作。

### 6.2 修改目录

```
backend/app/
├── hypothesis/                  【新增目录】
│   ├── __init__.py
│   ├── hypothesis_agent.py      # Hypothesis Agent
│   ├── hypothesis_generator.py  # 假设生成
│   ├── experiment_designer.py   # 实验设计建议
│   └── falsifiability_checker.py # 可证伪性检查
├── agent_orchestration/         【新增目录】
│   ├── __init__.py
│   ├── dynamic_router.py        # 动态路由（替代 v3 固定流水线）
│   ├── agent_registry_v4.py     # 13 个 Agent 注册表
│   ├── pathway_class_dispatcher.py  # 基于 pathway_class 的分支
│   └── fail_safe.py             # 失败短路 + 回退
├── agents_v4/                   【新增目录：v4 Agent 集合】
│   ├── __init__.py
│   ├── mechanism_builder.py     # Mechanism Builder Agent
│   ├── ode_builder.py           # ODE Builder Agent
│   ├── simulation_planner.py    # Simulation Planner Agent
│   └── parameter_agent.py       # Parameter Agent（强制 pathway 隔离）
├── graph_v3.py                  【修改：接入 v4 动态路由 hook】
└── state.py                     【修改：新增 v4_hypothesis_list, v4_agent_dispatches 字段】
```

### 6.3 删除

无。

**关键约束**：v3 的 `graph_v3.py` 固定流水线**不删除**，仅添加 v4 动态路由作为并行路径。flag=false 时走 v3 固定流水线。

### 6.4 新增

- `backend/app/hypothesis/` 完整目录
- `backend/app/agent_orchestration/` 完整目录
- `backend/app/agents_v4/` 完整目录
- `state.py` 新增字段：`v4_hypothesis_list: list`、`v4_agent_dispatches: list`
- `.env.example` 新增 `V4_HYPOTHESIS_AGENT_ENABLED=false` + `V4_DYNAMIC_ROUTING_ENABLED=false`
- 新增 SSE 事件：`v4_hypothesis_generated`（前端可不订阅）

### 6.5 是否 Breaking Change

**否**（内部）。

- v3 的 `graph_v3.py` 固定流水线不动
- v3 的 SSE 事件不变
- v3 的 `supervisor.py` 不动

**新增 SSE 事件**：`v4_hypothesis_generated`（前端可不订阅，不影响现有功能）

### 6.6 如何验证

**冒烟测试**（P6）：

| 用例 | 输入 | 期望 |
|------|------|------|
| 1 | "EGF activates EGFR" | v3 行为不变（flag=false） |
| 2 | 任意仿真完成（flag=true） | Hypothesis Agent 输出假设列表（≥1 条） |
| 3 | 假设质量检查 | 每个假设含预测 + 实验设计 + 验证方式 + 预期结果 |
| 4 | "EGF + PI3K crosstalk"（flag=true） | 动态路由激活：Pathway Planner → 2 Specialist → Cross-talk Coordinator → ... |
| 5 | "apoptosis caspase"（flag=true） | 动态路由：Pathway Planner → Apoptosis Specialist → Caspase_Cascade 模板 |
| 6 | 仿真失败（人为）（flag=true） | fail_safe 触发；短路到错误报告；不推进到 Hypothesis |
| 7 | v3 路径回归 | flag=false 时全量 v3 测试通过 |

**回归测试**：
- 全量 v3 测试套件
- EGF-EGFR 端到端（v3 路径）
- 13 Agent 协同测试（v4 路径）

### 6.7 是否可回滚

**是**。

- 回滚方式：`V4_HYPOTHESIS_AGENT_ENABLED=false` + `V4_DYNAMIC_ROUTING_ENABLED=false`
- 回滚后：完全走 v3 固定流水线
- 数据库：无影响
- Git：`git revert` P6 commit

### 6.8 不可碰清单（P6）

- **全部 v3 文件**（只新增 v4 文件）
- v3 的 `graph_v3.py` 核心路由逻辑（仅加 hook，不改路由）
- v3 的 `supervisor.py`
- v3 的 `nodes_v2.py`
- v3 的 `prompts_v2.py`
- 前端代码（仅新增事件订阅，不改现有）

### 6.9 完成标准

- [ ] Hypothesis Agent（含假设生成 + 实验设计 + 可证伪性检查）
- [ ] 动态路由（基于 pathway_class 分支）
- [ ] 13 Agent 注册表
- [ ] pathway_class_dispatcher
- [ ] fail_safe（失败短路 + 回退）
- [ ] Mechanism Builder / ODE Builder / Simulation Planner / Parameter Agent
- [ ] state.py 新增 2 个 v4 字段
- [ ] Feature Flag 接入
- [ ] 新增 SSE 事件 v4_hypothesis_generated
- [ ] 冒烟测试 7 个用例全部通过
- [ ] 回归测试 0 失败
- [ ] git commit

---

## Phase 7: v3 弃用标记 + 清理

### 7.1 目标

P1-P6 全部完成且生产环境稳定运行 ≥ 30 天后，开始 v3 弃用清理。**这是唯一的 Breaking Change Phase**。

### 7.2 修改目录

```
backend/app/
├── reaction_ir.py               【标记 deprecated，保留 90 天】
├── sbml_validator.py            【标记 deprecated，保留 90 天】
├── biomodels_reactions.py       【标记 deprecated，ALLOWED_PATHWAY_SET 保留但不再使用】
├── ode_templates/
│   ├── cascade_activation.j2    【标记 deprecated】
│   └── cascade_inhibition.j2    【标记 deprecated】
├── graph_v3.py                  【标记 deprecated，保留 90 天】
├── state.py                     【删除 v3 字段：network_json, parameters, rag_retrieved_params 等】
└── ...
```

### 7.3 删除

**P7 阶段才允许删除**：

- v3 的 `reaction_ir.py`（已被 `reaction_ir_v2/` 替代）
- v3 的 `sbml_validator.py`（已被 `validation_v2/` 替代）
- v3 的 `biomodels_reactions.py`（已被 `pathways/` 替代）
- v3 的 `cascade_activation.j2` / `cascade_inhibition.j2`（已被废弃）
- v3 的 State 字段（`network_json` / `parameters` / `rag_retrieved_params` 等）
- Adapter（`adapters/`，保留 90 天后删除）

**保留**：
- v3 的 `Signaling_Cascade_Phos.j2`（仍可用于 EGF-EGFR-MAPK 简单场景）
- v3 的 `PKPD_*.j2`（仍可用于简单 PK/PD）
- v3 的 `rag_client.py`（v4 仍依赖其底层检索能力）

### 7.4 新增

无（P7 仅清理）。

### 7.5 是否 Breaking Change

**是**（最终清理）。

- 删除 v3 State 字段 → 旧 checkpoint 不可用（需迁移脚本）
- 删除 v3 模块 → 旧 API 端点不可用（如有）
- 删除 Adapter → v3 → v4 转换不可用

### 7.6 如何验证

**冒烟测试**（P7）：

| 用例 | 输入 | 期望 |
|------|------|------|
| 1 | 任意 v3 输入 | v3 字段被弃用标记；运行时 warning 但不阻塞 |
| 2 | 任意 v4 输入 | 完全走 v4 路径；无 v3 字段残留 |
| 3 | 全量 v4 测试 | 全部通过 |
| 4 | State 迁移脚本 | 旧 checkpoint 成功迁移到 v4 格式 |

### 7.7 是否可回滚

**部分可回滚**。

- 删除的 v3 代码可通过 `git revert` 恢复
- 但 State 字段删除后，旧 checkpoint 不可逆（需迁移脚本）
- **建议**：P7 前 7 天完整备份数据库 + checkpoint

### 7.8 不可碰清单（P7）

- **全部 v4 文件**（只清理 v3 deprecated）

### 7.9 完成标准

- [ ] v3 模块全部标记 deprecated
- [ ] v3 State 字段删除
- [ ] State 迁移脚本（v3 → v4）
- [ ] Adapter 保留 90 天后删除
- [ ] v3 ChromaDB collection 保留 30 天后删除
- [ ] 全量 v4 测试通过
- [ ] 生产环境稳定运行 30 天
- [ ] git commit

---

## 风险矩阵（Risk Matrix）

| 风险 | Phase | 严重度 | 概率 | 缓解策略 |
|------|-------|--------|------|---------|
| Adapter 转换丢失信息 | P2 | 高 | 中 | 冒烟测试用例 3 校验往返 diff < 5%；fail-safe 降级 |
| DDE 求解器依赖缺失 | P3 | 中 | 高 | try-import 降级为 ODE 近似 + warning |
| Pathway Planner 误分类 | P4 | 高 | 中 | 规则优先 + LLM 兜底 + 冒烟测试 6 个用例 |
| Cross-talk shared species 冲突 | P4 | 高 | 中 | shared_species_sync 策略 + 单元测试 |
| SBML Grounder 五级映射断裂 | P5 | 中 | 中 | grounding_ledger 完整性检查 |
| Validation 误阻塞（false negative） | P5 | 高 | 低 | 通路特异阈值 + 人工 review |
| Hypothesis Agent 生成低质量假设 | P6 | 中 | 高 | falsifiability_checker + 文献检索验证 |
| 动态路由死锁 | P6 | 高 | 低 | fail_safe + 超时回退到 v3 路径 |
| State 字段共存导致内存膨胀 | P2-P6 | 低 | 高 | P7 清理；监控内存 |
| ChromaDB 数据迁移失败 | P7 | 高 | 低 | 旧 collection 保留 30 天 + 备份 |
| 依赖冲突（jitcdde / SALib / lmfit） | P3/P5 | 中 | 中 | try-import 降级 + requirements_v4.txt 独立 |

---

## 回滚预案（Rollback Plan）

### 全局回滚策略

任何 Phase 出问题，按以下优先级回滚：

1. **第一优先级：Feature Flag 回滚**（秒级）
   - 将对应 Phase 的 flag 切回 `false`
   - 系统立即走 v3 路径
   - 无需 git revert，无需重新部署

2. **第二优先级：Git revert**（分钟级）
   - `git revert` 对应 Phase 的 commit
   - 适用于 Feature Flag 无法解决的问题（如 State 字段冲突）

3. **第三优先级：数据库回滚**（小时级）
   - ChromaDB：从备份恢复 v3 collection
   - Checkpoint：从备份恢复
   - 适用于 P7 阶段的不可逆操作

### 各 Phase 回滚命令

| Phase | Feature Flag 回滚 | Git Revert |
|-------|------------------|------------|
| P1 | `V4_ONTOLOGY_AGENT_ENABLED=false` | `git revert <P1_commit>` |
| P2 | `V4_REACTION_IR_ENABLED=false` + `V4_REACTION_IR_ADAPTER_ENABLED=false` | `git revert <P2_commit>` |
| P3 | `V4_ODE_TEMPLATE_V2_ENABLED=false` + `V4_DDE_SOLVER_ENABLED=false` | `git revert <P3_commit>` |
| P4 | `V4_PATHWAY_PLANNER_ENABLED=false` + `V4_CROSSTALK_COORDINATOR_ENABLED=false` | `git revert <P4_commit>` |
| P5 | `V4_VALIDATION_PYRAMID_ENABLED=false` + `V4_SBML_GROUNDER_ENABLED=false` | `git revert <P5_commit>` |
| P6 | `V4_HYPOTHESIS_AGENT_ENABLED=false` + `V4_DYNAMIC_ROUTING_ENABLED=false` | `git revert <P6_commit>` |
| P7 | `V3_DEPRECATED_CLEANUP=false` | `git revert <P7_commit>` + 数据库恢复 |

### 回滚验证

回滚后必须验证：
- v3 路径全量测试通过
- EGF-EGFR 端到端仿真正常（pEGFR 5-10 min 达峰）
- SSE 事件正常
- 前端无报错

---

## 依赖隔离总表（Dependency Isolation）

| 依赖 | 版本 | 用途 | Phase | 降级策略 |
|------|------|------|-------|---------|
| `jitcdde` | ≥1.8 | DDE 求解器 | P3 | 降级为 ODE 近似 + warning |
| `SALib` | ≥1.4 | Sobol/Morris 全局灵敏度 | P5 | 降级为 local sensitivity only |
| `lmfit` | ≥1.2 | 参数校准 | P5 | 降级为简单最小二乘 |
| `pyviabilit` | ≥0.1 | 双稳态检测 | P5 | 降级为多初值扫描 |
| `bioservices` | ≥1.10 | KEGG/Reactome API | P1 | 降级为本地缓存 |
| `lxml` | ≥4.9 | SBML XML 解析 | P5 | 降级为正则解析 |
| `emcee` | ≥3.1 | MCMC（v5 预留） | P5+ | 降级为 grid search |
| `roadrunner` | ≥2.2 | SBML 仿真 | P5 | 降级到 Track B |

**降级原则**：
- 任何依赖缺失时系统不崩
- 降级时记录 warning 日志
- 降级不影响 v3 路径
- 降级功能在 UI 上标注"limited mode"

---

## 不可碰清单总表（Do-Not-Touch Master List）

| Phase | 不可碰文件 | 原因 |
|-------|-----------|------|
| P1 | sandbox.py / ode_templates/ / rag_client.py / graph_v3.py 核心路由 / nodes_v2.py / 前端 | 仅新增 Ontology Agent，不碰核心 |
| P2 | sandbox.py / ode_templates/ / nodes_v2.py 核心仿真 / rag_client.py / 前端 | 仅新增 Reaction IR v2 + Adapter |
| P3 | graph_v3.py 核心路由 / rag_client.py / sandbox.py 核心 / nodes_v2.py 核心 / 前端 | 仅新增模板 + 求解器 |
| P4 | ode_templates/（不改旧模板）/ rag_client.py / sbml_validator.py / report_renderer.py / 前端 | 仅新增 Pathway 模块 |
| P5 | ode_templates/ / report_renderer.py / rag_client.py / biomodels_reactions.py / 前端 | 仅新增 Validation Layer |
| P6 | 全部 v3 文件（只新增 v4 文件）/ 前端（仅新增订阅） | 仅新增 Hypothesis + 动态路由 |
| P7 | 全部 v4 文件（只清理 v3 deprecated） | 仅清理 v3 |

**违反规则**：任何 Phase 修改了"不可碰"清单中的文件 = Phase 失败，必须 `git revert` 重做。

---

## State 字段共存映射表

| v3 字段 | v4 字段 | 共存策略 | 清理 Phase |
|---------|---------|---------|-----------|
| `network_json` | `v4_reaction_ir` | v4 写入时通过 Adapter 同步到 v3 | P7 |
| `parameters` | `v4_parameters` | v4 写入时同步到 v3 | P7 |
| `rag_retrieved_params` | `v4_rag_parameters` | v4 写入时同步到 v3 | P7 |
| `rag_selected_params` | `v4_selected_parameters` | v4 写入时同步到 v3 | P7 |
| `mechanism` | `v4_mechanism_context` | 独立字段，不同步 | P7 |
| `simulation_ci`（已 deprecated） | `v4_confidence_interval` | 独立字段 | P7 |
| `next_worker` | `v4_agent_dispatches` | 独立字段 | P7 |
| `validation_report` | `v4_validation_report` | 独立字段 | P7 |
| 无 | `v4_ontology_entities` | 新增 | - |
| 无 | `v4_pathway_class` | 新增 | - |
| 无 | `v4_crosstalk_edges` | 新增 | - |
| 无 | `v4_shared_species` | 新增 | - |
| 无 | `v4_ode_system` | 新增 | - |
| 无 | `v4_grounding_ledger` | 新增 | - |
| 无 | `v4_calibration_result` | 新增 | - |
| 无 | `v4_hypothesis_list` | 新增 | - |

**State 共存原则**：
- v3 字段保持原名，v3 代码读 v3 字段
- v4 字段以 `v4_` 前缀共存，v4 代码读 v4 字段
- v4 代码写入 v4 字段时，通过 Adapter 同步到 v3 字段（保证旧代码可读）
- v3 代码写入 v3 字段时，**不同步到 v4 字段**（v4 字段保持 None，新代码检测到 None 时走 v3 路径）
- P7 阶段统一删除 v3 字段

---

## SSE 事件契约冻结清单

| 事件名 | 冻结状态 | 允许的变更 | 新增字段规则 |
|--------|---------|-----------|------------|
| `chat_message` | 冻结 | 仅允许追加字段 | nullable + 前端忽略 |
| `workflow_v3_state` | 冻结 | 事件名不变（即使内部 v4） | 仅追加字段 |
| `clarification_needed` | 冻结 | 不变 | 不允许新字段 |
| `clarification_resolved` | 冻结 | 不变 | 不允许新字段 |
| `pkpd_profile` | 冻结 | 不变 | 仅追加字段 |
| `drug_regimen` | 冻结 | 不变 | 仅追加字段 |
| `dose_response` | 冻结 | 不变 | 仅追加字段 |
| `combination_synergy` | 冻结 | 不变 | 仅追加字段 |
| `v4_hypothesis_generated` | 新增（P6） | 新事件 | 前端可不订阅 |

**SSE 契约原则**：
- 前端在 P1-P6 期间不动
- v4 新数据以追加字段方式注入现有事件
- 不得删除或重命名现有事件字段
- 新字段必须 nullable，前端不识别时忽略
- 新增事件类型必须以 `v4_` 前缀

---

## ChromaDB 迁移策略

### 不可逆操作禁令

迁移期间**禁止**：
- `delete_collection()`
- `recreate()`
- `drop_collection()`
- 任何会删除数据的操作

### v4 collection 命名

| v3 collection | v4 collection | 灌库时机 |
|---------------|---------------|---------|
| `biodynamics_mechanism` | `biodynamics_mechanism_v4` | P1 |
| `biodynamics_params` | `biodynamics_parameter_v4` | P2 |
| `biodynamics_evidence` | `biodynamics_evidence_v4` | P2 |
| （空） | `biodynamics_experiment_v4` | P2 |
| （无） | `biodynamics_pathway_graph_v4` | P4 |

### 数据迁移流程

1. **P1-P6**：v4 collection 独立灌库，v3 collection 不动
2. **每个 Phase 验证**：v4 检索正常才推进
3. **P7 阶段**：
   - v3 collection 标记 read-only
   - 提供 `migrate_v3_to_v4.py` 脚本（带 dry-run 模式）
   - v3 collection 保留 30 天（只读）
   - 30 天后删除 v3 collection

### 回滚策略

- v4 collection 数据问题：禁用对应 v4 flag，回到 v3 collection
- v3 collection 数据问题：从备份恢复（迁移前必须备份）
- 任何时候不删除 v3 collection（P7 之前）

---

## 总结

本 Migration Plan 遵循 Architecture-driven Development 方法论，将 v4 架构落地分为 7 个 Phase：

- **P1-P6**：纯新增共存模式，不删旧代码，通过 Feature Flag 控制
- **P7**：最终清理 v3 deprecated 代码

**核心保证**：
1. 每个 Phase 完成前不启动下一 Phase
2. 每个 Phase 必须 git commit + 冒烟测试通过
3. 每个 Phase 保证系统可运行（不崩）
4. 每个 Phase 可独立回滚（Feature Flag 秒级回滚）
5. 每个 Phase 明确"不可碰"清单
6. 8 项 Compatibility Strategy 约束全程遵守

**预期结果**：v4 迁移完成后，系统具备 10 通路完整覆盖 + 17 类机制 + 9 个新模板 + Cross-talk + Validation Pyramid + Hypothesis Layer，且迁移过程中几乎不会崩。

---

**文档版本**：v4.0-migration
**完成日期**：2026-07-05
**前置文档**：BioDynamics_v4_Scientific_Architecture.md
**下一步**：启动 P1（Scientific Layer 建立与 Ontology Agent）
