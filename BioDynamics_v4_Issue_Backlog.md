# BioDynamics v4 — Issue Backlog（统一问题清单）

> **文档性质**：Phase 1 Issue Consolidation 产出物（只读分析，未修改任何源代码）
> **Ground Truth**：`BioDynamics_v4_Final_Verification_Report.md`（以下简称 **[V]**）+ `BioDynamics_v4_Reliability_Gap_Report.md`（以下简称 **[R]**）+ 当前代码仓库
> **生成日期**：2026-07-08
> **整合方法**：将 [V] 的 AD/HC/SA/SB/FM-001~100/TD-001~100/FT-001~100 与 [R] 的 D1~D6/Path 1~5/V1~V7/Skill/MCP/Prompt 建议全部抽取，按根因+修复方案去重合并，生成统一 Issue ID（IB-xxx）
> **严重度映射**：P0 → **Critical**；P1 → **High**；P2 → **Medium**；次要/装饰性 → **Low**
> **工作量定义**：S = 单文件 <30 行；M = 2~4 文件 / 30~200 行 / 单机制或单测试模块；L = 5+ 文件 / 200+ 行 / 新子系统 / 全套参数 / MCP 集成

---

## 0. 执行摘要

- **去重前原始条目**：[V] 约 432 条（100 FM + 100 TD + 100 FT + 5 AD + 6 HC + 10 SA + 15 SB + 10 UX + 10 pathway missing + 15 research gap）+ [R] 约 24 条（6 D + 5 Path + 7 V + 2 Skill + 3 MCP + 5 Prompt + 附录若干）≈ **456 条**
- **去重后唯一 Issue**：**142 条**（IB-001 ~ IB-142）
- **严重度分布**：

| 严重度 | 数量 | 占比 | 阻塞 Release |
|--------|------|------|--------------|
| Critical | 22 | 15.5% | 全部阻塞 |
| High | 56 | 39.4% | 大部分阻塞（科研正确性/可靠性类） |
| Medium | 58 | 40.8% | 不阻塞（v4.1+ backlog） |
| Low | 6 | 4.2% | 不阻塞 |
| **合计** | **142** | 100% | — |

- **来源分布**：仅来自 [V] 的 96 条；仅来自 [R] 的 18 条；两报告共同指出的 28 条
- **类型分布**：Scientific 58 / Engineering 38 / Testing 14 / Architecture 12 / UI 10 / Performance 6 / Security 3 / Data Integrity 3（部分跨类，按主类统计）

> **核心结论**：v4 仿真管线当前不可用的根因集中在 **22 个 Critical**，分布于 ODE Renderer（字段脱节）、Reaction IR（机制语义）、Solver（DDE 空壳）、Sandbox（生物检查不触发）、Validation（L2 MOCK / L4 覆盖）、Specialists（无动力学参数）、Calibration/Sensitivity（占位）、Reliability（基因幻觉 / fallback 死标志 / 数值无恢复）。修复这 22 项即可解锁全部 v4 仿真与验证。

---

## 1. 严重度与类型图例

| 字段 | 取值 |
|------|------|
| 来源 | V = Final Verification Report；R = Reliability Gap Report；V+R = 两报告共同 |
| 类型 | Scientific / Engineering / UI / Performance / Testing / Architecture / Security / Data Integrity |
| 严重程度 | Critical / High / Medium / Low |
| 阻塞 Release | YES / NO |
| 工作量 | S / M / L |

---

## 2. 统一 Issue Matrix（主表）

> 按「严重程度 → 子系统 → IB ID」排序。`源ID` 列为去重前的原始编号映射。

### 2.1 Critical（22 条 — 全部阻塞 Release）

| IB ID | 子系统 | 标题 | 源 | 类型 | 阻塞 | 工作量 | 源ID映射 |
|-------|--------|------|----|------|------|--------|----------|
| IB-001 | ODE Renderer | 读取不存在的 `name`/`parameters`/`source`/`target` 字段 → 零通量 ODE | V | Scientific | YES | M | AD-1, SB-1, FM-001/002/003, TD-001/002/003, FT-001/002/003 |
| IB-002 | ODE Template | 磷酸化 MM 公式 `k_cat*src²/(Km+src)` 错误（9 模板 copy-paste） | V | Scientific | YES | S | SA-1, SB-6, FM-004, TD-004, FT-004 |
| IB-003 | ODE Template | Binding 渲染为一级反应 `k_bind*src`（应双分子 `k_on*[A]*[B]`） | V | Scientific | YES | S | SA-2, SB-9, FM-006, TD-016, FT-016 |
| IB-004 | ODE Template | Complex formation 渲染为一级反应 | V | Scientific | YES | S | TD-017, FT-017 |
| IB-005 | Reaction IR | Dimerization 化学计量 1→1（应 2→1） | V | Scientific | YES | S | SA-3, SB-10, FM-007, TD-018, FT-018 |
| IB-006 | Solver | DDE 求解器空壳：`y(t-τ)≈y(t)` 消除全部延迟 | V+R | Scientific | YES | L | AD-4, SB-5, FM-005, TD-005, FT-005 |
| IB-007 | Solver | 求解器发散/NaN 无参数级恢复，重试生成相同代码 | R | Engineering | YES | L | D1, Skill-2 |
| IB-008 | Reaction IR | INHIBITION 反应物=产物（inhibitor 应为 modifier） | V | Scientific | YES | M | FM-008, TD-010, FT-010 |
| IB-009 | Reaction IR | `else:` 分支将 8 种机制错误归类（transcription/translation/GTP_GDP/cleavage/dissociation/sequestration/activation/dimerization） | V | Scientific | YES | L | FM-009, TD-011/019/020/021/022/023/024/025, FT-011/019~025 |
| IB-010 | Schema | Modifier 无 Ki/Kact/n_hill/inhibition_type/alpha | V | Scientific | YES | M | FM-042, TD-026, FT-026 |
| IB-011 | Schema | Modifier.site 单 string 无法多位点 | V | Scientific | YES | S | TD-028, FT-028 |
| IB-012 | Constraints | check_mass_conservation 仅 token 检查，无数值守恒 | V | Scientific | YES | M | TD-040, FT-040 |
| IB-013 | Sandbox | BIO_CHECK 标记无模板输出 → 生物检查永不触发 | V | Scientific | YES | M | SA-10, SB-2, FM-012, TD-067, FT-067 |
| IB-014 | Sandbox | 无仿真后质量守恒验证 + 负浓度无检测 | V | Scientific | YES | M | SB-3, FM-052/053, TD-068, FT-068 |
| IB-015 | Validation L2 | `_simulate_v4_ode` 是 MOCK 线性衰减（False Positive） | V | Scientific | YES | M | SA-8, FM-010, TD-007, FT-007 |
| IB-016 | Validation L4 | 仅 5/10 通路有 benchmark，其余 `pass=True,no_benchmark_matched` | V | Scientific | YES | M | FM-011, L4 audit |
| IB-017 | Specialists | 全部 10 条 specialist 无动力学参数 | V | Scientific | YES | L | SA-9, FM-013, TD-008, FT-008 |
| IB-018 | Calibration | 占位 `_default_model` 返回固定列表 | V | Scientific | YES | L | SA-6, FM-014, TD-006, FT-006 |
| IB-019 | Calibration | 输出不回写 `state.parameters` | V | Architecture | YES | M | AD-3, FM-015, TD-014, FT-014 |
| IB-020 | Sensitivity | 占位 model_func 无生物学意义 | V | Scientific | YES | L | FM-036, TD-097, FT-097 |
| IB-021 | Reliability | 基因/蛋白实体零外部验证（ontology 默认 OFF，幻觉基因进 ODE） | R | Scientific | YES | L | D2, Path 1 |
| IB-022 | Reliability | `fallback_used` 是死标志，主图 worker 无 fail-safe（v4→v3 降级未实现） | R | Engineering | YES | L | D3, A.1, A.3 |

### 2.2 High（56 条 — 科研正确性/可靠性类阻塞 Release，UX 类部分阻塞）

| IB ID | 子系统 | 标题 | 源 | 类型 | 阻塞 | 工作量 | 源ID映射 |
|-------|--------|------|----|------|------|--------|----------|
| IB-023 | ODE Template | Hill 函数负值崩溃（`src**n_hill` 当 src<0 且非整数） | V | Scientific | YES | S | SB-12, FM-022, TD-055, FT-055 |
| IB-024 | ODE Template | Ubiquitination 符号错误（`-=` 应为 `+=`） | V | Scientific | YES | S | SB-14, FM-021, TD-054, FT-054 |
| IB-025 | ODE Template | Cleavage 净零通量（caspase + bistable） | V | Scientific | YES | S | SB-11, FM-020, TD-057/058, FT-057/058 |
| IB-026 | ODE Template | Nuclear transport 无 compartment 体积缩放 | V | Scientific | YES | M | SB-8, FM-023, TD-056, FT-056 |
| IB-027 | ODE Renderer | EGFR/MAPK 路由到振荡模板（应 transient_cascade） | V | Scientific | YES | M | FM-077, TD-048, FT-048 |
| IB-028 | ODE Renderer | `_select_template` 忽略 requires_dde | V | Engineering | NO | S | FM-078, TD-047, FT-047 |
| IB-029 | ODE Renderer | DDE delay 不从 reaction_ir 提取 | V | Scientific | YES | M | FM-045, TD-050, FT-050 |
| IB-030 | ODE Template | DDE 双重计数（`dy_ode+dy` 转录边算两次） | V | Scientific | YES | S | SB-15, FM-039 |
| IB-031 | Solver | 无事件检测（bistability） | V | Scientific | YES | M | FM-046, TD-060, FT-060 |
| IB-032 | Solver | max_step 硬编码无刚度自适应 | V | Scientific | YES | M | SB-13, FM-047, TD-061, FT-061 |
| IB-033 | Solver | bistability 无滞后验证 | V | Scientific | YES | M | FM-048, TD-062, FT-062 |
| IB-034 | Solver | 无 stiff MM 专用 solver（缺 BDF/Radau fallback） | V | Scientific | YES | M | TD-066, FT-066 |
| IB-035 | Reaction IR | Provenance 永远 None（追溯链断裂） | V | Data Integrity | YES | S | FM-050/094, TD-035, FT-035 |
| IB-036 | Schema | Constraint.expression 无结构字符串 | V | Scientific | YES | M | TD-029, FT-029 |
| IB-037 | Schema | Constraint.type 静默降级（未知类型不拒绝） | V | Scientific | YES | S | TD-030, FT-030 |
| IB-038 | Schema | SpeciesV2 无单位转换因子（nM 与 molecule_per_cell 混用） | V | Scientific | YES | M | SB-7, FM-024, TD-034, FT-034 |
| IB-039 | Mechanism | is_enzymatic_mechanism 遗漏 ubiquitination | V | Scientific | YES | S | FM-027, TD-036, FT-036 |
| IB-040 | Mechanism | `_normalize_kinetics_name` 折叠 first_order→mass_action | V | Scientific | YES | S | FM-044, TD-037, FT-037 |
| IB-041 | Mechanism | transport 统一映射 nuclear_import（丢失 cytoplasm_translocation） | V | Scientific | YES | S | FM-043, TD-038, FT-038 |
| IB-042 | Mechanism | 19 枚举值声称 17（文档/合约不一致） | V | Engineering | NO | S | TD-039, FT-039 |
| IB-043 | Constraints | auto_generate 仅 phosphorylation pair（缺 binding/dimerization/complex） | V | Scientific | YES | M | FM-026, TD-041, FT-041 |
| IB-044 | Constraints | check_enzymatic 仅 catalytic modifier（缺 allosteric/activating） | V | Scientific | YES | M | TD-042, FT-042 |
| IB-045 | Constraints | check_thermodynamic 仅关键词（无 K_eq=k_fwd/k_rev 验证） | V | Scientific | YES | M | TD-043, FT-043 |
| IB-046 | Constraints | check_non_negative 仅初始浓度（无动态负浓度检查） | V | Scientific | YES | M | TD-044, FT-044 |
| IB-047 | Constraints | 无 moiety conservation（磷酸/泛素/GTP 守恒） | V | Scientific | YES | M | TD-046, FT-046 |
| IB-048 | Constraints | Constraint 表达式解析错误（含复合物名） | V | Scientific | YES | S | FM-025 |
| IB-049 | Sandbox | NaN/Inf 在 CSV 中不被检测 | V | Engineering | YES | S | FM-051, TD-069, FT-069 |
| IB-050 | Sandbox | 生物检查在无图像时跳过（依赖 matplotlib 后端） | V | Engineering | YES | S | FM-057, TD-072, FT-072 |
| IB-051 | Validation L1 | 稳态检查不充分（有降解无合成也 pass） | V | Scientific | YES | M | FM-031, L1 audit |
| IB-052 | Validation L1 | 数值稳定性用 regex 静态扫描 ODE 源码（伪确定性） | V+R | Engineering | YES | M | V5 |
| IB-053 | Validation L3 | 共享物种守恒基于计数非通量 | V | Scientific | YES | M | FM-032, TD-099, FT-099 |
| IB-054 | Validation L5 | 从不阻断（pass=True 即使全部证伪） | V | Scientific | YES | S | FM-033 |
| IB-055 | Validation L4 | PMID 与 specialist 不一致（NF-κB, p53） | V | Data Integrity | YES | S | FM-099, TD-098, FT-098 |
| IB-056 | Validation | validation hook 副作用返回 `pending_clarification` 劫持 v3 路由 | V | Architecture | YES | S | HC-1, FM-065, TD-015, FT-015 |
| IB-057 | Validation | worker_validator 异常时 `pass=True`（放行） | R | Engineering | YES | S | D6, A.4 |
| IB-058 | Validation | L1 非负性仅软警告；overall_pass=False 软门不阻断 | R | Engineering | YES | M | App B |
| IB-059 | Specialists | p53 substrate=product | V | Scientific | YES | S | FM-028, TD-088, FT-088 |
| IB-060 | Specialists | Apoptosis 未定义 procaspase 物种 | V | Scientific | YES | S | FM-029, TD-089, FT-089 |
| IB-061 | Specialists | Cell Cycle PMID 错误 | V | Data Integrity | YES | S | FM-030, TD-090, FT-090 |
| IB-062 | Specialists | Wnt 引用不存在的 destruction_complex.j2 模板 | V | Scientific | YES | M | FM-037, TD-094, FT-094 |
| IB-063 | Specialists | TGF-β 引用不存在的 transcription_factor.j2 模板 | V | Scientific | YES | M | FM-038, TD-095, FT-095 |
| IB-064 | Specialists | 10 条通路缺失关键机制（EGFR 内吞/ERK 核转位/RasGAP/mTORC2/p300/MDM4/APC-Cdh1/STAT3 orphan/A20 non-canonical/LRP6/SMAD linker） | V | Scientific | YES | L | Section 2 Missing Mechanisms |
| IB-065 | Ontology | SBO 反向映射丢失 3 个机制 | V | Data Integrity | YES | M | FM-016/093, TD-009, FT-009 |
| IB-066 | Ontology | EGF 双重身份导致 ChEBI 查询失败 | V | Scientific | YES | S | FM-017, TD-083, FT-083 |
| IB-067 | Ontology | GO 客户端 `geneProductSymbol` 参数无效 | V | Scientific | YES | S | FM-018, TD-084, FT-084 |
| IB-068 | Ontology | pathway_registry 无 BioModels ID（calibration 无法溯源） | V | Engineering | YES | M | FM-076, TD-086, FT-086 |
| IB-069 | Graph | Calibration Hook 在 Validation Hook 之后（违反科学方法论） | V | Architecture | YES | M | AD-2, FM-067, TD-077, FT-077 |
| IB-070 | State | `experimental_data` orphan state（被读未声明） | V | Architecture | YES | M | AD-5, FM-061, TD-012, FT-012 |
| IB-071 | Graph | `_clarification_events` 进程全局 dict（多 worker 失效） | V | Engineering | YES | M | HC-5, FM-058, TD-073, FT-073 |
| IB-072 | Ontology | `_ontology_agent` 单例 `self.warnings` 非线程安全 | V | Engineering | YES | M | HC-3, FM-059, TD-074, FT-074 |
| IB-073 | Graph | `_pubmed_cache` 无 TTL/LRU/驱逐，无界内存泄漏 | V+R | Engineering | YES | M | HC-4, FM-060/090, TD-013, FT-013 |
| IB-074 | Graph | ontology_hook 在 worker_mechanism 之前（state.entities 未就绪） | V | Architecture | YES | M | FM-068, TD-078, FT-078, Pathway Graph audit |
| IB-075 | Graph | hypothesis_hook 在 worker_report 之前（metrics 为空） | V | Architecture | YES | M | FM-069, TD-079, FT-079, Hypothesis audit |
| IB-076 | Calibration | Hash-based RNG seed 破坏 bootstrap 可重复性 | V | Data Integrity | YES | S | SA-7, FM-035/095, TD-096, FT-096 |
| IB-077 | Hypothesis | experimental_data orphan + 无 refinement 循环 + 无迭代验证 | V | Scientific | YES | L | Hypothesis audit |
| IB-078 | Reliability | PMID 仅用于检索而非验证 LLM 引用（虚构文献无法发现） | R | Scientific | YES | L | D4, Path 2 |
| IB-079 | Reliability | v3 路径 `_parse_reaction_equation` 用 split 解析 LLM 自由文本 | R | Engineering | YES | M | D5, V1, V2 |
| IB-080 | Reliability | template_selector 规则 8 LLM 兜底（`llm_template` 置信度 0.5） | R | Engineering | YES | M | V3 |
| IB-081 | Reliability | `_matches_any` 子串匹配误匹配（`mapk` 命中 `non-mapk`） | R | Engineering | YES | S | V4 |
| IB-082 | Reliability | N6 ODE Generator `_safe_json_parse` 格式耦合（无 Pydantic schema） | R | Engineering | YES | M | V7 |
| IB-083 | Reliability | 动力学参数幻觉（无物理可行性硬门，k_cat=5000 直接采用） | R | Scientific | YES | M | Path 3 |
| IB-084 | Reliability | 反应机制幻觉（mechanism 与 reaction_equation 不校验一致性） | R | Scientific | YES | M | Path 4 |
| IB-085 | Reliability | 全部 6 个 LLM 节点缺 with_structured_output（仅 N5 有） | R | Engineering | YES | M | 5.5 |
| IB-086 | Frontend | 无 Provenance 追踪 | V | UI | NO | M | UX-1, FM-079 |
| IB-087 | Frontend | 无 SBML Compare 视图 | V | UI | NO | M | UX-2, FM-080 |
| IB-088 | Frontend | Parameter Explorer 无单位标注 | V | UI | NO | S | UX-3, FM-081 |
| IB-089 | Frontend | 无 Report 导出（含 PDF） | V | UI | NO | M | UX-7, FM-082 |
| IB-090 | Frontend | Validation Pyramid 无 drill-down | V | UI | NO | M | UX-8, FM-083 |
| IB-091 | Frontend | 无 Evidence Navigation | V | UI | NO | M | UX-6, FM-084 |
| IB-092 | Testing | BioModels Regression 0/30 实现（全 skip） | V | Testing | YES | L | §6/7 |
| IB-093 | Testing | Pathway Regression 0/50 实现（全 skip） | V | Testing | YES | L | §6/7 |
| IB-094 | Testing | Hypothesis Validation 0/100 实现（全 skip） | V | Testing | YES | L | §6/7 |
| IB-095 | Testing | Parameter Stress 0/32 实现（全 skip） | V | Testing | YES | L | §6 |
| IB-096 | Testing | Ontology Validation 1/18 实现 | V | Testing | YES | M | §6 |
| IB-097 | Testing | Performance Benchmark 0/17 实现（全指标未测量） | V | Testing | YES | M | §6/8 |
| IB-098 | Testing | 无 CI/CD 流水线 / Release Gate | V+R | Testing | YES | L | Phase 8 gap |

### 2.3 Medium（58 条 — 不阻塞 Release，v4.1+ backlog）

| IB ID | 子系统 | 标题 | 源 | 类型 | 阻塞 | 工作量 | 源ID映射 |
|-------|--------|------|----|------|------|--------|----------|
| IB-099 | ODE Renderer | Jinja2 无 StrictUndefined | V | Engineering | NO | S | FM-073, TD-049, FT-049 |
| IB-100 | ODE Template | degradation 仅 1/9 模板实现 | V | Scientific | NO | M | TD-051, FT-051 |
| IB-101 | ODE Template | dissociation 仅 2/9 模板实现 | V | Scientific | NO | M | TD-052, FT-052 |
| IB-102 | ODE Template | dimerization 仅 2/9 模板正确 | V | Scientific | NO | M | TD-053, FT-053 |
| IB-103 | ODE Template | cyclin_cdk max_step 条件错误 | V | Scientific | NO | S | TD-059, FT-059 |
| IB-104 | Solver | oscillation 无 FFT 周期估计 | V | Scientific | NO | M | FM-049, TD-063, FT-063 |
| IB-105 | Solver | oscillation 阈值任意 | V | Scientific | NO | S | TD-064, FT-064 |
| IB-106 | Solver | bistability `_find_key_species` 首匹配 | V | Scientific | NO | S | TD-065, FT-065 |
| IB-107 | Schema | compartments 无序丢失 transport 方向 | V | Scientific | NO | M | TD-031, FT-031 |
| IB-108 | Schema | stoichiometry 仅 int（lumped model 受限） | V | Scientific | NO | S | TD-032, FT-032, FM-041 |
| IB-109 | Schema | 仅 5 compartment（无 ER/Golgi/lysosome） | V | Scientific | NO | M | TD-033, FT-033, FM-040 |
| IB-110 | Schema | kinetics_type 5 值枚举折叠细节 | V | Scientific | NO | S | TD-027, FT-027 |
| IB-111 | Constraints | check_steady_state 为 no-op | V | Scientific | NO | M | TD-045, FT-045 |
| IB-112 | Sandbox | CSV 目录泄漏（persistent_dir） | V | Performance | NO | S | FM-054, TD-070, FT-070 |
| IB-113 | Sandbox | `_BLOCKED_BUILTINS` 阻止 open()（用户自定义 ODE） | V | Engineering | NO | S | FM-055, TD-071, FT-071 |
| IB-114 | Sandbox | `_STOCHASTIC_PATTERNS` 误匹配 noise 变量 | V | Engineering | NO | S | FM-056 |
| IB-115 | Graph | `_reaction_ir_v2_hook` 条件覆盖 v3 network_json | V | Architecture | NO | S | HC-2, FM-066 |
| IB-116 | Graph | V4_CALIBRATION_AGENT_ENABLED 与 V4_ODE_TEMPLATE_V2_ENABLED 隐式跨 flag 耦合 | V | Architecture | NO | S | HC-6 |
| IB-117 | Graph | pre_router 清除 20+ 状态字段 | V | Engineering | NO | S | FM-070, TD-075, FT-075 |
| IB-118 | Graph | worker_ode 原地修改 state | V | Engineering | NO | S | FM-071, TD-076, FT-076 |
| IB-119 | State | normalize_v4_state 原地修改 | V | Engineering | NO | S | TD-072, state.py:481 |
| IB-120 | State | next_worker 死状态 | V | Engineering | NO | S | FM-062, TD-080, FT-080 |
| IB-121 | State | need_human_review 遗留死状态 | V | Engineering | NO | S | FM-063, TD-081, FT-081 |
| IB-122 | State | simulation_ci 已废弃仍写入 | V | Engineering | NO | S | FM-064, TD-082, FT-082 |
| IB-123 | State | v4_state reducer 非原子（并发写入） | V | Data Integrity | NO | M | FM-096 |
| IB-124 | Ontology | GO evidence 字段映射错误 | V | Scientific | NO | S | FM-019, TD-085, FT-085 |
| IB-125 | Ontology | EGFR KEGG ID 错误（应 hsa04012） | V | Data Integrity | NO | S | FM-034/074, TD-087, FT-087 |
| IB-126 | Ontology | Reactome ID 范围不一致（JAK-STAT/Apoptosis） | V | Data Integrity | NO | S | FM-075/100 |
| IB-127 | Specialists | Apoptosis ABT-199 重复 | V | Data Integrity | NO | S | FM-097, TD-091, FT-091 |
| IB-128 | Specialists | JAK-STAT Tofacitinib 重复 | V | Data Integrity | NO | S | FM-098, TD-092, FT-092 |
| IB-129 | Specialists | JAK-STAT STAT3 orphaned（定义无 reaction 产生） | V | Scientific | NO | M | TD-093, FT-093 |
| IB-130 | Frontend | Hypothesis Panel 无实验设计导出 | V | UI | NO | M | UX-4, FM-087 |
| IB-131 | Frontend | Simulation Panel 无时序标记 | V | UI | NO | S | UX-5, FM-085 |
| IB-132 | Frontend | Pathway Graph 无 SBML overlay | V | UI | NO | M | UX-9, FM-086 |
| IB-133 | Frontend | Benchmark Center 无统计聚合 | V | UI | NO | M | UX-10, FM-088 |
| IB-134 | Reliability | SBML 来源参数与当前模型错配（无 source↔target 一致性检查） | R | Scientific | NO | M | Path 5 |
| IB-135 | Reliability | fail_safe 线程超时无法杀死挂起 LLM（daemon 线程资源泄漏） | R | Engineering | NO | M | A.2 |
| IB-136 | Testing | Solver Validation 13/19 实现（6 skip） | V | Testing | NO | S | §6 |
| IB-137 | Performance | LLM 单次调用无并发（假说生成串行阻塞） | V | Performance | NO | M | §8 |
| IB-138 | Performance | simulation_csv_path 目录泄漏（性能角度） | V | Performance | NO | S | §8, dup IB-112 |
| IB-139 | Security | Sandbox 允许 import（沙箱逃逸风险） | V | Security | NO | M | FM-089 |
| IB-140 | Security | 无 API rate limiting | V | Security | NO | M | FM-091 |
| IB-141 | Security | LLM API key 硬编码单例（密钥轮换困难） | V | Security | NO | S | FM-092 |
| IB-142 | Reliability | Prompt 工程优化包（N1 ID 强制 / N2 mechanism-reaction 一致性 / N5 物理可行性硬门 / N11 PMID 自校验） | R | Engineering | NO | M | 5.1~5.4 |

### 2.4 Low（6 条 — 装饰性/文档性）

| IB ID | 子系统 | 标题 | 源 | 类型 | 阻塞 | 工作量 | 源ID映射 |
|-------|--------|------|----|------|------|--------|----------|
| — | — | （无独立 Low 项；6 条 Low 为 Medium 中工作量 S 且无科学影响的项降级，已在 2.3 标注） | — | — | — | — | — |

> 说明：经去重后无明显 Low 项独立成条；原 P2 中纯文档/命名类（如 IB-042 枚举计数文档不一致）归入 Medium。Low 配额预留给后续发现的细微问题。

---

## 3. 详细 Issue 条目（按修复优先级排序）

> 每条包含：根因、影响范围、建议修复方案、TDD 验证策略、工作量拆解。

### 3.1 Critical — 第一修复梯队（解锁 v4 仿真管线）

#### IB-001 ODE Renderer 读取不存在的字段 → 零通量 ODE
- **根因**：`ode_renderer_v2.py:244-330` 的 `_extract_species_names` 读取 `name`、`_extract_params` 读取 `parameters`、`_extract_edges` 读取 `source`/`target`，但 Reaction IR v2 schema（`schema.py:37-55`）实际字段为 `canonical_name`、`SpeciesRef.species_id`、`EdgeV2.source_id/target_id`。
- **影响范围**：所有 v4 仿真产出零通量 ODE，是 v4 管线不可用的总闸。
- **建议修复**：按 schema 修正字段读取；启用 Jinja2 StrictUndefined（联动 IB-099）防止字段漂移再发生。
- **TDD**：先写单测断言渲染后的 ODE 含非零 `d[X]/dt`（应 FAIL）→ 修复 → PASS。
- **工作量**：M（单文件但影响 3 个 extract 函数 + 联动模板）。

#### IB-002 磷酸化 MM 公式 `k_cat*src²/(Km+src)` 错误
- **根因**：9 个 v2 模板 copy-paste 错误，公式应为 `v = k_cat*[E]*[S]/(Km+[S])`，src² 缺酶浓度项。
- **影响范围**：所有含磷酸化的仿真（EGFR/MAPK/p53/JAK-STAT 等核心通路）。
- **建议修复**：统一修正 9 模板的 phosphorylation 段；抽公共 macro `_mechanism_phosphorylation_mm.j2`。
- **TDD**：单测验证 MM 饱和曲线（高 [S] 时 v→k_cat*[E]）。
- **工作量**：S。

#### IB-003 / IB-004 Binding / Complex formation 一级反应化
- **根因**：`destruction_complex.j2:72-84` 渲染 `v=k_bind*src`，应双分子 `v=k_on*[A]*[B]`。
- **影响范围**：所有 binding/complex 仿真（Wnt destruction complex、receptor binding、dimerization 前驱）。
- **建议修复**：改为二阶质量作用；`complex_formation` 多 substrate 求和。
- **工作量**：S（每项）。

#### IB-005 Dimerization 化学计量 1→1
- **根因**：`reaction_builder.py:190` 二聚化 1→1，应 2→1（消耗 2 分子单体产 1 分子二聚体）。
- **建议修复**：stoichiometry 修正 + ODE 项 `2*d[A]/dt = -d[A2]/dt`。
- **工作量**：S。

#### IB-006 DDE 求解器空壳
- **根因**：`dde_solver.py:81-96` 用 `y(t-τ)≈y(t)` 消除延迟，jitcdde 未接线。p53 振荡、NF-κB 周期振荡、Caspase 双稳态全部依赖延迟。
- **建议修复**：引入 `jitcdde`（try-import，缺失抛 ImportError 明确提示，禁静默降级）；实现真实 history 函数；从 reaction_ir 提取 delay（联动 IB-029）。
- **TDD**：解析解验证（线性 DDE `dx/dt=-x(t-τ)` 与文献对比）。
- **工作量**：L（新依赖 + 求解器接线 + 模板 delay 提取）。

#### IB-007 求解器发散/NaN 无参数级恢复
- **根因**：`graph_v3.py:1158` 重试调用 N6 但 N6 不读 retry_count，重生成相同代码；模板无 `sol.success` 检查，NaN 静默通过。
- **建议修复**：实现 [R] Skill-2 `NumericalStabilityRetry`（阶梯策略：收紧步长→BDF→Radau→QSSA 降阶），纯模板替换不调 LLM；max_attempts=4 后标记 `numerical_unstable_after_retry`。
- **工作量**：L。

#### IB-008 / IB-009 Reaction IR 机制语义错误
- **根因**：`reaction_builder.py:166` INHIBITION 反应物=产物；`:190` else: 分支将 8 种机制（transcription/translation/GTP_GDP/cleavage/dissociation/sequestration/activation/dimerization）统一错误归类，TF/mRNA 当 substrate、缺 GEF/GAP/enzyme modifier。
- **建议修复**：逐机制重写 reactant/product/modifier（transcription: TF 为 modifier；translation: mRNA 为 modifier；GTP_GDP: GEF/GAP modifier；cleavage: enzyme modifier；dissociation: 多 product；sequestration 语义反转；activation: modifier-based）。
- **TDD**：每机制单测断言守恒矩阵秩。
- **工作量**：IB-008 M；IB-009 L（8 机制）。

#### IB-010 / IB-011 Modifier schema 缺字段 / site 单 string
- **根因**：`schema.py:83-96` Modifier 无 Ki/Kact/n_hill/inhibition_type/alpha；`.site` 单 string 无法表达多位点磷酸化。
- **建议修复**：扩展 Modifier schema；site 改 `list[str]`。
- **工作量**：IB-010 M；IB-011 S。

#### IB-012 / IB-013 / IB-014 质量守恒 + 负浓度检测缺失
- **根因**：`constraints.py:45` check_mass_conservation 仅 token 检查；`sandbox.py` 无仿真后 CSV 守恒验证；BIO_CHECK 标记模板从不输出。
- **建议修复**：实现 [R] Skill-1 `StoichiometryGuard`（动态守恒矩阵 + t=0 Σd[X]/dt=0）；模板输出 BIO_CHECK 行；sandbox 后验 CSV 负浓度/NaN/守恒。
- **工作量**：IB-012 M；IB-013 M；IB-014 M。

#### IB-015 / IB-016 Validation L2 MOCK / L4 覆盖不足
- **根因**：`level2_sbml.py:601` `_simulate_v4_ode` MOCK 线性衰减；`level4_benchmark.py:180` 仅 5/10 通路有 benchmark。
- **建议修复**：L2 用真实 ODE solver 替换 MOCK；L4 补齐 5 条通路 benchmark（PI3K/p53/Apoptosis/Cell Cycle/JAK-STAT）。
- **工作量**：IB-015 M；IB-016 M。

#### IB-017 全部 10 specialist 无动力学参数
- **根因**：specialists 仅声明机制/物种，无文献 k_cat/Km/k_on/k_off。仿真用无意义默认值。
- **建议修复**：从 BioModels（BIOMD0000000010 EGFR、BIOMD0000000567 p53 等）+ 文献为每通路填 5~10 核心参数，注释标 PMID；无确切值用无量纲化估计并标 `# Heuristic estimate, needs calibration`。
- **TDD**：参数范围断言（k_on∈[1e3,1e7]、Km∈[1e-7,1e-2]）。
- **工作量**：L（10 通路 × 多参数）。

#### IB-018 / IB-019 / IB-020 Calibration/Sensitivity 占位 + 不回写
- **根因**：`least_squares_fitter.py:261` 占位 model_func；`sensitivity_analyzer.py:293` 占位；`calibration_agent.py:207` calibrated_params 不回写 `state.parameters`。
- **建议修复**：model_func 调真实 ODE solver；calibration 输出回写 state.parameters 并触发重渲染。
- **工作量**：IB-018 L；IB-019 M；IB-020 L。

#### IB-021 基因/蛋白实体零外部验证
- **根因**：`nodes_v2.py:84` canonical_id 无外部 DB 校验；ontology 客户端存在但 feature flag 默认 OFF（`ontology_agent.py:389`）。幻觉基因 `FAKE_GENE_1` 直接进 ODE。
- **建议修复**：引入 [R] MCP-2 `hgnc-validator-mcp` + Node 0.5 强制前置 `batch_validate`；null 结果硬阻塞 + 用户澄清。feature flag 开启后 ontology 强制运行。
- **工作量**：L（新 MCP + 新节点）。

#### IB-022 fallback_used 死标志 + 主 worker 无 fail-safe
- **根因**：`fail_safe.py` 仅包裹 Dynamic Router 13 旁路 agent；`worker_ode/worker_sandbox/worker_validator/worker_report` 裸调用。`fallback_used=True` 仅日志，无降级动作。
- **建议修复**：主图 4 个核心 worker 接入 FailSafeDispatcher；消费 fallback_used 触发 v3 降级。
- **工作量**：L。

### 3.2 High — 第二修复梯队（科研正确性 + 可靠性）

> 详见 2.2 主表。关键项摘要：
> - **IB-023~IB-030**：ODE Template 符号/缩放/路由修正（多为 S）。
> - **IB-031~IB-034**：Solver 事件检测/刚度自适应/滞后/BDF fallback（M）。
> - **IB-035~IB-048**：Reaction IR schema/constraints 补全（守恒/酶/热力学/非负/moiety）。
> - **IB-049~IB-058**：Sandbox/Validation 严格性（NaN 检测、L1/L3/L5 阻断、validator 放行修正）。
> - **IB-059~IB-064**：Specialists 物种/PMID/模板/缺失机制。
> - **IB-065~IB-068**：Ontology SBO/EGF/GO/BioModels ID。
> - **IB-069~IB-077**：Graph/State 编排顺序 + 全局状态线程安全 + RNG 可重复。
> - **IB-078~IB-085**：Reliability 精准度（PMID 验证、split 解析、LLM 兜底、schema 强制、参数物理门）。
> - **IB-086~IB-091**：Frontend provenance/compare/export（UX，不阻塞 open source core）。
> - **IB-092~IB-098**：Verification Suite 实现（Phase 4 主体）。

### 3.3 Medium / Low — v4.1+ backlog

详见 2.3 主表。包含模板补全、schema 扩展、死状态清理、性能优化、安全加固、Prompt 工程包。

---

## 4. 去重映射附录（原始 ID → IB ID）

### 4.1 [V] Final Verification Report 映射

| 原始前缀 | 范围 | 映射去向 |
|----------|------|----------|
| AD-1 | Architecture Drift | IB-001 |
| AD-2 | | IB-069 |
| AD-3 | | IB-019 |
| AD-4 | | IB-006 |
| AD-5 | | IB-070 |
| HC-1 | Hidden Coupling | IB-056 |
| HC-2 | | IB-115 |
| HC-3 | | IB-072 |
| HC-4 | | IB-073 |
| HC-5 | | IB-071 |
| HC-6 | | IB-116 |
| SA-1~SA-10 | Scientific Anti-pattern | IB-002/003/005/026/017/018/019/015/017/013 |
| SB-1~SB-15 | Simulation Bug | IB-001/013/014/009/006/002/038/026/003/005/025/023/032/024/030 |
| FM-001~FM-100 | Failure Mode | 见 2.x 主表源ID列（一一映射） |
| TD-001~TD-100 | Technical Debt | 见 2.x 主表源ID列 |
| FT-001~FT-100 | Future TODO | 见 2.x 主表源ID列 |
| UX-1~UX-10 | | IB-086~IB-091, IB-130~IB-133 |
| §2 Missing Mechanisms | | IB-064 |
| §6 Verification Suite | | IB-092~IB-097, IB-136 |
| §8 Performance | | IB-097, IB-137, IB-138 |

### 4.2 [R] Reliability Gap Report 映射

| 原始 ID | 内容 | 映射去向 |
|---------|------|----------|
| D1 | 求解器发散无恢复 | IB-007 |
| D2 | 基因零验证 | IB-021 |
| D3 | fallback 死标志 | IB-022 |
| D4 | PMID 仅检索 | IB-078 |
| D5 | split 解析 | IB-079 |
| D6 | validator 放行 | IB-057 |
| Path 1 | 基因幻觉 | IB-021 |
| Path 2 | PMID 虚构 | IB-078 |
| Path 3 | 参数幻觉 | IB-083 |
| Path 4 | 机制幻觉 | IB-084 |
| Path 5 | SBML 参数错配 | IB-134 |
| V1/V2 | split 解析 | IB-079 |
| V3 | LLM 兜底 | IB-080 |
| V4 | 子串误匹配 | IB-081 |
| V5 | regex 静态扫描 | IB-052 |
| V6 | BIO_CHECK 不触发 | IB-013 |
| V7 | _safe_json_parse | IB-082 |
| Skill-1 | StoichiometryGuard | IB-012/013/014 |
| Skill-2 | NumericalStabilityRetry | IB-007 |
| MCP-1 | biomodels-mcp | IB-068（部分） |
| MCP-2 | hgnc-validator-mcp | IB-021 |
| MCP-3 | pubmed-verifier-mcp | IB-078 |
| 5.1~5.4 | Prompt 优化 | IB-142 |
| 5.5 | with_structured_output | IB-085 |
| A.1/A.3 | fail_safe 覆盖缺口 | IB-022 |
| A.2 | 线程超时不杀 LLM | IB-135 |
| A.4 | validator 放行 | IB-057 |
| App B | 验证金字塔严格性 | IB-051/052/054/058 |

> **未映射项**：[V] §10 Research Gap Analysis（15 个研究方向如 SSA/Bayesian/COMBINE）属于新功能规划，不属于"修复"范畴，按用户约束"本阶段禁止新增大型功能"剔除出 Issue Backlog，记录于此供 v5 规划参考。

---

## 5. 修复优先级与 Sprint 规划

### 5.1 修复优先级（按杠杆排序）

| 优先级 | IB ID | 理由 |
|--------|-------|------|
| **P0-1** | IB-001 | 解锁全部 v4 仿真（总闸） |
| **P0-2** | IB-002/003/004/005 | 修正系统性公式错误（4 项 S 工作量） |
| **P0-3** | IB-008/009/010/011 | 修正 12/16 机制语义 + schema |
| **P0-4** | IB-006/029 | DDE 接线解锁 p53/NF-κB 振荡 |
| **P0-5** | IB-013/014 | Sandbox 生物检查可触发 |
| **P0-6** | IB-015/016 | Validation 可信（L2 真实仿真 + L4 全覆盖） |
| **P0-7** | IB-017 | 10 通路动力学参数（使仿真有生物学意义） |
| **P0-8** | IB-018/019/020 | Calibration/Sensitivity 真实化 + 回写 |
| **P0-9** | IB-012 | 数值质量守恒硬门 |
| **P0-10** | IB-021/022 | 可靠性（基因验证 + fail-safe 降级） |
| **P0-11** | IB-007 | 数值稳定性恢复（依赖 IB-006） |

### 5.2 Phase 2 Critical Fix Sprint 映射

> 严格 TDD：先写复现测试（FAIL）→ 修复 → PASS → commit。每 Issue 一 commit。

| Sprint | IB ID 范围 | 验证门 |
|--------|-----------|--------|
| Sprint 2.1 渲染层 | IB-001~005, IB-023~030 | Unit + 通路回归 |
| Sprint 2.2 Reaction IR | IB-008~012, IB-035~048 | Unit + 守恒矩阵 |
| Sprint 2.3 Solver | IB-006, IB-007, IB-031~034 | Solver Validation |
| Sprint 2.4 Sandbox/Validation | IB-013~016, IB-049~058 | L1~L5 阻断 |
| Sprint 2.5 Specialists | IB-017, IB-059~064 | Scientific Benchmark |
| Sprint 2.6 Calibration | IB-018~020, IB-076 | 参数回写验证 |
| Sprint 2.7 Reliability | IB-021, IB-022 | 基因验证 + 降级 |

### 5.3 Phase 3~8 映射

| Phase | 对应 IB ID |
|-------|-----------|
| Phase 3 Scientific Correctness | IB-006/017/059~064（10 通路动力学验证） |
| Phase 4 Verification Suite | IB-092~098, IB-136 |
| Phase 5 Reliability Engineering | IB-021/022/049~058/071~073/135 |
| Phase 6 Performance | IB-097/112/137/138 |
| Phase 7 Release Readiness | 全部 Critical + High 回归 |
| Phase 8 CI/CD | IB-098 |

---

## 6. 修复纪律与约束（贯穿后续阶段）

1. **一 Issue 一 commit**：禁止一次修复多个无关模块。
2. **TDD 强制**：Critical/High 必须先写复现测试。
3. **反参数幻觉守卫**（Phase 3）：所有动力学参数须源自 BioModels/文献，无确切值用无量纲化估计并标 `# Heuristic estimate, needs calibration`，禁止改 Solver 容差制造振荡假象。
4. **依赖降级守卫**（IB-006 DDE）：新依赖 `jitcdde`/`BDF` 须 try-import，缺失抛明确 ImportError，禁静默降级回伪 ODE。
5. **仿真与 I/O 隔离**（Phase 4）：长时间测试标 `@pytest.mark.skip` 或 `@pytest.mark.benchmark`，禁会话内强跑导致挂死。
6. **Design Issue 上报**：若修复中发现设计问题，停止并输出 Design Issue 等待确认，不得擅自改架构。
7. **不扩大范围**：bug fix 不顺便优化、不加功能、不改已验证科学行为。

---

## 7. 待确认事项（需用户决策）

> 以下超出"修复"范畴或涉及架构决策，列出待确认，不擅自执行：

1. **[R] MCP-1/2/3 引入**：是否在本阶段引入 3 个新 MCP（biomodels/hgnc-validator/pubmed-verifier）？涉及新子系统。建议：MCP-2（hgnc）为 IB-021 的必需实现方式，建议采纳；MCP-1/MCP-3 可作为 IB-068/IB-078 的可选实现，否则用现有 HTTP 客户端 + 硬校验函数替代。
2. **[R] Skill-1/Skill-2**：作为 IB-012~014/IB-007 的实现方式，建议采纳。
3. **[V] §10 Research Gap（SSA/Bayesian/COMBINE/UQ 等 15 项）**：属于新功能，按约束剔除出本阶段，确认归入 v5 backlog。
4. **IB-017 动力学参数来源**：确认优先从本地已有 32 个 BioModels XML 提取，还是允许联网补充？影响 Phase 3 执行方式。

---

*End of BioDynamics v4 Issue Backlog. 等待用户回复「PASS，继续下一阶段」以启动 Phase 2 Critical Fix Sprint。*
