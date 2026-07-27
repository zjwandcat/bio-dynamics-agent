# Development Rules

## 必须遵守

1. Scientific correctness > feature count。
2. Structured evidence / BioModels / canonical benchmark > LLM assertion。
3. Validation failure 必须保持为 failure；不得用 skipped、degraded 或报告生成掩盖。
4. 修改科学行为必须有对应 pathway regression；修改共享模块需跑受影响的多通路测试。
5. 所有参数必须保留 value、unit、source、confidence/fallback provenance。
6. ODE 代码应由受控 template/rule 渲染；LLM 只输出结构化关系，不直接成为最终代码真相源。
7. 新状态字段必须同步 state、initial reset、producer、SSE、store、UI 和 tests。
8. 新通路必须同步 registry、specialist、IR/template、canonical、benchmark、frontend type 和 tests。
9. 所有 Feature Flag 必须可回退，测试 flag-on 与 flag-off。
10. 任何 benchmark pass 必须关联 commit/worktree、flags、依赖、data version 和结构化结果。

## 禁止

- 禁止为了展示增加 placeholder/fake feature/fake data。
- 禁止 hardcode benchmark expected result 到实现路径。
- 禁止将 synthetic benchmark 作为真实科学通过。
- 禁止在无测试/benchmark 的情况下新增 Agent、workflow 分支或隐式 fallback。
- 禁止把 HTTP 200、报告 markdown、PNG 或“无异常”视为科学通过。
- 禁止编造 PMID、BioModels ID、参数来源或置信度。
- 禁止在模板、Prompt、specialist、canonical 中重复维护同一真相而无一致性检查。
- 禁止输出或提交 `.env`、API key、credentials。
- 禁止 reset/clean/checkout 覆盖用户未提交工作树。
- 禁止默认搜索/修改 runtime logs、vector DB、CSV/PNG 和历史报告。

## 变更检查表

### Graph / Agent

- 更新 `build_workflow_v3` 拓扑说明。
- 检查 execution plan、step advance、clarification 和 recursion limit。
- 测 flag on/off、HITL、stop/cleanup、跨请求隔离。

### RAG / Parameter

- 保留 source/unit/confidence/fallback。
- 测无网络、无 collection、provider failure、online timeout。
- 运行受影响通路的动态 benchmark，不只测试 retrieval hit。

### ODE / Solver

- 不绕过 template selector/rule engine。
- 测 initial condition、units、mass conservation、non-negative、finite、peak/order。
- 保存修复前后 CSV/metrics，不用截图代替。

### Validation / Benchmark

- 不改变 skipped/degraded/fail 语义而不升级 schema。
- 负例必须继续失败。
- benchmark runner 不读取期望值后直接构造通过结果。

### Frontend / SSE

- 编辑前读 `frontend/AGENTS.md` 和本地 Next docs。
- 同步 backend event、parser、store、reset、component、unit test、API doc。
- 不硬编码 API host。

## Definition of Done

代码能运行不是完成。完成至少包括：问题可复现、最窄测试通过、相邻集成通过、科学
变更有 benchmark 证据、失败语义未弱化、无新增 runtime artifact、文档/PROJECT_STATE
与事实同步。
