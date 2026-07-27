# Design Decisions

本文记录当前代码体现的决策及其理由。它描述“为什么保持此方向”，不是说现有实现已
完全达到目标。改变决策前，应先提供 benchmark 和 migration plan。

## DD-01：结构化关系优先于 LLM 直接代码

Decision：LLM 在 N6 输出定性 `network_relations` JSON；Python 代码由 selector、rule
engine 和 Jinja template 渲染。

Why：减少任意代码、数值幻觉、不可复现方程和安全风险；让 template、参数和验证可审计。

Evidence：`prompts_v2.py::N6_ODE_PROMPT` 明确禁止 Python；
`nodes_v2.py::n6_ode_generator` 调用 `render_template`。

## DD-02：Reaction Graph / IR 是科学中间表示

Decision：V4 使用 Reaction IR v2 表达 species、reaction、mechanism、constraint、
composite reaction 和 state machine，再映射到 PathwayGraph/ODE。

Why：自然语言、检索证据和数值方程之间需要可验证层；没有 IR 时无法系统检查
守恒、compartment、mechanism 和 cross-talk。

## DD-03：Supervisor 负责调度，不负责科学计算

Decision：`pre_router/supervisor` 只生成计划、调度 worker、处理 HITL；科学产出由
workers/nodes 完成。

Why：规划和科学计算解耦，便于 mode 裁剪、追踪和失败定位。

## DD-04：v3 主干 + V4 feature-gated migration

Decision：保留 v3 Supervisor-Worker contract，V4 通过 coarse/fine flags 插入 hooks。

Why：允许逐层迁移和回滚，避免一次重写破坏 `/api/chat`、前端和既有 tests。

Consequence：产生 flat/grouped state、旧/新 template 和多套 naming 的维护债务。

## DD-05：BioModels 是可用时的 quantitative reference，不是所有通路的绝对真相

Decision：对已验证、语义匹配的 SBML 使用 simulation oracle/calibration reference；
没有正确模型时允许 skipped/structural path，但必须保留 role/method。

Why：BioModels 提供可执行方程和参数，比 LLM claim 强；但一个 BIOMD 模型可能只覆盖
子通路或属于错误生物系统，不能按 ID 存在就强行作为 ground truth。

## DD-06：Validation 必须是结构化结果

Decision：L1-L5 和 12-axis 输出结构化 status、metrics、method、evidence，而不是报告文字。

Why：报告生成不能替代数值判断；结构化结果才能驱动 benchmark、regression 和 UI。

Gap：当前 Pyramid 是软门，尚未完全实现“失败阻断/重试”的目标。

## DD-07：Evidence 必须带 provenance

Decision：参数和文献保留 source/PMID/unit/confidence/fallback；SA 提供 evidence
fusion、parameter provenance 和 decision log。

Why：无法追溯的正确数字也不能被当作可靠科学结论。

## DD-08：Prompt 优化排在机制/数据/仿真之后

Decision：loop controller 的修复顺序把 Prompt 放最后。

Why：peak time、数值爆炸、错误 BioModels、丢失 evidence 通常不是靠润色 Prompt 修复；
优先修机制、参数、simulation、validation。

## DD-09：确定性和可复现优先

Decision：sandbox 默认禁止随机性，设置 solver/timeout/max step，并导出 CSV。

Why：benchmark 和科学审计需要可重复 time series，不接受无法重放的随机输出。

## DD-10：失败可以降级，但不能冒充通过

Decision：外部服务和 optional dependency 失败时允许 fallback；fallback 必须被标记。

Why：研究工具应尽量返回可检查结果，同时不能隐瞒证据等级下降。

## DD-11：前端 SSE contract 保持兼容

Decision：`/api/chat` 保持 v3 prefix/wire frame；V4 新 REST 使用 `/api/v4`。

Why：允许后端科学层迁移时不同时重写主 UI。

## DD-12：真实 benchmark 优先于 synthetic schema test

Decision：真实发布判断由 `ScientificBenchmarkOrchestrator` 调完整工作流；synthetic 仅
显式 opt-in，用于 schema/快速检查。

Why：构造 metrics 不能证明 retrieval、ODE、solver、validation 和 report 真实协同。

## 决策变更模板

新增/修改决策需记录：Context、Decision、Alternatives、Scientific impact、Migration、
Rollback、Tests、Benchmarks、Data/Config version。
