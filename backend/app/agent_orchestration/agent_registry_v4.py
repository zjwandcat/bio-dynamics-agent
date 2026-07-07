# BioDynamics Agent v4 - Agent Registry V4（Phase 6 / Task 6.5.2）
#
# 13 Agent 注册表：定义 v4 Dynamic Router 编排的 13 个智能体元信息。
#
# 设计原则（铁律）：
# 1. 注册表为静态元数据，不持有 Agent 实例（Agent 模块通过 mapped_module 懒加载）
# 2. 不修改 v3 任何字段；仅向 Dynamic Router 提供分派计划与元信息
# 3. 13 Agent = Ontology + Pathway Planner + Specialist Group(10) + Cross-talk Coordinator
#              + Reaction Builder + Mechanism Builder + ODE Builder + SBML Grounder
#              + Calibration + Simulation Planner + Validation + Hypothesis + Parameter Agent
# 4. Specialist Group 是一个"组"条目，运行时按 v4_pathway_class 分发到对应 Specialist
# 5. P6/Task 6.6 未实现的 Agent（mechanism_builder/ode_builder/simulation_planner/
#    parameter_agent）以 lazy stub 形式注册，DynamicRouter.execute_agent 会处理 ImportError
#
# 参考：
# - spec.md Part 6 Dynamic Router（第 393-398 行）
# - tasks.md SubTask 6.5.2

from __future__ import annotations

from dataclasses import dataclass


# =============================================================================
# AgentSpecV4 数据类
# =============================================================================
@dataclass(frozen=True)
class AgentSpecV4:
    """单个 v4 Agent 的元信息。

    与 ``app.supervisor.AgentSpec`` 的区别：
    - 增加 ``agent_id``（路由键）与 ``phase``（P1/P4/P5/P6）
    - 使用 ``mapped_module``（模块路径）支持懒加载，避免循环依赖
    - 增加 ``timeout_seconds``（fail_safe 超时阈值）
    - ``mapped_node`` 改为 ``mapped_module``（v4 Agent 不直接对应 LangGraph 节点）

    Attributes:
        agent_id: 路由键（如 ``"ontology"``），DynamicRouter 用此键查找 Agent
        name: 前端展示名（英文）
        cn_label: 中文短标签
        description: 职责描述
        phase: 所属阶段（``"P1"`` / ``"P4"`` / ``"P5"`` / ``"P6"``）
        mapped_module: Agent 主类的完整模块路径（用于 ``importlib.import_module`` 懒加载）
        timeout_seconds: 单次调度超时阈值（秒），默认 30s
        icon: 前端图标标识（lucide 名称）
    """

    agent_id: str
    name: str
    cn_label: str
    description: str
    phase: str
    mapped_module: str
    timeout_seconds: int = 30
    icon: str = "cpu"


# =============================================================================
# Specialist Group 常量
# =============================================================================
# Specialist Group 是一个"组"条目，运行时按 v4_pathway_class 分发到 10 个 Specialist 之一
# （或多个，多通路场景）。具体分发逻辑由 PathwayClassDispatcher 负责。
SPECIALIST_AGENT_ID: str = "pathway_specialist_group"


# =============================================================================
# 13 Agent 注册表
# =============================================================================
# 注意：13 Agent 按类型计数，其中 pathway_specialist_group 是一个组条目，
# 内部按 v4_pathway_class 分发到 10 个 Specialist（EGFR/MAPK/PI3K/p53/Apoptosis/
# CellCycle/JAKSTAT/NFKB/Wnt/TGFbeta）。
#
# Agent → Phase 映射：
# - P1: Ontology Agent
# - P4: Pathway Planner / Specialist Group / Cross-talk Coordinator
# - P2: Reaction Builder（被 v4 复用，phase 标注为 "P5" 因为在 v4 流水线中归属 SBML Grounder 段；
#        实际属 P2，此处按 v4 编排阶段标注为 "P5"）
# - P5: SBML Grounder / Calibration / Validation
# - P6: Mechanism Builder / ODE Builder / Simulation Planner / Hypothesis / Parameter Agent
#
# P6/Task 6.6 未实现的 Agent（mapped_module 指向尚不存在的模块）：
# - mechanism_builder  → app.agents_v4.mechanism_builder（Task 6.6 创建）
# - ode_builder        → app.agents_v4.ode_builder（Task 6.6 创建）
# - simulation_planner → app.agents_v4.simulation_planner（Task 6.6 创建）
# - parameter_agent    → app.agents_v4.parameter_agent（Task 6.6 创建）
# DynamicRouter.execute_agent 会捕获 ImportError 返回 {}（stub）
AGENT_REGISTRY_V4: dict[str, AgentSpecV4] = {
    # 1. Ontology Agent（P1）
    "ontology": AgentSpecV4(
        agent_id="ontology",
        name="Ontology Agent",
        cn_label="本体标准化",
        description="HGNC/UniProt/ChEBI/GO/SBO 标准化，输出 v4_ontology_entities",
        phase="P1",
        mapped_module="app.ontology.ontology_agent",
        timeout_seconds=30,
        icon="book-open",
    ),
    # 2. Pathway Planner（P4）
    "pathway_planner": AgentSpecV4(
        agent_id="pathway_planner",
        name="Pathway Planner Agent",
        cn_label="通路识别",
        description="规则优先 + LLM 兜底，输出 v4_pathway_class（单通路/多通路/UNKNOWN）",
        phase="P4",
        mapped_module="app.pathways.pathway_planner",
        timeout_seconds=30,
        icon="route",
    ),
    # 3. Pathway Specialist Group（P4，10 Specialist 的组入口）
    "pathway_specialist_group": AgentSpecV4(
        agent_id="pathway_specialist_group",
        name="Pathway Specialist Group",
        cn_label="通路专家组",
        description="按 v4_pathway_class 分发到 10 Specialist（EGFR/MAPK/PI3K/p53/Apoptosis/"
        "CellCycle/JAKSTAT/NFKB/Wnt/TGFbeta），输出通路特异 Reaction IR 片段",
        phase="P4",
        mapped_module="app.pathways.specialist_hook",
        timeout_seconds=30,
        icon="users",
    ),
    # 4. Cross-talk Coordinator（P4）
    "crosstalk_coordinator": AgentSpecV4(
        agent_id="crosstalk_coordinator",
        name="Cross-talk Coordinator Agent",
        cn_label="通路协调",
        description="多通路 shared species 同步 + cross-talk edges 注入 + 时间尺度对齐",
        phase="P4",
        mapped_module="app.crosstalk.coordinator",
        timeout_seconds=30,
        icon="share-2",
    ),
    # 5. Reaction Builder（P2，被 v4 复用）
    "reaction_builder": AgentSpecV4(
        agent_id="reaction_builder",
        name="Reaction Builder Agent",
        cn_label="反应构建",
        description="从 PathwayGraph 构建 Reaction IR v2（species/reactions/composite_reactions）",
        phase="P2",
        mapped_module="app.reaction_ir_v2.reaction_builder",
        timeout_seconds=30,
        icon="git-branch",
    ),
    # 6. Mechanism Builder（P6/Task 6.6，lazy stub）
    "mechanism_builder": AgentSpecV4(
        agent_id="mechanism_builder",
        name="Mechanism Builder Agent",
        cn_label="机制构建",
        description="从 Reaction IR 推断机制类型（phosphorylation/bistable/oscillatory 等）",
        phase="P6",
        mapped_module="app.agents_v4.mechanism_builder",
        timeout_seconds=30,
        icon="workflow",
    ),
    # 7. ODE Builder（P6/Task 6.6，lazy stub）
    "ode_builder": AgentSpecV4(
        agent_id="ode_builder",
        name="ODE Builder Agent",
        cn_label="ODE 构建",
        description="从 Reaction IR + Pathway Graph 渲染 v4 ODE 代码（ode_templates_v2）",
        phase="P6",
        mapped_module="app.agents_v4.ode_builder",
        timeout_seconds=30,
        icon="code",
    ),
    # 8. SBML Grounder（P5）
    "sbml_grounder": AgentSpecV4(
        agent_id="sbml_grounder",
        name="SBML Grounder Agent",
        cn_label="SBML 溯源",
        description="建立 ODE ↔ Reaction ↔ SBML ↔ Parameter ↔ PMID 五级映射链",
        phase="P5",
        mapped_module="app.sbml_grounder.grounder_agent",
        timeout_seconds=30,
        icon="link",
    ),
    # 9. Calibration（P5）
    "calibration": AgentSpecV4(
        agent_id="calibration",
        name="Calibration Agent",
        cn_label="参数校准",
        description="用 BioModels reference 或用户实验数据拟合参数，输出置信区间",
        phase="P5",
        mapped_module="app.calibration.calibration_agent",
        timeout_seconds=30,
        icon="sliders",
    ),
    # 10. Simulation Planner（P6/Task 6.6，lazy stub）
    "simulation_planner": AgentSpecV4(
        agent_id="simulation_planner",
        name="Simulation Planner Agent",
        cn_label="仿真规划",
        description="规划仿真场景（dose-response/time-course/knockout），输出 simulation_plan",
        phase="P6",
        mapped_module="app.agents_v4.simulation_planner",
        timeout_seconds=30,
        icon="calendar-clock",
    ),
    # 11. Validation（P5）
    "validation": AgentSpecV4(
        agent_id="validation",
        name="Validation Pyramid Agent",
        cn_label="五层验证",
        description="5 层 Validation Pyramid（internal/SBML/crosstalk/benchmark/hypothesis）",
        phase="P5",
        mapped_module="app.validation_v2.validation_agent",
        timeout_seconds=30,
        icon="shield-check",
    ),
    # 12. Hypothesis（P6）
    "hypothesis": AgentSpecV4(
        agent_id="hypothesis",
        name="Hypothesis Agent",
        cn_label="假设生成",
        description="基于 metrics + validation 生成可证伪的实验假设列表",
        phase="P6",
        mapped_module="app.hypothesis.hypothesis_agent",
        timeout_seconds=30,
        icon="lightbulb",
    ),
    # 13. Parameter Agent（P6/Task 6.6，lazy stub）
    "parameter_agent": AgentSpecV4(
        agent_id="parameter_agent",
        name="Parameter Agent",
        cn_label="参数管理",
        description="管理参数优先级（RAG > SBML > Inferred）与跨通路参数隔离",
        phase="P6",
        mapped_module="app.agents_v4.parameter_agent",
        timeout_seconds=30,
        icon="database",
    ),
}


# =============================================================================
# 查询接口
# =============================================================================
def get_agent_spec(agent_id: str) -> AgentSpecV4 | None:
    """按 agent_id 查询 Agent 元信息。

    Args:
        agent_id: 路由键（如 ``"ontology"`` / ``"pathway_planner"``）

    Returns:
        ``AgentSpecV4`` 实例；未注册时返回 ``None``（调用方负责降级）
    """
    return AGENT_REGISTRY_V4.get(agent_id)


def list_agent_ids() -> list[str]:
    """列出所有已注册的 agent_id。

    Returns:
        agent_id 字符串列表（按注册顺序，Python 3.7+ dict 保序）。
        长度恒为 13（13 Agent 注册表）
    """
    return list(AGENT_REGISTRY_V4.keys())


def count_agents() -> int:
    """返回注册表中 Agent 总数（恒为 13）。"""
    return len(AGENT_REGISTRY_V4)


def is_agent_registered(agent_id: str) -> bool:
    """检查指定 agent_id 是否已注册。

    Args:
        agent_id: 路由键

    Returns:
        ``True`` 表示已注册；``False`` 表示未注册
    """
    return agent_id in AGENT_REGISTRY_V4


__all__ = [
    "AgentSpecV4",
    "AGENT_REGISTRY_V4",
    "SPECIALIST_AGENT_ID",
    "get_agent_spec",
    "list_agent_ids",
    "count_agents",
    "is_agent_registered",
]
