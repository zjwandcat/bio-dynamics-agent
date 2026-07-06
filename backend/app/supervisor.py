# BioDynamics Agent - 多智能体编排器（Supervisor）
# 对应 1233.md 第一部分规范：定义智能体分工、任务分发与状态透明化。
#
# 设计思路：现有 LangGraph 6 节点工作流作为执行引擎保留，本模块在其之上
# 叠加一层"编排语义"：将节点映射为 4 个专业智能体，并在每次节点执行前后
# 生成 agent_dispatch 事件，供前端工作流追踪器渲染。

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentSpec:
    """单个专业智能体的元信息。"""

    name: str                # 前端展示名（英文，与 1233.md 一致）
    cn_label: str            # 中文短标签
    description: str         # 职责描述
    mapped_node: str         # 对应的 LangGraph 节点名
    icon: str = "cpu"        # 前端图标标识（lucide 名称）


# 智能体注册表：严格遵循 1233.md 第一部分定义的 4 个专业智能体
# Node0（MCP 术语查询）与 Node5（报告生成）作为辅助步骤，
# 通过 emit_dispatch 动态出现在前端工作流追踪器中
AGENT_REGISTRY: list[AgentSpec] = [
    AgentSpec(
        name="Terminology Agent",
        cn_label="术语查询",
        description="调用 MCP 工具标准化生物医学术语，注入定义上下文",
        mapped_node="node0_mcp_term_lookup",
        icon="book-open",
    ),
    AgentSpec(
        name="Mechanism Analysis Agent",
        cn_label="机制解析",
        description="解析自然语言，提取生物实体与相互作用，输出网络 JSON",
        mapped_node="node1_parse_network",
        icon="network",
    ),
    AgentSpec(
        name="Knowledge Retrieval Agent",
        cn_label="知识检索",
        description="高阶 RAG：查询重写 + 混合检索 + 重排序，提取动力学参数",
        mapped_node="node1_5_rag_search",
        icon="search",
    ),
    AgentSpec(
        name="PK/PD Modeling Agent",
        cn_label="PK/PD建模",
        description="推断给药途径、房室模型与 PK/PD 参数，支持联合用药协同分析",
        mapped_node="node1_6_pkpd_inference",
        icon="syringe",
    ),
    AgentSpec(
        name="Simulation Engineer Agent",
        cn_label="仿真工程",
        description="生成 ODE 代码并执行仿真，优先使用 RAG 真实参数",
        mapped_node="node2_generate_code",  # 含 node3 沙箱执行
        icon="flask-conical",
    ),
    AgentSpec(
        name="Biology Validator Agent",
        cn_label="生物审计",
        description="审计仿真结果的生物学合理性，失败则路由回工程节点重试",
        mapped_node="node4_audit_and_correct",
        icon="shield-check",
    ),
]


# -----------------------------------------------------------------------------
# v2 升级：12 节点 → 9 专业智能体的 AGENT_REGISTRY_V2
# 节点映射：
#   Entity & Planning Agent     → N1 + N2
#   Mechanism Retrieval Agent   → N3
#   Knowledge Graph Engineer    → N4
#   Parameter Retrieval Agent   → N5
#   Simulation Engineer Agent   → N6 + N7
#   Scientific Analytics Agent  → N8
#   Experimental Design Agent   → N9
#   Evidence Synthesis Agent    → N10
#   Scientific Report Agent     → N11
# -----------------------------------------------------------------------------
AGENT_REGISTRY_V2: list[AgentSpec] = [
    AgentSpec(
        name="Entity & Planning Agent",
        cn_label="实体与规划",
        description="NER 实体识别 + 机制规划器，输出 simulation_type / template / required_outputs",
        mapped_node="n1_ner_entity_normalize",
        icon="network",
    ),
    AgentSpec(
        name="Mechanism Retrieval Agent",
        cn_label="机制检索",
        description="基于 Mechanism RAG 集合检索通路/拓扑知识",
        mapped_node="n3_mechanism_rag",
        icon="search",
    ),
    AgentSpec(
        name="Knowledge Graph Engineer",
        cn_label="知识图谱",
        description="由实体与关系构建纯 Python 知识图谱（拓扑签名 / 环路检测）",
        mapped_node="n4_kg_builder",
        icon="git-branch",
    ),
    AgentSpec(
        name="Parameter Retrieval Agent",
        cn_label="参数检索",
        description="为每条边查询 Parameter RAG 真实动力学参数，程序注入禁止 LLM 修改",
        mapped_node="n5_parameter_rag",
        icon="sliders",
    ),
    AgentSpec(
        name="Simulation Engineer Agent",
        cn_label="仿真工程",
        description="LLM 输出 Network JSON + Jinja2 模板 + Rule Engine + AST 预检 + 沙箱执行",
        mapped_node="n6_ode_generator",
        icon="flask-conical",
    ),
    AgentSpec(
        name="Scientific Analytics Agent",
        cn_label="特征提取",
        description="从 simulation.csv 纯 NumPy 提取 Peak / Tpeak / t1/2 / AUC / Fold Change",
        mapped_node="n8_scientific_features",
        icon="bar-chart-3",
    ),
    AgentSpec(
        name="Experimental Design Agent",
        cn_label="实验设计",
        description="基于 Experiment RAG 推荐 Western/ELISA/qPCR/Flow 等验证手段",
        mapped_node="n9_experiment_rag",
        icon="beaker",
    ),
    AgentSpec(
        name="Evidence Synthesis Agent",
        cn_label="证据综合",
        description="基于 Evidence RAG 检索 PMID/DOI/Figure 支撑报告结论",
        mapped_node="n10_evidence_rag",
        icon="library",
    ),
    AgentSpec(
        name="Scientific Report Agent",
        cn_label="报告生成",
        description="LLM 输出 JSON 字段 + Python Markdown 模板渲染，禁止自由写 Markdown",
        mapped_node="n11_scientific_report",
        icon="file-text",
    ),
    AgentSpec(
        name="SBML Validator Agent",
        cn_label="SBML验证",
        description="P0-4 新增：对比模板仿真与 SBML 真实仿真，输出 error_diff / peak_time_diff / amplification_diff",
        mapped_node="worker_validator",
        icon="shield-check",
    ),
]


def get_agent_by_node(node_name: str) -> AgentSpec | None:
    """根据 LangGraph 节点名反查对应的智能体定义。"""
    # node3_execute_sandbox 归属 Simulation Engineer Agent
    if node_name == "node3_execute_sandbox":
        for agent in AGENT_REGISTRY:
            if agent.name == "Simulation Engineer Agent":
                return agent
        return None
    for agent in AGENT_REGISTRY:
        if agent.mapped_node == node_name:
            return agent
    return None


def get_agent_by_node_v2(node_name: str) -> AgentSpec | None:
    """v2：根据 12 节点名反查对应的智能体。"""
    # n7_sandbox_execute 归属 Simulation Engineer Agent
    if node_name == "n7_sandbox_execute":
        for agent in AGENT_REGISTRY_V2:
            if agent.name == "Simulation Engineer Agent":
                return agent
        return None
    for agent in AGENT_REGISTRY_V2:
        if agent.mapped_node == node_name:
            return agent
    return None


# v2 节点的标准 reasoning 文案
_REASONING_V2: dict[str, str] = {
    "n1_ner_entity_normalize": "Initial dispatch: 命名实体识别与归一化，输出 entity_id/name/type/canonical_id",
    "n2_mechanistic_planner": "Planning dispatch: 选择 simulation_type / template / required_outputs",
    "n3_mechanism_rag": "Mechanism RAG: 从四路集合的 mechanism 子集检索通路知识",
    "n4_kg_builder": "KG build: 纯 Python 构建知识图谱、拓扑签名与环路检测",
    "n5_parameter_rag": "Parameter RAG: 按边查询 Kd/Km/Vmax 等动力学参数，程序注入",
    "n6_ode_generator": "ODE template: LLM 输出 Network JSON + Jinja2 模板 + Rule Engine",
    "n7_sandbox_execute": "Sandbox: AST 预检 + 沙箱执行 + 错误分类",
    "n8_scientific_features": "Feature extraction: 纯 NumPy 提取 Peak/Tpeak/t1/2/AUC/Fold Change",
    "n9_experiment_rag": "Experiment RAG: 推荐 Western/ELISA/qPCR/Flow 验证手段",
    "n10_evidence_rag": "Evidence RAG: 检索 PMID/DOI/Figure 支撑报告结论",
    "n11_scientific_report": "Report: LLM 输出 JSON 字段 + Python Markdown 模板渲染",
}


@dataclass
class DispatchRecord:
    """单次智能体调度记录，会被写入 state.agent_dispatches 供前端渲染。"""

    target_agent: str
    reasoning: str
    status: str                 # idle / in_progress / completed / failed
    timestamp: float = field(default_factory=time.time)
    node_name: str = ""         # 关联的 LangGraph 节点
    latency_ms: float = 0.0     # 该智能体本次执行耗时（毫秒）

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_agent": self.target_agent,
            "reasoning": self.reasoning,
            "status": self.status,
            "timestamp": self.timestamp,
            "node_name": self.node_name,
            "latency_ms": round(self.latency_ms, 1),
        }


class BioDynamicsOrchestrator:
    """编排器：在节点执行前后生成 agent_dispatch 事件。

    无状态工具类，由 nodes.py 在各节点入口/出口调用。
    """

    @staticmethod
    def emit_dispatch(
        target_agent: str,
        reasoning: str,
        status: str = "in_progress",
        node_name: str = "",
        latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        """生成一条 agent_dispatch 事件记录（写入 state.agent_dispatches）。"""
        record = DispatchRecord(
            target_agent=target_agent,
            reasoning=reasoning,
            status=status,
            node_name=node_name,
            latency_ms=latency_ms,
        )
        return record.to_dict()

    @staticmethod
    def dispatch_for_node(node_name: str, status: str = "in_progress") -> dict[str, Any] | None:
        """根据节点名生成标准调度事件，附带的 reasoning 由编排策略决定。"""
        agent = get_agent_by_node(node_name)
        if agent is None:
            return None

        reasoning_map = {
            "node0_mcp_term_lookup": "Terminology dispatch: 调用 MCP 工具标准化生物医学术语",
            "node1_parse_network": "Initial dispatch: 解析用户假说，提取生物实体与相互作用网络",
            "node1_5_rag_search": "Parallel retrieval: 机制已解析，立即检索文献动力学参数",
            "node1_6_pkpd_inference": "PK/PD dispatch: 基于 RAG 药物参数推断房室模型与 Emax 效应参数",
            "node2_generate_code": "Engineering loop: 基于 RAG 真实参数生成 ODE 仿真代码",
            "node3_execute_sandbox": "Engineering loop: 在沙箱中执行仿真代码并捕获结果",
            "node4_audit_and_correct": "Validation check: 审计仿真结果生物学合理性",
        }
        reasoning = reasoning_map.get(node_name, f"Routing to {agent.name}")
        return BioDynamicsOrchestrator.emit_dispatch(
            target_agent=agent.name,
            reasoning=reasoning,
            status=status,
            node_name=node_name,
        )

    @staticmethod
    def complete_dispatch(node_name: str, latency_ms: float) -> dict[str, Any] | None:
        """节点执行完成时生成 completed 状态的调度记录。"""
        agent = get_agent_by_node(node_name)
        if agent is None:
            return None
        return BioDynamicsOrchestrator.emit_dispatch(
            target_agent=agent.name,
            reasoning=f"{agent.name} completed in {latency_ms:.0f}ms",
            status="completed",
            node_name=node_name,
            latency_ms=latency_ms,
        )

    @staticmethod
    def fail_dispatch(node_name: str, error: str, latency_ms: float = 0.0) -> dict[str, Any] | None:
        """节点执行失败时生成 failed 状态的调度记录。"""
        agent = get_agent_by_node(node_name)
        if agent is None:
            return None
        return BioDynamicsOrchestrator.emit_dispatch(
            target_agent=agent.name,
            reasoning=f"{agent.name} failed: {error}",
            status="failed",
            node_name=node_name,
            latency_ms=latency_ms,
        )

    # -------------------------------------------------------------------------
    # v2 编排方法
    # -------------------------------------------------------------------------
    @staticmethod
    def dispatch_for_node_v2(node_name: str, status: str = "in_progress") -> dict[str, Any] | None:
        """v2：根据节点名生成 12 节点标准调度事件。"""
        agent = get_agent_by_node_v2(node_name)
        if agent is None:
            return None
        reasoning = _REASONING_V2.get(node_name, f"Routing to {agent.name}")
        return BioDynamicsOrchestrator.emit_dispatch(
            target_agent=agent.name,
            reasoning=reasoning,
            status=status,
            node_name=node_name,
        )

    @staticmethod
    def complete_dispatch_v2(node_name: str, latency_ms: float) -> dict[str, Any] | None:
        """v2：节点完成时生成 completed 调度记录。"""
        agent = get_agent_by_node_v2(node_name)
        if agent is None:
            return None
        return BioDynamicsOrchestrator.emit_dispatch(
            target_agent=agent.name,
            reasoning=f"{agent.name} completed in {latency_ms:.0f}ms",
            status="completed",
            node_name=node_name,
            latency_ms=latency_ms,
        )

    @staticmethod
    def fail_dispatch_v2(node_name: str, error: str, latency_ms: float = 0.0) -> dict[str, Any] | None:
        """v2：节点失败时生成 failed 调度记录。"""
        agent = get_agent_by_node_v2(node_name)
        if agent is None:
            return None
        return BioDynamicsOrchestrator.emit_dispatch(
            target_agent=agent.name,
            reasoning=f"{agent.name} failed: {error}",
            status="failed",
            node_name=node_name,
            latency_ms=latency_ms,
        )


# 全局编排器实例，供 nodes.py 直接导入使用
orchestrator = BioDynamicsOrchestrator()
