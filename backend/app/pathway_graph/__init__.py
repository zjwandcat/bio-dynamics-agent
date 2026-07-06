# BioDynamics Agent v4 - Pathway Graph Layer（Phase 3 新增）
# 对应 v4 Scientific Architecture Part 2 Layer 2 + Part 3 通路模块化设计。
#
# 职责：构建通路拓扑图（nodes + edges + cross-talk edges），为 Reaction IR v2 / ODE Template v2
# 提供层级化的通路表示。Pathway Graph 是 v4 的中枢层，取代 v3 的扁平 network_json。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_GRAPH_ENABLED=false 时完全不执行，系统行为同 v3
# 2. Pathway Graph 不替代 network_json，而是与之共存（v4_pathway_graph 字段）
# 3. 不调用 LLM，纯规则构建（从 ontology + pathway_registry + reaction_ir 派生）
# 4. 不破坏 P1/P2 不可碰清单（不改 sandbox.py / ode_templates/ / nodes_v2.py 核心 / rag_client.py）
# 5. Pathway Graph 是 Reaction IR 的输入，不是 ODE 的直接输入（ODE 仍由 Reaction IR 渲染）

__all__ = [
    "schema",
    "builder",
    "initializer",
]
