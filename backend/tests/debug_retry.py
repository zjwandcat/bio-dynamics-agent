import uuid
from langgraph.graph import END, START, StateGraph

from app.graph import _route_after_audit
from app.nodes import node3_execute_sandbox, node4_audit_and_correct, node5_generate_report
from app.state import BioDynamicsState

call_count = {"n": 0}


def mock_node2(state: BioDynamicsState) -> dict:
    call_count["n"] += 1
    print(f"mock_node2 called #{call_count['n']}")
    if call_count["n"] == 1:
        return {"python_code": "for i in range(10)\n    print(i)"}
    return {
        "python_code": """import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 50)
y = np.sin(x)
plt.plot(x, y)
plt.savefig('simulation.png')
print('BIO_CHECK: X = 0.5')
print('fixed')
"""
    }


workflow = StateGraph(BioDynamicsState)
workflow.add_node("node2_generate_code", mock_node2)
workflow.add_node("node3_execute_sandbox", node3_execute_sandbox)
workflow.add_node("node4_audit_and_correct", node4_audit_and_correct)
workflow.add_node("node5_generate_report", node5_generate_report)
workflow.add_edge(START, "node2_generate_code")
workflow.add_edge("node2_generate_code", "node3_execute_sandbox")
workflow.add_edge("node3_execute_sandbox", "node4_audit_and_correct")
workflow.add_conditional_edges(
    "node4_audit_and_correct",
    _route_after_audit,
    {END: END, "node2_generate_code": "node2_generate_code", "node5_generate_report": "node5_generate_report"},
)
workflow.add_edge("node5_generate_report", END)
compiled = workflow.compile()

initial_state: BioDynamicsState = {
    "user_input": "A activates B",
    "network_json": {
        "nodes": [
            {"id": "A", "name": "A", "type": "Protein"},
            {"id": "B", "name": "B", "type": "Protein"},
        ],
        "edges": [{"source": "A", "target": "B", "interaction": "activation"}],
    },
    "retry_count": 0,
    "messages": [],
}
final_state = compiled.invoke(initial_state, {"configurable": {"thread_id": str(uuid.uuid4())}})
print("final execution_status:", final_state.get("execution_status"))
print("final retry_count:", final_state.get("retry_count"))
print("final auditor_status:", final_state.get("auditor_status"))
print("call_count:", call_count["n"])
