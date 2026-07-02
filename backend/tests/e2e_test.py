# BioDynamics Agent - 端到端自动化测试脚本
# 覆盖：正常路径、代码纠错重试、生物学常识约束、沙箱安全、RAG 检索。

import json
import os
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

# 必须在导入 app 之前配置本地向量库与本地 Embedding
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("CHROMA_PERSIST_DIR", str(BACKEND_DIR / "data" / "vector_db"))
os.environ.setdefault("CHROMA_COLLECTION_NAME", "biodynamics_params")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.graph import _route_after_audit, _route_after_parse
from app.nodes import (
    AuditorOutput,
    NetworkOutput,
    RAGDecisionOutput,
    node1_5_rag_search,
    node1_parse_network,
    node2_generate_code,
    node3_execute_sandbox,
    node4_audit_and_correct,
    node5_generate_report,
)
from app.prompts import NODE2_GENERATOR_PROMPT, NODE6_REPORT_PROMPT
from app.sandbox import execute_simulation_code
from app.state import BioDynamicsState


# -----------------------------------------------------------------------------
# 确定性 LLM 测试替身
# -----------------------------------------------------------------------------
class DeterministicLLM(Runnable):
    """用于正常路径测试的确定性 LLM 替身，避免端到端测试依赖外部 API 稳定性。"""

    _VALID_SIMULATION_CODE = """import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def ode(t, y):
    Tumor, TGF_beta, CD8, Inhibitor = y
    Tumor = np.maximum(Tumor, 0)
    TGF_beta = np.maximum(TGF_beta, 0)
    CD8 = np.maximum(CD8, 0)
    Inhibitor = np.maximum(Inhibitor, 0)

    dTumor = 0.0
    dTGF = 0.1 * Tumor - 0.3 * TGF_beta * Inhibitor - 0.15 * TGF_beta
    dCD8 = 0.05 * CD8 * (1 - CD8 / 100.0) - 0.08 * TGF_beta * CD8 + 0.02
    dInhibitor = -0.05 * Inhibitor
    return [dTumor, dTGF, dCD8, dInhibitor]


y0 = [10.0, 5.0, 20.0, 10.0]
sol = solve_ivp(ode, [0, 48], y0, t_eval=np.linspace(0, 48, 200), method='RK45')

plt.figure(figsize=(6, 4))
plt.plot(sol.t, sol.y[2], label='CD8+ T cell')
plt.plot(sol.t, sol.y[1], label='TGF-beta')
plt.xlabel('Time (h)')
plt.ylabel('Concentration')
plt.legend()
plt.tight_layout()
plt.savefig('simulation.png')

print(f"BIO_CHECK: CD8 = {sol.y[2][-1]:.4f}")
print(f"BIO_CHECK: TGF_beta = {sol.y[1][-1]:.4f}")
print('simulation done')
"""

    def __init__(self, schema=None):
        self.schema = schema

    def invoke(self, input, config=None, **kwargs):
        if self.schema is not None:
            return self._structured_output(self.schema)

        text = ""
        if hasattr(input, "to_string"):
            text = input.to_string()
        else:
            text = str(input)

        # Node 1：机制解析（返回 JSON 字符串）
        if "计算系统生物学家" in text or "解析用户输入" in text:
            return AIMessage(
                content=json.dumps(
                    {
                        "need_human_review": False,
                        "review_question": "",
                        "nodes": [
                            {"id": "Tumor", "name": "Tumor", "type": "Cell"},
                            {"id": "TGF_beta", "name": "TGF-beta", "type": "Molecule"},
                            {"id": "CD8", "name": "CD8+ T cell", "type": "Cell"},
                            {"id": "Inhibitor", "name": "TGF-beta inhibitor", "type": "Molecule"},
                        ],
                        "edges": [
                            {"source": "Tumor", "target": "TGF_beta", "interaction": "activation"},
                            {"source": "TGF_beta", "target": "CD8", "interaction": "inhibition"},
                            {"source": "Inhibitor", "target": "TGF_beta", "interaction": "inhibition"},
                        ],
                    },
                    ensure_ascii=False,
                )
            )

        # Node 2：方程生成
        if "生物数学建模专家" in text or "ODE" in text or "scipy" in text:
            return AIMessage(content=f"```python\n{self._VALID_SIMULATION_CODE}\n```")

        # Node 5/6：报告生成
        return AIMessage(
            content=(
                "# 仿真预测报告\n\n"
                "TGF-β 抑制剂给药后，CD8+ T 细胞浓度随时间逐步恢复，"
                "提示抑制肿瘤分泌的 TGF-β 可有效解除免疫抑制。"
            )
        )

    def with_structured_output(self, schema, **kwargs):
        return DeterministicLLM(schema=schema)

    @staticmethod
    def _structured_output(schema):
        if schema is NetworkOutput:
            return NetworkOutput(
                need_human_review=False,
                review_question="",
                nodes=[
                    {"id": "Tumor", "name": "Tumor", "type": "Cell"},
                    {"id": "TGF_beta", "name": "TGF-beta", "type": "Molecule"},
                    {"id": "CD8", "name": "CD8+ T cell", "type": "Cell"},
                    {"id": "Inhibitor", "name": "TGF-beta inhibitor", "type": "Molecule"},
                ],
                edges=[
                    {"source": "Tumor", "target": "TGF_beta", "interaction": "activation"},
                    {"source": "TGF_beta", "target": "CD8", "interaction": "inhibition"},
                    {"source": "Inhibitor", "target": "TGF_beta", "interaction": "inhibition"},
                ],
            )
        if schema is RAGDecisionOutput:
            return RAGDecisionOutput(
                param_found=False,
                selected_params=[],
                reasoning="未检索到匹配参数，使用估算值。",
                fallback_to_estimation=True,
            )
        if schema is AuditorOutput:
            return AuditorOutput(
                status="retry",
                correction_suggestion="请检查代码并修正错误。",
                failure_report="",
            )
        return schema()


# -----------------------------------------------------------------------------
# 通用辅助
# -----------------------------------------------------------------------------
def _build_retry_workflow(mock_node2):
    """构建一个最小化的重试测试工作流：从 node2 开始，经过执行/审计后可回退到 node2。"""
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
        {
            END: END,
            "node2_generate_code": "node2_generate_code",
            "node5_generate_report": "node5_generate_report",
        },
    )
    workflow.add_edge("node5_generate_report", END)
    return workflow.compile()


# -----------------------------------------------------------------------------
# 测试用例
# -----------------------------------------------------------------------------
class TestSandboxSecurity(unittest.TestCase):
    """沙箱安全测试：危险模块导入必须被拦截。"""

    def test_block_os_system(self):
        malicious_code = 'import os\nos.system("rm -rf /")\nprint("done")'
        result = execute_simulation_code(malicious_code)
        self.assertEqual(result["status"], "error")
        self.assertIn("安全拦截", result["stdout_stderr"])
        self.assertIn("os", result["stdout_stderr"])

    def test_block_subprocess(self):
        malicious_code = 'import subprocess\nsubprocess.run(["whoami"])\nprint("done")'
        result = execute_simulation_code(malicious_code)
        self.assertEqual(result["status"], "error")
        self.assertIn("安全拦截", result["stdout_stderr"])
        self.assertIn("subprocess", result["stdout_stderr"])

    def test_block_eval(self):
        malicious_code = 'eval("__import__(\'os\').system(\'dir\')")\nprint("done")'
        result = execute_simulation_code(malicious_code)
        self.assertEqual(result["status"], "error")
        self.assertIn("安全拦截", result["stdout_stderr"])
        self.assertIn("eval", result["stdout_stderr"])


class TestBiologicalConstraints(unittest.TestCase):
    """生物学常识约束测试：负浓度 / NaN / Inf 必须被识别为错误。"""

    def _valid_code_with_bio_output(self, bio_lines: str) -> str:
        """构造一段能正常运行、生成图片，但 BIO_CHECK 违规的代码。"""
        return f"""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 模拟一个能产生图片的合法脚本
x = np.linspace(0, 10, 50)
y = np.sin(x)
plt.plot(x, y)
plt.savefig('simulation.png')
{bio_lines}
print('simulation done')
"""

    def test_negative_concentration(self):
        code = self._valid_code_with_bio_output("print('BIO_CHECK: CD8 = -0.5')")
        result = execute_simulation_code(code)
        self.assertEqual(result["status"], "error")
        self.assertIn("生物学常识检查未通过", result["stdout_stderr"])
        self.assertIn("CD8", result["stdout_stderr"])

    def test_nan_concentration(self):
        code = self._valid_code_with_bio_output("print('BIO_CHECK: TGF_beta = nan')")
        result = execute_simulation_code(code)
        self.assertEqual(result["status"], "error")
        self.assertIn("生物学常识检查未通过", result["stdout_stderr"])

    def test_inf_concentration(self):
        code = self._valid_code_with_bio_output("print('BIO_CHECK: A = inf')")
        result = execute_simulation_code(code)
        self.assertEqual(result["status"], "error")
        self.assertIn("生物学常识检查未通过", result["stdout_stderr"])

    def test_node4_triggers_retry_on_bio_violation(self):
        code = self._valid_code_with_bio_output("print('BIO_CHECK: CD8 = -1.0')")
        result = execute_simulation_code(code)
        state: BioDynamicsState = {
            "execution_status": result["status"],
            "stdout_stderr": result["stdout_stderr"],
            "retry_count": 0,
        }
        audit = node4_audit_and_correct(state)
        self.assertIn(audit["auditor_status"], ("retry", "failed"))
        if audit["auditor_status"] == "retry":
            self.assertEqual(audit["retry_count"], 1)


class TestCodeErrorRetry(unittest.TestCase):
    """代码纠错与重试测试：模拟 node2 生成错误代码，验证重试不进入死循环。"""

    def test_syntax_error_then_success(self):
        call_count = {"n": 0}

        def mock_node2(state: BioDynamicsState) -> dict:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # 故意漏写冒号，触发 SyntaxError
                return {"python_code": "for i in range(10)\n    print(i)"}
            # 第二次返回合法代码
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

        workflow = _build_retry_workflow(mock_node2)
        initial_state: BioDynamicsState = {
            "user_input": "A 激活 B",
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
        final_state = workflow.invoke(initial_state, {"configurable": {"thread_id": str(uuid.uuid4())}})

        self.assertLessEqual(call_count["n"], 3, "重试次数不应超过 3 次")
        self.assertEqual(final_state["execution_status"], "success")
        self.assertTrue(final_state.get("image_base64", ""))
        self.assertEqual(final_state.get("retry_count", 0), 1)

    def test_persistent_error_graceful_failure(self):
        """连续 3 次语法错误后应优雅失败，不出现死循环。"""

        def mock_node2_always_bad(_state: BioDynamicsState) -> dict:
            return {"python_code": "for i in range(10)\n    print(i)"}

        workflow = _build_retry_workflow(mock_node2_always_bad)
        initial_state: BioDynamicsState = {
            "user_input": "A 激活 B",
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
        final_state = workflow.invoke(initial_state, {"configurable": {"thread_id": str(uuid.uuid4())}})

        self.assertEqual(final_state.get("auditor_status"), "failed")
        self.assertEqual(final_state.get("retry_count"), 3)
        self.assertIn("重试", final_state.get("failure_report", ""))


class TestRAGRetrieval(unittest.TestCase):
    """RAG 检索测试：输入与种子数据相关的查询，应能检索到真实参数。"""

    def test_mapk_params_retrieved(self):
        state: BioDynamicsState = {
            "user_input": "MAPK activation in ERK signaling pathway",
            "network_json": {
                "nodes": [
                    {"id": "MAPK", "name": "MAPK", "type": "Protein"},
                    {"id": "ERK", "name": "ERK", "type": "Protein"},
                ],
                "edges": [
                    {"source": "MAPK", "target": "ERK", "interaction": "activation"}
                ],
            },
            "messages": [],
        }
        result = node1_5_rag_search(state)
        self.assertTrue(result.get("rag_retrieved_params"))
        self.assertIsInstance(result["rag_retrieved_params"], list)
        # 至少有一条来自 MAPK 经典模型 BIOMD0000000010 或 BIOMD0000000012
        sources = [p.get("source_model", "") for p in result["rag_retrieved_params"]]
        self.assertTrue(
            any(s in ("BIOMD0000000010", "BIOMD0000000012") for s in sources),
            f"未检索到 MAPK 模型参数，检索结果来源: {sources}",
        )

    def test_rag_summary_shows_found(self):
        state: BioDynamicsState = {
            "user_input": "MAPK ERK activation kinetic parameter",
            "network_json": {
                "nodes": [
                    {"id": "MAPK", "name": "MAPK", "type": "Protein"},
                    {"id": "ERK", "name": "ERK", "type": "Protein"},
                ],
                "edges": [
                    {"source": "MAPK", "target": "ERK", "interaction": "activation"}
                ],
            },
            "messages": [],
        }
        result = node1_5_rag_search(state)
        self.assertIn("检索到", result.get("rag_summary", ""))


class TestNormalPath(unittest.TestCase):
    """正常路径端到端测试：完整走通 LangGraph 工作流。"""

    def test_tgf_beta_inhibitor_recovery(self):
        from app.graph import compiled_workflow

        user_input = "肿瘤细胞分泌TGF-β抑制CD8+ T细胞，测试TGF-β抑制剂给药后CD8的恢复。"
        thread_id = f"e2e-normal-{uuid.uuid4().hex[:8]}"
        initial_state: BioDynamicsState = {
            "user_input": user_input,
            "retry_count": 0,
            "messages": [],
        }

        # 正常路径使用确定性 LLM 替身，确保端到端链路（解析→RAG→生成→沙箱→审计→报告）
        # 不依赖外部 API 的实时可用性，同时真实执行沙箱并生成图片。
        with patch("app.nodes.llm", DeterministicLLM()):
            final_state = compiled_workflow.invoke(
                initial_state,
                {"configurable": {"thread_id": thread_id}},
            )

        self.assertFalse(final_state.get("need_human_review", False))
        self.assertEqual(final_state.get("execution_status"), "success")
        self.assertTrue(final_state.get("image_base64", ""))
        self.assertTrue(final_state.get("final_report", ""))
        self.assertLessEqual(final_state.get("retry_count", 0), 3)


# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 打印关键配置，便于排查
    print("=" * 60)
    print("BioDynamics Agent E2E 测试启动")
    print(f"CHROMA_PERSIST_DIR: {settings.CHROMA_PERSIST_DIR}")
    print(f"CHROMA_COLLECTION_NAME: {settings.CHROMA_COLLECTION_NAME}")
    print(f"EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")
    print(f"EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
    print(f"LLM MODEL: {settings.OPENAI_MODEL}")
    print("=" * 60)
    unittest.main(verbosity=2)
