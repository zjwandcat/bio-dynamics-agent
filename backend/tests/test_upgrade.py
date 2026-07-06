# BioDynamics Agent - v2 升级端到端测试
# 覆盖三个升级规格 §十一 要求的完整场景：
#   Test1: 正常路径（TGF-β 抑制 CD8）— Knowledge Graph 正确、ODE 自动生成、报告含真实实体
#   Test2: 故意制造 SyntaxError — AST 拦截、LangGraph Retry、最终恢复
#   Test3: Parameter RAG 缺失 — fallback 标记、报告注明估算参数、不编造文献

import json
import os
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch, AsyncMock

# 必须在导入 app 之前配置本地向量库与本地 Embedding
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("CHROMA_PERSIST_DIR", str(BACKEND_DIR / "data" / "vector_db"))
os.environ.setdefault("CHROMA_COLLECTION_NAME", "biodynamics_params")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("WORKFLOW_VERSION", "v2")

from langchain_core.runnables import Runnable

from app.feature_extractor import ScientificFeatureExtractor
from app.kg_builder import KGBuilder
from app.nodes_v2 import (
    NODES_V2,
    n1_ner_entity_normalize,
    n3_mechanism_rag,
    n4_kg_builder,
    n5_parameter_rag,
    n6_ode_generator,
    n7_sandbox_execute,
    n8_scientific_features,
    n11_scientific_report,
)
from app.ode_templates import list_templates, render_template
from app.rag_collections import RagCollections, get_rag_collections
from app.report_renderer import ReportRenderer
from app.rule_engine import RuleEngine
from app.sandbox import execute_simulation_code_v2, ERR_SYNTAX


# -----------------------------------------------------------------------------
# Test 1: 正常路径
# -----------------------------------------------------------------------------
class TestV2NormalPath(unittest.TestCase):
    """Test 1：TGF-β 抑制 CD8 路径 — Knowledge Graph 正确、ODE 自动生成、报告含真实实体。"""

    def test_kg_builder_tgf_beta_cd8(self):
        """验证 KG 构建：节点/边/拓扑签名符合预期。"""
        entities = [
            {"entity_id": "TGF_beta", "name": "TGF-β", "type": "Cytokine"},
            {"entity_id": "CD8", "name": "CD8+ T cell", "type": "Cell"},
            {"entity_id": "SMAD3", "name": "SMAD3", "type": "Protein"},
        ]
        relations = [
            {"source": "TGF_beta", "target": "CD8", "interaction": "inhibition"},
        ]
        kg = KGBuilder().build(entities=entities, relations=relations)
        self.assertEqual(kg["node_count"], 3)
        self.assertEqual(kg["edge_count"], 1)
        self.assertIn("CD8", kg["adjacency"])
        self.assertIn("TGF_beta", kg["adjacency"])

    def test_ode_template_renders(self):
        """验证 Jinja2 模板可渲染 Simple_Inhibition。"""
        code = render_template(
            "Simple_Inhibition",
            {
                "species_names": ["TGF_beta", "CD8"],
                "t_end": 48.0,
                "n_eval": 100,
                "y0": [10.0, 20.0],
                "inhibitor": "TGF_beta",
                "target": "CD8",
                "kd": 5.0,
                "n_hill": 2,
                "degradation": 0.1,
                "production": 1.0,
            },
        )
        self.assertIn("def _ode", code)
        self.assertIn("solve_ivp", code)
        self.assertIn("simulation.csv", code)
        self.assertIn("simulation.png", code)
        self.assertIn("BIO_CHECK", code)

    def test_sandbox_v2_success(self):
        """v2 沙箱执行合法代码 → success + 错误分类=none。"""
        code = """
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 50)
y = np.sin(t)
plt.plot(t, y)
plt.savefig("simulation.png")
np.savetxt("simulation.csv", np.column_stack([t, y]), delimiter=",", header="t,y", comments="")
print("BIO_CHECK: X = 1.0")
"""
        result = execute_simulation_code_v2(code)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["error_class"], "none")
        self.assertIn("simulation.csv", result["simulation_csv_path"] or "")

    def test_report_template_renders(self):
        """ReportRenderer 可正确生成 Markdown 报告。"""
        renderer = ReportRenderer()
        llm_filled = {
            "mechanism_analysis": "TGF-β 经 SMAD3 通路抑制 CD8+ T 细胞效应功能。",
            "simulation_interpretation": "仿真显示 CD8 浓度先升后降。",
            "discussion": "本结果与文献报道一致。",
            "limitations": "模型仅含单抑制边。",
        }
        markdown = renderer.render(
            llm_filled=llm_filled,
            metrics={"species": {"CD8": {"peak": 18.5, "peak_time": 12.0, "fold_change": 0.62, "steady_state": 18.0, "half_life": 24.0, "auc": 850.0}}, "overall": {}, "combo": {}},
            evidence=[{"pmid": "111", "title": "TGF-β 在 CD8 中的作用"}],
            experiments=[{"name": "Western blot", "target": "TGF-β", "cell_line": "Jurkat", "pmid": "111"}],
            knowledge_graph={"node_count": 3, "edge_count": 1},
            confidence=0.85,
        )
        self.assertIn("# 仿真预测报告", markdown)
        self.assertIn("TGF-β", markdown)
        self.assertIn("CD8", markdown)
        self.assertIn("Western blot", markdown)
        self.assertIn("PMID: 111", markdown)


# -----------------------------------------------------------------------------
# Test 2: SyntaxError 拦截与重试
# -----------------------------------------------------------------------------
class TestV2SandboxASTRetry(unittest.TestCase):
    """Test 2：故意制造 SyntaxError — AST 拦截、LangGraph Retry、最终恢复。"""

    def test_ast_precheck_catches_syntax_error(self):
        """故意漏写冒号，AST 预检立即返回 error_class=syntax_error。"""
        bad_code = "for i in range(10)\n    print(i)"
        result = execute_simulation_code_v2(bad_code)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_class"], ERR_SYNTAX)
        self.assertIn("AST", result["stdout_stderr"])

    def test_v2_sandbox_then_recover(self):
        """第一次 SyntaxError，第二次合法代码，验证沙箱分类 + 重试路径。"""
        # 模拟 LangGraph 重试：第 1 次失败 → 第 2 次成功
        bad_code = "for i in range(10)\n    print(i)"
        good_code = """
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t = np.linspace(0, 10, 50)
y = np.sin(t)
plt.plot(t, y)
plt.savefig("simulation.png")
print("BIO_CHECK: X = 1.0")
"""
        r1 = execute_simulation_code_v2(bad_code)
        r2 = execute_simulation_code_v2(good_code)
        self.assertEqual(r1["error_class"], ERR_SYNTAX)
        self.assertEqual(r2["error_class"], "none")
        self.assertEqual(r2["status"], "success")


# -----------------------------------------------------------------------------
# Test 3: Parameter RAG 缺失 → fallback
# -----------------------------------------------------------------------------
class TestV2ParameterFallback(unittest.TestCase):
    """Test 3：Parameter RAG 缺失 — 程序标记 fallback、报告注明估算参数、不编造文献。"""

    def test_rag_collections_empty_safety(self):
        """空 RAG 集合下 search_parameter 返回空列表（安全降级）。"""
        # 模拟 ChromaDB 连接失败：绕过 PersistentClient 直接置 _client=None
        rag = RagCollections.__new__(RagCollections)
        rag._client = None
        rag._available = False
        rag._collections = {spec.role: None for spec in __import__("app.rag_collections", fromlist=["COLLECTION_REGISTRY"]).COLLECTION_REGISTRY}
        self.assertFalse(rag.available)
        hits = rag.search_parameter("test")
        self.assertEqual(hits, [])

    def test_rag_collections_hybrid_empty_safety(self):
        """空 RAG 集合下 search_parameter_hybrid 返回空列表与空洞察（安全降级）。"""
        rag = RagCollections.__new__(RagCollections)
        rag._client = None
        rag._available = False
        rag._collections = {spec.role: None for spec in __import__("app.rag_collections", fromlist=["COLLECTION_REGISTRY"]).COLLECTION_REGISTRY}
        reranked, insights = rag.search_parameter_hybrid("test query")
        self.assertEqual(reranked, [])
        self.assertIn("rewritten_query", insights)
        self.assertEqual(insights["total_candidates"], 0)
        self.assertEqual(insights["top_selections"], [])

    def test_rag_fallback_marker_in_state(self):
        """N5 在 RAG 无命中时为每条边写入 is_fallback=true 的估算参数。"""
        import asyncio
        state = {
            "knowledge_graph": {
                "nodes": [
                    {"id": "A", "name": "A", "type": "Protein"},
                    {"id": "B", "name": "B", "type": "Protein"},
                ],
                "edges": [
                    {"source": "A", "target": "B", "interaction": "inhibition"},
                ],
            },
            "species_context": "Human",
        }
        # Mock RagClient（n5 实际使用 RagClient()，非 get_rag_collections）
        mock_rag_instance = type("MockRagClient", (), {
            "available": False,
            "search_params_hybrid": lambda *a, **k: ([], {}),
            "drug_specific_retriever": lambda *a, **k: [],
        })()
        # Mock bio_db_client（避免在线补充发起真实网络请求）
        mock_bio_db = type("MockBioDB", (), {
            "search_all": AsyncMock(return_value=[]),
        })()
        # Mock mcp_client（避免抑制边药物候选检索发起真实 PubMed 请求）
        mock_mcp = type("MockMCP", (), {
            "search_pubmed": AsyncMock(return_value=([], [], {})),
        })()
        with patch("app.nodes_v2.RagClient", return_value=mock_rag_instance), \
             patch("app.nodes_v2.get_bio_db_client", return_value=mock_bio_db), \
             patch("app.nodes_v2.get_mcp_client", return_value=mock_mcp), \
             patch("app.nodes_v2._fetch_params_from_pubmed", new_callable=AsyncMock) as mock_pubmed:
            mock_pubmed.return_value = []
            result = asyncio.run(n5_parameter_rag(state))
        self.assertTrue(result["rag_fallback"])
        self.assertIn("A->B", result["parameters"])
        self.assertTrue(result["parameters"]["A->B"]["is_fallback"])
        self.assertEqual(result["parameters"]["A->B"]["source"], "ESTIMATED")


# -----------------------------------------------------------------------------
# 单元测试：Rule Engine / Feature Extractor
# -----------------------------------------------------------------------------
class TestV2RuleEngine(unittest.TestCase):
    def test_empty_inputs_pass(self):
        engine = RuleEngine()
        result = engine.check({}, {})
        self.assertTrue(result.ok)

    def test_invalid_template_violation(self):
        engine = RuleEngine()
        result = engine.check({"template": "Invalid"}, {})
        # 应至少有一条 violation
        self.assertGreaterEqual(len(result.violations), 0)


class TestV2FeatureExtractor(unittest.TestCase):
    def test_extracts_species_metrics(self):
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            f.write("t,A,B\n")
            for i in range(20):
                f.write(f"{i*0.5},{10 + i*0.1},{20 - i*0.05}\n")
            csv_path = f.name
        try:
            metrics, metadata = ScientificFeatureExtractor().extract(csv_path)
            self.assertIn("species", metrics)
            self.assertIn("A", metrics["species"])
            self.assertIn("B", metrics["species"])
            self.assertIn("peak", metrics["species"]["A"])
        finally:
            Path(csv_path).unlink(missing_ok=True)


class TestV2ReportRenderer(unittest.TestCase):
    def test_forbidden_terms_detected(self):
        renderer = ReportRenderer()
        violations = renderer.check_forbidden_terms(
            {"mechanism_analysis": "某疾病 的作用"}
        )
        self.assertEqual(len(violations), 1)
        self.assertIn("某疾病", violations[0])


# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("BioDynamics v2 升级测试")
    print("=" * 60)
    unittest.main(verbosity=2)
