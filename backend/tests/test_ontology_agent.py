# BioDynamics Agent v4 - Ontology Agent 单元测试
# 覆盖 P1 完成标准的 5 个测试用例：
# 1. HGNC API 查询 "EGFR" → HGNC:3236（mock，避免依赖网络）
# 2. UniProt API 查询 "EGFR" → P00533（mock）
# 3. API 失败时降级为 verified=false，不抛异常
# 4. 缓存命中时不调 API
# 5. Feature Flag false 时 Ontology Agent 不执行
#
# 运行：cd backend && python -m pytest tests/test_ontology_agent.py -v
# 或：  cd backend && python tests/test_ontology_agent.py

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class TestSboTerms(unittest.TestCase):
    """SBO term 常量定义测试。"""

    def test_17_mechanisms_defined(self):
        """验证 17 类机制全部定义（架构文档 Part 4 树状图列出 17 类，
        实现含 cytoplasm_translocation 复用 nuclear_import SBO，共 18 条映射）。"""
        from app.ontology.sbo_terms import MECHANISM_TO_SBO
        self.assertGreaterEqual(len(MECHANISM_TO_SBO), 17)

    def test_phosphorylation_sbo(self):
        """磷酸化对应 SBO:0000216。"""
        from app.ontology.sbo_terms import get_sbo_term
        self.assertEqual(get_sbo_term("phosphorylation"), "SBO:0000216")

    def test_inhibition_sbo(self):
        """抑制对应 SBO:0000169。"""
        from app.ontology.sbo_terms import get_sbo_term
        self.assertEqual(get_sbo_term("inhibition"), "SBO:0000169")

    def test_unknown_mechanism_returns_none(self):
        """未知机制返回 None。"""
        from app.ontology.sbo_terms import get_sbo_term
        self.assertIsNone(get_sbo_term("nonexistent_mechanism"))

    def test_reverse_lookup(self):
        """反向查询：SBO term → 机制名。"""
        from app.ontology.sbo_terms import get_mechanism_name
        self.assertEqual(get_mechanism_name("SBO:0000216"), "phosphorylation")


class TestPathwayRegistry(unittest.TestCase):
    """通路注册表测试。"""

    def test_10_pathways_defined(self):
        """验证 10 条通路全部定义。"""
        from app.ontology.pathway_registry import PATHWAY_REGISTRY
        self.assertEqual(len(PATHWAY_REGISTRY), 10)

    def test_egfr_pathway_keywords(self):
        """EGFR_RTK 通路关键词 ≥8 个。"""
        from app.ontology.pathway_registry import PATHWAY_REGISTRY
        egfr = PATHWAY_REGISTRY["EGFR_RTK"]
        self.assertGreaterEqual(len(egfr.keywords), 8)

    def test_lookup_pathway_egfr(self):
        """文本含 EGFR 命中 EGFR_RTK 通路。"""
        from app.ontology.pathway_registry import lookup_pathway
        self.assertEqual(lookup_pathway("EGF activates EGFR"), "EGFR_RTK")

    def test_lookup_pathway_nfkb(self):
        """文本含 NF-κB 命中 NF_KB 通路。"""
        from app.ontology.pathway_registry import lookup_pathway
        self.assertEqual(lookup_pathway("NF-κB signaling pathway"), "NF_KB")

    def test_lookup_pathway_no_match(self):
        """无关键词命中返回 None。"""
        from app.ontology.pathway_registry import lookup_pathway
        self.assertIsNone(lookup_pathway("random unrelated text"))


class TestHgncClient(unittest.TestCase):
    """HGNC API 客户端测试（mock）。"""

    def setUp(self):
        """每个测试前清空缓存，避免缓存干扰。"""
        from app.ontology._cache import cache_clear
        cache_clear("hgnc")

    def test_hgnc_query_egfr_mock(self):
        """测试 1：HGNC 查询 "EGFR" → HGNC:3236（mock）。"""
        from app.ontology.hgnc_client import query_hgnc
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": {
                "docs": [{
                    "hgnc_id": "HGNC:3236",
                    "symbol": "EGFR",
                    "name": "epidermal growth factor receptor",
                    "uniprot_ids": ["P00533"],
                    "entrez_id": 1956,
                    "ensembl_gene_id": "ENSG00000146648",
                    "gene_group": [],
                }]
            }
        }
        mock_response.raise_for_status.return_value = None
        with patch("app.ontology.hgnc_client.requests.get", return_value=mock_response):
            result = query_hgnc("EGFR")
        self.assertIsNotNone(result)
        self.assertEqual(result["hgnc_id"], "HGNC:3236")
        self.assertEqual(result["uniprot_id"], "P00533")

    def test_hgnc_query_empty_symbol(self):
        """空符号返回 None。"""
        from app.ontology.hgnc_client import query_hgnc
        self.assertIsNone(query_hgnc(""))
        self.assertIsNone(query_hgnc("   "))

    def test_hgnc_query_no_results(self):
        """查询无结果返回 None。"""
        from app.ontology.hgnc_client import query_hgnc
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": {"docs": []}}
        mock_response.raise_for_status.return_value = None
        with patch("app.ontology.hgnc_client.requests.get", return_value=mock_response):
            result = query_hgnc("NONEXISTENT_GENE_XYZ")
        self.assertIsNone(result)

    def test_hgnc_query_failure_degrades_gracefully(self):
        """测试 3：API 失败时降级返回 None，不抛异常。"""
        from app.ontology.hgnc_client import query_hgnc
        with patch(
            "app.ontology.hgnc_client.requests.get",
            side_effect=Exception("network error"),
        ):
            # 减少 backoff 等待时间
            with patch("app.ontology.hgnc_client.time.sleep"):
                result = query_hgnc("EGFR")
        self.assertIsNone(result)


class TestUniprotClient(unittest.TestCase):
    """UniProt API 客户端测试（mock）。"""

    def setUp(self):
        from app.ontology._cache import cache_clear
        cache_clear("uniprot")

    def test_uniprot_query_egfr_mock(self):
        """测试 2：UniProt 查询 "EGFR" → P00533（mock）。"""
        from app.ontology.uniprot_client import query_uniprot
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{
                "primaryAccession": "P00533",
                "proteinDescription": {
                    "recommendedName": {"fullName": {"value": "Epidermal growth factor receptor"}}
                },
                "genes": [{"geneName": {"value": "EGFR"}}],
                "organism": {"scientificName": "Homo sapiens"},
                "sequence": {"length": 1210},
                "comments": [],
            }]
        }
        mock_response.raise_for_status.return_value = None
        with patch("app.ontology.uniprot_client.requests.get", return_value=mock_response):
            result = query_uniprot("EGFR")
        self.assertIsNotNone(result)
        self.assertEqual(result["accession"], "P00533")

    def test_uniprot_query_failure_degrades(self):
        """UniProt 失败时降级返回 None。"""
        from app.ontology.uniprot_client import query_uniprot
        with patch(
            "app.ontology.uniprot_client.requests.get",
            side_effect=Exception("timeout"),
        ):
            with patch("app.ontology.uniprot_client.time.sleep"):
                result = query_uniprot("EGFR")
        self.assertIsNone(result)


class TestCacheBehavior(unittest.TestCase):
    """测试 4：缓存命中时不调 API。"""

    def setUp(self):
        from app.ontology._cache import cache_clear
        cache_clear("hgnc")

    def test_cache_hit_skips_api(self):
        """缓存命中时不发起 API 请求。"""
        from app.ontology._cache import cache_set
        from app.ontology.hgnc_client import query_hgnc

        # 预写缓存
        cache_set("hgnc", "EGFR", {
            "hgnc_id": "HGNC:3236",
            "symbol": "EGFR",
            "name": "cached",
            "uniprot_id": "P00533",
            "entrez_id": "1956",
            "ensembl_gene_id": "",
            "gene_group": [],
        })

        # mock requests.get，若被调用则测试失败
        with patch(
            "app.ontology.hgnc_client.requests.get",
            side_effect=AssertionError("API 不应被调用，缓存应命中"),
        ):
            result = query_hgnc("EGFR")
        self.assertIsNotNone(result)
        self.assertEqual(result["hgnc_id"], "HGNC:3236")
        self.assertEqual(result["name"], "cached")  # 来自缓存


class TestOntologyAgent(unittest.TestCase):
    """Ontology Agent 主逻辑测试。"""

    def setUp(self):
        from app.ontology._cache import cache_clear
        cache_clear()  # 清所有缓存

    def test_agent_annotate_with_mocked_apis(self):
        """Ontology Agent 端到端标注（mock API）。"""
        from app.ontology.ontology_agent import OntologyAgent

        agent = OntologyAgent()
        # mock 所有 API
        with patch("app.ontology.ontology_agent.query_hgnc", return_value={
            "hgnc_id": "HGNC:3236", "symbol": "EGFR",
            "name": "epidermal growth factor receptor",
            "uniprot_id": "P00533",
        }), patch("app.ontology.ontology_agent.query_uniprot", return_value={
            "accession": "P00533", "protein_name": "EGFR",
            "gene": "EGFR", "organism": "Homo sapiens",
            "length": 1210, "function": "",
        }), patch("app.ontology.ontology_agent.query_go", return_value=[
            {"go_id": "GO:0007179", "aspect": "biological_process",
             "term_name": "ERBB2 signaling pathway", "evidence": "EXP"},
        ]):
            result = agent.annotate("EGF activates EGFR")

        self.assertIn("entities", result)
        self.assertGreater(len(result["entities"]), 0)
        # 至少 EGFR 被 verified
        egfr_entities = [e for e in result["entities"] if e["name"] == "EGFR"]
        self.assertGreater(len(egfr_entities), 0)
        self.assertTrue(egfr_entities[0]["verified"])
        self.assertEqual(egfr_entities[0]["hgnc_id"], "HGNC:3236")
        self.assertEqual(egfr_entities[0]["uniprot_id"], "P00533")
        # 通路识别
        self.assertEqual(result["pathway_class"], "EGFR_RTK")

    def test_agent_degrades_on_all_api_failure(self):
        """测试 3：所有 API 失败时降级 verified=false，不抛异常。"""
        from app.ontology.ontology_agent import OntologyAgent

        agent = OntologyAgent()
        with patch("app.ontology.ontology_agent.query_hgnc", return_value=None), \
             patch("app.ontology.ontology_agent.query_uniprot", return_value=None), \
             patch("app.ontology.ontology_agent.query_chebi", return_value=None), \
             patch("app.ontology.ontology_agent.query_go", return_value=[]):
            result = agent.annotate("EGF activates EGFR")

        self.assertIn("entities", result)
        # 所有实体 verified=False
        for ent in result["entities"]:
            self.assertFalse(ent["verified"])

    def test_agent_empty_input(self):
        """空输入返回空实体列表。"""
        from app.ontology.ontology_agent import OntologyAgent
        agent = OntologyAgent()
        result = agent.annotate("")
        self.assertEqual(result["entities"], [])

    def test_agent_merges_v3_entities(self):
        """合并 v3 NER 实体与本模块抽取的实体。"""
        from app.ontology.ontology_agent import OntologyAgent
        agent = OntologyAgent()
        v3_entities = [
            {"entity_id": "1", "name": "EGFR", "type": "protein",
             "aliases": [], "canonical_id": ""},
            {"entity_id": "2", "name": "CustomProtein", "type": "protein",
             "aliases": [], "canonical_id": ""},
        ]
        with patch("app.ontology.ontology_agent.query_hgnc", return_value=None), \
             patch("app.ontology.ontology_agent.query_uniprot", return_value=None), \
             patch("app.ontology.ontology_agent.query_go", return_value=[]):
            result = agent.annotate("EGFR", v3_entities=v3_entities)
        names = [e["name"] for e in result["entities"]]
        self.assertIn("EGFR", names)
        self.assertIn("CustomProtein", names)
        # v3 实体 source 标记
        egfr = next(e for e in result["entities"] if e["name"] == "EGFR")
        self.assertEqual(egfr["source"], "v3_ner")


class TestOntologyHookNode(unittest.TestCase):
    """测试 5：Feature Flag false 时 Ontology Agent 不执行。"""

    def setUp(self):
        from app.ontology._cache import cache_clear
        cache_clear()

    def test_hook_disabled_returns_empty(self):
        """Feature Flag false 时 hook 返回空 dict，不执行 Ontology Agent。"""
        from app.ontology.ontology_agent import ontology_hook_node
        with patch("app.ontology.ontology_agent.settings") as mock_settings:
            mock_settings.V4_ONTOLOGY_AGENT_ENABLED = False
            # mock OntologyAgent，若被调用则测试失败
            with patch(
                "app.ontology.ontology_agent._get_ontology_agent",
                side_effect=AssertionError("Ontology Agent 不应被调用"),
            ):
                result = ontology_hook_node({"user_input": "EGF activates EGFR"})
        self.assertEqual(result, {})

    def test_hook_enabled_returns_v4_entities(self):
        """Feature Flag true 时 hook 返回 v4_ontology_entities。"""
        from app.ontology.ontology_agent import ontology_hook_node
        with patch("app.ontology.ontology_agent.settings") as mock_settings:
            mock_settings.V4_ONTOLOGY_AGENT_ENABLED = True
            with patch("app.ontology.ontology_agent.query_hgnc", return_value={
                "hgnc_id": "HGNC:3236", "symbol": "EGFR",
                "name": "epidermal growth factor receptor",
                "uniprot_id": "P00533",
            }), patch("app.ontology.ontology_agent.query_uniprot", return_value={
                "accession": "P00533", "protein_name": "EGFR",
                "gene": "EGFR", "organism": "Homo sapiens",
                "length": 1210, "function": "",
            }), patch("app.ontology.ontology_agent.query_go", return_value=[]):
                result = ontology_hook_node({"user_input": "EGF activates EGFR"})
        self.assertIn("v4_ontology_entities", result)
        entities = result["v4_ontology_entities"]["entities"]
        self.assertGreater(len(entities), 0)

    def test_hook_does_not_modify_v3_fields(self):
        """hook 返回的 dict 不含任何 v3 字段。"""
        from app.ontology.ontology_agent import ontology_hook_node
        with patch("app.ontology.ontology_agent.settings") as mock_settings:
            mock_settings.V4_ONTOLOGY_AGENT_ENABLED = True
            with patch("app.ontology.ontology_agent.query_hgnc", return_value=None), \
                 patch("app.ontology.ontology_agent.query_uniprot", return_value=None), \
                 patch("app.ontology.ontology_agent.query_go", return_value=[]):
                result = ontology_hook_node({"user_input": "EGF activates EGFR"})
        # v3 字段黑名单（绝对不能出现在返回中）
        v3_field_blacklist = [
            "network_json", "parameters", "entities", "mechanism",
            "execution_plan", "next_worker", "rag_retrieved_params",
            "user_input", "messages", "knowledge_graph",
        ]
        for field in v3_field_blacklist:
            self.assertNotIn(field, result, f"hook 不应修改 v3 字段 {field}")

    def test_hook_failure_does_not_raise(self):
        """hook 内部异常时不抛出，降级返回空 dict。"""
        from app.ontology.ontology_agent import ontology_hook_node
        with patch("app.ontology.ontology_agent.settings") as mock_settings:
            mock_settings.V4_ONTOLOGY_AGENT_ENABLED = True
            with patch(
                "app.ontology.ontology_agent._get_ontology_agent",
                side_effect=RuntimeError("simulated failure"),
            ):
                # 不应抛异常
                result = ontology_hook_node({"user_input": "EGF activates EGFR"})
        self.assertEqual(result, {})


class TestStateField(unittest.TestCase):
    """验证 state.py 新增字段。"""

    def test_v4_ontology_entities_field_exists(self):
        """BioDynamicsState 包含 v4_ontology_entities 字段。"""
        from app.state import BioDynamicsState
        # TypedDict 在 total=False 时字段都是可选的，检查 __annotations__
        self.assertIn("v4_ontology_entities", BioDynamicsState.__annotations__)


class TestConfigFlag(unittest.TestCase):
    """验证 config.py Feature Flag。"""

    def test_flag_default_false(self):
        """V4_ONTOLOGY_AGENT_ENABLED 默认 false。"""
        from app.config import settings
        # 默认应为 false（.env 中设为 false）
        self.assertFalse(settings.V4_ONTOLOGY_AGENT_ENABLED)

    def test_pathway_graph_flag_default_false(self):
        """V4_PATHWAY_GRAPH_ENABLED 默认 false。"""
        from app.config import settings
        self.assertFalse(settings.V4_PATHWAY_GRAPH_ENABLED)


class TestGraphHookIntegration(unittest.TestCase):
    """验证 graph_v3.py 已集成 ontology_hook 节点（不实际编译图）。"""

    def test_ontology_hook_node_imported(self):
        """graph_v3 模块成功导入 ontology_hook_node。"""
        import app.graph_v3 as graph_v3
        self.assertTrue(hasattr(graph_v3, "ontology_hook_node"))

    def test_build_workflow_contains_ontology_hook(self):
        """build_workflow_v3 注册了 ontology_hook 节点。

        通过检查源码确认（避免实际编译图触发外部依赖）。
        """
        import inspect
        import app.graph_v3 as graph_v3
        source = inspect.getsource(graph_v3.build_workflow_v3)
        self.assertIn("ontology_hook", source)
        self.assertIn('workflow.add_node("ontology_hook"', source)
        self.assertIn('workflow.add_edge(START, "ontology_hook")', source)
        self.assertIn('workflow.add_edge("ontology_hook", "pre_router")', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
