"""RAG → ODE 参数传递链路集成测试。

验证三个端到端场景的关键链路：
1. 奥希替尼抑制 EGFR（单药物，RAG 参数注入 ODE）
2. 他莫昔芬抑制 ERα（文献检索触发，变量名 ASCII 化）
3. Auto Fast 模式（TGF-β/SMAD/CD8，plan 含 worker_rag）

不调用真实 LLM / ChromaDB，用 mock 注入预设参数，验证节点间数据流正确性。
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# 场景 1：奥希替尼抑制 EGFR（单药物，RAG 参数注入 ODE）
# =============================================================================
class TestScenario1OsimertinibEgfr:
    """验证 RAG 提取的 IC50 能正确注入 ODE 的 Kd 参数。"""

    def test_rag_to_ode_parameter_flow(self):
        """场景 1 验证点：RAG 提取 IC50=12nM → ODE 中 Kd=12.0（非 10.0 占位符）。"""
        from app.nodes_v2 import n6_ode_generator

        # 模拟 n5_parameter_rag 产出的 parameters
        # 奥希替尼抑制 EGFR，RAG 提取 IC50=12nM
        state = {
            "knowledge_graph": {
                "nodes": [
                    {"id": "Osimertinib", "name": "Osimertinib", "type": "Drug"},
                    {"id": "EGFR", "name": "EGFR", "type": "Protein"},
                ],
                "edges": [
                    {
                        "source": "Osimertinib",
                        "target": "EGFR",
                        "interaction": "inhibition",
                    }
                ],
            },
            "parameters": {
                "Osimertinib->EGFR": {
                    "edge_key": "Osimertinib->EGFR",
                    "param_name": "ic50",
                    "value": 12.0,
                    "unit": "nM",
                    "source": "PMID:12345",
                    "is_fallback": False,
                }
            },
            "mechanism": {"template": "Simple_Inhibition"},
            "pkpd_profile": {},  # 不激活 PKPD 路径
            "drug_regimen": [],
        }
        result = n6_ode_generator(state)
        ode_code = result["ode_model"]["code"]

        # 验证点 1：ODE 中 Kd 应为 12.0（RAG 提取值），非 10.0（占位符）
        assert "12.0" in ode_code, f"ODE 代码缺少 RAG 提取的 Kd=12.0: {ode_code}"
        # 验证点 2：模板为 Simple_Inhibition
        assert result["ode_model"]["template"] == "Simple_Inhibition"
        # 验证点 3：ODE 变量名为英文 ASCII（无中文/占位符）
        assert "Osimertinib" in ode_code or "EGFR" in ode_code
        assert "e1" not in ode_code.split("=")[0]  # 变量名区域无 e1 占位符

    def test_dose_ec50_ratio_in_pkpd_mode(self):
        """场景 1 验证点：PKPD 模式下 DOSE 与 EC50 比值在 5-20 倍区间。"""
        from app.nodes_v2 import n6_ode_generator

        # 模拟 pkpd_profile 含 EC50=12nM
        state = {
            "knowledge_graph": {
                "nodes": [
                    {"id": "Osimertinib", "name": "Osimertinib", "type": "Drug"},
                    {"id": "EGFR", "name": "EGFR", "type": "Protein"},
                ],
                "edges": [
                    {
                        "source": "Osimertinib",
                        "target": "EGFR",
                        "interaction": "inhibition",
                    }
                ],
            },
            "parameters": {},
            "mechanism": {"template": "Simple_Inhibition"},
            "pkpd_profile": {
                "drug_name": "Osimertinib",
                "drug_target": "EGFR",
                "compartment": "1-compartment",
                "pk_params": {"k10": 0.1, "k12": 0.0, "k21": 0.0},
                "pd_params": {"EC50": 12.0, "Emax": 1.0, "gamma": 1.0},
            },
            "drug_regimen": [{"dose": 50.0}],  # 初始 dose=50，低于 10*12=120
        }
        result = n6_ode_generator(state)
        ode_code = result["ode_model"]["code"]

        # 验证：dose 应被提升到 10*EC50=120（P1-1 修复）
        assert "120.0" in ode_code, f"DOSE 未提升到 10×EC50=120: {ode_code}"
        # EC50 应为 12.0
        assert "12.0" in ode_code


# =============================================================================
# 场景 2：他莫昔芬抑制 ERα（文献检索触发，变量名 ASCII 化）
# =============================================================================
class TestScenario2TamoxifenErAlpha:
    """验证中文实体名能正确映射为 ASCII ODE 变量名。"""

    def test_chinese_entity_name_mapped_to_ascii(self):
        """场景 2 验证点：ERα → ER_alpha（ASCII 标识符），变量名无中文。"""
        from app.nodes_v2 import n6_ode_generator

        state = {
            "knowledge_graph": {
                "nodes": [
                    {"id": "Tamoxifen", "name": "Tamoxifen", "type": "Drug"},
                    {"id": "ERα", "name": "ERα", "type": "Protein"},
                ],
                "edges": [
                    {
                        "source": "Tamoxifen",
                        "target": "ERα",
                        "interaction": "inhibition",
                    }
                ],
            },
            "parameters": {
                "Tamoxifen->ERα": {
                    "edge_key": "Tamoxifen->ERα",
                    "param_name": "ic50",
                    "value": 10.0,
                    "unit": "nM",
                    "source": "PMID:67890",
                    "is_fallback": False,
                }
            },
            "mechanism": {"template": "Simple_Inhibition"},
            "pkpd_profile": {},
            "drug_regimen": [],
        }
        result = n6_ode_generator(state)
        ode_code = result["ode_model"]["code"]

        # 验证点 1：SPECIES_NAMES 中的变量名不含中文（ERα → ER_alpha 或类似 ASCII）
        # 只检查 SPECIES_NAMES 赋值行和变量引用，不检查 docstring/注释
        lines = ode_code.split("\n")
        for line in lines:
            stripped = line.strip()
            # 跳过 docstring（三引号包裹）和注释行
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith('"'):
                continue
            # 只检查变量赋值行（含 = 号）和 SPECIES_NAMES 定义
            if "=" in stripped and not stripped.startswith("def"):
                # 检查赋值行右侧是否含非 ASCII（变量名区域）
                # 允许字符串字面量中含中文（如 title），只检查标识符
                non_ascii = [c for c in stripped if ord(c) > 127]
                if non_ascii:
                    # 如果是字符串字面量内的中文，跳过
                    if '"' in stripped or "'" in stripped:
                        continue
                    assert not non_ascii, (
                        f"ODE 代码行含非 ASCII 字符（变量名区域）: {stripped} -> {non_ascii}"
                    )

    def test_rule_based_rag_check_triggers_for_literature_keywords(self):
        """场景 2 验证点：含'文献'关键词时触发 _rule_based_rag_check。"""
        from app.graph_v3 import _rule_based_rag_check

        # 含文献关键词
        assert _rule_based_rag_check("他莫昔芬对ERα的抑制机制文献研究")
        assert _rule_based_rag_check("tamoxifen ERα literature study")
        # 不含文献关键词
        assert not _rule_based_rag_check("他莫昔芬抑制ERα")


# =============================================================================
# 场景 3：Auto Fast 模式（TGF-β/SMAD/CD8）
# =============================================================================
class TestScenario3AutoFastMode:
    """验证 Auto Fast 模式的 plan 和参数占位。"""

    def test_auto_fast_plan_contains_worker_rag(self):
        """场景 3 验证点：Auto Fast plan 包含 worker_rag，parameters 非空。"""
        from app.graph_v3 import pre_router

        state = {
            "mode": "auto_fast",
            "user_input": "TGF-β如何通过SMAD信号通路抑制CD8+ T细胞",
            "manual_modules": [],
        }
        result = pre_router(state)
        plan = result["execution_plan"]

        # 验证 plan 含 worker_rag
        assert "worker_rag" in plan, f"Auto Fast plan 缺少 worker_rag: {plan}"
        # 验证顺序：mechanism -> rag -> ode
        assert plan.index("worker_mechanism") < plan.index("worker_rag")
        assert plan.index("worker_rag") < plan.index("worker_ode")

    def test_auto_fast_worker_rag_produces_parameters(self):
        """场景 3 验证点：Fast 模式 worker_rag 产出非空 parameters。"""
        import asyncio
        from app.graph_v3 import worker_rag

        state = {
            "mode": "auto_fast",
            "user_input": "TGF-β 抑制 CD8+ T cells",
            "knowledge_graph": {
                "nodes": [
                    {"id": "TGFb", "name": "TGFb", "type": "Protein"},
                    {"id": "CD8", "name": "CD8", "type": "Cell"},
                ],
                "edges": [
                    {
                        "source": "TGFb",
                        "target": "CD8",
                        "interaction": "inhibition",
                    }
                ],
            },
        }
        result = asyncio.run(worker_rag(state))

        # 验证 parameters 非空
        # result 可能被 compress_worker_output 包装，检查关键字段
        parameters = result.get("parameters", {})
        assert len(parameters) > 0, "Fast 模式 worker_rag 未产出 parameters"
        # 验证含 edge_key
        edge_key = "TGFb->CD8"
        assert edge_key in parameters, f"parameters 缺少 edge_key {edge_key}: {parameters}"
        # 验证含 rag_insights 占位
        rag_insights = result.get("rag_insights", {})
        assert rag_insights.get("rewritten_query") is not None

    def test_auto_fast_ode_uses_simple_inhibition_template(self):
        """场景 3 验证点：单 inhibition 边强制 Simple_Inhibition，无幻觉蛋白。"""
        from app.nodes_v2 import n6_ode_generator

        # 模拟 LLM 误选 Cascade_Inhibition，但 KG 只有单条 inhibition 边
        state = {
            "knowledge_graph": {
                "nodes": [
                    {"id": "TGFb", "name": "TGFb", "type": "Protein"},
                    {"id": "CD8", "name": "CD8", "type": "Cell"},
                ],
                "edges": [
                    {
                        "source": "TGFb",
                        "target": "CD8",
                        "interaction": "inhibition",
                    }
                ],
            },
            "parameters": {
                "TGFb->CD8": {
                    "edge_key": "TGFb->CD8",
                    "param_name": "kd",
                    "value": 10.0,
                    "unit": "nM",
                    "source": "FAST_MODE_ESTIMATED",
                    "is_fallback": True,
                }
            },
            "mechanism": {"template": "Cascade_Inhibition"},  # LLM 误选
            "pkpd_profile": {},
            "drug_regimen": [],
        }
        result = n6_ode_generator(state)

        # 验证 P2-5 修复：单 inhibition 边强制 Simple_Inhibition
        assert result["ode_model"]["template"] == "Simple_Inhibition", (
            f"P2-5 修复未生效，模板应为 Simple_Inhibition 而非 Cascade_Inhibition"
        )


# =============================================================================
# 场景 4：network_json schema 统一（v1/v3 一致性）
# =============================================================================
class TestScenario4NetworkJsonSchemaUnification:
    """验证 v1/v3 出口的 network_json schema 一致。"""

    def test_v1_normalize_network_json_id_priority(self):
        """场景 4 验证点：v1 出口 id 优先取 name（与 v3 一致）。"""
        from app.nodes import _normalize_network_json

        network_json = {
            "nodes": [
                {"id": "e1", "name": "EGFR", "type": "Protein"},
                {"id": "e2", "name": "Drug", "type": "Drug"},
            ],
            "edges": [
                {"source": "e2", "target": "e1", "interaction": "inhibition"}
            ],
        }
        result = _normalize_network_json(network_json)
        # v1 出口 id 应为 name（与 v3 worker_mechanism 一致）
        assert result["nodes"][0]["id"] == "EGFR"
        assert result["nodes"][1]["id"] == "Drug"
        # edge 引用同步更新
        assert result["edges"][0]["source"] == "Drug"
        assert result["edges"][0]["target"] == "EGFR"

    def test_v3_worker_mechanism_uses_normalize_function(self):
        """场景 4 验证点：v3 worker_mechanism 调用 _normalize_network_json。"""
        # 通过检查 graph_v3.py 源码确认调用了公共函数
        import inspect
        from app.graph_v3 import worker_mechanism

        source = inspect.getsource(worker_mechanism)
        assert "_normalize_network_json" in source, (
            "v3 worker_mechanism 未调用 _normalize_network_json 公共函数"
        )


# =============================================================================
# 场景 5：n10 证据检索正向过滤
# =============================================================================
class TestScenario5EvidenceRagFiltering:
    """验证 n10 证据检索的正向过滤和 PMID 提取。"""

    def test_evidence_filtered_by_kg_entities(self):
        """场景 5 验证点：仅保留含 KG 实体名的证据。"""
        from app.nodes_v2 import n10_evidence_rag
        from app import nodes_v2
        from unittest.mock import patch, MagicMock

        state = {
            "user_input": "Osimertinib 抑制 EGFR",
            "mechanism": {"pathway": "EGFR inhibition"},
            "knowledge_graph": {
                "nodes": [
                    {"id": "Osimertinib", "name": "Osimertinib", "aliases": []},
                    {"id": "EGFR", "name": "EGFR", "aliases": []},
                ],
                "edges": [],
            },
            "mcp_term_definitions": [],
        }
        mock_rag = MagicMock()
        mock_rag.available = True
        mock_rag.search_evidence.return_value = [
            {"title": "Osimertinib EGFR NSCLC", "context": "EGFR T790M"},
            {"title": "TGF-β CD8 SMAD", "context": "irrelevant"},
        ]
        with patch.object(nodes_v2, "get_rag_collections", return_value=mock_rag):
            result = n10_evidence_rag(state)
        evidence = result["paper_evidence"]
        # EGFR 相关证据保留
        assert any("EGFR" in ev.get("title", "") for ev in evidence)
        # TGF-β 无关证据丢弃
        assert not any("TGF-β" in ev.get("title", "") for ev in evidence)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
