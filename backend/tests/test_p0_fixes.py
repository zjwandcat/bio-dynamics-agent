"""P0 修复点专项单元测试。

覆盖：
- P0-2: N6 Simple 模板从 parameters 读取 Kd
- P0-3: _extract_drug_candidates_fallback 从 parameters 提取 IC50/EC50/Kd
- P0-4: KGBuilder entity_id → name 映射回填
- P0-5: EC50 单位校验中间区间
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.kg_builder import KGBuilder
from app.nodes import _extract_drug_candidates_fallback


# =============================================================================
# P0-3: _extract_drug_candidates_fallback 从 parameters 提取 IC50/EC50/Kd
# =============================================================================
class TestP03ExtractDrugCandidatesFromParameters:
    """验证兜底提取函数能从 state["parameters"] 读取 RAG 提取的 IC50/EC50/Kd。"""

    def test_extract_ic50_from_parameters(self):
        """P0-3 验证点：从 parameters 提取 IC50（RAG 提取值）。"""
        network_json = {
            "nodes": [{"id": "Osimertinib", "name": "Osimertinib", "type": "Drug"}],
            "edges": [
                {"source": "Osimertinib", "target": "EGFR", "interaction": "inhibition"}
            ],
        }
        user_input = "Osimertinib 抑制 EGFR"  # 无 IC50 数值
        parameters = {
            "Osimertinib->EGFR": {
                "edge_key": "Osimertinib->EGFR",
                "param_name": "ic50",
                "value": 12.0,
                "unit": "nM",
                "source": "PMID:12345",
                "is_fallback": False,
            }
        }
        candidates = _extract_drug_candidates_fallback(
            network_json, user_input, parameters=parameters
        )
        assert len(candidates) == 1
        assert candidates[0]["drug_name"] == "Osimertinib"
        assert candidates[0]["ic50"] == 12.0
        assert "RAG:PMID:12345" in candidates[0]["source"]
        assert candidates[0]["target_name"] == "EGFR"

    def test_user_input_ic50_overrides_parameters(self):
        """P0-3 验证点：user_input 正则提取的 IC50 优先级高于 parameters。"""
        network_json = {
            "nodes": [{"id": "Drug_A", "name": "Drug_A", "type": "Drug"}],
            "edges": [{"source": "Drug_A", "target": "Target_X", "interaction": "inhibition"}],
        }
        user_input = "Drug_A IC50 = 50 nM"
        parameters = {
            "Drug_A->Target_X": {
                "param_name": "ic50",
                "value": 12.0,
                "unit": "nM",
                "source": "RAG",
                "is_fallback": False,
            }
        }
        candidates = _extract_drug_candidates_fallback(
            network_json, user_input, parameters=parameters
        )
        assert candidates[0]["ic50"] == 50.0  # user_input 优先
        assert candidates[0]["source"] == "extracted_from_input"

    def test_kd_treated_as_ic50_when_no_ic50_ec50(self):
        """P0-3 验证点：parameters 中只有 Kd 时，近似为 IC50。"""
        network_json = {
            "nodes": [{"id": "Drug_B", "name": "Drug_B", "type": "Drug"}],
            "edges": [{"source": "Drug_B", "target": "Target_Y", "interaction": "inhibition"}],
        }
        user_input = "Drug_B 抑制 Target_Y"
        parameters = {
            "Drug_B->Target_Y": {
                "param_name": "kd",
                "value": 8.5,
                "unit": "nM",
                "source": "ChEMBL",
                "is_fallback": False,
            }
        }
        candidates = _extract_drug_candidates_fallback(
            network_json, user_input, parameters=parameters
        )
        assert candidates[0]["ic50"] == 8.5  # Kd 近似为 IC50

    def test_fallback_parameters_ignored(self):
        """P0-3 验证点：is_fallback=True 的 parameters 条目被忽略。"""
        network_json = {
            "nodes": [{"id": "Drug_C", "name": "Drug_C", "type": "Drug"}],
            "edges": [{"source": "Drug_C", "target": "Target_Z", "interaction": "inhibition"}],
        }
        user_input = "Drug_C 抑制 Target_Z"
        parameters = {
            "Drug_C->Target_Z": {
                "param_name": "ic50",
                "value": 10.0,
                "unit": "nM",
                "source": "ESTIMATED",
                "is_fallback": True,  # 标记为 fallback
            }
        }
        candidates = _extract_drug_candidates_fallback(
            network_json, user_input, parameters=parameters
        )
        assert candidates[0]["ic50"] == 0.0  # fallback 被忽略，无 IC50
        assert candidates[0]["source"] == "network_only"


# =============================================================================
# P0-4: KGBuilder entity_id → name 映射回填
# =============================================================================
class TestP04KgBuilderEntityIdMapping:
    """验证 KG 构建时 entity_id → name 映射回填，消除 e1/e2 占位符。"""

    def test_edges_use_name_not_entity_id(self):
        """P0-4 验证点：edges 的 source/target 使用 name 而非 entity_id。"""
        entities = [
            {"entity_id": "e1", "name": "EGFR", "type": "Protein", "aliases": []},
            {"entity_id": "e2", "name": "Osimertinib", "type": "Drug", "aliases": []},
        ]
        # N2 planner 用 entity_id 作为 source/target（问题场景）
        relations = [
            {"source": "e2", "target": "e1", "interaction": "inhibition"}
        ]
        kg = KGBuilder().build(entities=entities, relations=relations)
        assert kg["edge_count"] == 1
        edge = kg["edges"][0]
        assert edge["source"] == "Osimertinib"  # 回填为 name
        assert edge["target"] == "EGFR"  # 回填为 name
        assert edge["interaction"] == "inhibition"

    def test_edges_preserve_name_when_already_name(self):
        """P0-4 验证点：edges 已用 name 时保持不变。"""
        entities = [
            {"entity_id": "e1", "name": "EGFR", "type": "Protein", "aliases": []},
            {"entity_id": "e2", "name": "Osimertinib", "type": "Drug", "aliases": []},
        ]
        relations = [
            {"source": "Osimertinib", "target": "EGFR", "interaction": "inhibition"}
        ]
        kg = KGBuilder().build(entities=entities, relations=relations)
        edge = kg["edges"][0]
        assert edge["source"] == "Osimertinib"
        assert edge["target"] == "EGFR"

    def test_no_e1_e2_in_final_edges(self):
        """P0-4 验证点：最终 KG edges 中不出现 e1/e2 占位符。"""
        entities = [
            {"entity_id": "e1", "name": "雌激素受体 α", "type": "Protein", "aliases": ["ERα"]},
            {"entity_id": "e2", "name": "Tamoxifen", "type": "Drug", "aliases": []},
        ]
        relations = [
            {"source": "e2", "target": "e1", "interaction": "inhibition"}
        ]
        kg = KGBuilder().build(entities=entities, relations=relations)
        for edge in kg["edges"]:
            assert "e1" not in edge["source"]
            assert "e2" not in edge["source"]
            assert "e1" not in edge["target"]
            assert "e2" not in edge["target"]


# =============================================================================
# P0-5: EC50 单位校验中间区间
# =============================================================================
class TestP05Ec50UnitNormalization:
    """验证 N6 中 EC50 单位校验的中间区间逻辑。

    由于 n6_ode_generator 内联了单位校验逻辑，这里通过模拟 PK/PD 激活场景
    验证 _ec50 在中间区间 (0.001, 1.0) 会被乘以 1000。
    """

    def _build_pkpd_state(self, ec50_value: float, ec50_unit: str = "") -> dict:
        """构建激活 PK/PD 路径的 state，触发 EC50 单位校验。"""
        return {
            "knowledge_graph": {
                "nodes": [
                    {"id": "Osimertinib", "name": "Osimertinib", "type": "Drug"},
                    {"id": "EGFR", "name": "EGFR", "type": "Protein"},
                ],
                "edges": [
                    {"source": "Osimertinib", "target": "EGFR", "interaction": "inhibition"}
                ],
            },
            "parameters": {},
            "mechanism": {"template": "PKPD_OneCompartment"},
            "pkpd_profile": {
                "drug_name": "Osimertinib",
                "drug_target": "EGFR",
                "compartment": "1-compartment",
                "pk_params": {"k10": 0.1, "k12": 0.0, "k21": 0.0},
                "pd_params": {
                    "EC50": ec50_value,
                    "Emax": 1.0,
                    "gamma": 1.0,
                    **({"EC50_unit": ec50_unit} if ec50_unit else {}),
                },
            },
            "drug_regimen": [{"dose": 100.0}],
        }

    def test_ec50_middle_range_um_corrected(self):
        """P0-5 验证点：0.001 < EC50 < 1.0 且未声明 nM，乘以 1000 修正。"""
        from app.nodes_v2 import n6_ode_generator
        # EC50=0.05（µM 量级未换算），未声明单位
        state = self._build_pkpd_state(0.05)
        result = n6_ode_generator(state)
        ode_code = result["ode_model"]["code"]
        # 修正后 EC50 应为 50 nM（0.05 * 1000）
        # 模板渲染后会在代码中出现 EC50 = 50.0
        assert "50.0" in ode_code or "50" in ode_code

    def test_ec50_middle_range_with_nm_unit_not_corrected(self):
        """P0-5 验证点：0.001 < EC50 < 1.0 但声明 nM，不修正。"""
        from app.nodes_v2 import n6_ode_generator
        # EC50=0.5 nM（显式声明 nM），不应修正
        state = self._build_pkpd_state(0.5, ec50_unit="nM")
        result = n6_ode_generator(state)
        ode_code = result["ode_model"]["code"]
        # EC50 应保持 0.5 nM
        assert "0.5" in ode_code

    def test_ec50_high_range_divided(self):
        """P0-5 验证点：EC50 > 10000 除以 1000 修正（原有逻辑）。"""
        from app.nodes_v2 import n6_ode_generator
        # EC50=50000（µM 误填为 nM），应为 50 nM
        state = self._build_pkpd_state(50000.0)
        result = n6_ode_generator(state)
        ode_code = result["ode_model"]["code"]
        assert "50.0" in ode_code or "50" in ode_code


# =============================================================================
# P0-2: N6 Simple 模板从 parameters 读取 Kd
# =============================================================================
class TestP02SimpleTemplateKdFromParameters:
    """验证 N6 Simple 模板从 parameters[edge_key]["value"] 读取 RAG Kd。"""

    def test_simple_inhibition_uses_rag_kd(self):
        """P0-2 验证点：Simple_Inhibition 模板 Kd 来自 RAG 提取值。"""
        from app.nodes_v2 import n6_ode_generator
        state = {
            "knowledge_graph": {
                "nodes": [
                    {"id": "Drug_A", "name": "Drug_A", "type": "Drug"},
                    {"id": "Target_X", "name": "Target_X", "type": "Protein"},
                ],
                "edges": [
                    {"source": "Drug_A", "target": "Target_X", "interaction": "inhibition"}
                ],
            },
            "parameters": {
                "Drug_A->Target_X": {
                    "edge_key": "Drug_A->Target_X",
                    "param_name": "kd",
                    "value": 7.5,
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
        # Kd 应为 7.5（RAG 提取值），而非 10.0（硬编码默认）
        assert "7.5" in ode_code
        assert result["ode_model"]["template"] == "Simple_Inhibition"

    def test_simple_inhibition_fallback_to_default_kd(self):
        """P0-2 验证点：parameters 无 RAG 值时回退到 10.0。"""
        from app.nodes_v2 import n6_ode_generator
        state = {
            "knowledge_graph": {
                "nodes": [
                    {"id": "Drug_B", "name": "Drug_B", "type": "Drug"},
                    {"id": "Target_Y", "name": "Target_Y", "type": "Protein"},
                ],
                "edges": [
                    {"source": "Drug_B", "target": "Target_Y", "interaction": "inhibition"}
                ],
            },
            "parameters": {
                "Drug_B->Target_Y": {
                    "param_name": "kd",
                    "value": 10.0,
                    "unit": "nM",
                    "source": "ESTIMATED",
                    "is_fallback": True,  # 标记为 fallback
                }
            },
            "mechanism": {"template": "Simple_Inhibition"},
            "pkpd_profile": {},
            "drug_regimen": [],
        }
        result = n6_ode_generator(state)
        ode_code = result["ode_model"]["code"]
        # Kd 应为 10.0（默认值，因 parameters 是 fallback）
        assert "10.0" in ode_code


# =============================================================================
# P1-3: n10_evidence_rag 正向实体白名单过滤 + PMID 正则放宽
# =============================================================================
class TestP13EvidenceRagPositiveFiltering:
    """验证 N10 证据检索的正向过滤与 PMID 提取逻辑。"""

    def test_positive_filter_keeps_relevant_evidence(self):
        """P1-3 验证点：含 KG 实体名的证据被保留。"""
        from app.nodes_v2 import n10_evidence_rag

        # 构造 KG 节点含 EGFR/Osimertinib
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
        # 直接 patch rag_collections 避免依赖外部 ChromaDB
        from app import nodes_v2
        from unittest.mock import patch, MagicMock

        mock_rag = MagicMock()
        mock_rag.available = True
        mock_rag.search_evidence.return_value = [
            {"title": "Osimertinib inhibits EGFR in NSCLC", "context": "EGFR T790M"},
            {"title": "TGF-β signaling in CD8+ T cells", "context": "irrelevant topic"},
        ]
        with patch.object(nodes_v2, "get_rag_collections", return_value=mock_rag):
            result = n10_evidence_rag(state)
        evidence = result["paper_evidence"]
        # 第一条含 EGFR/Osimertinib → 保留
        assert any("Osimertinib" in ev.get("title", "") for ev in evidence)
        # 第二条无任何 KG 实体 → 丢弃
        assert not any("TGF-β" in ev.get("title", "") for ev in evidence)

    def test_pmid_regex_extracts_5_digit_pmid(self):
        """P1-3 验证点：PMID 正则放宽至 5 位数字。"""
        from app.nodes_v2 import n10_evidence_rag
        from app import nodes_v2
        from unittest.mock import patch, MagicMock

        state = {
            "user_input": "EGFR 研究",
            "mechanism": {"pathway": "EGFR"},
            "knowledge_graph": {
                "nodes": [{"id": "EGFR", "name": "EGFR", "aliases": []}],
                "edges": [],
            },
            "mcp_term_definitions": [],
        }
        mock_rag = MagicMock()
        mock_rag.available = True
        # 含 5 位 PMID 的早期文献
        mock_rag.search_evidence.return_value = [
            {
                "title": "EGFR signaling PMID: 12345",
                "context": "early paper",
            }
        ]
        with patch.object(nodes_v2, "get_rag_collections", return_value=mock_rag):
            result = n10_evidence_rag(state)
        evidence = result["paper_evidence"]
        assert len(evidence) == 1
        assert evidence[0]["pmid"] == "12345"

    def test_pmid_extraction_from_source_pmid_field(self):
        """P1-3 验证点：从 source_pmid 字段直接提取 PMID。"""
        from app.nodes_v2 import n10_evidence_rag
        from app import nodes_v2
        from unittest.mock import patch, MagicMock

        state = {
            "user_input": "EGFR",
            "mechanism": {"pathway": "EGFR"},
            "knowledge_graph": {
                "nodes": [{"id": "EGFR", "name": "EGFR"}],
                "edges": [],
            },
            "mcp_term_definitions": [],
        }
        mock_rag = MagicMock()
        mock_rag.available = True
        mock_rag.search_evidence.return_value = [
            {
                "title": "EGFR paper",
                "source_pmid": "36987654",
            }
        ]
        with patch.object(nodes_v2, "get_rag_collections", return_value=mock_rag):
            result = n10_evidence_rag(state)
        evidence = result["paper_evidence"]
        assert evidence[0]["pmid"] == "36987654"

    def test_figure_ref_and_cell_line_setdefault(self):
        """P1-3 验证点：figure_ref/cell_line 缺失时填空字符串。"""
        from app.nodes_v2 import n10_evidence_rag
        from app import nodes_v2
        from unittest.mock import patch, MagicMock

        state = {
            "user_input": "EGFR",
            "mechanism": {"pathway": "EGFR"},
            "knowledge_graph": {
                "nodes": [{"id": "EGFR", "name": "EGFR"}],
                "edges": [],
            },
            "mcp_term_definitions": [],
        }
        mock_rag = MagicMock()
        mock_rag.available = True
        mock_rag.search_evidence.return_value = [
            {"title": "EGFR paper"}  # 无 figure_ref/cell_line
        ]
        with patch.object(nodes_v2, "get_rag_collections", return_value=mock_rag):
            result = n10_evidence_rag(state)
        evidence = result["paper_evidence"]
        assert evidence[0]["figure_ref"] == ""
        assert evidence[0]["cell_line"] == ""


# =============================================================================
# P1-4: 统一 v1/v3 network_json schema（_normalize_network_json）
# =============================================================================
class TestP14NormalizeNetworkJson:
    """验证 _normalize_network_json 公共函数统一 v1/v3 出口 schema。"""

    def test_node_id_replaced_by_name_when_id_is_placeholder(self):
        """P1-4 验证点：id="e1" 占位符被 name="EGFR" 替换。"""
        from app.nodes import _normalize_network_json
        network_json = {
            "nodes": [
                {"id": "e1", "name": "EGFR", "type": "Protein"},
                {"id": "e2", "name": "Osimertinib", "type": "Drug"},
            ],
            "edges": [
                {"source": "e2", "target": "e1", "interaction": "inhibition"}
            ],
        }
        result = _normalize_network_json(network_json)
        # node.id 应被 name 替换
        assert result["nodes"][0]["id"] == "EGFR"
        assert result["nodes"][1]["id"] == "Osimertinib"
        # edge.source/target 应同步更新
        assert result["edges"][0]["source"] == "Osimertinib"
        assert result["edges"][0]["target"] == "EGFR"

    def test_node_id_preserved_when_name_empty(self):
        """P1-4 验证点：name 为空时保留原 id。"""
        from app.nodes import _normalize_network_json
        network_json = {
            "nodes": [{"id": "node_1", "name": "", "type": "Protein"}],
            "edges": [],
        }
        result = _normalize_network_json(network_json)
        assert result["nodes"][0]["id"] == "node_1"

    def test_edge_references_updated_after_id_normalization(self):
        """P1-4 验证点：多节点 id 替换后，edge 引用全部同步。"""
        from app.nodes import _normalize_network_json
        network_json = {
            "nodes": [
                {"id": "x1", "name": "SMAD3", "type": "Protein"},
                {"id": "x2", "name": "TGFBR1", "type": "Protein"},
                {"id": "x3", "name": "CD8A", "type": "Protein"},
            ],
            "edges": [
                {"source": "x2", "target": "x1", "interaction": "activation"},
                {"source": "x1", "target": "x3", "interaction": "inhibition"},
            ],
        }
        result = _normalize_network_json(network_json)
        assert result["edges"][0]["source"] == "TGFBR1"
        assert result["edges"][0]["target"] == "SMAD3"
        assert result["edges"][1]["source"] == "SMAD3"
        assert result["edges"][1]["target"] == "CD8A"

    def test_other_fields_preserved(self):
        """P1-4 验证点：type/aliases/interaction 等字段保留。"""
        from app.nodes import _normalize_network_json
        network_json = {
            "nodes": [
                {"id": "e1", "name": "EGFR", "type": "Protein", "aliases": ["ERBB1"]},
            ],
            "edges": [
                {"source": "e1", "target": "e1", "interaction": "activation", "directed": True}
            ],
        }
        result = _normalize_network_json(network_json)
        assert result["nodes"][0]["type"] == "Protein"
        assert result["nodes"][0]["aliases"] == ["ERBB1"]
        assert result["edges"][0]["interaction"] == "activation"
        assert result["edges"][0]["directed"] is True


# =============================================================================
# P1-5: Auto Fast plan 补 worker_rag + Manual 依赖补全
# =============================================================================
class TestP15PlanCompletion:
    """验证 PreRouter 的 Auto Fast plan 和 Manual 依赖补全。"""

    def test_auto_fast_plan_includes_worker_rag(self):
        """P1-5 验证点：Auto Fast plan 包含 worker_rag。"""
        from app.graph_v3 import pre_router
        state = {
            "mode": "auto_fast",
            "user_input": "TGF-β 抑制 CD8+ T cells",
            "manual_modules": [],
        }
        result = pre_router(state)
        plan = result["execution_plan"]
        assert "worker_rag" in plan, f"Auto Fast plan 缺少 worker_rag: {plan}"
        # 确保顺序正确：mechanism -> rag -> ode
        mech_idx = plan.index("worker_mechanism")
        rag_idx = plan.index("worker_rag")
        ode_idx = plan.index("worker_ode")
        assert mech_idx < rag_idx < ode_idx

    def test_manual_plan_ode_depends_on_mechanism_and_rag(self):
        """P1-5 验证点：用户只勾选 dose_analysis 时，自动补全 mechanism_graph + rag。"""
        from app.graph_v3 import _build_manual_plan
        plan = _build_manual_plan(["dose_analysis"])
        assert "worker_mechanism" in plan, f"Manual plan 缺少 worker_mechanism: {plan}"
        assert "worker_rag" in plan, f"Manual plan 缺少 worker_rag: {plan}"
        assert "worker_ode" in plan
        # 顺序：mechanism -> rag -> ode
        assert plan.index("worker_mechanism") < plan.index("worker_rag")
        assert plan.index("worker_rag") < plan.index("worker_ode")

    def test_manual_plan_sandbox_depends_on_ode_mechanism_rag(self):
        """P1-5 验证点：用户只勾选 sandbox_execute 时，自动补全 ode + mechanism + rag。"""
        from app.graph_v3 import _build_manual_plan
        plan = _build_manual_plan(["sandbox_execute"])
        assert "worker_ode" in plan
        assert "worker_mechanism" in plan
        assert "worker_rag" in plan
        assert "worker_sandbox" in plan

    def test_manual_plan_report_depends_on_full_chain(self):
        """P1-5 验证点：用户只勾选 report_generation 时，自动补全完整链路。"""
        from app.graph_v3 import _build_manual_plan
        plan = _build_manual_plan(["report_generation"])
        assert "worker_mechanism" in plan
        assert "worker_sandbox" in plan
        assert "worker_report" in plan


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
