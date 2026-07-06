"""EGF-EGFR 级联场景修复验证测试。

覆盖 2026-07-04 实际运行时暴露的缺陷：
1. Cascade 模板 edges_json 与 SPECIES_NAMES 不一致（中文边 vs ASCII 物种名）
2. Y0 全部硬编码为 10.0，未读取用户输入的初始浓度
3. params_json key 与 SPECIES_NAMES 不匹配导致 kd 回退到 10.0
4. _build_rag_params_context 无法解析 v3 的 "source->target" key
5. KGBuilder 对 NER 占位符（e1/e2）未使用 aliases 回填
"""

from __future__ import annotations

from typing import Any

import pytest

from app.kg_builder import KGBuilder
from app.nodes import _build_rag_params_context
from app.nodes_v2 import n6_ode_generator


# =============================================================================
# 1. Cascade 模板一致性：边 source/target 与 SPECIES_NAMES 使用相同 ODE 标识符
# =============================================================================
class TestCascadeTemplateIdentifierConsistency:
    """验证 Cascade_Activation 模板渲染出的代码内部一致。"""

    def test_egf_egfr_cascade_edges_use_ascii_identifiers(self):
        """EDGES 中的 source/target 必须出现在 SPECIES_NAMES 中。"""
        user_input = (
            "表皮生长因子（EGF）结合 EGFR 受体后诱导其二聚化和自磷酸化，"
            "激活下游 Shc-Grb2-SOS-Ras-MAPK 信号级联。"
            "初始条件：EGF=0.008 nM，EGFR=0.3 nM。"
        )
        state = {
            "user_input": user_input,
            "knowledge_graph": {
                "nodes": [
                    {"id": "EGF", "name": "EGF", "type": "Protein", "aliases": []},
                    {"id": "EGFR", "name": "EGFR 受体", "type": "Protein", "aliases": ["EGFR"]},
                    {"id": "pEGFR", "name": "磷酸化 EGFR", "type": "Protein", "aliases": ["pEGFR"]},
                    {"id": "Shc", "name": "Shc", "type": "Protein", "aliases": []},
                    {"id": "Grb2", "name": "Grb2", "type": "Protein", "aliases": []},
                    {"id": "SOS", "name": "SOS", "type": "Protein", "aliases": []},
                    {"id": "Ras", "name": "Ras", "type": "Protein", "aliases": []},
                    {"id": "MAPK", "name": "MAPK", "type": "Protein", "aliases": []},
                ],
                "edges": [
                    {"source": "EGF", "target": "EGFR 受体", "interaction": "activation"},
                    {"source": "EGFR 受体", "target": "磷酸化 EGFR", "interaction": "activation"},
                    {"source": "磷酸化 EGFR", "target": "Shc", "interaction": "activation"},
                    {"source": "Shc", "target": "Grb2", "interaction": "activation"},
                    {"source": "Grb2", "target": "SOS", "interaction": "activation"},
                    {"source": "SOS", "target": "Ras", "interaction": "activation"},
                    {"source": "Ras", "target": "MAPK", "interaction": "activation"},
                ],
            },
            "parameters": {},
            "mechanism": {"template": "Cascade_Activation"},
            "pkpd_profile": {},
            "drug_regimen": [],
        }
        result = n6_ode_generator(state)
        ode_code = result["ode_model"]["code"]

        # 解析 SPECIES_NAMES 与 EDGES
        species_line = next(
            (line for line in ode_code.splitlines() if "SPECIES_NAMES" in line), ""
        )
        edges_line_start = ode_code.find("EDGES = [")
        edges_line_end = ode_code.find("]", edges_line_start) + 1
        edges_block = ode_code[edges_line_start:edges_line_end]

        # SPECIES_NAMES 应为 ASCII 标识符
        assert "'EGF'" in species_line
        assert "'EGFR'" in species_line
        assert "'pEGFR'" in species_line
        assert "表皮生长" not in species_line

        # EDGES 中的 source/target 必须是 ASCII 标识符且属于 SPECIES_NAMES
        assert "'EGF'" in edges_block
        assert "'EGFR'" in edges_block
        assert "'pEGFR'" in edges_block
        assert "表皮生长" not in edges_block
        assert "磷酸化" not in edges_block

    def test_egf_egfr_y0_parsed_from_user_input(self):
        """Y0 必须包含用户输入的 EGF=0.008 nM 和 EGFR=0.3 nM。"""
        user_input = "初始条件：EGF=0.008 nM，EGFR=0.3 nM。"
        state = {
            "user_input": user_input,
            "knowledge_graph": {
                "nodes": [
                    {"id": "EGF", "name": "EGF", "type": "Protein", "aliases": []},
                    {"id": "EGFR", "name": "EGFR", "type": "Protein", "aliases": []},
                    {"id": "pEGFR", "name": "pEGFR", "type": "Protein", "aliases": []},
                ],
                "edges": [
                    {"source": "EGF", "target": "EGFR", "interaction": "activation"},
                    {"source": "EGFR", "target": "pEGFR", "interaction": "activation"},
                ],
            },
            "parameters": {},
            "mechanism": {"template": "Cascade_Activation"},
            "pkpd_profile": {},
            "drug_regimen": [],
        }
        result = n6_ode_generator(state)
        ode_code = result["ode_model"]["code"]

        y0_line = next(
            (line for line in ode_code.splitlines() if "Y0 =" in line), ""
        )
        assert "0.008" in y0_line, f"Y0 未包含 EGF=0.008: {y0_line}"
        assert "0.3" in y0_line, f"Y0 未包含 EGFR=0.3: {y0_line}"

    def test_cascade_params_keys_match_species_names(self):
        """params_json 的 key 必须与 SPECIES_NAMES 对齐，才能被模板正确 lookup。"""
        state = {
            "user_input": "EGF=1.0 nM，EGFR=2.0 nM，pEGFR=0.1 nM。",
            "knowledge_graph": {
                "nodes": [
                    {"id": "EGF", "name": "EGF", "type": "Protein", "aliases": []},
                    {"id": "EGFR", "name": "EGFR", "type": "Protein", "aliases": []},
                    {"id": "pEGFR", "name": "pEGFR", "type": "Protein", "aliases": []},
                ],
                "edges": [
                    {"source": "EGF", "target": "EGFR", "interaction": "activation"},
                    {"source": "EGFR", "target": "pEGFR", "interaction": "activation"},
                ],
            },
            "parameters": {
                "EGF->EGFR": {
                    "edge_key": "EGF->EGFR",
                    "param_name": "Kd",
                    "value": 0.5,
                    "unit": "nM",
                    "source": "BIOMD0000000205",
                    "is_fallback": False,
                },
                "EGFR->pEGFR": {
                    "edge_key": "EGFR->pEGFR",
                    "param_name": "Kd",
                    "value": 1.5,
                    "unit": "nM",
                    "source": "BIOMD0000000205",
                    "is_fallback": False,
                },
            },
            "mechanism": {"template": "Cascade_Activation"},
            "pkpd_profile": {},
            "drug_regimen": [],
        }
        result = n6_ode_generator(state)
        ode_code = result["ode_model"]["code"]

        # PARAMS 字典应使用 ASCII 物种名作为 key
        assert "'EGFR': {'kd': 0.5" in ode_code or '"EGFR": {"kd": 0.5' in ode_code
        assert "'pEGFR': {'kd': 1.5" in ode_code or '"pEGFR": {"kd": 1.5' in ode_code


# =============================================================================
# 2. _build_rag_params_context 兼容 v3 "source->target" key
# =============================================================================
class TestRagParamsContextKeyFormats:
    """验证 _build_rag_params_context 同时支持 '|' 与 '->' 分隔符。"""

    def test_arrow_separator_supported(self):
        """v3 n5 使用 'source->target'，不应再抛 ValueError。"""
        rag_selected = {
            "EGF->EGFR": {
                "param_found": True,
                "selected_params": [
                    {"param_name": "Kd", "value": 0.5, "unit": "nM", "source": "RAG"}
                ],
                "reasoning": "命中 BIOMD0000000205",
                "fallback_to_estimation": False,
            }
        }
        ctx = _build_rag_params_context(rag_selected)
        assert "EGF" in ctx
        assert "EGFR" in ctx
        assert "0.5" in ctx

    def test_pipe_separator_still_supported(self):
        """v1 使用 'source|target' 仍需兼容。"""
        rag_selected = {
            "EGF|EGFR": {
                "param_found": False,
                "selected_params": [],
                "reasoning": "无命中",
                "fallback_to_estimation": True,
            }
        }
        ctx = _build_rag_params_context(rag_selected)
        assert "EGF" in ctx
        assert "EGFR" in ctx


# =============================================================================
# 3. KGBuilder 占位符回填
# =============================================================================
class TestKgBuilderPlaceholderFallback:
    """验证 NER 实体 name 缺失或占位时使用 aliases 回填。"""

    def test_placeholder_name_uses_alias(self):
        """entity_id=e1 且 name=e1 时，应使用 aliases 中的 EGFR。"""
        builder = KGBuilder()
        entities = [
            {
                "entity_id": "e1",
                "name": "e1",
                "type": "Protein",
                "aliases": ["EGFR"],
            }
        ]
        relations = [{"source": "e1", "target": "Shc", "interaction": "activation"}]
        kg = builder.build(entities, relations)

        node_names = {n["name"] for n in kg["nodes"]}
        assert "EGFR" in node_names
        assert "e1" not in node_names
        assert kg["edges"][0]["source"] == "EGFR"

    def test_empty_name_uses_alias(self):
        """name 为空字符串时，应使用 aliases[0]。"""
        builder = KGBuilder()
        entities = [
            {
                "entity_id": "e2",
                "name": "",
                "type": "Protein",
                "aliases": ["MAPK"],
            }
        ]
        relations = [{"source": "Ras", "target": "e2", "interaction": "activation"}]
        kg = builder.build(entities, relations)

        node_names = {n["name"] for n in kg["nodes"]}
        assert "MAPK" in node_names
        assert kg["edges"][0]["target"] == "MAPK"
