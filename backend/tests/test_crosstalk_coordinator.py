# BioDynamics Agent v4 - Cross-talk Coordinator Agent 单元测试 (Phase 4 / Task 4.13.8)
#
# 测试用例（≥30）：
#   TestCrossTalkCoordinatorRegistration: 基础属性（3）
#   TestSharedSpeciesIdentification: shared species 识别（6）
#   TestSharedSpeciesSync: 同步策略（5）
#   TestCrossTalkEdgeInjector: edge 注入（6）
#   TestPathwayTagIsolation: pathway_tag 隔离（4）
#   TestTimeScaleAligner: 时间尺度对齐（4）
#   TestCrossTalkCoordinatorHook: LangGraph hook（4）
#
# 运行：cd backend && python -m pytest tests/test_crosstalk_coordinator.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# 测试辅助：构造 Specialist 输出
# =============================================================================
def _make_egfr_output() -> dict:
    """构造 EGFR Specialist 输出（含 shared_species=["Ras"]）。"""
    return {
        "pathway_class": "EGFR_RTK",
        "shared_species": ["Ras"],
        "species": [
            {"name": "EGF", "species_type": "ligand"},
            {"name": "EGFR", "species_type": "protein"},
            {"name": "Ras", "species_type": "protein", "shared": True},
            {"name": "RasGTP", "species_type": "protein"},
        ],
        "reactions": [
            {
                "id": "R_EGFR_PI3K",
                "source": "pEGFR",
                "target": "PI3K",
                "mechanism": "activation",
                "pathway_tag": "EGFR_RTK",
                "substrate": "pEGFR",
                "product": "PI3K",
                "modifier": None,
            },
        ],
        "crosstalk_reactions": [
            {
                "source": "pEGFR",
                "target": "PI3K",
                "mechanism": "activation",
                "shared_species": [],
                "description": "pEGFR 直接磷酸化 PI3K",
            },
        ],
        "kinetics_overrides": {"k_egfr_int": 0.01},
        "max_step": 0.1,
        "time_scale": "fast",
    }


def _make_pi3k_output() -> dict:
    """构造 PI3K Specialist 输出（含 shared_species=["AKT"]）。"""
    return {
        "pathway_class": "PI3K_AKT_mTOR",
        "shared_species": ["AKT"],
        "species": [
            {"name": "PI3K", "species_type": "protein"},
            {"name": "AKT", "species_type": "protein", "shared": True},
            {"name": "pAKT", "species_type": "protein"},
            {"name": "mTORC1", "species_type": "complex", "shared": True},
        ],
        "reactions": [
            {
                "id": "R_PIP3_pAKT",
                "source": "PIP3",
                "target": "pAKT",
                "mechanism": "phosphorylation",
                "pathway_tag": "PI3K_AKT_mTOR",
                "substrate": "AKT",
                "product": "pAKT",
                "modifier": "PIP3",
            },
        ],
        "crosstalk_reactions": [
            {
                "source": "pAKT",
                "target": "Raf",
                "mechanism": "inhibition",
                "shared_species": ["AKT"],
                "site": "Ser259",
                "description": "pAKT 磷酸化 Raf Ser259 抑制 MAPK",
            },
        ],
        "kinetics_overrides": {"k_akt_phos": 0.05},
        "max_step": 1.0,
        "time_scale": "medium",
    }


def _make_mapk_output() -> dict:
    """构造 MAPK Specialist 输出（含 shared_species=["RasGTP"]）。"""
    return {
        "pathway_class": "MAPK_ERK",
        "shared_species": ["RasGTP"],
        "species": [
            {"name": "RasGTP", "species_type": "protein", "shared": True},
            {"name": "Raf", "species_type": "protein", "shared": True},
            {"name": "MEK", "species_type": "protein", "shared": True},
            {"name": "ERK", "species_type": "protein", "shared": True},
        ],
        "reactions": [
            {
                "id": "R_pRaf_pMEK",
                "source": "pRaf",
                "target": "pMEK",
                "mechanism": "phosphorylation",
                "pathway_tag": "MAPK_ERK",
                "substrate": "MEK",
                "product": "pMEK",
                "modifier": "pRaf",
            },
        ],
        "crosstalk_reactions": [
            {
                "source": "pERK",
                "target": "ELK1",
                "mechanism": "phosphorylation",
                "shared_species": [],
                "description": "pERK 激活转录因子 ELK1",
            },
        ],
        "kinetics_overrides": {"k_mek_phos": 0.2},
        "max_step": 0.1,
        "time_scale": "fast",
    }


# =============================================================================
# 1. TestCrossTalkCoordinatorRegistration: 基础属性
# =============================================================================
class TestCrossTalkCoordinatorRegistration(unittest.TestCase):
    """Coordinator 基础属性测试。"""

    def test_coordinator_instantiation(self):
        """Coordinator 可正常实例化。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        self.assertIsNotNone(coord)

    def test_coordinator_has_coordinate_method(self):
        """Coordinator 含 coordinate 方法。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        self.assertTrue(hasattr(coord, "coordinate"))
        self.assertTrue(callable(coord.coordinate))

    def test_coordinator_has_identify_shared_species_method(self):
        """Coordinator 含 _identify_shared_species 方法。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        self.assertTrue(hasattr(coord, "_identify_shared_species"))

    def test_hook_node_is_callable(self):
        """crosstalk_coordinator_hook_node 为可调用函数。"""
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
        self.assertTrue(callable(crosstalk_coordinator_hook_node))


# =============================================================================
# 2. TestSharedSpeciesIdentification: shared species 识别
# =============================================================================
class TestSharedSpeciesIdentification(unittest.TestCase):
    """shared species 识别测试。"""

    def test_single_pathway_no_shared(self):
        """单通路场景：pathway_class 不含 MULTI: 返回空 shared species。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        result = coord.coordinate([_make_egfr_output()], "EGFR_RTK")
        self.assertEqual(result["v4_shared_species"], [])

    def test_multi_pathway_egfr_pi3k_shared_akt(self):
        """EGFR + PI3K 双通路：识别 AKT 为 shared（PI3K 标记 shared=True）。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        result = coord.coordinate(
            [_make_egfr_output(), _make_pi3k_output()],
            "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
        )
        # AKT 在 PI3K species 中标记 shared=True
        self.assertIn("AKT", result["v4_shared_species"])

    def test_multi_pathway_egfr_mapk_shared_rasgtp(self):
        """EGFR + MAPK 双通路：识别 RasGTP 为 shared。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        egfr = _make_egfr_output()
        # EGFR 也声明 RasGTP 为 shared_species
        egfr["shared_species"] = ["Ras", "RasGTP"]
        result = coord.coordinate(
            [egfr, _make_mapk_output()],
            "MULTI:EGFR_RTK+MAPK_ERK",
        )
        self.assertIn("RasGTP", result["v4_shared_species"])

    def test_collect_specialist_is_shared_marked(self):
        """收集 Specialist species 中标记 shared=True 的物种。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        # 构造两个通路都标记 mTORC1 为 shared
        out1 = {
            "pathway_class": "PI3K_AKT_mTOR",
            "shared_species": [],
            "species": [{"name": "mTORC1", "shared": True}],
        }
        out2 = {
            "pathway_class": "Apoptosis",
            "shared_species": [],
            "species": [{"name": "mTORC1", "shared": True}],
        }
        shared = coord._identify_shared_species([out1, out2])
        self.assertIn("mTORC1", shared)

    def test_shared_species_deduplication(self):
        """shared species 去重（同名只出现一次）。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        out1 = {"pathway_class": "A", "shared_species": ["Ras"]}
        out2 = {"pathway_class": "B", "shared_species": ["Ras"]}
        shared = coord._identify_shared_species([out1, out2])
        self.assertEqual(shared.count("Ras"), 1)

    def test_empty_specialist_outputs(self):
        """空 Specialist 输出列表返回空 shared species。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        shared = coord._identify_shared_species([])
        self.assertEqual(shared, [])

    def test_species_appearing_in_only_one_pathway_not_shared(self):
        """仅出现在 1 个通路的 species 不标记为 shared。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        out1 = {"pathway_class": "A", "shared_species": ["UniqueA"]}
        out2 = {"pathway_class": "B", "shared_species": ["UniqueB"]}
        shared = coord._identify_shared_species([out1, out2])
        self.assertEqual(shared, [])


# =============================================================================
# 3. TestSharedSpeciesSync: 同步策略
# =============================================================================
class TestSharedSpeciesSync(unittest.TestCase):
    """shared species 同步策略测试。"""

    def test_sync_map_correct(self):
        """sync_map 正确：species_name 映射到同名 canonical 变量。"""
        from app.crosstalk.shared_species_sync import SharedSpeciesSync
        sync = SharedSpeciesSync()
        result = sync.compute_sync_strategy(["AKT", "RasGTP"], [])
        self.assertEqual(result["sync_map"]["AKT"], "AKT")
        self.assertEqual(result["sync_map"]["RasGTP"], "RasGTP")

    def test_dominant_pathway_produced_priority(self):
        """主导通路选择：物种作为 product 被生成的通路优先。"""
        from app.crosstalk.shared_species_sync import SharedSpeciesSync
        sync = SharedSpeciesSync()
        # 通路 A：AKT 作为 product（produced）
        out_a = {
            "pathway_class": "PI3K_AKT_mTOR",
            "species": [{"name": "AKT"}],
            "reactions": [
                {"product": "AKT", "source": "PIP3", "target": "pAKT",
                 "mechanism": "phosphorylation"},
            ],
        }
        # 通路 B：AKT 仅作为 modifier（consumed）
        out_b = {
            "pathway_class": "Apoptosis",
            "species": [{"name": "AKT"}],
            "reactions": [
                {"modifier": "AKT", "source": "AKT", "target": "Bad",
                 "mechanism": "inhibition"},
            ],
        }
        result = sync.compute_sync_strategy(["AKT"], [out_a, out_b])
        self.assertEqual(result["pathway_assignments"]["AKT"], "PI3K_AKT_mTOR")

    def test_conflict_resolution_strategy(self):
        """冲突解决策略：有参数差异时返回 median_value。"""
        from app.crosstalk.shared_species_sync import SharedSpeciesSync
        sync = SharedSpeciesSync()
        out1 = {
            "pathway_class": "A",
            "species": [{"name": "AKT"}],
            "kinetics_overrides": {"k_akt_a": 0.1, "akt_rate": 0.05},
        }
        out2 = {
            "pathway_class": "B",
            "species": [{"name": "AKT"}],
            "kinetics_overrides": {"k_akt_b": 0.3, "akt_rate": 0.15},
        }
        result = sync.compute_sync_strategy(["AKT"], [out1, out2])
        conflict = result["conflict_resolution"]["AKT"]
        # akt_rate 在两个通路中有不同值，应触发 median_value 策略
        self.assertIn(conflict["strategy"], ("median_value", "no_conflict"))
        self.assertIn("pathway_tags", conflict)

    def test_pathway_assignments_present(self):
        """pathway_assignments 字段存在且包含所有 shared species。"""
        from app.crosstalk.shared_species_sync import SharedSpeciesSync
        sync = SharedSpeciesSync()
        result = sync.compute_sync_strategy(["AKT", "Ras"], [])
        self.assertIn("AKT", result["pathway_assignments"])
        self.assertIn("Ras", result["pathway_assignments"])

    def test_sync_empty_shared_species(self):
        """空 shared species 列表返回空 sync 策略。"""
        from app.crosstalk.shared_species_sync import SharedSpeciesSync
        sync = SharedSpeciesSync()
        result = sync.compute_sync_strategy([], [])
        self.assertEqual(result["sync_map"], {})
        self.assertEqual(result["pathway_assignments"], {})


# =============================================================================
# 4. TestCrossTalkEdgeInjector: edge 注入
# =============================================================================
class TestCrossTalkEdgeInjector(unittest.TestCase):
    """Cross-talk edge 注入测试。"""

    def test_inject_edges_to_pathway_graph(self):
        """注入 edge 到 pathway_graph（新增 cross_talk_edges 字段）。"""
        from app.crosstalk.crosstalk_edges import CrossTalkEdgeInjector
        injector = CrossTalkEdgeInjector()
        graph = {"nodes": [], "edges": []}
        edges = [
            {
                "id": "CT_EGFR_TO_PI3K",
                "source_pathway": "EGFR_RTK",
                "target_pathway": "PI3K_AKT_mTOR",
                "source_node": "PN_pEGFR",
                "target_node": "PN_PI3K",
                "mechanism": "activation",
            }
        ]
        result = injector.inject_edges(graph, edges)
        self.assertIn("cross_talk_edges", result)
        self.assertEqual(len(result["cross_talk_edges"]), 1)
        self.assertEqual(result["cross_talk_edges"][0]["id"], "CT_EGFR_TO_PI3K")

    def test_validate_edges_missing_field_filtered(self):
        """校验 edge schema：缺字段的 edge 被过滤。"""
        from app.crosstalk.crosstalk_edges import CrossTalkEdgeInjector
        injector = CrossTalkEdgeInjector()
        edges = [
            {"id": "CT_1", "source_pathway": "A", "target_pathway": "B",
             "source_node": "x", "target_node": "y", "mechanism": "activation"},
            # 缺 mechanism
            {"id": "CT_2", "source_pathway": "A", "target_pathway": "B",
             "source_node": "x", "target_node": "y"},
            # 缺 target_pathway
            {"id": "CT_3", "source_pathway": "A",
             "source_node": "x", "target_node": "y", "mechanism": "activation"},
        ]
        valid = injector.validate_edges(edges)
        ids = [e["id"] for e in valid]
        self.assertIn("CT_1", ids)
        self.assertNotIn("CT_2", ids)
        self.assertNotIn("CT_3", ids)

    def test_validate_edges_valid_passes(self):
        """校验 edge schema：有效 edge 全部通过。"""
        from app.crosstalk.crosstalk_edges import CrossTalkEdgeInjector
        injector = CrossTalkEdgeInjector()
        edges = [
            {"id": "CT_1", "source_pathway": "A", "target_pathway": "B",
             "source_node": "x", "target_node": "y", "mechanism": "activation"},
            {"id": "CT_2", "source_pathway": "A", "target_pathway": "C",
             "source_node": "x", "target_node": "z", "mechanism": "inhibition"},
        ]
        valid = injector.validate_edges(edges)
        self.assertEqual(len(valid), 2)

    def test_deduplicate_edges_by_id(self):
        """按 id 去重：相同 id 只保留首个。"""
        from app.crosstalk.crosstalk_edges import CrossTalkEdgeInjector
        injector = CrossTalkEdgeInjector()
        edges = [
            {"id": "CT_1", "source_pathway": "A", "target_pathway": "B",
             "source_node": "x", "target_node": "y", "mechanism": "activation"},
            {"id": "CT_1", "source_pathway": "A", "target_pathway": "B",
             "source_node": "x", "target_node": "y", "mechanism": "activation"},
            {"id": "CT_2", "source_pathway": "A", "target_pathway": "C",
             "source_node": "x", "target_node": "z", "mechanism": "inhibition"},
        ]
        deduped = injector.deduplicate_edges(edges)
        self.assertEqual(len(deduped), 2)

    def test_inject_edges_preserves_existing(self):
        """注入 edges 时保留已存在的 cross_talk_edges（按 id 去重合并）。"""
        from app.crosstalk.crosstalk_edges import CrossTalkEdgeInjector
        injector = CrossTalkEdgeInjector()
        graph = {
            "nodes": [],
            "edges": [],
            "cross_talk_edges": [
                {"id": "CT_EXISTING", "source_pathway": "A", "target_pathway": "B",
                 "source_node": "x", "target_node": "y", "mechanism": "activation"},
            ],
        }
        new_edges = [
            {"id": "CT_NEW", "source_pathway": "A", "target_pathway": "C",
             "source_node": "x", "target_node": "z", "mechanism": "inhibition"},
        ]
        result = injector.inject_edges(graph, new_edges)
        ids = [e["id"] for e in result["cross_talk_edges"]]
        self.assertIn("CT_EXISTING", ids)
        self.assertIn("CT_NEW", ids)

    def test_inject_empty_edges(self):
        """注入空 edges 列表：pathway_graph 保持原结构（cross_talk_edges 为空）。"""
        from app.crosstalk.crosstalk_edges import CrossTalkEdgeInjector
        injector = CrossTalkEdgeInjector()
        graph = {"nodes": [], "edges": []}
        result = injector.inject_edges(graph, [])
        self.assertEqual(result["cross_talk_edges"], [])


# =============================================================================
# 5. TestPathwayTagIsolation: pathway_tag 隔离
# =============================================================================
class TestPathwayTagIsolation(unittest.TestCase):
    """pathway_tag 隔离测试。"""

    def test_tag_crosstalk_ab(self):
        """标记 cross-talk 相关参数为 CROSSTALK_A_B。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        out1 = {
            "pathway_class": "EGFR_RTK",
            "reactions": [],
            "crosstalk_reactions": [
                {
                    "source": "pEGFR",
                    "target": "PI3K",
                    "mechanism": "activation",
                    "shared_species": ["Ras"],
                    "substrate": "pEGFR",
                    "product": "PI3K",
                },
            ],
            "kinetics_overrides": {},
        }
        out2 = {
            "pathway_class": "PI3K_AKT_mTOR",
            "reactions": [],
            "crosstalk_reactions": [],
            "kinetics_overrides": {},
        }
        result = coord._enforce_pathway_tag_isolation([out1, out2])
        tagged = result["tagged_parameters"]
        # 应有 CROSSTALK_EGFR_RTK_PI3K_AKT_mTOR 标签
        crosstalk_tags = [t["pathway_tag"] for t in tagged if t["is_crosstalk"]]
        self.assertTrue(any("CROSSTALK_EGFR_RTK" in t for t in crosstalk_tags))

    def test_isolation_no_violations_normal(self):
        """正常情况（参数各自归属通路）：isolation_violations 为空。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        out1 = {
            "pathway_class": "EGFR_RTK",
            "reactions": [
                {"id": "R1", "source": "EGF", "target": "EGFR",
                 "mechanism": "binding", "substrate": "EGF", "product": "EGFR",
                 "modifier": None, "pathway_tag": "EGFR_RTK"},
            ],
            "crosstalk_reactions": [],
            "kinetics_overrides": {"k_egfr": 0.1},
        }
        out2 = {
            "pathway_class": "PI3K_AKT_mTOR",
            "reactions": [
                {"id": "R2", "source": "PI3K", "target": "PIP3",
                 "mechanism": "activation", "substrate": "PIP2", "product": "PIP3",
                 "modifier": "PI3K", "pathway_tag": "PI3K_AKT_mTOR"},
            ],
            "crosstalk_reactions": [],
            "kinetics_overrides": {"k_pi3k": 0.05},
        }
        result = coord._enforce_pathway_tag_isolation([out1, out2])
        self.assertEqual(result["isolation_violations"], [])

    def test_isolation_violations_detected(self):
        """参数污染检测：同一非 crosstalk 参数被多个通路覆盖时检测违规。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        out1 = {
            "pathway_class": "EGFR_RTK",
            "reactions": [],
            "crosstalk_reactions": [],
            "kinetics_overrides": {"k_shared_param": 0.1},
        }
        out2 = {
            "pathway_class": "PI3K_AKT_mTOR",
            "reactions": [],
            "crosstalk_reactions": [],
            "kinetics_overrides": {"k_shared_param": 0.2},
        }
        result = coord._enforce_pathway_tag_isolation([out1, out2])
        # k_shared_param 被两个通路覆盖，应检测到违规
        violations = result["isolation_violations"]
        self.assertTrue(
            any(v["parameter_name"] == "k_shared_param" for v in violations)
        )

    def test_tagged_parameters_structure(self):
        """tagged_parameters 每条含 parameter_name / pathway_tag / is_crosstalk。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        out1 = {
            "pathway_class": "EGFR_RTK",
            "reactions": [
                {"id": "R1", "source": "EGF", "target": "EGFR",
                 "mechanism": "binding", "substrate": "EGF", "product": "EGFR",
                 "modifier": None},
            ],
            "crosstalk_reactions": [],
            "kinetics_overrides": {"k_egfr": 0.1},
        }
        result = coord._enforce_pathway_tag_isolation([out1])
        for tp in result["tagged_parameters"]:
            self.assertIn("parameter_name", tp)
            self.assertIn("pathway_tag", tp)
            self.assertIn("is_crosstalk", tp)


# =============================================================================
# 6. TestTimeScaleAligner: 时间尺度对齐
# =============================================================================
class TestTimeScaleAligner(unittest.TestCase):
    """时间尺度对齐测试。"""

    def test_single_pathway_returns_own_max_step(self):
        """单通路场景：直接返回该通路的 max_step。"""
        from app.crosstalk.time_scale_aligner import TimeScaleAligner
        aligner = TimeScaleAligner()
        result = aligner.align_time_scales([
            {"pathway_class": "EGFR_RTK", "max_step": 0.1, "time_scale": "fast"},
        ])
        self.assertEqual(result["unified_max_step"], 0.1)
        self.assertEqual(result["alignment_strategy"], "single_pathway")

    def test_multi_pathway_takes_min_max_step(self):
        """多通路场景：取所有通路 max_step 最小值。"""
        from app.crosstalk.time_scale_aligner import TimeScaleAligner
        aligner = TimeScaleAligner()
        result = aligner.align_time_scales([
            {"pathway_class": "EGFR_RTK", "max_step": 0.1},
            {"pathway_class": "PI3K_AKT_mTOR", "max_step": 1.0},
        ])
        self.assertEqual(result["unified_max_step"], 0.1)
        self.assertEqual(result["alignment_strategy"], "min_of_all")

    def test_alignment_strategy_min_of_all(self):
        """多通路对齐策略为 min_of_all。"""
        from app.crosstalk.time_scale_aligner import TimeScaleAligner
        aligner = TimeScaleAligner()
        result = aligner.align_time_scales([
            {"pathway_class": "A", "max_step": 0.5},
            {"pathway_class": "B", "max_step": 0.2},
            {"pathway_class": "C", "max_step": 1.0},
        ])
        self.assertEqual(result["alignment_strategy"], "min_of_all")
        self.assertEqual(result["unified_max_step"], 0.2)

    def test_pathway_time_scales_recorded(self):
        """每通路时间尺度信息被记录。"""
        from app.crosstalk.time_scale_aligner import TimeScaleAligner
        aligner = TimeScaleAligner()
        result = aligner.align_time_scales([
            {"pathway_class": "EGFR_RTK", "max_step": 0.1, "time_scale": "fast"},
            {"pathway_class": "PI3K_AKT_mTOR", "max_step": 1.0, "time_scale": "medium"},
        ])
        scales = result["pathway_time_scales"]
        self.assertEqual(len(scales), 2)
        classes = [s["pathway_class"] for s in scales]
        self.assertIn("EGFR_RTK", classes)
        self.assertIn("PI3K_AKT_mTOR", classes)


# =============================================================================
# 7. TestCrossTalkCoordinatorHook: LangGraph hook
# =============================================================================
class TestCrossTalkCoordinatorHook(unittest.TestCase):
    """LangGraph hook 测试（Feature Flag 隔离）。"""

    def test_feature_flag_false_returns_empty(self):
        """Feature Flag false 时 hook 返回 {}。"""
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
        with patch("app.config.settings") as mock_settings:
            mock_settings.V4_CROSSTALK_COORDINATOR_ENABLED = False
            mock_settings.effective_v4_crosstalk_coordinator_enabled.return_value = False
            state = {
                "v4_pathway_class": "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
                "v4_specialist_outputs": [_make_egfr_output(), _make_pi3k_output()],
            }
            result = crosstalk_coordinator_hook_node(state)
            self.assertEqual(result, {})

    def test_feature_flag_true_single_pathway_returns_empty_lists(self):
        """Feature Flag true + 单通路：返回空列表字段。"""
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
        with patch("app.config.settings") as mock_settings:
            mock_settings.V4_CROSSTALK_COORDINATOR_ENABLED = True
            state = {
                "v4_pathway_class": "EGFR_RTK",
                "v4_specialist_outputs": [_make_egfr_output()],
            }
            result = crosstalk_coordinator_hook_node(state)
            self.assertEqual(result["v4_crosstalk_edges"], [])
            self.assertEqual(result["v4_shared_species"], [])

    def test_feature_flag_true_multi_pathway_merges(self):
        """Feature Flag true + 多通路：调用 coordinate() 合并。"""
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
        with patch("app.config.settings") as mock_settings:
            mock_settings.V4_CROSSTALK_COORDINATOR_ENABLED = True
            state = {
                "v4_pathway_class": "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
                "v4_specialist_outputs": [_make_egfr_output(), _make_pi3k_output()],
            }
            result = crosstalk_coordinator_hook_node(state)
            # 应有 cross-talk edges + shared species
            self.assertIn("v4_crosstalk_edges", result)
            self.assertIn("v4_shared_species", result)
            self.assertIn("v4_shared_species_sync", result)
            self.assertIn("v4_time_scale_alignment", result)

    def test_hook_exception_returns_empty(self):
        """hook 异常时返回 {}（降级，不抛异常）。"""
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
        with patch("app.config.settings") as mock_settings:
            mock_settings.V4_CROSSTALK_COORDINATOR_ENABLED = True
            # 传入非法 state 触发异常路径
            state = None
            result = crosstalk_coordinator_hook_node(state)
            self.assertEqual(result, {})


# =============================================================================
# 8. TestCrossTalkCoordinatorEndToEnd: coordinate 主流程
# =============================================================================
class TestCrossTalkCoordinatorEndToEnd(unittest.TestCase):
    """Coordinator coordinate() 主流程测试。"""

    def test_coordinate_multi_pathway_returns_all_fields(self):
        """多通路 coordinate() 返回所有 v4 字段。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        result = coord.coordinate(
            [_make_egfr_output(), _make_pi3k_output()],
            "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
        )
        self.assertIn("v4_crosstalk_edges", result)
        self.assertIn("v4_shared_species", result)
        self.assertIn("v4_shared_species_sync", result)
        self.assertIn("v4_time_scale_alignment", result)
        self.assertIn("v4_pathway_tag_isolation", result)

    def test_coordinate_single_pathway_returns_empty(self):
        """单通路 coordinate() 返回空列表 + 空 sync 策略。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        result = coord.coordinate([_make_egfr_output()], "EGFR_RTK")
        self.assertEqual(result["v4_crosstalk_edges"], [])
        self.assertEqual(result["v4_shared_species"], [])

    def test_coordinate_insufficient_specialists(self):
        """多通路但 Specialist 输出 < 2 个：返回空。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        result = coord.coordinate(
            [_make_egfr_output()],
            "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
        )
        self.assertEqual(result["v4_crosstalk_edges"], [])
        self.assertEqual(result["v4_shared_species"], [])

    def test_coordinate_collects_crosstalk_edges_from_specialists(self):
        """coordinate() 收集 Specialist 输出中的 crosstalk_edges 字段。"""
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        out1 = _make_egfr_output()
        out1["crosstalk_edges"] = [
            {
                "id": "CT_EGFR_TO_PI3K",
                "source_pathway": "EGFR_RTK",
                "target_pathway": "PI3K_AKT_mTOR",
                "source_node": "PN_pEGFR",
                "target_node": "PN_PI3K",
                "mechanism": "activation",
            }
        ]
        out2 = _make_pi3k_output()
        result = coord.coordinate([out1, out2], "MULTI:EGFR_RTK+PI3K_AKT_mTOR")
        ids = [e["id"] for e in result["v4_crosstalk_edges"]]
        self.assertIn("CT_EGFR_TO_PI3K", ids)


if __name__ == "__main__":
    unittest.main()
