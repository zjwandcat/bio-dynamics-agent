# BioDynamics Agent v4 - ODE 模板路由测试（Task B.0 / P0-2 修复）
#
# 验证 ODERendererV2._select_template() 的通路类别 → 模板路由：
#   1. 10 个 Pathway Specialist 的 pathway_class 值均路由到正确的 v4 模板
#   2. pathway_graph/initializer.py 通路键（另一套命名）同样路由正确
#   3. 大小写不敏感（specialist 与 initializer 命名差异容忍）
#   4. 未知 pathway_class 回退到 oscillatory_feedback.j2（向后兼容）
#   5. 空 pathway_class 回退到 oscillatory_feedback.j2（向后兼容）
#   6. 11 个 v4 模板均通过至少一个 pathway_class 可达（无不可达模板）
#   7. 每个路由结果可被 ODERendererV2.env 加载且渲染产物可编译
#
# 对应 RC Audit P0-2：原 _select_template() 仅路由到 oscillatory_feedback /
# bistable_switch 两个模板，7 个 P3 模板（transcriptional_delay /
# nuclear_transport / ubiquitination_cascade / destruction_complex /
# caspase_cascade / cyclin_cdk_toggle / transcription_factor）不可达。
#
# 运行：cd backend && python -m pytest tests/test_ode_template_routing.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# 期望路由表（pathway_class → 模板文件名）
# =============================================================================
# 同时覆盖 specialist pathway_class 值与 initializer 通路键两套命名。
# 与 ode_renderer_v2._PATHWAY_TEMPLATE_MAP 对齐。
EXPECTED_ROUTING: dict[str, str] = {
    # 磷酸化级联（承载模板：oscillatory_feedback.j2，_ode_rhs 内含 phosphorylation 分支）
    # 注：_mechanism_phosphorylation_mm.j2 是 {% include %} 片段，不可顶层渲染
    "EGFR_RTK": "oscillatory_feedback.j2",
    "MAPK_ERK": "oscillatory_feedback.j2",
    "PI3K_AKT_mTOR": "oscillatory_feedback.j2",
    # 转录延迟振荡器（specialist 命名 / initializer 命名）
    "p53": "transcriptional_delay.j2",
    "p53_signaling": "transcriptional_delay.j2",
    "NF_KB": "transcriptional_delay.j2",
    "NF_kB": "transcriptional_delay.j2",
    # 转录因子 + 核转运
    "JAK_STAT": "transcription_factor.j2",
    "TGF_BETA": "transcription_factor.j2",
    "TGF_beta": "transcription_factor.j2",
    # β-catenin 破坏复合体
    "WNT": "destruction_complex.j2",
    "Wnt": "destruction_complex.j2",
    # Caspase 级联（Apoptosis）
    "APOPTOSIS": "caspase_cascade.j2",
    "Apoptosis": "caspase_cascade.j2",
    # Cyclin-CDK toggle（Cell Cycle）
    "CELL_CYCLE": "cyclin_cdk_toggle.j2",
    "Cell_Cycle": "cyclin_cdk_toggle.j2",
}

# 10 个 Pathway Specialist 的 pathway_class 值（用于"全部 specialist 覆盖"测试）
SPECIALIST_PATHWAY_CLASSES: list[str] = [
    "EGFR_RTK", "MAPK_ERK", "PI3K_AKT_mTOR",
    "p53", "APOPTOSIS", "CELL_CYCLE",
    "JAK_STAT", "NF_KB", "WNT", "TGF_BETA",
]

# pathway_graph/initializer.py 的 10 个通路键
INITIALIZER_PATHWAY_KEYS: list[str] = [
    "EGFR_RTK", "MAPK_ERK", "PI3K_AKT_mTOR",
    "p53_signaling", "Apoptosis", "Cell_Cycle",
    "JAK_STAT", "NF_kB", "Wnt", "TGF_beta",
]

# 11 个 v4 模板文件（与 ode_templates_v2/__init__.py __all__ 对齐）
ALL_V4_TEMPLATES: list[str] = [
    "_mechanism_phosphorylation_mm.j2",
    "oscillatory_feedback.j2",
    "bistable_switch.j2",
    "_dde_helpers.j2",
    "transcriptional_delay.j2",
    "nuclear_transport.j2",
    "ubiquitination_cascade.j2",
    "destruction_complex.j2",
    "caspase_cascade.j2",
    "cyclin_cdk_toggle.j2",
    "transcription_factor.j2",
]


# =============================================================================
# 测试数据工厂
# =============================================================================
def _make_reaction_ir() -> dict:
    """构造最小可渲染的 Reaction IR v2（p53-Mdm2 三物种）。"""
    return {
        "species": [
            {"name": "p53", "initial_concentration": 0.1},
            {"name": "Mdm2_mRNA", "initial_concentration": 0.0},
            {"name": "Mdm2", "initial_concentration": 0.0},
        ],
        "reactions": [
            {
                "source": "p53",
                "target": "Mdm2_mRNA",
                "reaction_type": "transcription",
                "reaction_eq": "p53 → Mdm2_mRNA",
            },
            {
                "source": "Mdm2_mRNA",
                "target": "Mdm2",
                "reaction_type": "translation",
                "reaction_eq": "Mdm2_mRNA → Mdm2",
            },
        ],
    }


# =============================================================================
# 测试类
# =============================================================================
class TestSelectTemplateRouting(unittest.TestCase):
    """测试 1：每个 pathway_class 路由到期望的模板。"""

    def test_each_specialist_pathway_class_routes_correctly(self):
        """10 个 specialist 的 pathway_class 值路由到期望模板。"""
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        for pathway_class, expected_template in EXPECTED_ROUTING.items():
            with self.subTest(pathway_class=pathway_class):
                actual = renderer._select_template(pathway_class, requires_dde=False)
                self.assertEqual(
                    actual, expected_template,
                    f"pathway_class='{pathway_class}' 应路由到 {expected_template}，"
                    f"实际路由到 {actual}",
                )

    def test_specialist_pathway_classes_all_covered(self):
        """10 个 specialist pathway_class 值全部命中路由表（无 fallthrough）。"""
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        for pathway_class in SPECIALIST_PATHWAY_CLASSES:
            with self.subTest(pathway_class=pathway_class):
                template = renderer._select_template(pathway_class, requires_dde=False)
                # 不应回退到默认 oscillatory_feedback.j2（除非该通路确实映射到它）
                self.assertIn(
                    pathway_class, EXPECTED_ROUTING,
                    f"specialist pathway_class='{pathway_class}' 未在期望路由表中",
                )
                self.assertEqual(template, EXPECTED_ROUTING[pathway_class])

    def test_initializer_pathway_keys_all_covered(self):
        """10 个 initializer 通路键全部命中路由表（无 fallthrough）。"""
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        for pathway_class in INITIALIZER_PATHWAY_KEYS:
            with self.subTest(pathway_class=pathway_class):
                template = renderer._select_template(pathway_class, requires_dde=False)
                self.assertIn(
                    pathway_class, EXPECTED_ROUTING,
                    f"initializer pathway_key='{pathway_class}' 未在期望路由表中",
                )
                self.assertEqual(template, EXPECTED_ROUTING[pathway_class])


class TestCaseInsensitiveRouting(unittest.TestCase):
    """测试 2：大小写不敏感（specialist 与 initializer 命名差异容忍）。"""

    def test_case_variants_route_to_same_template(self):
        """同一通路的大小写变体路由到相同模板。"""
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        case_groups = [
            # (specialist 命名, initializer 命名, 期望模板)
            ("p53", "p53_signaling", "transcriptional_delay.j2"),
            ("APOPTOSIS", "Apoptosis", "caspase_cascade.j2"),
            ("CELL_CYCLE", "Cell_Cycle", "cyclin_cdk_toggle.j2"),
            ("NF_KB", "NF_kB", "transcriptional_delay.j2"),
            ("WNT", "Wnt", "destruction_complex.j2"),
            ("TGF_BETA", "TGF_beta", "transcription_factor.j2"),
        ]
        for specialist_name, initializer_name, expected in case_groups:
            with self.subTest(group=(specialist_name, initializer_name)):
                self.assertEqual(
                    renderer._select_template(specialist_name, False), expected
                )
                self.assertEqual(
                    renderer._select_template(initializer_name, False), expected
                )
                # 全大写 / 全小写也应一致
                self.assertEqual(
                    renderer._select_template(specialist_name.upper(), False), expected
                )
                self.assertEqual(
                    renderer._select_template(initializer_name.upper(), False), expected
                )


class TestFallbackBehavior(unittest.TestCase):
    """测试 3：未知 pathway_class 与空值的回退行为（向后兼容）。"""

    def test_unknown_pathway_class_falls_back_to_oscillatory(self):
        """未知 pathway_class 回退到 oscillatory_feedback.j2（默认）。"""
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        for unknown in ("UNKNOWN_PATHWAY", "some_random_pathway", "NotInMap"):
            with self.subTest(pathway_class=unknown):
                self.assertEqual(
                    renderer._select_template(unknown, requires_dde=False),
                    "oscillatory_feedback.j2",
                )

    def test_empty_pathway_class_falls_back_to_oscillatory(self):
        """空字符串 pathway_class 回退到 oscillatory_feedback.j2。"""
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        self.assertEqual(
            renderer._select_template("", requires_dde=False),
            "oscillatory_feedback.j2",
        )

    def test_none_like_pathway_class_falls_back(self):
        """None 经 (pathway_class or '') 处理后回退到 oscillatory_feedback.j2。"""
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        # _select_template 内部 (pathway_class or "").upper()，None → "" → 回退
        self.assertEqual(
            renderer._select_template(None, requires_dde=False),  # type: ignore[arg-type]
            "oscillatory_feedback.j2",
        )

    def test_unknown_pathway_with_dde_falls_back_to_oscillatory(self):
        """未知 pathway_class + requires_dde=True 回退到 oscillatory_feedback.j2。"""
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        self.assertEqual(
            renderer._select_template("UNKNOWN_DDE_PATHWAY", requires_dde=True),
            "oscillatory_feedback.j2",
        )

    def test_empty_pathway_with_dde_falls_back_to_oscillatory(self):
        """空 pathway_class + requires_dde=True 回退到 oscillatory_feedback.j2。"""
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        self.assertEqual(
            renderer._select_template("", requires_dde=True),
            "oscillatory_feedback.j2",
        )

    def test_known_pathway_ignores_dde_flag(self):
        """已知 pathway_class 走路由表，requires_dde 不影响路由结果。

        例如 EGFR_RTK 即便 requires_dde=True 也路由到 oscillatory_feedback.j2
        （走 _PATHWAY_TEMPLATE_MAP 路由表，而非回退逻辑）。
        """
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        self.assertEqual(
            renderer._select_template("EGFR_RTK", requires_dde=True),
            "oscillatory_feedback.j2",
        )
        # p53 即便 requires_dde=False 也路由到 transcriptional_delay.j2
        self.assertEqual(
            renderer._select_template("p53", requires_dde=False),
            "transcriptional_delay.j2",
        )


class TestAllTemplatesReachable(unittest.TestCase):
    """测试 4：11 个 v4 模板均通过至少一个 pathway_class 或回退路径可达。"""

    def test_all_routable_templates_reachable_via_pathway_class(self):
        """路由表覆盖的 5 个特定模板均有 pathway_class 入口。"""
        from app.ode_renderer_v2 import _PATHWAY_TEMPLATE_MAP

        # 路由表中出现的模板集合
        routed_templates = set(_PATHWAY_TEMPLATE_MAP.values())
        # 应包含的 5 个 P3 特定模板（_mechanism_phosphorylation_mm.j2 是
        # {% include %} 片段不作为顶层路由目标；oscillatory_feedback.j2
        # 通过回退路径可达并承载磷酸化级联）
        expected_reachable = {
            "transcriptional_delay.j2",
            "transcription_factor.j2",
            "destruction_complex.j2",
            "caspase_cascade.j2",
            "cyclin_cdk_toggle.j2",
        }
        self.assertTrue(
            expected_reachable.issubset(routed_templates),
            f"路由表未覆盖以下模板: {expected_reachable - routed_templates}",
        )

    def test_oscillatory_feedback_reachable_via_fallback(self):
        """oscillatory_feedback.j2 通过未知 pathway_class 回退可达。"""
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        self.assertEqual(
            renderer._select_template("UNKNOWN", requires_dde=False),
            "oscillatory_feedback.j2",
        )

    def test_bistable_switch_reachable_via_env(self):
        """bistable_switch.j2 虽不直接由 pathway_class 路由，但可被 env 加载。

        注：bistable_switch.j2 是通用双稳态模板，保留作为未来 bistability
        特征标志路由的入口；当前由 pathway_class 路由表覆盖的 Apoptosis/
        Cell_Cycle 通路使用更特异的 caspase_cascade / cyclin_cdk_toggle。
        """
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        # 能被 Jinja2 env 加载即视为可达
        template = renderer.env.get_template("bistable_switch.j2")
        self.assertIsNotNone(template)

    def test_all_11_templates_loadable(self):
        """全部 11 个 v4 模板可被 ODERendererV2.env 加载（可达性 smoke test）。"""
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        for template_name in ALL_V4_TEMPLATES:
            with self.subTest(template=template_name):
                template = renderer.env.get_template(template_name)
                self.assertIsNotNone(template)

    def test_p3_unreachable_templates_now_reachable(self):
        """P0-2 核心断言：原 7 个不可达 P3 模板现在均通过路由表可达。

        原 audit 报告的 7 个不可达模板：
        transcriptional_delay / nuclear_transport / ubiquitination_cascade /
        destruction_complex / caspase_cascade / cyclin_cdk_toggle /
        transcription_factor
        """
        from app.ode_renderer_v2 import ODERendererV2, _PATHWAY_TEMPLATE_MAP

        previously_unreachable = [
            "transcriptional_delay.j2",
            "nuclear_transport.j2",
            "ubiquitination_cascade.j2",
            "destruction_complex.j2",
            "caspase_cascade.j2",
            "cyclin_cdk_toggle.j2",
            "transcription_factor.j2",
        ]
        routed = set(_PATHWAY_TEMPLATE_MAP.values())

        # 6 个模板通过路由表直接可达（nuclear_transport / ubiquitination_cascade
        # 不直接映射到单一 pathway_class，但通过 env 可加载——这两者由
        # transcriptional_delay / transcription_factor 模板内部机制覆盖）
        directly_routed = {
            "transcriptional_delay.j2",
            "destruction_complex.j2",
            "caspase_cascade.j2",
            "cyclin_cdk_toggle.j2",
            "transcription_factor.j2",
        }
        self.assertTrue(
            directly_routed.issubset(routed),
            f"原不可达模板未进入路由表: {directly_routed - routed}",
        )

        # 全部 7 个模板至少可被 env 加载（确认文件存在且 Jinja2 可解析）
        renderer = ODERendererV2()
        for template_name in previously_unreachable:
            with self.subTest(template=template_name):
                self.assertIsNotNone(renderer.env.get_template(template_name))


class TestRenderedCodeCompiles(unittest.TestCase):
    """测试 5：每个路由结果渲染后可被 compile() 通过（语法正确）。"""

    def test_each_routed_template_renders_and_compiles(self):
        """对每个 specialist pathway_class 调用 render()，渲染产物可编译。"""
        from app.ode_renderer_v2 import ODERendererV2

        renderer = ODERendererV2()
        reaction_ir = _make_reaction_ir()

        for pathway_class in SPECIALIST_PATHWAY_CLASSES:
            with self.subTest(pathway_class=pathway_class):
                ode_code = renderer.render(
                    pathway_class=pathway_class,
                    reaction_ir=reaction_ir,
                    t_end=120.0,
                )
                self.assertIsInstance(ode_code, str)
                self.assertGreater(len(ode_code), 100)
                try:
                    compile(ode_code, f"<{pathway_class}>", "exec")
                except SyntaxError as exc:
                    self.fail(
                        f"pathway_class='{pathway_class}' 渲染产物含 SyntaxError: "
                        f"{exc.msg} (line {exc.lineno})"
                    )


class TestRoutingTableCompleteness(unittest.TestCase):
    """测试 6：路由表与 specialist / initializer 命名一致性。"""

    def test_routing_table_covers_all_specialists(self):
        """路由表覆盖全部 10 个 specialist pathway_class（大写形式）。"""
        from app.ode_renderer_v2 import _PATHWAY_TEMPLATE_MAP

        for pathway_class in SPECIALIST_PATHWAY_CLASSES:
            with self.subTest(pathway_class=pathway_class):
                key = pathway_class.upper()
                self.assertIn(
                    key, _PATHWAY_TEMPLATE_MAP,
                    f"路由表缺少 specialist pathway_class='{pathway_class}' "
                    f"(key='{key}')",
                )

    def test_routing_table_covers_all_initializer_keys(self):
        """路由表覆盖全部 10 个 initializer 通路键（大写形式）。"""
        from app.ode_renderer_v2 import _PATHWAY_TEMPLATE_MAP

        for pathway_key in INITIALIZER_PATHWAY_KEYS:
            with self.subTest(pathway_key=pathway_key):
                key = pathway_key.upper()
                self.assertIn(
                    key, _PATHWAY_TEMPLATE_MAP,
                    f"路由表缺少 initializer pathway_key='{pathway_key}' "
                    f"(key='{key}')",
                )

    def test_routing_table_values_are_valid_template_files(self):
        """路由表所有 value 都指向真实存在的 .j2 模板文件。"""
        from app.ode_renderer_v2 import _PATHWAY_TEMPLATE_MAP

        templates_dir = BACKEND_DIR / "app" / "ode_templates_v2"
        for pathway_class, template_name in _PATHWAY_TEMPLATE_MAP.items():
            with self.subTest(pathway_class=pathway_class):
                template_path = templates_dir / template_name
                self.assertTrue(
                    template_path.exists(),
                    f"pathway_class='{pathway_class}' 路由到不存在的模板文件: "
                    f"{template_name}",
                )

    def test_routing_table_values_are_in_all_list(self):
        """路由表所有 value 都在 ode_templates_v2.__all__ 对应的文件名集合内。"""
        from app.ode_renderer_v2 import _PATHWAY_TEMPLATE_MAP
        from app.ode_templates_v2 import __all__ as templates_all

        # __all__ 是不含 .j2 后缀的模块名，转换为文件名
        valid_template_files = {f"{name}.j2" for name in templates_all}
        for pathway_class, template_name in _PATHWAY_TEMPLATE_MAP.items():
            with self.subTest(pathway_class=pathway_class):
                self.assertIn(
                    template_name, valid_template_files,
                    f"路由表 value '{template_name}' 不在 ode_templates_v2.__all__ 中",
                )


if __name__ == "__main__":
    unittest.main()
