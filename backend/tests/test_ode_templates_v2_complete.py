# BioDynamics Agent v4 - ODE Templates v2 完整性测试（P3 模板补全）
#
# 对应 spec.md Part 5 + Migration Plan §Phase 3：验证 7 个新 ODE 模板的
# 渲染、编译、结构完整性与 detect_* 函数行为。
#
# 测试用例：
#   1. 文件存在性：7 个新模板 + 4 个基础模板均存在
#   2. __init__.py __all__ 完整性：包含全部 11 个模板名
#   3. Jinja2 渲染：每个新模板可通过 ODERendererV2.env 渲染为非空字符串
#   4. 渲染产物为合法 Python：compile() 不抛 SyntaxError
#   5. 渲染产物结构完整：含 SPECIES_NAMES / _ode_rhs / detect_* / __main__
#   6. detect_* 函数行为：使用 mock 时序数据验证返回结构正确
#
# 运行：cd backend && python -m pytest tests/test_ode_templates_v2_complete.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# 常量：7 个新模板 + 4 个基础模板
# =============================================================================
_V4_TEMPLATES_DIR = BACKEND_DIR / "app" / "ode_templates_v2"

_BASE_TEMPLATES = [
    "_mechanism_phosphorylation_mm.j2",
    "oscillatory_feedback.j2",
    "bistable_switch.j2",
    "_dde_helpers.j2",
]

_NEW_TEMPLATES = [
    "transcriptional_delay.j2",
    "nuclear_transport.j2",
    "ubiquitination_cascade.j2",
    "destruction_complex.j2",
    "caspase_cascade.j2",
    "cyclin_cdk_toggle.j2",
    "transcription_factor.j2",
]

_ALL_TEMPLATES = _BASE_TEMPLATES + _NEW_TEMPLATES

# 每个新模板的 detect_* 函数名
_DETECT_FUNCTIONS = {
    "transcriptional_delay.j2": "detect_delay_effect",
    "nuclear_transport.j2": "detect_nuclear_ratio",
    "ubiquitination_cascade.j2": "detect_ubiquitination_cascade",
    "destruction_complex.j2": "detect_bcatenin_steady_state",
    "caspase_cascade.j2": "detect_momp_and_bistability",
    "cyclin_cdk_toggle.j2": "detect_toggle_and_oscillation",
    "transcription_factor.j2": "detect_tf_activation",
}


# =============================================================================
# 测试数据工厂
# =============================================================================
def _make_render_kwargs(**overrides):
    """构造 Jinja2 渲染参数（与 ODERendererV2.render 一致）。"""
    kwargs = {
        "species_names": ["p53", "Mdm2_mRNA", "Mdm2"],
        "y0": [0.1, 0.0, 0.0],
        "edges_json": [
            {
                "source": "p53",
                "target": "Mdm2_mRNA",
                "mechanism": "transcription",
                "reaction_eq": "p53 → Mdm2_mRNA",
                "sbo_term": None,
            },
            {
                "source": "Mdm2_mRNA",
                "target": "Mdm2",
                "mechanism": "translation",
                "reaction_eq": "Mdm2_mRNA → Mdm2",
                "sbo_term": None,
            },
        ],
        "params_json": {
            "Mdm2_mRNA": {"k_trans": 0.1, "k_mRNA_deg": 0.01, "n_hill": 2.0, "K_d": 0.5},
            "Mdm2": {"k_transl": 0.05, "k_prot_deg": 0.005},
        },
        "t_end": 360.0,
        "n_eval": 200,
        "dde_delay_minutes": 60.0,
        "requires_dde": True,
        "pathway_class": "p53_signaling",
    }
    kwargs.update(overrides)
    return kwargs


def _make_jinja_env():
    """构造 Jinja2 Environment（与 ODERendererV2 相同的配置）。"""
    from jinja2 import Environment, FileSystemLoader
    return Environment(
        loader=FileSystemLoader(str(_V4_TEMPLATES_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


# =============================================================================
# 测试类
# =============================================================================
class TestTemplateFileExistence(unittest.TestCase):
    """测试 1：所有 11 个模板文件存在。"""

    def test_base_templates_exist(self):
        """4 个基础模板文件存在。"""
        for name in _BASE_TEMPLATES:
            path = _V4_TEMPLATES_DIR / name
            self.assertTrue(
                path.exists(),
                f"基础模板缺失: {name}",
            )

    def test_new_templates_exist(self):
        """7 个新模板文件存在（P3 补全）。"""
        for name in _NEW_TEMPLATES:
            path = _V4_TEMPLATES_DIR / name
            self.assertTrue(
                path.exists(),
                f"P3 新模板缺失: {name}",
            )

    def test_all_templates_count(self):
        """ode_templates_v2 目录下至少 11 个 .j2 文件。"""
        j2_files = list(_V4_TEMPLATES_DIR.glob("*.j2"))
        self.assertGreaterEqual(
            len(j2_files),
            11,
            f"应有至少 11 个 .j2 文件，实际 {len(j2_files)}",
        )


class TestInitAllCompleteness(unittest.TestCase):
    """测试 2：__init__.py __all__ 包含全部 11 个模板名。"""

    def test_all_list_contains_base_templates(self):
        """__all__ 包含 4 个基础模板名。"""
        from app.ode_templates_v2 import __all__ as templates_all
        for name in ["_mechanism_phosphorylation_mm", "oscillatory_feedback",
                     "bistable_switch", "_dde_helpers"]:
            self.assertIn(name, templates_all, f"__all__ 缺少基础模板: {name}")

    def test_all_list_contains_new_templates(self):
        """__all__ 包含 7 个新模板名（P3 补全）。"""
        from app.ode_templates_v2 import __all__ as templates_all
        for name in ["transcriptional_delay", "nuclear_transport",
                     "ubiquitination_cascade", "destruction_complex",
                     "caspase_cascade", "cyclin_cdk_toggle",
                     "transcription_factor"]:
            self.assertIn(name, templates_all, f"__all__ 缺少新模板: {name}")

    def test_all_list_total_count(self):
        """__all__ 至少含 11 个模板名。"""
        from app.ode_templates_v2 import __all__ as templates_all
        self.assertGreaterEqual(
            len(templates_all),
            11,
            f"__all__ 应含至少 11 个模板名，实际 {len(templates_all)}",
        )


class TestTemplateRendering(unittest.TestCase):
    """测试 3：每个新模板可通过 Jinja2 渲染为非空字符串。"""

    def test_each_new_template_renders_nonempty(self):
        """7 个新模板均可渲染为非空字符串。"""
        env = _make_jinja_env()
        kwargs = _make_render_kwargs()

        for template_name in _NEW_TEMPLATES:
            with self.subTest(template=template_name):
                template = env.get_template(template_name)
                rendered = template.render(**kwargs)
                self.assertIsInstance(rendered, str, f"{template_name} 渲染产物应为 str")
                self.assertGreater(
                    len(rendered),
                    100,
                    f"{template_name} 渲染产物过短（<100 字符），可能渲染失败",
                )

    def test_render_includes_species_names(self):
        """渲染产物含 SPECIES_NAMES 与物种名。"""
        env = _make_jinja_env()
        kwargs = _make_render_kwargs()

        for template_name in _NEW_TEMPLATES:
            with self.subTest(template=template_name):
                rendered = env.get_template(template_name).render(**kwargs)
                self.assertIn("SPECIES_NAMES", rendered)
                self.assertIn("p53", rendered)
                self.assertIn("Mdm2_mRNA", rendered)

    def test_render_includes_t_end(self):
        """渲染产物含 T_END = 360.0。"""
        env = _make_jinja_env()
        kwargs = _make_render_kwargs(t_end=360.0)

        for template_name in _NEW_TEMPLATES:
            with self.subTest(template=template_name):
                rendered = env.get_template(template_name).render(**kwargs)
                self.assertIn("T_END", rendered)
                self.assertIn("360", rendered)


class TestRenderedCodeValidity(unittest.TestCase):
    """测试 4：渲染产物为合法 Python（compile() 不抛 SyntaxError）。"""

    def test_each_new_template_compiles(self):
        """7 个新模板的渲染产物均可通过 compile()。"""
        env = _make_jinja_env()
        kwargs = _make_render_kwargs()

        for template_name in _NEW_TEMPLATES:
            with self.subTest(template=template_name):
                rendered = env.get_template(template_name).render(**kwargs)
                try:
                    compile(rendered, f"<{template_name}>", "exec")
                except SyntaxError as e:
                    self.fail(
                        f"{template_name} 渲染产物含 SyntaxError: {e.msg} "
                        f"(line {e.lineno})"
                    )


class TestRenderedCodeStructure(unittest.TestCase):
    """测试 5：渲染产物结构完整（含关键函数与 __main__ 块）。"""

    def test_rendered_code_has_ode_rhs(self):
        """每个新模板渲染后含 _ode_rhs 函数定义。"""
        env = _make_jinja_env()
        kwargs = _make_render_kwargs()

        for template_name in _NEW_TEMPLATES:
            with self.subTest(template=template_name):
                rendered = env.get_template(template_name).render(**kwargs)
                self.assertIn("def _ode_rhs", rendered,
                              f"{template_name} 缺少 _ode_rhs 函数")

    def test_rendered_code_has_detect_function(self):
        """每个新模板渲染后含对应的 detect_* 函数。"""
        env = _make_jinja_env()
        kwargs = _make_render_kwargs()

        for template_name, detect_fn in _DETECT_FUNCTIONS.items():
            with self.subTest(template=template_name):
                rendered = env.get_template(template_name).render(**kwargs)
                self.assertIn(
                    f"def {detect_fn}",
                    rendered,
                    f"{template_name} 缺少 {detect_fn} 函数",
                )

    def test_rendered_code_has_main_block(self):
        """每个新模板渲染后含 if __name__ == '__main__' 块。"""
        env = _make_jinja_env()
        kwargs = _make_render_kwargs()

        for template_name in _NEW_TEMPLATES:
            with self.subTest(template=template_name):
                rendered = env.get_template(template_name).render(**kwargs)
                self.assertIn('if __name__ == "__main__"', rendered,
                              f"{template_name} 缺少 __main__ 块")

    def test_rendered_code_has_solve_ivp(self):
        """每个新模板渲染后含 solve_ivp 调用（ODE 求解器）。"""
        env = _make_jinja_env()
        kwargs = _make_render_kwargs()

        for template_name in _NEW_TEMPLATES:
            with self.subTest(template=template_name):
                rendered = env.get_template(template_name).render(**kwargs)
                self.assertIn("solve_ivp", rendered,
                              f"{template_name} 缺少 solve_ivp 调用")


class TestDetectFunctions(unittest.TestCase):
    """测试 6：detect_* 函数行为（使用 mock 时序数据）。

    通过 exec() 加载渲染后的代码（不执行 __main__），然后调用 detect_*
    函数验证返回结构。
    """

    def _exec_template(self, template_name, kwargs):
        """渲染并 exec 模板代码，返回命名空间 dict。

        注意：__main__ 块不会执行（因为 __name__ != '__main__'）。
        """
        env = _make_jinja_env()
        rendered = env.get_template(template_name).render(**kwargs)
        namespace = {}
        exec(compile(rendered, f"<{template_name}>", "exec"), namespace)
        return namespace

    def test_detect_delay_effect_returns_dict(self):
        """transcriptional_delay.j2 的 detect_delay_effect 返回 dict。"""
        import numpy as np
        kwargs = _make_render_kwargs(
            species_names=["p53", "Mdm2_mRNA", "Mdm2"],
        )
        ns = self._exec_template("transcriptional_delay.j2", kwargs)

        t_arr = np.linspace(0, 360, 200)
        y_arr = np.zeros((200, 3))
        y_arr[:, 0] = np.exp(-t_arr / 100) * 0.5  # p53 早期峰值
        y_arr[:, 1] = np.exp(-(t_arr - 60) / 100) * 0.3  # Mdm2_mRNA 延迟峰值
        y_arr[:, 2] = 0.0

        result = ns["detect_delay_effect"](t_arr, y_arr)
        self.assertIsInstance(result, dict)
        self.assertIn("delay_effective", result)
        self.assertIn("tf_peak_time", result)
        self.assertIn("mrna_peak_time", result)

    def test_detect_nuclear_ratio_returns_dict(self):
        """nuclear_transport.j2 的 detect_nuclear_ratio 返回 dict。"""
        import numpy as np
        kwargs = _make_render_kwargs(
            species_names=["STAT1_cyto", "STAT1_nuc", "STAT1_mRNA"],
        )
        ns = self._exec_template("nuclear_transport.j2", kwargs)

        t_arr = np.linspace(0, 120, 100)
        y_arr = np.zeros((100, 3))
        y_arr[:, 0] = 0.5  # cyto
        y_arr[:, 1] = 1.5  # nuc (核占优)
        y_arr[:, 2] = 0.1

        result = ns["detect_nuclear_ratio"](t_arr, y_arr)
        self.assertIsInstance(result, dict)
        self.assertIn("has_nuclear_transport", result)
        self.assertIn("ratios", result)
        self.assertTrue(result["has_nuclear_transport"])

    def test_detect_ubiquitination_cascade_returns_dict(self):
        """ubiquitination_cascade.j2 的 detect_ubiquitination_cascade 返回 dict。"""
        import numpy as np
        kwargs = _make_render_kwargs(
            species_names=["p53", "Ub_p53", "Mdm2"],
        )
        ns = self._exec_template("ubiquitination_cascade.j2", kwargs)

        t_arr = np.linspace(0, 120, 100)
        y_arr = np.zeros((100, 3))
        y_arr[:, 0] = np.exp(-t_arr / 50)  # p53 下降
        y_arr[:, 1] = 1.0 - np.exp(-t_arr / 50)  # Ub_p53 上升
        y_arr[:, 2] = 0.1

        result = ns["detect_ubiquitination_cascade"](t_arr, y_arr)
        self.assertIsInstance(result, dict)
        self.assertIn("has_ubiquitination", result)
        self.assertTrue(result["has_ubiquitination"])
        self.assertIn("Ub_p53", result["ub_species"])

    def test_detect_bcatenin_steady_state_returns_dict(self):
        """destruction_complex.j2 的 detect_bcatenin_steady_state 返回 dict。"""
        import numpy as np
        kwargs = _make_render_kwargs(
            species_names=["Axin", "beta_catenin", "APC"],
        )
        ns = self._exec_template("destruction_complex.j2", kwargs)

        t_arr = np.linspace(0, 240, 100)
        y_arr = np.zeros((100, 3))
        y_arr[:, 0] = 0.5  # Axin
        y_arr[:, 1] = 5.0  # beta_catenin < 10 nM threshold
        y_arr[:, 2] = 0.3  # APC

        result = ns["detect_bcatenin_steady_state"](t_arr, y_arr)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["bcatenin_detected"])
        self.assertIn("steady_state_nM", result)
        self.assertIn("below_threshold", result)

    def test_detect_momp_and_bistability_returns_dict(self):
        """caspase_cascade.j2 的 detect_momp_and_bistability 返回 dict。"""
        import numpy as np
        kwargs = _make_render_kwargs(
            species_names=["Caspase3_active", "Cyt_c", "Bax"],
        )
        ns = self._exec_template("caspase_cascade.j2", kwargs)

        t_arr = np.linspace(0, 120, 100)
        y_arr = np.zeros((100, 3))
        y_arr[:, 0] = 1.0 - np.exp(-t_arr / 20)  # Caspase3 上升
        y_arr[:, 1] = np.where(t_arr > 5, 1.0, 0.0)  # Cyt c 早期释放
        y_arr[:, 2] = 0.1

        result = ns["detect_momp_and_bistability"](t_arr, y_arr)
        self.assertIsInstance(result, dict)
        self.assertIn("momp_detected", result)
        self.assertIn("bistable", result)
        self.assertIn("final_state", result)

    def test_detect_toggle_and_oscillation_returns_dict(self):
        """cyclin_cdk_toggle.j2 的 detect_toggle_and_oscillation 返回 dict。"""
        import numpy as np
        kwargs = _make_render_kwargs(
            species_names=["E2F", "CyclinB", "Rb"],
        )
        ns = self._exec_template("cyclin_cdk_toggle.j2", kwargs)

        t_arr = np.linspace(0, 480, 500)
        y_arr = np.zeros((500, 3))
        y_arr[:, 0] = 1.0  # E2F 高态（ON）
        y_arr[:, 1] = 0.5 + 0.5 * np.sin(t_arr * 0.05)  # CyclinB 振荡
        y_arr[:, 2] = 0.0

        result = ns["detect_toggle_and_oscillation"](t_arr, y_arr)
        self.assertIsInstance(result, dict)
        self.assertIn("toggle_detected", result)
        self.assertIn("oscillation_detected", result)

    def test_detect_tf_activation_returns_dict(self):
        """transcription_factor.j2 的 detect_tf_activation 返回 dict。"""
        import numpy as np
        kwargs = _make_render_kwargs(
            species_names=["pSTAT1", "STAT1_mRNA", "STAT1"],
        )
        ns = self._exec_template("transcription_factor.j2", kwargs)

        t_arr = np.linspace(0, 120, 100)
        y_arr = np.zeros((100, 3))
        # pSTAT1 在 10 min 达峰
        y_arr[:, 0] = np.exp(-((t_arr - 10) / 5) ** 2)
        # mRNA 在 45 min 达峰
        y_arr[:, 1] = np.exp(-((t_arr - 45) / 10) ** 2)
        y_arr[:, 2] = 0.1

        result = ns["detect_tf_activation"](t_arr, y_arr)
        self.assertIsInstance(result, dict)
        self.assertIn("tf_active_detected", result)
        self.assertIn("mrna_detected", result)
        self.assertTrue(result["tf_active_detected"])
        self.assertTrue(result["mrna_detected"])


class TestDdeDegradationInTemplates(unittest.TestCase):
    """补充测试：含 DDE 需求的模板支持 jitcdde 降级。"""

    def test_transcriptional_delay_dde_flag_true(self):
        """transcriptional_delay.j2 在 requires_dde=True 时含 DDE 相关代码。"""
        env = _make_jinja_env()
        kwargs = _make_render_kwargs(requires_dde=True)
        rendered = env.get_template("transcriptional_delay.j2").render(**kwargs)
        # 渲染后应含 DDE_DELAY 或 jitcdde 引用
        self.assertIn("DDE_DELAY", rendered)
        self.assertIn("REQUIRES_DDE", rendered)

    def test_transcriptional_delay_dde_flag_false(self):
        """transcriptional_delay.j2 在 requires_dde=False 时仍可渲染。"""
        env = _make_jinja_env()
        kwargs = _make_render_kwargs(requires_dde=False, dde_delay_minutes=0.0)
        rendered = env.get_template("transcriptional_delay.j2").render(**kwargs)
        self.assertIn("REQUIRES_DDE", rendered)
        # 应可编译
        compile(rendered, "<transcriptional_delay>", "exec")

    def test_all_new_templates_compile_without_dde(self):
        """所有 7 个新模板在 requires_dde=False 时均可编译。"""
        env = _make_jinja_env()
        kwargs = _make_render_kwargs(requires_dde=False, dde_delay_minutes=0.0)

        for template_name in _NEW_TEMPLATES:
            with self.subTest(template=template_name):
                rendered = env.get_template(template_name).render(**kwargs)
                try:
                    compile(rendered, f"<{template_name}>", "exec")
                except SyntaxError as e:
                    self.fail(
                        f"{template_name} 在 DDE=False 时含 SyntaxError: {e.msg}"
                    )


class TestODERendererV2Integration(unittest.TestCase):
    """补充测试：ODERendererV2 可加载所有 11 个模板。"""

    def test_renderer_env_can_load_all_templates(self):
        """ODERendererV2 的 Jinja2 env 可加载全部 11 个模板。"""
        from app.ode_renderer_v2 import ODERendererV2
        renderer = ODERendererV2()

        for template_name in _ALL_TEMPLATES:
            with self.subTest(template=template_name):
                # get_template 在模板不存在时抛 TemplateNotFound
                template = renderer.env.get_template(template_name)
                self.assertIsNotNone(template)

    def test_v4_ode_available_flag(self):
        """is_v4_ode_available 返回 bool（依赖 feature flag）。"""
        from app.ode_renderer_v2 import is_v4_ode_available
        result = is_v4_ode_available()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
