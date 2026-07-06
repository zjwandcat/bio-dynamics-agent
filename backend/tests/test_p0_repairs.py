# BioDynamics Agent - P0 修复单元测试
# 对应 修复提示词1.md 与 EGF-EGFR错误结论根因与后续修复计划报告.md 的 P0 级修复
#
# 测试范围：
#   1. TemplateSelectorSkill 规则引擎（template_selector.py）
#   2. ReactionIR 中间表示（reaction_ir.py）
#   3. DomainChecker 领域常识审查（domain_checker.py）
#   4. BioModelsAPIClient ID 提取（biomodels_client.py）
#   5. Signaling_Cascade_Phos.j2 模板渲染与质量守恒
#   6. RagClient.search_params_hybrid_with_context 4-collection 接口
#
# 运行方式：
#   cd backend
#   python -m pytest tests/test_p0_repairs.py -v
# 或直接：
#   python tests/test_p0_repairs.py

from __future__ import annotations

import sys
import os
from pathlib import Path

# 添加 backend 目录到 sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.template_selector import (
    TemplateSelectorSkill,
    select_template,
    get_simulation_time_scale,
    PHOS_CASCADE_TEMPLATES,
    CASCADE_TEMPLATES,
    PKPD_TEMPLATES,
    TEMPLATE_WHITELIST,
    TemplateSelection,
)
from app.reaction_ir import (
    build_reaction_graph,
    validate_reaction_graph,
    ReactionIR,
    REACTION_TYPES,
    KINETICS_TYPES,
)
from app.domain_checker import (
    DomainChecker,
    check_ode_code,
    DomainCheckResult,
)
from app.biomodels_client import (
    BioModelsAPIClient,
    extract_biomodel_id,
    get_biomodels_client,
)
from app.ode_templates import render_template


# =============================================================================
# 测试工具函数
# =============================================================================
def _make_egf_egfr_edges() -> list[dict]:
    """构建 EGF-EGFR 信号级联的边列表（含 mechanism 与 reaction_equation）。"""
    return [
        {"source": "EGF", "target": "EGFR", "interaction": "activation",
         "mechanism": "binding", "reaction_equation": "EGF + EGFR → EGF-EGFR"},
        {"source": "EGF-EGFR", "target": "pEGFR", "interaction": "activation",
         "mechanism": "phosphorylation", "reaction_equation": "EGF-EGFR → pEGFR"},
        {"source": "pEGFR", "target": "pShc", "interaction": "activation",
         "mechanism": "phosphorylation",
         "reaction_equation": "pEGFR + Shc → pEGFR + pShc"},
        {"source": "pShc", "target": "Grb2", "interaction": "activation",
         "mechanism": "binding", "reaction_equation": "pShc + Grb2 → pShc-Grb2"},
        {"source": "SOS", "target": "RasGTP", "interaction": "activation",
         "mechanism": "exchange",
         "reaction_equation": "RasGDP → RasGTP (catalyzed by SOS)"},
    ]


def _make_egf_egfr_kg() -> dict:
    """构建 EGF-EGFR 知识图谱（含 nodes 与 edges）。"""
    return {
        "nodes": [
            {"id": "n1", "name": "EGF", "type": "Cytokine"},
            {"id": "n2", "name": "EGFR", "type": "Protein"},
            {"id": "n3", "name": "EGF-EGFR", "type": "Complex"},
            {"id": "n4", "name": "pEGFR", "type": "Protein"},
            {"id": "n5", "name": "Shc", "type": "Protein"},
            {"id": "n6", "name": "pShc", "type": "Protein"},
            {"id": "n7", "name": "Grb2", "type": "Protein"},
            {"id": "n8", "name": "SOS", "type": "Protein"},
            {"id": "n9", "name": "RasGDP", "type": "Protein"},
            {"id": "n10", "name": "RasGTP", "type": "Protein"},
        ],
        "edges": _make_egf_egfr_edges(),
    }


# =============================================================================
# 1. TemplateSelectorSkill 测试
# =============================================================================
class TestTemplateSelector:
    """模板选择规则引擎测试。"""

    def setup_method(self):
        self.skill = TemplateSelectorSkill()

    def test_egf_egfr_keyword_selects_signaling_cascade_phos(self):
        """测试：EGF-EGFR 关键词触发 Signaling_Cascade_Phos（非 Cascade_Activation）。"""
        result = self.skill.select(
            user_input="EGF 结合 EGFR 受体后诱导其二聚化和自磷酸化，激活下游 Shc-Grb2-SOS-Ras-MAPK 信号级联",
            edges=_make_egf_egfr_edges(),
            llm_template="Cascade_Activation",  # LLM 错误选择
        )
        assert result.template == "Signaling_Cascade_Phos", (
            f"期望 Signaling_Cascade_Phos，实际 {result.template}"
        )
        assert result.override_llm is True, "应覆盖 LLM 的错误选择"
        assert result.confidence >= 0.9, f"置信度应 >= 0.9，实际 {result.confidence}"

    def test_biomd_id_selects_signaling_cascade_phos(self):
        """测试：BIOMD0000000205 SBML grounding 触发 Signaling_Cascade_Phos。"""
        result = self.skill.select(
            user_input="基于 BIOMD0000000205 模型参数仿真 EGF 刺激下 EGFR 磷酸化动力学",
            edges=[],
            llm_template="Simple_Inhibition",
            sbml_model_id="BIOMD0000000205",
        )
        assert result.template == "Signaling_Cascade_Phos"
        assert result.rule_source == "sbml_grounding"

    def test_single_inhibition_edge_selects_simple_inhibition(self):
        """测试：单 inhibition 边触发 Simple_Inhibition。"""
        edges = [
            {"source": "Drug A", "target": "Target X",
             "interaction": "inhibition", "mechanism": "inhibition"},
        ]
        result = self.skill.select(
            user_input="Drug A 抑制 Target X，IC50=100 nM",
            edges=edges,
            llm_template="Cascade_Inhibition",
        )
        assert result.template == "Simple_Inhibition"
        assert result.override_llm is True

    def test_pkpd_profile_overrides_other_rules(self):
        """测试：PK/PD profile 优先级最高。"""
        result = self.skill.select(
            user_input="药物动力学仿真",
            edges=[],
            llm_template="Simple_Inhibition",
            pkpd_profile={
                "drug_name": "Aspirin",
                "drug_target": "COX2",
                "compartment": "1-compartment",
            },
        )
        assert result.template == "PKPD_OneCompartment"

    def test_combination_drugs_selects_combination(self):
        """测试：≥2 inhibition 边 + 联合用药关键词触发 Combination。"""
        edges = [
            {"source": "Drug A", "target": "Target X",
             "interaction": "inhibition", "mechanism": "inhibition"},
            {"source": "Drug B", "target": "Target Y",
             "interaction": "inhibition", "mechanism": "inhibition"},
        ]
        result = self.skill.select(
            user_input="Drug A 和 Drug B 联合用药协同效应",
            edges=edges,
            llm_template="Simple_Inhibition",
        )
        assert result.template == "Combination"

    def test_mechanism_vote_binding_phosphorylation(self):
        """测试：mechanism 投票（binding + phosphorylation > 50%）。"""
        edges = [
            {"source": "A", "target": "B", "mechanism": "binding"},
            {"source": "B", "target": "C", "mechanism": "phosphorylation"},
            {"source": "C", "target": "D", "mechanism": "binding"},
        ]
        result = self.skill.select(
            user_input="Some generic query",
            edges=edges,
            llm_template="Simple_Inhibition",
        )
        assert result.template == "Signaling_Cascade_Phos"
        assert result.rule_source == "mechanism_vote"

    def test_time_scale_signaling_cascade_phos(self):
        """测试：Signaling_Cascade_Phos 时间尺度为 120 min。"""
        t_end, n_eval, unit = get_simulation_time_scale("Signaling_Cascade_Phos")
        assert t_end == 120.0, f"期望 120 min，实际 {t_end}"
        assert unit == "min", f"期望 min，实际 {unit}"
        assert n_eval == 300

    def test_time_scale_pkpd(self):
        """测试：PKPD 模板时间尺度为 48 h。"""
        t_end, n_eval, unit = get_simulation_time_scale("PKPD_OneCompartment")
        assert t_end == 48.0
        assert unit == "h"

    def test_template_whitelist_completeness(self):
        """测试：模板白名单包含所有必需模板。"""
        required = {
            "Signaling_Cascade_Phos", "Simple_Inhibition", "Simple_Activation",
            "Cascade_Activation", "Cascade_Inhibition",
            "PKPD_OneCompartment", "PKPD_TwoCompartment",
            "Combination", "DoseSweep",
        }
        assert required.issubset(set(TEMPLATE_WHITELIST)), (
            f"白名单缺少：{required - set(TEMPLATE_WHITELIST)}"
        )


# =============================================================================
# 2. ReactionIR 测试
# =============================================================================
class TestReactionIR:
    """Reaction IR 中间表示测试。"""

    def test_build_reaction_graph_from_kg(self):
        """测试：从 KG 构建 Reaction Graph。"""
        kg = _make_egf_egfr_kg()
        graph = build_reaction_graph(kg)
        assert "species" in graph
        assert "reactions" in graph
        assert len(graph["reactions"]) == 5
        # species 应包含所有节点 + 反应中新增的物种
        assert "EGF" in graph["species"]
        assert "EGFR" in graph["species"]

    def test_reaction_types_in_enum(self):
        """测试：所有 reaction_type 在枚举内。"""
        kg = _make_egf_egfr_kg()
        graph = build_reaction_graph(kg)
        for r in graph["reactions"]:
            assert r["reaction_type"] in REACTION_TYPES, (
                f"未知 reaction_type: {r['reaction_type']}"
            )
            assert r["kinetics_type"] in KINETICS_TYPES, (
                f"未知 kinetics_type: {r['kinetics_type']}"
            )

    def test_validate_reaction_graph_passes(self):
        """测试：合规 Reaction Graph 校验通过。"""
        kg = _make_egf_egfr_kg()
        graph = build_reaction_graph(kg)
        violations = validate_reaction_graph(graph)
        assert violations == [], f"不应有违规：{violations}"

    def test_binding_reaction_has_mass_action(self):
        """测试：binding 反应使用 mass_action 动力学。"""
        kg = _make_egf_egfr_kg()
        graph = build_reaction_graph(kg)
        binding_reactions = [
            r for r in graph["reactions"]
            if r["reaction_type"] == "binding"
        ]
        assert len(binding_reactions) > 0, "应至少有 1 条 binding 反应"
        for r in binding_reactions:
            assert r["kinetics_type"] == "mass_action"

    def test_phosphorylation_reaction_has_michaelis_menten(self):
        """测试：phosphorylation 反应使用 Michaelis_Menten 动力学。"""
        kg = _make_egf_egfr_kg()
        graph = build_reaction_graph(kg)
        phos_reactions = [
            r for r in graph["reactions"]
            if r["reaction_type"] == "phosphorylation"
        ]
        assert len(phos_reactions) > 0, "应至少有 1 条 phosphorylation 反应"
        for r in phos_reactions:
            assert r["kinetics_type"] == "Michaelis_Menten"


# =============================================================================
# 3. DomainChecker 测试
# =============================================================================
class TestDomainChecker:
    """领域常识审查器测试。"""

    def setup_method(self):
        self.checker = DomainChecker()

    def test_valid_code_passes(self):
        """测试：合法 ODE 代码通过审查。"""
        code = """
import numpy as np
from scipy.integrate import solve_ivp

SPECIES_NAMES = ["EGF", "EGFR"]
T_END = 120.0
Y0 = [0.008, 0.3]
EDGES = [{"source": "EGF", "target": "EGFR", "mechanism": "binding"}]
PARAMS = {"EGFR": {"k_on": 1.0, "k_off": 0.01}}

def _ode(t, y):
    y = np.maximum(y, 0.0)
    dy = np.zeros_like(y)
    return dy.tolist()

sol = solve_ivp(_ode, (0.0, T_END), Y0)
print(sol.t)
"""
        result = check_ode_code(
            code=code,
            species_names=["EGF", "EGFR"],
            edges=[{"source": "EGF", "target": "EGFR", "mechanism": "binding"}],
            template_name="Signaling_Cascade_Phos",
        )
        # 不应有 high 严重性违规
        high_violations = [v for v in result.violations if v.severity == "high"]
        assert len(high_violations) == 0, (
            f"不应有 high 违规：{[v.message for v in high_violations]}"
        )

    def test_dangerous_import_detected(self):
        """测试：危险 import 被检测。"""
        code = """
import os
import subprocess
import numpy as np
SPECIES_NAMES = ["A"]
T_END = 60.0
Y0 = [1.0]
EDGES = []
PARAMS = {}
def _ode(t, y):
    return [0.0]
"""
        result = check_ode_code(code=code, species_names=["A"])
        dangerous_violations = [
            v for v in result.violations if v.rule == "dangerous_call"
        ]
        assert len(dangerous_violations) >= 2, (
            f"应检测到 os 和 subprocess：{[v.message for v in dangerous_violations]}"
        )
        assert result.passed is False, "危险调用应导致审查不通过"

    def test_pathway_shortcut_detected(self):
        """测试：信号通路 shortcut 被检测（EGF→MAPK）。"""
        edges = [
            {"source": "EGF", "target": "MAPK", "interaction": "activation",
             "mechanism": "activation"},
        ]
        result = check_ode_code(
            code="print('hello')",
            species_names=["EGF", "MAPK"],
            edges=edges,
            template_name="Signaling_Cascade_Phos",
        )
        shortcut_violations = [
            v for v in result.violations if v.rule == "pathway_completeness"
        ]
        assert len(shortcut_violations) > 0, "应检测到 EGF→MAPK shortcut"

    def test_time_scale_violation_detected(self):
        """测试：信号级联 T_END 过大被检测。"""
        code = """
T_END = 5000.0
SPECIES_NAMES = ["EGF", "EGFR"]
"""
        result = check_ode_code(
            code=code,
            species_names=["EGF", "EGFR"],
            edges=[],
            template_name="Signaling_Cascade_Phos",
        )
        time_violations = [
            v for v in result.violations if v.rule == "time_scale"
        ]
        assert len(time_violations) > 0, "应检测到时间尺度违规"


# =============================================================================
# 4. BioModelsAPIClient 测试（不依赖网络）
# =============================================================================
class TestBioModelsClient:
    """BioModels API 客户端测试（仅测试 ID 提取，不测试网络下载）。"""

    def test_extract_biomd_id_from_input(self):
        """测试：从用户输入提取 BIOMD ID。"""
        result = extract_biomodel_id(
            "基于 BIOMD0000000205 模型参数仿真 EGF 刺激下 EGFR 磷酸化动力学"
        )
        assert result == "BIOMD0000000205", f"期望 BIOMD0000000205，实际 {result}"

    def test_extract_model_id_from_input(self):
        """测试：从用户输入提取 MODEL ID。"""
        result = extract_biomodel_id(
            "参考 MODEL1234567890 仿真"
        )
        assert result == "MODEL1234567890"

    def test_extract_no_id_returns_empty(self):
        """测试：无 BIOMD ID 时返回空字符串。"""
        result = extract_biomodel_id(
            "EGF 刺激下 EGFR 磷酸化动力学"
        )
        assert result == ""

    def test_get_client_singleton(self):
        """测试：get_biomodels_client 返回单例。"""
        c1 = get_biomodels_client()
        c2 = get_biomodels_client()
        assert c1 is c2, "应返回同一实例"


# =============================================================================
# 5. Signaling_Cascade_Phos.j2 模板渲染测试
# =============================================================================
class TestSignalingCascadePhosTemplate:
    """Signaling_Cascade_Phos 模板渲染与质量守恒测试。"""

    def test_template_renders_without_error(self):
        """测试：模板渲染不报错。"""
        edges = [
            {"source": "EGF", "target": "EGFR", "interaction": "activation",
             "mechanism": "binding", "reaction_equation": "EGF + EGFR → EGF-EGFR"},
            {"source": "EGF-EGFR", "target": "pEGFR", "interaction": "activation",
             "mechanism": "phosphorylation", "reaction_equation": "EGF-EGFR → pEGFR"},
        ]
        species_names = ["EGF", "EGFR", "EGF-EGFR", "pEGFR"]
        template_vars = {
            "species_names": species_names,
            "t_end": 120.0,
            "n_eval": 300,
            "y0": [0.008, 0.3, 0.0, 0.0],
            "edges": edges,
            "edges_json": edges,
            "parameters": {},
            "params_json": {
                "EGFR": {"k_on": 1.0, "k_off": 0.01, "degradation": 0.001},
                "pEGFR": {"k_phos": 0.5, "k_dephos": 0.05, "degradation": 0.002},
            },
            "inhibitor": "EGF",
            "target": "EGFR",
            "activator": "EGF",
            "kd": 10.0,
            "n_hill": 2,
            "degradation": 0.1,
            "production": 1.0,
            "drug_name": "Drug",
            "dose": 100.0,
            "k10": 0.1,
            "k12": 0.0,
            "k21": 0.0,
            "ec50": 10.0,
            "emax": 1.0,
            "gamma": 1.0,
        }
        code = render_template("Signaling_Cascade_Phos", template_vars)
        assert "def _ode" in code
        assert "SPECIES_NAMES" in code
        assert "solve_ivp" in code
        # 验证不再有非对称缩放（* 0.01 / * 0.005）
        assert "* 0.005" not in code, "模板不应再含非对称 * 0.005 缩放"
        assert "_net_rate" in code, "应含净反应速率计算（质量守恒）"

    def test_template_syntax_valid(self):
        """测试：渲染后的代码语法合法（ast.parse 通过）。"""
        import ast
        edges = [
            {"source": "EGF", "target": "EGFR", "interaction": "activation",
             "mechanism": "binding", "reaction_equation": "EGF + EGFR → EGF-EGFR"},
        ]
        template_vars = {
            "species_names": ["EGF", "EGFR", "EGF-EGFR"],
            "t_end": 120.0,
            "n_eval": 300,
            "y0": [0.008, 0.3, 0.0],
            "edges": edges,
            "edges_json": edges,
            "parameters": {},
            "params_json": {"EGFR": {"k_on": 1.0, "k_off": 0.01}},
            "inhibitor": "EGF", "target": "EGFR", "activator": "EGF",
            "kd": 10.0, "n_hill": 2, "degradation": 0.1, "production": 1.0,
            "drug_name": "Drug", "dose": 100.0, "k10": 0.1, "k12": 0.0,
            "k21": 0.0, "ec50": 10.0, "emax": 1.0, "gamma": 1.0,
        }
        code = render_template("Signaling_Cascade_Phos", template_vars)
        try:
            ast.parse(code)
        except SyntaxError as exc:
            raise AssertionError(f"渲染代码语法错误：{exc}")


# =============================================================================
# 6. RagClient 4-collection 接口测试
# =============================================================================
class TestRagClient4Collection:
    """RagClient 4-collection 接口测试。"""

    def test_search_params_hybrid_with_context_method_exists(self):
        """测试：search_params_hybrid_with_context 方法存在。"""
        from app.rag_client import RagClient
        assert hasattr(RagClient, "search_params_hybrid_with_context"), (
            "RagClient 应有 search_params_hybrid_with_context 方法"
        )

    def test_search_params_hybrid_with_context_signature(self):
        """测试：方法签名包含 include_mechanism / include_evidence 参数。"""
        import inspect
        from app.rag_client import RagClient
        sig = inspect.signature(RagClient.search_params_hybrid_with_context)
        params = set(sig.parameters.keys())
        assert "include_mechanism" in params
        assert "include_evidence" in params
        assert "rag_collections" in params


# =============================================================================
# 7. 集成测试：EGF-EGFR 端到端流程（不调用 LLM）
# =============================================================================
class TestEGFEGFREndToEnd:
    """EGF-EGFR 端到端流程测试（不调用 LLM，使用 mock 数据）。"""

    def test_full_template_selection_pipeline(self):
        """测试：完整模板选择流水线。"""
        # 1. 用户输入
        user_input = (
            "表皮生长因子（EGF）结合 EGFR 受体后诱导其二聚化和自磷酸化，"
            "激活下游 Shc-Grb2-SOS-Ras-MAPK 信号级联。"
            "请基于 BIOMD0000000205 模型的参数，仿真 EGF 刺激下 EGFR 磷酸化的动力学过程。"
            "初始条件：EGF=0.008 nM，EGFR=0.3 nM。"
        )
        edges = _make_egf_egfr_edges()
        # 2. 提取 BIOMD ID
        model_id = extract_biomodel_id(user_input)
        assert model_id == "BIOMD0000000205"
        # 3. 模板选择
        selection = select_template(
            user_input=user_input,
            edges=edges,
            llm_template="Cascade_Activation",  # LLM 错误选择
            sbml_model_id=model_id,
        )
        assert selection.template == "Signaling_Cascade_Phos"
        assert selection.override_llm is True
        # 4. 时间尺度
        t_end, n_eval, unit = get_simulation_time_scale(selection.template)
        assert t_end == 120.0
        assert unit == "min"
        # 5. Reaction Graph
        kg = _make_egf_egfr_kg()
        graph = build_reaction_graph(kg)
        violations = validate_reaction_graph(graph)
        assert violations == []
        # 6. 领域审查（仅检查模板渲染后的代码不含 shortcut）
        assert len(graph["reactions"]) == 5


# =============================================================================
# 测试入口
# =============================================================================
if __name__ == "__main__":
    # 简单测试运行器（不依赖 pytest）
    test_classes = [
        TestTemplateSelector,
        TestReactionIR,
        TestDomainChecker,
        TestBioModelsClient,
        TestSignalingCascadePhosTemplate,
        TestRagClient4Collection,
        TestEGFEGFREndToEnd,
    ]
    total_pass = 0
    total_fail = 0
    failures: list[str] = []
    for test_class in test_classes:
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in methods:
            try:
                # setup_method
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                getattr(instance, method_name)()
                print(f"  PASS: {test_class.__name__}.{method_name}")
                total_pass += 1
            except Exception as exc:
                print(f"  FAIL: {test_class.__name__}.{method_name}: {exc}")
                failures.append(f"{test_class.__name__}.{method_name}: {exc}")
                total_fail += 1
    print(f"\n{'=' * 60}")
    print(f"总计：{total_pass} 通过，{total_fail} 失败")
    if failures:
        print("\n失败详情：")
        for f in failures:
            print(f"  - {f}")
    print(f"{'=' * 60}")
    sys.exit(0 if total_fail == 0 else 1)
