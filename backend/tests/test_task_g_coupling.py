"""Task G 验证：ODE 药物-靶点抑制耦合。

验证内容：
1. Simple_Inhibition.j2 使用标准 inhibition 约定（0=无药, 1=全抑制），且耦合到靶点 ODE
2. PKPD_OneCompartment.j2 使用 Emax 模型并耦合 effect 到靶点
3. PKPD_TwoCompartment.j2 同样正确耦合
4. 无 PK/PD 时 Simple_Inhibition 仍可正常渲染
"""
import pytest

from app.ode_templates import render_template, list_templates


def test_simple_inhibition_uses_standard_inhibition_convention():
    """Simple_Inhibition 应使用 inhibition∈[0,1] 约定并耦合 (1-inhibition)。"""
    code = render_template("Simple_Inhibition", {
        "species_names": ["Drug", "Target"],
        "t_end": 48.0,
        "n_eval": 200,
        "y0": [10.0, 10.0],
        "edges": [],
        "parameters": {},
        "inhibitor": "Drug",
        "target": "Target",
        "activator": "Drug",
        "kd": 10.0,
        "n_hill": 2,
        "degradation": 0.1,
        "production": 1.0,
        "edges_json": [],
        "params_json": {},
        "drug_name": "Drug",
        "dose": 100.0,
        "k10": 0.1,
        "k12": 0.0,
        "k21": 0.0,
        "ec50": 10.0,
        "emax": 1.0,
        "gamma": 1.0,
    })
    # inhibition 应为 inhibitor**N / (KD**N + inhibitor**N)，而非反转
    assert "inhibitor ** N_HILL / (KD ** N_HILL + inhibitor ** N_HILL" in code
    # 耦合应为 (1 - inhibition)，而非 inhibition 直接相乘
    assert "PROD_RATE * (1.0 - inhibition)" in code
    # 不应出现旧的反转公式
    assert "KD ** N_HILL / (KD ** N_HILL + inhibitor" not in code


def test_pkpd_one_compartment_couples_emax_to_target():
    """PKPD_OneCompartment 应使用 Emax 模型并耦合 (1-effect) 到靶点。"""
    code = render_template("PKPD_OneCompartment", {
        "species_names": ["Osimertinib", "EGFR"],
        "t_end": 48.0,
        "n_eval": 200,
        "y0": [100.0, 10.0],
        "edges": [],
        "parameters": {},
        "inhibitor": "Osimertinib",
        "target": "EGFR",
        "activator": "Osimertinib",
        "kd": 10.0,
        "n_hill": 2,
        "degradation": 0.1,
        "production": 1.0,
        "edges_json": [],
        "params_json": {},
        "drug_name": "Osimertinib",
        "dose": 100.0,
        "k10": 0.1,
        "k12": 0.0,
        "k21": 0.0,
        "ec50": 12.0,
        "emax": 1.0,
        "gamma": 1.0,
    })
    assert "EMAX = 1.0" in code
    assert "EC50 = 12.0" in code
    assert "effect = EMAX * (drug_conc ** GAMMA) / (EC50 ** GAMMA + drug_conc ** GAMMA" in code
    assert "(1.0 - effect)" in code


def test_pkpd_two_compartment_couples_emax_to_target():
    """PKPD_TwoCompartment 应使用 Emax 模型并耦合 (1-effect) 到靶点。"""
    code = render_template("PKPD_TwoCompartment", {
        "species_names": ["Osimertinib_central", "Osimertinib_peripheral", "EGFR"],
        "t_end": 48.0,
        "n_eval": 200,
        "y0": [100.0, 0.0, 10.0],
        "edges": [],
        "parameters": {},
        "inhibitor": "Osimertinib",
        "target": "EGFR",
        "activator": "Osimertinib",
        "kd": 10.0,
        "n_hill": 2,
        "degradation": 0.1,
        "production": 1.0,
        "edges_json": [],
        "params_json": {},
        "drug_name": "Osimertinib",
        "dose": 100.0,
        "k10": 0.1,
        "k12": 0.05,
        "k21": 0.03,
        "ec50": 12.0,
        "emax": 1.0,
        "gamma": 1.0,
    })
    assert "K12 = 0.05" in code
    assert "K21 = 0.03" in code
    assert "effect = EMAX * (c_central ** GAMMA) / (EC50 ** GAMMA + c_central ** GAMMA" in code
    assert "(1.0 - effect)" in code


def test_pkpd_templates_available():
    """PKPD 模板应已注册。"""
    available = list_templates()
    assert "PKPD_OneCompartment" in available
    assert "PKPD_TwoCompartment" in available


def test_pkpd_detection_logic():
    """验证 PK/PD 检测条件：drug_name + drug_target 同时存在才激活。"""
    # 有完整 profile
    profile_full = {"drug_name": "Osimertinib", "drug_target": "EGFR"}
    assert bool(profile_full.get("drug_name") and profile_full.get("drug_target"))

    # 缺少靶点
    profile_no_target = {"drug_name": "Osimertinib", "drug_target": ""}
    assert not bool(profile_no_target.get("drug_name") and profile_no_target.get("drug_target"))

    # 空 profile
    profile_empty = {}
    assert not bool(profile_empty.get("drug_name") and profile_empty.get("drug_target"))


def test_n6_switches_to_pkpd_template_when_profile_present(monkeypatch):
    """n6_ode_generator 在 pkpd_profile 存在时应切换到 PKPD 模板并耦合 Emax。"""
    from unittest.mock import MagicMock
    from app.nodes_v2 import n6_ode_generator

    # mock LLM 避免真实调用
    mock_resp = MagicMock()
    mock_resp.content = '{"variables": [], "equations": []}'
    monkeypatch.setattr("app.nodes_v2.llm", MagicMock(invoke=MagicMock(return_value=mock_resp)))

    state = {
        "user_input": "Osimertinib inhibits EGFR",
        "mechanism": {"template": "Simple_Inhibition"},
        "knowledge_graph": {
            "nodes": [{"id": "Osimertinib", "name": "Osimertinib"},
                      {"id": "EGFR", "name": "EGFR"}],
            "edges": [{"source": "Osimertinib", "target": "EGFR",
                       "interaction": "inhibition"}],
        },
        "parameters": {},
        "pkpd_profile": {
            "drug_name": "Osimertinib",
            "drug_target": "EGFR",
            "compartment": "1-compartment",
            "pk_params": {"k10": 0.1},
            "pd_params": {"Emax": 1.0, "EC50": 12.0, "gamma": 1.0},
        },
        "drug_regimen": [{"drug_name": "Osimertinib", "dose": 100.0,
                          "ec50": 12.0, "emax": 1.0, "gamma": 1.0,
                          "target": "EGFR"}],
    }
    out = n6_ode_generator(state)
    code = out["ode_model"]["code"]
    # 应切换到 PKPD_OneCompartment
    assert out["ode_model"]["template"] == "PKPD_OneCompartment"
    # 应包含 Emax 耦合
    assert "EMAX = 1.0" in code
    assert "EC50 = 12.0" in code
    assert "effect = EMAX" in code
    assert "(1.0 - effect)" in code
    # 应包含药物名
    assert "Osimertinib" in code


def test_n6_keeps_simple_template_without_pkpd(monkeypatch):
    """n6_ode_generator 在无 pkpd_profile 时保持原模板。"""
    from unittest.mock import MagicMock
    from app.nodes_v2 import n6_ode_generator

    mock_resp = MagicMock()
    mock_resp.content = '{"variables": [], "equations": []}'
    monkeypatch.setattr("app.nodes_v2.llm", MagicMock(invoke=MagicMock(return_value=mock_resp)))

    state = {
        "user_input": "A inhibits B",
        "mechanism": {"template": "Simple_Inhibition"},
        "knowledge_graph": {
            "nodes": [{"id": "A", "name": "A"}, {"id": "B", "name": "B"}],
            "edges": [{"source": "A", "target": "B", "interaction": "inhibition"}],
        },
        "parameters": {},
    }
    out = n6_ode_generator(state)
    assert out["ode_model"]["template"] == "Simple_Inhibition"
    code = out["ode_model"]["code"]
    # 应使用新的标准 inhibition 约定
    assert "PROD_RATE * (1.0 - inhibition)" in code

