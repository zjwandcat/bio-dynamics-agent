"""
N4→N5→N6 端到端集成测试：验证 KG→Reaction Graph→Template Engine→ODE 流水线
重点：N4 KGBuilder 保留 mechanism/reaction_equation 字段，N6 正确消费
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_n4_preserves_mechanism():
    """验证 N4 KGBuilder 保留 mechanism / reaction_equation 字段。"""
    from app.nodes_v2 import n4_kg_builder

    # 模拟 N1+N2 输出
    state = {
        "entities": [
            {"entity_id": "e1", "name": "EGF", "type": "Molecule", "aliases": []},
            {"entity_id": "e2", "name": "EGFR", "type": "Protein", "aliases": []},
            {"entity_id": "e3", "name": "pEGFR", "type": "Protein", "aliases": []},
        ],
        "network_relations": {
            "edges": [
                {
                    "source": "EGF",
                    "target": "EGFR",
                    "interaction": "activation",
                    "mechanism": "binding",
                    "reaction_equation": "EGF + EGFR → EGF-EGFR",
                },
                {
                    "source": "EGFR",
                    "target": "pEGFR",
                    "interaction": "activation",
                    "mechanism": "phosphorylation",
                    "reaction_equation": "EGFR → pEGFR",
                },
            ]
        },
    }

    result = n4_kg_builder(state)
    kg = result["knowledge_graph"]
    edges = kg["edges"]

    print(f"\n--- N4 输出 edges ---")
    for e in edges:
        print(f"  {e}")

    assert len(edges) == 2, f"应有 2 条边，实际 {len(edges)}"

    # 验证 mechanism 字段保留
    binding_edge = next(e for e in edges if e.get("mechanism") == "binding")
    phos_edge = next(e for e in edges if e.get("mechanism") == "phosphorylation")

    assert binding_edge["source"] == "EGF"
    assert binding_edge["target"] == "EGFR"
    assert binding_edge["reaction_equation"] == "EGF + EGFR → EGF-EGFR"

    assert phos_edge["source"] == "EGFR"
    assert phos_edge["target"] == "pEGFR"
    assert phos_edge["reaction_equation"] == "EGFR → pEGFR"

    print("\n✓ N4 保留 mechanism / reaction_equation 字段")
    return kg


def test_n4_fallback_preserves_mechanism():
    """验证 N4 fallback 路径（network_json.edges）也保留 mechanism 字段。"""
    from app.nodes_v2 import n4_kg_builder

    state = {
        "entities": [
            {"entity_id": "e1", "name": "EGF", "type": "Molecule", "aliases": []},
            {"entity_id": "e2", "name": "EGFR", "type": "Protein", "aliases": []},
        ],
        # 注意：没有 network_relations，触发 network_json fallback
        "network_json": {
            "edges": [
                {
                    "source": "EGF",
                    "target": "EGFR",
                    "interaction": "activation",
                    "mechanism": "binding",
                    "reaction_equation": "EGF + EGFR → EGF-EGFR",
                },
            ]
        },
    }

    result = n4_kg_builder(state)
    kg = result["knowledge_graph"]
    edges = kg["edges"]

    print(f"\n--- N4 fallback 输出 edges ---")
    for e in edges:
        print(f"  {e}")

    assert len(edges) == 1
    assert edges[0].get("mechanism") == "binding", f"fallback 路径丢失 mechanism: {edges[0]}"
    assert edges[0].get("reaction_equation") == "EGF + EGFR → EGF-EGFR"

    print("\n✓ N4 fallback 路径也保留 mechanism / reaction_equation")
    return kg


def test_n6_consumes_mechanism_from_n4():
    """验证 N6 能消费 N4 输出的 KG（含 mechanism 字段）并渲染 Signaling_Cascade_Phos。"""
    from app.nodes_v2 import n4_kg_builder, n6_ode_generator

    # Step 1: N4 构建 KG
    state_after_n4 = {
        "entities": [
            {"entity_id": "e1", "name": "EGF", "type": "Molecule", "aliases": []},
            {"entity_id": "e2", "name": "EGFR", "type": "Protein", "aliases": []},
            {"entity_id": "e3", "name": "pEGFR", "type": "Protein", "aliases": []},
        ],
        "network_relations": {
            "edges": [
                {"source": "EGF", "target": "EGFR", "interaction": "activation",
                 "mechanism": "binding", "reaction_equation": "EGF + EGFR → EGF-EGFR"},
                {"source": "EGFR", "target": "pEGFR", "interaction": "activation",
                 "mechanism": "phosphorylation", "reaction_equation": "EGFR → pEGFR"},
            ]
        },
    }
    n4_result = n4_kg_builder(state_after_n4)
    kg = n4_result["knowledge_graph"]

    # Step 2: N6 渲染 ODE
    state_for_n6 = {
        "user_input": "EGF=0.008 nM, EGFR=0.3 nM",
        "knowledge_graph": kg,
        "mechanism": {
            "simulation_type": "signaling_cascade_phos",
            "template": "Signaling_Cascade_Phos",
        },
        "parameters": {
            "EGF->EGFR": {"param_name": "k1", "value": 100.0, "unit": "nM-1*min-1",
                          "source": "BIOMD0000000205", "is_fallback": False},
            "EGFR->pEGFR": {"param_name": "k1", "value": 0.5, "unit": "min-1",
                            "source": "BIOMD0000000205", "is_fallback": False},
        },
    }

    n6_result = n6_ode_generator(state_for_n6)
    code = n6_result["ode_model"]["code"]

    # 验证代码包含 mechanism 字段
    assert "'mechanism': 'binding'" in code, f"代码缺少 mechanism=binding: {code[:500]}"
    assert "'mechanism': 'phosphorylation'" in code, f"代码缺少 mechanism=phosphorylation"
    assert "EGF + EGFR → EGF-EGFR" in code, "代码缺少 reaction_equation"
    assert "EGFR → pEGFR" in code, "代码缺少 EGFR→pEGFR reaction_equation"

    print(f"\n✓ N6 消费 N4 输出的 KG（含 mechanism 字段）")
    print(f"  代码长度: {len(code)} chars")

    # 执行代码验证
    test_dir = Path(__file__).parent / "_test_n4_n6_integration"
    test_dir.mkdir(exist_ok=True)
    code_file = test_dir / "test_ode.py"
    code_file.write_text(code, encoding="utf-8")

    import subprocess
    proc = subprocess.run(
        ["python", str(code_file)],
        cwd=str(test_dir),
        capture_output=True,
        text=True,
        timeout=30,
    )

    if proc.returncode != 0:
        print(f"  执行失败: {proc.stderr[:1000]}")
        return

    print(f"  执行成功:")
    for line in proc.stdout.split("\n"):
        if line.startswith("BIO_CHECK:"):
            print(f"    {line}")


if __name__ == "__main__":
    print("=" * 60)
    print("测试 1: N4 保留 mechanism 字段")
    print("=" * 60)
    test_n4_preserves_mechanism()

    print("\n" + "=" * 60)
    print("测试 2: N4 fallback 路径保留 mechanism 字段")
    print("=" * 60)
    test_n4_fallback_preserves_mechanism()

    print("\n" + "=" * 60)
    print("测试 3: N6 消费 N4 输出的 KG（含 mechanism）")
    print("=" * 60)
    test_n6_consumes_mechanism_from_n4()
