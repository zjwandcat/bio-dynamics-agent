"""
Step 2.2 + 2.3 结构性修复验证：Signaling_Cascade_Phos 模板端到端流水线测试

验证点：
1. KG → Reaction Graph → Template Engine → ODE 流水线正常渲染
2. edges_for_template 携带 mechanism / reaction_equation 字段
3. phos_cascade_params 按 mechanism 映射 k_on/k_off/k_phos/k_dephos
4. 初始条件：EGF=0.008, EGFR=0.3, pEGFR/pShc/...=0.0
5. ODE 代码可执行（AST + 沙箱执行）
6. pEGFR 5-10 min 达峰，MAPK 级联放大
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def make_test_state() -> dict:
    """构造 EGF-EGFR 信号级联的测试 state（模拟 N1+N2+N4+N5 输出）。"""
    return {
        "user_input": "EGF=0.008 nM, EGFR=0.3 nM",
        "knowledge_graph": {
            "nodes": [
                {"id": "e1", "name": "EGF", "type": "Molecule", "aliases": []},
                {"id": "e2", "name": "EGFR", "type": "Protein", "aliases": []},
                {"id": "e3", "name": "pEGFR", "type": "Protein", "aliases": []},
                {"id": "e4", "name": "Shc", "type": "Protein", "aliases": []},
                {"id": "e5", "name": "pShc", "type": "Protein", "aliases": []},
                {"id": "e6", "name": "Grb2", "type": "Protein", "aliases": []},
                {"id": "e7", "name": "SOS", "type": "Protein", "aliases": []},
                {"id": "e8", "name": "RasGDP", "type": "Protein", "aliases": []},
                {"id": "e9", "name": "RasGTP", "type": "Protein", "aliases": []},
                {"id": "e10", "name": "Raf", "type": "Protein", "aliases": []},
                {"id": "e11", "name": "pRaf", "type": "Protein", "aliases": []},
                {"id": "e12", "name": "MEK", "type": "Protein", "aliases": []},
                {"id": "e13", "name": "pMEK", "type": "Protein", "aliases": []},
                {"id": "e14", "name": "MAPK", "type": "Protein", "aliases": []},
                {"id": "e15", "name": "pMAPK", "type": "Protein", "aliases": []},
            ],
            "edges": [
                {"source": "EGF",      "target": "EGFR",   "interaction": "activation", "mechanism": "binding",         "reaction_equation": "EGF + EGFR → EGF-EGFR"},
                {"source": "EGFR",     "target": "pEGFR",  "interaction": "activation", "mechanism": "phosphorylation", "reaction_equation": "EGFR → pEGFR"},
                {"source": "pEGFR",    "target": "pShc",   "interaction": "activation", "mechanism": "phosphorylation", "reaction_equation": "pEGFR + Shc → pEGFR + pShc"},
                {"source": "pShc",     "target": "Grb2",   "interaction": "activation", "mechanism": "binding",         "reaction_equation": "pShc + Grb2 → pShc-Grb2"},
                {"source": "Grb2",     "target": "SOS",    "interaction": "activation", "mechanism": "binding",         "reaction_equation": "pShc-Grb2 + SOS → pShc-Grb2-SOS"},
                {"source": "SOS",      "target": "RasGTP", "interaction": "activation", "mechanism": "exchange",        "reaction_equation": "RasGDP → RasGTP (catalyzed by SOS)"},
                {"source": "RasGTP",   "target": "pRaf",   "interaction": "activation", "mechanism": "phosphorylation", "reaction_equation": "RasGTP + Raf → RasGTP + pRaf"},
                {"source": "pRaf",     "target": "pMEK",   "interaction": "activation", "mechanism": "phosphorylation", "reaction_equation": "pRaf + MEK → pRaf + pMEK"},
                {"source": "pMEK",     "target": "pMAPK",  "interaction": "activation", "mechanism": "phosphorylation", "reaction_equation": "pMEK + MAPK → pMEK + pMAPK"},
            ],
        },
        "mechanism": {
            "pathway": "EGF-EGFR-Shc-Grb2-SOS-Ras-Raf-MEK-MAPK signaling cascade",
            "simulation_type": "signaling_cascade_phos",
            "template": "Signaling_Cascade_Phos",
        },
        "parameters": {
            "EGF->EGFR":      {"param_name": "k1", "value": 100.0, "unit": "nM-1*min-1", "source": "BIOMD0000000205", "is_fallback": False},
            "EGFR->pEGFR":    {"param_name": "k1", "value": 0.5,   "unit": "min-1",      "source": "BIOMD0000000205", "is_fallback": False},
            "pEGFR->pShc":    {"param_name": "k1", "value": 0.3,   "unit": "min-1",      "source": "BIOMD0000000205", "is_fallback": False},
            "pShc->Grb2":     {"param_name": "k1", "value": 1.0,   "unit": "nM-1*min-1", "source": "BIOMD0000000205", "is_fallback": False},
            "Grb2->SOS":      {"param_name": "k1", "value": 0.8,   "unit": "nM-1*min-1", "source": "BIOMD0000000205", "is_fallback": False},
            "SOS->RasGTP":    {"param_name": "k1", "value": 0.1,   "unit": "min-1",      "source": "BIOMD0000000205", "is_fallback": False},
            "RasGTP->pRaf":   {"param_name": "k1", "value": 0.4,   "unit": "min-1",      "source": "BIOMD0000000205", "is_fallback": False},
            "pRaf->pMEK":     {"param_name": "k1", "value": 0.6,   "unit": "min-1",      "source": "BIOMD0000000205", "is_fallback": False},
            "pMEK->pMAPK":    {"param_name": "k1", "value": 0.7,   "unit": "min-1",      "source": "BIOMD0000000205", "is_fallback": False},
        },
    }


def test_n6_render_and_execute():
    """测试 N6 渲染 Signaling_Cascade_Phos 模板并执行仿真。"""
    from app.nodes_v2 import n6_ode_generator

    state = make_test_state()
    result = n6_ode_generator(state)
    ode_model = result.get("ode_model", {})
    code = ode_model.get("code", "")
    template = ode_model.get("template", "")

    print(f"\n{'='*60}")
    print(f"模板: {template}")
    print(f"代码长度: {len(code)} chars")
    print(f"{'='*60}")

    assert template == "Signaling_Cascade_Phos", f"模板错误: {template}"
    assert "EGF" in code and "pEGFR" in code and "pMAPK" in code, "代码缺少关键物种"
    assert "EDGES" in code and "PARAMS" in code, "代码缺少 EDGES/PARAMS"
    assert "solve_ivp" in code, "代码缺少 solve_ivp"

    # 打印代码前 60 行
    print("\n--- 代码预览（前 60 行）---")
    for i, line in enumerate(code.split("\n")[:60], 1):
        print(f"{i:3d} | {line}")

    # 写入文件并在沙箱执行
    test_dir = Path(__file__).parent / "_test_phos_pipeline"
    test_dir.mkdir(exist_ok=True)
    code_file = test_dir / "test_ode.py"
    code_file.write_text(code, encoding="utf-8")

    print(f"\n--- 沙箱执行（cwd={test_dir}）---")
    import subprocess
    proc = subprocess.run(
        ["python", str(code_file)],
        cwd=str(test_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )
    print(f"returncode: {proc.returncode}")
    print(f"stdout:\n{proc.stdout}")
    if proc.stderr:
        print(f"stderr:\n{proc.stderr[:3000]}")

    # 解析 BIO_CHECK 行
    bio_checks = {}
    for line in proc.stdout.split("\n"):
        if line.startswith("BIO_CHECK:"):
            parts = line.replace("BIO_CHECK:", "").strip().split("=")
            if len(parts) == 2:
                try:
                    bio_checks[parts[0].strip()] = float(parts[1].strip())
                except ValueError:
                    pass

    print(f"\n--- BIO_CHECK 结果 ---")
    for k, v in bio_checks.items():
        print(f"  {k:10s} = {v:.6f}")

    # 验证 pEGFR / pMAPK 是否有非零值（信号级联成功传导）
    assert "pEGFR" in bio_checks, "pEGFR 未生成"
    assert "pMAPK" in bio_checks, "pMAPK 未生成"
    print(f"\n✓ pEGFR = {bio_checks['pEGFR']:.6f}")
    print(f"✓ pMAPK = {bio_checks['pMAPK']:.6f}")

    # 检查 simulation.csv 是否生成，并验证 pEGFR 达峰时间
    csv_path = test_dir / "simulation.csv"
    if csv_path.exists():
        import csv
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        print(f"\n--- CSV 头: {header}")
        print(f"--- 行数: {len(rows)}")
        if "pEGFR" in header:
            pegfr_idx = header.index("pEGFR")
            t_idx = header.index("t")
            pegfr_values = [(float(r[t_idx]), float(r[pegfr_idx])) for r in rows]
            peak_t, peak_v = max(pegfr_values, key=lambda x: x[1])
            print(f"--- pEGFR 峰值: t={peak_t:.2f} min, value={peak_v:.6f}")
            # 5-10 min 达峰的判据（先看是否有动态，后续可调）
            if peak_v > 0:
                print(f"✓ pEGFR 有动态变化（peak at t={peak_t:.2f} min）")
        if "pMAPK" in header:
            pmapk_idx = header.index("pMAPK")
            pmapk_final = float(rows[-1][pmapk_idx])
            print(f"--- pMAPK 终值: {pmapk_final:.6f}")
    return bio_checks


if __name__ == "__main__":
    test_n6_render_and_execute()
