"""
Root Cause Re-Audit Script: BIOMD0000000205 结构性审计
检查三层一致性：(1) KG→ODE (2) Parameter Ontology (3) RAG semantic correctness
"""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

BACKEND = Path(__file__).resolve().parent / "backend"
SBML_PATH = BACKEND / "data" / "raw" / "BIOMD0000000205.xml"
PROCESSED_JSON = BACKEND / "data" / "processed" / "BIOMD0000000205.json"
TEST_OUTPUT = Path(__file__).resolve().parent / "test_outputs_egf"

# =============================================================================
# 1. 解析 SBML：提取物种、反应、参数
# =============================================================================
def parse_sbml():
    tree = ET.parse(SBML_PATH)
    root = tree.getroot()
    ns_match = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    ns = {"sbml": ns_match} if ns_match else {}

    model = root.find(".//sbml:model", ns) or root.find(".//model")

    # --- Species ---
    species = []
    species_list = model.find(".//sbml:listOfSpecies", ns) or model.find(".//listOfSpecies")
    if species_list is not None:
        for sp in species_list:
            sid = sp.get("id", "")
            name = sp.get("name", sid)
            init_conc = sp.get("initialConcentration", "0")
            species.append({
                "id": sid,
                "name": name,
                "initial_concentration": float(init_conc) if init_conc else 0.0,
            })

    # --- Reactions ---
    reactions = []
    reaction_list = model.find(".//sbml:listOfReactions", ns) or model.find(".//listOfReactions")
    if reaction_list is not None:
        for rxn in reaction_list:
            rid = rxn.get("id", "")
            rname = rxn.get("name", rid)

            # Reactants
            reactants = []
            reactant_list = rxn.find(".//sbml:listOfReactants", ns) or rxn.find(".//listOfReactants")
            if reactant_list is not None:
                for ref in reactant_list:
                    sp_ref = ref.get("species", "")
                    stoich = ref.get("stoichiometry", "1")
                    reactants.append({"species": sp_ref, "stoichiometry": float(stoich)})

            # Products
            products = []
            product_list = rxn.find(".//sbml:listOfProducts", ns) or rxn.find(".//listOfProducts")
            if product_list is not None:
                for ref in product_list:
                    sp_ref = ref.get("species", "")
                    stoich = ref.get("stoichiometry", "1")
                    products.append({"species": sp_ref, "stoichiometry": float(stoich)})

            # Local parameters in kinetic law
            local_params = []
            kl = rxn.find(".//sbml:kineticLaw", ns) or rxn.find(".//kineticLaw")
            if kl is not None:
                lp_list = kl.find(".//sbml:listOfLocalParameters", ns) or kl.find(".//listOfLocalParameters")
                if lp_list is None:
                    lp_list = kl.find(".//sbml:listOfParameters", ns) or kl.find(".//listOfParameters")
                if lp_list is not None:
                    for p in lp_list:
                        pid = p.get("id", "")
                        pval = p.get("value", "")
                        punit = p.get("units", "")
                        local_params.append({
                            "id": pid,
                            "value": float(pval) if pval else None,
                            "unit": punit,
                        })

            reactions.append({
                "id": rid,
                "name": rname,
                "reactants": reactants,
                "products": products,
                "local_params": local_params,
            })

    # --- Global parameters ---
    global_params = []
    param_list = model.find(".//sbml:listOfParameters", ns) or model.find(".//listOfParameters")
    if param_list is not None:
        for p in param_list:
            pid = p.get("id", "")
            pval = p.get("value", "")
            punit = p.get("units", "")
            global_params.append({
                "id": pid,
                "value": float(pval) if pval else None,
                "unit": punit,
            })

    return species, reactions, global_params


def species_id_to_name(species):
    return {s["id"]: s["name"] for s in species}


# =============================================================================
# 2. 加载当前测试输出的 KG 和 ODE
# =============================================================================
def load_current_kg_and_ode():
    ode_path = TEST_OUTPUT / "ode_code.py"
    summary_path = TEST_OUTPUT / "summary.json"

    ode_code = ode_path.read_text(encoding="utf-8") if ode_path.exists() else ""

    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # 从 ODE 代码中提取 SPECIES_NAMES, EDGES, PARAMS, Y0
    species_names = []
    edges = []
    params = {}
    y0 = []

    sp_match = re.search(r"SPECIES_NAMES\s*=\s*(\[[^\]]+\])", ode_code)
    if sp_match:
        species_names = eval(sp_match.group(1))

    edge_match = re.search(r"EDGES\s*=\s*(\[.+?\])\s*\n", ode_code, re.DOTALL)
    if edge_match:
        edges = eval(edge_match.group(1))

    param_match = re.search(r"PARAMS\s*=\s*(\{.+?\})\s*\n", ode_code, re.DOTALL)
    if param_match:
        params = eval(param_match.group(1))

    y0_match = re.search(r"Y0\s*=\s*(\[[^\]]+\])", ode_code)
    if y0_match:
        y0 = eval(y0_match.group(1))

    return {
        "species_names": species_names,
        "edges": edges,
        "params": params,
        "y0": y0,
        "kg_node_count": summary.get("kg_node_count", 0),
        "kg_edge_count": summary.get("kg_edge_count", 0),
    }


# =============================================================================
# 3. 三层一致性审计
# =============================================================================
def audit_kg_ode_consistency(sbml_species, sbml_reactions, current):
    """Audit 1: KG → ODE consistency"""
    sp_name_map = species_id_to_name(sbml_species)
    sbml_species_names = {s["name"] for s in sbml_species}
    current_species = set(current["species_names"])

    # Missing species: 关键中间态缺失
    key_intermediates = [
        "EGF-EGFR", "EGF-EGFR-2", "EGF-pEGFR-2", "EGF-pEGFR-2-SHP",
        "pEGFR", "pShc", "pMAPK", "pRas", "Ras-GTP", "Ras-GDP",
        "Grb2-SOS", "Shc-Grb2-SOS",
    ]
    # 在 SBML 中查找含 pEGFR/pShc/pMAPK 等的物种
    phosphorylated_in_sbml = [s["name"] for s in sbml_species if "p" in s["name"].lower() and ("EGFR" in s["name"] or "MAPK" in s["name"] or "Shc" in s["name"])]

    missing_nodes = []
    for s in sbml_species:
        if s["name"] not in current_species and s["initial_concentration"] == 0.0:
            # 只标记初始浓度=0 的中间态（这些是被生成的）
            missing_nodes.append({
                "name": s["name"],
                "id": s["id"],
                "reason": f"SBML 中存在但 KG 中缺失（初始浓度=0，为中间态）",
            })

    # 同时检查关键磷酸化形式
    for kw in ["pEGFR", "pShc", "pMAPK", "pRas"]:
        found = any(kw.lower() in n.lower() for n in current_species)
        if not found:
            missing_nodes.append({
                "name": kw,
                "id": "N/A",
                "reason": f"磷酸化中间体 {kw} 在 KG 中完全缺失",
            })

    # Invalid edges: 检查是否有错误的 interaction 类型
    invalid_edges = []
    for e in current["edges"]:
        # SBML 反应中没有简单的 "activation" 类型，都是 mass-action binding/phosphorylation
        if e.get("interaction") == "activation":
            # 检查是否跳过了中间态
            src = e.get("source", "")
            tgt = e.get("target", "")
            # EGF → EGFR 应该是 binding 不是 activation
            if src == "EGF" and tgt == "EGFR":
                invalid_edges.append({
                    "edge": f"{src}→{tgt}",
                    "reason": "EGF→EGFR 应为 binding（mass action）而非 activation",
                })
            # HRAS → MAPK1 跳过了 pMAPK
            if "MAPK" in tgt and "RAS" in src.upper():
                invalid_edges.append({
                    "edge": f"{src}→{tgt}",
                    "reason": "Ras→MAPK 跳过了 MAPK 磷酸化（pMAPK）中间态",
                })

    # Collapsed pathways: 检查是否有多步合并为一步
    collapsed_pathways = []
    # EGF → EGFR → SHC1 → GRB2 → SOS1 → HRAS → MAPK1 是 6 步简化
    # 实际 SBML 有更多步骤
    sbml_reaction_count = len(sbml_reactions)
    current_edge_count = len(current["edges"])
    if sbml_reaction_count > current_edge_count * 3:
        collapsed_pathways.append({
            "current": f"{current_edge_count} 条 activation 边",
            "sbml": f"{sbml_reaction_count} 条 mass-action/phosphorylation 反应",
            "reason": f"通路被压缩为 {current_edge_count} 步简单 activation，丢失 {sbml_reaction_count - current_edge_count} 个反应节点",
        })

    return {
        "missing_nodes": missing_nodes[:20],  # 限制输出
        "invalid_edges": invalid_edges,
        "collapsed_pathways": collapsed_pathways,
        "sbml_species_count": len(sbml_species),
        "current_species_count": len(current_species),
        "sbml_reaction_count": sbml_reaction_count,
        "current_edge_count": current_edge_count,
    }


def audit_parameter_ontology(sbml_species, sbml_reactions, global_params):
    """Audit 2: Parameter Ontology Audit"""
    # 分类参数
    param_types = defaultdict(list)

    # initial_concentration（来自 species）
    for s in sbml_species:
        param_types["initial_concentration"].append({
            "param_name": f"initial_concentration_{s['id']}",
            "species": s["name"],
            "value": s["initial_concentration"],
        })

    # kinetic_rate（来自反应的 local params: k1, k2 等）
    for rxn in sbml_reactions:
        for lp in rxn["local_params"]:
            pid = lp["id"]
            ptype = "kinetic_rate" if re.match(r"^k\d*$", pid) else "other"
            param_types[ptype].append({
                "param_name": pid,
                "reaction": rxn["id"],
                "value": lp["value"],
                "unit": lp["unit"],
            })

    # global params
    for gp in global_params:
        pid = gp["id"]
        if re.match(r"^k\d*$", pid):
            param_types["kinetic_rate"].append({
                "param_name": pid,
                "value": gp["value"],
                "unit": gp["unit"],
            })
        else:
            param_types["other"].append({
                "param_name": pid,
                "value": gp["value"],
                "unit": gp["unit"],
            })

    # 检查 misused parameters：当前 ODE 中哪些参数被误用
    current = load_current_kg_and_ode()
    misused_parameters = []
    for target, pinfo in current["params"].items():
        param_name = pinfo.get("param_name", "")
        value = pinfo.get("value", 0)
        # 检查是否是 initial_concentration 被用作 Kd
        if param_name.startswith("initial_concentration"):
            misused_parameters.append({
                "target": target,
                "param_name": param_name,
                "value": value,
                "issue": "initial_concentration 被误用为 Kd（动力学常数）",
                "correct_type": "initial_concentration",
                "used_as": "Kd",
            })
        # 检查 Kd=0（会导致 Hill 函数饱和）
        if value == 0.0 and param_name.lower() in ("kd", "k1", "k2"):
            misused_parameters.append({
                "target": target,
                "param_name": param_name,
                "value": value,
                "issue": "Kd=0 会导致 Hill 函数饱和（分母=0）",
                "correct_type": "binding_affinity",
                "used_as": "Kd=0 (saturated)",
            })

    return {
        "param_type_distribution": {k: len(v) for k, v in param_types.items()},
        "misused_parameters": misused_parameters,
        "invalid_bindings": [
            {
                "target": tgt,
                "param_name": p.get("param_name", ""),
                "value": p.get("value", 0),
                "issue": "非动力学参数被绑定为 Kd",
            }
            for tgt, p in current["params"].items()
            if p.get("param_name", "").startswith("initial_concentration")
        ],
    }


def audit_rag_semantic(sbml_species, sbml_reactions):
    """Audit 3: RAG semantic correctness"""
    # 模拟 RAG 查询：EGF activation EGFR kinetic parameter Kd
    # 检查 embedding 相似度是否 override type constraint

    # 构建 document 文本（模拟 ChromaDB 中的存储格式）
    documents = []
    for s in sbml_species:
        doc = f"initial_concentration_{s['id']} Species '{s['name']}' (id={s['id']}) from SBML model BIOMD0000000205. Initial concentration/amount used in the model."
        documents.append({"type": "initial_concentration", "name": s["name"], "doc": doc})

    for rxn in sbml_reactions:
        for lp in rxn["local_params"]:
            doc = f"Local parameter '{lp['id']}' in reaction '{rxn['id']}' from SBML model BIOMD0000000205."
            documents.append({"type": "kinetic_rate", "name": lp["id"], "doc": doc, "reaction": rxn["id"]})

    # 模拟查询
    query = "EGF activation EGFR kinetic parameter Kd"

    # 检查哪些 document 包含 EGF/EGFR 关键词
    type_conflicts = []
    for d in documents:
        doc_lower = d["doc"].lower()
        has_egf = "egf" in doc_lower
        has_egfr = "egfr" in doc_lower
        has_kinetic = "kinetic" in doc_lower or "parameter" in doc_lower or "reaction" in doc_lower
        if has_egf and has_egfr and d["type"] == "initial_concentration":
            type_conflicts.append({
                "doc_type": d["type"],
                "name": d["name"],
                "issue": "initial_concentration 文档含 EGF/EGFR 关键词，embedding 会优先返回（语义近）",
                "doc_snippet": d["doc"][:120],
            })
        if d["type"] == "kinetic_rate" and not has_egf and not has_egfr:
            type_conflicts.append({
                "doc_type": d["type"],
                "name": d["name"],
                "issue": "kinetic_rate 文档不含 EGF/EGFR 关键词，embedding 会漏掉（语义远）",
                "doc_snippet": d["doc"][:120],
            })

    # Embedding failures: 统计有多少 kinetic_rate 参数的 document 缺少反应物信息
    kinetic_without_reactants = 0
    kinetic_total = 0
    for d in documents:
        if d["type"] == "kinetic_rate":
            kinetic_total += 1
            if "EGF" not in d["doc"] and "EGFR" not in d["doc"]:
                kinetic_without_reactants += 1

    return {
        "type_conflicts": type_conflicts[:15],
        "embedding_failures": [
            {
                "issue": f"{kinetic_without_reactants}/{kinetic_total} kinetic_rate 参数的 document 不含反应物/产物信息",
                "impact": "embedding 模型无法将 k1/k2 与 EGF/EGFR 关联，导致语义检索失败",
            }
        ],
        "total_documents": len(documents),
        "kinetic_rate_total": kinetic_total,
        "kinetic_rate_without_reactants": kinetic_without_reactants,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 80)
    print("ROOT CAUSE RE-AUDIT: BIOMD0000000205 EGF-EGFR 信号通路")
    print("=" * 80)

    species, reactions, global_params = parse_sbml()
    current = load_current_kg_and_ode()

    print(f"\nSBML 模型: BIOMD0000000205 (Ung2008_EGFR_Endocytosis)")
    print(f"  Species: {len(species)}")
    print(f"  Reactions: {len(reactions)}")
    print(f"  Global params: {len(global_params)}")

    print(f"\n当前 KG/ODE 输出:")
    print(f"  Species: {current['species_names']}")
    print(f"  Edges: {len(current['edges'])}")
    print(f"  Params: {list(current['params'].keys())}")

    # --- Audit 1: KG → ODE consistency ---
    print("\n" + "=" * 80)
    print("AUDIT 1: KG → ODE Consistency")
    print("=" * 80)
    audit1 = audit_kg_ode_consistency(species, reactions, current)
    print(json.dumps(audit1, ensure_ascii=False, indent=2))

    # --- Audit 2: Parameter Ontology ---
    print("\n" + "=" * 80)
    print("AUDIT 2: Parameter Ontology Audit")
    print("=" * 80)
    audit2 = audit_parameter_ontology(species, reactions, global_params)
    print(json.dumps(audit2, ensure_ascii=False, indent=2))

    # --- Audit 3: RAG semantic correctness ---
    print("\n" + "=" * 80)
    print("AUDIT 3: RAG Semantic Correctness")
    print("=" * 80)
    audit3 = audit_rag_semantic(species, reactions)
    print(json.dumps(audit3, ensure_ascii=False, indent=2))

    # --- 汇总 ---
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    full_report = {
        "audit_1_kg_ode_consistency": audit1,
        "audit_2_parameter_ontology": audit2,
        "audit_3_rag_semantic": audit3,
    }
    output_path = Path(__file__).resolve().parent / "audit_report_biomd205.json"
    output_path.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整审计报告已保存: {output_path}")


if __name__ == "__main__":
    main()
