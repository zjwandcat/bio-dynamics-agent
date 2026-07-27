"""任务 C：SBML Parser Pipeline 参数库扩充脚本

策略（用户明确要求）：
    BioModels → SBML Parser → Species → Reactions → Parameters → JSON → Embedding

铁律：
    1. 不用 LLM 提取参数（参数不能猜，必须可重复）
    2. 存结构化 JSON，不存一段文字
    3. 每个参数必须有 source_model + confidence + reaction context

输出格式（存入 ChromaDB v2 parameter collection）：
    {
        "pathway": "EGFR",
        "param_name": "k1",
        "value": 100.0,
        "unit": "uM_1_s_1",
        "species": "EGF, EGFR",
        "reactions": "EGF + EGFR -> EGF-EGFR",
        "source": "BioModels:BIOMD0000000205",
        "source_model": "BIOMD0000000205",
        "confidence": "HIGH",
        "context": "Reaction R1: EGF + EGFR -> EGF-EGFR (k1=100.0 uM_1_s_1)",
    }

用法：
    python scripts/sbml_pipeline_indexer.py                          # 索引 data/raw/ 顶层
    python scripts/sbml_pipeline_indexer.py --recursive              # 递归索引子目录
    python scripts/sbml_pipeline_indexer.py --file path/to/model.xml # 索引单个文件
    python scripts/sbml_pipeline_indexer.py --dry-run                # 只解析不入库
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

# 确保从 backend 目录运行
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

from app.rag_collections import get_rag_collections

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# SBML Parser（纯 ElementTree，不用 LLM，可重复）
# =============================================================================

# 动力学参数名白名单（大小写不敏感）
_KINETIC_PARAM_NAMES: set[str] = {
    "k1", "k2", "k_1", "k_2", "k3", "k4",
    "k_on", "k_off", "kon", "koff",
    "kcat", "k_cat",
    "Kd", "kd", "Ki", "ki", "Km", "km",
    "Vmax", "vmax",
    "V1", "V2", "V3",
    "k_deg", "kdegr", "k_prod", "k_syn", "k_sec",
    "k_act", "k_inact",
    "EC50", "IC50", "KEC50",
}

# 通路关键词映射（BIOMD ID → pathway name）
_BIOMD_PATHWAY_MAP: dict[str, str] = {
    "BIOMD0000000008": "WNT",
    "BIOMD0000000010": "MAPK",
    "BIOMD0000000022": "EGFR",
    # [Round 5] 新增：BIOMD0000000102 (Legewie 2006, PMID:17038645) 才是真正的
    # 凋亡 caspase 级联双稳态模型（含 Caspase-3/Apaf-1/XIAP，bistable switch）。
    # 原先 apoptosis_specialist.py 注释引用 BIOMD0000000335 是错误的 ——
    # 该 ID 在 BioModels 实为 Hockin2002_BloodCoagulation（凝血级联，非凋亡）。
    "BIOMD0000000102": "Apoptosis",
    "BIOMD0000000048": "CellCycle",
    "BIOMD0000000056": "CellCycle",
    "BIOMD0000000205": "EGFR",
    "BIOMD0000000224": "JAK_STAT",
    "BIOMD0000000250": "PI3K_AKT_MTOR",
    "BIOMD0000000252": "TGF_BETA",
    # [Round 5] 修复：BIOMD0000000255 实为 "Chen2009 - ErbB Signaling"（EGFR 家族信号通路），
    # 不是凋亡模型。原先误标记为 Apoptosis 导致 v2 RAG 参数库包含错误的凋亡参数。
    # BIOMD0000000332 实为凝血级联反应模型，非凋亡模型，已移除
    "BIOMD0000000255": "EGFR",
    "BIOMD0000000258": "NFKB",
    "BIOMD0000000262": "EGFR",
    "BIOMD0000000263": "EGFR",
    "BIOMD0000000264": "EGFR",
    # BIOMD0000000332 实为凝血级联反应模型，非凋亡模型，已移除
    # [Round 5] 注：BIOMD0000000335 在 BioModels 实为 Hockin2002_BloodCoagulation
    # （凝血级联），非 Eissing 2004 凋亡模型。改用 BIOMD0000000102 (Legewie2006)。
    "BIOMD0000000382": "P53",
    "BIOMD0000000666": "EGFR",
}


def _get_sbml_namespace(root: ET.Element) -> dict[str, str]:
    """提取 SBML 命名空间。"""
    if "}" in root.tag:
        ns_uri = root.tag.split("}")[0][1:]
        return {"sbml": ns_uri}
    return {}


def _find_all(elem: ET.Element, path: str, ns: dict[str, str]) -> list[ET.Element]:
    """兼容带命名空间和不带命名空间的查找。

    自动处理 .// 前缀：
      path="species" → "sbml:species" 或 "species"
      path=".//species" → ".//sbml:species" 或 ".//species"
      path="listOfReactants" → "sbml:listOfReactants" 或 "listOfReactants"
    """
    if ns:
        # 带 .// 前缀的 path
        if path.startswith(".//"):
            ns_path = f".//sbml:{path[3:]}"
        else:
            ns_path = f"sbml:{path}"
        result = elem.findall(ns_path, ns)
        if result:
            return result
    # 不带命名空间
    return elem.findall(path)


def _extract_species(model_elem: ET.Element, ns: dict[str, str]) -> dict[str, str]:
    """提取所有物种 id → name 映射。"""
    species_map: dict[str, str] = {}
    for sp in _find_all(model_elem, ".//species", ns):
        sp_id = sp.get("id", "")
        sp_name = sp.get("name") or sp_id
        if sp_id:
            species_map[sp_id] = sp_name
    return species_map


def _extract_compartments(model_elem: ET.Element, ns: dict[str, str]) -> set[str]:
    """[Round 5] 提取所有 compartment id（用于排除 kineticLaw math 中的体积引用）。

    许多 SBML 模型（如 Legewie2006）在 kineticLaw math 中引用 compartment 体积
    （例如 `cytosol * k1 * A * C9`），这些不是动力学参数，必须排除。
    """
    comp_ids: set[str] = set()
    for comp in _find_all(model_elem, ".//compartment", ns):
        comp_id = comp.get("id", "")
        if comp_id:
            comp_ids.add(comp_id)
    return comp_ids


def _extract_model_parameters(
    model_elem: ET.Element, ns: dict[str, str]
) -> dict[str, tuple[float, str]]:
    """[Round 5] 提取 model 级别的全局参数（listOfParameters）。

    许多 SBML 模型（如 Legewie2006 BIOMD0000000102）将所有动力学参数定义在
    `<model>/<listOfParameters>` 而非 `<kineticLaw>/<listOfParameters>`，
    kineticLaw 仅通过 MathML `<ci>` 引用它们。

    注意：必须只取 model 直接子级的 listOfParameters，不能深入 reaction 内部
    的 kineticLaw/listOfParameters（那些是反应局部参数）。

    Returns:
        {param_id: (value, units)} 字典
    """
    params: dict[str, tuple[float, str]] = {}
    # 遍历 model 直接子元素，找到 listOfParameters
    for child in model_elem:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag != "listOfParameters":
            continue
        # 在 listOfParameters 下查找 parameter 元素
        for param in child:
            ptag = param.tag.split("}")[-1] if "}" in param.tag else param.tag
            if ptag != "parameter":
                continue
            pname = param.get("id") or param.get("name", "")
            pvalue = param.get("value", "")
            punit = param.get("units", "")
            if not pname or pvalue == "":
                continue
            try:
                pvalue_float = float(pvalue)
            except (TypeError, ValueError):
                continue
            params[pname] = (pvalue_float, punit or "dimensionless")
    return params


def _parse_kineticlaw_ci_refs(
    rxn: ET.Element, ns: dict[str, str]
) -> list[str]:
    """[Round 5] 从 kineticLaw 的 MathML `<math>` 中提取所有 `<ci>` 引用的变量 ID。

    SBML kineticLaw 的速率方程用 MathML 表示，`<ci>` 元素引用变量（物种、
    compartment、参数）。此函数提取所有引用的 ID（去除前后空白）。

    例如 `<ci> k1 </ci>` → "k1"
    """
    refs: list[str] = []
    for kl in _find_all(rxn, ".//kineticLaw", ns):
        # MathML 命名空间独立于 SBML
        for ci in kl.iter():
            tag = ci.tag.split("}")[-1] if "}" in ci.tag else ci.tag
            if tag == "ci":
                text = (ci.text or "").strip()
                if text:
                    refs.append(text)
    return refs


def _extract_reaction_info(
    rxn: ET.Element, ns: dict[str, str], species_map: dict[str, str]
) -> dict[str, Any]:
    """提取单个反应的完整信息：reactants → products + reaction name。"""
    rxn_id = rxn.get("id", "")
    rxn_name = rxn.get("name") or rxn_id

    reactants: list[str] = []
    products: list[str] = []

    # 分别从 listOfReactants 和 listOfProducts 提取
    for lr in _find_all(rxn, "listOfReactants", ns):
        for sr in _find_all(lr, "speciesReference", ns):
            sp_ref = sr.get("species", "")
            sp_name = species_map.get(sp_ref, sp_ref)
            if sp_name:
                reactants.append(sp_name)

    for lp in _find_all(rxn, "listOfProducts", ns):
        for sr in _find_all(lp, "speciesReference", ns):
            sp_ref = sr.get("species", "")
            sp_name = species_map.get(sp_ref, sp_ref)
            if sp_name:
                products.append(sp_name)

    # 构建反应描述：A + B -> C
    reactant_str = " + ".join(reactants) if reactants else "?"
    product_str = " + ".join(products) if products else "?"
    reaction_desc = f"{reactant_str} -> {product_str}"

    return {
        "rxn_id": rxn_id,
        "rxn_name": rxn_name,
        "reactants": reactants,
        "products": products,
        "reaction_desc": reaction_desc,
    }


def _extract_parameters(
    rxn: ET.Element,
    ns: dict[str, str],
    rxn_info: dict[str, Any],
    model_id: str,
    pathway: str,
    model_params: dict[str, tuple[float, str]] | None = None,
    species_ids: set[str] | None = None,
    compartment_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """从 kineticLaw 中提取动力学参数。

    [Round 5] 支持两种 SBML 参数定义方式：
    1. 参数定义在 `<kineticLaw>/<listOfParameters>` 内（原有逻辑）
    2. 参数定义在 `<model>/<listOfParameters>` 全局，kineticLaw 通过
       MathML `<ci>` 引用（新增逻辑，如 Legewie2006 BIOMD0000000102）

    对于方式 2，从 kineticLaw math 的 `<ci>` 引用中找出参数引用，
    排除物种和 compartment，然后在 model_params 字典中查找其值。
    被 kineticLaw math 引用的参数按定义是动力学参数，跳过白名单过滤。
    """
    params: list[dict[str, Any]] = []

    # --- 方式 1：kineticLaw 内部局部参数（原有逻辑，保持向后兼容）---
    has_local_params = False
    for kl in _find_all(rxn, ".//kineticLaw", ns):
        for param in _find_all(kl, ".//parameter", ns):
            has_local_params = True
            pname = param.get("id") or param.get("name", "")
            pvalue = param.get("value", "")
            punit = param.get("units", "")

            if not pname or pvalue == "":
                continue

            # 仅保留动力学相关参数名（大小写不敏感）
            if pname not in _KINETIC_PARAM_NAMES and pname.lower() not in {p.lower() for p in _KINETIC_PARAM_NAMES}:
                continue

            try:
                pvalue_float = float(pvalue)
            except (TypeError, ValueError):
                continue

            # 构建结构化参数记录
            species_str = ", ".join(
                set(rxn_info["reactants"] + rxn_info["products"])
            )[:200]  # 截断防止 metadata 过大

            context = (
                f"Reaction {rxn_info['rxn_id']}: {rxn_info['reaction_desc']} "
                f"({pname}={pvalue_float} {punit})"
            )

            params.append({
                "pathway": pathway,
                "param_name": pname,
                "value": pvalue_float,
                "unit": punit or "dimensionless",
                "species": species_str,
                "reactions": rxn_info["reaction_desc"],
                "source": f"BioModels:{model_id}",
                "source_model": model_id,
                "confidence": "HIGH",  # SBML 已发表模型，置信度高
                "context": context,
                "type": "kinetic_rate",
            })

    # --- 方式 2：model 级别全局参数（通过 MathML <ci> 引用，新增）---
    # 仅当该反应的 kineticLaw 没有局部参数时启用
    if not has_local_params and model_params:
        # 收集需要排除的 ID（物种 + compartment）
        excluded_ids: set[str] = set()
        if species_ids:
            excluded_ids |= species_ids
        if compartment_ids:
            excluded_ids |= compartment_ids

        # 从 kineticLaw math 中提取 <ci> 引用
        ci_refs = _parse_kineticlaw_ci_refs(rxn, ns)

        # 去重，保持顺序
        seen: set[str] = set()
        for ref_id in ci_refs:
            if ref_id in seen:
                continue
            seen.add(ref_id)
            # 排除物种和 compartment
            if ref_id in excluded_ids:
                continue
            # 必须在 model_params 中存在
            if ref_id not in model_params:
                continue
            pvalue_float, punit = model_params[ref_id]

            # 被 kineticLaw math 引用的参数按定义是动力学参数，跳过白名单

            species_str = ", ".join(
                set(rxn_info["reactants"] + rxn_info["products"])
            )[:200]

            context = (
                f"Reaction {rxn_info['rxn_id']}: {rxn_info['reaction_desc']} "
                f"({ref_id}={pvalue_float} {punit})"
            )

            params.append({
                "pathway": pathway,
                "param_name": ref_id,
                "value": pvalue_float,
                "unit": punit,
                "species": species_str,
                "reactions": rxn_info["reaction_desc"],
                "source": f"BioModels:{model_id}",
                "source_model": model_id,
                "confidence": "HIGH",
                "context": context,
                "type": "kinetic_rate",
            })

    return params


def parse_sbml_pipeline(
    sbml_text: str, model_id: str, model_name: str = ""
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """SBML Parser Pipeline（纯规则，不用 LLM，可重复）。

    Pipeline:
        SBML XML → ElementTree parse → Species → Reactions → Parameters → JSON

    Args:
        sbml_text: SBML XML 文本。
        model_id: BioModels ID（如 BIOMD0000000205）。
        model_name: 模型名称（可选）。

    Returns:
        (mechanism_records, parameter_records, summary)
    """
    try:
        root = ET.fromstring(sbml_text)
    except ET.ParseError as exc:
        logger.warning("SBML 解析失败 (%s): %s", model_id, exc)
        return [], [], {"error": str(exc)}

    ns = _get_sbml_namespace(root)
    model_elem = _find_all(root, "model", ns)
    model_elem = model_elem[0] if model_elem else root

    # 1. 提取物种
    species_map = _extract_species(model_elem, ns)
    species_list = list(species_map.values())
    species_ids = set(species_map.keys())

    # [Round 5] 提取 compartment（排除 kineticLaw math 中的体积引用）
    compartment_ids = _extract_compartments(model_elem, ns)

    # [Round 5] 提取 model 级别全局参数（Legewie2006 等模型使用此方式）
    model_params = _extract_model_parameters(model_elem, ns)

    # 2. 确定通路
    pathway = _BIOMD_PATHWAY_MAP.get(model_id, model_name or model_id)

    # 3. 提取反应 + 参数
    mechanism_records: list[dict[str, Any]] = []
    parameter_records: list[dict[str, Any]] = []
    reactions_info: list[str] = []

    for rxn in _find_all(model_elem, ".//reaction", ns):
        rxn_info = _extract_reaction_info(rxn, ns, species_map)
        reactions_info.append(rxn_info["reaction_desc"])

        # 提取参数（[Round 5] 传入 model 级别参数、物种 ID、compartment ID
        # 以支持 Legewie2006 等模型的全局参数引用方式）
        params = _extract_parameters(
            rxn, ns, rxn_info, model_id, pathway,
            model_params=model_params,
            species_ids=species_ids,
            compartment_ids=compartment_ids,
        )
        parameter_records.extend(params)

    # 4. 构建机制记录
    if species_list or reactions_info:
        mechanism_records.append({
            "pathway": pathway,
            "entities": species_list[:50],
            "interactions": reactions_info[:50],
            "description": f"SBML model {model_id} ({pathway}): "
                          f"{len(species_list)} species, {len(reactions_info)} reactions, "
                          f"{len(parameter_records)} kinetic parameters",
            "source": f"BioModels:{model_id}",
            "source_model": model_id,
            "pmid": "",
        })

    summary = {
        "model_id": model_id,
        "pathway": pathway,
        "species_count": len(species_list),
        "reaction_count": len(reactions_info),
        "parameter_count": len(parameter_records),
    }

    return mechanism_records, parameter_records, summary


# =============================================================================
# 索引入库
# =============================================================================

def _read_sbml_text(sbml_path: Path) -> str | None:
    """读取 SBML 文本，自动识别 ZIP 归档与纯 XML。

    - 纯 XML：尝试 utf-8-sig / utf-8 / latin-1 解码
    - ZIP 归档（BioModels 完整包）：提取 {model_id}_url.xml 或 _urn.xml
    """
    # 先读前 4 字节判断是否为 ZIP
    try:
        with open(sbml_path, "rb") as f:
            magic = f.read(4)
    except Exception:
        return None

    # ZIP magic: PK\x03\x04
    if magic[:2] == b"PK":
        import zipfile
        try:
            with zipfile.ZipFile(sbml_path) as zf:
                # 优先 _url.xml，其次 _urn.xml
                names = zf.namelist()
                target = None
                for suffix in ("_url.xml", "_urn.xml"):
                    for n in names:
                        if n.endswith(suffix):
                            target = n
                            break
                    if target:
                        break
                if not target:
                    return None
                return zf.read(target).decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("ZIP 解压失败 %s: %s", sbml_path.name, exc)
            return None

    # 纯 XML：尝试多种编码
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return sbml_path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None


def index_sbml_file(
    sbml_path: Path,
    rag_collections,
    dry_run: bool = False,
) -> dict[str, Any]:
    """解析单个 SBML 文件并入库。"""
    model_id = sbml_path.stem  # 文件名即 BIOMD ID
    model_name = _BIOMD_PATHWAY_MAP.get(model_id, model_id)

    logger.info("解析: %s (pathway=%s)", sbml_path.name, model_name)

    sbml_text = _read_sbml_text(sbml_path)
    if sbml_text is None:
        logger.error("读取失败 %s: 无法解码或非 XML/ZIP", sbml_path)
        return {"file": str(sbml_path), "error": "read failed"}

    mech_records, param_records, summary = parse_sbml_pipeline(
        sbml_text, model_id, model_name
    )

    logger.info(
        "  → %d species, %d reactions, %d parameters",
        summary.get("species_count", 0),
        summary.get("reaction_count", 0),
        summary.get("parameter_count", 0),
    )

    if dry_run:
        # 打印前 3 个参数示例
        for p in param_records[:3]:
            logger.info(
                "  [DRY] %s=%s %s | %s | rxn=%s",
                p["param_name"], p["value"], p["unit"],
                p["source"], p["reactions"],
            )
        return summary

    # 入库（upsert_parameter/upsert_mechanism 接收 list[dict]）
    if not rag_collections.available:
        logger.error("RagCollections 不可用，跳过入库")
        return summary

    if mech_records:
        try:
            n = rag_collections.upsert_mechanism(mech_records)
            logger.info("  机制入库: %d", n)
        except Exception as exc:
            logger.warning("  机制入库失败: %s", exc)

    if param_records:
        try:
            n = rag_collections.upsert_parameter(param_records)
            logger.info("  参数入库: %d", n)
        except Exception as exc:
            logger.warning("  参数入库失败: %s", exc)

    return summary


def main():
    parser = argparse.ArgumentParser(description="SBML Parser Pipeline 参数库扩充")
    parser.add_argument(
        "--raw-dir", default="data/raw",
        help="SBML 文件目录（默认 data/raw）"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="递归扫描子目录（索引 data/raw/biomodels/ 等）"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="索引单个文件"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只解析不入库"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SBML Parser Pipeline 参数库扩充")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("模式: DRY RUN（只解析不入库）")
    else:
        rag_collections = get_rag_collections()
        if not rag_collections.available:
            logger.error("RagCollections 不可用，退出")
            return
        before_stats = rag_collections.stats()
        logger.info("入库前: %s", before_stats)

    # 收集 SBML 文件
    sbml_files: list[Path] = []
    if args.file:
        sbml_files = [Path(args.file)]
    else:
        raw_dir = Path(args.raw_dir)
        if not raw_dir.exists():
            logger.error("目录不存在: %s", raw_dir)
            return
        pattern = "**/*.xml" if args.recursive else "*.xml"
        sbml_files = sorted(raw_dir.glob(pattern))
        # 也匹配 .sbml 文件
        sbml_files.extend(sorted(raw_dir.glob(pattern.replace(".xml", ".sbml"))))

    logger.info("找到 %d 个 SBML 文件", len(sbml_files))

    # 逐个解析入库
    all_summaries: list[dict] = []
    for sbml_path in sbml_files:
        try:
            summary = index_sbml_file(
                sbml_path,
                rag_collections if not args.dry_run else None,
                dry_run=args.dry_run,
            )
            all_summaries.append(summary)
        except Exception as exc:
            logger.error("处理失败 %s: %s", sbml_path, exc)

    # 汇总
    logger.info("=" * 60)
    logger.info("汇总")
    logger.info("=" * 60)

    total_species = sum(s.get("species_count", 0) for s in all_summaries)
    total_reactions = sum(s.get("reaction_count", 0) for s in all_summaries)
    total_params = sum(s.get("parameter_count", 0) for s in all_summaries)

    logger.info("处理文件: %d", len(all_summaries))
    logger.info("总物种: %d", total_species)
    logger.info("总反应: %d", total_reactions)
    logger.info("总参数: %d", total_params)

    if not args.dry_run:
        after_stats = rag_collections.stats()
        logger.info("入库后: %s", after_stats)
        logger.info(
            "增量: mechanism +%d, parameter +%d",
            after_stats.get("mechanism", 0) - before_stats.get("mechanism", 0),
            after_stats.get("parameter", 0) - before_stats.get("parameter", 0),
        )


if __name__ == "__main__":
    main()
