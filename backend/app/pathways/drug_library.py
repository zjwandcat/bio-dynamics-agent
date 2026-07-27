# BioDynamics Agent v4 - Canonical Drug Library Loader (N6 缺口 1)
# 加载 backend/knowledge/canonical/drug_library.yaml，提供药物-靶点 canonical 查询。
#
# 设计原则：
# 1. canonical 知识库驱动（非 per-case 硬编码），所有 specialist 共享同一份 drug library
# 2. 单例缓存：首次 load 时读取 YAML 并缓存 dict，避免重复磁盘 I/O
# 3. 容错降级：YAML 缺失或解析失败时返回空 dict，不阻塞 specialist 注册
# 4. 药物名归一化：查询时将 "-"/空格 统一为 "_" 比较（如 "ABT-199" == "ABT_199"，
#    "Nutlin-3" == "Nutlin_3"），同时保留原始名供 specialist 引用
#
# 依赖：
# - backend/knowledge/canonical/drug_library.yaml（canonical 药物-靶点知识库）
#
# 使用示例：
#   from app.pathways.drug_library import get_drug_entry, build_inhibitor_edge
#   trametinib = get_drug_entry("Trametinib")
#   edge = build_inhibitor_edge("Trametinib", "MEK", rate_constant=0.1)
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


# drug_library.yaml 的绝对路径（与 specialists 同包相对定位）
_DRUG_LIBRARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "knowledge",
    "canonical",
    "drug_library.yaml",
)


@lru_cache(maxsize=1)
def _load_drug_library() -> dict[str, dict[str, Any]]:
    """加载并缓存 drug_library.yaml，返回 {drug_name: {fields...}} 字典。

    容错：YAML 缺失或解析失败时返回空 dict，记录 warning。
    """
    try:
        import yaml  # PyYAML
    except ImportError:
        logger.warning(
            "drug_library: PyYAML 未安装，drug_library.yaml 加载失败（返回空 dict）"
        )
        return {}
    if not os.path.exists(_DRUG_LIBRARY_PATH):
        logger.warning(
            "drug_library: 文件不存在 %s（返回空 dict）", _DRUG_LIBRARY_PATH
        )
        return {}
    try:
        with open(_DRUG_LIBRARY_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("drug_library: 解析 YAML 失败: %s（返回空 dict）", exc)
        return {}
    drugs = data.get("drugs", {}) if isinstance(data, dict) else {}
    if not isinstance(drugs, dict):
        logger.warning("drug_library: 'drugs' 字段非 dict 类型（返回空 dict）")
        return {}
    logger.info("drug_library: 加载 %d 个药物", len(drugs))
    return drugs


def _normalize_drug_name(name: str) -> str:
    """药物名归一化：'-'/空格 → '_'，便于跨源匹配（ABT-199 ↔ ABT_199）。"""
    if not name:
        return ""
    return name.replace("-", "_").replace(" ", "_")


def get_drug_entry(drug_name: str) -> dict[str, Any]:
    """按药物名查询 canonical 条目（含 primary_target/mechanism/ic50/source_pmid）。

    Args:
        drug_name: 药物名（支持 ABT-199 / ABT_199 / Nutlin-3 / Nutlin_3 等变体）

    Returns:
        canonical 药物 dict（深拷贝，避免调用方污染缓存）。
        未命中时返回空 dict（不抛异常）。
    """
    if not drug_name:
        return {}
    drugs = _load_drug_library()
    # 1. 直接命中
    if drug_name in drugs:
        return dict(drugs[drug_name])
    # 2. 归一化匹配（ABT-199 ↔ ABT_199）
    norm = _normalize_drug_name(drug_name)
    for key, val in drugs.items():
        if _normalize_drug_name(key) == norm:
            return dict(val) if isinstance(val, dict) else {}
    return {}


def list_drugs() -> list[str]:
    """返回 drug_library 中所有药物名（原始键名，保留连字符/下划线变体）。"""
    return list(_load_drug_library().keys())


def build_inhibitor_edge(
    drug_name: str,
    target: str,
    rate_constant: float | None = None,
    product: str | None = None,
) -> dict[str, Any]:
    """构建 inhibitor edge Reaction IR（canonical 知识库驱动）。

    集成 drug_library 的 ic50/source_pmid/mechanism，生成符合 specialist 核心反应
    schema 的 inhibition 边，供 apply_core 返回 → specialist_hook 写回 KG/network_json。

    Args:
        drug_name: 药物名（如 "Trametinib"，支持 ABT-199/Nutlin-3 变体）
        target: 抑制靶点物种名（如 "MEK"，须与 specialist species 对齐）
        rate_constant: 抑制速率常数 kon（M^-1 min^-1），None 时按 IC50 估算
        product: 产物名（默认 "<drug>_<target>_inactive"，代表药物-靶点失活复合物）

    Returns:
        Reaction IR dict，含 source/target/mechanism/kinetics_type/substrate/product/
        modifier/drug_name/ic50_nM/ki_nM/source_pmid/rate_constant/mechanism_detail/
        pathway_tag 字段。drug_library 未命中时退化为最小 inhibition 边。
    """
    entry = get_drug_entry(drug_name)
    # 药物名规范化为物种 ID（ODE 变量名安全：仅字母数字下划线）
    drug_species = _normalize_drug_name(drug_name)
    target_safe = target or "Target"
    product_name = product or f"{drug_species}_{target_safe}_inactive"
    # 速率常数：显式传入 > IC50 估算 > 默认 1e6 M^-1 min^-1（k_on 期望区间内）
    ic50_nM = entry.get("ic50_nM")
    if rate_constant is None:
        # IC50(nM) → 近似 kon：k_on = ln(2) / (IC50_s)
        # IC50 越小（强抑制），kon 越大；用 1e6 / IC50_nM 作 heuristic
        if isinstance(ic50_nM, (int, float)) and ic50_nM > 0:
            rate_constant = 1e6 / max(ic50_nM, 0.1)
        else:
            rate_constant = 1e6  # 抗体或缺失 IC50 时用保守默认
    return {
        "source": drug_species,
        "target": target_safe,
        "mechanism": "inhibition",
        "kinetics_type": "mass_action",
        "substrate": target_safe,
        "product": product_name,
        "modifier": drug_species,
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        # === drug_library canonical 字段（显式 inhibitor edge 必备） ===
        "drug_name": drug_name,
        "ic50_nM": ic50_nM,
        "ki_nM": entry.get("ki_nM"),
        "source_pmid": entry.get("source_pmid"),
        "rate_constant": rate_constant,
        "mechanism_detail": entry.get("mechanism", "inhibition"),
        "primary_target": entry.get("primary_target", target_safe),
        "atc_code": entry.get("atc_code"),
        "fda_approved": entry.get("fda_approved"),
        "description": (
            f"{drug_name} 抑制 {target_safe}（{entry.get('mechanism', 'inhibition')}, "
            f"IC50={ic50_nM} nM, PMID:{entry.get('source_pmid', 'N/A')}, "
            f"canonical drug_library 驱动）"
            if entry
            else f"{drug_name} 抑制 {target_safe}（drug_library 未命中，最小 inhibition 边）"
        ),
    }


def build_drug_species(drug_name: str, compartment: str = "extracellular") -> dict[str, Any]:
    """构建药物物种定义（species_type="drug"，供 specialist core species 加入）。

    Args:
        drug_name: 药物名（如 "Trametinib"）
        compartment: 房室（默认 extracellular，小分子药物经血液循环到达靶组织）

    Returns:
        species dict，含 name/species_type/compartment/drug_name/source_pmid 字段。
    """
    entry = get_drug_entry(drug_name)
    drug_species = _normalize_drug_name(drug_name)
    return {
        "name": drug_species,
        "species_type": "drug",
        "compartment": compartment,
        # canonical 元数据（供 KG node 渲染与 PK/PD 推断）
        "drug_name": drug_name,
        "primary_target": entry.get("primary_target", ""),
        "ic50_nM": entry.get("ic50_nM"),
        "source_pmid": entry.get("source_pmid"),
        "fda_approved": entry.get("fda_approved", False),
    }


__all__ = [
    "get_drug_entry",
    "list_drugs",
    "build_inhibitor_edge",
    "build_drug_species",
    "_load_drug_library",
]
