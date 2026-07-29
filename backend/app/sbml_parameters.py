"""Deterministic kinetic-parameter extraction from SBML.

The benchmark pipeline already loads governed BioModels SBML before parameter
grounding. This module turns that local model into auditable per-edge
parameters without issuing one remote RAG request per reaction edge.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from defusedxml.ElementTree import fromstring

logger = logging.getLogger(__name__)


# =============================================================================
# [RCA-20 修复 A] SBML 参数生物合理性校验层
# =============================================================================
# 根因（详见 P1_OVERACTIVATION_RCA.md 根因 1）：
#   SBML 原生参数（如 JAK2ActEpo=633,250）被无校验直接注入 ODE，
#   导致 fold_change 爆炸到 57~605x（期望 5~15x）。
#
# 修复：在 ground_sbml_parameters_to_edges 注入参数前，按 mechanism 类型
# 对 value 做生物学合理性 clamp。超界值截断到范围内并记录 warning，
# 标记 is_fallback=True 以提示下游降级处理。
#
# Feature Flag: V4_SBML_PARAM_RANGE_CHECK_ENABLED（默认 OFF）
# 回退安全：Flag OFF 时直接走原逻辑（无校验），完全恢复 v3 行为
# =============================================================================

_V4_SBML_PARAM_RANGE_CHECK_ENABLED = (
    os.environ.get("V4_SBML_PARAM_RANGE_CHECK_ENABLED", "false").lower() == "true"
)

# 按 mechanism 类型定义参数的生物学合理范围（基于文献动力学常数典型值）
# 来源：BioNumbers、SBioBox、Kegg Reaction 参数库
# 范围宽松（10.0 上限）以避免误伤合法参数，仅截断明显异常值（如 633,250）
_MECHANISM_PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "activation": {
        "k_cat": (1e-4, 10.0), "kact": (1e-4, 10.0), "k_act": (1e-4, 10.0),
        "k1": (1e-4, 10.0), "k1f": (1e-4, 10.0), "kf": (1e-4, 10.0),
        "vmax": (1e-4, 100.0), "k": (1e-4, 10.0),
    },
    "phosphorylation": {
        "k_cat": (1e-4, 10.0), "kcat": (1e-4, 10.0),
        "k_phos": (1e-4, 10.0), "kphos": (1e-4, 10.0),
        "km": (1e-6, 100.0), "Km": (1e-6, 100.0),
        "k_dephos": (1e-4, 1.0), "kdephos": (1e-4, 1.0), "k2": (1e-4, 1.0),
    },
    "dephosphorylation": {
        "k_dephos": (1e-4, 1.0), "kdephos": (1e-4, 1.0), "k2": (1e-4, 1.0),
    },
    "inhibition": {
        "k_inhibit": (1e-4, 10.0), "ki": (1e-4, 10.0), "kd": (1e-4, 100.0),
        "ic50": (1e-4, 100.0), "koff": (1e-4, 10.0), "k1b": (1e-4, 10.0),
    },
    "binding": {
        "kon": (1e-4, 100.0), "k1f": (1e-4, 100.0), "kf": (1e-4, 100.0),
        "koff": (1e-4, 10.0), "k1b": (1e-4, 10.0), "kb": (1e-4, 10.0),
        "kd": (1e-4, 100.0),
    },
    "degradation": {
        "k_deg": (1e-4, 1.0), "kdeg": (1e-4, 1.0),
        "kdegr": (1e-4, 1.0), "decay": (1e-4, 1.0),
    },
}


def _clamp_sbml_param_value(
    selected: SBMLKineticParameter,
    mechanism: str,
) -> tuple[SBMLKineticParameter, str, float | None]:
    """对 SBML 参数值做生物学合理性 clamp。

    Args:
        selected: 原始 SBML 参数（dataclass，frozen）
        mechanism: 边的 mechanism 类型（如 "activation"/"phosphorylation"）

    Returns:
        tuple: (处理后的 selected, clamp_status, original_value)
        - clamp_status: "pass" | "clamped" | "no_range_defined"
        - original_value: 原始值（仅 clamp 时非 None）
    """
    if not _V4_SBML_PARAM_RANGE_CHECK_ENABLED:
        return selected, "disabled", None

    mechanism_lower = mechanism.lower()
    param_ranges = _MECHANISM_PARAM_RANGES.get(mechanism_lower)
    if not param_ranges:
        # 未定义范围的 mechanism 不做校验（保守策略，避免误伤）
        return selected, "no_range_defined", None

    param_name_lower = selected.param_name.lower()
    range_info = param_ranges.get(param_name_lower)
    if not range_info:
        # 参数名不在范围字典中，尝试模糊匹配（如 JAK2ActEpo 含 "act"）
        # 仅对 activation 机制尝试模糊匹配（因为 JAK2ActEpo 类参数最常见）
        if mechanism_lower == "activation":
            for key, rng in param_ranges.items():
                if key in param_name_lower or param_name_lower in key:
                    range_info = rng
                    break
        if not range_info:
            return selected, "no_range_defined", None

    lo, hi = range_info
    original_value = selected.value
    if not math.isfinite(original_value):
        # 非有限值（inf/nan）直接降级到下限
        clamped_value = lo
        logger.warning(
            "SBML param %s=%r non-finite for mechanism=%s, clamped to %g",
            selected.param_name, original_value, mechanism, clamped_value,
        )
        return replace(selected, value=clamped_value), "clamped", original_value

    if lo <= original_value <= hi:
        return selected, "pass", None

    # 超界：clamp 到范围内
    clamped_value = max(lo, min(hi, original_value))
    logger.warning(
        "SBML param %s=%g out of range [%g, %g] for mechanism=%s, clamped to %g",
        selected.param_name, original_value, lo, hi, mechanism, clamped_value,
    )
    return replace(selected, value=clamped_value), "clamped", original_value


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


@dataclass(frozen=True)
class SBMLKineticParameter:
    model_id: str
    reaction_id: str
    reaction_name: str
    participants: tuple[str, ...]
    param_name: str
    value: float
    unit: str
    scope: str
    # cache 治理字段（用于版本追踪与 cache 一致性校验）
    sbml_sha256: str = ""
    sbml_source_url: str = ""
    cache_timestamp: str = ""
    sbml_version: str = ""
    cache_stale: bool = False


def compute_sbml_cache_provenance(
    sbml_text: str,
    model_id: str,
) -> dict[str, Any]:
    """Compute cache governance fields for an SBML document.

    与 ``biomodels_client._checksum_matches_manifest`` 配合：当本地 cache 的
    sha256 与 manifest 不一致时，``cache_stale=True`` 并触发重新下载（调用
    ``BioModelsAPIClient.download(model_id)`` 刷新本地缓存文件）。

    Returns:
        dict with keys: sbml_sha256 / sbml_source_url / cache_timestamp /
        sbml_version / cache_stale
    """
    payload = (sbml_text or "").encode("utf-8")
    sha256 = hashlib.sha256(payload).hexdigest() if payload else ""
    source_url = (
        f"https://www.ebi.ac.uk/biomodels/model/download/{model_id}.xml"
        if model_id
        else ""
    )
    cache_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    sbml_version = ""
    cache_stale = False
    try:
        root = fromstring(sbml_text) if sbml_text.strip() else None
    except Exception as exc:  # pragma: no cover - defensive: callers already parse
        logger.debug("cache provenance 解析 SBML 失败 (%s): %s", model_id, exc)
        root = None
    if root is not None:
        level = root.get("level", "")
        version = root.get("version", "")
        if level and version:
            sbml_version = f"L{level}V{version}"

    # 与 manifest 比对：manifest 无条目视为通过（cache_stale=False）
    if model_id and payload:
        try:
            from app.biomodels_client import BioModelsAPIClient
            if not BioModelsAPIClient._checksum_matches_manifest(model_id, payload):
                cache_stale = True
                logger.warning(
                    "SBML cache sha256 与 manifest 不一致 (model=%s)，标记 cache_stale",
                    model_id,
                )
                # 触发重新下载：download() 会重新校验本地缓存并在不一致时在线拉取
                try:
                    refreshed = BioModelsAPIClient().download(model_id)
                    if refreshed:
                        logger.info(
                            "SBML cache 已触发重新下载 (model=%s, bytes=%d)",
                            model_id, len(refreshed),
                        )
                except Exception as exc:
                    logger.warning(
                        "SBML cache 重新下载失败 (model=%s): %s", model_id, exc,
                    )
        except Exception as exc:
            logger.debug("cache provenance manifest 校验失败 (%s): %s", model_id, exc)

    return {
        "sbml_sha256": sha256,
        "sbml_source_url": source_url,
        "cache_timestamp": cache_timestamp,
        "sbml_version": sbml_version,
        "cache_stale": cache_stale,
    }


def extract_sbml_kinetic_parameters(
    sbml_text: str,
    model_id: str,
) -> list[SBMLKineticParameter]:
    """Extract reaction-local and referenced global parameters from SBML."""
    if not sbml_text.strip():
        return []
    root = fromstring(sbml_text)
    model = next((e for e in root.iter() if _local_name(e.tag) == "model"), None)
    if model is None:
        return []

    # 计算 cache 治理字段（同一模型的所有参数共享同一份 provenance）
    provenance = compute_sbml_cache_provenance(sbml_text, model_id)

    species_names: dict[str, str] = {}
    global_params: dict[str, tuple[float, str]] = {}
    for child in model:
        child_name = _local_name(child.tag)
        if child_name == "listOfSpecies":
            for species in child:
                if _local_name(species.tag) != "species":
                    continue
                species_id = species.get("id", "")
                if species_id:
                    species_names[species_id] = species.get("name", "") or species_id
        elif child_name == "listOfParameters":
            for parameter in child:
                parsed = _parse_parameter(parameter)
                if parsed is not None:
                    global_params[parsed[0]] = (parsed[1], parsed[2])

    extracted: list[SBMLKineticParameter] = []
    for reaction in (e for e in model.iter() if _local_name(e.tag) == "reaction"):
        reaction_id = reaction.get("id", "")
        reaction_name = reaction.get("name", "") or reaction_id
        participants: list[str] = []
        kinetic_law = None
        for element in reaction.iter():
            name = _local_name(element.tag)
            if name in {"speciesReference", "modifierSpeciesReference"}:
                species_id = element.get("species", "")
                if species_id:
                    participants.extend((species_id, species_names.get(species_id, species_id)))
            elif name == "kineticLaw":
                kinetic_law = element
        if kinetic_law is None:
            continue

        local_found = False
        for element in kinetic_law.iter():
            if _local_name(element.tag) not in {"parameter", "localParameter"}:
                continue
            parsed = _parse_parameter(element)
            if parsed is None:
                continue
            local_found = True
            extracted.append(
                SBMLKineticParameter(
                    model_id=model_id,
                    reaction_id=reaction_id,
                    reaction_name=reaction_name,
                    participants=tuple(dict.fromkeys(participants)),
                    param_name=parsed[0],
                    value=parsed[1],
                    unit=parsed[2],
                    scope="local",
                    sbml_sha256=provenance["sbml_sha256"],
                    sbml_source_url=provenance["sbml_source_url"],
                    cache_timestamp=provenance["cache_timestamp"],
                    sbml_version=provenance["sbml_version"],
                    cache_stale=provenance["cache_stale"],
                )
            )

        referenced: set[str] = set()
        for element in kinetic_law.iter():
            if _local_name(element.tag) == "ci" and element.text:
                candidate = element.text.strip()
                if candidate in global_params:
                    referenced.add(candidate)
        formula = kinetic_law.get("formula", "")
        if formula:
            for candidate in re.findall(r"[A-Za-z_]\w*", formula):
                if candidate in global_params:
                    referenced.add(candidate)
        if not local_found or referenced:
            for param_name in sorted(referenced):
                value, unit = global_params[param_name]
                extracted.append(
                    SBMLKineticParameter(
                        model_id=model_id,
                        reaction_id=reaction_id,
                        reaction_name=reaction_name,
                        participants=tuple(dict.fromkeys(participants)),
                        param_name=param_name,
                        value=value,
                        unit=unit,
                        scope="global_reference",
                        sbml_sha256=provenance["sbml_sha256"],
                        sbml_source_url=provenance["sbml_source_url"],
                        cache_timestamp=provenance["cache_timestamp"],
                        sbml_version=provenance["sbml_version"],
                        cache_stale=provenance["cache_stale"],
                    )
                )

    # Some governed ODE models (notably BIOMD0000000252) encode the complete
    # system as rate rules rather than reactions. Treat each rule as a
    # reaction-like kinetic context so it receives the same provenance path.
    for rule in (e for e in model.iter() if _local_name(e.tag) == "rateRule"):
        variable = rule.get("variable", "")
        participants = [variable, species_names.get(variable, variable)] if variable else []
        referenced: set[str] = set()
        for element in rule.iter():
            if _local_name(element.tag) != "ci" or not element.text:
                continue
            candidate = element.text.strip()
            if candidate in global_params:
                referenced.add(candidate)
            if candidate in species_names:
                participants.extend((candidate, species_names[candidate]))
        for param_name in sorted(referenced):
            value, unit = global_params[param_name]
            extracted.append(
                SBMLKineticParameter(
                    model_id=model_id,
                    reaction_id=f"rateRule:{variable}",
                    reaction_name=f"rate rule for {species_names.get(variable, variable)}",
                    participants=tuple(dict.fromkeys(participants)),
                    param_name=param_name,
                    value=value,
                    unit=unit,
                    scope="rate_rule",
                    sbml_sha256=provenance["sbml_sha256"],
                    sbml_source_url=provenance["sbml_source_url"],
                    cache_timestamp=provenance["cache_timestamp"],
                    sbml_version=provenance["sbml_version"],
                    cache_stale=provenance["cache_stale"],
                )
            )
    return extracted


def _parse_parameter(element: Any) -> tuple[str, float, str] | None:
    param_name = element.get("id", "") or element.get("name", "")
    raw_value = element.get("value")
    if not param_name or raw_value is None:
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value == 0.0:
        return None
    return param_name, value, element.get("units", "") or "SBML_native"


def ground_sbml_parameters_to_edges(
    edges: list[dict[str, Any]],
    models: list[dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Map governed SBML parameters onto a reduced mechanism graph."""
    candidates: list[SBMLKineticParameter] = []
    parse_errors: list[str] = []
    for model in models:
        model_id = str(model.get("model_id", ""))
        sbml_text = str(model.get("sbml_text", ""))
        try:
            candidates.extend(extract_sbml_kinetic_parameters(sbml_text, model_id))
        except Exception as exc:
            parse_errors.append(f"{model_id}:{type(exc).__name__}")
    if not candidates:
        return {}, {}, {"candidate_count": 0, "parse_errors": parse_errors}

    parameters: dict[str, dict[str, Any]] = {}
    decisions: dict[str, dict[str, Any]] = {}
    direct_matches = 0
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        edge_key = f"{source}->{target}"
        mechanism = str(edge.get("mechanism") or edge.get("interaction") or "activation")
        ranked = sorted(
            candidates,
            key=lambda item: (
                -_reaction_match_score(item, source, target),
                _parameter_preference(item.param_name, mechanism),
                item.model_id,
                item.reaction_id,
                item.param_name,
            ),
        )
        selected = ranked[0]
        match_score = _reaction_match_score(selected, source, target)
        mapping_method = "participant_match" if match_score > 0 else "model_mechanism_reuse"
        direct_matches += int(match_score > 0)
        origin = f"{selected.model_id}:{selected.reaction_id}:{selected.param_name}"

        # [RCA-20 修复 A] SBML 参数生物合理性校验
        # 在注入 parameters 字典前对 selected.value 做 clamp
        selected, clamp_status, original_value = _clamp_sbml_param_value(selected, mechanism)
        is_clamped = clamp_status == "clamped"

        parameters[edge_key] = {
            "edge_key": edge_key,
            "param_name": selected.param_name,
            "value": selected.value,
            "unit": selected.unit,
            "source": "SBML",
            "confidence": 0.95 if match_score > 0 else 0.8,
            "confidence_label": "HIGH",
            "origin": origin,
            "biomd_id": selected.model_id if selected.model_id else None,
            "source_model": selected.model_id,
            "reaction_id": selected.reaction_id,
            "parameter_scope": selected.scope,
            "mapping_method": mapping_method,
            # [RCA-20 修复 A] clamp 后的参数标记为 fallback 以提示下游降级
            "is_fallback": is_clamped,
            "missing_parameter": False,
            # 新增字段：参数校验状态（供审计追溯）
            "param_clamp_status": clamp_status,
            "param_original_value": original_value,
        }
        decisions[edge_key] = {
            "param_found": True,
            "selected_params": [{
                "param_name": selected.param_name,
                "value": selected.value,
                "unit": selected.unit,
                "source": origin,
            }],
            "reasoning": (
                f"Governed canonical SBML parameter ({mapping_method})"
                + (f"; [RCA-20] clamped from {original_value}" if is_clamped else "")
            ),
            "fallback_to_estimation": False,
        }
    return parameters, decisions, {
        "candidate_count": len(candidates),
        "direct_match_count": direct_matches,
        "reuse_count": len(edges) - direct_matches,
        "models": sorted({candidate.model_id for candidate in candidates}),
        "parse_errors": parse_errors,
    }


def _reaction_match_score(item: SBMLKineticParameter, source: str, target: str) -> int:
    haystack = {_token(value) for value in (*item.participants, item.reaction_id, item.reaction_name)}
    score = 0
    for needle in (_token(source), _token(target)):
        if not needle:
            continue
        if needle in haystack:
            score += 4
        elif any(needle in value or value in needle for value in haystack if value):
            score += 2
    return score


def _parameter_preference(param_name: str, mechanism: str) -> int:
    name = _token(param_name)
    mechanism = mechanism.lower()
    if mechanism in {"binding", "recruitment", "dissociation"}:
        preferred = ("kon", "k1f", "kf", "koff", "k1b", "kb", "kd")
    elif mechanism in {"phosphorylation", "dephosphorylation"}:
        preferred = ("kcat", "kphos", "kdephos", "vmax", "km", "k1f", "kf")
    elif mechanism == "degradation":
        preferred = ("kdeg", "kdegr", "decay", "degradation")
    elif mechanism == "inhibition":
        preferred = ("ki", "kd", "ic50", "koff", "k1b")
    else:
        preferred = ("kcat", "kact", "k1f", "kf", "vmax", "k")
    for index, marker in enumerate(preferred):
        if marker in name:
            return index
    return len(preferred) + 1


__all__ = [
    "SBMLKineticParameter",
    "compute_sbml_cache_provenance",
    "extract_sbml_kinetic_parameters",
    "ground_sbml_parameters_to_edges",
]
