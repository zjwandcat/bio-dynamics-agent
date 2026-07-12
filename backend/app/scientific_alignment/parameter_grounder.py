# BioDynamics Agent v4 - Scientific Alignment Loop: Parameter Grounding (Task 7 SubTask 7.1)
#
# 参数接地（Parameter Grounding）：为 LLM 生成的 ODE 参数提供"先验分布"依据，
# 避免 LLM 直接"拍脑袋"出数字。
#
# 设计目标：
#   1. BioModels 中位数（confidence=High）：从已校准的 SBML 模型提取参数中位数
#   2. 文献范围（confidence=Medium）：从 RAG 检索的文献参数取 min/max 范围
#   3. 反应类型先验（confidence=Low）：硬编码的通用生物学先验（如 binding 的 kd 范围）
#   4. 估算 fallback（confidence=Low）：三层均缺失时，返回 value=0 的占位先验
#
# 优先级：BioModels > Literature > Reaction_type > fallback
# 同一 param_name 仅保留最高优先级来源的先验。
#
# Feature Flag 守护：
#   SA_PARAMETER_PRIOR 默认 OFF。关闭时返回空 Prior 报告（skipped=True），不阻塞主流程。
#   铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
#         （由 settings.is_sa_feature_enabled 强制校验）。
#
# 单位约定（本模块）：
#   - 浓度：nM（与 rag_client._to_nM 一致）
#   - 时间：分钟 min（注意：rag_client._to_hours 用 h，本模块按 Task 7 约定用 min）
#   - 速率：per minute（不使用 per second）
#   生成 ParameterPrior 时对已知单位做归一化；未知/复合单位原样返回。
#
# 安全设计：
#   - 不引入新依赖（仅标准库 + app.config.settings）
#   - 不 import rag_client（避免循环依赖），本文件内实现轻量单位归一化
#   - _REACTION_TYPE_PRIORS 为通用生物学先验（经典文献共识），非通路特定数据
#   - 所有通路特定数据来自输入参数（biomodels_params / literature_params）
#
# 依赖：app.config.settings；不引入新依赖。
#
# 核心导出：
#   from app.scientific_alignment.parameter_grounder import (
#       ParameterPrior,
#       ParameterPriorReport,
#       build_parameter_prior,
#       get_reaction_type_prior,
#   )

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 反应类型先验表（通用生物学先验，非通路特定）
# =============================================================================
# 数据来源：经典系统生物学文献共识（如 Tyson et al. 2011《Design principles of
# biochemical oscillators》等综述），反映常见反应类型参数的典型数量级范围。
# 表内单位为文献常用单位（nM / uM / per_s / per_min / min），生成 ParameterPrior
# 时由 _convert_from_table_unit 统一归一到项目约定单位（nM / min-1 / min）。
#
# 说明：
#   - 每个 key 形如 "{param_key}_range_{unit_suffix}"，代表该反应类型下一个典型
#     参数的取值范围；"distribution" 字段指定推荐分布类型。
#   - 表可扩展：新增反应类型时在此追加即可，无需改动主逻辑。
_REACTION_TYPE_PRIORS: dict[str, dict] = {
    "binding": {
        "kd_range_nM": (0.1, 1000.0),
        "kon_range_per_uM_per_s": (1e-3, 1e6),
        "koff_range_per_s": (1e-4, 1.0),
        "distribution": "LogNormal",
    },
    "phosphorylation": {
        "kcat_range_per_s": (1e-3, 10.0),
        "km_range_uM": (0.01, 100.0),
        "distribution": "LogNormal",
    },
    "dephosphorylation": {
        "kcat_range_per_s": (1e-3, 5.0),
        "km_range_uM": (0.01, 50.0),
        "distribution": "LogNormal",
    },
    "transcription": {
        "k_synthesis_range_per_min": (1e-3, 1.0),
        "kd_mrna_range_per_min": (1e-3, 0.5),
        "distribution": "LogNormal",
    },
    "degradation": {
        "k_deg_range_per_min": (1e-4, 0.5),
        "half_life_range_min": (5.0, 1000.0),
        "distribution": "LogNormal",
    },
    "transport": {
        "k_transport_range_per_min": (1e-3, 1.0),
        "distribution": "LogNormal",
    },
}


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class ParameterPrior:
    """单个参数的先验分布。

    Attributes:
        param_name: 参数名（如 "k_bind_EGF_EGFR"）。
        value: 推荐值（BioModels 中位数 或 文献均值 或 反应类型几何均值）。
        unit: 单位（已归一化到项目约定：nM / min-1 / min 等）。
        source: 来源类型，取值：
            "BioModels_median" / "Literature_range" /
            "Reaction_type_prior" / "estimation_fallback"。
        source_detail: 详细来源（如 "BIOMD0000000010 median" 或 "PMID:12451180 range"）。
        reaction_type: 反应类型（binding / phosphorylation / dephosphorylation /
            transcription / degradation / transport）。
        confidence: 置信度 High / Medium / Low。
        distribution: 推荐分布 LogNormal / Uniform / Point。
        reference: PMID 或 BIOMD ID。
        range_min: 文献/反应类型范围下限（如有），已归一化。
        range_max: 文献/反应类型范围上限（如有），已归一化。
    """

    param_name: str = ""
    value: float = 0.0
    unit: str = ""
    source: str = ""
    source_detail: str = ""
    reaction_type: str = ""
    confidence: str = "Low"
    distribution: str = "Point"
    reference: str = ""
    range_min: float | None = None
    range_max: float | None = None


@dataclass
class ParameterPriorReport:
    """参数先验报告。

    Attributes:
        pathway: 通路标识（如 "egfr"）。
        priors: 合并后的 ParameterPrior 列表（按 param_name 排序）。
        biomodels_consulted: 本次咨询的 BioModels ID 去重列表。
        literature_consulted: 本次咨询的 PMID 去重列表。
        fallback_count: estimation_fallback 先验的数量。
        high_confidence_count: confidence=High 的先验数量。
        skipped: Feature Flag 关闭时为 True（不阻塞主流程）。
        skip_reason: 跳过原因（Flag 关闭时填充）。
    """

    pathway: str = ""
    priors: list[ParameterPrior] = field(default_factory=list)
    biomodels_consulted: list[str] = field(default_factory=list)
    literature_consulted: list[str] = field(default_factory=list)
    fallback_count: int = 0
    high_confidence_count: int = 0
    skipped: bool = False
    skip_reason: str = ""


# =============================================================================
# 单位归一化辅助函数（轻量实现，不 import rag_client 以避免循环依赖）
# =============================================================================
def _to_nM(value: float, unit: str) -> tuple[float, str]:
    """将常见浓度单位统一转换为 nM。

    与 rag_client._to_nM 保持一致的换算系数，确保跨模块浓度单位可比。
    无法识别的单位原样返回，避免丢失信息。

    Args:
        value: 原始数值。
        unit: 原始单位字符串。

    Returns:
        (归一化数值, 归一化单位)。
    """
    u = (unit or "").lower().strip()
    if u == "nm":
        return float(value), "nM"
    if u in ("um", "μm", "µm", "micromolar"):
        return float(value) * 1000.0, "nM"
    if u in ("mm", "millimolar"):
        return float(value) * 1_000_000.0, "nM"
    if u in ("pm", "picomolar"):
        return float(value) / 1000.0, "nM"
    return float(value), unit


def _to_minutes(value: float, unit: str) -> tuple[float, str]:
    """将常见时间单位统一转换为分钟 min。

    注意：rag_client._to_hours 转为小时，本模块按 Task 7 约定转为分钟。

    Args:
        value: 原始数值。
        unit: 原始单位字符串。

    Returns:
        (归一化数值, 归一化单位)。
    """
    u = (unit or "").lower().strip()
    if u in ("min", "minute", "minutes"):
        return float(value), "min"
    if u in ("h", "hr", "hour", "hours"):
        return float(value) * 60.0, "min"
    if u in ("s", "sec", "second", "seconds"):
        return float(value) / 60.0, "min"
    if u in ("d", "day", "days"):
        return float(value) * 1440.0, "min"
    return float(value), unit


def _normalize_param_value(param_name: str, value, unit: str) -> tuple[float, str]:
    """对单条参数值做轻量单位归一化（浓度→nM，时间→min）。

    复用 rag_client.normalize_param 的关键词策略，但时间统一到 min（本模块约定），
    且不 import rag_client 以避免循环依赖。

    规则：
      - 参数名含 kd/km/vmax/ec50/ic50 → 浓度类，转 nM
      - 参数名含 half-life/degradation/secretion → 时间类，转 min
      - 其余（含复合速率单位如 'nM-1min-1'）原样返回，避免误转

    Args:
        param_name: 参数名（用于关键词判定）。
        value: 原始数值（int/float/str 均可，内部转 float）。
        unit: 原始单位字符串。

    Returns:
        (归一化后的数值, 归一化后的单位字符串)。无法解析数值时返回 (0.0, unit)。
    """
    name = (param_name or "").lower()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0, str(unit or "")
    # 浓度相关参数统一转为 nM
    if any(k in name for k in ("kd", "km", "vmax", "ec50", "ic50")):
        return _to_nM(numeric, unit)
    # 时间/降解相关参数统一转为 min
    if any(
        k in name
        for k in ("half-life", "half_life", "halflife", "degradation", "secretion")
    ):
        return _to_minutes(numeric, unit)
    # 其余（含复合速率单位）原样返回
    return numeric, str(unit or "")


def _convert_from_table_unit(value: float, unit_suffix: str) -> tuple[float, str]:
    """将反应类型先验表中的数值从表内单位转换为项目约定单位。

    表内单位（文献常用）→ 项目约定：
      nM            → nM（浓度，不变）
      uM            → nM（浓度，×1000）
      per_s         → min-1（一阶速率，×60）
      per_uM_per_s  → nM-1min-1（二阶速率，×0.06）
      per_min       → min-1（一阶速率，不变）
      min           → min（时间，不变）

    二阶速率换算推导：
      1 uM^-1 s^-1 = 1/(uM·s) = 1/(1000 nM · s)
                   = (1/1000) nM^-1 s^-1
                   = (1/1000) × 60 nM^-1 min^-1
                   = 0.06 nM^-1 min^-1

    Args:
        value: 表内数值。
        unit_suffix: 表 key 中 "_range_" 之后的单位后缀（如 "nM" / "per_s"）。

    Returns:
        (归一化数值, 归一化单位字符串)。未知后缀原样返回。
    """
    u = (unit_suffix or "").lower().strip()
    if u == "nm":
        return float(value), "nM"
    if u in ("um", "μm", "µm", "micromolar"):
        return float(value) * 1000.0, "nM"
    if u in ("per_s", "/s", "s-1", "per second", "per_second"):
        return float(value) * 60.0, "min-1"
    if u in ("per_um_per_s", "um-1s-1", "μm-1s-1", "per_um_per_second"):
        return float(value) * 0.06, "nM-1min-1"
    if u in ("per_min", "/min", "min-1", "per minute", "per_minute"):
        return float(value), "min-1"
    if u in ("min", "minute", "minutes"):
        return float(value), "min"
    return float(value), unit_suffix


# =============================================================================
# 数学辅助函数
# =============================================================================
def _geometric_mean(a: float, b: float) -> float:
    """几何平均数 sqrt(a*b)，用于从范围 [a, b] 取代表值。

    动力学参数通常跨多个数量级，几何均值在对数空间上是中心，比算术均值更合理。
    对非正值（≤0，数学上几何均值无定义）回退到算术均值，避免 math domain error。

    Args:
        a: 范围下限。
        b: 范围上限。

    Returns:
        几何均值 sqrt(a*b)；a 或 b 非正时回退为 (a+b)/2。
    """
    if a > 0 and b > 0:
        return math.sqrt(a * b)
    return (a + b) / 2.0


# =============================================================================
# 反应类型先验查表
# =============================================================================
def get_reaction_type_prior(reaction_type: str) -> dict:
    """查反应类型先验表（大小写不敏感），未知反应类型返回空 dict。

    Args:
        reaction_type: 反应类型（如 "binding" / "phosphorylation"）。

    Returns:
        该反应类型的先验表 dict（含各参数范围与 distribution）；
        未知类型返回 {}，调用方据此跳过该层。
    """
    if not reaction_type:
        return {}
    return _REACTION_TYPE_PRIORS.get(reaction_type.lower().strip(), {})


# =============================================================================
# 三层先验提取
# =============================================================================
def _extract_biomodels_priors(
    biomodels_params: list[dict],
    reaction_type: str,
) -> tuple[list[ParameterPrior], list[str]]:
    """从 BioModels 参数列表提取中位数先验（第一层，confidence=High）。

    流程：
      1. 按 param_name 分组
      2. 对每组的 value 取中位数（statistics.median），单位归一化
      3. source=BioModels_median，source_detail="{BIOMD ID} median"
      4. distribution=LogNormal（动力学参数在对数空间更合理）
      5. 记录去重的 biomodels_consulted 列表

    Args:
        biomodels_params: 从 BioModels SBML 提取的参数列表，每个 dict 含
            {param_name, value, unit, biomodel_id}。
        reaction_type: 反应类型（写入先验的 reaction_type 字段）。

    Returns:
        (先验列表, 去重的 BIOMD ID 列表)。输入为空时返回 ([], [])。
    """
    if not biomodels_params:
        return [], []

    # 按 param_name 分组，同时收集所有 BIOMD ID（保持插入顺序去重）
    groups: dict[str, list[dict]] = {}
    biomodel_ids: list[str] = []
    for rec in biomodels_params:
        pname = str(rec.get("param_name", "")).strip()
        if not pname:
            continue
        groups.setdefault(pname, []).append(rec)
        bid = str(rec.get("biomodel_id", "")).strip()
        if bid and bid not in biomodel_ids:
            biomodel_ids.append(bid)

    priors: list[ParameterPrior] = []
    for pname, records in groups.items():
        values: list[float] = []
        unit = ""
        bids: list[str] = []
        for rec in records:
            try:
                v = float(rec.get("value", 0))
            except (TypeError, ValueError):
                continue
            # 单位归一化（浓度→nM，时间→min，复合单位原样）
            v_norm, u_norm = _normalize_param_value(
                pname, v, str(rec.get("unit", ""))
            )
            values.append(v_norm)
            unit = u_norm
            b = str(rec.get("biomodel_id", "")).strip()
            if b and b not in bids:
                bids.append(b)
        if not values:
            continue
        # 中位数：对参数跨模型取集中趋势，比均值对离群值更稳健
        median = statistics.median(values)
        source_detail = (
            f"{', '.join(bids)} median" if bids else "BioModels median"
        )
        priors.append(
            ParameterPrior(
                param_name=pname,
                value=median,
                unit=unit,
                source="BioModels_median",
                source_detail=source_detail,
                reaction_type=reaction_type,
                confidence="High",
                distribution="LogNormal",
                reference=bids[0] if bids else "",
            )
        )
    return priors, biomodel_ids


def _extract_literature_priors(
    literature_params: list[dict],
    reaction_type: str,
) -> tuple[list[ParameterPrior], list[str]]:
    """从文献参数列表提取范围先验（第二层，confidence=Medium）。

    流程：
      1. 按 param_name 分组
      2. 对每组取显式 range_min/range_max（如有，跨记录取极值）或从 values 推导
      3. value 取算术均值，单位归一化
      4. source=Literature_range，source_detail="{PMID} range"
      5. distribution=LogNormal
      6. 记录去重的 literature_consulted（PMID）列表

    Args:
        literature_params: 从文献 RAG 检索的参数列表，每个 dict 含
            {param_name, value, unit, pmid, range_min?, range_max?}。
        reaction_type: 反应类型。

    Returns:
        (先验列表, 去重的 PMID 列表)。输入为空时返回 ([], [])。
    """
    if not literature_params:
        return [], []

    groups: dict[str, list[dict]] = {}
    pmids: list[str] = []
    for rec in literature_params:
        pname = str(rec.get("param_name", "")).strip()
        if not pname:
            continue
        groups.setdefault(pname, []).append(rec)
        pmid = str(rec.get("pmid", "")).strip()
        if pmid and pmid not in pmids:
            pmids.append(pmid)

    priors: list[ParameterPrior] = []
    for pname, records in groups.items():
        values: list[float] = []
        unit = ""
        rmin: float | None = None
        rmax: float | None = None
        pmid = ""
        for rec in records:
            rec_unit = str(rec.get("unit", ""))
            try:
                v = float(rec.get("value", 0))
            except (TypeError, ValueError):
                continue
            v_norm, u_norm = _normalize_param_value(pname, v, rec_unit)
            values.append(v_norm)
            unit = u_norm
            # 显式 range_min / range_max 优先，按同一记录单位归一化后取极值
            rmin_raw = rec.get("range_min")
            if rmin_raw is not None:
                try:
                    rmin_v, _ = _normalize_param_value(pname, float(rmin_raw), rec_unit)
                    rmin = rmin_v if rmin is None else min(rmin, rmin_v)
                except (TypeError, ValueError):
                    pass
            rmax_raw = rec.get("range_max")
            if rmax_raw is not None:
                try:
                    rmax_v, _ = _normalize_param_value(pname, float(rmax_raw), rec_unit)
                    rmax = rmax_v if rmax is None else max(rmax, rmax_v)
                except (TypeError, ValueError):
                    pass
            if not pmid:
                pmid = str(rec.get("pmid", "")).strip()
        if not values:
            continue
        # 无显式 range 时从 values 推导 min/max
        if rmin is None:
            rmin = min(values)
        if rmax is None:
            rmax = max(values)
        mean_val = sum(values) / len(values)
        source_detail = f"{pmid} range" if pmid else "Literature range"
        priors.append(
            ParameterPrior(
                param_name=pname,
                value=mean_val,
                unit=unit,
                source="Literature_range",
                source_detail=source_detail,
                reaction_type=reaction_type,
                confidence="Medium",
                distribution="LogNormal",
                reference=pmid,
                range_min=rmin,
                range_max=rmax,
            )
        )
    return priors, pmids


def _extract_reaction_type_priors(reaction_type: str) -> list[ParameterPrior]:
    """从反应类型先验表生成典型参数先验（第三层，confidence=Low）。

    对该反应类型的每个典型参数（表内形如 "{param}_range_{unit}" 的 key）生成一个
    ParameterPrior：
      - value = 范围几何均值（对数空间中心，动力学参数跨数量级时更合理）
      - range_min / range_max = 表值（经 _convert_from_table_unit 归一化）
      - distribution 从表取（通常 LogNormal）
      - source=Reaction_type_prior

    Args:
        reaction_type: 反应类型。

    Returns:
        先验列表；未知反应类型返回 []。
    """
    table = get_reaction_type_prior(reaction_type)
    if not table:
        return []
    dist = table.get("distribution", "LogNormal")
    priors: list[ParameterPrior] = []
    for key, val in table.items():
        if key == "distribution":
            continue
        if not isinstance(val, tuple) or len(val) != 2:
            continue
        # 解析 key: "{param_name}_range_{unit_suffix}"
        if "_range_" not in key:
            continue
        param_part, unit_suffix = key.split("_range_", 1)
        try:
            lo = float(val[0])
            hi = float(val[1])
        except (TypeError, ValueError):
            continue
        lo_norm, unit_norm = _convert_from_table_unit(lo, unit_suffix)
        hi_norm, _ = _convert_from_table_unit(hi, unit_suffix)
        geo = _geometric_mean(lo_norm, hi_norm)
        priors.append(
            ParameterPrior(
                param_name=param_part,
                value=geo,
                unit=unit_norm,
                source="Reaction_type_prior",
                source_detail=f"{reaction_type} typical range",
                reaction_type=reaction_type,
                confidence="Low",
                distribution=dist,
                range_min=lo_norm,
                range_max=hi_norm,
            )
        )
    return priors


# =============================================================================
# 合并策略
# =============================================================================
def _merge_priors(
    biomodels: dict[str, ParameterPrior],
    literature: dict[str, ParameterPrior],
    reaction: dict[str, ParameterPrior],
) -> list[ParameterPrior]:
    """按 param_name 合并三层先验，优先级 BioModels > Literature > Reaction_type。

    同一 param_name 仅保留最高优先级来源的先验。返回列表按 param_name 排序，
    保证输出稳定（便于测试与日志对比）。

    Args:
        biomodels: {param_name: ParameterPrior}（最高优先级）。
        literature: {param_name: ParameterPrior}。
        reaction: {param_name: ParameterPrior}（最低优先级）。

    Returns:
        合并后的 ParameterPrior 列表，按 param_name 升序排列。
    """
    all_names = set(biomodels) | set(literature) | set(reaction)
    merged: list[ParameterPrior] = []
    for name in sorted(all_names):
        if name in biomodels:
            merged.append(biomodels[name])
        elif name in literature:
            merged.append(literature[name])
        elif name in reaction:
            merged.append(reaction[name])
    return merged


# =============================================================================
# 主函数
# =============================================================================
def build_parameter_prior(
    pathway: str,
    reaction_type: str,
    param_name: str = "",
    biomodels_params: list[dict] | None = None,
    literature_params: list[dict] | None = None,
) -> ParameterPriorReport:
    """为指定通路的某个反应类型构建参数先验。

    优先级：
      1. BioModels 中位数（confidence=High，source=BioModels_median）
      2. 文献范围（confidence=Medium，source=Literature_range）
      3. 反应类型先验（confidence=Low，source=Reaction_type_prior）
      4. 估算 fallback（confidence=Low，source=estimation_fallback）

    仅当 BioModels 与文献均缺失某 param_name 时，才用反应类型先验；三层均无且
    param_name 显式指定时，才返回 estimation_fallback 先验。

    Args:
        pathway: 通路标识（如 "egfr"）。
        reaction_type: 反应类型（binding/phosphorylation/dephosphorylation/
            transcription/degradation/transport）。
        param_name: 可选，特定参数名；为空则返回该反应类型的所有先验。
        biomodels_params: 从 BioModels SBML 提取的参数列表，每个 dict 含
            {param_name, value, unit, biomodel_id}。
        literature_params: 从文献 RAG 检索的参数列表，每个 dict 含
            {param_name, value, unit, pmid, range_min?, range_max?}。

    Returns:
        ParameterPriorReport。Feature Flag 关闭时返回 skipped=True 的空报告。
    """
    # -------------------------------------------------------------------------
    # 1. Feature Flag 守护：默认 OFF，关闭时返回空报告不阻塞
    #    铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
    # -------------------------------------------------------------------------
    if not settings.is_sa_feature_enabled("PARAMETER_PRIOR"):
        logger.debug(
            "parameter prior skipped: SA_PARAMETER_PRIOR disabled (pathway=%s)",
            pathway,
        )
        return ParameterPriorReport(
            pathway=pathway,
            skipped=True,
            skip_reason="SA_PARAMETER_PRIOR disabled",
        )

    biomodels_params = biomodels_params or []
    literature_params = literature_params or []

    # -------------------------------------------------------------------------
    # 2. 三层先验提取
    # -------------------------------------------------------------------------
    bm_priors, biomodels_consulted = _extract_biomodels_priors(
        biomodels_params, reaction_type
    )
    lit_priors, literature_consulted = _extract_literature_priors(
        literature_params, reaction_type
    )
    rxn_priors = _extract_reaction_type_priors(reaction_type)

    # -------------------------------------------------------------------------
    # 3. 合并：同一 param_name 优先级高的覆盖低的
    # -------------------------------------------------------------------------
    merged = _merge_priors(
        {p.param_name: p for p in bm_priors},
        {p.param_name: p for p in lit_priors},
        {p.param_name: p for p in rxn_priors},
    )

    # -------------------------------------------------------------------------
    # 4. 按 param_name 过滤（若显式指定）
    #    三层均无且 param_name 显式指定 → 生成 estimation_fallback 先验
    # -------------------------------------------------------------------------
    fallback_count = 0
    if param_name:
        merged = [p for p in merged if p.param_name == param_name]
        if not merged:
            fallback = ParameterPrior(
                param_name=param_name,
                value=0.0,
                unit="",
                source="estimation_fallback",
                source_detail="no BioModels / literature / reaction-type prior available",
                reaction_type=reaction_type,
                confidence="Low",
                distribution="Point",
            )
            merged = [fallback]
            fallback_count = 1

    # -------------------------------------------------------------------------
    # 5. 统计 high_confidence_count
    # -------------------------------------------------------------------------
    high_confidence_count = sum(1 for p in merged if p.confidence == "High")

    logger.debug(
        "parameter prior built: pathway=%s reaction_type=%s priors=%d "
        "biomodels=%d literature=%d fallback=%d high_conf=%d",
        pathway,
        reaction_type,
        len(merged),
        len(biomodels_consulted),
        len(literature_consulted),
        fallback_count,
        high_confidence_count,
    )

    return ParameterPriorReport(
        pathway=pathway,
        priors=merged,
        biomodels_consulted=biomodels_consulted,
        literature_consulted=literature_consulted,
        fallback_count=fallback_count,
        high_confidence_count=high_confidence_count,
        skipped=False,
    )


__all__ = [
    "ParameterPrior",
    "ParameterPriorReport",
    "build_parameter_prior",
    "get_reaction_type_prior",
]
