# BioDynamics Agent v4 - Scientific Alignment Loop 包（Task 3 / Task 22）
#
# 文献级 Gold Standard 与 Evidence Ranking Score：约束 Retriever 不能自行决定
# 什么文献最好，必须按经典综述 > 机制论文 > BioModels 来源 > 近期应用 > 案例报告
# 的固定优先级排序，PubMed 检索结果仅作为补充。
#
# Task 22 新增：Canonical Reference Library — 每通路唯一权威参考，
# 所有 Benchmark 的"标准答案"必须基于 Canonical。
#
# Task 7 新增：Parameter Grounding — 从 BioModels 中位数 + 文献范围 + 反应类型先验
# 构建 Parameter Prior，禁止 LLM 直接拍脑袋出参数。
#
# Task 12 新增：7 轴 Scientific Validation Pyramid —
# Mechanism / Dynamics / BioModels / Literature / Experiment / Discussion / Evidence
# 每轴独立评分，任一轴 Fail 整体降 Confidence。
#
# 核心导出：
#   from app.scientific_alignment import (
#       EvidenceType, EvidenceDoc, EvidenceRanker,
#       load_literature_gold_standard,
#       CanonicalReference, ConsistencyRule, load_canonical,
#       get_consistency_rules, validate_canonical,
#       cross_check_consistency,
#       ConsistencyViolation, ConsistencyReport,
#       check_consistency, extract_peak_times_from_simulation,
#       MechanismAlignmentResult, check_mechanism_alignment, normalize_node_name,
#       ParameterPrior, ParameterPriorReport, build_parameter_prior,
#       AxisScore, SevenAxisReport, run_seven_axis_validation,
#   )

from app.scientific_alignment.canonical_loader import (
    CanonicalMechanism,
    CanonicalMissingError,
    CanonicalNotFoundError,
    CanonicalPathwayNameError,
    CanonicalReference,
    ConsistencyRule,
    MechanismEdge,
    get_consistency_rules,
    load_canonical,
    validate_canonical,
)
from app.scientific_alignment.canonical_cross_check import (
    cross_check_consistency,
)
from app.scientific_alignment.consistency_checker import (
    ConsistencyReport,
    ConsistencyViolation,
    check_consistency,
    extract_peak_times_from_simulation,
)
from app.scientific_alignment.evidence_ranker import (
    EvidenceDoc,
    EvidenceRanker,
    EvidenceType,
    load_literature_gold_standard,
)
from app.scientific_alignment.mechanism_checker import (
    MechanismAlignmentResult,
    check_mechanism_alignment,
    normalize_node_name,
)
from app.scientific_alignment.parameter_grounder import (
    ParameterPrior,
    ParameterPriorReport,
    build_parameter_prior,
)
from app.scientific_alignment.seven_axis_validator import (
    AxisScore,
    SevenAxisReport,
    run_seven_axis_validation,
)

__all__ = [
    "EvidenceType",
    "EvidenceDoc",
    "EvidenceRanker",
    "load_literature_gold_standard",
    "CanonicalReference",
    "CanonicalMechanism",
    "ConsistencyRule",
    "MechanismEdge",
    "CanonicalMissingError",
    "CanonicalNotFoundError",
    "CanonicalPathwayNameError",
    "load_canonical",
    "get_consistency_rules",
    "validate_canonical",
    "cross_check_consistency",
    "ConsistencyViolation",
    "ConsistencyReport",
    "check_consistency",
    "extract_peak_times_from_simulation",
    "MechanismAlignmentResult",
    "check_mechanism_alignment",
    "normalize_node_name",
    "ParameterPrior",
    "ParameterPriorReport",
    "build_parameter_prior",
    "AxisScore",
    "SevenAxisReport",
    "run_seven_axis_validation",
]
