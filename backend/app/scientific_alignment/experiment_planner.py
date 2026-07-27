# BioDynamics Agent v4 - Scientific Alignment Loop: Mechanism-based Experiment Planner (Task 11)
#
# 机制驱动实验规划器：按机制节点（非蛋白存在）生成实验链。
# 设计目标：每条实验验证一个机制节点（如 EGFR 磷酸化、ERK 双磷酸化、DUSP 负反馈转录），
#   而非验证某个蛋白是否存在。禁止用 qPCR 作为磷酸化激活的主要验证。
#
# 与 experiment_designer.py 的关系：
#   experiment_designer.py 面向单条假设生成 6 字段实验设计（perturbation/readout/...）；
#   本模块面向通路级机制节点生成完整实验链（多条实验），并提供检测函数校验已有实验列表。
#
# Feature Flag 守护：
#   SA_SEVEN_AXIS 默认 OFF。关闭时返回 skipped=True，不阻塞。
#   铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
#         （由 settings.is_sa_feature_enabled 强制校验）。
#
# 依赖：canonical_loader.py、mechanism_checker.py、app.config.settings；不引入新依赖。
#
# 核心导出：
#   from app.scientific_alignment.experiment_planner import (
#       MechanismExperiment, ExperimentPlan,
#       plan_experiments, check_experiments, get_experiment_chain_template,
#       EGFR_EXPERIMENT_CHAIN, FORBIDDEN_EXPERIMENTS,
#   )

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import yaml

from app.config import settings
from app.scientific_alignment.canonical_loader import load_canonical
from app.scientific_alignment.mechanism_checker import normalize_node_name

logger = logging.getLogger(__name__)


# =============================================================================
# Sprint 4 — Experiment Rule Engine：YAML 规则目录
# =============================================================================
# 实验规则 YAML 根目录：backend/knowledge/experiments/
# 与 canonical_loader.py 同样的路径解析模式（Path(__file__).resolve()）
EXPERIMENTS_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent / "knowledge" / "experiments"
)

# pathway 白名单正则：仅允许 [a-zA-Z0-9_]，防止路径遍历
_EXPERIMENT_PATHWAY_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_]+$")


# =============================================================================
# 通路实验链模板（模块级常量）
# =============================================================================
# 每条实验验证一个机制节点，含 justification 说明为何该实验验证该节点。
# 模板字段：name / technique / target_node / mechanism_step / justification /
#           expected_result

EGFR_EXPERIMENT_CHAIN: list[dict] = [
    {
        "name": "pEGFR Western Blot",
        "technique": "Western Blot",
        "target_node": "EGFR",
        "mechanism_step": "EGF-EGFR binding & autophosphorylation",
        "justification": "pEGFR WB 直接检测 EGFR 磷酸化激活",
        "expected_result": "EGF 刺激后 pEGFR <5min 达峰",
    },
    {
        "name": "ppERK Western Blot",
        "technique": "Western Blot",
        "target_node": "ERK",
        "mechanism_step": "Raf-MEK-ERK cascade phosphorylation",
        "justification": "ppERK WB 验证 ERK 双磷酸化活化",
        "expected_result": "ERK 10-20min 达峰",
    },
    {
        "name": "DUSP1 qPCR",
        "technique": "qPCR",
        "target_node": "DUSP",
        "mechanism_step": "DUSP negative feedback transcription",
        "justification": "DUSP1 mRNA 诱导是负反馈激活标志",
        "expected_result": "DUSP1 30-60min 上调",
    },
    {
        "name": "U0126 MEK inhibitor",
        "technique": "Inhibitor",
        "target_node": "MEK",
        "mechanism_step": "Raf-MEK-ERK cascade blockade",
        "justification": "U0126 阻断 MEK 验证 ERK 下游依赖",
        "expected_result": "U0126 后 ppERK 消失",
    },
    {
        "name": "Gefitinib EGFR inhibitor",
        "technique": "Inhibitor",
        "target_node": "EGFR",
        "mechanism_step": "EGFR kinase activity blockade",
        "justification": "Gefitinib 阻断 EGFR 验证上游依赖",
        "expected_result": "Gefitinib 后 pEGFR/pERK 消失",
    },
    {
        "name": "Time Course",
        "technique": "Time Course",
        "target_node": "ERK",
        "mechanism_step": "ERK dynamics over time",
        "justification": "时间进程验证 ERK 瞬态峰值",
        "expected_result": "ERK 峰值后下降",
    },
    {
        "name": "Dose Response",
        "technique": "Dose Response",
        "target_node": "EGFR",
        "mechanism_step": "EGF dose-EGFR activation",
        "justification": "剂量响应验证 EGF-EGFR 剂量依赖",
        "expected_result": "EC50 在生理范围",
    },
]

MAPK_EXPERIMENT_CHAIN: list[dict] = [
    {
        "name": "ppERK Western Blot",
        "technique": "Western Blot",
        "target_node": "ERK",
        "mechanism_step": "Raf-MEK-ERK cascade phosphorylation",
        "justification": "ppERK WB 验证 ERK 双磷酸化活化",
        "expected_result": "ERK 10-20min 达峰",
    },
    {
        "name": "ppMEK Western Blot",
        "technique": "Western Blot",
        "target_node": "MEK",
        "mechanism_step": "Raf-mediated MEK phosphorylation",
        "justification": "ppMEK WB 验证 MEK 磷酸化活化",
        "expected_result": "MEK 磷酸化先于 ERK",
    },
    {
        "name": "DUSP1 qPCR",
        "technique": "qPCR",
        "target_node": "DUSP",
        "mechanism_step": "DUSP negative feedback transcription",
        "justification": "DUSP1 mRNA 诱导是负反馈激活标志",
        "expected_result": "DUSP1 30-60min 上调",
    },
    {
        "name": "U0126 MEK inhibitor",
        "technique": "Inhibitor",
        "target_node": "MEK",
        "mechanism_step": "Raf-MEK-ERK cascade blockade",
        "justification": "U0126 阻断 MEK 验证 ERK 下游依赖",
        "expected_result": "U0126 后 ppERK 消失",
    },
    {
        "name": "Time Course",
        "technique": "Time Course",
        "target_node": "ERK",
        "mechanism_step": "ERK dynamics over time",
        "justification": "时间进程验证 ERK 瞬态峰值与回落",
        "expected_result": "ERK 峰值后下降",
    },
    {
        "name": "Dose Response",
        "technique": "Dose Response",
        "target_node": "ERK",
        "mechanism_step": "Stimulus dose-ERK activation",
        "justification": "剂量响应验证 MAPK 通路剂量依赖",
        "expected_result": "EC50 在生理范围",
    },
]

PI3K_AKT_MTOR_EXPERIMENT_CHAIN: list[dict] = [
    {
        "name": "pAKT Ser473 Western Blot",
        "technique": "Western Blot",
        "target_node": "AKT",
        "mechanism_step": "PI3K-PIP3-PDK1 mediated AKT phosphorylation",
        "justification": "pAKT WB 直接检测 AKT 磷酸化激活",
        "expected_result": "pAKT 刺激后 5-15min 升高",
    },
    {
        "name": "pS6 Western Blot",
        "technique": "Western Blot",
        "target_node": "MTORC1",
        "mechanism_step": "mTORC1-S6K cascade activation",
        "justification": "pS6 WB 验证 mTORC1 下游信号活化",
        "expected_result": "pS6 30-60min 持续升高",
    },
    {
        "name": "LY294002 PI3K inhibitor",
        "technique": "Inhibitor",
        "target_node": "PI3K",
        "mechanism_step": "PI3K kinase activity blockade",
        "justification": "LY294002 阻断 PI3K 验证 AKT 上游依赖",
        "expected_result": "LY294002 后 pAKT 消失",
    },
    {
        "name": "Rapamycin mTOR inhibitor",
        "technique": "Inhibitor",
        "target_node": "MTORC1",
        "mechanism_step": "mTORC1 activity blockade",
        "justification": "Rapamycin 阻断 mTORC1 验证 S6K 下游依赖",
        "expected_result": "Rapamycin 后 pS6 消失",
    },
    {
        "name": "Time Course",
        "technique": "Time Course",
        "target_node": "AKT",
        "mechanism_step": "AKT dynamics over time",
        "justification": "时间进程验证 AKT 活化动力学",
        "expected_result": "AKT 峰值后部分回落",
    },
    {
        "name": "Dose Response",
        "technique": "Dose Response",
        "target_node": "AKT",
        "mechanism_step": "Growth factor dose-AKT activation",
        "justification": "剂量响应验证 PI3K-AKT 剂量依赖",
        "expected_result": "EC50 在生理范围",
    },
]

P53_EXPERIMENT_CHAIN: list[dict] = [
    {
        "name": "p53 Western Blot",
        "technique": "Western Blot",
        "target_node": "P53",
        "mechanism_step": "DNA damage induced p53 stabilization",
        "justification": "p53 WB 直接检测 p53 蛋白稳定积累",
        "expected_result": "p53 损伤后 1-4h 升高",
    },
    {
        "name": "MDM2 co-Immunoprecipitation",
        "technique": "co-IP",
        "target_node": "MDM2",
        "mechanism_step": "p53-MDM2 feedback interaction",
        "justification": "MDM2 co-IP 验证 p53-MDM2 反馈结合",
        "expected_result": "MDM2 与 p53 共沉淀",
    },
    {
        "name": "p21 qPCR",
        "technique": "qPCR",
        "target_node": "P21",
        "mechanism_step": "p53 transcriptional target induction",
        "justification": "p21 mRNA 诱导是 p53 转录活性标志",
        "expected_result": "p21 2-6h 上调",
    },
    {
        "name": "Nutlin-3 MDM2 inhibitor",
        "technique": "Inhibitor",
        "target_node": "MDM2",
        "mechanism_step": "MDM2-p53 interaction blockade",
        "justification": "Nutlin-3 阻断 MDM2 验证 p53 稳定化依赖",
        "expected_result": "Nutlin-3 后 p53 积累",
    },
    {
        "name": "Time Course",
        "technique": "Time Course",
        "target_node": "P53",
        "mechanism_step": "p53 dynamics over time",
        "justification": "时间进程验证 p53 稳定化动力学",
        "expected_result": "p53 持续升高后回落",
    },
]

APOPTOSIS_EXPERIMENT_CHAIN: list[dict] = [
    {
        "name": "Cleaved Caspase-3 Western Blot",
        "technique": "Western Blot",
        "target_node": "CASPASE3",
        "mechanism_step": "Executioner caspase cleavage activation",
        "justification": "Cleaved Caspase-3 WB 验证执行者 caspase 活化",
        "expected_result": "Cleaved Caspase-3 刺激后 2-6h 出现",
    },
    {
        "name": "Cytochrome C release assay",
        "technique": "Fractionation",
        "target_node": "BAX",
        "mechanism_step": "Mitochondrial outer membrane permeabilization",
        "justification": "CytoC 释放验证 MOMP 线粒体外膜透化",
        "expected_result": "胞浆 CytoC 增加",
    },
    {
        "name": "Cleaved PARP Western Blot",
        "technique": "Western Blot",
        "target_node": "CASPASE3",
        "mechanism_step": "Caspase substrate cleavage",
        "justification": "Cleaved PARP WB 验证 caspase 底物切割",
        "expected_result": "Cleaved PARP 与 Caspase-3 同步出现",
    },
    {
        "name": "zVAD-fmk pan-caspase inhibitor",
        "technique": "Inhibitor",
        "target_node": "CASPASE3",
        "mechanism_step": "Caspase activity blockade",
        "justification": "zVAD 阻断 caspase 验证凋亡执行依赖",
        "expected_result": "zVAD 后 Cleaved Caspase-3/PARP 消失",
    },
    {
        "name": "Time Course",
        "technique": "Time Course",
        "target_node": "CASPASE3",
        "mechanism_step": "Apoptosis dynamics over time",
        "justification": "时间进程验证凋亡执行动力学",
        "expected_result": "Caspase-3 峰值后细胞死亡",
    },
]


# =============================================================================
# 禁止模式
# =============================================================================
# 禁止用 qPCR 作为磷酸化激活的主要验证：
#   - EGFR 机制是磷酸化激活，qPCR 测 mRNA 不能验证激活状态
#   - qPCR 只能作为补充（如 DUSP1/p21 转录靶标），不能替代磷酸化 WB

FORBIDDEN_EXPERIMENTS: list[dict] = [
    {
        "pattern": "EGFR qPCR",
        "reason": (
            "EGFR 机制是磷酸化激活，qPCR 测 mRNA 不能验证激活状态，应用 pEGFR WB"
        ),
    },
    {
        "pattern": "qPCR as primary validation",
        "reason": "qPCR 只能作为补充，不能替代磷酸化 WB",
    },
]

# 磷酸化激活类节点：用 qPCR 验证这些节点属于 "qPCR as primary validation"
_PHOSPHO_NODES: set[str] = {
    normalize_node_name(n)
    for n in (
        "EGFR", "ERK", "MEK", "RAF", "AKT", "S6K",
        "MTORC1", "RSK", "JAK", "STAT", "P53",
    )
}

# 模板话术黑名单：justification 过于笼统时视为无效（experiment_unjustified）
_CLICHE_JUSTIFICATIONS: set[str] = {
    "", "N/A", "n/a", "TBD", "TODO", "-", "--",
    "validate", "confirm", "test", "yes",
}


# =============================================================================
# 通路名 → Canonical 文件 key 映射
# =============================================================================
# plan_experiments 接收展示用通路名（如 "EGFR" / "PI3K-AKT-mTOR"），
# load_canonical 需要文件名 key（如 "egfr" / "pi3k_akt_mtor"）。
# 含 "-" 的通路名不满足 canonical_loader 的 [a-zA-Z0-9_]+ 白名单，需映射。
_PATHWAY_TO_CANONICAL: dict[str, str] = {
    "EGFR": "egfr",
    "EGFR_RTK": "egfr",
    "MAPK": "mapk",
    "MAPK_CASCADE": "mapk",
    "PI3K-AKT-mTOR": "pi3k_akt_mtor",
    "PI3K_AKT_mTOR": "pi3k_akt_mtor",
    "PI3K_AKT_MTOR": "pi3k_akt_mtor",
    "p53": "p53",
    "P53": "p53",
    "P53_SIGNALING": "p53",
    "Apoptosis": "apoptosis",
    "APOPTOSIS": "apoptosis",
    # Sprint 4 扩展：覆盖剩余 5 条通路
    "JAK_STAT": "jak_stat",
    "JAK-STAT": "jak_stat",
    "JAKSTAT": "jak_stat",
    "NFKB": "nf_kappa_b",
    "NF-KB": "nf_kappa_b",
    "NF_KB": "nf_kappa_b",
    "NFKB_SIGNALING": "nf_kappa_b",
    "TGF_BETA": "tgf_beta",
    "TGF-BETA": "tgf_beta",
    "TGFB": "tgf_beta",
    "TGFβ": "tgf_beta",
    "WNT": "wnt",
    "WNT_SIGNALING": "wnt",
    "CELL_CYCLE": "cell_cycle",
    "CELLCYCLE": "cell_cycle",
    "CELL-CYCLE": "cell_cycle",
}


def _pathway_to_canonical_key(pathway: str) -> str:
    """将展示用通路名映射为 Canonical 文件 key。

    支持大小写不敏感与常见变体（如 ``"PI3K-AKT-mTOR"`` → ``"pi3k_akt_mtor"``）。
    未知通路返回小写形式（load_canonical 会再次校验合法性）。

    Args:
        pathway: 通路展示名。

    Returns:
        Canonical 文件 key（如 ``"egfr"``）；空输入返回空字符串。
    """
    if not pathway:
        return ""
    key = _PATHWAY_TO_CANONICAL.get(pathway)
    if key:
        return key
    # 大小写不敏感匹配
    for k, v in _PATHWAY_TO_CANONICAL.items():
        if k.lower() == pathway.lower():
            return v
    return pathway.lower()


# =============================================================================
# 数据类
# =============================================================================
@dataclass
class MechanismExperiment:
    """单条机制驱动实验。

    Attributes:
        name: 实验名称（如 ``"pEGFR Western Blot"``）。
        technique: 实验技术（Western Blot / qPCR / Inhibitor /
            Time Course / Dose Response）。
        target_node: 验证的机制节点（如 ``"EGFR"`` / ``"ERK"`` / ``"DUSP"``）。
        mechanism_step: 验证的机制步骤描述（如
            ``"EGF-EGFR binding & autophosphorylation"``）。
        justification: 为何该实验验证该节点（如
            ``"pEGFR WB 直接检测 EGFR 磷酸化激活"``）。
        expected_result: 预期结果（如 ``"EGF 刺激后 pEGFR 5min 内升高"``）。
        forbidden: 是否为禁止的实验类型（如 EGFR qPCR 作为主要验证）。
        forbidden_reason: 禁止原因（forbidden=False 时为空）。
    """

    name: str = ""
    technique: str = ""
    target_node: str = ""
    mechanism_step: str = ""
    justification: str = ""
    expected_result: str = ""
    forbidden: bool = False
    forbidden_reason: str = ""


@dataclass
class ExperimentPlan:
    """通路级实验规划结果。

    Attributes:
        enabled: Feature Flag 是否启用。
        skipped: 是否跳过（Flag 关闭时为 True）。
        pathway: 通路标识。
        experiments: 机制驱动实验列表。
        chain_complete: 实验链是否覆盖关键机制步骤。
        unjustified_count: 无 justification 的实验数。
        forbidden_count: 含禁止模式的实验数。
        coverage: 机制节点覆盖率（0.0-1.0）。
    """

    enabled: bool = False
    skipped: bool = False
    pathway: str = ""
    experiments: List[MechanismExperiment] = field(default_factory=list)
    chain_complete: bool = False
    unjustified_count: int = 0
    forbidden_count: int = 0
    coverage: float = 0.0


# =============================================================================
# Sprint 4 — YAML 规则加载（Feature Flag 守护）
# =============================================================================
def _load_experiments_from_yaml(pathway_key: str) -> tuple[list[dict], list[dict]]:
    """从 knowledge/experiments/{pathway_key}.yaml 加载实验链与禁止规则。

    安全设计（与 canonical_loader.py 一致）：
      1. pathway_key 白名单校验（仅 [a-zA-Z0-9_]）
      2. Path.resolve() 后校验归属 EXPERIMENTS_DIR
      3. yaml.safe_load

    Args:
        pathway_key: Canonical 文件 key（如 ``"egfr"`` / ``"pi3k_akt_mtor"``）。

    Returns:
        ``(experiments, forbidden)`` 元组。文件不存在或解析失败时返回
        ``([], [])``，由调用方降级到硬编码模板。
    """
    if not pathway_key or not _EXPERIMENT_PATHWAY_PATTERN.match(pathway_key):
        return [], []

    yaml_path = (EXPERIMENTS_DIR / f"{pathway_key}.yaml").resolve()
    try:
        yaml_path.relative_to(EXPERIMENTS_DIR.resolve())
    except ValueError:
        logger.warning(
            "[Sprint4] 实验 YAML 路径越界，拒绝加载: %s", yaml_path,
        )
        return [], []

    if not yaml_path.exists():
        logger.debug("[Sprint4] 实验 YAML 不存在: %s", yaml_path)
        return [], []

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("[Sprint4] 实验 YAML 解析失败 (%s): %s", yaml_path, exc)
        return [], []

    experiments = data.get("experiments", []) or []
    forbidden = data.get("forbidden", []) or []

    # 深拷贝避免修改缓存
    return (
        [dict(item) if isinstance(item, dict) else {} for item in experiments],
        [dict(item) if isinstance(item, dict) else {} for item in forbidden],
    )


def _check_forbidden_yaml(
    name: str, technique: str, target_node: str, yaml_forbidden: list[dict],
) -> tuple[bool, str]:
    """按 YAML forbidden 规则检测实验是否禁止。

    YAML forbidden 每条含 name / technique / target_node / reason 字段。
    匹配规则（大小写不敏感）：
      - name 精确包含匹配，或
      - technique 精确匹配且 target_node 匹配

    Args:
        name: 实验名称。
        technique: 实验技术。
        target_node: 验证的机制节点。
        yaml_forbidden: YAML 加载的禁止规则列表。

    Returns:
        ``(forbidden, reason)`` 元组。
    """
    name_lower = (name or "").lower().strip()
    technique_lower = (technique or "").lower().strip()
    target_norm = normalize_node_name(target_node)

    for rule in yaml_forbidden:
        r_name = (rule.get("name", "") or "").lower().strip()
        r_technique = (rule.get("technique", "") or "").lower().strip()
        r_target = rule.get("target_node", "") or ""
        r_reason = rule.get("reason", "") or ""
        r_target_norm = normalize_node_name(r_target)

        # 匹配 1: name 精确包含
        if r_name and r_name in name_lower:
            return True, r_reason
        # 匹配 2: technique + target_node 双匹配
        if (
            r_technique
            and r_technique == technique_lower
            and r_target_norm
            and r_target_norm == target_norm
        ):
            return True, r_reason

    return False, ""


# =============================================================================
# 通路实验链模板选择
# =============================================================================
def get_experiment_chain_template(pathway: str) -> list[dict]:
    """返回通路对应的实验链模板。未知通路返回 EGFR 默认链。

    Sprint 4 扩展：``SA_SPRINT4_EXPERIMENT_RULE_ENGINE`` Flag ON 时，
    优先从 ``knowledge/experiments/{pathway_key}.yaml`` 加载 Mechanism-aware
    实验链（覆盖全部 10 条通路）；YAML 不存在时降级到硬编码模板。
    Flag OFF 时使用硬编码模板（v3/v4 行为不变）。

    Args:
        pathway: 通路标识（如 ``"EGFR"`` / ``"MAPK"`` /
            ``"PI3K-AKT-mTOR"``）。

    Returns:
        实验模板 dict 列表（深拷贝，调用方可安全修改）。
    """
    # -------------------------------------------------------------------------
    # Sprint 4: Flag ON 时优先从 YAML 加载（100% Rule-driven，Mechanism-aware）
    # -------------------------------------------------------------------------
    if settings.is_sa_feature_enabled("SPRINT4_EXPERIMENT_RULE_ENGINE"):
        canonical_key = _pathway_to_canonical_key(pathway)
        yaml_experiments, _ = _load_experiments_from_yaml(canonical_key)
        if yaml_experiments:
            logger.debug(
                "[Sprint4] 从 YAML 加载实验链: pathway=%s key=%s count=%d",
                pathway, canonical_key, len(yaml_experiments),
            )
            # [D3] 修复：critic 期望 mechanism_node 字段，YAML 用 target_node
            for exp in yaml_experiments:
                if "mechanism_node" not in exp and "target_node" in exp:
                    exp["mechanism_node"] = exp["target_node"]
            return yaml_experiments
        # YAML 不存在 → 降级到硬编码模板（下方逻辑）

    key = (pathway or "").strip().upper()
    # 归一化 PI3K 系列：去分隔符后比较
    compact = key.replace("-", "").replace("_", "").replace(" ", "")

    if key in ("EGFR", "EGFR_RTK"):
        template = EGFR_EXPERIMENT_CHAIN
    elif key == "MAPK":
        template = MAPK_EXPERIMENT_CHAIN
    elif compact == "PI3KAKTMTOR":
        template = PI3K_AKT_MTOR_EXPERIMENT_CHAIN
    elif key in ("P53", "P53_SIGNALING"):
        template = P53_EXPERIMENT_CHAIN
    elif key == "APOPTOSIS":
        template = APOPTOSIS_EXPERIMENT_CHAIN
    else:
        # 未知通路返回 EGFR 默认链
        template = EGFR_EXPERIMENT_CHAIN

    # 深拷贝避免修改模块级常量
    result = [dict(item) for item in template]
    # [D3] 修复：critic 期望 mechanism_node 字段，模板用 target_node
    for exp in result:
        if "mechanism_node" not in exp and "target_node" in exp:
            exp["mechanism_node"] = exp["target_node"]
    return result


# =============================================================================
# 禁止模式检测
# =============================================================================
def _check_forbidden(
    name: str, technique: str, target_node: str
) -> tuple[bool, str]:
    """检查实验是否匹配禁止模式。

    检测规则：
      1. 名称含 ``"EGFR qPCR"``（大小写不敏感）→ 禁止
      2. 名称含 ``"qpcr"`` 且 target_node 为 EGFR → 禁止
      3. 技术含 ``"qpcr"`` 且 target_node 为磷酸化激活类节点
         （EGFR/ERK/MEK/AKT 等，非转录靶标）→ 禁止（qPCR as primary validation）

    Args:
        name: 实验名称。
        technique: 实验技术。
        target_node: 验证的机制节点。

    Returns:
        ``(forbidden, reason)`` 元组。forbidden=True 时 reason 为禁止原因。
    """
    name_lower = (name or "").lower()
    technique_lower = (technique or "").lower()
    target_norm = normalize_node_name(target_node)

    # 模式 1: 名称中直接含 "EGFR qPCR"
    if "egfr qpcr" in name_lower:
        return True, FORBIDDEN_EXPERIMENTS[0]["reason"]

    # 模式 1 兜底: 名称含 "qpcr" 且 target 为 EGFR
    if "qpcr" in name_lower and target_norm == normalize_node_name("EGFR"):
        return True, FORBIDDEN_EXPERIMENTS[0]["reason"]

    # 模式 2: qPCR 用于验证磷酸化激活类节点（qPCR as primary validation）
    if "qpcr" in technique_lower and target_norm in _PHOSPHO_NODES:
        return True, FORBIDDEN_EXPERIMENTS[1]["reason"]

    return False, ""


# =============================================================================
# Justification 有效性检测
# =============================================================================
def _is_justified(justification: Any) -> bool:
    """检查 justification 是否有效（非空且非模板话术）。

    判定规则：
      1. 必须为字符串类型
      2. strip 后非空
      3. 不在模板话术黑名单中（如 ``"N/A"`` / ``"TBD"`` / ``"validate"``）
      4. 长度 >= 5 字符（过短视为无效）

    Args:
        justification: justification 字段值。

    Returns:
        True 表示 justification 有效。
    """
    if not isinstance(justification, str):
        return False
    text = justification.strip()
    if not text:
        return False
    if text.lower() in _CLICHE_JUSTIFICATIONS:
        return False
    if len(text) < 5:
        return False
    return True


# =============================================================================
# 覆盖率计算
# =============================================================================
def _compute_coverage(
    provided_nodes: list[str], required_nodes: list[str]
) -> float:
    """计算 provided_nodes 对 required_nodes 的覆盖率。

    使用 normalize_node_name 归一化后做集合匹配（大小写不敏感 + 同义词）。

    Args:
        provided_nodes: 提供的节点列表（extracted_nodes 或
            experiments 的 target_node）。
        required_nodes: Canonical 要求的节点列表。

    Returns:
        覆盖率 ``matched / required``（0.0-1.0）；required 为空时返回 0.0。
    """
    if not required_nodes:
        return 0.0
    required_norm: set[str] = set()
    for rn in required_nodes:
        n = normalize_node_name(rn)
        if n:
            required_norm.add(n)
    if not required_norm:
        return 0.0
    provided_norm: set[str] = set()
    for pn in provided_nodes:
        n = normalize_node_name(pn)
        if n:
            provided_norm.add(n)
    matched = required_norm & provided_norm
    return len(matched) / len(required_norm)


# =============================================================================
# 主函数
# =============================================================================
def plan_experiments(
    pathway: str, extracted_nodes: list[str] | None = None
) -> ExperimentPlan:
    """按机制节点生成实验链。从 Canonical 加载 required_nodes，按通路选择实验模板。

    流程：
      1. Feature Flag 守护（SA_SEVEN_AXIS）
      2. 按通路选择实验链模板（未知通路用 EGFR 默认链）
      3. 将模板转换为 MechanismExperiment，检测禁止模式与 justification
      4. 从 Canonical 加载 required_nodes，计算 extracted_nodes 覆盖率
      5. chain_complete = 覆盖率 >= 0.7 且 unjustified_count==0 且 forbidden_count==0

    Args:
        pathway: 通路标识（如 ``"EGFR"`` / ``"MAPK"`` /
            ``"PI3K-AKT-mTOR"``）。
        extracted_nodes: Agent 提取的机制节点列表，用于计算覆盖率。
            为 None 时覆盖率记 0.0。

    Returns:
        ExperimentPlan。Flag 关闭时返回 ``ExperimentPlan(enabled=False,
        skipped=True)``。
    """
    # -------------------------------------------------------------------------
    # Feature Flag 守护：SA_SEVEN_AXIS 默认 OFF
    # 铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
    # -------------------------------------------------------------------------
    if not settings.is_sa_feature_enabled("SEVEN_AXIS"):
        return ExperimentPlan(
            enabled=False,
            skipped=True,
            pathway=pathway,
        )

    # 1. 获取通路实验链模板
    template = get_experiment_chain_template(pathway)

    # Sprint 4: Flag ON 时加载 YAML forbidden 规则（Mechanism-aware 禁止模式）
    _yaml_forbidden: list[dict] = []
    if settings.is_sa_feature_enabled("SPRINT4_EXPERIMENT_RULE_ENGINE"):
        _canonical_key_for_yaml = _pathway_to_canonical_key(pathway)
        _, _yaml_forbidden = _load_experiments_from_yaml(_canonical_key_for_yaml)

    # 2. 转换为 MechanismExperiment 对象并检测禁止模式 / justification
    experiments: list[MechanismExperiment] = []
    unjustified_count = 0
    forbidden_count = 0
    for item in template:
        exp = MechanismExperiment(
            name=item.get("name", ""),
            technique=item.get("technique", ""),
            target_node=item.get("target_node", ""),
            mechanism_step=item.get("mechanism_step", ""),
            justification=item.get("justification", ""),
            expected_result=item.get("expected_result", ""),
        )
        # 检测禁止模式（硬编码规则）
        forbidden, reason = _check_forbidden(
            exp.name, exp.technique, exp.target_node
        )
        # Sprint 4: Flag ON 时追加 YAML forbidden 规则检测
        if not forbidden and _yaml_forbidden:
            forbidden, reason = _check_forbidden_yaml(
                exp.name, exp.technique, exp.target_node, _yaml_forbidden
            )
        if forbidden:
            exp.forbidden = True
            exp.forbidden_reason = reason
            forbidden_count += 1
        # 检测 justification
        if not _is_justified(exp.justification):
            unjustified_count += 1
        experiments.append(exp)

    # 3. 加载 Canonical required_nodes 计算覆盖率
    coverage = 0.0
    canonical_key = _pathway_to_canonical_key(pathway)
    try:
        canonical = load_canonical(canonical_key)
        required_nodes = canonical.required_nodes
        coverage = _compute_coverage(extracted_nodes or [], required_nodes)
    except Exception as exc:
        # Canonical 加载失败时不阻塞，coverage 记 0.0
        logger.warning(
            "plan_experiments: 加载 Canonical 失败 (pathway=%s, key=%s): %s",
            pathway, canonical_key, exc,
        )
        coverage = 0.0

    # 4. chain_complete 判定
    chain_complete = (
        coverage >= 0.7
        and unjustified_count == 0
        and forbidden_count == 0
    )

    logger.debug(
        "plan_experiments: pathway=%s experiments=%d coverage=%.2f "
        "unjustified=%d forbidden=%d chain_complete=%s",
        pathway, len(experiments), coverage,
        unjustified_count, forbidden_count, chain_complete,
    )

    return ExperimentPlan(
        enabled=True,
        skipped=False,
        pathway=pathway,
        experiments=experiments,
        chain_complete=chain_complete,
        unjustified_count=unjustified_count,
        forbidden_count=forbidden_count,
        coverage=coverage,
    )


def check_experiments(
    experiments: list[dict], pathway: str = ""
) -> ExperimentPlan:
    """检测已有实验列表是否机制驱动、是否有 justification、是否含禁止模式。

    检测流程：
      1. Feature Flag 守护（SA_SEVEN_AXIS）
      2. 遍历 experiments，检查每条是否有 justification（非空且非模板话术）
      3. 检查 name 是否匹配 FORBIDDEN_EXPERIMENTS 模式 → forbidden=True
      4. 从 Canonical 加载 required_nodes，计算 target_node 覆盖率
      5. chain_complete = 覆盖率 >= 0.7 且 unjustified_count==0 且
         forbidden_count==0

    Args:
        experiments: dict 列表，每个 dict 含 name/technique/target_node
            等字段。
        pathway: 通路标识（用于加载 Canonical 计算覆盖率）。

    Returns:
        ExperimentPlan。Flag 关闭时返回 ``ExperimentPlan(enabled=False,
        skipped=True)``。
    """
    # -------------------------------------------------------------------------
    # Feature Flag 守护
    # -------------------------------------------------------------------------
    if not settings.is_sa_feature_enabled("SEVEN_AXIS"):
        return ExperimentPlan(
            enabled=False,
            skipped=True,
            pathway=pathway,
        )

    unjustified_count = 0
    forbidden_count = 0
    target_nodes: list[str] = []
    checked: list[MechanismExperiment] = []

    # Sprint 4: Flag ON 时加载 YAML forbidden 规则
    _yaml_forbidden: list[dict] = []
    if settings.is_sa_feature_enabled("SPRINT4_EXPERIMENT_RULE_ENGINE") and pathway:
        _yaml_canonical_key = _pathway_to_canonical_key(pathway)
        _, _yaml_forbidden = _load_experiments_from_yaml(_yaml_canonical_key)

    for item in experiments:
        if not isinstance(item, dict):
            continue
        exp = MechanismExperiment(
            name=item.get("name", ""),
            technique=item.get("technique", ""),
            target_node=item.get("target_node", ""),
            mechanism_step=item.get("mechanism_step", ""),
            justification=item.get("justification", ""),
            expected_result=item.get("expected_result", ""),
        )
        # 检测禁止模式（硬编码规则）
        forbidden, reason = _check_forbidden(
            exp.name, exp.technique, exp.target_node
        )
        # Sprint 4: Flag ON 时追加 YAML forbidden 规则检测
        if not forbidden and _yaml_forbidden:
            forbidden, reason = _check_forbidden_yaml(
                exp.name, exp.technique, exp.target_node, _yaml_forbidden
            )
        if forbidden:
            exp.forbidden = True
            exp.forbidden_reason = reason
            forbidden_count += 1
        # 检测 justification
        if not _is_justified(exp.justification):
            unjustified_count += 1
        if exp.target_node:
            target_nodes.append(exp.target_node)
        checked.append(exp)

    # 加载 Canonical 计算覆盖率
    coverage = 0.0
    if pathway:
        canonical_key = _pathway_to_canonical_key(pathway)
        try:
            canonical = load_canonical(canonical_key)
            required_nodes = canonical.required_nodes
            coverage = _compute_coverage(target_nodes, required_nodes)
        except Exception as exc:
            logger.warning(
                "check_experiments: 加载 Canonical 失败 "
                "(pathway=%s, key=%s): %s",
                pathway, canonical_key, exc,
            )
            coverage = 0.0

    chain_complete = (
        coverage >= 0.7
        and unjustified_count == 0
        and forbidden_count == 0
    )

    logger.debug(
        "check_experiments: pathway=%s experiments=%d coverage=%.2f "
        "unjustified=%d forbidden=%d chain_complete=%s",
        pathway, len(checked), coverage,
        unjustified_count, forbidden_count, chain_complete,
    )

    return ExperimentPlan(
        enabled=True,
        skipped=False,
        pathway=pathway,
        experiments=checked,
        chain_complete=chain_complete,
        unjustified_count=unjustified_count,
        forbidden_count=forbidden_count,
        coverage=coverage,
    )


__all__ = [
    "MechanismExperiment",
    "ExperimentPlan",
    "plan_experiments",
    "check_experiments",
    "get_experiment_chain_template",
    "EGFR_EXPERIMENT_CHAIN",
    "MAPK_EXPERIMENT_CHAIN",
    "PI3K_AKT_MTOR_EXPERIMENT_CHAIN",
    "P53_EXPERIMENT_CHAIN",
    "APOPTOSIS_EXPERIMENT_CHAIN",
    "FORBIDDEN_EXPERIMENTS",
]
