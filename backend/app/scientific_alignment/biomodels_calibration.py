# BioDynamics Agent v4 - BioModels Parameter Calibration (Task F)
#
# 用户核心诉求：
#   你的ODE → RoadRunner → BioModels → Compare
#   输出 Peak Time / Peak Value / RMSE / Correlation 差异
#   如果差太大，自动进入 Calibration
#
# 设计：
#   1. 加载 Agent ODE 仿真 CSV（时间序列）
#   2. 加载对应 BioModels SBML 文件
#   3. 用 RoadRunner 仿真 SBML
#   4. 对比对物种（fuzzy match 物种名）
#      - Peak Time Difference
#      - Peak Value Difference (%)
#      - RMSE
#      - Pearson Correlation
#   5. 输出对比报告
#   6. 若差异超阈值 → 标记 needs_calibration=True
#
# 核心导出：
#   from app.scientific_alignment.biomodels_calibration import (
#       BioModelsComparator, ComparisonResult, SpeciesComparison,
#       compare_with_biomodels,
#   )

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================
# BioModels SBML 文件根目录
_BIOMODELS_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent / "data" / "raw"
)

# v4_pathway_class → BIOMD ID 映射（取每通路的代表性模型）
_PATHWAY_TO_BIOMD: dict[str, str] = {
    "EGFR_RTK": "BIOMD0000000205",
    "EGFR": "BIOMD0000000205",
    "MAPK_ERK": "BIOMD0000000010",
    "MAPK": "BIOMD0000000010",
    "PI3K_AKT_MTOR": "BIOMD0000000250",
    "TGF_BETA": "BIOMD0000000252",
    "JAK_STAT": "BIOMD0000000224",
    "NFKB": "BIOMD0000000258",
    "NF_KAPPA_B": "BIOMD0000000258",
    "P53": "BIOMD0000000382",
    "APOPTOSIS": "BIOMD0000000255",
    "CELL_CYCLE": "BIOMD0000000056",
    "WNT": "BIOMD0000000008",
}

# 默认校准阈值
_DEFAULT_PEAK_TIME_THRESHOLD: float = 5.0    # 分钟
_DEFAULT_PEAK_VALUE_THRESHOLD: float = 0.20  # 20%
_DEFAULT_RMSE_THRESHOLD: float = 0.15
_DEFAULT_CORRELATION_THRESHOLD: float = 0.80


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class SpeciesComparison:
    """单个物种的对比结果。"""

    agent_species: str
    biomodel_species: str
    matched: bool                  # 是否成功匹配
    peak_time_diff: float = 0.0    # 峰值时间差（分钟）
    peak_value_diff_pct: float = 0.0  # 峰值差异（%）
    rmse: float = 0.0             # 归一化 RMSE
    correlation: float = 0.0      # Pearson 相关系数
    needs_calibration: bool = False


@dataclass
class ComparisonResult:
    """整体对比结果。

    对接 validation_report 兼容字段（_c4/_c8 评估器期望的命名）：
      - rmse / correlation: overall_rmse / overall_correlation 的别名
      - sbml_sim_available: 与 biomodel_loaded 同义
      - checksum_verified: SBML cache sha256 是否匹配 manifest
      - method: "libroadrunner"（仿真成功）或 "blocked"（失败/降级）

    注意：blocked 不伪装 pass。pass_ 仅在真实仿真成功且无需校准时为 True。
    """

    pathway: str
    biomodel_id: str
    biomodel_loaded: bool
    agent_species_count: int = 0
    biomodel_species_count: int = 0
    matched_count: int = 0
    species_comparisons: list[SpeciesComparison] = field(default_factory=list)
    overall_rmse: float = 0.0
    overall_correlation: float = 0.0
    needs_calibration: bool = False
    error: str = ""
    # validation_report 兼容字段（由 compare() 填充）
    checksum_verified: bool = False
    method: str = ""

    @property
    def rmse(self) -> float:
        """validation_report 兼容字段（overall_rmse 别名）。"""
        return self.overall_rmse

    @property
    def correlation(self) -> float:
        """validation_report 兼容字段（overall_correlation 别名）。"""
        return self.overall_correlation

    @property
    def sbml_sim_available(self) -> bool:
        """validation_report 兼容字段（biomodel_loaded 别名）。"""
        return self.biomodel_loaded

    @property
    def pass_(self) -> bool:
        """validation_report 兼容：blocked 不伪装 pass。

        仅当 SBML 真实仿真成功且无需校准时为 True。
        """
        return self.biomodel_loaded and not self.needs_calibration


# =============================================================================
# BioModelsComparator
# =============================================================================
class BioModelsComparator:
    """BioModels 参考对比器。

    用 RoadRunner 仿真 BioModels SBML，与 Agent ODE 仿真结果对比。

    Usage::

        comparator = BioModelsComparator()
        result = comparator.compare(
            agent_csv_path="path/to/simulation.csv",
            pathway="EGFR",
            duration=120,
            n_points=500,
        )
        if result.needs_calibration:
            print("需要校准！")
    """

    def __init__(
        self,
        peak_time_threshold: float = _DEFAULT_PEAK_TIME_THRESHOLD,
        peak_value_threshold: float = _DEFAULT_PEAK_VALUE_THRESHOLD,
        rmse_threshold: float = _DEFAULT_RMSE_THRESHOLD,
        correlation_threshold: float = _DEFAULT_CORRELATION_THRESHOLD,
    ) -> None:
        self._peak_time_threshold = peak_time_threshold
        self._peak_value_threshold = peak_value_threshold
        self._rmse_threshold = rmse_threshold
        self._correlation_threshold = correlation_threshold

    # -------------------------------------------------------------------------
    # 公开接口
    # -------------------------------------------------------------------------
    def compare(
        self,
        agent_csv_path: str | Path,
        pathway: str,
        duration: float = 120.0,
        n_points: int = 500,
    ) -> ComparisonResult:
        """对比 Agent ODE 仿真与 BioModels 参考仿真。

        Args:
            agent_csv_path: Agent ODE 仿真 CSV 路径。
            pathway: 通路标识（v4_pathway_class）。
            duration: 仿真时长（分钟）。
            n_points: 仿真点数。

        Returns:
            ComparisonResult。
        """
        # 1. 确定 BioModels ID
        biomodel_id = _resolve_biomodel_id(pathway)
        result = ComparisonResult(
            pathway=pathway,
            biomodel_id=biomodel_id,
            biomodel_loaded=False,
            method="blocked",  # 默认 blocked，仿真成功后升级为 libroadrunner
        )

        if not biomodel_id:
            result.error = f"通路 {pathway} 无 BioModels 映射"
            return result

        # 2. 加载 Agent 仿真 CSV
        try:
            agent_data = _load_csv_timeseries(agent_csv_path)
        except Exception as exc:
            result.error = f"加载 Agent CSV 失败: {exc}"
            return result

        if not agent_data:
            result.error = "Agent CSV 无有效数据"
            return result

        result.agent_species_count = len(agent_data) - 1  # 减去 time 列

        # 3. 加载并仿真 BioModels SBML
        try:
            biomodel_data = self._simulate_biomodels(
                biomodel_id, duration=duration, n_points=n_points
            )
        except Exception as exc:
            result.error = f"BioModels 仿真失败: {exc}"
            logger.warning("BioModelsComparator: %s", exc)
            return result

        if not biomodel_data:
            result.error = "BioModels 仿真无数据"
            return result

        result.biomodel_loaded = True
        result.method = "libroadrunner"
        # 对接 validation_report: 校验 SBML cache sha256 与 manifest 是否一致
        result.checksum_verified = _verify_sbml_checksum(biomodel_id)
        result.biomodel_species_count = len(biomodel_data) - 1

        # 4. 物种匹配 + 对比
        agent_time = agent_data.get("time", np.array([]))
        biomodel_time = biomodel_data.get("time", np.array([]))

        if len(agent_time) == 0 or len(biomodel_time) == 0:
            result.error = "时间序列为空"
            return result

        for agent_spec, agent_vals in agent_data.items():
            if agent_spec == "time":
                continue

            # Fuzzy match 物种名
            biomodel_spec = _fuzzy_match_species(agent_spec, biomodel_data)
            if not biomodel_spec:
                result.species_comparisons.append(SpeciesComparison(
                    agent_species=agent_spec,
                    biomodel_species="",
                    matched=False,
                ))
                continue

            biomodel_vals = biomodel_data[biomodel_spec]

            # 归一化时间轴（0-1）以便对比不同时间分辨率
            agent_norm = _normalize_time(agent_time, agent_vals)
            biomodel_norm = _normalize_time(biomodel_time, biomodel_vals)

            # 重采样到相同点数
            agent_resampled = _resample(agent_norm, n_points)
            biomodel_resampled = _resample(biomodel_norm, n_points)

            # 计算对比指标
            comp = self._compare_species(
                agent_spec, biomodel_spec,
                agent_time, agent_vals,
                biomodel_time, biomodel_vals,
                agent_resampled, biomodel_resampled,
            )
            result.species_comparisons.append(comp)
            result.matched_count += 1

        # 5. 汇总
        matched = [c for c in result.species_comparisons if c.matched]
        if matched:
            result.overall_rmse = float(np.mean([c.rmse for c in matched]))
            result.overall_correlation = float(np.mean([c.correlation for c in matched]))
            result.needs_calibration = any(c.needs_calibration for c in matched)

        return result

    # -------------------------------------------------------------------------
    # 内部方法
    # -------------------------------------------------------------------------
    def _simulate_biomodels(
        self,
        biomodel_id: str,
        duration: float,
        n_points: int,
    ) -> dict[str, np.ndarray]:
        """用 RoadRunner 仿真 BioModels SBML。

        Returns:
            {species_name: np.ndarray, "time": np.ndarray}
        """
        try:
            import roadrunner
        except ImportError as exc:
            raise RuntimeError(f"RoadRunner 不可用: {exc}") from exc

        # 查找 SBML 文件
        sbml_path = _find_sbml_file(biomodel_id)
        if not sbml_path:
            raise FileNotFoundError(f"未找到 SBML 文件: {biomodel_id}")

        # 加载 SBML
        sbml_text = sbml_path.read_text(encoding="utf-8", errors="replace")
        rr = roadrunner.RoadRunner()
        rr.load(sbml_text)

        # 仿真（duration 转为秒，因为 BioModels SBML 通常以秒为单位）
        # Agent ODE 通常以分钟为单位，所以需要转换
        duration_sec = duration * 60.0  # min → sec
        result = rr.simulate(0, duration_sec, n_points)

        # 构建 species_id → species_name 映射（从 SBML 解析）
        id_to_name = _build_species_id_name_map(sbml_text)

        # 提取物种数据
        # RoadRunner 返回的列名可能带 "[" 和 "]"，需要清理
        data: dict[str, np.ndarray] = {}
        col_names = result.colnames
        for i, col in enumerate(col_names):
            clean_name = col.strip("[]")
            if clean_name.lower() == "time":
                data["time"] = np.array(result[:, i])
            else:
                # 优先使用 SBML name 属性（如有），否则用 ID
                display_name = id_to_name.get(clean_name, clean_name)
                data[display_name] = np.array(result[:, i])

        return data

    def _compare_species(
        self,
        agent_spec: str,
        biomodel_spec: str,
        agent_time: np.ndarray,
        agent_vals: np.ndarray,
        biomodel_time: np.ndarray,
        biomodel_vals: np.ndarray,
        agent_resampled: np.ndarray,
        biomodel_resampled: np.ndarray,
    ) -> SpeciesComparison:
        """计算单个物种的对比指标。"""
        # Peak Time（在原始时间轴上计算）
        agent_peak_idx = int(np.argmax(agent_vals))
        biomodel_peak_idx = int(np.argmax(biomodel_vals))

        # BioModels 时间是秒，转为分钟
        agent_peak_time = float(agent_time[agent_peak_idx]) if len(agent_time) > 0 else 0.0
        biomodel_peak_time = float(biomodel_time[biomodel_peak_idx]) / 60.0 if len(biomodel_time) > 0 else 0.0
        peak_time_diff = abs(agent_peak_time - biomodel_peak_time)

        # Peak Value（归一化后对比）
        agent_peak_val = float(np.max(agent_resampled)) if len(agent_resampled) > 0 else 0.0
        biomodel_peak_val = float(np.max(biomodel_resampled)) if len(biomodel_resampled) > 0 else 0.0
        if biomodel_peak_val > 0:
            peak_value_diff_pct = abs(agent_peak_val - biomodel_peak_val) / biomodel_peak_val
        else:
            peak_value_diff_pct = 0.0

        # RMSE（归一化后）
        if len(agent_resampled) > 0 and len(biomodel_resampled) > 0:
            rmse = float(np.sqrt(np.mean((agent_resampled - biomodel_resampled) ** 2)))
        else:
            rmse = 1.0

        # Pearson Correlation
        if len(agent_resampled) > 1 and len(biomodel_resampled) > 1:
            try:
                corr_matrix = np.corrcoef(agent_resampled, biomodel_resampled)
                correlation = float(corr_matrix[0, 1])
            except Exception:
                correlation = 0.0
        else:
            correlation = 0.0

        # 判断是否需要校准
        needs_calibration = (
            peak_time_diff > self._peak_time_threshold
            or peak_value_diff_pct > self._peak_value_threshold
            or rmse > self._rmse_threshold
            or correlation < self._correlation_threshold
        )

        return SpeciesComparison(
            agent_species=agent_spec,
            biomodel_species=biomodel_spec,
            matched=True,
            peak_time_diff=peak_time_diff,
            peak_value_diff_pct=peak_value_diff_pct,
            rmse=rmse,
            correlation=correlation,
            needs_calibration=needs_calibration,
        )

    # -------------------------------------------------------------------------
    # 报告
    # -------------------------------------------------------------------------
    def format_report(self, result: ComparisonResult) -> str:
        """格式化对比报告为可读文本。"""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("BioModels Parameter Calibration — 对比报告")
        lines.append("=" * 60)
        lines.append(f"通路: {result.pathway}")
        lines.append(f"BioModels ID: {result.biomodel_id}")
        lines.append(f"BioModels 加载: {'成功' if result.biomodel_loaded else '失败'}")

        if result.error:
            lines.append(f"错误: {result.error}")
            return "\n".join(lines)

        lines.append(f"Agent 物种数: {result.agent_species_count}")
        lines.append(f"BioModels 物种数: {result.biomodel_species_count}")
        lines.append(f"匹配物种数: {result.matched_count}")
        lines.append("")

        if result.species_comparisons:
            lines.append(f"{'物种':<25} {'Peak Time差':>12} {'Peak Value差':>14} {'RMSE':>8} {'Corr':>8} {'校准':>6}")
            lines.append("-" * 80)
            for comp in result.species_comparisons:
                if not comp.matched:
                    lines.append(f"{comp.agent_species:<25} {'(未匹配)':>12}")
                    continue
                lines.append(
                    f"{comp.agent_species:<25} "
                    f"{comp.peak_time_diff:>10.1f}m "
                    f"{comp.peak_value_diff_pct:>12.1%} "
                    f"{comp.rmse:>8.3f} "
                    f"{comp.correlation:>8.3f} "
                    f"{'是' if comp.needs_calibration else '否':>6}"
                )

        lines.append("")
        lines.append(f"整体 RMSE: {result.overall_rmse:.3f}")
        lines.append(f"整体 Correlation: {result.overall_correlation:.3f}")
        lines.append(f"需要校准: {'是' if result.needs_calibration else '否'}")

        return "\n".join(lines)


# =============================================================================
# 辅助函数
# =============================================================================
def _verify_sbml_checksum(biomodel_id: str) -> bool:
    """对接 validation_report: 校验本地 SBML cache 的 sha256 是否匹配 manifest。

    复用 biomodels_client.BioModelsAPIClient.cache_provenance 读取 manifest
    并比对本地缓存文件的 sha256。manifest 无该模型条目时返回 False（不伪装通过）。
    """
    if not biomodel_id:
        return False
    try:
        from app.biomodels_client import BioModelsAPIClient
        provenance = BioModelsAPIClient.cache_provenance(biomodel_id)
        return bool(provenance.get("checksum_verified", False))
    except Exception as exc:
        logger.debug("SBML checksum 校验失败 (%s): %s", biomodel_id, exc)
        return False


def _resolve_biomodel_id(pathway: str) -> str:
    """将 pathway 标识解析为 BioModels ID。"""
    if not pathway:
        return ""

    p = pathway.strip()

    # MULTI 通路：取首个子通路
    if p.startswith("MULTI:"):
        parts = p[6:].split("+")
        if parts:
            p = parts[0].strip()

    # 查映射表
    if p in _PATHWAY_TO_BIOMD:
        return _PATHWAY_TO_BIOMD[p]
    p_upper = p.upper()
    if p_upper in _PATHWAY_TO_BIOMD:
        return _PATHWAY_TO_BIOMD[p_upper]

    return ""


def _find_sbml_file(biomodel_id: str) -> Path | None:
    """在 data/raw/ 目录查找 SBML 文件。"""
    if not biomodel_id:
        return None

    # 优先顶层 XML
    xml_path = _BIOMODELS_DIR / f"{biomodel_id}.xml"
    if xml_path.exists():
        return xml_path

    # 递归查找
    for p in _BIOMODELS_DIR.rglob(f"{biomodel_id}*.xml"):
        return p

    return None


def _build_species_id_name_map(sbml_text: str) -> dict[str, str]:
    """从 SBML XML 构建 species_id → species_name 映射。

    若 species 有 name 属性且与 id 不同，使用 name；否则用 id。
    """
    try:
        from xml.etree import ElementTree as ET
        root = ET.fromstring(sbml_text)
        # 获取命名空间
        ns: dict[str, str] = {}
        if "}" in root.tag:
            ns_uri = root.tag.split("}")[0][1:]
            ns = {"sbml": ns_uri}

        # 定位 model 元素
        if ns:
            model_elem = root.find("sbml:model", ns)
        else:
            model_elem = root.find("model")
        if model_elem is None:
            model_elem = root

        # 查找所有 species
        if ns:
            species_list = model_elem.findall(".//sbml:species", ns)
        else:
            species_list = model_elem.findall(".//species")

        id_to_name: dict[str, str] = {}
        for sp in species_list:
            sp_id = sp.get("id", "")
            sp_name = sp.get("name", "")
            if sp_id and sp_name and sp_name != sp_id:
                id_to_name[sp_id] = sp_name
        return id_to_name
    except Exception as exc:
        logger.debug("构建 species_id→name 映射失败: %s", exc)
        return {}


def _load_csv_timeseries(csv_path: str | Path) -> dict[str, np.ndarray]:
    """加载 CSV 时间序列数据。

    Returns:
        {column_name: np.ndarray}，其中 "time" 列为时间轴。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")

    data: dict[str, list[float]] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col, val in row.items():
                if col not in data:
                    data[col] = []
                try:
                    data[col].append(float(val))
                except (ValueError, TypeError):
                    data[col].append(0.0)

    # 归一化列名：time/t/Time → "time"
    result: dict[str, np.ndarray] = {}
    for col, vals in data.items():
        col_lower = col.lower().strip()
        if col_lower in ("time", "t", "times"):
            result["time"] = np.array(vals)
        else:
            result[col] = np.array(vals)

    return result


def _extract_species_tokens(name: str) -> list[str]:
    """从物种名提取生物学 token（大写缩写 + 数字组合）。

    拆分规则：
      - camelCase 边界（pEGFR → ["p", "EGFR"]）
      - 非字母数字字符（EGF-EGFR → ["EGF", "EGFR"]）
      - 数字边界（ERK1 → ["ERK", "1"]）

    仅保留长度 >= 3 的字母 token（如 EGF, EGFR, ERK, MAPK），
    过滤掉 "p"（磷酸化前缀）等短 token。
    """
    if not name:
        return []
    # camelCase 拆分 + 非字母数字拆分
    # 先在 camelCase 边界插入空格：pEGFR → p EGFR
    split1 = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    # 在非字母数字处拆分
    tokens = re.split(r"[^A-Za-z0-9]+", split1)
    # 过滤：仅保留长度 >= 3 的字母 token
    return [t for t in tokens if len(t) >= 3 and t.isalpha()]


def _strip_phospho_prefix(name: str) -> str:
    """去除磷酸化前缀（pEGFR → EGFR, pERK → ERK）。

    仅当 "p" 后紧跟大写字母时才去除（避免误伤 "p53" 等）。
    """
    if not name:
        return name
    # p 后跟大写字母：pEGFR → EGFR, pERK → ERK
    # 但 p53 保留（p 后跟数字）
    m = re.match(r"^p([A-Z][A-Za-z0-9]*)$", name)
    if m:
        return m.group(1)
    return name


def _fuzzy_match_species(
    agent_species: str,
    biomodel_data: dict[str, np.ndarray],
) -> str:
    """Fuzzy match Agent 物种名到 BioModels 物种名。

    策略（按严格度递减）：
      1. 精确匹配（忽略大小写）
      2. 磷酸化感知匹配：pEGFR → EGFR（去前缀后精确匹配）
      3. Token 匹配：提取核心 token（EGF/EGFR/ERK/MAPK），主 token 相同才算匹配
      4. 关键词匹配：至少 2 个关键词命中（避免 EGF 误匹配 pEGFR）

    严格约束：pEGFR 不得匹配 EGF（不同分子），EGF 不得匹配 EGFR（不同分子）。
    """
    if not agent_species:
        return ""

    agent_lower = agent_species.lower().strip()
    biomodel_species = [s for s in biomodel_data.keys() if s != "time"]

    # 1. 精确匹配（忽略大小写）
    for bm in biomodel_species:
        if bm.lower().strip() == agent_lower:
            return bm

    # 2. 磷酸化感知匹配：pEGFR → EGFR
    agent_stripped = _strip_phospho_prefix(agent_species)
    if agent_stripped != agent_species:  # 前缀被去除
        for bm in biomodel_species:
            bm_stripped = _strip_phospho_prefix(bm)
            # 去前缀后精确匹配（pEGFR ↔ EGFR，或 pEGFR ↔ pEGFR）
            if bm.lower().strip() == agent_stripped.lower().strip():
                return bm
            if bm_stripped.lower().strip() == agent_stripped.lower().strip():
                return bm

    # 3. Token 匹配：主 token（最长 token）必须相同
    agent_tokens = _extract_species_tokens(agent_species)
    if agent_tokens:
        # 主 token = 最长的 token（如 "EGFR" > "EGF"）
        agent_main = max(agent_tokens, key=len)
        agent_main_lower = agent_main.lower()

        # 先找主 token 完全相同的
        for bm in biomodel_species:
            bm_tokens = _extract_species_tokens(bm)
            if not bm_tokens:
                continue
            bm_main = max(bm_tokens, key=len)
            if bm_main.lower() == agent_main_lower:
                return bm

        # 再找主 token 是对方主 token 的前缀/后缀（EGFR ↔ EGFR_complex）
        for bm in biomodel_species:
            bm_tokens = _extract_species_tokens(bm)
            if not bm_tokens:
                continue
            bm_main = max(bm_tokens, key=len)
            bm_main_lower = bm_main.lower()
            # 仅当长度 >= 4 时允许前缀匹配（避免 EGF↔EGFR）
            if len(agent_main_lower) >= 4 and len(bm_main_lower) >= 4:
                if (agent_main_lower.startswith(bm_main_lower)
                        or bm_main_lower.startswith(agent_main_lower)):
                    return bm

    # 4. 关键词匹配：至少 2 个关键词命中（严格阈值，避免误匹配）
    keywords = [t.lower() for t in agent_tokens if len(t) >= 3]
    if len(keywords) >= 2:
        best_match = ""
        best_score = 0
        for bm in biomodel_species:
            bm_lower = bm.lower()
            score = sum(1 for kw in keywords if kw in bm_lower)
            if score > best_score:
                best_score = score
                best_match = bm
        # 至少匹配 2 个关键词
        if best_score >= 2:
            return best_match

    return ""


def _normalize_time(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """归一化时间序列值到 0-1 范围。"""
    if len(values) == 0:
        return values
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < 1e-10:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _resample(values: np.ndarray, n_points: int) -> np.ndarray:
    """重采样到 n_points 个点（线性插值）。"""
    if len(values) <= 1:
        return values
    if len(values) == n_points:
        return values
    try:
        x_old = np.linspace(0, 1, len(values))
        x_new = np.linspace(0, 1, n_points)
        return np.interp(x_new, x_old, values)
    except Exception:
        return values


# =============================================================================
# 便捷函数
# =============================================================================
def compare_with_biomodels(
    agent_csv_path: str | Path,
    pathway: str,
    duration: float = 120.0,
    n_points: int = 500,
) -> ComparisonResult:
    """便捷函数：对比 Agent ODE 仿真与 BioModels 参考仿真。

    Args:
        agent_csv_path: Agent ODE 仿真 CSV 路径。
        pathway: 通路标识（v4_pathway_class）。
        duration: 仿真时长（分钟）。
        n_points: 仿真点数。

    Returns:
        ComparisonResult。
    """
    comparator = BioModelsComparator()
    return comparator.compare(
        agent_csv_path=agent_csv_path,
        pathway=pathway,
        duration=duration,
        n_points=n_points,
    )


__all__ = [
    "SpeciesComparison",
    "ComparisonResult",
    "BioModelsComparator",
    "compare_with_biomodels",
]
