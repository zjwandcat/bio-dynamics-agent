# BioDynamics Agent v4 - Scientific Alignment Loop: Evidence Ranker (Task 3.3)
#
# 文献级 Gold Standard 与 Evidence Ranking Score 实现。
# 设计目标：约束 Retriever 不能自行决定什么文献最好，必须按
#   Review(5) > Mechanism(4) > BioModels Source(3) > Recent Application(2) > Case Report(1)
# 的固定优先级排序。PubMed 检索结果仅当经典文献不足时补充，
# 不得让随机新论文排在经典文献前面。
#
# 依赖：PyYAML（已存在）、Python 标准库；不引入新依赖。
#
# 核心导出：
#   from app.scientific_alignment.evidence_ranker import (
#       EvidenceType, EvidenceDoc, EvidenceRanker,
#       load_literature_gold_standard,
#   )

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================
# 文献级 Gold Standard YAML 根目录：backend/knowledge/gold_standard/
GOLD_STANDARD_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent / "knowledge" / "gold_standard"
)

# pathway 白名单正则：仅允许 [a-zA-Z0-9_]，防止路径遍历与非法字符注入
_PATHWAY_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_]+$")

# 评分调参常量
_CITATION_THRESHOLD: int = 100       # 引用数达到该阈值给予 citation_bonus
_CITATION_BONUS: float = 0.5         # 高引用文献加分
_RECENT_YEAR_THRESHOLD: int = 2020   # 太新文献的年份阈值
_RECENCY_PENALTY: float = 0.3        # 太新但非经典文献的减分


# =============================================================================
# EvidenceType 枚举
# =============================================================================
class EvidenceType(IntEnum):
    """文献证据类型枚举，数值即基础分（base_score）。

    优先级与 ranking_policy 一致：
      REVIEW > MECHANISM_PAPER > BIOMODELS_SOURCE > RECENT_APPLICATION > CASE_REPORT
    """

    REVIEW = 5              # 经典综述（最高优先级）
    MECHANISM_PAPER = 4     # 机制论文
    BIOMODELS_SOURCE = 3    # BioModels 来源论文
    RECENT_APPLICATION = 2  # 近期应用论文（仅补充）
    CASE_REPORT = 1         # 案例报告（一般不用）


# =============================================================================
# EvidenceDoc 数据类
# =============================================================================
@dataclass
class EvidenceDoc:
    """文献证据文档。

    Attributes:
        pmid: PubMed ID（可为 "PMID:xxxxxxxx" 或纯数字字符串）。
        title: 文献标题。
        year: 发表年份。
        journal: 期刊名称（可空）。
        evidence_type: 证据类型，默认 RECENT_APPLICATION。
        base_score: 基础分；若 <=0 则在 __post_init__ 中从 evidence_type 推导。
        citation_count: 引用数（可选，默认 0）。
        recency_bonus: 近期相关性加分（外部设置，默认 0.0）。
    """

    pmid: str
    title: str
    year: int
    journal: str = ""
    evidence_type: EvidenceType = EvidenceType.RECENT_APPLICATION
    base_score: float = 0.0
    citation_count: int = 0
    recency_bonus: float = 0.0

    def __post_init__(self) -> None:
        # 若未显式设置 base_score，则从 evidence_type 数值推导
        if self.base_score <= 0:
            self.base_score = float(int(self.evidence_type))


# =============================================================================
# EvidenceRanker 排序器
# =============================================================================
class EvidenceRanker:
    """Evidence Ranking Score 排序器。

    根据文献级 Gold Standard 元数据对 EvidenceDoc 进行分类、评分与排序，
    确保经典文献永远排在近期 PubMed 应用论文之前。

    Usage::

        gold = load_literature_gold_standard("egfr")
        ranker = EvidenceRanker(gold_standard=gold)
        doc_type = ranker.classify("PMID:7657691")
        score = ranker.score(EvidenceDoc(pmid="PMID:7657691", ...))
        ranked = ranker.rank(docs)
        trace = ranker.get_ranking_trace(docs)
    """

    def __init__(
        self,
        gold_standard: dict[str, Any] | None = None,
        citation_threshold: int = _CITATION_THRESHOLD,
        citation_bonus: float = _CITATION_BONUS,
        recent_year_threshold: int = _RECENT_YEAR_THRESHOLD,
        recency_penalty: float = _RECENCY_PENALTY,
    ) -> None:
        """初始化排序器。

        Args:
            gold_standard: 文献级 Gold Standard dict（来自
                ``load_literature_gold_standard``）。为 None 或空 dict 时
                退化为纯 PubType 推断模式。
            citation_threshold: 引用数阈值，达到则给予 citation_bonus。
            citation_bonus: 高引用文献加分。
            recent_year_threshold: 太新文献的年份阈值（>= 该年份视为太新）。
            recency_penalty: 太新但非经典文献的减分（正值，从 score 中扣除）。
        """
        self._gold_standard: dict[str, Any] = gold_standard or {}
        self._citation_threshold = citation_threshold
        self._citation_bonus = citation_bonus
        self._recent_year_threshold = recent_year_threshold
        self._recency_penalty = recency_penalty
        # PMID -> EvidenceType 索引（按优先级 review > mechanism > biomodels_source）
        self._pmid_index: dict[str, EvidenceType] = {}
        self._build_pmid_index()

    # -------------------------------------------------------------------------
    # Gold Standard 索引构建
    # -------------------------------------------------------------------------
    def _build_pmid_index(self) -> None:
        """从 Gold Standard 构建 PMID -> EvidenceType 索引。

        按优先级顺序填充：classical_reviews > mechanism_papers >
        biomodels_source_papers。若同一 PMID 出现在多个分类中，保留最高优先级。
        """
        gs = self._gold_standard
        for pmid in self._extract_pmids(gs.get("classical_reviews", [])):
            self._pmid_index[pmid] = EvidenceType.REVIEW
        for pmid in self._extract_pmids(gs.get("mechanism_papers", [])):
            if pmid not in self._pmid_index:
                self._pmid_index[pmid] = EvidenceType.MECHANISM_PAPER
        for pmid in self._extract_pmids(gs.get("biomodels_source_papers", [])):
            if pmid not in self._pmid_index:
                self._pmid_index[pmid] = EvidenceType.BIOMODELS_SOURCE

    @staticmethod
    def _extract_pmids(items: Any) -> set[str]:
        """从 YAML 列表中提取并归一化所有 PMID。

        归一化：去除 "PMID:" 前缀，保留纯数字字符串。跳过 "TODO" 占位符。
        """
        result: set[str] = set()
        if not items or not isinstance(items, list):
            return result
        for item in items:
            if not isinstance(item, dict):
                continue
            pmid = item.get("pmid", "")
            normalized = EvidenceRanker._normalize_pmid(pmid)
            if normalized and not normalized.upper().startswith("TODO"):
                result.add(normalized)
        return result

    @staticmethod
    def _normalize_pmid(pmid: Any) -> str:
        """归一化 PMID：去除 'PMID:' 前缀，strip 空白。"""
        if not pmid:
            return ""
        return str(pmid).replace("PMID:", "").replace("pmid:", "").strip()

    # -------------------------------------------------------------------------
    # 分类
    # -------------------------------------------------------------------------
    def classify(
        self,
        pmid: str,
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceType:
        """根据文献级 Gold Standard 元数据分类文献。

        分类优先级：
          1. 若 PMID 在 classical_reviews -> REVIEW
          2. 若 PMID 在 mechanism_papers -> MECHANISM_PAPER
          3. 若 PMID 在 biomodels_source_papers -> BIOMODELS_SOURCE
          4. 否则根据 metadata['pub_types'] 推断（含 'review' -> REVIEW；
             含 'case report' -> CASE_REPORT）
          5. 默认 RECENT_APPLICATION

        Args:
            pmid: PubMed ID（可带 "PMID:" 前缀）。
            metadata: 文献元数据 dict，可含 'pub_types' 列表。

        Returns:
            EvidenceType 枚举值。
        """
        normalized = self._normalize_pmid(pmid)

        # 1-3. Gold Standard 索引查找
        if normalized and normalized in self._pmid_index:
            return self._pmid_index[normalized]

        # 4. 根据 PubType 推断
        metadata = metadata or {}
        pub_types = metadata.get("pub_types") or metadata.get("publication_types") or []
        if isinstance(pub_types, list):
            for pt in pub_types:
                pt_lower = str(pt).lower()
                if "review" in pt_lower:
                    return EvidenceType.REVIEW
                if "case report" in pt_lower or "case reports" in pt_lower:
                    return EvidenceType.CASE_REPORT

        # 5. 默认 RECENT_APPLICATION
        return EvidenceType.RECENT_APPLICATION

    # -------------------------------------------------------------------------
    # 评分
    # -------------------------------------------------------------------------
    def score(self, doc: EvidenceDoc) -> float:
        """计算 Evidence Ranking Score。

        公式：base_score + citation_bonus + recency_bonus - recency_penalty

        - citation_bonus：citation_count >= 阈值时加 ``citation_bonus``
        - recency_penalty：仅对非经典文献（非 REVIEW/MECHANISM）且
          year >= recent_year_threshold 时扣 ``recency_penalty``；经典文献不受影响

        Args:
            doc: EvidenceDoc 实例。

        Returns:
            最终评分（float）。
        """
        # base_score：优先使用 doc.base_score，否则从 evidence_type 推导
        base = (
            float(doc.base_score)
            if doc.base_score and doc.base_score > 0
            else float(int(doc.evidence_type))
        )
        score = base

        # citation_bonus：高引用文献加分
        if doc.citation_count and doc.citation_count >= self._citation_threshold:
            score += self._citation_bonus

        # recency_bonus：外部设置的近期相关性加分（直接累加）
        score += float(doc.recency_bonus or 0.0)

        # recency_penalty：太新但非经典文献减分（经典文献不受影响）
        is_classical = doc.evidence_type in (
            EvidenceType.REVIEW,
            EvidenceType.MECHANISM_PAPER,
        )
        if (
            not is_classical
            and doc.year
            and doc.year >= self._recent_year_threshold
        ):
            score -= self._recency_penalty

        return score

    # -------------------------------------------------------------------------
    # 排序
    # -------------------------------------------------------------------------
    def rank(self, docs: list[EvidenceDoc]) -> list[EvidenceDoc]:
        """按 Evidence Ranking Score 降序排序。

        强制约束：经典 Review 永远排在近期 Application 前。
        实现方式：主排序键为 evidence_type 数值降序（保证
        Review > Mechanism > BioModels Source > Recent Application > Case Report
        的绝对层级），次排序键为 score 降序（同层级内按分数细化）。

        Args:
            docs: EvidenceDoc 列表。

        Returns:
            排序后的 EvidenceDoc 列表（新列表，不修改输入）。
        """
        return sorted(
            docs,
            key=lambda d: (-int(d.evidence_type), -self.score(d)),
        )

    # -------------------------------------------------------------------------
    # 排序轨迹（供 Evidence Fusion 读取）
    # -------------------------------------------------------------------------
    def get_ranking_trace(self, docs: list[EvidenceDoc]) -> list[dict[str, Any]]:
        """返回每个 doc 的 per-doc score/rank/evidence_type 中间态。

        供 Evidence Fusion 模块读取，用于可解释性与审计。

        Args:
            docs: EvidenceDoc 列表。

        Returns:
            排序后的中间态 dict 列表，每项包含::

                {
                    "rank": int,                # 1-based 排名
                    "pmid": str,
                    "title": str,
                    "year": int,
                    "evidence_type": str,       # 枚举名（如 "REVIEW"）
                    "base_score": float,
                    "final_score": float,
                    "citation_count": int,
                    "recency_bonus": float,
                }
        """
        ranked = self.rank(docs)
        trace: list[dict[str, Any]] = []
        for idx, doc in enumerate(ranked, start=1):
            trace.append(
                {
                    "rank": idx,
                    "pmid": doc.pmid,
                    "title": doc.title,
                    "year": doc.year,
                    "evidence_type": doc.evidence_type.name,
                    "base_score": (
                        float(doc.base_score)
                        if doc.base_score and doc.base_score > 0
                        else float(int(doc.evidence_type))
                    ),
                    "final_score": self.score(doc),
                    "citation_count": doc.citation_count,
                    "recency_bonus": doc.recency_bonus,
                }
            )
        return trace


# =============================================================================
# 文献级 Gold Standard 加载（模块级函数）
# =============================================================================
def load_literature_gold_standard(pathway: str) -> dict[str, Any]:
    """从 ``knowledge/gold_standard/literature_<pathway>.yaml`` 加载 Gold Standard。

    做路径遍历防护：
      1. pathway 必须匹配 ``[a-zA-Z0-9_]+`` 白名单（拒绝 ``..`` / ``/`` / ``\\``）
      2. 二次校验：解析后的绝对路径必须在 GOLD_STANDARD_DIR 内

    Args:
        pathway: 通路标识（如 ``"egfr"``、``"pi3k_akt_mtor"``）。

    Returns:
        解析后的 YAML dict；文件不存在或解析失败时返回空 dict。

    Raises:
        ValueError: pathway 含非法字符或检测到路径遍历攻击。
    """
    # 1. 白名单校验
    if not pathway or not _PATHWAY_PATTERN.match(pathway):
        raise ValueError(
            f"非法 pathway 标识: {pathway!r}（仅允许 [a-zA-Z0-9_]）"
        )

    # 2. 构造文件路径（pathway 已通过白名单，无法包含路径分隔符）
    file_path = GOLD_STANDARD_DIR / f"literature_{pathway}.yaml"

    # 3. 二次防护：确保解析后路径仍在 GOLD_STANDARD_DIR 内
    try:
        file_path.resolve().relative_to(GOLD_STANDARD_DIR.resolve())
    except ValueError as exc:
        raise ValueError(
            f"路径遍历攻击检测: pathway={pathway!r} 解析后越出 gold_standard 目录"
        ) from exc

    if not file_path.exists():
        logger.warning(
            "load_literature_gold_standard: 文件不存在: %s", file_path
        )
        return {}

    try:
        with file_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            logger.warning(
                "load_literature_gold_standard: %s 顶层不是 dict（%s），返回空 dict",
                file_path.name,
                type(data).__name__,
            )
            return {}
        return data
    except yaml.YAMLError as exc:
        logger.warning(
            "load_literature_gold_standard: %s YAML 解析错误: %s",
            file_path.name,
            exc,
        )
        return {}
    except OSError as exc:
        logger.warning(
            "load_literature_gold_standard: %s 读取失败: %s",
            file_path.name,
            exc,
        )
        return {}


__all__ = [
    "EvidenceType",
    "EvidenceDoc",
    "EvidenceRanker",
    "load_literature_gold_standard",
]
