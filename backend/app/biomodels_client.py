# BioDynamics Agent - BioModels REST API 客户端
# 对应 EGF-EGFR错误结论根因与后续修复计划报告.md §5.1.4 与 §5.4.1：
# 通过 EBI BioModels REST API 按需下载 SBML，不依赖本地文件作为默认数据源。
# 本地 backend/data/raw/ 仅作为 API 不可用时的缓存兜底。

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


# EBI BioModels REST API 端点（参考 https://www.ebi.ac.uk/biomodels/)
_BIOMODELS_BASE = "https://www.ebi.ac.uk/biomodels"
_SEARCH_ENDPOINT = f"{_BIOMODELS_BASE}/search"
_MODEL_ENDPOINT = f"{_BIOMODELS_BASE}/{{model_id}}"
_DOWNLOAD_ENDPOINT = f"{_BIOMODELS_BASE}/model/download/{{model_id}}"

# 默认超时（秒），BioModels 偶尔较慢
_DEFAULT_TIMEOUT = 20.0

# 本地缓存目录：API 失败时兜底
_LOCAL_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# 用户输入中的 BIOMD/MODEL ID 正则
_BIOMD_ID_RE = re.compile(r"\b(BIOMD\d{10,}|MODEL\d{10,})\b", re.IGNORECASE)


# -----------------------------------------------------------------------------
# SBML 三角色定位（修复提示词1.md §一评论 #1：必须采纳）
# SBML 在系统中有 3 种角色，而非 2 种：
#   1. Primary Ground Truth — 用户显式提到 BIOMD*/MODEL* ID，SBML 是 ground truth
#   2. Calibration Reference — 自由建模时，用 SBML 参数校准模板 ODE
#   3. Validation Oracle — 仿真后用 SBML 仿真作为"金标准"对比模板仿真
# -----------------------------------------------------------------------------
SBML_ROLE_PRIMARY_GROUND_TRUTH = "primary_ground_truth"
SBML_ROLE_CALIBRATION_REFERENCE = "calibration_reference"
SBML_ROLE_VALIDATION_ORACLE = "validation_oracle"
SBML_ROLE_NONE = "none"


def detect_sbml_role(
    user_input: str,
    has_simulation_run: bool = False,
    sbml_model_id: str = "",
    has_loaded_sbml: bool = False,
) -> str:
    """根据用户输入与流水线阶段严格判断 SBML 在本次任务中的角色。

    深度审核报告 §1.2 三角色定位逻辑加固：
    - primary_ground_truth: 建模阶段，用户显式提供 BIOMD*/MODEL* ID，
      SBML 作为参数注入的唯一权威源，禁止与 validation_oracle 混用
    - calibration_reference: 用户未提供 ID 但通路关键词能匹配 BioModels，
      SBML 仅用于参数校准（不作为金标准）
    - validation_oracle: 仿真已跑完且 SBML 可用，SBML 仿真作为金标准对比模板仿真
    - none: 无 SBML 可用

    严格区分规则（防止角色混用）：
    1. has_simulation_run=False + 有 BIOMD* ID → primary_ground_truth（建模阶段）
    2. has_simulation_run=True + 有 BIOMD* ID + 已加载 SBML → validation_oracle（仿真后验证）
    3. has_simulation_run=True + 有 BIOMD* ID + 未加载 SBML → calibration_reference（兜底）
    4. 无 ID 但有通路关键词 → calibration_reference（始终为校准参考，不升级为 ground truth）
    5. 都不匹配 → none

    Args:
        user_input: 用户原始问题文本。
        has_simulation_run: 当前流水线是否已经跑完 sandbox 仿真
            （False=处于建模/参数注入阶段，True=处于仿真后验证阶段）。
        sbml_model_id: 已加载的 SBML 模型 ID（来自 state.sbml_model_id）。
            若与 user_input 中的 ID 一致，说明 SBML 已在建模阶段被使用。
        has_loaded_sbml: SBML XML 文本是否已成功加载到 state.sbml_model_text。

    Returns:
        SBML_ROLE_* 枚举值之一。
    """
    if not user_input:
        return SBML_ROLE_NONE

    has_biomd_id = bool(_BIOMD_ID_RE.search(user_input))

    if has_biomd_id:
        if not has_simulation_run:
            # 建模阶段：SBML 作为参数注入的权威源
            return SBML_ROLE_PRIMARY_GROUND_TRUTH
        else:
            # 仿真后阶段：严格区分验证 vs 校准
            # 仅当 SBML 已加载且 ID 匹配时才升级为 validation_oracle
            # 否则降级为 calibration_reference（避免角色混用）
            if has_loaded_sbml and sbml_model_id:
                return SBML_ROLE_VALIDATION_ORACLE
            else:
                # 仿真后但 SBML 未加载：只能做参数校准参考
                return SBML_ROLE_CALIBRATION_REFERENCE

    # 无显式 ID：若通路关键词可匹配 BioModels，则为 calibration reference
    # 严格规则：calibration_reference 永远不升级为 primary_ground_truth 或 validation_oracle
    _PATHWAY_KEYWORDS = (
        "egf", "egfr", "mapk", "erk", "ras", "raf", "mek", "shc", "grb2", "sos",
        "tgf", "smad", "pi3k", "akt", "mtor", "nfkb", "apoptosis", "cas9", "crispr",
        "erk", "jnkk", "jak", "stat", "vegf", "tnf", "il-6", "il-2",
        # 深度审核报告 §4.3 扩充：覆盖更多细分领域
        "pd1", "pd-l1", "ctla4", "tumor immun", "metabolism", "insulin",
        "wnt", "notch", "hedgehog", "cell cycle", "p53",
    )
    text_lower = user_input.lower()
    if any(kw in text_lower for kw in _PATHWAY_KEYWORDS):
        return SBML_ROLE_CALIBRATION_REFERENCE

    return SBML_ROLE_NONE


class BioModelsAPIClient:
    """EBI BioModels REST API 客户端。

    提供三个核心能力：
    1. search(query) — 按关键词搜索模型，返回 top-k 候选；
    2. download(model_id) — 下载指定模型的 SBML XML 文本；
    3. load_sbml_for_user_input(user_input) — 高阶入口：从用户输入识别
       BIOMD* / MODEL* ID 或通路关键词，自动选择并下载 SBML。

    所有网络调用失败时安全降级：
    - search 失败 → 返回空列表；
    - download 失败 → 优先从本地 backend/data/raw/{model_id}.xml 读取；
    - 二者皆失败 → 返回空字符串，调用方继续走 LLM 推断路径。
    """

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self._session = requests.Session()
        # EBI 建议携带 User-Agent
        self._session.headers.update({
            "User-Agent": "BioDynamics-Agent/1.0 (research)",
            "Accept": "application/json, application/xml, */*",
        })

    # -------------------------------------------------------------------------
    # 公开 API
    # -------------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """按关键词搜索 BioModels，返回候选模型列表。

        Args:
            query: 通路/机制关键词，如 "EGF EGFR MAPK signaling"。
            top_k: 返回最多多少条候选。

        Returns:
            [{model_id, name, format, submissionDate, publication_id}, ...]
            失败返回空列表。
        """
        params = {
            "query": query,
            "format": "json",
            "numResults": min(top_k, 100),
        }
        try:
            resp = self._session.get(
                _SEARCH_ENDPOINT,
                params=params,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # BioModels search API: "matches" 是总命中数（int），模型列表在 "models"
            models = data.get("models", []) or []
            results: list[dict[str, Any]] = []
            for m in models[:top_k]:
                results.append({
                    "model_id": m.get("id", ""),
                    "name": m.get("name", ""),
                    "format": m.get("format", ""),
                    "submissionDate": m.get("submissionDate", ""),
                    "publication_id": m.get("publicationId", ""),
                })
            return results
        except Exception as exc:
            logger.warning("BioModels search 失败 (query=%s): %s", query[:50], exc)
            return []

    def download(self, model_id: str) -> str:
        """下载指定模型的 SBML XML 文本。

        优先调用 BioModels /model/download/{id} 端点获取 SBML；
        失败时回退到本地 backend/data/raw/{model_id}.xml 缓存。

        Returns:
            SBML XML 文本；失败返回空字符串。
        """
        if not model_id:
            return ""
        model_id = model_id.strip()
        # 1. 在线下载
        try:
            url = _DOWNLOAD_ENDPOINT.format(model_id=model_id)
            # BioModels 下载要求 filename={model_id}_url.xml，否则返回 400 或 zip 包
            resp = self._session.get(
                url,
                params={"filename": f"{model_id}_url.xml"},
                timeout=self.timeout,
            )
            if resp.status_code == 200 and resp.text.strip():
                # 简单校验是否是 XML
                text = resp.text
                if "<sbml" in text[:2000].lower() or "<?xml" in text[:200]:
                    logger.info("BioModels 下载成功: %s (size=%d)", model_id, len(text))
                    self._cache_to_local(model_id, text)
                    return text
                logger.warning(
                    "BioModels 返回非 SBML 内容: %s, 前100字符: %s",
                    model_id, text[:100],
                )
        except Exception as exc:
            logger.warning("BioModels download 失败 (model_id=%s): %s", model_id, exc)

        # 2. 本地缓存兜底
        cached = self._load_from_local_cache(model_id)
        if cached:
            logger.info("BioModels 在线下载失败，使用本地缓存: %s", model_id)
            return cached
        return ""

    def load_sbml_for_user_input(
        self,
        user_input: str,
        max_keywords_attempts: int = 1,
    ) -> tuple[str, str]:
        """高阶入口：从用户输入识别模型 ID 或通路关键词，自动下载 SBML。

        识别顺序：
        1. 若 user_input 含 BIOMD*/MODEL* ID → 直接下载该 ID；
        2. 否则用通路关键词（如 "EGF EGFR MAPK"）调 /search 取 top-1，再下载；
        3. 失败返回空字符串。

        Returns:
            (model_id, sbml_text)；二者皆失败时 sbml_text 为空字符串。
        """
        if not user_input:
            return "", ""

        # 1. 直接识别 BIOMD*/MODEL*
        match = _BIOMD_ID_RE.search(user_input)
        if match:
            model_id = match.group(1).upper()
            sbml = self.download(model_id)
            if sbml:
                return model_id, sbml
            # 在线失败 + 本地缓存也失败 → 返回空
            return model_id, ""

        # 2. 关键词搜索 top-1
        # 提取通路关键词：去除常见停用词，保留生物学实体
        keywords = self._extract_pathway_keywords(user_input)
        if not keywords:
            return "", ""

        for attempt in range(max_keywords_attempts):
            query = " ".join(keywords)
            candidates = self.search(query, top_k=3)
            if not candidates:
                continue
            # 优先选择 SBML 格式 + 有 publication 的候选
            best = self._pick_best_candidate(candidates)
            if not best:
                continue
            model_id = best.get("model_id", "")
            if not model_id:
                continue
            sbml = self.download(model_id)
            if sbml:
                logger.info(
                    "BioModels 关键词检索命中: query=%s → model_id=%s name=%s",
                    query, model_id, best.get("name", ""),
                )
                return model_id, sbml
        return "", ""

    # -------------------------------------------------------------------------
    # 内部辅助
    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_pathway_keywords(text: str) -> list[str]:
        """从用户输入提取通路关键词，去掉停用词。"""
        # 简单分词与停用词过滤
        stop_words = {
            "the", "a", "an", "of", "and", "or", "to", "in", "on", "for",
            "with", "by", "is", "are", "was", "were", "be", "been",
            "this", "that", "these", "those", "as", "at", "from",
            "model", "simulation", "process", "dynamics", "kinetic",
            "的", "了", "在", "与", "和", "或", "基于", "模拟", "动力学",
            "信号", "级联", "通路",
        }
        # 英文词 + 中文连续字符（粗略）
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]*|[\u4e00-\u9fa5]+", text)
        keywords = [
            t for t in tokens
            if t.lower() not in stop_words and len(t) >= 2
        ]
        # 去重保序
        seen: set[str] = set()
        result: list[str] = []
        for k in keywords:
            kl = k.lower()
            if kl not in seen:
                seen.add(kl)
                result.append(k)
        return result[:8]  # 限制关键词数量避免 query 过长

    @staticmethod
    def _pick_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        """从候选列表中挑选最佳模型：优先 SBML 格式 + 有 publication。"""
        if not candidates:
            return None
        # 评分：format=SBML +2，有 publication +1
        def _score(c: dict[str, Any]) -> int:
            s = 0
            fmt = str(c.get("format", "")).upper()
            if "SBML" in fmt or fmt == "SBML":
                s += 2
            if c.get("publication_id"):
                s += 1
            return s
        return max(candidates, key=_score)

    @staticmethod
    def _load_from_local_cache(model_id: str) -> str:
        """从 backend/data/raw/{model_id}.xml 读取缓存的 SBML。"""
        path = _LOCAL_CACHE_DIR / f"{model_id}.xml"
        try:
            if path.exists():
                return path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("读取本地 SBML 缓存失败 (%s): %s", path, exc)
        return ""

    @staticmethod
    def _cache_to_local(model_id: str, sbml_text: str) -> None:
        """把下载的 SBML 写入本地缓存，便于离线复用。"""
        try:
            _LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path = _LOCAL_CACHE_DIR / f"{model_id}.xml"
            path.write_text(sbml_text, encoding="utf-8")
        except Exception as exc:
            logger.warning("写入本地 SBML 缓存失败 (%s): %s", model_id, exc)


# -----------------------------------------------------------------------------
# 全局实例（懒加载）
# -----------------------------------------------------------------------------
_global_client: BioModelsAPIClient | None = None


def get_biomodels_client() -> BioModelsAPIClient:
    """获取（或懒加载）全局 BioModelsAPIClient 实例。"""
    global _global_client
    if _global_client is None:
        _global_client = BioModelsAPIClient()
    return _global_client


# -----------------------------------------------------------------------------
# 便捷函数：用户输入中是否显式提到 BIOMD*/MODEL*
# -----------------------------------------------------------------------------
def extract_biomodel_id(user_input: str) -> str:
    """从用户输入提取 BIOMD*/MODEL* 模型 ID；不存在返回空字符串。"""
    if not user_input:
        return ""
    match = _BIOMD_ID_RE.search(user_input)
    return match.group(1).upper() if match else ""
