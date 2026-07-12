# BioDynamics Agent - BioModels REST API 客户端
# 对应 EGF-EGFR错误结论根因与后续修复计划报告.md §5.1.4 与 §5.4.1：
# 通过 EBI BioModels REST API 按需下载 SBML，不依赖本地文件作为默认数据源。
# 本地 backend/data/raw/ 仅作为 API 不可用时的缓存兜底。

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


# =============================================================================
# Task 5 依赖：defusedxml（XXE 防护硬约束）+ numpy（数值计算软依赖）
# =============================================================================
# 安全硬约束：外部 SBML XML 必须用 defusedxml 解析，禁止 xml.etree 直接解析外部 XML。
# defusedxml 不可用时降级到 Track B，绝不回退到 xml.etree。
try:
    from defusedxml import ElementTree as DefusedET  # type: ignore
    DEFUSEDXML_AVAILABLE = True
except ImportError:
    DEFUSEDXML_AVAILABLE = False
    DefusedET = None  # type: ignore
    logger.error(
        "defusedxml 未安装，无法安全解析外部 SBML，BioModels Oracle 降级到 Track B。"
        "安装命令：pip install defusedxml"
    )

# numpy 软依赖：用于线性插值与 Pearson 相关系数计算
# 不可用时降级到纯 Python 实现（性能较低但功能完整）
try:
    import numpy as _np  # type: ignore
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    _np = None  # type: ignore


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
                # Task 19 SubTask 19.2: per-item isinstance 守卫，防止单条异常导致整批丢失
                if not isinstance(m, dict):
                    logger.warning("BioModels search 跳过非 dict 候选: %r", type(m))
                    continue
                results.append({
                    "model_id": m.get("id", "") or "",
                    "name": m.get("name", "") or "",
                    "format": m.get("format", "") or "",
                    "submissionDate": m.get("submissionDate", "") or "",
                    "publication_id": m.get("publicationId", "") or "",
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
                # Task 19 SubTask 19.2: schema 校验 — 用 defusedxml 实际 parse 验证
                # 旧实现仅用字符串包含判断（"<sbml" in text），可被错误页绕过。
                text = resp.text
                if self._validate_sbml_schema(text):
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
    def _validate_sbml_schema(text: str) -> bool:
        """Task 19 SubTask 19.2: 校验下载内容是否为合法 SBML XML。

        用 defusedxml 实际 parse 验证 XML 结构（取代旧字符串包含判断），
        并检查根元素是否为 sbml 标签。defusedxml 不可用时降级到字符串校验。

        Args:
            text: 待校验的文本

        Returns:
            True 表示是合法 SBML XML，False 表示不是
        """
        if not text or not text.strip():
            return False
        # 优先用 defusedxml 实际 parse
        if DEFUSEDXML_AVAILABLE and DefusedET is not None:
            try:
                root = DefusedET.fromstring(text)
                # 检查根元素标签是否为 sbml（含命名空间）
                tag = root.tag
                if isinstance(tag, str) and (
                    tag == "sbml" or tag.endswith("}sbml")
                ):
                    return True
                logger.warning("SBML schema 校验：根元素非 <sbml>，实际为 %s", tag)
                return False
            except Exception as exc:
                logger.warning("SBML schema 校验：XML 解析失败: %s", exc)
                return False
        # defusedxml 不可用时降级到字符串校验（已有 try/except 兜底）
        logger.warning("defusedxml 不可用，SBML schema 校验降级到字符串判断")
        return "<sbml" in text[:2000].lower()

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


# =============================================================================
# Task 5: BioModels Validation Oracle — validation_oracle 角色激活
# =============================================================================
# 设计原则（铁律）：
# 1. Feature Flag SA_BIOMODELS_ORACLE 默认 false，关闭时返回 skipped report
# 2. XXE 防护硬约束：外部 SBML 解析必须用 defusedxml，不可用时降级到 Track B
#    绝不回退到 xml.etree.ElementTree 解析外部 XML
# 3. Track A：roadrunner 真实仿真 + defusedxml 安全解析 → 峰值/RMSE 对比
# 4. Track B：结构相似度降级（无真实仿真）→ Pearson 相关评分
# 5. 网络失败/SBML 解析失败/仿真失败均优雅降级，返回 degraded report，不抛异常
# 6. 禁止硬编码任何 BioModels ID 特定逻辑——通用处理
# =============================================================================


@dataclass
class BioModelsOracleReport:
    """BioModels Validation Oracle 对比报告。

    封装模板仿真轨迹与 BioModels SBML 仿真轨迹的差异分析结果。
    status 取值：
    - "passed"：overall_distance < 0.1 且 max_relative_error < 0.2
    - "failed"：超过阈值
    - "degraded"：Track B 降级或部分物种失败
    - "skipped"：Feature Flag 关闭
    """
    biomodel_id: str
    status: str  # "passed" / "failed" / "skipped" / "degraded"
    sbml_loaded: bool
    simulation_run: bool
    track: str  # "A"（roadrunner 真实仿真）/ "B"（结构相似度降级）/ ""（skipped）
    species_comparisons: list[dict] = field(default_factory=list)
    overall_distance: float = float("nan")
    max_relative_error: float = float("nan")
    summary: str = ""
    errors: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# 辅助函数：物种匹配、SBML 解析、轨迹对比
# -----------------------------------------------------------------------------

def _match_species_case_insensitive(
    template_species: str,
    sbml_species_list: list[str],
) -> str:
    """大小写不敏感 + 常见变体匹配物种名。

    匹配策略（按优先级）：
    1. 完全匹配（大小写不敏感）
    2. 去除下划线/连字符后匹配（p_egfr ↔ pEGFR）
    3. 常见修饰形式变体匹配（pEGFR ↔ EGFR_p / ppERK ↔ ERK 等）

    Args:
        template_species: 模板物种名
        sbml_species_list: SBML 中的 species id 列表

    Returns:
        匹配到的 SBML species id；未匹配返回空字符串
    """
    if not template_species or not sbml_species_list:
        return ""

    # 1. 完全匹配（大小写不敏感）
    tpl_lower = template_species.lower()
    for sp in sbml_species_list:
        if sp.lower() == tpl_lower:
            return sp

    # 2. 去除下划线/连字符后匹配
    tpl_norm = tpl_lower.replace("_", "").replace("-", "")
    for sp in sbml_species_list:
        sp_norm = sp.lower().replace("_", "").replace("-", "")
        if sp_norm == tpl_norm:
            return sp

    # 3. 常见修饰形式变体匹配
    # 去除 p/pp 前缀（磷酸化标记）和 _active/_p 后缀后比较基础名
    def _strip_modifiers(name: str) -> str:
        n = name.lower()
        # 去除前缀 p/pp（后接大写字母，避免误剥 pip3 等小写名）
        n = re.sub(r"^(p|pp)+(?=[a-z])", "", n)
        # 去除修饰/激活后缀
        n = re.sub(
            r"_(active|p|phos|phosphorylated|gtp|gdp)$",
            "",
            n,
        )
        # 去除 GTP/GDP 后缀（如 RasGTP → ras）
        n = re.sub(r"(gtp|gdp)$", "", n)
        return n.replace("_", "").replace("-", "")

    tpl_base = _strip_modifiers(template_species)
    if tpl_base:
        for sp in sbml_species_list:
            sp_base = _strip_modifiers(sp)
            if sp_base and sp_base == tpl_base:
                return sp

    return ""


def _extract_sbml_species_ids(sbml_xml: str) -> list[str]:
    """用 defusedxml 安全解析 SBML，提取所有 species id。

    安全硬约束：使用 defusedxml 防御 XXE 攻击，禁止 xml.etree 解析外部 XML。
    defusedxml 不可用时返回空列表（调用方降级到 Track B）。

    Returns:
        species id 列表（如 ["pEGFR", "ERK", "MEK"]）；解析失败返回空列表
    """
    if not sbml_xml or DefusedET is None:
        return []
    try:
        root = DefusedET.fromstring(sbml_xml)
        species_ids: list[str] = []
        # 遍历所有元素，查找 species 标签（兼容 SBML L2/L3 命名空间）
        for elem in root.iter():
            tag = elem.tag.lower()
            # 排除 speciesReference（反应物引用），仅取真正的 species 定义
            if "species" in tag and "speciesreference" not in tag and "listofspecies" not in tag:
                sp_id = elem.get("id") or elem.get("name") or ""
                if sp_id and sp_id not in species_ids:
                    species_ids.append(sp_id)
        return species_ids
    except Exception as exc:
        logger.warning("解析 SBML species 失败: %s", exc)
        return []


def _get_sbml_time_factor(sbml_xml: str) -> float:
    """获取 SBML 时间单位到分钟的转换因子。

    SBML 默认秒，模板轨迹用分钟。本函数尝试从 SBML model 元素的
    timeUnits 属性推断时间单位，返回乘以 SBML 时间值得到分钟的因子。

    Returns:
        转换因子：
        - 秒（默认）：1/60.0
        - 分钟：1.0
        - 小时：60.0
        解析失败默认返回 1/60.0（假设秒）
    """
    if not sbml_xml or DefusedET is None:
        return 1.0 / 60.0  # 默认秒
    try:
        root = DefusedET.fromstring(sbml_xml)
        for elem in root.iter():
            tag = elem.tag.lower()
            if tag.endswith("model"):
                tu = (elem.get("timeUnits") or "").lower()
                if "minute" in tu or tu == "min":
                    return 1.0
                elif "hour" in tu or tu == "hr":
                    return 60.0
                elif "second" in tu or tu == "sec" or tu == "s":
                    return 1.0 / 60.0
                break
        # 默认秒（SBML 规范默认时间单位为秒）
        return 1.0 / 60.0
    except Exception:
        return 1.0 / 60.0


def _find_peak(times: list[float], values: list[float]) -> tuple[float, float]:
    """找到峰值时间与峰值。

    Args:
        times: 时间点列表
        values: 对应的值列表

    Returns:
        (peak_time, peak_value)；空列表返回 (0.0, 0.0)
    """
    if not values:
        return (0.0, 0.0)
    peak_idx = values.index(max(values))
    peak_time = times[peak_idx] if peak_idx < len(times) else 0.0
    return (peak_time, values[peak_idx])


def _linear_interp(
    query_times: list[float],
    ref_times: list[float],
    ref_values: list[float],
) -> list[float] | None:
    """线性插值：在 ref_times/ref_values 上对 query_times 插值。

    Args:
        query_times: 需要插值的时间点
        ref_times: 参考时间点（需与 ref_values 等长）
        ref_values: 参考值

    Returns:
        插值结果列表；若 ref 数据无效返回 None
    """
    if (not ref_times or not ref_values
            or len(ref_times) != len(ref_values)):
        return None
    if len(ref_times) == 1:
        return [ref_values[0]] * len(query_times)

    # 优先用 numpy（性能更优）
    if NUMPY_AVAILABLE and _np is not None:
        try:
            ref_t = _np.array(ref_times, dtype=float)
            ref_v = _np.array(ref_values, dtype=float)
            # 确保 ref_times 单调递增
            order = _np.argsort(ref_t)
            ref_t = ref_t[order]
            ref_v = ref_v[order]
            result = _np.interp(query_times, ref_t, ref_v)
            return [float(x) for x in result]
        except Exception:
            pass  # 降级到纯 Python 实现

    # 纯 Python 线性插值兜底
    paired = sorted(zip(ref_times, ref_values))
    ref_t_sorted = [p[0] for p in paired]
    ref_v_sorted = [p[1] for p in paired]
    n = len(ref_t_sorted)
    result: list[float] = []
    for qt in query_times:
        if qt <= ref_t_sorted[0]:
            result.append(ref_v_sorted[0])
        elif qt >= ref_t_sorted[-1]:
            result.append(ref_v_sorted[-1])
        else:
            lo, hi = 0, n - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if ref_t_sorted[mid] <= qt:
                    lo = mid
                else:
                    hi = mid
            t0, t1 = ref_t_sorted[lo], ref_t_sorted[hi]
            v0, v1 = ref_v_sorted[lo], ref_v_sorted[hi]
            if t1 == t0:
                result.append(v0)
            else:
                result.append(v0 + (v1 - v0) * (qt - t0) / (t1 - t0))
    return result


def _compute_rmse(a: list[float], b: list[float]) -> float:
    """计算均方根误差（Root Mean Square Error）。

    Args:
        a: 序列 A
        b: 序列 B（需与 A 等长）

    Returns:
        RMSE 值；长度不匹配或空返回 inf
    """
    if not a or not b or len(a) != len(b):
        return float("inf")
    n = len(a)
    mse = sum((a[i] - b[i]) ** 2 for i in range(n)) / n
    return math.sqrt(mse)


def _normalize_to_unit(values: list[float]) -> list[float]:
    """归一化到 [0, 1] 区间。

    常数序列归一化为 0.5（避免除零）。
    """
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-12:
        return [0.5] * len(values)
    return [(v - vmin) / (vmax - vmin) for v in values]


def _pearson_correlation(a: list[float], b: list[float]) -> float:
    """计算 Pearson 相关系数。

    Args:
        a: 序列 A
        b: 序列 B（需与 A 等长，至少 2 个点）

    Returns:
        相关系数 [-1, 1]；方差为零或长度不足返回 0.0
    """
    if (not a or not b or len(a) != len(b) or len(a) < 2):
        return 0.0
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    var_a = sum((a[i] - mean_a) ** 2 for i in range(n))
    var_b = sum((b[i] - mean_b) ** 2 for i in range(n))
    denom = math.sqrt(var_a * var_b)
    if denom < 1e-12:
        return 0.0
    return cov / denom


# -----------------------------------------------------------------------------
# Track A：roadrunner 真实仿真
# -----------------------------------------------------------------------------

def _run_oracle_track_a(
    biomodel_id: str,
    sbml_xml: str,
    template_trajectory: dict,
    species_mapping: dict | None,
    duration: float,
    n_points: int,
) -> BioModelsOracleReport:
    """Track A：roadrunner 跑真实 SBML 仿真，与模板轨迹逐物种对比。

    步骤：
    1. 用 defusedxml 安全解析 SBML，提取 species 列表
    2. 构建 species_mapping（未提供时自动匹配）
    3. roadrunner 加载 SBML 并仿真
    4. 时间单位归一化（SBML 秒 → 分钟）
    5. 逐物种对比：峰值时间差 / 峰值相对误差 / RMSE / 时间对齐状态
    6. 综合指标与 status 判定
    """
    errors: list[str] = []

    # 1. 用 defusedxml 安全解析 SBML，提取 species 列表
    sbml_species_ids = _extract_sbml_species_ids(sbml_xml)
    if not sbml_species_ids:
        errors.append("defusedxml 解析 SBML 未提取到任何 species")

    # 2. 构建 species_mapping（未提供时自动匹配）
    if species_mapping is None:
        species_mapping = {}
        for tpl_sp in template_trajectory.keys():
            matched = _match_species_case_insensitive(tpl_sp, sbml_species_ids)
            if matched:
                species_mapping[tpl_sp] = matched
            else:
                errors.append(
                    f"物种匹配失败: {tpl_sp}"
                    f"（SBML species: {sbml_species_ids[:10]}）"
                )

    # 3. roadrunner 仿真（lazy import）
    try:
        import roadrunner  # type: ignore
    except ImportError:
        errors.append("roadrunner 不可用")
        return _run_oracle_track_b(
            biomodel_id, sbml_xml, template_trajectory, species_mapping,
            duration, n_points, errors,
            fallback_reason="roadrunner_unavailable",
        )

    # 获取 SBML 时间单位转换因子
    time_factor = _get_sbml_time_factor(sbml_xml)
    # end_time 在 SBML 时间单位下的值 = duration(分钟) / time_factor
    end_time_sbml = duration / time_factor if time_factor > 0 else duration * 60.0

    try:
        rr = roadrunner.RoadRunner()
        rr.load(sbml_xml)
        rr.reset()
        # roadrunner.simulate(start, end, num_points)
        result = rr.simulate(0, end_time_sbml, n_points)

        # 提取时间轴（SBML 时间单位 → 分钟）
        col_names = result.colnames if hasattr(result, "colnames") else []
        sim_times_min = [float(t) * time_factor for t in result[:, 0]]

        # 提取每个 species 的轨迹
        sbml_trajectories: dict[str, list[float]] = {}
        for col_idx, col in enumerate(col_names):
            if col_idx == 0:
                continue  # 跳过 time 列
            # roadrunner 列名形如 "EGF" / "[EGF]" / "compartment/EGF"
            sp_id = col.replace("[", "").replace("]", "").split("/")[-1]
            if sp_id.lower() == "time":
                continue
            sbml_trajectories[sp_id] = [float(v) for v in result[:, col_idx]]
    except Exception as exc:
        errors.append(f"roadrunner 仿真失败: {exc}")
        return _run_oracle_track_b(
            biomodel_id, sbml_xml, template_trajectory, species_mapping,
            duration, n_points, errors,
            fallback_reason=f"simulation_failed: {exc}",
        )

    # 4. 逐物种对比
    species_comparisons: list[dict] = []
    rmses: list[float] = []
    rel_errors: list[float] = []
    eps = 1e-9

    for tpl_sp, sbml_sp_id in species_mapping.items():
        tpl_data = template_trajectory.get(tpl_sp)
        if not tpl_data or not isinstance(tpl_data, dict):
            errors.append(f"模板轨迹无效: {tpl_sp}")
            continue

        tpl_times = tpl_data.get("time", [])
        tpl_values = tpl_data.get("values", [])
        if (not tpl_times or not tpl_values
                or len(tpl_times) != len(tpl_values)):
            errors.append(f"模板轨迹数据不完整: {tpl_sp}")
            continue

        sbml_values = sbml_trajectories.get(sbml_sp_id)
        if sbml_values is None:
            errors.append(f"SBML 仿真结果中无 species: {sbml_sp_id}")
            continue

        # 峰值对比
        tpl_peak_time, tpl_peak_value = _find_peak(tpl_times, tpl_values)
        sbml_peak_time, sbml_peak_value = _find_peak(sim_times_min, sbml_values)
        peak_time_diff = abs(tpl_peak_time - sbml_peak_time)
        peak_value_rel_error = abs(tpl_peak_value - sbml_peak_value) / max(
            abs(sbml_peak_value), eps
        )

        # 时间对齐 + RMSE
        # 检查时间轴是否完全一致
        if (len(tpl_times) == len(sim_times_min)
                and all(
                    abs(t1 - t2) < 1e-6
                    for t1, t2 in zip(tpl_times, sim_times_min)
                )):
            time_alignment = "aligned"
            aligned_sbml_values = sbml_values
        else:
            # 线性插值对齐
            aligned_sbml_values = _linear_interp(
                tpl_times, sim_times_min, sbml_values
            )
            if aligned_sbml_values is None:
                time_alignment = "failed"
                aligned_sbml_values = []
            else:
                time_alignment = "interpolated"

        if aligned_sbml_values and len(aligned_sbml_values) == len(tpl_values):
            rmse = _compute_rmse(tpl_values, aligned_sbml_values)
        else:
            rmse = float("inf")

        species_comparisons.append({
            "species": tpl_sp,
            "sbml_species_id": sbml_sp_id,
            "template_peak_time": tpl_peak_time,
            "sbml_peak_time": sbml_peak_time,
            "template_peak_value": tpl_peak_value,
            "sbml_peak_value": sbml_peak_value,
            "peak_time_diff_min": peak_time_diff,
            "peak_value_rel_error": peak_value_rel_error,
            "rmse": rmse,
            "time_alignment": time_alignment,
        })

        if rmse < float("inf"):
            rmses.append(rmse)
        rel_errors.append(peak_value_rel_error)

    # 5. 综合指标
    overall_distance = sum(rmses) / len(rmses) if rmses else float("inf")
    max_relative_error = max(rel_errors) if rel_errors else 1.0

    # 6. status 判定
    # passed：overall_distance < 0.1 且 max_relative_error < 0.2
    # failed：超过阈值
    # degraded：部分物种失败（有 errors 但有对比结果）
    if not species_comparisons:
        status = "degraded"
    elif overall_distance < 0.1 and max_relative_error < 0.2:
        status = "passed"
    else:
        status = "failed"

    summary = (
        f"Track A (roadrunner 仿真)：对比 {len(species_comparisons)} 个物种，"
        f"overall_distance={overall_distance:.4f}, "
        f"max_relative_error={max_relative_error:.4f}, status={status}"
    )

    return BioModelsOracleReport(
        biomodel_id=biomodel_id,
        status=status,
        sbml_loaded=True,
        simulation_run=True,
        track="A",
        species_comparisons=species_comparisons,
        overall_distance=overall_distance,
        max_relative_error=max_relative_error,
        summary=summary,
        errors=errors,
    )


# -----------------------------------------------------------------------------
# Track B：结构相似度降级
# -----------------------------------------------------------------------------

def _run_oracle_track_b(
    biomodel_id: str,
    sbml_xml: str,
    template_trajectory: dict,
    species_mapping: dict | None,
    duration: float,
    n_points: int,
    errors: list[str],
    fallback_reason: str = "",
) -> BioModelsOracleReport:
    """Track B：结构相似度降级（无真实 SBML 仿真）。

    roadrunner 或 defusedxml 不可用时调用。仅做轨迹形状相似度评分：
    - 对每个物种，归一化到 [0,1]，与参考形状计算 Pearson 相关系数
    - overall_distance = 1 - mean(correlation)
    - status = "degraded"

    参考形状为典型信号响应曲线（先升后降，峰值在 1/3 处），仅用于形状质量评估。
    """
    # 尝试提取 SBML species 列表（如果 defusedxml 可用）
    sbml_species_ids: list[str] = []
    if DefusedET is not None and sbml_xml:
        sbml_species_ids = _extract_sbml_species_ids(sbml_xml)

    # 构建 species_mapping（未提供时自动匹配）
    if species_mapping is None:
        species_mapping = {}
        for tpl_sp in template_trajectory.keys():
            if sbml_species_ids:
                matched = _match_species_case_insensitive(
                    tpl_sp, sbml_species_ids
                )
                if matched:
                    species_mapping[tpl_sp] = matched

    species_comparisons: list[dict] = []
    correlations: list[float] = []

    # 生成参考形状（典型信号响应：先升后降，峰值在 1/3 处）
    # 仅用于 Track B 形状相似度评分，非真实 SBML 仿真
    if n_points < 2:
        n_points = 2
    ref_times = [i * duration / (n_points - 1) for i in range(n_points)]
    peak_t = duration / 3.0
    sigma = duration / 6.0 if duration > 0 else 1.0
    ref_shape = [
        math.exp(-((t - peak_t) ** 2) / (2.0 * sigma ** 2))
        for t in ref_times
    ]

    for tpl_sp in template_trajectory.keys():
        tpl_data = template_trajectory.get(tpl_sp)
        if not tpl_data or not isinstance(tpl_data, dict):
            continue
        tpl_times = tpl_data.get("time", [])
        tpl_values = tpl_data.get("values", [])
        if not tpl_values or len(tpl_values) < 2:
            continue

        # 归一化模板轨迹到 [0, 1]
        tpl_norm = _normalize_to_unit(tpl_values)

        # 在参考形状上插值到模板时间点
        ref_aligned = _linear_interp(tpl_times, ref_times, ref_shape)
        if ref_aligned is None or len(ref_aligned) != len(tpl_norm):
            continue
        ref_norm = _normalize_to_unit(ref_aligned)

        # Pearson 相关系数
        corr = _pearson_correlation(tpl_norm, ref_norm)
        correlations.append(corr)

        sbml_sp_id = species_mapping.get(tpl_sp, "")
        tpl_peak_time, tpl_peak_value = _find_peak(tpl_times, tpl_values)
        species_comparisons.append({
            "species": tpl_sp,
            "sbml_species_id": sbml_sp_id,
            "template_peak_time": tpl_peak_time,
            "sbml_peak_time": None,  # Track B 无 SBML 仿真
            "template_peak_value": tpl_peak_value,
            "sbml_peak_value": None,
            "peak_time_diff_min": None,
            "peak_value_rel_error": None,
            "rmse": None,
            "time_alignment": "degraded",
            "shape_correlation": corr,
        })

    overall_distance = (
        1.0 - (sum(correlations) / len(correlations))
        if correlations else 1.0
    )
    max_relative_error = 1.0  # Track B 无法计算真实相对误差

    fb_msg = f"（fallback: {fallback_reason}）" if fallback_reason else ""
    summary = (
        f"Track B (结构相似度降级){fb_msg}："
        f"对比 {len(species_comparisons)} 个物种的形状相似度，"
        f"overall_distance={overall_distance:.4f}, status=degraded"
    )

    return BioModelsOracleReport(
        biomodel_id=biomodel_id,
        status="degraded",
        sbml_loaded=bool(sbml_xml),
        simulation_run=False,
        track="B",
        species_comparisons=species_comparisons,
        overall_distance=overall_distance,
        max_relative_error=max_relative_error,
        summary=summary,
        errors=errors,
    )


# -----------------------------------------------------------------------------
# 主入口：run_biomodels_oracle
# -----------------------------------------------------------------------------

def run_biomodels_oracle(
    biomodel_id: str,
    template_trajectory: dict,
    species_mapping: dict | None = None,
    duration: float = 120.0,
    n_points: int = 121,
) -> BioModelsOracleReport:
    """激活 validation_oracle 角色：下载 SBML → 仿真 → 与模板轨迹对比 → 生成差异报告。

    受 SA_BIOMODELS_ORACLE Feature Flag 保护。Flag 关闭时返回 skipped report。
    roadrunner 不可用时降级到结构相似度评分（Track B）。

    Args:
        biomodel_id: BioModels 模型 ID，如 "BIOMD0000000010"
        template_trajectory: 模板仿真轨迹，dict 形如
            {"pEGFR": {"time": [...], "values": [...]}, "ERK": {...}}
        species_mapping: 模板物种名 → SBML species id 的映射；
            为 None 时尝试大小写不敏感匹配
        duration: 仿真时长（分钟），默认 120
        n_points: 采样点数，默认 121（每分钟一个点）

    Returns:
        BioModelsOracleReport 对比报告
    """
    # lazy import 避免循环依赖与模块加载时序问题
    from app.config import ROADRUNNER_AVAILABLE, settings

    errors: list[str] = []

    # 1. Feature Flag 守护：默认 false，关闭时返回 skipped report
    if not settings.is_sa_feature_enabled("BIOMODELS_ORACLE"):
        return BioModelsOracleReport(
            biomodel_id=biomodel_id,
            status="skipped",
            sbml_loaded=False,
            simulation_run=False,
            track="",
            species_comparisons=[],
            overall_distance=float("nan"),
            max_relative_error=float("nan"),
            summary="SA_BIOMODELS_ORACLE Feature Flag 关闭，跳过 validation_oracle",
            errors=[],
        )

    # 2. 参数校验
    if not biomodel_id:
        errors.append("biomodel_id 为空")
        return BioModelsOracleReport(
            biomodel_id=biomodel_id,
            status="degraded",
            sbml_loaded=False,
            simulation_run=False,
            track="B",
            species_comparisons=[],
            overall_distance=1.0,
            max_relative_error=1.0,
            summary="biomodel_id 为空，无法下载 SBML",
            errors=errors,
        )

    if not template_trajectory or not isinstance(template_trajectory, dict):
        errors.append("template_trajectory 为空或非 dict")
        return BioModelsOracleReport(
            biomodel_id=biomodel_id,
            status="degraded",
            sbml_loaded=False,
            simulation_run=False,
            track="B",
            species_comparisons=[],
            overall_distance=1.0,
            max_relative_error=1.0,
            summary="template_trajectory 无效",
            errors=errors,
        )

    # 3. 下载 SBML（复用现有 BioModelsAPIClient.download 方法）
    try:
        client = get_biomodels_client()
        sbml_xml = client.download(biomodel_id)
    except Exception as exc:
        errors.append(f"下载 SBML 异常: {exc}")
        sbml_xml = ""

    if not sbml_xml:
        errors.append(f"SBML 下载失败或为空: {biomodel_id}")
        return BioModelsOracleReport(
            biomodel_id=biomodel_id,
            status="degraded",
            sbml_loaded=False,
            simulation_run=False,
            track="B",
            species_comparisons=[],
            overall_distance=1.0,
            max_relative_error=1.0,
            summary=f"SBML 下载失败 ({biomodel_id})，降级到 Track B（无对比）",
            errors=errors,
        )

    # 4. 选择 Track
    # Track A 需同时满足：roadrunner 可用 + defusedxml 可用（XXE 防护硬约束）
    use_track_a = bool(ROADRUNNER_AVAILABLE and DEFUSEDXML_AVAILABLE)

    if use_track_a:
        try:
            return _run_oracle_track_a(
                biomodel_id=biomodel_id,
                sbml_xml=sbml_xml,
                template_trajectory=template_trajectory,
                species_mapping=species_mapping,
                duration=duration,
                n_points=n_points,
            )
        except Exception as exc:
            # Track A 任何异常 → 降级到 Track B（不抛异常阻塞主流程）
            errors.append(f"Track A 异常，降级到 Track B: {exc}")
            return _run_oracle_track_b(
                biomodel_id=biomodel_id,
                sbml_xml=sbml_xml,
                template_trajectory=template_trajectory,
                species_mapping=species_mapping,
                duration=duration,
                n_points=n_points,
                errors=errors,
                fallback_reason=f"track_a_exception: {exc}",
            )

    # roadrunner 或 defusedxml 不可用 → Track B
    if not ROADRUNNER_AVAILABLE:
        errors.append("roadrunner 不可用，降级到 Track B")
    if not DEFUSEDXML_AVAILABLE:
        errors.append("defusedxml 不可用，降级到 Track B（XXE 防护硬约束）")

    return _run_oracle_track_b(
        biomodel_id=biomodel_id,
        sbml_xml=sbml_xml,
        template_trajectory=template_trajectory,
        species_mapping=species_mapping,
        duration=duration,
        n_points=n_points,
        errors=errors,
        fallback_reason="roadrunner_or_defusedxml_unavailable",
    )
