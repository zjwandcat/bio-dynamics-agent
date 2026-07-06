# BioDynamics Agent v4 - ChEBI REST API 客户端
# 查询化学实体（配体/药物）的 ChEBI ID（如 "EGF" → "CHEBI:132945"）。
# API 文档：https://www.ebi.ac.uk/chebi/webServices.do
# 失败策略：重试 3 次后降级返回 None，不抛异常，不阻塞流水线。

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from app.ontology._cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# ChEBI REST API 端点（GET 搜索接口）
CHEBI_BASE_URL = "https://www.ebi.ac.uk/chebi/ws/rest/search"

# 请求超时（秒）
CHEBI_TIMEOUT = 10.0

# 最大重试次数
CHEBI_MAX_RETRIES = 3

# 重试退避基数（秒）
CHEBI_RETRY_BACKOFF = 1.0


def query_chebi(name: str) -> dict[str, Any] | None:
    """查询 ChEBI API 获取化学实体标准化信息。

    Args:
        name: 化学实体名称（如 "EGF" / "imatinib" / "doxorubicin"）

    Returns:
        标准化信息字典，包含 chebi_id / name / formula 等；
        查询失败或未命中返回 None。返回字段示例：
        {
            "chebi_id": "CHEBI:132945",
            "name": "epidermal growth factor",
            "formula": "C270H408N76O83S8",
            "mass": 6200.0,
            "definition": "..."
        }
    """
    if not name or not name.strip():
        return None
    name_clean = name.strip()

    # 缓存优先
    cached = cache_get("chebi", name_clean)
    if cached is not None:
        logger.debug("ChEBI 缓存命中: %s", name_clean)
        return cached

    params = {
        "searchQuery": name_clean,
        "maxResults": 1,
        "stars": "3",  # 仅查询三星级以上（人工审核）记录
    }

    last_error: Exception | None = None
    for attempt in range(CHEBI_MAX_RETRIES):
        try:
            response = requests.get(
                CHEBI_BASE_URL, params=params,
                headers={"Accept": "application/json"},
                timeout=CHEBI_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            # ChEBI search 返回 {searchResults: [{chebiId, ...}]}
            results = data.get("searchResults", [])
            if not results:
                logger.debug("ChEBI 查询无结果: %s", name_clean)
                return None
            entry = results[0]
            result = {
                "chebi_id": entry.get("chebiId", ""),
                "name": entry.get("chebiAsciiName", entry.get("name", name_clean)),
                "formula": entry.get("formulae", ""),
                "mass": entry.get("mass", 0.0),
                "definition": entry.get("definition", ""),
                "status": entry.get("status", ""),
            }
            cache_set("chebi", name_clean, result)
            logger.info(
                "ChEBI 查询成功: %s → %s",
                name_clean, result.get("chebi_id"),
            )
            return result
        except Exception as exc:
            last_error = exc
            logger.warning(
                "ChEBI 查询失败 (attempt=%d/%d, name=%s): %s",
                attempt + 1, CHEBI_MAX_RETRIES, name_clean, exc,
            )
            if attempt < CHEBI_MAX_RETRIES - 1:
                time.sleep(CHEBI_RETRY_BACKOFF * (2 ** attempt))

    logger.error("ChEBI 查询最终失败 (name=%s): %s", name_clean, last_error)
    return None
