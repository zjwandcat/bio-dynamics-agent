# BioDynamics Agent v4 - HGNC REST API 客户端
# 查询基因符号 → HGNC ID（如 "EGFR" → "HGNC:3236"）。
# API 文档：https://rest.genenames.org/
# 失败策略：重试 3 次后降级返回 None，不抛异常，不阻塞流水线。

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from app.ontology._cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# HGNC REST API 端点
HGNC_BASE_URL = "https://rest.genenames.org/fetch/symbol"

# 请求超时（秒）
HGNC_TIMEOUT = 10.0

# 最大重试次数（指数退避）
HGNC_MAX_RETRIES = 3

# 请求间隔（退避基数，秒）
HGNC_RETRY_BACKOFF = 1.0


def query_hgnc(symbol: str) -> dict[str, Any] | None:
    """查询 HGNC API 获取基因标准化信息。

    Args:
        symbol: 基因符号（如 "EGFR" / "TP53"）

    Returns:
        标准化信息字典，包含 hgnc_id / uniprot_id / gene_name 等；
        查询失败或未命中返回 None。返回字段示例：
        {
            "hgnc_id": "HGNC:3236",
            "symbol": "EGFR",
            "name": "epidermal growth factor receptor",
            "uniprot_id": "P00533",
            "entrez_id": "1956",
            "ensembl_gene_id": "ENSG00000146648"
        }
    """
    if not symbol or not symbol.strip():
        return None
    symbol_clean = symbol.strip()

    # 缓存优先
    cached = cache_get("hgnc", symbol_clean)
    if cached is not None:
        logger.debug("HGNC 缓存命中: %s", symbol_clean)
        return cached

    # 重试查询
    url = f"{HGNC_BASE_URL}/{symbol_clean}"
    headers = {"Accept": "application/json"}
    last_error: Exception | None = None
    for attempt in range(HGNC_MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=HGNC_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            records = data.get("response", {}).get("docs", [])
            if not records:
                logger.debug("HGNC 查询无结果: %s", symbol_clean)
                return None
            doc = records[0]
            # 提取关键字段，统一输出结构
            result = {
                "hgnc_id": doc.get("hgnc_id", ""),
                "symbol": doc.get("symbol", symbol_clean),
                "name": doc.get("name", ""),
                "uniprot_id": doc.get("uniprot_ids", [""])[0] if doc.get("uniprot_ids") else "",
                "entrez_id": str(doc.get("entrez_id", "")),
                "ensembl_gene_id": doc.get("ensembl_gene_id", ""),
                "gene_group": doc.get("gene_group", []),
            }
            cache_set("hgnc", symbol_clean, result)
            logger.info("HGNC 查询成功: %s → %s", symbol_clean, result.get("hgnc_id"))
            return result
        except Exception as exc:
            last_error = exc
            logger.warning(
                "HGNC 查询失败 (attempt=%d/%d, symbol=%s): %s",
                attempt + 1, HGNC_MAX_RETRIES, symbol_clean, exc,
            )
            if attempt < HGNC_MAX_RETRIES - 1:
                time.sleep(HGNC_RETRY_BACKOFF * (2 ** attempt))

    logger.error("HGNC 查询最终失败 (symbol=%s): %s", symbol_clean, last_error)
    return None
