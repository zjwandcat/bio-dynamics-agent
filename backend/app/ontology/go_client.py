# BioDynamics Agent v4 - Gene Ontology (GO) REST API 客户端
# 查询基因/蛋白的功能术语（如 "EGFR" → GO:0007179 等）。
# API 文档：https://www.ebi.ac.uk/QuickGO/api/
# 失败策略：重试 3 次后降级返回空列表，不抛异常，不阻塞流水线。

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from app.ontology._cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# QuickGO REST API 端点：按 gene product 查询 GO annotations
GO_BASE_URL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"

# 请求超时（秒）
GO_TIMEOUT = 10.0

# 最大重试次数
GO_MAX_RETRIES = 3

# 重试退避基数（秒）
GO_RETRY_BACKOFF = 1.0


def query_go(gene_symbol: str) -> list[dict[str, Any]]:
    """查询 QuickGO API 获取基因的 GO 功能注释。

    Args:
        gene_symbol: 基因符号（如 "EGFR"）

    Returns:
        GO 注释列表，每项含 go_id / aspect / term_name；
        查询失败或未命中返回空列表。返回字段示例：
        [
            {
                "go_id": "GO:0007179",
                "aspect": "biological_process",
                "term_name": "ERBB2 signaling pathway",
                "evidence": "EXP"
            },
            ...
        ]
    """
    if not gene_symbol or not gene_symbol.strip():
        return []
    symbol_clean = gene_symbol.strip()

    # 缓存优先
    cached = cache_get("go", symbol_clean)
    if cached is not None:
        logger.debug("GO 缓存命中: %s", symbol_clean)
        return cached if isinstance(cached, list) else []

    # QuickGO 按 geneProductId 查询（symbol → UniProtKB:XXX）
    # 为避免依赖 UniProt，这里用 withGeneProductSymbol 过滤
    params = {
        "geneProductSymbol": symbol_clean,
        "taxonId": "9606",  # 人类
        "limit": 20,        # 限制结果数，避免超大响应
    }

    last_error: Exception | None = None
    for attempt in range(GO_MAX_RETRIES):
        try:
            response = requests.get(
                GO_BASE_URL, params=params,
                headers={"Accept": "application/json"},
                timeout=GO_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            annotations = data.get("results", [])
            # 简化输出结构，只保留关键字段
            simplified: list[dict[str, Any]] = []
            for ann in annotations:
                go_term = ann.get("goId", "")
                aspect = ann.get("goAspect", "")
                # aspect 形如 "biological_process" / "molecular_function" / "cellular_component"
                evidence = ann.get("qualifier", "")
                simplified.append({
                    "go_id": go_term,
                    "aspect": aspect,
                    "term_name": ann.get("goName", ""),
                    "evidence": evidence,
                })
            cache_set("go", symbol_clean, simplified)
            logger.info(
                "GO 查询成功: %s → %d 条注释",
                symbol_clean, len(simplified),
            )
            return simplified
        except Exception as exc:
            last_error = exc
            logger.warning(
                "GO 查询失败 (attempt=%d/%d, symbol=%s): %s",
                attempt + 1, GO_MAX_RETRIES, symbol_clean, exc,
            )
            if attempt < GO_MAX_RETRIES - 1:
                time.sleep(GO_RETRY_BACKOFF * (2 ** attempt))

    logger.error("GO 查询最终失败 (symbol=%s): %s", symbol_clean, last_error)
    return []
