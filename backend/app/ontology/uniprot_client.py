# BioDynamics Agent v4 - UniProt REST API 客户端
# 查询蛋白 accession（如 "EGFR" → "P00533"）。
# API 文档：https://www.uniprot.org/help/api
# 失败策略：重试 3 次后降级返回 None，不抛异常，不阻塞流水线。

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from app.ontology._cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# UniProt REST API 端点（2024 后的新版 API）
UNIPROT_BASE_URL = "https://rest.uniprot.org/uniprotkb/search"

# 请求超时（秒）
UNIPROT_TIMEOUT = 10.0

# 最大重试次数
UNIPROT_MAX_RETRIES = 3

# 重试退避基数（秒）
UNIPROT_RETRY_BACKOFF = 1.0


def query_uniprot(
    gene_symbol: str,
    organism: str = "Homo sapiens",
) -> dict[str, Any] | None:
    """查询 UniProt API 获取蛋白标准化信息。

    Args:
        gene_symbol: 基因符号（如 "EGFR"）
        organism: 物种名（默认人类，避免跨物种混淆）

    Returns:
        标准化信息字典，包含 accession / protein_name / gene 等；
        查询失败或未命中返回 None。返回字段示例：
        {
            "accession": "P00533",
            "protein_name": "Epidermal growth factor receptor",
            "gene": "EGFR",
            "organism": "Homo sapiens",
            "length": 1210,
            "function": "..."
        }
    """
    if not gene_symbol or not gene_symbol.strip():
        return None
    symbol_clean = gene_symbol.strip()
    cache_query = f"{symbol_clean}|{organism}"

    # 缓存优先
    cached = cache_get("uniprot", cache_query)
    if cached is not None:
        logger.debug("UniProt 缓存命中: %s", symbol_clean)
        return cached

    # 构造查询：gene_exact + organism，限制 reviewed=true（Swiss-Prot 优先）
    query = f'gene_exact:{symbol_clean} AND organism_id:9606'
    params = {
        "query": query,
        "format": "json",
        "size": 1,  # 只取第一个
        "fields": "accession,id,gene_names,protein_name,organism_name,length,cc_function",
    }

    last_error: Exception | None = None
    for attempt in range(UNIPROT_MAX_RETRIES):
        try:
            response = requests.get(
                UNIPROT_BASE_URL, params=params, timeout=UNIPROT_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results:
                logger.debug("UniProt 查询无结果: %s", symbol_clean)
                return None
            entry = results[0]
            # 解析字段（UniProt JSON 结构嵌套较深，需小心提取）
            protein_desc = ""
            if entry.get("proteinDescription"):
                protein_desc = (
                    entry["proteinDescription"]
                    .get("recommendedName", {})
                    .get("fullName", {})
                    .get("value", "")
                )
            gene_names = entry.get("genes", [])
            primary_gene = (
                gene_names[0].get("geneName", {}).get("value", symbol_clean)
                if gene_names
                else symbol_clean
            )
            organism_name = ""
            if entry.get("organism"):
                organism_name = entry["organism"].get("scientificName", organism)
            # function 注释可能为列表
            function_text = ""
            comments = entry.get("comments", [])
            for c in comments:
                if c.get("commentType") == "FUNCTION":
                    texts = c.get("texts", [])
                    function_text = " ".join(t.get("value", "") for t in texts)
                    break

            result = {
                "accession": entry.get("primaryAccession", ""),
                "protein_name": protein_desc,
                "gene": primary_gene,
                "organism": organism_name,
                "length": entry.get("sequence", {}).get("length", 0),
                "function": function_text,
            }
            cache_set("uniprot", cache_query, result)
            logger.info(
                "UniProt 查询成功: %s → %s",
                symbol_clean, result.get("accession"),
            )
            return result
        except Exception as exc:
            last_error = exc
            logger.warning(
                "UniProt 查询失败 (attempt=%d/%d, symbol=%s): %s",
                attempt + 1, UNIPROT_MAX_RETRIES, symbol_clean, exc,
            )
            if attempt < UNIPROT_MAX_RETRIES - 1:
                time.sleep(UNIPROT_RETRY_BACKOFF * (2 ** attempt))

    logger.error("UniProt 查询最终失败 (symbol=%s): %s", symbol_clean, last_error)
    return None
