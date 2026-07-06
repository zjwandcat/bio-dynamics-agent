# BioDynamics Agent v4 - 本体客户端共享缓存工具
# 提供 TTL 7 天的 JSON 文件缓存，避免重复查询 HGNC/UniProt/ChEBI/GO API。
# 缓存命中时不发起网络请求；缓存 miss 时调用方负责写回。

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 缓存目录：与 ontology 模块同级的 cache/ 子目录
CACHE_DIR: Path = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 缓存 TTL：7 天（秒）
CACHE_TTL_SECONDS: int = 7 * 24 * 3600


def _cache_key(provider: str, query: str) -> str:
    """生成缓存文件名：provider + query 的 md5 哈希，避免文件名包含非法字符。

    Args:
        provider: 数据源标识（如 "hgnc" / "uniprot"）
        query: 查询字符串（如 "EGFR"）

    Returns:
        缓存文件名（如 "hgnc_a1b2c3d4e5f6.json"）
    """
    digest = hashlib.md5(query.lower().encode("utf-8")).hexdigest()[:12]
    return f"{provider}_{digest}.json"


def cache_get(provider: str, query: str) -> dict[str, Any] | None:
    """读取缓存。

    Args:
        provider: 数据源标识
        query: 查询字符串

    Returns:
        缓存的 dict，缓存未命中或过期返回 None
    """
    path = CACHE_DIR / _cache_key(provider, query)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        # 检查 TTL
        if time.time() - payload.get("_ts", 0) > CACHE_TTL_SECONDS:
            return None
        return payload.get("data")
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("缓存读取失败 %s: %s", path.name, exc)
        return None


def cache_set(provider: str, query: str, data: dict[str, Any]) -> None:
    """写入缓存。

    Args:
        provider: 数据源标识
        query: 查询字符串
        data: 待缓存的 dict
    """
    path = CACHE_DIR / _cache_key(provider, query)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump({"_ts": time.time(), "data": data}, f, ensure_ascii=False)
    except OSError as exc:
        logger.debug("缓存写入失败 %s: %s", path.name, exc)


def cache_clear(provider: str | None = None) -> int:
    """清除缓存文件（主要用于测试）。

    Args:
        provider: 指定 provider 则只清该 provider 的缓存；None 清全部

    Returns:
        清除的文件数
    """
    count = 0
    for p in CACHE_DIR.glob("*.json"):
        if provider is None or p.name.startswith(f"{provider}_"):
            try:
                p.unlink()
                count += 1
            except OSError:
                pass
    return count
