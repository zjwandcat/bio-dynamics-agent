"""离线 PMID 向量补全脚本（缺口 2）。

读取 backend/benchmarks/case_manifest.json 获取所有 43 案例的
canonical_required_pmids（去重后约 80-120 个），调 PubMed E-utilities
efetch 批量下载文献元数据，使用项目已有的 embedding 模型生成向量，
写入 ChromaDB biomodels_evidence collection。

特性:
  - 断点续传：已存在的 PMID 自动跳过（用 collection.get 检查）
  - 离线模式：有本地 cache 目录 backend/data/pmid_cache/，优先读 cache
  - 失败容错：失败 PMID 记录到 failed_pmids.json，不中断批处理
  - 命令行参数：
      --dry-run   只打印待处理 PMID，不写入 ChromaDB
      --force     强制重建（忽略已存在的 PMID）
      --pathway   只处理某通路（如 EGFR / MAPK）
      --batch-size  efetch 批量大小（默认 50）

使用方式:
  cd backend
  python -m scripts.build_pmid_vector_db --dry-run
  python -m scripts.build_pmid_vector_db --pathway EGFR
  python -m scripts.build_pmid_vector_db --force

Python 3.14 兼容，仅依赖 requests + xml.etree（非 biopython）。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# 让 backend/ 成为 sys.path 第一个元素，便于 import app.*
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# NO_PROXY=* 绕过系统代理，与 main.py 启动时一致
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

import chromadb  # noqa: E402
import requests  # noqa: E402

try:  # Task 19 SEC-1.4: 优先 defusedxml
    from defusedxml import ElementTree as ET  # type: ignore
except ImportError:  # pragma: no cover
    import xml.etree.ElementTree as ET  # type: ignore

from app.config import embedding_model, settings  # noqa: E402

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 路径配置
# -----------------------------------------------------------------------------
CASE_MANIFEST_PATH = _BACKEND_DIR / "benchmarks" / "case_manifest.json"
CACHE_DIR = _BACKEND_DIR / "data" / "pmid_cache"
FAILED_PMIDS_PATH = CACHE_DIR / "failed_pmids.json"
PERSIST_DIR = Path(settings.CHROMA_PERSIST_DIR)
EVIDENCE_COLLECTION = settings.CHROMA_COLLECTION_EVIDENCE

# efetch 批量上限（NCBI 单次 id 参数最长 ~200 字符，按 50 个一批安全）
DEFAULT_BATCH_SIZE = 50
EFETCH_TIMEOUT_SEC = 30
RATE_LIMIT_SLEEP_SEC = 0.4  # 无 API key 时 3 req/s，留余量


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_case_manifest() -> dict[str, Any]:
    """加载 case_manifest.json。"""
    if not CASE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"case_manifest.json 未找到: {CASE_MANIFEST_PATH}"
        )
    with CASE_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_pmids_from_manifest(
    manifest: dict[str, Any],
    pathway_filter: str | None = None,
) -> dict[str, str]:
    """从 manifest 抽取 PMID -> pathway 映射。

    Args:
        manifest: case_manifest.json 加载后的 dict
        pathway_filter: 仅保留指定通路（大小写敏感）。None 表示全部。

    Returns:
        {pmid: pathway} 字典（已去重）
    """
    pmid_to_pathway: dict[str, str] = {}
    cases = manifest.get("cases", {}) or {}
    filter_lower = pathway_filter.lower() if pathway_filter else None
    for case_id, case in cases.items():
        pathway = str(case.get("pathway", "")).strip()
        if filter_lower and pathway.lower() != filter_lower:
            continue
        pmids = case.get("canonical_required_pmids", []) or []
        for pmid in pmids:
            pmid_str = str(pmid).strip()
            if pmid_str and pmid_str not in pmid_to_pathway:
                pmid_to_pathway[pmid_str] = pathway
    return pmid_to_pathway


# -----------------------------------------------------------------------------
# ChromaDB 操作
# -----------------------------------------------------------------------------
def get_evidence_collection() -> Any:
    """获取（或创建）biomodels_evidence collection。"""
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    client.heartbeat()
    return client.get_or_create_collection(
        name=EVIDENCE_COLLECTION,
        metadata={"hnsw:space": "cosine", "role": "evidence"},
    )


def list_existing_pmids(coll: Any) -> set[str]:
    """查询 collection 中已存在的所有 PMID（用于断点续传）。"""
    existing: set[str] = set()
    try:
        got = coll.get(include=["metadatas"])
    except Exception as exc:
        logger.warning("查询 collection.get 失败，假定无已有数据：%s", exc)
        return existing
    metas = got.get("metadatas", []) or []
    for meta in metas:
        if not meta:
            continue
        pmid = str(meta.get("pmid", "")).strip()
        if pmid:
            existing.add(pmid)
    return existing


def upsert_evidence_records(
    coll: Any,
    records: list[dict[str, Any]],
) -> int:
    """把解析后的 PubMed 文献记录写入 ChromaDB evidence collection。

    每条记录需含 pmid / title / abstract 等字段。
    """
    if not records:
        return 0

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for record in records:
        pmid = str(record.get("pmid", "")).strip()
        if not pmid:
            continue
        title = str(record.get("title", ""))
        abstract = str(record.get("abstract", ""))
        pathway = str(record.get("pathway", ""))
        pub_year = str(record.get("pub_year", ""))
        journal = str(record.get("journal", ""))
        authors = record.get("authors", []) or []
        mesh_terms = record.get("mesh_terms", []) or []
        authors_str = "; ".join(authors) if isinstance(authors, list) else str(authors)
        mesh_str = "; ".join(mesh_terms) if isinstance(mesh_terms, list) else str(mesh_terms)

        # 拼接 embedding 文本：title + abstract 优先，缺失则回退到 mesh + journal
        search_text = " ".join(
            p for p in [title, abstract] if p
        ).strip()
        if not search_text:
            search_text = " ".join(
                p for p in [mesh_str, journal, pmid] if p
            ).strip()
        if not search_text:
            logger.debug("跳过空文本 PMID=%s", pmid)
            continue

        try:
            vector = embedding_model.embed_query(search_text)
        except Exception as exc:
            logger.warning("Embedding 失败 (PMID=%s)：%s", pmid, exc)
            continue

        ids.append(f"pmid_{pmid}")
        embeddings.append(vector)
        documents.append(search_text)
        # ChromaDB metadata 仅支持 str/int/float/bool
        metadatas.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "source": f"PMID:{pmid}",
            "source_role": "PubMed",
            "pathway": pathway,
            "pub_year": pub_year,
            "journal": journal,
            "authors": authors_str[:500],  # 截断防止 metadata 过长
            "mesh_terms": mesh_str[:500],
        })

    if not ids:
        return 0

    try:
        coll.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)
    except Exception as exc:
        logger.error("写入 ChromaDB evidence collection 失败：%s", exc)
        return 0


# -----------------------------------------------------------------------------
# PubMed E-utilities 离线拉取（requests + xml.etree，非 biopython）
# -----------------------------------------------------------------------------
def _build_ncbi_params(extra: dict[str, Any]) -> dict[str, Any]:
    """附加 NCBI email / api_key / tool 参数。"""
    params = dict(extra)
    if settings.NCBI_EMAIL:
        params.setdefault("email", settings.NCBI_EMAIL)
        params.setdefault("tool", "BioDynamicsAgent")
    if settings.NCBI_API_KEY:
        params.setdefault("api_key", settings.NCBI_API_KEY)
    return params


def _cache_path_for_pmid(pmid: str) -> Path:
    """单个 PMID 的本地 cache 文件路径。"""
    return CACHE_DIR / f"pmid_{pmid}.xml"


def _load_from_cache(pmid: str) -> str | None:
    """从本地 cache 读取 PubMed XML 文本，未命中返回 None。"""
    cache_path = _cache_path_for_pmid(pmid)
    if not cache_path.exists():
        return None
    try:
        return cache_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("读取 cache 失败 (%s)：%s", cache_path, exc)
        return None


def _save_to_cache(pmid: str, xml_text: str) -> None:
    """把 PubMed XML 文本写入本地 cache。"""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _cache_path_for_pmid(pmid)
        cache_path.write_text(xml_text, encoding="utf-8")
    except Exception as exc:
        logger.warning("写入 cache 失败 (PMID=%s)：%s", pmid, exc)


def efetch_pmids_batch(pmids: list[str]) -> dict[str, str]:
    """批量调 PubMed efetch 拉取 XML，返回 {pmid: xml_text}。

    优先读本地 cache，缺失的 PMID 在线拉取并写回 cache。
    在线失败时该 PMID 不出现在返回 dict 中。
    """
    if not pmids:
        return {}

    result: dict[str, str] = {}
    online_needed: list[str] = []

    # 1. 先读 cache
    for pmid in pmids:
        cached = _load_from_cache(pmid)
        if cached:
            result[pmid] = cached
        else:
            online_needed.append(pmid)

    if not online_needed:
        return result

    # 2. 批量 efetch（NCBI 支持单次多 id）
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = _build_ncbi_params({
        "db": "pubmed",
        "id": ",".join(online_needed),
        "retmode": "xml",
    })

    try:
        resp = requests.get(efetch_url, params=params, timeout=EFETCH_TIMEOUT_SEC)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning(
            "PubMed efetch 批量拉取失败 (count=%d)：%s", len(online_needed), exc
        )
        return result

    # 3. 拆分为单 PMID 的 XML 片段并写 cache
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        logger.warning("PubMed efetch XML 解析失败：%s", exc)
        return result

    for article_elem in root.findall(".//PubmedArticle"):
        pmid_elem = article_elem.find(".//PMID")
        if pmid_elem is None or not pmid_elem.text:
            continue
        pmid = pmid_elem.text.strip()
        # 序列化为单篇文章 XML 字符串（包裹 <PubmedArticleSet>）
        single_xml = ET.tostring(article_elem, encoding="unicode")
        wrapped = f"<?xml version='1.0'?><PubmedArticleSet>{single_xml}</PubmedArticleSet>"
        result[pmid] = wrapped
        _save_to_cache(pmid, wrapped)

    # 限速
    time.sleep(RATE_LIMIT_SLEEP_SEC)
    return result


def parse_pubmed_xml(xml_text: str, pathway: str = "") -> dict[str, Any] | None:
    """解析单篇 PubMed XML，提取结构化字段。

    Returns:
        dict with pmid/title/abstract/authors/journal/pub_year/mesh_terms/pathway
        或 None（解析失败）
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("XML 解析失败：%s", exc)
        return None

    article_elem = root.find(".//PubmedArticle")
    if article_elem is None:
        return None

    pmid_elem = article_elem.find(".//PMID")
    pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else ""
    if not pmid:
        return None

    title_elem = article_elem.find(".//ArticleTitle")
    title = (
        "".join(title_elem.itertext()).strip()
        if title_elem is not None else ""
    )
    abstract_parts = article_elem.findall(".//AbstractText")
    abstract = " ".join(
        "".join(p.itertext()) for p in abstract_parts
    )[:2000]

    authors: list[str] = []
    for au in article_elem.findall(".//Author"):
        last = au.findtext("LastName") or ""
        fore = au.findtext("ForeName") or ""
        full = (fore + " " + last).strip() or (au.findtext("CollectiveName") or "")
        if full:
            authors.append(full)
    journal = article_elem.findtext(".//Journal/Title") or ""
    pub_year = (
        article_elem.findtext(".//PubDate/Year")
        or article_elem.findtext(".//PubDate/MedlineDate") or ""
    )[:4]
    mesh_terms: list[str] = []
    for mh in article_elem.findall(".//MeshHeading/DescriptorName"):
        if mh.text:
            mesh_terms.append(mh.text)

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "pub_year": pub_year,
        "mesh_terms": mesh_terms,
        "pathway": pathway,
    }


# -----------------------------------------------------------------------------
# 失败 PMID 记录
# -----------------------------------------------------------------------------
def load_failed_pmids() -> dict[str, str]:
    """加载 failed_pmids.json。"""
    if not FAILED_PMIDS_PATH.exists():
        return {}
    try:
        with FAILED_PMIDS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("读取 failed_pmids.json 失败：%s", exc)
        return {}


def save_failed_pmids(failed: dict[str, str]) -> None:
    """持久化 failed_pmids.json。"""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with FAILED_PMIDS_PATH.open("w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("写入 failed_pmids.json 失败：%s", exc)


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def process_pmids(
    pmid_to_pathway: dict[str, str],
    *,
    dry_run: bool = False,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """处理 PMID 列表，拉取并写入 ChromaDB evidence collection。

    Returns:
        统计 dict: {total, processed, written, cached, failed}
    """
    total = len(pmid_to_pathway)
    if total == 0:
        logger.info("没有待处理的 PMID")
        return {"total": 0, "processed": 0, "written": 0, "cached": 0, "failed": 0}

    logger.info(
        "待处理 PMID 共 %d 个 (force=%s, dry_run=%s, batch_size=%d)",
        total, force, dry_run, batch_size,
    )

    if dry_run:
        for pmid, pathway in pmid_to_pathway.items():
            print(f"  [DRY-RUN] PMID={pmid}  pathway={pathway}")
        return {
            "total": total,
            "processed": 0,
            "written": 0,
            "cached": 0,
            "failed": 0,
        }

    # 准备 collection
    try:
        coll = get_evidence_collection()
    except Exception as exc:
        logger.error("ChromaDB 不可用，终止：%s", exc)
        return {
            "total": total,
            "processed": 0,
            "written": 0,
            "cached": 0,
            "failed": total,
        }

    # 断点续传：查询已存在的 PMID
    existing_pmids: set[str] = set()
    if not force:
        existing_pmids = list_existing_pmids(coll)
        logger.info("向量库已有 %d 个 PMID，将跳过", len(existing_pmids))

    # 待处理列表
    pending = [
        (pmid, pathway)
        for pmid, pathway in pmid_to_pathway.items()
        if pmid not in existing_pmids
    ]
    if not pending:
        logger.info("所有 PMID 已在向量库中，无需处理")
        return {
            "total": total,
            "processed": total,
            "written": 0,
            "cached": len(existing_pmids),
            "failed": 0,
        }

    logger.info("实际待处理 %d 个 PMID（已跳过 %d 个已存在）",
                len(pending), len(existing_pmids))

    failed = load_failed_pmids()
    written = 0
    cached = 0
    processed = 0

    # 分批处理
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        batch_pmids = [p for p, _ in batch]
        logger.info(
            "处理批次 %d/%d (size=%d, pmids=%s...)",
            i // batch_size + 1,
            (len(pending) + batch_size - 1) // batch_size,
            len(batch),
            batch_pmids[:3],
        )

        try:
            xml_map = efetch_pmids_batch(batch_pmids)
        except Exception as exc:
            logger.error("批次 efetch 失败：%s", exc)
            for pmid, _ in batch:
                failed[pmid] = str(exc)
            continue

        # 解析 + 收集
        records_to_write: list[dict[str, Any]] = []
        for pmid, pathway in batch:
            xml_text = xml_map.get(pmid)
            if not xml_text:
                # 可能是 cache 命中失败的 edge case，或在线返回缺该 PMID
                if not _load_from_cache(pmid):
                    failed[pmid] = "efetch returned no XML for this PMID"
                continue
            parsed = parse_pubmed_xml(xml_text, pathway=pathway)
            if parsed is None:
                failed[pmid] = "XML parse failed"
                continue
            records_to_write.append(parsed)
            processed += 1

        # 写入 ChromaDB
        if records_to_write:
            n = upsert_evidence_records(coll, records_to_write)
            written += n
            cached += len(records_to_write) - n  # 解析成功但写入失败的数量

        # 每批保存 failed_pmids.json
        save_failed_pmids(failed)

    logger.info(
        "完成: total=%d processed=%d written=%d cached_skipped=%d failed=%d",
        total, processed, written, len(existing_pmids), len(failed),
    )
    return {
        "total": total,
        "processed": processed,
        "written": written,
        "cached": len(existing_pmids),
        "failed": len(failed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="离线 PMID 向量补全脚本：从 case_manifest 抽取 PMID，"
                    "拉取 PubMed 元数据并写入 ChromaDB evidence collection。"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印待处理 PMID，不写入 ChromaDB",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重建，忽略已存在的 PMID",
    )
    parser.add_argument(
        "--pathway", type=str, default=None,
        help="只处理指定通路（如 EGFR、MAPK）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"efetch 批量大小（默认 {DEFAULT_BATCH_SIZE}）",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="启用 DEBUG 级别日志",
    )
    args = parser.parse_args(argv)

    _setup_logging(verbose=args.verbose)

    logger.info("=== build_pmid_vector_db 开始 ===")
    logger.info("case_manifest: %s", CASE_MANIFEST_PATH)
    logger.info("cache_dir: %s", CACHE_DIR)
    logger.info("chroma_persist_dir: %s", PERSIST_DIR)
    logger.info("evidence_collection: %s", EVIDENCE_COLLECTION)

    # 1. 加载 manifest 并抽取 PMID
    try:
        manifest = load_case_manifest()
    except Exception as exc:
        logger.error("加载 case_manifest.json 失败：%s", exc)
        return 2

    pmid_to_pathway = collect_pmids_from_manifest(
        manifest, pathway_filter=args.pathway
    )
    total_cases = len(manifest.get("cases", {}) or {})
    logger.info(
        "manifest: cases=%d unique_pmids=%d (pathway_filter=%s)",
        total_cases, len(pmid_to_pathway), args.pathway or "ALL",
    )

    if not pmid_to_pathway:
        logger.warning("没有匹配的 PMID，结束")
        return 0

    # 2. 处理
    stats = process_pmids(
        pmid_to_pathway,
        dry_run=args.dry_run,
        force=args.force,
        batch_size=args.batch_size,
    )

    # 3. 输出汇总
    print("\n=== 汇总 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["failed"] > 0:
        print(f"\n失败 PMID 详情见: {FAILED_PMIDS_PATH}")
    logger.info("=== build_pmid_vector_db 完成 ===")
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
