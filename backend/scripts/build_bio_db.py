# BioDynamics Agent - 从 KEGG/Reactome/UniProt/ChEMBL 批量下载并写入 ChromaDB
# 离线建库脚本：将在线数据库的结构化数据写入四路 collection（mechanism/parameter/evidence）。
#
# 用法：
#   python scripts/build_bio_db.py --databases kegg reactome uniprot chembl --max-entries 50 --recreate
#   python scripts/build_bio_db.py --databases kegg  # 仅建 KEGG
#   python scripts/build_bio_db.py --databases chembl --recreate  # 重建 ChEMBL collection

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# 将 backend 目录加入 Python 路径
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.bio_db_client import BioDBClient  # noqa: E402
from app.config import settings  # noqa: E402
from app.rag_collections import get_rag_collections  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def build_kegg(client: BioDBClient, rag_cols: Any, max_entries: int) -> int:
    """从 KEGG 下载通路数据，写入 mechanism collection。"""
    total = 0
    pathway_ids = client.KEGG_PATHWAY_IDS[:max_entries]
    logger.info("KEGG：准备下载 %d 个通路", len(pathway_ids))

    for pw_id in pathway_ids:
        try:
            records = await client.search_kegg_mechanism(pw_id)
            if records:
                n = rag_cols.upsert_mechanism(records)
                total += n
                logger.info("KEGG %s 写入 %d 条 mechanism", pw_id, n)
            time.sleep(0.5)  # 避免请求过快
        except Exception as exc:
            logger.warning("KEGG %s 处理失败：%s", pw_id, exc)

    return total


async def build_reactome(client: BioDBClient, rag_cols: Any, max_entries: int) -> int:
    """从 Reactome 下载通路数据，写入 mechanism collection。"""
    total = 0
    # Reactome 搜索关键词列表（肿瘤免疫相关）
    queries = [
        "TGF-beta signaling",
        "MAPK signaling",
        "T cell receptor signaling",
        "NF-kappa B signaling",
        "PI3K-Akt signaling",
        "apoptosis",
        "cell cycle",
        "immune system",
        "cancer",
    ]
    queries = queries[:max_entries]
    logger.info("Reactome：准备搜索 %d 个查询", len(queries))

    for query in queries:
        try:
            raw = await client.search_reactome(query)
            # 转换为 mechanism schema
            records: list[dict] = []
            for item in raw:
                records.append({
                    "pathway": item.get("name", ""),
                    "entities": item.get("participants", []),
                    "interactions": [],
                    "description": f"Reactome pathway: {item.get('name', '')} ({item.get('stId', '')})",
                    "source": f"Reactome:{item.get('stId', '')}",
                    "pmid": "",
                })
            if records:
                n = rag_cols.upsert_mechanism(records)
                total += n
                logger.info("Reactome '%s' 写入 %d 条 mechanism", query, n)
            time.sleep(0.5)
        except Exception as exc:
            logger.warning("Reactome '%s' 处理失败：%s", query, exc)

    return total


async def build_uniprot(client: BioDBClient, rag_cols: Any, max_entries: int) -> int:
    """从 UniProt 下载蛋白功能数据，写入 evidence collection。"""
    total = 0
    queries = [
        "TGF-beta receptor",
        "SMAD protein",
        "CD8 T cell",
        "PD-1",
        "CTLA-4",
        "MAPK cascade",
        "NF-kappa B",
        "PI3K kinase",
        "AKT kinase",
        "p53 tumor suppressor",
    ]
    queries = queries[:max_entries]
    logger.info("UniProt：准备搜索 %d 个查询", len(queries))

    for query in queries:
        try:
            raw = await client.search_uniprot(query, "Human")
            # 转换为 evidence schema
            records: list[dict] = []
            for item in raw:
                accession = item.get("accession", "")
                gene_names = item.get("gene_names", [])
                gene_str = ", ".join(gene_names) if gene_names else ""
                records.append({
                    "pmid": "",
                    "doi": f"UniProt:{accession}",
                    "title": f"{item.get('protein_name', '')} ({gene_str})",
                    "figure_ref": "",
                    "cell_line": "",
                    "species": item.get("organism", "Human"),
                    "year": "",
                    "journal": "UniProt",
                })
            if records:
                n = rag_cols.upsert_evidence(records)
                total += n
                logger.info("UniProt '%s' 写入 %d 条 evidence", query, n)
            time.sleep(0.5)
        except Exception as exc:
            logger.warning("UniProt '%s' 处理失败：%s", query, exc)

    return total


async def build_chembl(client: BioDBClient, rag_cols: Any, max_entries: int) -> int:
    """从 ChEMBL 下载药物活性数据，写入 parameter collection。"""
    total = 0
    queries = [
        "TGF-beta receptor",
        "PD-1",
        "CTLA-4",
        "VEGF receptor",
        "EGFR",
        "HER2",
        "BRAF",
        "ALK",
        "MEK",
        "CDK4",
    ]
    queries = queries[:max_entries]
    logger.info("ChEMBL：准备搜索 %d 个靶点", len(queries))

    for query in queries:
        try:
            records = await client.search_chembl_parameters(query)
            if records:
                n = rag_cols.upsert_parameter(records)
                total += n
                logger.info("ChEMBL '%s' 写入 %d 条 parameter", query, n)
            time.sleep(0.5)
        except Exception as exc:
            logger.warning("ChEMBL '%s' 处理失败：%s", query, exc)

    return total


async def main_async(args: argparse.Namespace) -> None:
    """异步主入口。"""
    logger.info("=" * 60)
    logger.info("生物医学数据库离线建库器启动")
    logger.info("目标数据库：%s", ", ".join(args.databases))
    logger.info("最大条目数：%d", args.max_entries)
    logger.info("=" * 60)

    rag_cols = get_rag_collections()
    if not rag_cols.available:
        logger.error("RagCollections 不可用，无法写入")
        return

    if args.recreate:
        logger.info("重建模式：删除四路 collection 后重建")
        for spec in rag_cols._collections:
            try:
                coll = rag_cols._collection(spec)
                if coll is not None:
                    rag_cols._client.delete_collection(name=coll.name)  # type: ignore[union-attr]
                    rag_cols._collections[spec] = None
                    logger.info("已删除 collection：%s", spec)
            except Exception as exc:
                logger.info("删除 %s 跳过：%s", spec, exc)

    client = BioDBClient()
    total_written = 0

    builders = {
        "kegg": build_kegg,
        "reactome": build_reactome,
        "uniprot": build_uniprot,
        "chembl": build_chembl,
    }

    for db_name in args.databases:
        builder = builders.get(db_name)
        if builder is None:
            logger.warning("未知数据库：%s，跳过", db_name)
            continue
        logger.info("--- 开始构建 %s ---", db_name.upper())
        try:
            count = await builder(client, rag_cols, args.max_entries)
            total_written += count
            logger.info("%s 完成，写入 %d 条", db_name.upper(), count)
        except Exception as exc:
            logger.error("%s 构建失败：%s", db_name.upper(), exc)

    logger.info("=" * 60)
    logger.info("建库完成，总计写入 %d 条", total_written)
    logger.info("四路 collection 状态：%s", rag_cols.stats())
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 KEGG/Reactome/UniProt/ChEMBL 批量下载并写入 ChromaDB"
    )
    parser.add_argument(
        "--databases",
        nargs="+",
        default=["kegg", "reactome", "uniprot", "chembl"],
        choices=["kegg", "reactome", "uniprot", "chembl"],
        help="要构建的数据库列表（默认全部）",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=50,
        help="每个数据库最大下载条目数（默认 50）",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="是否删除旧 collection 重新创建",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
