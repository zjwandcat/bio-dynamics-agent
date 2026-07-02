# BioDynamics Agent - 离线 RAG 参数库构建脚本
# 从 PubMed 抓取摘要，使用 LLM 提取动力学参数，并写入 ChromaDB 向量库。

import argparse
import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode

import requests
from langchain_core.prompts import ChatPromptTemplate

# 将 backend 目录加入 Python 路径，以导入 app 包
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import llm, settings  # noqa: E402
from app.nodes import RAGExtractionOutput  # noqa: E402
from app.prompts import RAG_EXTRACTION_PROMPT  # noqa: E402
from app.rag_client import RagClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _ncbi_get(params: dict[str, str]) -> requests.Response:
    """向 NCBI E-utilities 发送 GET 请求，并处理常见错误。"""
    url = f"{NCBI_BASE_URL}/{'/'.join(params.pop('endpoint', 'esearch.fcgi').split())}"
    query = urlencode(params)
    response = requests.get(f"{url}?{query}", timeout=60)
    response.raise_for_status()
    return response


def fetch_pmids(query: str, max_results: int) -> list[str]:
    """通过 E-utilities esearch 获取 PubMed PMID 列表。"""
    params = {
        "endpoint": "esearch.fcgi",
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
    }
    response = _ncbi_get(params)
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: list[str], batch_size: int = 50) -> list[tuple[str, str]]:
    """分批获取 PubMed 摘要，返回 (pmid, abstract_text) 列表。"""
    results: list[tuple[str, str]] = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        ids = ",".join(batch)
        params = {
            "endpoint": "efetch.fcgi",
            "db": "pubmed",
            "id": ids,
            "rettype": "abstract",
            "retmode": "xml",
        }
        response = _ncbi_get(params)
        root = ET.fromstring(response.text)

        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            pmid = str(pmid_elem.text) if pmid_elem is not None else ""
            abstract_parts: list[str] = []
            for abstract_text in article.findall(".//AbstractText"):
                if abstract_text.text:
                    abstract_parts.append(abstract_text.text)
            abstract_text = " ".join(abstract_parts)
            if pmid and abstract_text:
                results.append((pmid, abstract_text))

        # 避免触发 NCBI 频率限制
        time.sleep(0.5)

    return results


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """按固定字符窗口切分文本，保留重叠。"""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def extract_params_from_chunk(chunk: str) -> list[dict]:
    """使用 LLM 从文本片段中提取参数；失败时返回空列表。"""
    structured_llm = llm.with_structured_output(RAGExtractionOutput)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_EXTRACTION_PROMPT),
            ("human", "请提取以下文献片段中的动力学参数。"),
        ]
    )
    chain = prompt.partial(document_chunk=chunk) | structured_llm
    try:
        result: RAGExtractionOutput = chain.invoke({})
        return [p.model_dump() for p in result.params]
    except Exception as exc:
        logger.warning("参数提取失败：%s", exc)
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="从 PubMed 摘要构建 BioDynamics RAG 参数库")
    parser.add_argument("--queries", nargs="+", required=True, help="PubMed 检索关键词列表")
    parser.add_argument("--max-results", type=int, default=50, help="每个关键词最大返回文章数")
    parser.add_argument("--chunk-size", type=int, default=800, help="文本切分窗口大小")
    parser.add_argument("--recreate", action="store_true", help="是否重新创建 collection")
    args = parser.parse_args()

    if not settings.NCBI_EMAIL:
        logger.error("请在 .env 中配置 NCBI_EMAIL")
        return

    rag_client = RagClient()
    if not rag_client.available:
        logger.error("ChromaDB 连接失败，无法建库")
        return

    if args.recreate:
        if rag_client.delete_collection():
            logger.info("已删除旧 collection：%s", rag_client.collection_name)
        else:
            logger.warning("旧 collection 不存在或删除失败：%s", rag_client.collection_name)

    rag_client.ensure_collection()

    total_inserted = 0
    for query in args.queries:
        logger.info("正在检索关键词：%s", query)
        pmids = fetch_pmids(query, args.max_results)
        logger.info("获取到 %d 篇文献", len(pmids))

        abstracts = fetch_abstracts(pmids)
        for pmid, abstract in abstracts:
            chunks = chunk_text(abstract, args.chunk_size)
            for idx, chunk in enumerate(chunks):
                params = extract_params_from_chunk(chunk)
                for p in params:
                    p["source_pmid"] = pmid
                    p["chunk_index"] = idx
                if params:
                    inserted = rag_client.upsert_params(params)
                    total_inserted += inserted
                    logger.info("PMID %s chunk %d 写入 %d 条参数", pmid, idx, inserted)

    logger.info("建库完成，共写入 %d 条参数", total_inserted)


if __name__ == "__main__":
    main()
