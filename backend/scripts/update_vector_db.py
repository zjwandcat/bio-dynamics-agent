# BioDynamics Agent - 手动触发式知识库更新脚本
# 读取 backend/data/raw 下的原始文件（SBML/XML/JSON/TXT），切块、Embedding 后 upsert 到 ChromaDB。
# XML/SBML 使用标准库 xml.etree.ElementTree 提取文本，不依赖 biopython。

import argparse
import json
import logging
import os
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

# 必须在导入 app.config 之前设置 ChromaDB 路径
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 100


def _set_chroma_persist_dir(path: Path) -> None:
    """设置本地 ChromaDB 持久化目录环境变量。"""
    abs_path = str(path.resolve())
    os.environ["CHROMA_PERSIST_DIR"] = abs_path
    logger.info("使用本地 ChromaDB 持久化目录: %s", abs_path)


def extract_text_from_file(file_path: Path) -> str:
    """从各类原始文件中提取纯文本内容。"""
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            # 若 JSON 是参数记录列表，逐条序列化拼接
            if isinstance(data, list):
                pieces = []
                for record in data:
                    if isinstance(record, dict):
                        pieces.append(
                            " ".join(f"{k}: {v}" for k, v in record.items() if v is not None)
                        )
                    else:
                        pieces.append(str(record))
                return "\n".join(pieces)
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            logger.warning("JSON 解析失败 %s: %s", file_path.name, exc)
            return file_path.read_text(encoding="utf-8", errors="ignore")

    if suffix in (".xml", ".sbml"):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            # 递归提取所有文本节点
            texts = [text.strip() for text in root.itertext() if text.strip()]
            return "\n".join(texts)
        except Exception as exc:
            logger.warning("XML 解析失败 %s: %s", file_path.name, exc)
            return file_path.read_text(encoding="utf-8", errors="ignore")

    # 默认按文本文件读取
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.warning("文件读取失败 %s: %s", file_path.name, exc)
        return ""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """按固定字符窗口切分文本，保留重叠。"""
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start <= 0:
            start = end
    return chunks


def update_vector_db(
    raw_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    recreate: bool = False,
) -> dict:
    """执行知识库更新，返回统计信息。"""
    # 延迟导入，确保环境变量已生效
    from app.config import embedding_model
    from app.rag_client import RagClient

    rag_client = RagClient()
    if not rag_client.available:
        raise RuntimeError("ChromaDB 客户端初始化失败")

    if recreate:
        if rag_client.delete_collection():
            logger.info("已删除旧 collection: %s", rag_client.collection_name)
        else:
            logger.warning("旧 collection 不存在或删除失败")

    rag_client.ensure_collection()

    if not raw_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {raw_dir}")

    supported_suffixes = {".txt", ".md", ".json", ".xml", ".sbml"}
    files = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_suffixes]
    logger.info("发现 %d 个原始文件待处理", len(files))

    total_chunks = 0
    total_inserted = 0

    for file_path in files:
        logger.info("正在处理: %s", file_path.name)
        text = extract_text_from_file(file_path)
        if not text.strip():
            logger.warning("%s 未提取到有效文本，跳过", file_path.name)
            continue

        chunks = chunk_text(text, chunk_size, overlap)
        if not chunks:
            continue

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            try:
                vector = embedding_model.embed_query(chunk)
            except Exception as exc:
                logger.warning("Embedding 失败，跳过 chunk %d (%s): %s", idx, file_path.name, exc)
                continue

            ids.append(str(uuid.uuid4()))
            embeddings.append(vector)
            documents.append(chunk)
            metadatas.append(
                {
                    "source_file": file_path.name,
                    "chunk_index": idx,
                    "source_type": file_path.suffix.lower().lstrip("."),
                }
            )

        if not ids:
            continue

        collection = rag_client._get_collection()
        if collection is None:
            raise RuntimeError("无法获取 ChromaDB collection")

        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            total_chunks += len(ids)
            total_inserted += len(ids)
            logger.info("%s 已写入 %d 个 chunk", file_path.name, len(ids))
        except Exception as exc:
            logger.warning("写入 ChromaDB 失败 (%s): %s", file_path.name, exc)

    return {
        "files_processed": len(files),
        "chunks_inserted": total_inserted,
        "collection_name": rag_client.collection_name,
        "persist_dir": rag_client.persist_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="手动更新 BioDynamics Agent 知识库（ChromaDB）")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=BACKEND_DIR / "data" / "raw",
        help="原始数据目录（默认 data/raw）",
    )
    parser.add_argument(
        "--chroma-persist-dir",
        type=Path,
        default=BACKEND_DIR / "data" / "vector_db",
        help="ChromaDB 持久化目录（默认 data/vector_db）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="文本切块大小",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="切块重叠大小",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="是否删除旧 collection 重新创建",
    )
    args = parser.parse_args()

    _set_chroma_persist_dir(args.chroma_persist_dir)

    try:
        stats = update_vector_db(
            raw_dir=args.raw_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            recreate=args.recreate,
        )
        logger.info("知识库更新完成: %s", stats)
    except Exception as exc:
        logger.error("知识库更新失败: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
