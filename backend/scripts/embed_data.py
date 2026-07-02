# BioDynamics Agent - 将解析后的 SBML 参数 JSON 嵌入 ChromaDB 向量库
# 使用 LangChain/OpenAI 兼容的 Embedding 模型，支持本地 sentence-transformers 模型。

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# 将 backend 目录加入 Python 路径，以便复用 app 包中的配置与工具
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _set_local_embeddings() -> None:
    """默认启用本地 Embedding，避免云端 API 余额/模型不匹配问题。"""
    os.environ.setdefault("EMBEDDING_PROVIDER", "local")
    os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # 国内网络优先使用 HuggingFace 镜像站
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    logger.info("使用本地 Embedding 模型: %s", os.environ["EMBEDDING_MODEL"])


def main() -> None:
    parser = argparse.ArgumentParser(description="将结构化参数 JSON 嵌入 ChromaDB 向量库")
    parser.add_argument(
        "--input",
        type=Path,
        default=BACKEND_DIR / "data" / "processed" / "all_params.json",
        help="待嵌入的 JSON 文件路径",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=BACKEND_DIR / "data" / "vector_db",
        help="ChromaDB 持久化目录（默认 data/vector_db）",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="biodynamics_params",
        help="ChromaDB collection 名称",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="是否删除旧 collection 重新创建",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("输入文件不存在: %s", args.input)
        sys.exit(1)

    # 配置本地 Embedding 与 ChromaDB 持久化路径
    _set_local_embeddings()
    os.environ.setdefault("CHROMA_PERSIST_DIR", str(args.chroma_dir.resolve()))
    os.environ.setdefault("CHROMA_COLLECTION_NAME", args.collection)
    logger.info("使用 ChromaDB 持久化目录: %s", os.environ["CHROMA_PERSIST_DIR"])

    # 延迟导入，确保环境变量已生效
    from app.rag_client import RagClient

    rag_client = RagClient()
    if not rag_client.available:
        logger.error("ChromaDB 客户端初始化失败，请检查配置或依赖")
        sys.exit(1)

    if args.recreate:
        rag_client.delete_collection()
        logger.info("已删除旧 collection: %s", args.collection)

    rag_client.ensure_collection()

    with args.input.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        logger.error("输入 JSON 应为记录列表")
        sys.exit(1)

    logger.info("读取到 %d 条参数记录，开始嵌入...", len(records))
    inserted = rag_client.upsert_params(records)
    logger.info(
        "成功嵌入 %d/%d 条记录到 collection '%s'",
        inserted,
        len(records),
        args.collection,
    )


if __name__ == "__main__":
    main()
