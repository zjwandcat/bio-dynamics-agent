"""Rebuild ChromaDB biodynamics_params collection with RAG type system.

This script:
1. Deletes the old biodynamics_params collection
2. Loads all processed JSON files from data/processed/
3. Upserts records into the new collection via rag_client.upsert_params()

Each record now includes:
- type: kinetic_rate / binding_affinity / degradation_rate / initial_concentration
- reaction_equation: "EGF + EGFR → EGF-EGFR" (for kinetic_rate)
- species: reactant/product names (for kinetic_rate)
"""
import json
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.rag_client import RagClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def rebuild_params_collection():
    """重建 biodynamics_params collection，带 type system。"""
    rag_client = RagClient()
    if not rag_client.available:
        logger.error("ChromaDB 连接失败，无法重建")
        return

    # 1. 删除旧 collection
    logger.info("删除旧 collection: %s", rag_client.collection_name)
    rag_client.delete_collection()
    rag_client.ensure_collection()

    # 2. 加载所有 processed JSON
    processed_dir = BACKEND_DIR / "data" / "processed"
    if not processed_dir.exists():
        logger.error("processed 目录不存在: %s", processed_dir)
        return

    json_files = list(processed_dir.glob("*.json"))
    logger.info("发现 %d 个 processed JSON 文件", len(json_files))

    total_inserted = 0
    for json_path in json_files:
        if json_path.name == "all_params.json":
            continue  # 跳过合并文件
        try:
            records = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(records, list) or not records:
                logger.warning("%s 无有效记录，跳过", json_path.name)
                continue

            # 确保每条记录都有 type 字段
            for r in records:
                if "type" not in r:
                    # 兼容旧记录：根据 param_name 推断 type
                    pname = str(r.get("param_name", "")).lower()
                    if pname.startswith("initial_concentration"):
                        r["type"] = "initial_concentration"
                    elif pname.startswith(("k1", "k2", "k_", "k_on", "k_off", "kcat")):
                        r["type"] = "kinetic_rate"
                    elif pname in ("kd", "ki", "km") or pname.startswith(("ec50", "ic50")):
                        r["type"] = "binding_affinity"
                    else:
                        r["type"] = "other"

            inserted = rag_client.upsert_params(records)
            total_inserted += inserted
            logger.info("%s 写入 %d/%d 条记录", json_path.name, inserted, len(records))
        except Exception as exc:
            logger.error("处理 %s 失败: %s", json_path.name, exc)

    logger.info("重建完成，共写入 %d 条记录", total_inserted)

    # 3. 验证 type 分布
    collection = rag_client._get_collection()
    if collection is not None:
        all_data = collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", []) or []
        from collections import Counter
        type_counts = Counter(str(m.get("type", "unknown")) for m in metadatas if m)
        logger.info("Type 分布: %s", dict(type_counts))

        # 验证 kinetic_rate 记录是否含反应物信息
        kinetic_with_reaction = sum(
            1 for m in metadatas if m and m.get("type") == "kinetic_rate"
            and ("EGF" in str(m.get("context", "")) or "EGFR" in str(m.get("context", "")))
        )
        kinetic_total = sum(1 for m in metadatas if m and m.get("type") == "kinetic_rate")
        logger.info(
            "Kinetic rate 验证: %d/%d 含 EGF/EGFR 关键词",
            kinetic_with_reaction, kinetic_total,
        )


if __name__ == "__main__":
    rebuild_params_collection()
