# BioDynamics Agent - 四路 RAG 集合数据灌入脚本（v2 升级）
# 用途：把历史单 collection（CHROMA_COLLECTION_NAME）中的数据按字段归类
#       迁移到 v2 拆分的四路 collection（Mechanism / Parameter / Experiment / Evidence）。
#
# 分类规则（按 param_name 关键字 + source 字段启发式判断）：
#   - 命中 experiment_/protocol_/detection_ → experiment
#   - 命中 pmid + figure_ref + 无 kd/ec50/km/vmax → evidence
#   - 命中 pathway/cell_line/species + 描述性字段 + 无数值 → mechanism
#   - 其它（带 value + unit + param_name 是 kd/km/...）→ parameter
#   - 无法分类的全部默认入 parameter（v1 主力是参数）
#
# 用法：
#   python scripts/seed_collections.py --recreate     # 重建四路 collection
#   python scripts/seed_collections.py --dry-run      # 仅打印分类结果，不写入
#   python scripts/seed_collections.py --limit 200    # 限制最多迁移 200 条

import argparse
import logging
import sys
from pathlib import Path

# 将 backend 目录加入 Python 路径，以导入 app 包
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.rag_collections import (  # noqa: E402
    COLLECTION_REGISTRY,
    RagCollections,
    get_rag_collections,
)
from app.rag_client import RagClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 归类关键字
_EXPERIMENT_KEYWORDS = (
    "experiment", "protocol", "detection", "assay",
    "western", "flow", "elisa", "qpcr", "cytometry", "blot",
)
_EVIDENCE_KEYWORDS = (
    "figure", "fig.", "fig ", "p.", "panel",
)
_MECHANISM_KEYWORDS = (
    "pathway", "cascade", "signaling", "topology", "interaction", "regulation",
)
_PARAMETER_PARAMNAMES = (
    "kd", "km", "vmax", "ec50", "ic50", "half-life", "half_life",
    "degradation", "secretion", "production", "rate", "kcat", "k_on", "k_off",
)


def _classify(record: dict, idx: int) -> str:
    """将单条历史 record 归类到四路 RAG 之一。"""
    param_name = str(record.get("param_name", "")).lower()
    source = str(record.get("source", "")).lower()
    context = str(record.get("context", "")).lower()
    has_value = record.get("value") not in (None, "", "None")

    blob = " ".join([param_name, source, context])

    if any(kw in blob for kw in _EXPERIMENT_KEYWORDS):
        return "experiment"
    if any(kw in blob for kw in _EVIDENCE_KEYWORDS):
        return "evidence"
    if any(kw in blob for kw in _MECHANISM_KEYWORDS) and not has_value:
        return "mechanism"
    if any(pn in param_name for pn in _PARAMETER_PARAMNAMES) and has_value:
        return "parameter"

    # 默认：v1 历史数据绝大多数是 kinetics parameter
    if has_value and param_name:
        return "parameter"
    if idx % 4 == 0:
        return "mechanism"
    if idx % 4 == 1:
        return "experiment"
    if idx % 4 == 2:
        return "evidence"
    return "parameter"


def _to_role_record(role: str, record: dict) -> dict:
    """把通用 record 转换为指定 role 的 schema。"""
    if role == "parameter":
        return {
            "param_name": record.get("param_name", ""),
            "value": record.get("value"),
            "unit": record.get("unit", ""),
            "context": record.get("context", ""),
            "species": record.get("species", ""),
            "cell_line": record.get("cell_line", ""),
            "source": record.get("source", ""),
            "source_model": record.get("source_model", ""),
            "confidence": record.get("confidence", "MEDIUM"),
        }
    if role == "experiment":
        return {
            "name": record.get("param_name", "Unknown Protocol"),
            "target": record.get("context", ""),
            "detection_method": record.get("param_name", ""),
            "cell_line": record.get("cell_line", ""),
            "species": record.get("species", ""),
            "pmid": record.get("source", ""),
            "description": record.get("context", ""),
        }
    if role == "evidence":
        return {
            "pmid": record.get("source", ""),
            "doi": "",
            "title": record.get("context", ""),
            "figure_ref": record.get("param_name", ""),
            "cell_line": record.get("cell_line", ""),
            "species": record.get("species", ""),
            "year": "",
            "journal": "",
        }
    # mechanism
    return {
        "pathway": record.get("context", "Unknown Pathway"),
        "entities": [],
        "interactions": [record.get("param_name", "")] if record.get("param_name") else [],
        "description": record.get("context", ""),
        "source": record.get("source", ""),
        "pmid": record.get("source", ""),
    }


def fetch_legacy_records(rag_client: RagClient) -> list[dict]:
    """从历史 collection 拉取全部记录（metadatas + documents）。"""
    if not rag_client.available or rag_client.client is None:
        logger.error("历史 collection 不可用：%s", rag_client.collection_name)
        return []

    try:
        coll = rag_client.client.get_collection(name=rag_client.collection_name)
    except Exception as exc:
        logger.error("获取历史 collection 失败：%s", exc)
        return []

    try:
        all_data = coll.get(include=["metadatas", "documents"])
    except Exception as exc:
        logger.error("读取历史 collection 内容失败：%s", exc)
        return []

    metadatas = all_data.get("metadatas", []) or []
    documents = all_data.get("documents", []) or []

    records: list[dict] = []
    for idx, meta in enumerate(metadatas):
        if not meta:
            continue
        rec = dict(meta)
        if idx < len(documents):
            rec["_document"] = documents[idx]
        records.append(rec)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="灌入 v2 四路 RAG 集合")
    parser.add_argument("--recreate", action="store_true", help="重建四路 collection（先删除）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印分类结果，不写入")
    parser.add_argument("--limit", type=int, default=0, help="限制最多迁移条数（0 = 不限制）")
    parser.add_argument("--source", type=str, default=None, help="自定义历史 collection 名（默认用 CHROMA_COLLECTION_NAME）")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("四路 RAG 灌入器启动")
    logger.info("历史 collection：%s", args.source or settings.CHROMA_COLLECTION_NAME)
    logger.info("目标四路 collection：%s", ", ".join(s.name for s in COLLECTION_REGISTRY))
    logger.info("=" * 60)

    # 1. 拉取历史数据
    rag_client = RagClient()
    if args.source:
        rag_client.collection_name = args.source
    records = fetch_legacy_records(rag_client)
    if not records:
        logger.warning("历史 collection 为空或不可用，结束。")
        return

    if args.limit > 0:
        records = records[: args.limit]
    logger.info("共拉取 %d 条历史记录", len(records))

    # 2. 分类
    buckets: dict[str, list[dict]] = {spec.role: [] for spec in COLLECTION_REGISTRY}
    for idx, rec in enumerate(records):
        role = _classify(rec, idx)
        buckets[role].append(rec)

    logger.info(
        "分类结果：%s",
        {k: len(v) for k, v in buckets.items()},
    )

    if args.dry_run:
        logger.info("--dry-run：跳过写入阶段")
        return

    # 3. 灌入
    rag_cols = get_rag_collections()
    if not rag_cols.available:
        logger.error("RagCollections 不可用，放弃写入")
        return

    if args.recreate:
        # 删除四路 collection 重建
        for spec in COLLECTION_REGISTRY:
            try:
                rag_cols._client.delete_collection(name=spec.name)  # type: ignore[union-attr]
                rag_cols._collections[spec.role] = None
                logger.info("已删除 collection：%s (%s)", spec.name, spec.role)
            except Exception as exc:
                logger.info("删除 %s 跳过（不存在）：%s", spec.name, exc)

    total_written = 0
    for spec in COLLECTION_REGISTRY:
        role_records = buckets.get(spec.role, [])
        if not role_records:
            logger.info("[%s] 无记录，跳过", spec.role)
            continue
        converted = [_to_role_record(spec.role, r) for r in role_records]
        # 按 role 调用对应 upsert
        if spec.role == "mechanism":
            n = rag_cols.upsert_mechanism(converted)
        elif spec.role == "parameter":
            n = rag_cols.upsert_parameter(converted)
        elif spec.role == "experiment":
            n = rag_cols.upsert_experiment(converted)
        else:  # evidence
            n = rag_cols.upsert_evidence(converted)
        logger.info("[%s] 写入 %d / %d 条", spec.role, n, len(converted))
        total_written += n

    logger.info("=" * 60)
    logger.info("灌入完成，总计写入 %d 条", total_written)
    logger.info("最终四路 collection 状态：%s", rag_cols.stats())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
