# BioDynamics Agent - RAG 知识库自动扩充脚本
#
# 对应深度审核报告 §4.1 RAG 扩充脚本（自动化）：
#   接收领域关键词（如 "Tumor Immunity"），自动从 BioModels / PubMed 抓取
#   SBML 或 XML 文件至 data/raw/，解析后更新至 4 个 ChromaDB Collection。
#   去重：在入库前必须基于文档 hash 去重。
#
# 设计原则：
#   1. 数据来源唯一性 —— 仅从 BioModels / PubMed 抓取，不进行 LLM 幻觉补全。
#   2. 基于文档 hash 去重 —— sha256(role + ":" + search_text) 持久化到
#      data/raw/.extend_rag_dedup.json，避免重复入库。
#   3. Python 3.14 兼容 —— 使用 requests + xml.etree.ElementTree，不依赖 biopython。
#   4. 失败安全 —— 任一外部源失败时仅 warning 日志，不阻塞整体流程。
#
# 用法：
#   python scripts/extend_rag_db.py --keywords "Tumor Immunity" "EGFR signaling" \
#       --max-models 10 --max-articles 50

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from langchain_core.prompts import ChatPromptTemplate

# 将 backend 目录加入 Python 路径，以导入 app 包
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.biomodels_client import get_biomodels_client  # noqa: E402
from app.config import llm, settings  # noqa: E402
from app.nodes import RAGExtractionOutput  # noqa: E402
from app.prompts import RAG_EXTRACTION_PROMPT  # noqa: E402
from app.rag_collections import get_rag_collections  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 本地存储路径
RAW_DIR = BACKEND_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
DEDUP_FILE = RAW_DIR / ".extend_rag_dedup.json"

# NCBI E-utilities 端点
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# SBML 命名空间（L2/L3 兼容）
_SBML_NS_V2 = "http://www.sbml.org/sbml/level2"
_SBML_NS_V3 = "http://www.sbml.org/sbml/level3/version1"
# 动力学参数白名单：仅这些 param_name 可入库 parameter collection
_KINETIC_PARAM_NAMES = {
    "k1", "k2", "k_1", "k_2", "k_on", "k_off", "kon", "koff",
    "kcat", "k_cat", "Kd", "kd", "Ki", "ki", "Km", "km",
    "Vmax", "vmax", "V1", "V2", "V3",
    "k_deg", "kdegr", "k_prod", "k_syn", "k_sec", "k_act", "k_inact",
    "EC50", "IC50", "KEC50",
}


# =============================================================================
# 去重管理器
# =============================================================================
class DedupStore:
    """基于文档 hash 的持久化去重器。

    hash key = sha256(role + ":" + search_text)
    持久化到 data/raw/.extend_rag_dedup.json，跨脚本运行保持一致。
    """

    def __init__(self, path: Path = DEDUP_FILE) -> None:
        self._path = path
        self._hashes: set[str] = set()
        self._load()

    def _load(self) -> None:
        """从磁盘加载已有 hash 集合。"""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._hashes = set(data.get("hashes", []))
                logger.info("加载去重记录 %d 条", len(self._hashes))
            except Exception as exc:
                logger.warning("去重记录加载失败，重新开始：%s", exc)
                self._hashes = set()
        else:
            logger.info("未找到去重记录，首次运行")

    def filter_new(
        self,
        role: str,
        records: list[dict[str, Any]],
        text_extractor: Any,
    ) -> list[dict[str, Any]]:
        """过滤掉已入库的记录，返回新增记录列表。

        Args:
            role: collection 角色（mechanism/parameter/experiment/evidence）
            records: 待入库记录列表
            text_extractor: 从记录构建检索文本的函数

        Returns:
            去重后的新增记录列表。
        """
        new_records: list[dict[str, Any]] = []
        new_hashes: list[str] = []
        for rec in records:
            search_text = text_extractor(rec)
            if not search_text:
                continue
            h = hashlib.sha256(
                f"{role}:{search_text}".encode("utf-8")
            ).hexdigest()
            if h in self._hashes:
                continue
            new_records.append(rec)
            new_hashes.append(h)
        # 立即加入内存集合，避免本次运行内重复
        self._hashes.update(new_hashes)
        return new_records

    def persist(self) -> None:
        """持久化 hash 集合到磁盘。"""
        try:
            self._path.write_text(
                json.dumps({"hashes": sorted(self._hashes)}, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("去重记录持久化 %d 条 → %s", len(self._hashes), self._path)
        except Exception as exc:
            logger.warning("去重记录持久化失败：%s", exc)


# =============================================================================
# Phase 1: BioModels SBML 抓取与解析
# =============================================================================
def fetch_biomodels_sbml(
    keywords: list[str],
    max_models: int,
    raw_dir: Path,
) -> list[dict[str, str]]:
    """从 BioModels 搜索并下载 SBML 文件。

    Returns:
        [{model_id, name, sbml_text}, ...] 列表。
    """
    client = get_biomodels_client()
    results: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for kw in keywords:
        logger.info("[BioModels] 搜索关键词：%s", kw)
        try:
            candidates = client.search(kw, top_k=max_models)
        except Exception as exc:
            logger.warning("[BioModels] 搜索失败 (%s)：%s", kw, exc)
            continue

        for cand in candidates:
            model_id = cand.get("model_id", "")
            if not model_id or model_id in seen_ids:
                continue
            seen_ids.add(model_id)
            try:
                sbml_text = client.download(model_id)
            except Exception as exc:
                logger.warning("[BioModels] 下载失败 (%s)：%s", model_id, exc)
                continue
            if not sbml_text:
                continue
            # 落盘到 data/raw/{model_id}.xml
            xml_path = raw_dir / f"{model_id}.xml"
            try:
                xml_path.write_text(sbml_text, encoding="utf-8")
            except Exception as exc:
                logger.warning("[BioModels] 落盘失败 (%s)：%s", xml_path, exc)
            results.append({
                "model_id": model_id,
                "name": cand.get("name", ""),
                "sbml_text": sbml_text,
            })
            logger.info("[BioModels] 下载成功 %s (%d bytes)", model_id, len(sbml_text))
            # 礼貌延迟，避免被 BioModels 限流
            time.sleep(0.5)

    logger.info("[BioModels] 共获取 %d 个 SBML 模型", len(results))
    return results


def _sbml_namespaces(sbml_text: str) -> dict[str, str]:
    """提取 SBML 根元素的命名空间映射。"""
    try:
        root = ET.fromstring(sbml_text)
        ns = {"sbml": root.tag.split("}")[0][1:]} if "}" in root.tag else {}
        return ns
    except ET.ParseError as exc:
        logger.warning("SBML 解析失败：%s", exc)
        return {}


def parse_sbml_to_records(
    sbml_text: str,
    model_id: str,
    model_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """解析 SBML XML，提取机制知识与动力学参数。

    SBML 三角色定位（深度审核报告 §1.2）：
    本脚本入库的 SBML 参数仅作为 calibration_reference（校准参考），
    不作为 primary_ground_truth 或 validation_oracle。

    Returns:
        (mechanism_records, parameter_records)
    """
    ns = _sbml_namespaces(sbml_text)
    if not ns:
        return [], []

    try:
        root = ET.fromstring(sbml_text)
    except ET.ParseError as exc:
        logger.warning("SBML 二次解析失败 (%s)：%s", model_id, exc)
        return [], []

    # 定位 model 元素
    model_elem = root.find("sbml:model", ns) or root.find("model", ns) or root
    if model_elem is None:
        return [], []

    # 提取物种列表
    species_list: list[str] = []
    for sp in model_elem.findall(".//sbml:species", ns) or model_elem.findall(".//species", ns) or []:
        name = sp.get("name") or sp.get("id", "")
        if name:
            species_list.append(name)

    # 提取反应列表（含 reaction name + reactants → products）
    interactions: list[str] = []
    for rxn in model_elem.findall(".//sbml:reaction", ns) or model_elem.findall(".//reaction", ns) or []:
        rxn_name = rxn.get("name") or rxn.get("id", "")
        reactants = [r.get("species", "") for r in (rxn.findall(".//sbml:speciesReference", ns) or rxn.findall(".//speciesReference", ns) or [])]
        # 简单记录 reaction name 作为 interaction
        if rxn_name:
            interactions.append(rxn_name)

    # 构建机制记录
    mechanism_records: list[dict[str, Any]] = []
    if species_list or interactions:
        mechanism_records.append({
            "pathway": model_name or model_id,
            "entities": species_list[:50],  # 截断防止 metadata 过大
            "interactions": interactions[:50],
            "description": f"SBML model {model_id} 包含 {len(species_list)} 个物种和 {len(interactions)} 个反应",
            "source": f"BioModels:{model_id}",
            "pmid": "",
        })

    # 提取动力学参数（kineticLaw 中的 parameters）
    parameter_records: list[dict[str, Any]] = []
    for rxn in model_elem.findall(".//sbml:reaction", ns) or model_elem.findall(".//reaction", ns) or []:
        rxn_id = rxn.get("id", "")
        for kl in rxn.findall(".//sbml:kineticLaw", ns) or rxn.findall(".//kineticLaw", ns) or []:
            for param in kl.findall(".//sbml:parameter", ns) or kl.findall(".//parameter", ns) or []:
                pname = param.get("id") or param.get("name", "")
                pvalue = param.get("value", "")
                if not pname or pvalue == "":
                    continue
                # 仅保留动力学相关参数名
                if pname not in _KINETIC_PARAM_NAMES and pname.lower() not in {p.lower() for p in _KINETIC_PARAM_NAMES}:
                    continue
                try:
                    pvalue_float = float(pvalue)
                except (TypeError, ValueError):
                    continue
                parameter_records.append({
                    "param_name": pname,
                    "value": pvalue_float,
                    "unit": "1/s",  # SBML 默认单位不确定，标记为通用
                    "context": f"SBML reaction {rxn_id} in {model_id}",
                    "species": "Unknown",
                    "cell_line": "",
                    "source": f"BioModels:{model_id}",
                    "source_model": model_id,
                    "confidence": "MEDIUM",  # SBML 参数置信度中等（校准参考）
                })

    return mechanism_records, parameter_records


# =============================================================================
# Phase 2: PubMed 摘要抓取与参数提取
# =============================================================================
def _ncbi_get(params: dict[str, str]) -> requests.Response:
    """向 NCBI E-utilities 发送 GET 请求。"""
    endpoint = params.pop("endpoint", "esearch.fcgi")
    url = f"{NCBI_BASE_URL}/{endpoint}"
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
    if settings.NCBI_EMAIL:
        params["email"] = settings.NCBI_EMAIL
    if settings.NCBI_API_KEY:
        params["api_key"] = settings.NCBI_API_KEY
    try:
        response = _ncbi_get(params)
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as exc:
        logger.warning("[PubMed] esearch 失败 (%s)：%s", query[:50], exc)
        return []


def fetch_articles_with_metadata(
    pmids: list[str], batch_size: int = 50
) -> list[dict[str, Any]]:
    """分批获取 PubMed 文章，返回含元数据的字典列表。

    返回字段：pmid, title, abstract, journal, year。
    """
    results: list[dict[str, Any]] = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i: i + batch_size]
        params = {
            "endpoint": "efetch.fcgi",
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "abstract",
            "retmode": "xml",
        }
        if settings.NCBI_EMAIL:
            params["email"] = settings.NCBI_EMAIL
        if settings.NCBI_API_KEY:
            params["api_key"] = settings.NCBI_API_KEY
        try:
            response = _ncbi_get(params)
            root = ET.fromstring(response.text)
        except Exception as exc:
            logger.warning("[PubMed] efetch 失败 (batch %d)：%s", i, exc)
            continue

        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            pmid = str(pmid_elem.text) if pmid_elem is not None and pmid_elem.text else ""
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None and title_elem.text else ""
            abstract_parts: list[str] = []
            for at in article.findall(".//AbstractText"):
                if at.text:
                    abstract_parts.append(at.text)
            abstract = " ".join(abstract_parts)
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None and journal_elem.text else ""
            year_elem = article.find(".//PubDate/Year")
            year = year_elem.text if year_elem is not None and year_elem.text else ""
            if pmid and (abstract or title):
                results.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                })
        time.sleep(0.5)  # 避免触发 NCBI 频率限制

    return results


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
    return chunks


def extract_params_from_chunk(chunk: str) -> list[dict]:
    """使用 LLM 从文本片段中提取动力学参数。"""
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


def process_pubmed_articles(
    keywords: list[str],
    max_articles: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """从 PubMed 抓取文章并提取参数与证据。

    Returns:
        (parameter_records, evidence_records, experiment_records)
    """
    parameter_records: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    experiment_records: list[dict[str, Any]] = []
    seen_pmids: set[str] = set()

    for kw in keywords:
        logger.info("[PubMed] 检索关键词：%s", kw)
        pmids = fetch_pmids(kw, max_articles)
        logger.info("[PubMed] 获取 %d 篇 PMID", len(pmids))
        # 跨关键词 PMID 去重
        new_pmids = [p for p in pmids if p not in seen_pmids]
        seen_pmids.update(new_pmids)

        articles = fetch_articles_with_metadata(new_pmids)
        logger.info("[PubMed] 获取 %d 篇文章元数据", len(articles))

        for art in articles:
            pmid = art["pmid"]
            # 证据记录
            evidence_records.append({
                "pmid": pmid,
                "doi": "",
                "title": art["title"],
                "figure_ref": "",
                "cell_line": "",
                "species": "Human",
                "year": art["year"],
                "journal": art["journal"],
            })

            # 参数提取
            abstract = art["abstract"]
            if not abstract or len(abstract) < 50:
                continue
            chunks = chunk_text(abstract)
            for idx, chunk in enumerate(chunks):
                params = extract_params_from_chunk(chunk)
                for p in params:
                    p["source_pmid"] = pmid
                    p["source"] = f"PMID:{pmid}"
                    p["confidence"] = "MEDIUM"
                    p["chunk_index"] = idx
                    # 补齐 RagCollections 必填字段
                    if "context" not in p:
                        p["context"] = art["title"][:100]
                    if "species" not in p:
                        p["species"] = "Human"
                    parameter_records.append(p)

    logger.info(
        "[PubMed] 共提取 %d 条参数 / %d 条证据 / %d 条实验方案",
        len(parameter_records), len(evidence_records), len(experiment_records),
    )
    return parameter_records, evidence_records, experiment_records


# =============================================================================
# 主流程
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="BioDynamics RAG 知识库自动扩充：从 BioModels / PubMed 抓取并入库"
    )
    parser.add_argument(
        "--keywords", nargs="+", required=True,
        help="领域关键词列表，如 'Tumor Immunity' 'EGFR signaling'",
    )
    parser.add_argument(
        "--max-models", type=int, default=10,
        help="每个关键词从 BioModels 抓取的最大模型数（默认 10）",
    )
    parser.add_argument(
        "--max-articles", type=int, default=50,
        help="每个关键词从 PubMed 抓取的最大文章数（默认 50）",
    )
    parser.add_argument(
        "--skip-biomodels", action="store_true",
        help="跳过 BioModels SBML 抓取阶段",
    )
    parser.add_argument(
        "--skip-pubmed", action="store_true",
        help="跳过 PubMed 文献抓取阶段",
    )
    args = parser.parse_args()

    # 初始化 RAG Collections
    rag = get_rag_collections()
    if not rag.available:
        logger.error("ChromaDB 不可用，无法扩充 RAG 知识库")
        return

    # 初始化去重器
    dedup = DedupStore()

    total_mechanism = 0
    total_parameter = 0
    total_evidence = 0

    # --------------------------------------------------------------------------
    # Phase 1: BioModels SBML → mechanism + parameter collections
    # --------------------------------------------------------------------------
    if not args.skip_biomodels:
        logger.info("=" * 60)
        logger.info("Phase 1: BioModels SBML 抓取与解析")
        logger.info("=" * 60)
        sbml_models = fetch_biomodels_sbml(args.keywords, args.max_models, RAW_DIR)

        all_mechanism: list[dict[str, Any]] = []
        all_sbml_params: list[dict[str, Any]] = []
        for model in sbml_models:
            mech_recs, param_recs = parse_sbml_to_records(
                model["sbml_text"], model["model_id"], model["name"]
            )
            all_mechanism.extend(mech_recs)
            all_sbml_params.extend(param_recs)

        # 去重后入库 mechanism
        def _mechanism_text(rec: dict[str, Any]) -> str:
            return " ".join(
                str(x) for x in [
                    rec.get("pathway"), rec.get("description"),
                    " ".join(rec.get("entities", []) or []),
                    " ".join(rec.get("interactions", []) or []),
                    rec.get("source"),
                ] if x
            )

        new_mechanism = dedup.filter_new("mechanism", all_mechanism, _mechanism_text)
        if new_mechanism:
            inserted = rag.upsert_mechanism(new_mechanism)
            total_mechanism += inserted
            logger.info("[mechanism] 新增 %d / %d 条（去重 %d 条）",
                        inserted, len(all_mechanism), len(all_mechanism) - len(new_mechanism))
        else:
            logger.info("[mechanism] 全部重复，跳过 %d 条", len(all_mechanism))

        # 去重后入库 parameter（SBML 来源）
        def _parameter_text(rec: dict[str, Any]) -> str:
            return " ".join(
                str(x) for x in [
                    rec.get("param_name"), rec.get("value"),
                    rec.get("context"), rec.get("species"),
                    rec.get("source_model"), rec.get("source"),
                ] if x is not None
            )

        new_params = dedup.filter_new("parameter", all_sbml_params, _parameter_text)
        if new_params:
            inserted = rag.upsert_parameter(new_params)
            total_parameter += inserted
            logger.info("[parameter/SBML] 新增 %d / %d 条（去重 %d 条）",
                        inserted, len(all_sbml_params), len(all_sbml_params) - len(new_params))
        else:
            logger.info("[parameter/SBML] 全部重复，跳过 %d 条", len(all_sbml_params))

    # --------------------------------------------------------------------------
    # Phase 2: PubMed → parameter + evidence collections
    # --------------------------------------------------------------------------
    if not args.skip_pubmed:
        logger.info("=" * 60)
        logger.info("Phase 2: PubMed 文献抓取与参数提取")
        logger.info("=" * 60)
        pubmed_params, pubmed_evidence, _ = process_pubmed_articles(
            args.keywords, args.max_articles
        )

        # 去重后入库 parameter（PubMed 来源）
        def _parameter_text(rec: dict[str, Any]) -> str:
            return " ".join(
                str(x) for x in [
                    rec.get("param_name"), rec.get("value"),
                    rec.get("context"), rec.get("species"),
                    rec.get("source_pmid"), rec.get("source"),
                ] if x is not None
            )

        new_params = dedup.filter_new("parameter", pubmed_params, _parameter_text)
        if new_params:
            inserted = rag.upsert_parameter(new_params)
            total_parameter += inserted
            logger.info("[parameter/PubMed] 新增 %d / %d 条（去重 %d 条）",
                        inserted, len(pubmed_params), len(pubmed_params) - len(new_params))
        else:
            logger.info("[parameter/PubMed] 全部重复，跳过 %d 条", len(pubmed_params))

        # 去重后入库 evidence
        def _evidence_text(rec: dict[str, Any]) -> str:
            return " ".join(
                str(x) for x in [
                    rec.get("title"), rec.get("pmid"),
                    rec.get("journal"), rec.get("year"),
                    rec.get("species"),
                ] if x
            )

        new_evidence = dedup.filter_new("evidence", pubmed_evidence, _evidence_text)
        if new_evidence:
            inserted = rag.upsert_evidence(new_evidence)
            total_evidence += inserted
            logger.info("[evidence] 新增 %d / %d 条（去重 %d 条）",
                        inserted, len(pubmed_evidence), len(pubmed_evidence) - len(new_evidence))
        else:
            logger.info("[evidence] 全部重复，跳过 %d 条", len(pubmed_evidence))

    # 持久化去重记录
    dedup.persist()

    logger.info("=" * 60)
    logger.info("扩充完成：mechanism=%d, parameter=%d, evidence=%d",
                total_mechanism, total_parameter, total_evidence)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
