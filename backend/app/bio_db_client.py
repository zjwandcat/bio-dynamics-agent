# BioDynamics Agent - 生物医学在线数据库统一客户端
# 封装 KEGG / Reactome / UniProt / ChEMBL 四个免费公开 REST API，
# 提供统一的异步检索接口，供 Node 1.5 在线补充与离线建库脚本使用。
#
# 设计要点：
# 1. 所有 API 均为免费公开，无需 API Key
# 2. 使用 asyncio + requests（通过 asyncio.to_thread）实现异步调用
# 3. 每个方法返回 list[dict]，dict 含 source_db 标识
# 4. 异常安全：所有调用失败返回空列表

from __future__ import annotations

import asyncio
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

# 请求超时（秒）
_DEFAULT_TIMEOUT = 30


# -----------------------------------------------------------------------------
# KEGG REST API
# https://www.kegg.jp/kegg/rest/keggapi.html
# -----------------------------------------------------------------------------
_KEGG_BASE = "https://rest.kegg.jp"


def _kegg_find(query: str, database: str = "pathway", max_results: int = 10) -> list[dict]:
    """KEGG find 操作：在指定数据库中搜索关键词，返回条目列表。"""
    url = f"{_KEGG_BASE}/find/{database}/{query}"
    try:
        resp = requests.get(url, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        results: list[dict] = []
        for line in lines[:max_results]:
            if "\t" not in line:
                continue
            entry_id, description = line.split("\t", 1)
            results.append({
                "entry_id": entry_id.strip(),
                "description": description.strip(),
                "source_db": "KEGG",
                "source_type": "mechanism",
            })
        return results
    except Exception as exc:
        logger.warning("KEGG find 失败（%s/%s）：%s", database, query, exc)
        return []


def _kegg_get(entry_id: str) -> dict[str, Any]:
    """KEGG get 操作：获取指定条目的详细信息。"""
    url = f"{_KEGG_BASE}/get/{entry_id}"
    try:
        resp = requests.get(url, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        text = resp.text.strip()
        # 解析 KEGG 平文格式
        entry: dict[str, Any] = {
            "entry_id": entry_id,
            "source_db": "KEGG",
            "source_type": "mechanism",
            "raw_text": text,
        }
        current_field = ""
        current_lines: list[str] = []
        for line in text.split("\n"):
            if line.startswith(" ") and current_field:
                current_lines.append(line.strip())
            elif line:
                if current_field:
                    entry[current_field] = "\n".join(current_lines)
                parts = line.split(None, 1)
                current_field = parts[0].lower() if parts else ""
                current_lines = [parts[1].strip()] if len(parts) > 1 else []
        if current_field:
            entry[current_field] = "\n".join(current_lines)
        return entry
    except Exception as exc:
        logger.warning("KEGG get 失败（%s）：%s", entry_id, exc)
        return {}


def _kegg_link(target_db: str, source_ids: list[str]) -> list[dict]:
    """KEGG link 操作：查找指定条目与目标数据库之间的关联。"""
    ids_str = "+".join(source_ids)
    url = f"{_KEGG_BASE}/link/{target_db}/{ids_str}"
    try:
        resp = requests.get(url, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        results: list[dict] = []
        for line in resp.text.strip().split("\n"):
            if "\t" not in line:
                continue
            source_id, target_id = line.split("\t", 1)
            results.append({
                "source_id": source_id.strip(),
                "linked_id": target_id.strip(),
                "source_db": "KEGG",
            })
        return results
    except Exception as exc:
        logger.warning("KEGG link 失败：%s", exc)
        return []


# -----------------------------------------------------------------------------
# Reactome ContentService API
# https://reactome.org/ContentService/
# -----------------------------------------------------------------------------
_REACTOME_BASE = "https://reactome.org/ContentService"


def _reactome_search(query: str, species: str = "Homo sapiens") -> list[dict]:
    """Reactome 搜索：返回与查询匹配的通路/事件列表。"""
    url = f"{_REACTOME_BASE}/search/query"
    params = {"query": query, "species": species, "types": "Pathway", "start": 0, "rows": 10}
    try:
        resp = requests.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results: list[dict] = []
        for entry in data.get("results", [])[:10]:
            results.append({
                "stId": entry.get("stId", ""),
                "name": entry.get("name", ""),
                "species": entry.get("speciesName", species),
                "source_db": "Reactome",
                "source_type": "mechanism",
            })
        return results
    except Exception as exc:
        logger.warning("Reactome search 失败（%s）：%s", query, exc)
        return []


def _reactome_get_pathway(st_id: str) -> dict[str, Any]:
    """Reactome 获取通路详情：事件层级与参与分子。"""
    url = f"{_REACTOME_BASE}/data/query/{st_id}"
    try:
        resp = requests.get(url, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # 提取参与分子（physicalEntity / referenceEntity）
        participants: list[str] = []
        for event in data.get("hasEvent", []):
            if isinstance(event, dict):
                participants.append(event.get("name", ""))
        return {
            "stId": st_id,
            "name": data.get("name", ""),
            "schemaClass": data.get("schemaClass", ""),
            "participants": participants,
            "source_db": "Reactome",
            "source_type": "mechanism",
            "raw_data": data,
        }
    except Exception as exc:
        logger.warning("Reactome get 失败（%s）：%s", st_id, exc)
        return {}


def _reactome_pathway_diagram(st_id: str) -> list[dict]:
    """Reactome 获取通路中的事件/参与者列表（简化版）。"""
    url = f"{_REACTOME_BASE}/data/pathways/{st_id}"
    try:
        resp = requests.get(url, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results: list[dict] = []
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict):
                results.append({
                    "stId": item.get("stId", ""),
                    "name": item.get("name", ""),
                    "source_db": "Reactome",
                    "source_type": "mechanism",
                })
        return results
    except Exception as exc:
        logger.warning("Reactome pathway diagram 失败（%s）：%s", st_id, exc)
        return []


# -----------------------------------------------------------------------------
# UniProt REST API
# https://www.uniprot.org/help/api_queries
# -----------------------------------------------------------------------------
_UNIPROT_BASE = "https://rest.uniprot.org"


def _uniprot_search(query: str, organism: str = "Human", size: int = 10) -> list[dict]:
    """UniProt 搜索：查询蛋白质功能注释、亚细胞定位、相互作用伙伴。"""
    # 构建 UniProt 查询字符串
    formatted_query = query
    if organism:
        formatted_query += f' AND (organism_name:"{organism}")'
    url = f"{_UNIPROT_BASE}/uniprotkb/search"
    params = {
        "query": formatted_query,
        "format": "json",
        "size": size,
        "fields": "accession,protein_name,gene_names,organism_name,cc_function,cc_interaction,cc_subcellular_location,xref_kegg,xref_reactome",
    }
    try:
        resp = requests.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results: list[dict] = []
        for entry in data.get("results", []):
            # 提取功能注释
            function_text = ""
            for cc in entry.get("comments", []):
                if cc.get("commentType") == "FUNCTION":
                    for text_val in cc.get("texts", []):
                        function_text += text_val.get("value", "") + " "
            # 提取相互作用伙伴
            interaction_partners: list[str] = []
            for cc in entry.get("comments", []):
                if cc.get("commentType") == "INTERACTION":
                    for inter in cc.get("interactions", []):
                        gene = inter.get("interactantOne", {}).get("geneName", "")
                        if gene:
                            interaction_partners.append(gene)
                        gene2 = inter.get("interactantTwo", {}).get("geneName", "")
                        if gene2:
                            interaction_partners.append(gene2)
            # 提取亚细胞定位
            locations: list[str] = []
            for cc in entry.get("comments", []):
                if cc.get("commentType") == "SUBCELLULAR LOCATION":
                    for loc in cc.get("subcellularLocations", []):
                        loc_name = loc.get("location", {}).get("value", "")
                        if loc_name:
                            locations.append(loc_name)
            # 提取 KEGG / Reactome 交叉引用
            kegg_ids: list[str] = []
            reactome_ids: list[str] = []
            for xref in entry.get("uniProtKBCrossReferences", []):
                db_name = xref.get("database", "")
                if db_name == "KEGG":
                    kegg_ids.append(xref.get("id", ""))
                elif db_name == "Reactome":
                    reactome_ids.append(xref.get("id", ""))

            results.append({
                "accession": entry.get("primaryAccession", ""),
                "protein_name": entry.get("proteinDescription", {}).get(
                    "recommendedName", {}
                ).get("fullName", {}).get("value", ""),
                "gene_names": [g.get("geneName", {}).get("value", "")
                               for g in entry.get("genes", [])],
                "organism": entry.get("organism", {}).get("scientificName", organism),
                "function": function_text.strip(),
                "interaction_partners": interaction_partners,
                "subcellular_locations": locations,
                "kegg_ids": kegg_ids,
                "reactome_ids": reactome_ids,
                "source_db": "UniProt",
                "source_type": "evidence",
            })
        return results
    except Exception as exc:
        logger.warning("UniProt search 失败（%s）：%s", query, exc)
        return []


# -----------------------------------------------------------------------------
# ChEMBL API
# https://www.ebi.ac.uk/chembl/api/data/
# -----------------------------------------------------------------------------
_CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def _chembl_search_target(query: str) -> list[dict]:
    """ChEMBL 搜索靶点：返回与查询匹配的蛋白靶点列表。"""
    url = f"{_CHEMBL_BASE}/target/search.json"
    params = {"q": query, "limit": 5}
    try:
        resp = requests.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results: list[dict] = []
        for target in data.get("targets", []):
            results.append({
                "chembl_id": target.get("target_chembl_id", ""),
                "pref_name": target.get("pref_name", ""),
                "target_type": target.get("target_type", ""),
                "organism": target.get("organism", ""),
                "source_db": "ChEMBL",
                "source_type": "parameter",
            })
        return results
    except Exception as exc:
        logger.warning("ChEMBL target search 失败（%s）：%s", query, exc)
        return []


def _chembl_get_activities(target_chembl_id: str, limit: int = 10) -> list[dict]:
    """ChEMBL 获取指定靶点的活性数据（IC50/Ki/Kd 等）。"""
    url = f"{_CHEMBL_BASE}/activity.json"
    params = {
        "target_chembl_id": target_chembl_id,
        "limit": limit,
        "order_by": "pchembl_value",
    }
    try:
        resp = requests.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results: list[dict] = []
        for act in data.get("activities", []):
            # 仅保留有数值的活性记录
            if not act.get("standard_value") or not act.get("standard_type"):
                continue
            results.append({
                "molecule_chembl_id": act.get("molecule_chembl_id", ""),
                "molecule_name": act.get("molecule_pref_name", ""),
                "target_chembl_id": target_chembl_id,
                "activity_type": act.get("standard_type", ""),
                "activity_value": float(act["standard_value"]) if act.get("standard_value") else None,
                "activity_unit": act.get("standard_units", ""),
                "pchembl_value": float(act["pchembl_value"]) if act.get("pchembl_value") else None,
                "assay_type": act.get("assay_type", ""),
                "source_db": "ChEMBL",
                "source_type": "parameter",
                "param_name": act.get("standard_type", "").lower().replace(" ", "_"),
                "value": float(act["standard_value"]) if act.get("standard_value") else None,
                "unit": act.get("standard_units", ""),
                "context": f"{act.get('molecule_pref_name', '')} vs {target_chembl_id}",
                "confidence": "HIGH" if act.get("pchembl_value") else "MEDIUM",
            })
        return results
    except Exception as exc:
        logger.warning("ChEMBL activities 失败（%s）：%s", target_chembl_id, exc)
        return []


# -----------------------------------------------------------------------------
# 统一客户端
# -----------------------------------------------------------------------------
class BioDBClient:
    """生物医学在线数据库统一客户端。

    封装 KEGG / Reactome / UniProt / ChEMBL 四个 API，
    提供单库检索与聚合并行检索接口。
    所有方法均为异步（通过 asyncio.to_thread 包装同步 requests 调用），
    异常安全——失败返回空列表。
    """

    # 已注册数据库元信息（供前端 /rag-status 使用）
    DB_REGISTRY: list[dict[str, str]] = [
        {"name": "KEGG", "type": "online_api", "base_url": _KEGG_BASE, "collection_role": "mechanism"},
        {"name": "Reactome", "type": "online_api", "base_url": _REACTOME_BASE, "collection_role": "mechanism"},
        {"name": "UniProt", "type": "online_api", "base_url": _UNIPROT_BASE, "collection_role": "evidence"},
        {"name": "ChEMBL", "type": "online_api", "base_url": _CHEMBL_BASE, "collection_role": "parameter"},
    ]

    # KEGG 常用人类通路 ID 映射（离线建库用）
    KEGG_PATHWAY_IDS: list[str] = [
        "hsa04350",  # TGF-beta signaling pathway
        "hsa04010",  # MAPK signaling pathway
        "hsa04650",  # Natural killer cell mediated cytotoxicity
        "hsa04660",  # T cell receptor signaling pathway
        "hsa04064",  # NF-kappa B signaling pathway
        "hsa04151",  # PI3K-Akt signaling pathway
        "hsa04066",  # HIF-1 signaling pathway
        "hsa04110",  # Cell cycle
        "hsa04210",  # Apoptosis
        "hsa05200",  # Pathways in cancer
        "hsa05230",  # Central carbon metabolism in cancer
        "hsa05165",  # Human papillomavirus infection
    ]

    async def search_kegg(self, query: str, pathway_id: str = "") -> list[dict]:
        """KEGG API: 查询通路、反应、化合物信息。

        若指定 pathway_id，直接获取该通路详情；
        否则先 find 再 get 最相关条目。
        """
        if pathway_id:
            entry = await asyncio.to_thread(_kegg_get, pathway_id)
            return [entry] if entry else []

        # 搜索通路
        found = await asyncio.to_thread(_kegg_find, query, "pathway", 5)
        if not found:
            return []

        # 获取最相关条目的详细信息
        results: list[dict] = []
        for item in found[:3]:
            entry_id = item.get("entry_id", "")
            if not entry_id:
                continue
            detail = await asyncio.to_thread(_kegg_get, entry_id)
            if detail:
                results.append(detail)
        return results

    async def search_reactome(self, query: str, species: str = "Homo sapiens") -> list[dict]:
        """Reactome API: 查询通路图、事件层级、参与分子。"""
        # 搜索匹配的通路
        pathways = await asyncio.to_thread(_reactome_search, query, species)
        if not pathways:
            return []

        # 获取前 3 个通路的详情
        results: list[dict] = []
        for pw in pathways[:3]:
            st_id = pw.get("stId", "")
            if not st_id:
                continue
            detail = await asyncio.to_thread(_reactome_get_pathway, st_id)
            if detail:
                results.append(detail)
        return results

    async def search_uniprot(self, query: str, organism: str = "Human") -> list[dict]:
        """UniProt REST API: 查询蛋白质功能、亚细胞定位、相互作用。"""
        return await asyncio.to_thread(_uniprot_search, query, organism, 10)

    async def search_chembl(self, query: str, target_chembl_id: str = "") -> list[dict]:
        """ChEMBL API: 查询药物活性数据（IC50/Ki/Kd）。

        若指定 target_chembl_id，直接获取活性数据；
        否则先搜索靶点再获取活性。
        """
        if target_chembl_id:
            return await asyncio.to_thread(_chembl_get_activities, target_chembl_id, 10)

        # 搜索靶点
        targets = await asyncio.to_thread(_chembl_search_target, query)
        if not targets:
            return []

        # 获取前 3 个靶点的活性数据
        results: list[dict] = []
        for target in targets[:3]:
            tid = target.get("chembl_id", "")
            if not tid:
                continue
            activities = await asyncio.to_thread(_chembl_get_activities, tid, 5)
            results.extend(activities)
        return results

    async def search_all(self, query: str, species: str = "Human") -> list[dict]:
        """并行查询所有在线数据库，合并去重后返回。

        用于 Node 1.5 自动在线补充：当本地 ChromaDB 命中不足时调用。
        """
        results = await asyncio.gather(
            self.search_kegg(query),
            self.search_reactome(query, species),
            self.search_uniprot(query, species),
            self.search_chembl(query),
            return_exceptions=True,
        )

        merged: list[dict] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("在线数据库查询异常：%s", result)
                continue
            if isinstance(result, list):
                merged.extend(result)

        return merged

    async def search_kegg_mechanism(self, query: str) -> list[dict]:
        """KEGG 机制专用：返回符合 mechanism collection schema 的记录。"""
        raw = await self.search_kegg(query)
        mechanism_records: list[dict] = []
        for entry in raw:
            entry_id = entry.get("entry_id", "")
            name = entry.get("name", entry_id)
            description = entry.get("description", entry.get("raw_text", ""))[:500]
            # 提取实体：从 gene 字段提取基因名（KEGG pathway 条目含 gene 而非 orthology）
            entities: list[str] = []
            gene_text = entry.get("gene", "")
            if gene_text:
                for line in gene_text.split("\n"):
                    # 格式：100532736  MICOS10-NBL1; MICOS10-NBL1 readthrough [KO:K19558]
                    parts = line.strip().split(";")
                    if parts:
                        gene_name = parts[0].strip()
                        # 去掉开头的数字 ID
                        tokens = gene_name.split()
                        if len(tokens) >= 2:
                            entities.append(tokens[1])
                        elif tokens:
                            entities.append(tokens[0])
            # 从 compound 字段提取化合物名
            compound_text = entry.get("compound", "")
            if compound_text:
                for line in compound_text.split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        entities.append(parts[1])
            # 从 drug 字段提取药物名
            drug_text = entry.get("drug", "")
            if drug_text:
                for line in drug_text.split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        entities.append(parts[1])
            # 提取交互关系：从 rel_pathway 字段
            interactions: list[str] = []
            rel_pathway = entry.get("rel_pathway", "")
            if rel_pathway:
                for line in rel_pathway.split("\n"):
                    interactions.append(line.strip())
            # 从 orthology 字段（非 pathway 条目可能存在）
            orthology = entry.get("orthology", "")
            if orthology:
                for line in orthology.split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        entities.append(parts[1] if len(parts) > 2 else parts[0])
            # 从 relation 字段
            relation = entry.get("relation", "")
            if relation:
                for line in relation.split("\n"):
                    interactions.append(line.strip())

            mechanism_records.append({
                "pathway": name if isinstance(name, str) else str(name),
                "entities": entities[:30],
                "interactions": interactions[:10],
                "description": description,
                "source": f"KEGG:{entry_id}",
                "pmid": "",
            })
        return mechanism_records

    async def search_chembl_parameters(self, query: str) -> list[dict]:
        """ChEMBL 参数专用：返回符合 parameter collection schema 的记录。"""
        raw = await self.search_chembl(query)
        param_records: list[dict] = []
        for act in raw:
            if act.get("value") is None:
                continue
            param_records.append({
                "param_name": act.get("param_name", act.get("activity_type", "ic50").lower()),
                "value": act["value"],
                "unit": act.get("unit", "nM"),
                "context": act.get("context", ""),
                "species": "Human",
                "cell_line": "",
                "source": f"ChEMBL:{act.get('molecule_chembl_id', '')}",
                "source_model": "",
                "confidence": act.get("confidence", "MEDIUM"),
            })
        return param_records


# -----------------------------------------------------------------------------
# 全局单例（懒加载）
# -----------------------------------------------------------------------------
_bio_db_client: BioDBClient | None = None


def get_bio_db_client() -> BioDBClient:
    """获取全局 BioDBClient 单例。"""
    global _bio_db_client
    if _bio_db_client is None:
        _bio_db_client = BioDBClient()
    return _bio_db_client
