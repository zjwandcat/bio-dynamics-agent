"""本体验证测试 — Ontology Validation Suite

验证 BioDynamics v4 本体层的查询正确性：
  - HGNC      基因符号 → HGNC ID / UniProt ID
  - UniProt   蛋白 accession → 序列 / 注释
  - ChEBI     化学名 → ChEBI ID
  - GO        基因产物 → GO 注释
  - SBO       机制 ↔ SBO 术语双向映射
  - KEGG      通路 ID 解析
  - Reactome  通路 ID 解析

需要网络的用例标记 @pytest.mark.requires_network + skip（离线 collect 可枚举）。
离线可运行的用例（SBO 映射表、species_type 覆盖）不打网络标记。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# --------------------------------------------------------------------------- #
# HGNC 测试用例（基因符号 → HGNC ID + UniProt ID）
# 数据来源：https://www.genenames.org（权威映射）
# --------------------------------------------------------------------------- #
HGNC_TEST_CASES = [
    {"name": "EGFR", "expected_hgnc_id": "3236", "expected_uniprot": "P00533"},
    {"name": "TP53", "expected_hgnc_id": "11998", "expected_uniprot": "P04637"},
    {"name": "AKT1", "expected_hgnc_id": "391", "expected_uniprot": "P31749"},
    {"name": "MAPK1", "expected_hgnc_id": "6871", "expected_uniprot": "P28482"},
    {"name": "NFKB1", "expected_hgnc_id": "7794", "expected_uniprot": "P19838"},
    {"name": "STAT3", "expected_hgnc_id": "11364", "expected_uniprot": "P40763"},
    {"name": "CTNNB1", "expected_hgnc_id": "2514", "expected_uniprot": "P35222"},
    {"name": "SMAD2", "expected_hgnc_id": "6768", "expected_uniprot": "Q15796"},
    {"name": "MDM2", "expected_hgnc_id": "6973", "expected_uniprot": "Q00987"},
    {"name": "BAX", "expected_hgnc_id": "953", "expected_uniprot": "Q07812"},
]

# --------------------------------------------------------------------------- #
# ChEBI 测试用例（化学名 → ChEBI ID）
# --------------------------------------------------------------------------- #
CHEBI_TEST_CASES = [
    {"name": "EGF", "expected_chebi_id": "CHEBI:132945"},
    {"name": "ATP", "expected_chebi_id": "CHEBI:15422"},
    {"name": "GTP", "expected_chebi_id": "CHEBI:15996"},
    {"name": "doxorubicin", "expected_chebi_id": "CHEBI:28748"},
    {"name": "imatinib", "expected_chebi_id": "CHEBI:31690"},
]

# --------------------------------------------------------------------------- #
# GO 测试用例（基因产物 → GO 术语）
# --------------------------------------------------------------------------- #
GO_TEST_CASES = [
    {"name": "EGFR", "expected_go_terms": ["GO:0007179", "GO:0007165"]},
    {"name": "TP53", "expected_go_terms": ["GO:0006915", "GO:0005634"]},
    {"name": "AKT1", "expected_go_terms": ["GO:0004672", "GO:0005515"]},
]

# --------------------------------------------------------------------------- #
# KEGG 通路 ID 测试用例
# --------------------------------------------------------------------------- #
KEGG_TEST_CASES = [
    {"kegg_id": "hsa04010", "expected_name_contains": "MAPK"},
    {"kegg_id": "hsa04110", "expected_name_contains": "Cell cycle"},
    {"kegg_id": "hsa04151", "expected_name_contains": "PI3K"},
    {"kegg_id": "hsa04210", "expected_name_contains": "Apoptosis"},
    {"kegg_id": "hsa04310", "expected_name_contains": "Wnt"},
]

# --------------------------------------------------------------------------- #
# Reactome 通路 ID 测试用例
# --------------------------------------------------------------------------- #
REACTOME_TEST_CASES = [
    {"reactome_id": "R-HSA-177929", "expected_name_contains": "EGFR"},
    {"reactome_id": "R-HSA-110057", "expected_name_contains": "MAPK"},
    {"reactome_id": "R-HSA-109582", "expected_name_contains": "AKT"},
    {"reactome_id": "R-HSA-2559583", "expected_name_contains": "p53"},
    {"reactome_id": "R-HSA-109606", "expected_name_contains": "NF-kB"},
]

# --------------------------------------------------------------------------- #
# SBO 机制映射（离线可验证）
# 来源：backend/app/ontology/sbo_terms.py 的 SBO_TO_MECHANISM
# --------------------------------------------------------------------------- #
# 完整的 SBO → 机制映射（v4 应支持的全部机制）
EXPECTED_MECHANISM_TO_SBO = {
    "phosphorylation": "SBO:0000216",
    "dephosphorylation": "SBO:0000330",
    "binding": "SBO:0000177",
    "dissociation": "SBO:0000180",
    "ubiquitination": "SBO:0000220",
    "proteasomal_degradation": "SBO:0000179",
    "complex_formation": "SBO:0000526",
    "dimerization": "SBO:0000434",
    "cleavage": "SBO:0000210",
    "activation": "SBO:0000170",
    "inactivation": "SBO:0000170",
    "translocation": "SBO:0000186",
    "nuclear_import": "SBO:0000186",
    "nuclear_export": "SBO:0000186",
    "gtp_gdp_exchange": "SBO:0000332",
    "sequestration": "SBO:0000177",
    "feedback": "SBO:0000168",
    "transcription": "SBO:0000183",
    "translation": "SBO:0000184",
}

# v4 应支持的全部 species_type
EXPECTED_SPECIES_TYPES = {
    "gene", "protein", "chemical", "pathway", "unknown",
    "ligand", "receptor", "kinase", "drug", "complex", "rna",
}


# --------------------------------------------------------------------------- #
# HTTP 客户端桩
# --------------------------------------------------------------------------- #
def _http_get(url: str, timeout: float = 15.0) -> dict[str, Any]:
    """通用 HTTP GET，返回 JSON；缺失 requests / 网络不可达时抛 RuntimeError。"""
    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"requests 未安装：{exc}") from exc
    import os
    if not os.environ.get("BIODYNAMICS_ONTOLOGY_NETWORK"):
        raise RuntimeError("网络访问未启用（设置 BIODYNAMICS_ONTOLOGY_NETWORK=1 开启）")
    resp = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {url}")
    return resp.json()


# --------------------------------------------------------------------------- #
# HGNC 查询测试
# --------------------------------------------------------------------------- #
@pytest.mark.requires_network
@pytest.mark.parametrize("case", HGNC_TEST_CASES,
                         ids=lambda c: c["name"])
def test_hgnc_lookup(case: dict[str, str]) -> None:
    """HGNC 基因符号 → HGNC ID + UniProt ID 映射正确性。"""
    try:
        url = f"https://rest.genenames.org/fetch/symbol/{case['name']}"
        data = _http_get(url)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    docs = data.get("response", {}).get("docs", [])
    assert docs, f"HGNC 未找到 {case['name']}"
    record = docs[0]
    assert str(record.get("hgnc_id", "")).endswith(case["expected_hgnc_id"]), (
        f"{case['name']} HGNC ID 不匹配：{record.get('hgnc_id')}"
    )
    uniprot = record.get("uniprot_ids", [])
    if uniprot:
        assert case["expected_uniprot"] in uniprot, (
            f"{case['name']} UniProt 不匹配：{uniprot}"
        )


# --------------------------------------------------------------------------- #
# UniProt 查询测试
# --------------------------------------------------------------------------- #
@pytest.mark.requires_network
@pytest.mark.parametrize("case", HGNC_TEST_CASES,
                         ids=lambda c: c["expected_uniprot"])
def test_uniprot_lookup(case: dict[str, str]) -> None:
    """UniProt accession → 蛋白信息查询。"""
    try:
        url = (f"https://rest.uniprot.org/uniprotkb/{case['expected_uniprot']}"
               f".json")
        data = _http_get(url)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    assert data.get("primaryAccession") == case["expected_uniprot"], (
        f"UniProt accession 不匹配"
    )
    # 基因名应包含查询符号
    genes = data.get("genes", [])
    gene_names = [g.get("geneName", {}).get("value", "") for g in genes]
    assert any(case["name"] in gn for gn in gene_names), (
        f"UniProt {case['expected_uniprot']} 基因名 {gene_names} 不含 {case['name']}"
    )


# --------------------------------------------------------------------------- #
# ChEBI 查询测试
# --------------------------------------------------------------------------- #
@pytest.mark.requires_network
@pytest.mark.parametrize("case", CHEBI_TEST_CASES,
                         ids=lambda c: c["name"])
def test_chebi_lookup(case: dict[str, str]) -> None:
    """ChEBI 化学名 → ChEBI ID 映射。"""
    try:
        url = (f"https://www.ebi.ac.uk/webservices/rest/search?searchString="
               f"{case['name']}&searchCategory=ALL&maxResults=5")
        import requests  # type: ignore
        resp = requests.get(url, timeout=15.0)
        if resp.status_code != 200:
            pytest.skip(f"ChEBI HTTP {resp.status_code}")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    except ImportError as exc:
        pytest.skip(str(exc))

    text = resp.text
    # ChEBI 返回 XML，简化检查：期望 ID 出现在响应中
    chebi_id_num = case["expected_chebi_id"].split(":")[-1]
    assert chebi_id_num in text, (
        f"ChEBI 未找到 {case['name']} → {case['expected_chebi_id']}"
    )


# --------------------------------------------------------------------------- #
# GO 查询测试
# --------------------------------------------------------------------------- #
@pytest.mark.requires_network
@pytest.mark.parametrize("case", GO_TEST_CASES,
                         ids=lambda c: c["name"])
def test_go_lookup(case: dict[str, str]) -> None:
    """GO 术语注释查询（QuickGO）。"""
    try:
        url = (f"https://www.ebi.ac.uk/QuickGO/services/annotation/"
               f"search?geneProductId={case['name']}&limit=50")
        data = _http_get(url)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    go_ids = {ann.get("goId") for ann in data.get("results", [])}
    found = any(term in go_ids for term in case["expected_go_terms"])
    assert found, (
        f"{case['name']} GO 注释 {go_ids} 不含期望术语 {case['expected_go_terms']}"
    )


# --------------------------------------------------------------------------- #
# KEGG 通路查询测试
# --------------------------------------------------------------------------- #
@pytest.mark.requires_network
@pytest.mark.parametrize("case", KEGG_TEST_CASES,
                         ids=lambda c: c["kegg_id"])
def test_kegg_pathway_lookup(case: dict[str, str]) -> None:
    """KEGG 通路 ID → 通路名解析。"""
    try:
        import requests  # type: ignore
        url = f"http://rest.kegg.jp/get/{case['kegg_id']}"
        resp = requests.get(url, timeout=15.0)
        if resp.status_code != 200:
            pytest.skip(f"KEGG HTTP {resp.status_code}")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    except ImportError as exc:
        pytest.skip(str(exc))

    text = resp.text
    assert case["expected_name_contains"].lower() in text.lower(), (
        f"KEGG {case['kegg_id']} 响应不含 '{case['expected_name_contains']}'"
    )


# --------------------------------------------------------------------------- #
# Reactome 通路查询测试
# --------------------------------------------------------------------------- #
@pytest.mark.requires_network
@pytest.mark.parametrize("case", REACTOME_TEST_CASES,
                         ids=lambda c: c["reactome_id"])
def test_reactome_pathway_lookup(case: dict[str, str]) -> None:
    """Reactome 通路 ID → 通路名解析。"""
    try:
        url = (f"https://reactome.org/ContentService/data/query/{case['reactome_id']}"
               f"/displayName")
        import requests  # type: ignore
        resp = requests.get(url, timeout=15.0)
        if resp.status_code != 200:
            pytest.skip(f"Reactome HTTP {resp.status_code}")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    except ImportError as exc:
        pytest.skip(str(exc))

    name = resp.text.strip()
    assert case["expected_name_contains"].lower() in name.lower(), (
        f"Reactome {case['reactome_id']} 名称 '{name}' 不含 "
        f"'{case['expected_name_contains']}'"
    )


# --------------------------------------------------------------------------- #
# SBO 机制映射测试（离线）
# --------------------------------------------------------------------------- #
def test_sbo_mechanism_mapping_completeness() -> None:
    """校验 SBO ↔ 机制双向映射覆盖全部 v4 机制。

    当 backend 可导入时做实际检查；否则校验本地 EXPECTED 表的内部一致性。
    """
    try:
        from app.ontology.sbo_terms import SBO_TO_MECHANISM, MECHANISM_TO_SBO  # type: ignore
        sbo_to_mech = SBO_TO_MECHANISM
        mech_to_sbo = MECHANISM_TO_SBO
    except ImportError:
        # backend 不可导入：校验本地期望表的内部一致性
        sbo_to_mech = {v: k for k, v in EXPECTED_MECHANISM_TO_SBO.items()}
        mech_to_sbo = dict(EXPECTED_MECHANISM_TO_SBO)

    # 每个期望机制都应在映射中
    for mech in EXPECTED_MECHANISM_TO_SBO:
        assert mech in mech_to_sbo, f"机制 {mech} 不在 MECHANISM_TO_SBO 中"
    # 关键机制必须在 SBO_TO_MECHANISM 的值集合中
    critical = {"nuclear_import", "ubiquitination", "sequestration",
                "phosphorylation", "binding", "cleavage"}
    mech_values = set(sbo_to_mech.values())
    for mech in critical:
        assert mech in mech_values, f"关键机制 {mech} 在 SBO_TO_MECHANISM 中丢失"


def test_sbo_reverse_map_no_loss() -> None:
    """校验 SBO 反向映射不丢失机制（文档化 FM-016）。"""
    try:
        from app.ontology.sbo_terms import SBO_TO_MECHANISM  # type: ignore
    except ImportError:
        pytest.skip("backend 不可导入，跳过实际 SBO 检查")

    values = set(SBO_TO_MECHANISM.values())
    for mech in ["nuclear_import", "ubiquitination", "sequestration"]:
        assert mech in values, (
            f"FM-016 复现：SBO_TO_MECHANISM 丢失机制 {mech}"
        )


def test_sbo_term_format() -> None:
    """校验 SBO 术语格式合法（SBO:7位数字）。"""
    import re
    pattern = re.compile(r"^SBO:\d{7}$")
    for mech, sbo in EXPECTED_MECHANISM_TO_SBO.items():
        assert pattern.match(sbo), f"机制 {mech} 的 SBO 术语 {sbo} 格式非法"


# --------------------------------------------------------------------------- #
# species_type 覆盖测试（离线）
# --------------------------------------------------------------------------- #
def test_species_type_coverage() -> None:
    """校验 species_type 接受全部 v3 NER 类型。"""
    try:
        from app.ontology.ontology_agent import _SPECIES_TYPES  # type: ignore
    except ImportError:
        pytest.skip("backend 不可导入，跳过 _SPECIES_TYPES 检查")

    missing = EXPECTED_SPECIES_TYPES - set(_SPECIES_TYPES)
    assert not missing, f"缺失 species_type：{missing}"


def test_species_type_local_coverage() -> None:
    """离线校验本地 EXPECTED_SPECIES_TYPES 完整性。"""
    assert "gene" in EXPECTED_SPECIES_TYPES
    assert "protein" in EXPECTED_SPECIES_TYPES
    assert "chemical" in EXPECTED_SPECIES_TYPES
    assert "complex" in EXPECTED_SPECIES_TYPES
    assert len(EXPECTED_SPECIES_TYPES) >= 11


# --------------------------------------------------------------------------- #
# 别名 / 同义词解析测试
# --------------------------------------------------------------------------- #
ALIAS_TEST_CASES = [
    {"alias": "ERBB1", "canonical": "EGFR", "source": "HGNC"},
    {"alias": "HER1", "canonical": "EGFR", "source": "HGNC"},
    {"alias": "p53", "canonical": "TP53", "source": "HGNC"},
    {"alias": "AKT", "canonical": "AKT1", "source": "HGNC"},
    {"alias": "MAPK", "canonical": "MAPK1", "source": "HGNC"},
    {"alias": "ERK2", "canonical": "MAPK1", "source": "UniProt"},
    {"alias": "ERK1", "canonical": "MAPK3", "source": "UniProt"},
]


@pytest.mark.requires_network
@pytest.mark.parametrize("case", ALIAS_TEST_CASES,
                         ids=lambda c: c["alias"])
def test_alias_resolution(case: dict[str, str]) -> None:
    """别名 → 规范基因符号解析。"""
    try:
        url = f"https://rest.genenames.org/fetch/symbol/{case['alias']}"
        data = _http_get(url)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    docs = data.get("response", {}).get("docs", [])
    if docs:
        # 直接命中
        symbol = docs[0].get("symbol", "")
        assert symbol == case["alias"], f"别名 {case['alias']} 直接解析异常"
    else:
        # 别名查询：通过 prev_symbol / alias_symbol
        url2 = (f"https://rest.genenames.org/fetch/alias_symbol/{case['alias']}")
        try:
            data2 = _http_get(url2)
        except RuntimeError as exc:
            pytest.skip(str(exc))
        docs2 = data2.get("response", {}).get("docs", [])
        symbols = {d.get("symbol", "") for d in docs2}
        assert case["canonical"] in symbols, (
            f"别名 {case['alias']} 未解析到 {case['canonical']}，得到 {symbols}"
        )


# --------------------------------------------------------------------------- #
# 数据集完整性（离线）
# --------------------------------------------------------------------------- #
def test_hgnc_dataset_completeness() -> None:
    """校验 HGNC 测试数据集字段完整。"""
    for case in HGNC_TEST_CASES:
        assert "name" in case and "expected_hgnc_id" in case
        assert "expected_uniprot" in case
        assert case["expected_uniprot"].startswith("P") or case["expected_uniprot"].startswith("Q")


def test_chebi_dataset_format() -> None:
    """校验 ChEBI ID 格式。"""
    import re
    pattern = re.compile(r"^CHEBI:\d+$")
    for case in CHEBI_TEST_CASES:
        assert pattern.match(case["expected_chebi_id"]), (
            f"ChEBI ID 格式非法：{case['expected_chebi_id']}"
        )


def test_kegg_reactome_id_format() -> None:
    """校验 KEGG / Reactome ID 格式。"""
    import re
    kegg_pattern = re.compile(r"^hsa\d{5}$")
    for case in KEGG_TEST_CASES:
        assert kegg_pattern.match(case["kegg_id"]), f"KEGG ID 格式非法：{case['kegg_id']}"
    reactome_pattern = re.compile(r"^R-HSA-\d+$")
    for case in REACTOME_TEST_CASES:
        assert reactome_pattern.match(case["reactome_id"]), (
            f"Reactome ID 格式非法：{case['reactome_id']}"
        )


# --------------------------------------------------------------------------- #
# 文档化已知 v4 限制
# --------------------------------------------------------------------------- #
@pytest.mark.skip(reason="已知 v4 限制：EGF 双重身份 (FM-017)")
def test_egf_dual_identity_documentation() -> None:
    """文档化 P1 bug：EGF 同时存在于 _KNOWN_PROTEINS 和 _KNOWN_CHEMICALS。"""
    pass


@pytest.mark.skip(reason="已知 v4 限制：GO geneProductSymbol 参数无效 (FM-018)")
def test_go_client_invalid_param_documentation() -> None:
    """文档化 P1 bug：QuickGO 不支持 geneProductSymbol 参数。"""
    pass
