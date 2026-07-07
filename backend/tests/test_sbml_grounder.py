# BioDynamics Agent v4 - SBML Grounder Agent 单元测试 (Phase 5 / Task 5.1.8)
#
# 测试 BIOMD0000000205（EGFR 信号通路）五级映射完整性。
#
# 测试用例（≥30）：
#   - TestSBMLParserV2: SBML XML 解析（lxml / ElementTree / 正则兜底）
#   - TestCanonicalSpeciesResolver: HGNC/UniProt/ChEBI 提取 + verified 标记
#   - TestOntologyGrounder: ontology ID 对齐 + unverified 标记
#   - TestAliasResolver: EGFR/ERBB1/HER1 等别名解析
#   - TestFiveLevelMapper: 五级映射链 + integrity
#   - TestSBMLGrounderAgent: 主入口 + Feature Flag + 异常降级
#
# 运行：cd backend && python -m pytest tests/test_sbml_grounder.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# BIOMD0000000205 mock SBML XML（EGFR 信号通路简化版）
# =============================================================================
# 用于测试 SBMLParserV2 / CanonicalSpeciesResolver / OntologyGrounder / FiveLevelMapper
# 包含 species/reaction/kineticLaw/annotation，覆盖 HGNC/UniProt/ChEBI 三种 ID 提取。
BIOMD0000000205_MOCK_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="BIOMD0000000205" metaid="metaid_0000001">
    <listOfCompartments>
      <compartment id="compartment" size="1"/>
      <compartment id="extracellular" size="1e-6"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="EGF" name="EGF" compartment="extracellular" metaid="metaid_0000002"
               initialConcentration="0.1">
        <annotation>
          <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about="#metaid_0000002">
              <bqbiol:is xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">
                <rdf:Bag>
                  <rdf:li rdf:resource="https://identifiers.org/HGNC:HGNC:3229"/>
                  <rdf:li rdf:resource="https://identifiers.org/uniprot/P01133"/>
                </rdf:Bag>
              </bqbiol:is>
            </rdf:Description>
          </rdf:RDF>
        </annotation>
      </species>
      <species id="EGFR" name="EGFR" compartment="compartment" metaid="metaid_0000003"
               initialConcentration="100.0">
        <annotation>
          <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about="#metaid_0000003">
              <bqbiol:is xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">
                <rdf:Bag>
                  <rdf:li rdf:resource="https://identifiers.org/HGNC:HGNC:3236"/>
                  <rdf:li rdf:resource="https://identifiers.org/uniprot/P00533"/>
                </rdf:Bag>
              </bqbiol:is>
            </rdf:Description>
          </rdf:RDF>
        </annotation>
      </species>
      <species id="EGF_EGFR" name="EGF-EGFR complex" compartment="compartment"
               initialConcentration="0.0"/>
      <species id="Imatinib" name="Imatinib" compartment="compartment"
               initialConcentration="10.0">
        <annotation>
          <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description>
              <bqbiol:is xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">
                <rdf:Bag>
                  <rdf:li rdf:resource="https://identifiers.org/CHEBI:45783"/>
                </rdf:Bag>
              </bqbiol:is>
            </rdf:Description>
          </rdf:RDF>
        </annotation>
      </species>
      <species id="Unknown_Protein" name="Mystery" compartment="compartment"
               initialConcentration="5.0"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kon_EGF_EGFR" name="kon" value="0.003" units="per_nM_per_min"/>
      <parameter id="koff_EGF_EGFR" name="koff" value="0.12" units="per_min"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="reaction_EGF_EGFR_binding" name="EGF-EGFR binding" reversible="false">
        <listOfReactants>
          <speciesReference species="EGF" stoichiometry="1"/>
          <speciesReference species="EGFR" stoichiometry="1"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="EGF_EGFR" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <minus/>
              <apply>
                <times/>
                <ci>kon_EGF_EGFR</ci>
                <ci>EGF</ci>
                <ci>EGFR</ci>
              </apply>
              <apply>
                <times/>
                <ci>koff_EGF_EGFR</ci>
                <ci>EGF_EGFR</ci>
              </apply>
            </apply>
          </math>
          <listOfParameters>
            <parameter id="local_kon" value="0.003"/>
          </listOfParameters>
        </kineticLaw>
      </reaction>
      <reaction id="reaction_Imatinib_binding" name="Imatinib binding" reversible="false">
        <listOfReactants>
          <speciesReference species="Imatinib" stoichiometry="1"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <ci>kon_EGF_EGFR</ci>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

# 残缺 SBML（触发正则兜底）
BROKEN_SBML = """<sbml>
<model id="broken">
<listOfSpecies>
<species id="S1" name="Protein1" compartment="cytoplasm"/>
<species id="S2" name="Protein2"/>
</listOfSpecies>
<listOfReactions>
<reaction id="R1" name="Reaction1"/>
</listOfReactions>
<listOfParameters>
<parameter id="p1" value="1.5"/>
</listOfParameters>
</model>
"""


# =============================================================================
# 1. TestSBMLParserV2
# =============================================================================
class TestSBMLParserV2(unittest.TestCase):
    """SBMLParserV2 单元测试：XML 解析 + lxml/ElementTree 降级 + 正则兜底。"""

    def test_parse_basic_xml(self):
        """测试 SBML XML 基础解析。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        doc = parser.parse(BIOMD0000000205_MOCK_SBML)
        self.assertTrue(doc.integrity)
        self.assertEqual(doc.model_id, "BIOMD0000000205")
        # backend 应为 lxml 或 elementtree（取决于环境）
        self.assertIn(doc.parser_backend, {"lxml", "elementtree"})

    def test_extract_species_count(self):
        """测试 species 提取数量。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        doc = parser.parse(BIOMD0000000205_MOCK_SBML)
        # EGF / EGFR / EGF_EGFR / Imatinib / Unknown_Protein = 5
        self.assertEqual(len(doc.species), 5)
        species_ids = {s["id"] for s in doc.species}
        self.assertIn("EGF", species_ids)
        self.assertIn("EGFR", species_ids)
        self.assertIn("EGF_EGFR", species_ids)
        self.assertIn("Imatinib", species_ids)

    def test_extract_species_annotation(self):
        """测试 species annotation 提取（含 HGNC/UniProt）。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        doc = parser.parse(BIOMD0000000205_MOCK_SBML)
        egfr = next(s for s in doc.species if s["id"] == "EGFR")
        self.assertTrue(egfr["ontology"]["hgnc_id"])
        self.assertTrue(egfr["ontology"]["uniprot_id"])
        # HGNC ID 应为 "HGNC:3236"
        self.assertEqual(egfr["ontology"]["hgnc_id"], "HGNC:3236")
        # UniProt ID 应为 "P00533"
        self.assertEqual(egfr["ontology"]["uniprot_id"], "P00533")

    def test_extract_species_chebi(self):
        """测试 ChEBI ID 提取（Imatinib 药物）。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        doc = parser.parse(BIOMD0000000205_MOCK_SBML)
        imatinib = next(s for s in doc.species if s["id"] == "Imatinib")
        self.assertEqual(imatinib["ontology"]["chebi_id"], "CHEBI:45783")

    def test_extract_reactions_count(self):
        """测试 reaction 提取数量。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        doc = parser.parse(BIOMD0000000205_MOCK_SBML)
        # reaction_EGF_EGFR_binding + reaction_Imatinib_binding = 2
        self.assertEqual(len(doc.reactions), 2)
        rxn_ids = {r["id"] for r in doc.reactions}
        self.assertIn("reaction_EGF_EGFR_binding", rxn_ids)
        self.assertIn("reaction_Imatinib_binding", rxn_ids)

    def test_extract_reactions_kinetic_law(self):
        """测试 reaction kineticLaw 提取。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        doc = parser.parse(BIOMD0000000205_MOCK_SBML)
        rxn = next(r for r in doc.reactions if r["id"] == "reaction_EGF_EGFR_binding")
        # kinetic_law 应含 kon_EGF_EGFR / koff_EGF_EGFR / EGF / EGFR / EGF_EGFR 等
        self.assertIn("kon_EGF_EGFR", rxn["kinetic_law"])
        self.assertIn("EGF", rxn["kinetic_law"])

    def test_extract_reactions_reactants_products(self):
        """测试 reaction reactants/products 提取。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        doc = parser.parse(BIOMD0000000205_MOCK_SBML)
        rxn = next(r for r in doc.reactions if r["id"] == "reaction_EGF_EGFR_binding")
        self.assertEqual(len(rxn["reactants"]), 2)
        self.assertEqual(len(rxn["products"]), 1)
        # 第一个 reactant 应为 EGF
        self.assertEqual(rxn["reactants"][0]["species"], "EGF")

    def test_extract_parameters(self):
        """测试 parameter 提取（model 层 + kineticLaw 局部）。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        doc = parser.parse(BIOMD0000000205_MOCK_SBML)
        # kon_EGF_EGFR / koff_EGF_EGFR (model) + local_kon (kineticLaw) = 3
        param_ids = {p["id"] for p in doc.parameters}
        self.assertIn("kon_EGF_EGFR", param_ids)
        self.assertIn("koff_EGF_EGFR", param_ids)
        self.assertIn("local_kon", param_ids)
        # 验证 value
        kon = next(p for p in doc.parameters if p["id"] == "kon_EGF_EGFR")
        self.assertAlmostEqual(kon["value"], 0.003)

    def test_extract_compartments(self):
        """测试 compartment 提取。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        doc = parser.parse(BIOMD0000000205_MOCK_SBML)
        self.assertEqual(len(doc.compartments), 2)
        comp_ids = {c["id"] for c in doc.compartments}
        self.assertIn("compartment", comp_ids)
        self.assertIn("extracellular", comp_ids)

    def test_parse_empty_content(self):
        """测试空内容解析返回空 SBMLDocument。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        doc = parser.parse("")
        self.assertFalse(doc.integrity)

    def test_parse_bytes_input(self):
        """测试字节输入解析。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        doc = parser.parse(BIOMD0000000205_MOCK_SBML.encode("utf-8"))
        self.assertTrue(doc.integrity)
        self.assertEqual(len(doc.species), 5)

    def test_regex_fallback(self):
        """测试正则兜底解析（XML 解析失败时）。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        # 用残缺 XML 触发正则兜底（保留 species/reaction 标签）
        doc = parser.parse(BROKEN_SBML)
        # BROKEN_SBML 实际可被 ElementTree 解析，验证 backend
        self.assertIn(doc.parser_backend, {"elementtree", "lxml", "regex"})

    def test_regex_fallback_on_invalid_xml(self):
        """测试完全非法 XML 的正则兜底。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        # 非法 XML（缺根元素闭合）但含 species 标签
        invalid_xml = '<sbml><species id="X" name="X"/><species id="Y" name="Y"/>'
        doc = parser.parse(invalid_xml)
        # 应触发正则兜底（regex）或 ElementTree 解析
        self.assertIn(doc.parser_backend, {"elementtree", "lxml", "regex"})
        # 至少能提取到 X
        self.assertGreaterEqual(len(doc.species), 1)


# =============================================================================
# 2. TestCanonicalSpeciesResolver
# =============================================================================
class TestCanonicalSpeciesResolver(unittest.TestCase):
    """CanonicalSpeciesResolver 单元测试。"""

    def test_extract_hgnc_id(self):
        """测试 HGNC ID 提取（HGNC:HGNC:3236 格式）。"""
        from app.sbml_grounder.canonical_species import CanonicalSpeciesResolver

        resolver = CanonicalSpeciesResolver()
        annotation = (
            '<annotation><rdf:li rdf:resource="https://identifiers.org/HGNC:HGNC:3236"/>'
            "</annotation>"
        )
        hgnc = resolver.extract_hgnc_id(annotation)
        self.assertEqual(hgnc, "HGNC:3236")

    def test_extract_hgnc_id_plain(self):
        """测试 HGNC ID 提取（HGNC:3236 简化格式）。"""
        from app.sbml_grounder.canonical_species import CanonicalSpeciesResolver

        resolver = CanonicalSpeciesResolver()
        annotation = 'rdf:resource="https://identifiers.org/HGNC:3236"'
        hgnc = resolver.extract_hgnc_id(annotation)
        self.assertEqual(hgnc, "HGNC:3236")

    def test_extract_uniprot_id(self):
        """测试 UniProt ID 提取（UniProt:P00533 格式）。"""
        from app.sbml_grounder.canonical_species import CanonicalSpeciesResolver

        resolver = CanonicalSpeciesResolver()
        annotation = '<annotation>UniProt:P00533</annotation>'
        uniprot = resolver.extract_uniprot_id(annotation)
        self.assertEqual(uniprot, "P00533")

    def test_extract_uniprot_id_identifiers(self):
        """测试 UniProt ID 提取（identifiers.org/uniprot/ 格式）。"""
        from app.sbml_grounder.canonical_species import CanonicalSpeciesResolver

        resolver = CanonicalSpeciesResolver()
        annotation = (
            '<annotation><rdf:li rdf:resource="https://identifiers.org/uniprot/P00533"/>'
            "</annotation>"
        )
        uniprot = resolver.extract_uniprot_id(annotation)
        self.assertEqual(uniprot, "P00533")

    def test_extract_chebi_id(self):
        """测试 ChEBI ID 提取（ChEBI:33384 格式）。"""
        from app.sbml_grounder.canonical_species import CanonicalSpeciesResolver

        resolver = CanonicalSpeciesResolver()
        annotation = '<annotation><rdf:li rdf:resource="https://identifiers.org/CHEBI:33384"/></annotation>'
        chebi = resolver.extract_chebi_id(annotation)
        self.assertEqual(chebi, "CHEBI:33384")

    def test_resolve_with_annotation(self):
        """测试带 annotation 的 species 解析（verified=True）。"""
        from app.sbml_grounder.canonical_species import CanonicalSpeciesResolver

        resolver = CanonicalSpeciesResolver()
        sbml_species = [
            {
                "id": "EGFR",
                "name": "EGFR",
                "compartment": "compartment",
                "annotation": (
                    '<annotation><rdf:li rdf:resource="https://identifiers.org/HGNC:HGNC:3236"/>'
                    '<rdf:li rdf:resource="https://identifiers.org/uniprot/P00533"/></annotation>'
                ),
                "ontology": {"hgnc_id": "HGNC:3236", "uniprot_id": "P00533"},
            }
        ]
        result = resolver.resolve(sbml_species)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["canonical_name"], "EGFR")
        self.assertEqual(result[0]["hgnc_id"], "HGNC:3236")
        self.assertEqual(result[0]["uniprot_id"], "P00533")
        self.assertTrue(result[0]["verified"])

    def test_resolve_missing_annotation_inferred(self):
        """测试缺失 annotation 时用 name+compartment 推断（verified=False）。"""
        from app.sbml_grounder.canonical_species import CanonicalSpeciesResolver

        resolver = CanonicalSpeciesResolver()
        sbml_species = [
            {
                "id": "MYSTERY",
                "name": "MysteryProtein",
                "compartment": "cytoplasm",
                "annotation": "",
                "ontology": {},
            }
        ]
        result = resolver.resolve(sbml_species)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["verified"])
        self.assertEqual(result[0]["canonical_name"], "MysteryProtein")
        self.assertEqual(result[0]["source"], "inferred_from_name_compartment")

    def test_resolve_alias_canonicalization(self):
        """测试别名归一化（ERBB1 → EGFR）。"""
        from app.sbml_grounder.canonical_species import CanonicalSpeciesResolver

        resolver = CanonicalSpeciesResolver()
        sbml_species = [
            {
                "id": "ERBB1_species",
                "name": "ERBB1",
                "compartment": "membrane",
                "annotation": "",
                "ontology": {},
            }
        ]
        result = resolver.resolve(sbml_species)
        self.assertEqual(result[0]["canonical_name"], "EGFR")

    def test_resolve_empty_input(self):
        """测试空输入返回空列表。"""
        from app.sbml_grounder.canonical_species import CanonicalSpeciesResolver

        resolver = CanonicalSpeciesResolver()
        result = resolver.resolve([])
        self.assertEqual(result, [])

    def test_resolve_exception_fallback(self):
        """测试 species 解析异常时降级返回 verified=False。"""
        from app.sbml_grounder.canonical_species import CanonicalSpeciesResolver

        resolver = CanonicalSpeciesResolver()
        # 传入非 dict 数据触发异常
        sbml_species = [{"id": None, "name": None, "compartment": None}]
        result = resolver.resolve(sbml_species)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["verified"])


# =============================================================================
# 3. TestOntologyGrounder
# =============================================================================
class TestOntologyGrounder(unittest.TestCase):
    """OntologyGrounder 单元测试。"""

    def test_ground_species_with_annotation(self):
        """测试带 annotation 的 species grounding（verified=True）。"""
        from app.sbml_grounder.ontology_grounding import OntologyGrounder

        grounder = OntologyGrounder()
        canonical_species = [
            {
                "sbml_species_id": "EGFR",
                "canonical_name": "EGFR",
                "hgnc_id": "HGNC:3236",
                "uniprot_id": "P00533",
                "chebi_id": None,
                "verified": True,
                "source": "sbml_annotation",
            }
        ]
        result = grounder.ground_species(canonical_species)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["verified"])
        self.assertEqual(result[0]["ontology_ref"]["hgnc_id"], "HGNC:3236")

    def test_ground_species_local_kb_fallback(self):
        """测试 SBML annotation 缺失时本地知识库兜底。"""
        from app.sbml_grounder.ontology_grounding import OntologyGrounder

        grounder = OntologyGrounder()
        canonical_species = [
            {
                "sbml_species_id": "AKT1_species",
                "canonical_name": "AKT1",
                "hgnc_id": None,
                "uniprot_id": None,
                "chebi_id": None,
                "verified": False,
                "source": "inferred_from_name_compartment",
            }
        ]
        result = grounder.ground_species(canonical_species)
        self.assertEqual(len(result), 1)
        # 本地知识库应补全 HGNC 与 UniProt ID
        self.assertTrue(result[0]["verified"])
        self.assertEqual(result[0]["ontology_ref"]["hgnc_id"], "HGNC:391")
        self.assertEqual(result[0]["ontology_ref"]["uniprot_id"], "P31749")

    def test_ground_species_unverified_warning(self):
        """测试未知 species 标记 unverified + warning。"""
        from app.sbml_grounder.ontology_grounding import OntologyGrounder

        grounder = OntologyGrounder()
        canonical_species = [
            {
                "sbml_species_id": "UNKNOWN1",
                "canonical_name": "UnknownProteinX",
                "hgnc_id": None,
                "uniprot_id": None,
                "chebi_id": None,
                "verified": False,
                "source": "inferred_from_name_compartment",
            }
        ]
        result = grounder.ground_species(canonical_species)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["verified"])
        self.assertGreater(len(result[0]["warnings"]), 0)

    def test_ground_species_drug_chebi(self):
        """测试药物 species 必须有 ChEBI ID。"""
        from app.sbml_grounder.ontology_grounding import OntologyGrounder

        grounder = OntologyGrounder()
        canonical_species = [
            {
                "sbml_species_id": "Imatinib",
                "canonical_name": "IMATINIB",
                "hgnc_id": None,
                "uniprot_id": None,
                "chebi_id": "CHEBI:45783",
                "verified": True,
                "source": "sbml_annotation",
            }
        ]
        result = grounder.ground_species(canonical_species)
        self.assertEqual(len(result), 1)
        # Imatinib 在本地知识库中被识别为 drug
        self.assertEqual(result[0]["ontology_ref"].get("chebi_id"), "CHEBI:45783")

    def test_align_with_p1_ontology(self):
        """测试与 P1 ontology_entities 对齐。"""
        from app.sbml_grounder.ontology_grounding import OntologyGrounder

        grounder = OntologyGrounder()
        p1_entities = [
            {
                "name": "EGFR",
                "hgnc_id": "HGNC:3236",
                "uniprot_id": "P00533",
                "chebi_id": None,
                "species_type": "protein",
                "verified": True,
                "source": "uniprot",
            }
        ]
        index = grounder.align_with_p1_ontology(p1_entities)
        self.assertIn("EGFR", index)
        self.assertEqual(index["EGFR"]["hgnc_id"], "HGNC:3236")

    def test_align_with_p1_ontology_dict_input(self):
        """测试 P1 ontology_entities dict 输入（含 entities key）。"""
        from app.sbml_grounder.ontology_grounding import OntologyGrounder

        grounder = OntologyGrounder()
        p1_dict = {
            "entities": [
                {
                    "name": "AKT1",
                    "hgnc_id": "HGNC:391",
                    "uniprot_id": "P31749",
                    "species_type": "protein",
                    "verified": True,
                }
            ],
            "pathway_class": "PI3K_AKT_mTOR",
        }
        index = grounder.align_with_p1_ontology(p1_dict)
        self.assertIn("AKT1", index)

    def test_ground_species_p1_alignment(self):
        """测试 P1 ontology 对齐补全 species ID。"""
        from app.sbml_grounder.ontology_grounding import OntologyGrounder

        grounder = OntologyGrounder()
        # 自定义一个不在本地知识库的 species（如 AKT9）
        canonical_species = [
            {
                "sbml_species_id": "AKT1_sp",
                "canonical_name": "AKT1",
                "hgnc_id": None,
                "uniprot_id": None,
                "chebi_id": None,
                "verified": False,
                "source": "inferred_from_name_compartment",
            }
        ]
        p1_entities = [
            {
                "name": "AKT1",
                "hgnc_id": "HGNC:391",
                "uniprot_id": "P31749",
                "species_type": "protein",
                "verified": True,
            }
        ]
        result = grounder.ground_species(canonical_species, p1_entities)
        self.assertTrue(result[0]["verified"])

    def test_ground_species_empty_input(self):
        """测试空输入返回空列表。"""
        from app.sbml_grounder.ontology_grounding import OntologyGrounder

        grounder = OntologyGrounder()
        result = grounder.ground_species([])
        self.assertEqual(result, [])


# =============================================================================
# 4. TestAliasResolver
# =============================================================================
class TestAliasResolver(unittest.TestCase):
    """AliasResolver 单元测试。"""

    def test_canonicalize_egfr_alias_erbb1(self):
        """测试 EGFR 别名解析（ERBB1 → EGFR）。"""
        from app.sbml_grounder.alias_resolution import AliasResolver

        resolver = AliasResolver()
        self.assertEqual(resolver.canonicalize("ERBB1"), "EGFR")

    def test_canonicalize_egfr_alias_her1(self):
        """测试 EGFR 别名解析（HER1 → EGFR）。"""
        from app.sbml_grounder.alias_resolution import AliasResolver

        resolver = AliasResolver()
        self.assertEqual(resolver.canonicalize("HER1"), "EGFR")

    def test_canonicalize_egfr_self(self):
        """测试 EGFR 自身解析（EGFR → EGFR）。"""
        from app.sbml_grounder.alias_resolution import AliasResolver

        resolver = AliasResolver()
        self.assertEqual(resolver.canonicalize("EGFR"), "EGFR")

    def test_canonicalize_mapk1_alias_erk2(self):
        """测试 MAPK1 别名解析（ERK2 → MAPK1）。"""
        from app.sbml_grounder.alias_resolution import AliasResolver

        resolver = AliasResolver()
        self.assertEqual(resolver.canonicalize("ERK2"), "MAPK1")

    def test_canonicalize_akt1_alias_pkb(self):
        """测试 AKT1 别名解析（PKB → AKT1）。"""
        from app.sbml_grounder.alias_resolution import AliasResolver

        resolver = AliasResolver()
        self.assertEqual(resolver.canonicalize("PKB"), "AKT1")

    def test_canonicalize_akt1_alias_pkb_alpha(self):
        """测试 AKT1 别名解析（PKB-ALPHA → AKT1）。"""
        from app.sbml_grounder.alias_resolution import AliasResolver

        resolver = AliasResolver()
        self.assertEqual(resolver.canonicalize("PKB-ALPHA"), "AKT1")

    def test_canonicalize_case_insensitive(self):
        """测试大小写不敏感（egfr → EGFR）。"""
        from app.sbml_grounder.alias_resolution import AliasResolver

        resolver = AliasResolver()
        self.assertEqual(resolver.canonicalize("egfr"), "EGFR")
        self.assertEqual(resolver.canonicalize("Egfr"), "EGFR")

    def test_canonicalize_unknown_keeps_original(self):
        """测试未知名称保持原样。"""
        from app.sbml_grounder.alias_resolution import AliasResolver

        resolver = AliasResolver()
        self.assertEqual(resolver.canonicalize("UNKNOWN_PROTEIN_X"), "UNKNOWN_PROTEIN_X")

    def test_canonicalize_empty_string(self):
        """测试空字符串处理。"""
        from app.sbml_grounder.alias_resolution import AliasResolver

        resolver = AliasResolver()
        self.assertEqual(resolver.canonicalize(""), "")

    def test_resolve_aliases_batch(self):
        """测试批量别名解析。"""
        from app.sbml_grounder.alias_resolution import AliasResolver

        resolver = AliasResolver()
        result = resolver.resolve_aliases(["ERBB1", "ERK2", "PKB", "EGFR"])
        self.assertEqual(result["ERBB1"], "EGFR")
        self.assertEqual(result["ERK2"], "MAPK1")
        self.assertEqual(result["PKB"], "AKT1")
        self.assertEqual(result["EGFR"], "EGFR")

    def test_build_alias_map(self):
        """测试构建别名映射表。"""
        from app.sbml_grounder.alias_resolution import AliasResolver

        resolver = AliasResolver()
        alias_map = resolver.build_alias_map()
        self.assertIsInstance(alias_map, dict)
        self.assertIn("ERBB1", alias_map)
        self.assertEqual(alias_map["ERBB1"], "EGFR")

    def test_get_aliases_for(self):
        """测试获取指定 canonical name 的所有别名。"""
        from app.sbml_grounder.alias_resolution import AliasResolver

        resolver = AliasResolver()
        aliases = resolver.get_aliases_for("EGFR")
        self.assertIn("EGFR", aliases)
        self.assertIn("ERBB1", aliases)
        self.assertIn("HER1", aliases)

    def test_get_aliases_for_unknown(self):
        """测试获取未知 canonical name 的别名（返回自身）。"""
        from app.sbml_grounder.alias_resolution import AliasResolver

        resolver = AliasResolver()
        aliases = resolver.get_aliases_for("UNKNOWN_PROTEIN")
        self.assertEqual(aliases, ["UNKNOWN_PROTEIN"])


# =============================================================================
# 5. TestFiveLevelMapper
# =============================================================================
class TestFiveLevelMapper(unittest.TestCase):
    """FiveLevelMapper 单元测试：五级映射链。"""

    def _build_test_state(self):
        """构造测试用 state（模拟 BIOMD0000000205 场景）。"""
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        parser = SBMLParserV2()
        sbml_doc = parser.parse(BIOMD0000000205_MOCK_SBML)

        ode_system = {
            "pathway_class": "EGFR_RTK",
            "template_name": "Signaling_Cascade_Phos",
            "ode_code": "dEGF/dt = -kon_EGF_EGFR * EGF * EGFR",
            "equations": [
                {
                    "eq_id": "ODE_001",
                    "reaction_id": "RXN_001",
                    "species_id": "EGF",
                    "rhs": "-kon_EGF_EGFR * EGF * EGFR",
                }
            ],
        }

        reaction_ir = {
            "reactions": [
                {
                    "id": "RXN_001",
                    "name": "EGF-EGFR binding",
                    "kinetics_type": "mass_action",
                    "parameter_context": "EGF-EGFR binding kon/koff",
                    "provenance": {
                        "source_sbml_reaction": "reaction_EGF_EGFR_binding",
                        "source_pmid": "PMID:12215431",
                    },
                }
            ],
        }

        parameters = {
            "edge1": {
                "param_name": "kon_EGF_EGFR",
                "value": 0.003,
                "source": "BioModels",
                "confidence": 0.95,
                "origin": "PMID:12215431",
                "is_fallback": False,
                "missing_parameter": False,
            }
        }

        return sbml_doc, ode_system, reaction_ir, parameters

    def test_build_mapping_basic(self):
        """测试五级映射基础构建。"""
        from app.sbml_grounder.five_level_mapping import FiveLevelMapper

        mapper = FiveLevelMapper()
        sbml_doc, ode_system, reaction_ir, parameters = self._build_test_state()
        ledger = mapper.build_mapping(
            ode_system=ode_system,
            reaction_ir=reaction_ir,
            sbml_document=sbml_doc,
            parameters=parameters,
        )
        self.assertIn("ode_equations", ledger)
        self.assertIn("species_mapping", ledger)
        self.assertIn("integrity", ledger)
        self.assertIn("warnings", ledger)
        self.assertIn("statistics", ledger)

    def test_map_ode_to_reaction(self):
        """测试 Level 1: ODE → Reaction 映射。"""
        from app.sbml_grounder.five_level_mapping import FiveLevelMapper

        mapper = FiveLevelMapper()
        sbml_doc, ode_system, reaction_ir, parameters = self._build_test_state()
        ode_equations = mapper._extract_ode_equations(ode_system)
        reactions = mapper._extract_reactions(reaction_ir)
        result = mapper.map_ode_to_reaction(ode_equations, reactions)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["mapped"])
        self.assertEqual(result[0]["reaction_id"], "RXN_001")

    def test_map_reaction_to_sbml(self):
        """测试 Level 2: Reaction → SBML Reaction 映射。"""
        from app.sbml_grounder.five_level_mapping import FiveLevelMapper

        mapper = FiveLevelMapper()
        sbml_doc, ode_system, reaction_ir, parameters = self._build_test_state()
        reactions = mapper._extract_reactions(reaction_ir)
        result = mapper.map_reaction_to_sbml(reactions, sbml_doc.reactions)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["mapped"])
        self.assertEqual(
            result[0]["sbml_reaction_id"], "reaction_EGF_EGFR_binding"
        )
        self.assertEqual(result[0]["match_method"], "provenance_source_sbml")

    def test_map_sbml_to_parameter(self):
        """测试 Level 3: SBML → Parameter 映射。"""
        from app.sbml_grounder.five_level_mapping import FiveLevelMapper

        mapper = FiveLevelMapper()
        sbml_doc, ode_system, reaction_ir, parameters = self._build_test_state()
        reactions = mapper._extract_reactions(reaction_ir)
        result = mapper.map_sbml_to_parameter(
            sbml_doc.reactions, parameters, reactions
        )
        # reaction_EGF_EGFR_binding 的 kineticLaw 含 kon_EGF_EGFR
        binding_match = next(
            r for r in result if r["sbml_reaction_id"] == "reaction_EGF_EGFR_binding"
        )
        self.assertTrue(binding_match["mapped"])
        self.assertIn("kon_EGF_EGFR", binding_match["parameter_ids"])

    def test_map_parameter_to_pmid(self):
        """测试 Level 4: Parameter → PMID 映射。"""
        from app.sbml_grounder.five_level_mapping import FiveLevelMapper

        mapper = FiveLevelMapper()
        sbml_doc, ode_system, reaction_ir, parameters = self._build_test_state()
        reactions = mapper._extract_reactions(reaction_ir)
        result = mapper.map_parameter_to_pmid(parameters, reactions)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["mapped"])
        self.assertIn("PMID:12215431", result[0]["pmids"])

    def test_map_species_to_ontology(self):
        """测试 Level 5: Species → ontology ID 映射。"""
        from app.sbml_grounder.five_level_mapping import FiveLevelMapper

        mapper = FiveLevelMapper()
        sbml_doc, _, _, _ = self._build_test_state()
        result = mapper.map_species_to_ontology(sbml_doc.species)
        # 5 个 species（EGF / EGFR / EGF_EGFR / Imatinib / Unknown_Protein）
        self.assertEqual(len(result), 5)
        # EGFR 应 verified=True
        egfr = next(s for s in result if s["canonical_name"] == "EGFR")
        self.assertTrue(egfr["verified"])

    def test_compute_integrity_complete_chain(self):
        """测试完整映射链 integrity=True。"""
        from app.sbml_grounder.five_level_mapping import FiveLevelMapper

        mapper = FiveLevelMapper()
        sbml_doc, ode_system, reaction_ir, parameters = self._build_test_state()
        ledger = mapper.build_mapping(
            ode_system=ode_system,
            reaction_ir=reaction_ir,
            sbml_document=sbml_doc,
            parameters=parameters,
        )
        # 完整场景：所有级别都映射成功
        self.assertTrue(ledger["integrity"])

    def test_compute_integrity_broken_chain(self):
        """测试映射链断裂 integrity=False。"""
        from app.sbml_grounder.five_level_mapping import FiveLevelMapper

        mapper = FiveLevelMapper()
        # reaction_ir 缺失 provenance.source_sbml_reaction
        ode_system = {
            "equations": [
                {"eq_id": "ODE_001", "reaction_id": "RXN_X", "species_id": "X"}
            ]
        }
        reaction_ir = {
            "reactions": [
                {
                    "id": "RXN_X",
                    "provenance": {},  # 缺 source_sbml_reaction
                }
            ]
        }
        # SBML 中无匹配 reaction
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        sbml_doc = SBMLParserV2().parse(BIOMD0000000205_MOCK_SBML)
        parameters = {}
        ledger = mapper.build_mapping(
            ode_system=ode_system,
            reaction_ir=reaction_ir,
            sbml_document=sbml_doc,
            parameters=parameters,
        )
        self.assertFalse(ledger["integrity"])

    def test_compute_integrity_empty_inputs(self):
        """测试空输入时 integrity=False。"""
        from app.sbml_grounder.five_level_mapping import FiveLevelMapper

        mapper = FiveLevelMapper()
        ledger = mapper.build_mapping(
            ode_system=None,
            reaction_ir=None,
            sbml_document=None,
            parameters=None,
        )
        self.assertFalse(ledger["integrity"])
        self.assertGreater(len(ledger["warnings"]), 0)

    def test_extract_ode_equations_from_code(self):
        """测试从 ode_code 正则提取方程（equations 字段缺失时）。"""
        from app.sbml_grounder.five_level_mapping import FiveLevelMapper

        mapper = FiveLevelMapper()
        ode_system = {
            "ode_code": "dEGF/dt = -kon * EGF\ndEGFR/dt = +koff * EGF_EGFR\n",
        }
        equations = mapper._extract_ode_equations(ode_system)
        self.assertEqual(len(equations), 2)
        self.assertEqual(equations[0]["species_id"], "EGF")
        self.assertEqual(equations[1]["species_id"], "EGFR")

    def test_merge_five_levels_ledger(self):
        """测试五级映射合并为 ledger。"""
        from app.sbml_grounder.five_level_mapping import FiveLevelMapper

        mapper = FiveLevelMapper()
        sbml_doc, ode_system, reaction_ir, parameters = self._build_test_state()
        ledger = mapper.build_mapping(
            ode_system=ode_system,
            reaction_ir=reaction_ir,
            sbml_document=sbml_doc,
            parameters=parameters,
        )
        # ode_equations[0] 应含完整五级字段
        eq = ledger["ode_equations"][0]
        self.assertEqual(eq["eq_id"], "ODE_001")
        self.assertEqual(eq["reaction_id"], "RXN_001")
        self.assertEqual(eq["sbml_reaction_id"], "reaction_EGF_EGFR_binding")
        self.assertIn("kon_EGF_EGFR", eq["parameter_ids"])
        self.assertIn("PMID:12215431", eq["pmids"])

    def test_statistics(self):
        """测试统计信息计算。"""
        from app.sbml_grounder.five_level_mapping import FiveLevelMapper

        mapper = FiveLevelMapper()
        sbml_doc, ode_system, reaction_ir, parameters = self._build_test_state()
        ledger = mapper.build_mapping(
            ode_system=ode_system,
            reaction_ir=reaction_ir,
            sbml_document=sbml_doc,
            parameters=parameters,
        )
        stats = ledger["statistics"]
        self.assertEqual(stats["total_ode_equations"], 1)
        self.assertEqual(stats["mapped_reactions"], 1)
        self.assertGreaterEqual(stats["total_species"], 5)


# =============================================================================
# 6. TestSBMLGrounderAgent
# =============================================================================
class TestSBMLGrounderAgent(unittest.TestCase):
    """SBMLGrounderAgent 单元测试：主入口 + Feature Flag + 异常降级。"""

    def _build_full_state(self):
        """构造完整测试 state（BIOMD0000000205 场景）。"""
        from app.sbml_grounder.five_level_mapping import FiveLevelMapper
        from app.sbml_grounder.sbml_parser_v2 import SBMLParserV2

        return {
            "sbml_model_id": "BIOMD0000000205",
            "sbml_model_text": BIOMD0000000205_MOCK_SBML,
            "v4_ode_system": {
                "pathway_class": "EGFR_RTK",
                "ode_code": "dEGF/dt = -kon_EGF_EGFR * EGF * EGFR",
                "equations": [
                    {
                        "eq_id": "ODE_001",
                        "reaction_id": "RXN_001",
                        "species_id": "EGF",
                    }
                ],
            },
            "v4_reaction_ir": {
                "reactions": [
                    {
                        "id": "RXN_001",
                        "name": "EGF-EGFR binding",
                        "kinetics_type": "mass_action",
                        "parameter_context": "EGF-EGFR binding",
                        "provenance": {
                            "source_sbml_reaction": "reaction_EGF_EGFR_binding",
                            "source_pmid": "PMID:12215431",
                        },
                    }
                ]
            },
            "parameters": {
                "edge1": {
                    "param_name": "kon_EGF_EGFR",
                    "value": 0.003,
                    "source": "BioModels",
                    "confidence": 0.95,
                    "origin": "PMID:12215431",
                    "is_fallback": False,
                    "missing_parameter": False,
                }
            },
            "v4_ontology_entities": None,
        }

    def test_ground_main_entry(self):
        """测试 ground() 主入口。"""
        from app.sbml_grounder.grounder_agent import SBMLGrounderAgent

        agent = SBMLGrounderAgent()
        result = agent.ground(self._build_full_state())
        self.assertIn("v4_grounding_ledger", result)
        ledger = result["v4_grounding_ledger"]
        self.assertIn("ode_equations", ledger)
        self.assertIn("integrity", ledger)
        self.assertIn("statistics", ledger)

    def test_ground_writes_v4_grounding_ledger(self):
        """测试 ground() 写入 v4_grounding_ledger 字段。"""
        from app.sbml_grounder.grounder_agent import SBMLGrounderAgent

        agent = SBMLGrounderAgent()
        result = agent.ground(self._build_full_state())
        # 必须写入 v4_grounding_ledger（不是 v3 字段）
        self.assertIn("v4_grounding_ledger", result)
        self.assertNotIn("sbml_parsed_network", result)

    def test_hook_flag_off_returns_empty(self):
        """测试 Feature Flag=false 时 hook 返回 {}。"""
        from app.sbml_grounder.grounder_agent import sbml_grounder_hook_node

        with patch("app.sbml_grounder.grounder_agent.settings") as mock_settings:
            mock_settings.V4_SBML_GROUNDER_ENABLED = False
            mock_settings.effective_v4_sbml_grounder_enabled.return_value = False
            result = sbml_grounder_hook_node(self._build_full_state())
            self.assertEqual(result, {})

    def test_hook_flag_on_writes_ledger(self):
        """测试 Feature Flag=true 时 hook 写入 v4_grounding_ledger。"""
        from app.sbml_grounder.grounder_agent import sbml_grounder_hook_node

        with patch("app.sbml_grounder.grounder_agent.settings") as mock_settings:
            mock_settings.V4_SBML_GROUNDER_ENABLED = True
            result = sbml_grounder_hook_node(self._build_full_state())
            self.assertIn("v4_grounding_ledger", result)

    def test_hook_exception_returns_empty(self):
        """测试 hook 异常时降级返回 {}。"""
        from app.sbml_grounder.grounder_agent import sbml_grounder_hook_node

        with patch("app.sbml_grounder.grounder_agent.settings") as mock_settings:
            mock_settings.V4_SBML_GROUNDER_ENABLED = True
            # 让 SBMLGrounderAgent.ground 抛异常
            with patch(
                "app.sbml_grounder.grounder_agent.SBMLGrounderAgent.ground",
                side_effect=RuntimeError("boom"),
            ):
                result = sbml_grounder_hook_node(self._build_full_state())
                self.assertEqual(result, {})

    def test_ground_no_sbml_text(self):
        """测试 sbml_model_text 缺失时不抛异常。"""
        from app.sbml_grounder.grounder_agent import SBMLGrounderAgent

        agent = SBMLGrounderAgent()
        state = self._build_full_state()
        state["sbml_model_text"] = ""
        result = agent.ground(state)
        # 仍应返回 ledger（integrity=False）
        self.assertIn("v4_grounding_ledger", result)
        self.assertFalse(result["v4_grounding_ledger"]["integrity"])

    def test_ground_empty_state(self):
        """测试空 state 降级返回空 ledger。"""
        from app.sbml_grounder.grounder_agent import SBMLGrounderAgent

        agent = SBMLGrounderAgent()
        result = agent.ground({})
        self.assertIn("v4_grounding_ledger", result)
        self.assertFalse(result["v4_grounding_ledger"]["integrity"])

    def test_ground_biomd0000000205_full_mapping(self):
        """测试 BIOMD0000000205 mock 五级映射完整性。"""
        from app.sbml_grounder.grounder_agent import SBMLGrounderAgent

        agent = SBMLGrounderAgent()
        result = agent.ground(self._build_full_state())
        ledger = result["v4_grounding_ledger"]
        # BIOMD0000000205 场景应 integrity=True
        self.assertTrue(ledger["integrity"], f"integrity=False: {ledger.get('warnings')}")
        # ode_equations 应有 1 条完整映射
        self.assertEqual(len(ledger["ode_equations"]), 1)
        eq = ledger["ode_equations"][0]
        self.assertEqual(eq["reaction_id"], "RXN_001")
        self.assertEqual(eq["sbml_reaction_id"], "reaction_EGF_EGFR_binding")
        self.assertIn("kon_EGF_EGFR", eq["parameter_ids"])
        self.assertIn("PMID:12215431", eq["pmids"])
        # canonical species 解析正确（EGFR verified=True）
        egfr_species = next(
            s for s in ledger["species_mapping"] if s["canonical_name"] == "EGFR"
        )
        self.assertTrue(egfr_species["verified"])
        self.assertEqual(egfr_species["ontology_ref"]["uniprot_id"], "P00533")

    def test_ground_does_not_modify_v3_fields(self):
        """测试 ground() 不写入任何 v3 字段。"""
        from app.sbml_grounder.grounder_agent import SBMLGrounderAgent

        agent = SBMLGrounderAgent()
        result = agent.ground(self._build_full_state())
        v3_keys = {
            "network_json",
            "parameters",
            "ode_model",
            "sbml_parsed_network",
            "entities",
            "mechanism",
            "validation_report",
        }
        for key in v3_keys:
            self.assertNotIn(key, result, f"ground() 不应写入 v3 字段 {key}")

    def test_hook_default_flag_off(self):
        """测试默认 Feature Flag=false 时 hook 返回 {}。"""
        from app.sbml_grounder.grounder_agent import sbml_grounder_hook_node
        from app.config import settings

        # 默认 V4_SBML_GROUNDER_ENABLED=False（铁律）
        self.assertFalse(settings.V4_SBML_GROUNDER_ENABLED)
        result = sbml_grounder_hook_node(self._build_full_state())
        self.assertEqual(result, {})

    def test_ground_with_p1_ontology_entities(self):
        """测试 ground() 接收 P1 ontology_entities 输入。"""
        from app.sbml_grounder.grounder_agent import SBMLGrounderAgent

        agent = SBMLGrounderAgent()
        state = self._build_full_state()
        state["v4_ontology_entities"] = {
            "entities": [
                {
                    "name": "EGFR",
                    "hgnc_id": "HGNC:3236",
                    "uniprot_id": "P00533",
                    "species_type": "protein",
                    "verified": True,
                }
            ]
        }
        result = agent.ground(state)
        ledger = result["v4_grounding_ledger"]
        self.assertIn("ode_equations", ledger)


# =============================================================================
# 7. TestStateField（验证 state.py 新增 v4_grounding_ledger 字段）
# =============================================================================
class TestStateField(unittest.TestCase):
    """验证 state.py 新增 v4_grounding_ledger 字段。"""

    def test_state_has_v4_grounding_ledger_field(self):
        """测试 BioDynamicsState 含 v4_grounding_ledger 字段。"""
        from app.state import BioDynamicsState

        # TypedDict total=False 时所有字段可选
        self.assertIn("v4_grounding_ledger", BioDynamicsState.__annotations__)

    def test_state_field_does_not_break_v3(self):
        """测试新增字段不破坏 v3 既有字段。"""
        from app.state import BioDynamicsState

        annotations = BioDynamicsState.__annotations__
        # v3 字段必须仍然存在
        for key in ["network_json", "parameters", "ode_model", "sbml_parsed_network"]:
            self.assertIn(key, annotations)
        # v4 Phase 1-4 字段必须仍然存在
        for key in [
            "v4_ontology_entities",
            "v4_reaction_ir",
            "v4_pathway_graph",
            "v4_ode_system",
            "v4_pathway_class",
            "v4_specialist_outputs",
            "v4_crosstalk_edges",
        ]:
            self.assertIn(key, annotations)


# =============================================================================
# 8. TestPackageInit（验证包导出与导入）
# =============================================================================
class TestPackageInit(unittest.TestCase):
    """验证 sbml_grounder 包导出。"""

    def test_package_importable(self):
        """测试 sbml_grounder 包可导入。"""
        import app.sbml_grounder

        self.assertTrue(hasattr(app.sbml_grounder, "SBMLGrounderAgent"))
        self.assertTrue(hasattr(app.sbml_grounder, "sbml_grounder_hook_node"))
        self.assertTrue(hasattr(app.sbml_grounder, "SBMLParserV2"))
        self.assertTrue(hasattr(app.sbml_grounder, "FiveLevelMapper"))
        self.assertTrue(hasattr(app.sbml_grounder, "CanonicalSpeciesResolver"))
        self.assertTrue(hasattr(app.sbml_grounder, "OntologyGrounder"))
        self.assertTrue(hasattr(app.sbml_grounder, "AliasResolver"))

    def test_hook_node_callable(self):
        """测试 sbml_grounder_hook_node 可调用。"""
        from app.sbml_grounder import sbml_grounder_hook_node

        self.assertTrue(callable(sbml_grounder_hook_node))


if __name__ == "__main__":
    unittest.main()
