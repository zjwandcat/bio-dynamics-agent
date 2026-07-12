# BioDynamics Agent v4 - SBML Parser V2 (Phase 5 / Task 5.1.2)
#
# 真正 XML 解析 SBML，替代 v3 LLM 解析（v3 依赖 LLM 从 SBML 文本提取网络拓扑，
# 不可重现且不可验证）。v2 优先用 lxml（更严格的 XML 校验），不可用时降级到
# xml.etree.ElementTree（标准库），最终降级到正则解析（提取 species/reaction 名）。
#
# 设计原则（铁律）：
# 1. 失败降级：lxml 不可用 → ElementTree → 正则兜底，不抛异常
# 2. 容错解析：单个元素解析失败不影响整体（跳过并记录 warning）
# 3. SBML L2/L3 兼容：namespaces 自动探测（sbml2/sbml3）
# 4. 输出 SBMLDocument dataclass，含 species/reactions/parameters/compartments
#
# 依赖：
# - app.config.LXML_AVAILABLE / LXML_VERSION（try-import 模板）
# - lxml.etree（可选，优先使用）/ xml.etree.ElementTree（标准库后备）

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import LXML_AVAILABLE

logger = logging.getLogger(__name__)


# =============================================================================
# SBML namespace 探测
# =============================================================================
# SBML L2 默认 namespace：http://www.sbml.org/sbml/level2/version2 (或 version3/version4)
# SBML L3 默认 namespace：http://www.sbml.org/sbml/level3/version1 (或 version2)
# 解析时自动探测实际 namespace（取 root tag 的 namespace 部分）。
_SBML_NS_PATTERN = re.compile(r"^\{([^}]+)\}")


def _detect_namespace(root_tag: str) -> str:
    """从 root tag 提取 namespace（如 '{http://...sbml/level2/version2}sbml' → namespace）。

    Args:
        root_tag: Element.tag 字符串，可能含 '{ns}localname' 形式

    Returns:
        namespace 字符串（含尾随 '/'），无 namespace 返回空字符串
    """
    m = _SBML_NS_PATTERN.match(root_tag)
    if m:
        ns = m.group(1)
        if not ns.endswith("/"):
            ns = ns + "/"
        return ns
    return ""


def _qname(ns: str, local: str) -> str:
    """构造带 namespace 的元素名（用于 lxml/ElementTree 查找）。

    Args:
        ns: namespace 字符串（含尾随 '/'）或空字符串
        local: 本地名（如 'species'）

    Returns:
        '{ns}local' 或 'local'（无 namespace 时）
    """
    if ns:
        return "{" + ns.rstrip("/") + "}" + local
    return local


# =============================================================================
# SBMLDocument dataclass
# =============================================================================
@dataclass
class SBMLDocument:
    """SBML 解析结果容器。

    所有字段均为 list[dict]，保持纯数据结构（无 Pydantic 依赖，便于序列化）。
    integrity=True 表示解析完整无错误；False 表示有元素解析失败但仍有部分结果。
    """

    species: list[dict[str, Any]] = field(default_factory=list)
    reactions: list[dict[str, Any]] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    compartments: list[dict[str, Any]] = field(default_factory=list)
    integrity: bool = True
    # 元信息
    level: str = ""
    version: str = ""
    model_id: str = ""
    parser_backend: str = "unknown"  # "lxml" / "elementtree" / "regex"
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# SBMLParserV2 主类
# =============================================================================
class SBMLParserV2:
    """SBML XML 解析器 v2。

    优先用 lxml（LXML_AVAILABLE=True 时提供更严格的 XML 校验与 schema 验证），
    降级到 xml.etree.ElementTree（标准库），最终降级到正则解析（仅提取 species/reaction 名）。

    用法：
        parser = SBMLParserV2()
        doc = parser.parse(sbml_xml_string)
        for sp in doc.species:
            print(sp["id"], sp.get("annotation"))
    """

    # SBML annotation 中常见的 ontology ID 提取正则
    # HGNC ID 格式：HGNC:HGNC:3236 或 HGNC:3236
    _HGNC_RE = re.compile(r"HGNC:HGNC:(\d+)", re.IGNORECASE)
    # UniProt accession：6 个常用字符（P00533 / Q15375 等），10 个字符（O14578 等）
    _UNIPROT_RE = re.compile(
        r"[opq][0-9][a-z0-9]{3}\d|[a-nr-z][0-9]([a-z][a-z0-9]{2}\d){1,2}",
        re.IGNORECASE,
    )
    # 显式 UniProt: 前缀（如 annotation 中 "UniProt:P00533"）
    _UNIPROT_PREFIX_RE = re.compile(r"UniProt[:\s]+([A-Z0-9]{6,10})", re.IGNORECASE)
    # ChEBI ID：CHEBI:33384 或 ChEBI:33384
    _CHEBI_RE = re.compile(r"CHEBI:(\d+)", re.IGNORECASE)

    def parse(self, sbml_content: str | bytes) -> SBMLDocument:
        """主解析入口：解析 SBML XML 字符串/字节。

        解析顺序：
        1. lxml（若 LXML_AVAILABLE=True）：更严格的 XML 校验
        2. xml.etree.ElementTree（标准库后备）
        3. 正则解析（XML 解析失败时的兜底，仅提取 species/reaction 名）

        Args:
            sbml_content: SBML XML 字符串或字节

        Returns:
            SBMLDocument 实例（integrity=False 表示部分解析失败）
        """
        if not sbml_content:
            return SBMLDocument(integrity=False, warnings=["empty sbml content"])

        # 优先用 lxml
        if LXML_AVAILABLE:
            try:
                return self._parse_with_lxml(sbml_content)
            except Exception as exc:
                logger.warning("lxml 解析失败，降级到 ElementTree: %s", exc)

        # 降级到 ElementTree
        try:
            return self._parse_with_elementtree(sbml_content)
        except Exception as exc:
            logger.warning(
                "ElementTree 解析失败，降级到正则解析: %s", exc
            )
            return self._parse_with_regex(sbml_content)

    # -------------------------------------------------------------------------
    # lxml 后端
    # -------------------------------------------------------------------------
    def _parse_with_lxml(self, sbml_content: str | bytes) -> SBMLDocument:
        """用 lxml.etree 解析 SBML（更严格的 XML 校验）。"""
        from lxml import etree as lxml_etree  # type: ignore

        if isinstance(sbml_content, str):
            sbml_content = sbml_content.encode("utf-8")
        # recover=False 严格模式；XML 错误时抛 XMLSyntaxError（被 parse 捕获降级）
        # Task 19 SEC-1.2: 显式禁用实体解析与网络访问（深度防御）
        parser = lxml_etree.XMLParser(
            remove_blank_text=False,
            remove_comments=False,
            recover=False,
            resolve_entities=False,
            no_network=True,
        )
        root = lxml_etree.fromstring(sbml_content, parser=parser)
        return self._build_document(root, backend="lxml")

    # -------------------------------------------------------------------------
    # ElementTree 后端（标准库后备）
    # -------------------------------------------------------------------------
    def _parse_with_elementtree(self, sbml_content: str | bytes) -> SBMLDocument:
        """用 defusedxml.ElementTree 解析 SBML（安全后备）。

        Task 19 SEC-1.1: 原 xml.etree.ElementTree 不防御内部实体扩展（Billion Laughs），
        改用 defusedxml.ElementTree。defusedxml 不可用时降级到正则解析，
        绝不回退到 xml.etree（与 biomodels_client.py 安全硬约束一致）。
        """
        try:
            from defusedxml import ElementTree as ET  # type: ignore
        except ImportError:
            # defusedxml 不可用：降级到正则解析，不回退到 xml.etree
            logger.warning(
                "defusedxml 未安装，SBML ElementTree 后备降级到正则解析"
            )
            if isinstance(sbml_content, bytes):
                sbml_content = sbml_content.decode("utf-8", errors="replace")
            return self._parse_with_regex(sbml_content)

        if isinstance(sbml_content, bytes):
            root = ET.fromstring(sbml_content)
        else:
            root = ET.fromstring(sbml_content)
        return self._build_document(root, backend="defusedxml")

    # -------------------------------------------------------------------------
    # 正则兜底（XML 解析全失败时）
    # -------------------------------------------------------------------------
    def _parse_with_regex(self, sbml_content: str) -> SBMLDocument:
        """正则兜底解析：仅提取 species/reaction 名，无 annotation/kineticLaw。

        触发条件：lxml 与 ElementTree 均解析失败（如 SBML 是伪 XML 片段）。
        输出 SBMLDocument.integrity=False，species/reactions 仅含 id/name。
        """
        if isinstance(sbml_content, bytes):
            sbml_content = sbml_content.decode("utf-8", errors="ignore")

        species: list[dict[str, Any]] = []
        reactions: list[dict[str, Any]] = []

        # 提取 species 标签（<species id="..." name="..."/>）
        for m in re.finditer(
            r"<species\b([^/>]*?)/?>", sbml_content, re.IGNORECASE | re.DOTALL
        ):
            attrs = m.group(1)
            sp_id = self._extract_xml_attr(attrs, "id")
            sp_name = self._extract_xml_attr(attrs, "name") or sp_id
            compartment = self._extract_xml_attr(attrs, "compartment")
            if sp_id:
                species.append(
                    {
                        "id": sp_id,
                        "name": sp_name,
                        "compartment": compartment,
                        "metaid": self._extract_xml_attr(attrs, "metaid"),
                        "annotation": "",
                        "ontology": {},
                    }
                )

        # 提取 reaction 标签
        for m in re.finditer(
            r"<reaction\b([^/>]*?)(?:/>|>.*?</reaction>)",
            sbml_content,
            re.IGNORECASE | re.DOTALL,
        ):
            attrs = m.group(1)
            rxn_id = self._extract_xml_attr(attrs, "id")
            rxn_name = self._extract_xml_attr(attrs, "name") or rxn_id
            if rxn_id:
                reactions.append(
                    {
                        "id": rxn_id,
                        "name": rxn_name,
                        "kinetic_law": "",
                        "reactants": [],
                        "products": [],
                        "modifiers": [],
                        "annotation": "",
                        "ontology": {},
                    }
                )

        # 提取 parameter 标签
        parameters: list[dict[str, Any]] = []
        for m in re.finditer(
            r"<parameter\b([^/>]*?)/?>", sbml_content, re.IGNORECASE | re.DOTALL
        ):
            attrs = m.group(1)
            p_id = self._extract_xml_attr(attrs, "id")
            p_name = self._extract_xml_attr(attrs, "name") or p_id
            p_value_str = self._extract_xml_attr(attrs, "value")
            try:
                p_value = float(p_value_str) if p_value_str else None
            except (TypeError, ValueError):
                p_value = None
            if p_id:
                parameters.append(
                    {
                        "id": p_id,
                        "name": p_name,
                        "value": p_value,
                        "units": self._extract_xml_attr(attrs, "units"),
                    }
                )

        return SBMLDocument(
            species=species,
            reactions=reactions,
            parameters=parameters,
            compartments=[],
            integrity=False,
            parser_backend="regex",
            warnings=["regex fallback: XML parse failed, partial extraction only"],
        )

    @staticmethod
    def _extract_xml_attr(attrs_str: str, attr_name: str) -> str:
        """从 XML 属性字符串中提取指定属性值（正则兜底用）。"""
        m = re.search(
            rf'\b{attr_name}\s*=\s*"([^"]*)"',
            attrs_str,
            re.IGNORECASE,
        )
        return m.group(1) if m else ""

    # -------------------------------------------------------------------------
    # 公共构建逻辑（lxml / ElementTree 共用）
    # -------------------------------------------------------------------------
    def _build_document(self, root: Any, backend: str) -> SBMLDocument:
        """从 XML root 构建完整 SBMLDocument（lxml / ElementTree 共用）。"""
        ns = _detect_namespace(root.tag)

        # SBML level/version 探测
        level = root.get("level", "")
        version = root.get("version", "")

        # model 元素（SBML root → model）
        model = root.find(_qname(ns, "model"))
        if model is None:
            # 部分文档可能直接以 model 为 root（非标准但容错）
            if root.tag.endswith("model") or "model" in root.tag:
                model = root
            else:
                return SBMLDocument(
                    integrity=False,
                    parser_backend=backend,
                    warnings=["no <model> element found"],
                )

        model_id = model.get("id", "") or model.get("metaid", "")

        compartments = self.extract_compartments(model, ns)
        species = self.extract_species(model, ns)
        reactions = self.extract_reactions(model, ns)
        parameters = self.extract_parameters(model, ns)

        warnings: list[str] = []
        integrity = True
        if not species and not reactions:
            warnings.append("no species or reactions found in model")
            integrity = False

        return SBMLDocument(
            species=species,
            reactions=reactions,
            parameters=parameters,
            compartments=compartments,
            integrity=integrity,
            level=level,
            version=version,
            model_id=model_id,
            parser_backend=backend,
            warnings=warnings,
        )

    # -------------------------------------------------------------------------
    # 元素提取接口（公开，供外部直接调用）
    # -------------------------------------------------------------------------
    def extract_species(self, model: Any, ns: str = "") -> list[dict[str, Any]]:
        """提取 model 下所有 species 元素。

        每个 species dict：
            {id, name, compartment, metaid, annotation, ontology, initial_concentration}
        """
        species_list: list[dict[str, Any]] = []
        list_of_species = model.find(_qname(ns, "listOfSpecies"))
        if list_of_species is None:
            return species_list

        for sp_elem in list_of_species.findall(_qname(ns, "species")):
            try:
                sp_id = sp_elem.get("id", "")
                if not sp_id:
                    continue
                annotation_data = self.extract_annotations(sp_elem)
                try:
                    init_conc = float(sp_elem.get("initialConcentration", "0") or "0")
                except (TypeError, ValueError):
                    init_conc = 0.0
                species_list.append(
                    {
                        "id": sp_id,
                        "name": sp_elem.get("name", "") or sp_id,
                        "compartment": sp_elem.get("compartment", ""),
                        "metaid": sp_elem.get("metaid", ""),
                        "initial_concentration": init_conc,
                        "annotation": annotation_data.get("raw", ""),
                        "ontology": annotation_data.get("ontology", {}),
                    }
                )
            except Exception as exc:
                logger.warning("species 解析失败 (id=%s): %s", sp_elem.get("id", "?"), exc)
        return species_list

    def extract_reactions(self, model: Any, ns: str = "") -> list[dict[str, Any]]:
        """提取 model 下所有 reaction 元素（含 kineticLaw / reactants / products / modifiers）。

        每个 reaction dict：
            {id, name, kinetic_law, reactants, products, modifiers, annotation, ontology}
        """
        reactions: list[dict[str, Any]] = []
        list_of_reactions = model.find(_qname(ns, "listOfReactions"))
        if list_of_reactions is None:
            return reactions

        for rxn_elem in list_of_reactions.findall(_qname(ns, "reaction")):
            try:
                rxn_id = rxn_elem.get("id", "")
                if not rxn_id:
                    continue
                reactants = self._extract_species_refs(rxn_elem, ns, "listOfReactants", "reactant")
                products = self._extract_species_refs(rxn_elem, ns, "listOfProducts", "product")
                modifiers = self._extract_species_refs(rxn_elem, ns, "listOfModifiers", "modifier")
                kinetic_law = self._extract_kinetic_law(rxn_elem, ns)
                annotation_data = self.extract_annotations(rxn_elem)
                reactions.append(
                    {
                        "id": rxn_id,
                        "name": rxn_elem.get("name", "") or rxn_id,
                        "reversible": rxn_elem.get("reversible", "false").lower() == "true",
                        "kinetic_law": kinetic_law,
                        "reactants": reactants,
                        "products": products,
                        "modifiers": modifiers,
                        "annotation": annotation_data.get("raw", ""),
                        "ontology": annotation_data.get("ontology", {}),
                    }
                )
            except Exception as exc:
                logger.warning("reaction 解析失败 (id=%s): %s", rxn_elem.get("id", "?"), exc)
        return reactions

    def _extract_species_refs(
        self, rxn_elem: Any, ns: str, list_tag: str, role: str
    ) -> list[dict[str, str]]:
        """提取 reaction 下的 reactants/products/modifiers 引用列表。

        SBML 规范：listOfReactants/listOfProducts/listOfModifiers 下的元素
        均为 <speciesReference>（统一元素名），role 由 list_tag 决定。

        Args:
            rxn_elem: reaction XML 元素
            ns: SBML namespace
            list_tag: listOfReactants / listOfProducts / listOfModifiers
            role: reactant / product / modifier（用于结果标记）
        """
        result: list[dict[str, str]] = []
        container = rxn_elem.find(_qname(ns, list_tag))
        if container is None:
            return result
        # SBML L2/L3 规范：所有 species 引用统一使用 speciesReference 元素
        # （modifiers 用 modifierSpeciesReference，但 speciesReference 更常见）
        for item in container.findall(_qname(ns, "speciesReference")):
            species_ref = item.get("species", "")
            if species_ref:
                try:
                    stoich = float(item.get("stoichiometry", "1") or "1")
                except (TypeError, ValueError):
                    stoich = 1.0
                result.append(
                    {
                        "species": species_ref,
                        "stoichiometry": stoich,
                        "role": role,
                    }
                )
        # 兼容 SBML modifiers 用 modifierSpeciesReference 的情况
        if list_tag == "listOfModifiers":
            for item in container.findall(_qname(ns, "modifierSpeciesReference")):
                species_ref = item.get("species", "")
                if species_ref:
                    result.append(
                        {
                            "species": species_ref,
                            "stoichiometry": 1.0,
                            "role": "modifier",
                        }
                    )
        return result

    def _extract_kinetic_law(self, rxn_elem: Any, ns: str) -> str:
        """提取 reaction 的 kineticLaw 数学表达式（math 元素的文本形式）。

        简化处理：直接拼接 math 元素的文本内容（含 CI 元素的变量名）。
        完整 MathML 解析超出本任务范围，仅用于后续 parameter 匹配。
        """
        kinetic_law_elem = rxn_elem.find(_qname(ns, "kineticLaw"))
        if kinetic_law_elem is None:
            return ""
        math_elem = kinetic_law_elem.find(_qname(ns, "math"))
        if math_elem is None:
            # 尝试 MML namespace（SBML L2 的 math 用 http://www.w3.org/1998/Math/MathML）
            for child in kinetic_law_elem:
                if "math" in child.tag.lower():
                    math_elem = child
                    break
        if math_elem is None:
            return ""
        # 提取所有文本与 <ci> 变量名（参数引用）
        text_parts: list[str] = []
        for elem in math_elem.iter():
            tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag_local == "ci" and elem.text:
                text_parts.append(elem.text.strip())
            elif elem.text and elem.text.strip():
                text_parts.append(elem.text.strip())
        return " ".join(text_parts)

    def extract_parameters(self, model: Any, ns: str = "") -> list[dict[str, Any]]:
        """提取 model 下所有 parameter 元素。

        每个 parameter dict：
            {id, name, value, units}
        """
        parameters: list[dict[str, Any]] = []
        # 参数可能位于 model 顶层或 kineticLaw 内部
        # 1. model 层参数（listOfParameters）
        list_of_params = model.find(_qname(ns, "listOfParameters"))
        if list_of_params is not None:
            for p_elem in list_of_params.findall(_qname(ns, "parameter")):
                try:
                    p_id = p_elem.get("id", "")
                    if not p_id:
                        continue
                    try:
                        value = float(p_elem.get("value", "0") or "0")
                    except (TypeError, ValueError):
                        value = None
                    parameters.append(
                        {
                            "id": p_id,
                            "name": p_elem.get("name", "") or p_id,
                            "value": value,
                            "units": p_elem.get("units", ""),
                            "scope": "model",
                        }
                    )
                except Exception as exc:
                    logger.warning("parameter 解析失败 (id=%s): %s", p_elem.get("id", "?"), exc)

        # 2. reaction kineticLaw 内的局部参数（listOfParameters in kineticLaw）
        list_of_reactions = model.find(_qname(ns, "listOfReactions"))
        if list_of_reactions is not None:
            for rxn_elem in list_of_reactions.findall(_qname(ns, "reaction")):
                rxn_id = rxn_elem.get("id", "")
                kinetic_law_elem = rxn_elem.find(_qname(ns, "kineticLaw"))
                if kinetic_law_elem is None:
                    continue
                local_params = kinetic_law_elem.find(_qname(ns, "listOfParameters"))
                if local_params is None:
                    continue
                for p_elem in local_params.findall(_qname(ns, "parameter")):
                    try:
                        p_id = p_elem.get("id", "")
                        if not p_id:
                            continue
                        try:
                            value = float(p_elem.get("value", "0") or "0")
                        except (TypeError, ValueError):
                            value = None
                        parameters.append(
                            {
                                "id": p_id,
                                "name": p_elem.get("name", "") or p_id,
                                "value": value,
                                "units": p_elem.get("units", ""),
                                "scope": "local",
                                "reaction_id": rxn_id,
                            }
                        )
                    except Exception as exc:
                        logger.warning(
                            "local parameter 解析失败 (rxn=%s, id=%s): %s",
                            rxn_id,
                            p_elem.get("id", "?"),
                            exc,
                        )
        return parameters

    def extract_compartments(self, model: Any, ns: str = "") -> list[dict[str, Any]]:
        """提取 model 下所有 compartment 元素。"""
        compartments: list[dict[str, Any]] = []
        list_of_compartments = model.find(_qname(ns, "listOfCompartments"))
        if list_of_compartments is None:
            return compartments
        for comp_elem in list_of_compartments.findall(_qname(ns, "compartment")):
            try:
                comp_id = comp_elem.get("id", "")
                if not comp_id:
                    continue
                try:
                    size = float(comp_elem.get("size", "1") or "1")
                except (TypeError, ValueError):
                    size = 1.0
                compartments.append(
                    {
                        "id": comp_id,
                        "name": comp_elem.get("name", "") or comp_id,
                        "size": size,
                    }
                )
            except Exception as exc:
                logger.warning("compartment 解析失败 (id=%s): %s", comp_elem.get("id", "?"), exc)
        return compartments

    def extract_annotations(self, sbml_element: Any) -> dict[str, Any]:
        """从 SBML 元素的 metaid / annotation 提取 HGNC/UniProt/ChEBI ID。

        SBML annotation 通常采用 RDF/BioModels qualifiers 格式：
            <annotation>
              <rdf:RDF>
                <rdf:Description rdf:about="#metaid">
                  <bqbiol:is>
                    <rdf:Bag>
                      <rdf:li rdf:resource="https://identifiers.org/HGNC:HGNC:3236"/>
                      <rdf:li rdf:resource="https://identifiers.org/uniprot/P00533"/>
                    </rdf:Bag>
                  </bqbiol:is>
                </rdf:Description>
              </rdf:RDF>
            </annotation>

        本方法简化处理：直接对 annotation 元素的所有文本（含 resource URI）做正则提取。

        Returns:
            {raw: str, ontology: {hgnc_id, uniprot_id, chebi_id}}
        """
        result: dict[str, Any] = {"raw": "", "ontology": {}}
        try:
            # annotation 子元素可能带 namespace
            annotation_elem = None
            for child in sbml_element:
                tag_local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if tag_local == "annotation":
                    annotation_elem = child
                    break
            if annotation_elem is None:
                return result

            # 序列化 annotation 为字符串（含所有子元素文本与 resource 属性）
            try:
                import xml.etree.ElementTree as ET

                raw = ET.tostring(annotation_elem, encoding="unicode", method="xml")
            except Exception:
                # 降级：拼接所有文本
                raw = "".join(annotation_elem.itertext()) if hasattr(annotation_elem, "itertext") else ""
            result["raw"] = raw

            ontology: dict[str, str] = {}

            # 提取 HGNC ID
            hgnc = self._extract_hgnc_from_text(raw)
            if hgnc:
                ontology["hgnc_id"] = hgnc

            # 提取 UniProt ID
            uniprot = self._extract_uniprot_from_text(raw)
            if uniprot:
                ontology["uniprot_id"] = uniprot

            # 提取 ChEBI ID
            chebi = self._extract_chebi_from_text(raw)
            if chebi:
                ontology["chebi_id"] = chebi

            result["ontology"] = ontology
            return result
        except Exception as exc:
            logger.debug("annotation 提取失败: %s", exc)
            return result

    # -------------------------------------------------------------------------
    # Ontology ID 正则提取（公开，供 canonical_species.py 复用）
    # -------------------------------------------------------------------------
    def _extract_hgnc_from_text(self, text: str) -> str | None:
        """从文本中提取 HGNC ID（如 'HGNC:HGNC:3236' → 'HGNC:3236'）。"""
        if not text:
            return None
        m = self._HGNC_RE.search(text)
        if m:
            return f"HGNC:{m.group(1)}"
        # 兼容无双重前缀的格式：HGNC:3236（非 identifiers.org 标准）
        m2 = re.search(r"(?<![A-Za-z])HGNC:(\d+)(?!\d)", text)
        if m2:
            return f"HGNC:{m2.group(1)}"
        return None

    def _extract_uniprot_from_text(self, text: str) -> str | None:
        """从文本中提取 UniProt accession（如 'UniProt:P00533' → 'P00533'）。"""
        if not text:
            return None
        # 优先匹配显式 UniProt: 前缀
        m = self._UNIPROT_PREFIX_RE.search(text)
        if m:
            return m.group(1).upper()
        # 降级：匹配 identifiers.org/uniprot/P00533 格式
        m2 = re.search(r"identifiers\.org/uniprot/([A-Z0-9]{6,10})", text, re.IGNORECASE)
        if m2:
            return m2.group(1).upper()
        # 最终兜底：裸 accession 匹配（高 false positive，仅在前缀未命中时启用）
        # 限制为 resource 属性场景：仅匹配 resource="...P00533" 形式
        m3 = re.search(r'resource="[^"]*?([A-NR-Z][0-9][A-Z][A-Z0-9]{2}\d)', text)
        if m3:
            return m3.group(1).upper()
        return None

    def _extract_chebi_from_text(self, text: str) -> str | None:
        """从文本中提取 ChEBI ID（如 'ChEBI:33384' → 'CHEBI:33384'）。"""
        if not text:
            return None
        m = self._CHEBI_RE.search(text)
        if m:
            return f"CHEBI:{m.group(1)}"
        return None


__all__ = ["SBMLDocument", "SBMLParserV2"]
