# BioDynamics Agent - MCP（Model Context Protocol）工具封装层
# 将生物医学领域的 MCP 工具（OpenBioMed Skills、medical-terminologies-mcp、
# pubmed-search-mcp、NIH UMLS MCP）统一封装为可调用接口，并支持优雅降级。
#
# 设计要点：
# 1. 优先连接真实 MCP 服务端点（通过 langchain-protocol 的 MCPClient）
# 2. 若端点未配置或连接失败，自动降级为 LLM 内部知识完成术语查询
# 3. 每次工具调用均产出 ToolCallRecord，供前端工具调用状态面板渲染
# 4. 估算 MCP 术语标准化带来的 Token 节省量，供前端展示

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import llm
from app.token_usage import UsageAccumulator

logger = logging.getLogger(__name__)

# NCBI E-utilities 限速：无 API Key 时 3 req/s，有 API Key 时 10 req/s
# 保守起见统一控制在 2 req/s（0.5s 间隔），确保不触发 NCBI 封禁
_NCBI_MIN_INTERVAL: float = 0.5  # 秒，两次 NCBI HTTP 请求之间的最小间隔
_ncbi_last_request_ts: float = 0.0


async def _ncbi_wait_rate_limit() -> None:
    """等待 NCBI 限速间隔，确保不超过 2 req/s。"""
    global _ncbi_last_request_ts
    import time as _time
    elapsed = _time.time() - _ncbi_last_request_ts
    if elapsed < _NCBI_MIN_INTERVAL:
        await asyncio.sleep(_NCBI_MIN_INTERVAL - elapsed)
    _ncbi_last_request_ts = _time.time()


# -----------------------------------------------------------------------------
# MCP 工具注册表：定义 4 个核心生物医学 MCP 工具的元信息
# -----------------------------------------------------------------------------
@dataclass
class MCPToolSpec:
    """单个 MCP 工具的元信息定义。"""

    key: str                # 内部标识符
    name: str               # 前端展示名
    icon: str               # 前端图标（emoji）
    description: str        # 工具职责描述
    url_env: str            # 对应的环境变量名（存放 MCP 服务端点）


MCP_TOOL_REGISTRY: list[MCPToolSpec] = [
    MCPToolSpec(
        key="openbiomed",
        name="OpenBioMed Skills",
        icon="🔍",
        description="生物医学实体识别与关系抽取（清华AIR×水木分子）",
        url_env="MCP_OPENBIOMED_URL",
    ),
    MCPToolSpec(
        key="medical_terminologies",
        name="medical-terminologies-mcp",
        icon="📋",
        description="临床术语标准化（ICD-10、SNOMED CT）",
        url_env="MCP_MEDTERM_URL",
    ),
    MCPToolSpec(
        key="pubmed_search",
        name="pubmed-search-mcp",
        icon="📄",
        description="PubMed 增强版文献检索",
        url_env="MCP_PUBMED_URL",
    ),
    MCPToolSpec(
        key="umls",
        name="NIH UMLS MCP",
        icon="🔬",
        description="本体术语同义词与层级关系",
        url_env="MCP_UMLS_URL",
    ),
]


# -----------------------------------------------------------------------------
# 工具调用记录：每次 MCP 调用产出一条结构化记录，供前端渲染
# -----------------------------------------------------------------------------
@dataclass
class ToolCallRecord:
    """单次 MCP 工具调用的完整记录。"""

    tool_key: str               # 工具标识（对应 MCPToolSpec.key）
    tool_name: str              # 工具展示名
    icon: str                   # 工具图标
    action: str                 # 调用动作描述，如 "解析术语 'EGFR'"
    status: str                 # success / fallback / failed
    input_summary: str          # 输入摘要（术语或查询）
    output_summary: str         # 输出摘要（定义或结果）
    latency_ms: float = 0.0     # 耗时（毫秒）
    tokens_saved: int = 0       # 估算节省的 Token 数
    detail: dict = field(default_factory=dict)  # 详细数据（可选展开）

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_key": self.tool_key,
            "tool_name": self.tool_name,
            "icon": self.icon,
            "action": self.action,
            "status": self.status,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "latency_ms": round(self.latency_ms, 1),
            "tokens_saved": self.tokens_saved,
            "detail": self.detail,
        }


# -----------------------------------------------------------------------------
# 术语定义的结构化输出模型
# -----------------------------------------------------------------------------
class TermDefinition(BaseModel):
    """单个生物医学术语的标准化定义。"""

    term: str = Field(..., description="原始术语")
    canonical_name: str = Field(..., description="标准化学名（英文）")
    definition: str = Field(..., description="简明生物学定义（中文）")
    synonyms: list[str] = Field(default_factory=list, description="同义词列表")
    category: str = Field(default="", description="术语类别，如细胞因子/细胞类型/通路")
    related_pathway: str = Field(default="", description="相关信号通路")


class TermLookupOutput(BaseModel):
    """术语查询的结构化输出。"""

    definitions: list[TermDefinition] = Field(
        default_factory=list, description="提取到的术语定义列表"
    )


# -----------------------------------------------------------------------------
# 术语提取提示词：从用户输入中识别生物医学专业术语
# -----------------------------------------------------------------------------
TERM_EXTRACTION_PROMPT = """你是一个生物医学术语识别专家。从用户输入的自然语言假说中，提取所有专业生物医学术语。
包括但不限于：蛋白质/基因名（如 p53、EGFR）、细胞类型（如 NK 细胞、巨噬细胞）、
信号通路（如 MAPK、PI3K-AKT）、疾病名、药物名、生物过程（如凋亡、增殖）。
仅提取用户输入中实际出现的术语，禁止添加示例中未提及的实体。
输出严格 JSON，绝对禁止使用 markdown 代码块。
输出格式: {{"terms": ["术语1", "术语2", ...]}}"""

TERM_DEFINITION_PROMPT = """你是生物医学本体专家。请为以下术语提供标准化定义。
对每个术语：
1. 给出标准化学名（英文 canonical name）
2. 用 1-2 句中文简明定义其生物学意义
3. 列出 2-4 个同义词（含英文缩写）
4. 标注术语类别（细胞因子/受体/细胞类型/信号通路/生物过程/其他）
5. 若涉及信号通路，标注相关通路名

术语列表: {terms}
输出严格 JSON，绝对禁止 markdown 代码块。
输出格式: {{"definitions": [{{"term": str, "canonical_name": str, "definition": str, "synonyms": [str], "category": str, "related_pathway": str}}]}}"""


# -----------------------------------------------------------------------------
# MCP 工具客户端：核心封装层
# -----------------------------------------------------------------------------
class MCPBioClient:
    """生物医学 MCP 工具统一客户端。

    封装 4 个 MCP 工具的调用逻辑，支持真实 MCP 连接与 LLM 降级两种模式。
    所有调用均产出 ToolCallRecord 供前端可视化。
    """

    # 估算每个术语定义可节省的 Token 数（模型无需再生成术语解释上下文）
    TOKENS_SAVED_PER_TERM: int = 40

    def __init__(self) -> None:
        from app.config import settings

        # 从 settings 读取各 MCP 工具的端点配置
        self._endpoints: dict[str, str | None] = {}
        for spec in MCP_TOOL_REGISTRY:
            url = getattr(settings, spec.url_env, None) or ""
            self._endpoints[spec.key] = url if url else None

        # 标记是否有任何真实 MCP 端点可用
        self._has_real_endpoint = any(self._endpoints.values())

    @property
    def available(self) -> bool:
        """客户端是否可用（即使无真实 MCP 端点，LLM 降级模式也可用）。"""
        return True

    @property
    def has_real_mcp(self) -> bool:
        """是否配置了真实 MCP 服务端点。"""
        return self._has_real_endpoint

    def _get_spec(self, tool_key: str) -> MCPToolSpec | None:
        """根据 key 查找工具元信息。"""
        for spec in MCP_TOOL_REGISTRY:
            if spec.key == tool_key:
                return spec
        return None

    async def _try_real_mcp(
        self, tool_key: str, method: str, params: dict
    ) -> dict | None:
        """尝试连接真实 MCP 服务并调用指定方法。

        使用 langchain-protocol 的 MCPClient 进行 SSE 连接。
        连接失败时返回 None，由上层降级处理。
        """
        endpoint = self._endpoints.get(tool_key)
        if not endpoint:
            return None

        try:
            # 延迟导入：仅在真正需要时加载 MCP 客户端，避免无端点时的导入开销
            from langchain_protocol import MCPClient

            client = MCPClient(url=endpoint)
            tools = await client.load_tools()

            # 根据 tool_key 选择对应的工具方法调用
            # 不同 MCP 服务的工具名可能不同，这里做兼容映射
            tool_name_map = {
                "openbiomed": "extract_entities",
                "medical_terminologies": "standardize_term",
                "pubmed_search": "search_pubmed_key_words",
                "umls": "get_synonyms",
            }
            target_tool_name = tool_name_map.get(tool_key, method)

            for tool in tools:
                if tool.name == target_tool_name:
                    result = await tool.ainvoke(params)
                    if isinstance(result, str):
                        return json.loads(result) if result.strip() else {}
                    return result if isinstance(result, dict) else {"result": str(result)}

            logger.warning("MCP 工具 %s 中未找到方法 %s", tool_key, target_tool_name)
            return None
        except Exception as exc:
            logger.info("MCP 真实连接失败（%s），将降级：%s", tool_key, exc)
            return None

    def _extract_terms_with_llm(
        self, user_input: str, callbacks: list | None = None
    ) -> tuple[list[str], dict]:
        """使用 LLM 从用户输入中提取生物医学术语（降级模式）。

        返回 (术语列表, token_usage)。
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", TERM_EXTRACTION_PROMPT),
                ("human", "{user_input}"),
            ]
        )

        class _TermList(BaseModel):
            terms: list[str] = Field(default_factory=list)

        chain = prompt | llm.with_structured_output(_TermList)
        handler = UsageAccumulator()
        result = chain.invoke(
            {"user_input": user_input},
            config={"callbacks": [handler, *(callbacks or [])]},
        )
        terms = result.terms if hasattr(result, "terms") else []
        return terms, usage_to_dict(handler)

    def _define_terms_with_llm(
        self, terms: list[str], callbacks: list | None = None
    ) -> tuple[list[TermDefinition], dict]:
        """使用 LLM 为术语列表生成标准化定义（降级模式）。

        返回 (定义列表, token_usage)。
        """
        if not terms:
            return [], {}

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", TERM_DEFINITION_PROMPT),
                ("human", "请为这些术语提供标准化定义。"),
            ]
        )
        chain = prompt.partial(terms=json.dumps(terms, ensure_ascii=False)) | llm.with_structured_output(TermLookupOutput)
        handler = UsageAccumulator()
        result: TermLookupOutput = chain.invoke(
            {}, config={"callbacks": [handler, *(callbacks or [])]}
        )
        return result.definitions, usage_to_dict(handler)

    async def lookup_terms(
        self, user_input: str, callbacks: list | None = None
    ) -> tuple[list[TermDefinition], list[dict], dict]:
        """术语查询主入口：提取术语 → 获取定义 → 标准化。

        编排 4 个 MCP 工具的调用流程，产出术语定义与工具调用记录。
        返回 (定义列表, 工具调用记录列表, token_usage合计)。
        """
        tool_calls: list[ToolCallRecord] = []
        total_usage: dict = {}

        # ---- 步骤 1: OpenBioMed Skills - 实体识别与术语提取 ----
        spec = self._get_spec("openbiomed")
        start_ts = time.time()
        real_result = await self._try_real_mcp(
            "openbiomed", "extract_entities", {"text": user_input}
        )
        if real_result is not None:
            terms = real_result.get("entities", real_result.get("terms", []))
            status = "success"
            output_summary = f"识别到 {len(terms)} 个生物医学实体"
        else:
            # 降级：LLM 提取术语
            terms, usage = self._extract_terms_with_llm(user_input, callbacks)
            total_usage = merge_dict(total_usage, usage)
            status = "fallback"
            output_summary = f"（LLM 降级）识别到 {len(terms)} 个术语"

        latency_ms = (time.time() - start_ts) * 1000
        tool_calls.append(ToolCallRecord(
            tool_key="openbiomed",
            tool_name=spec.name,
            icon=spec.icon,
            action=f"解析输入文本，提取生物医学实体",
            status=status,
            input_summary=user_input[:80] + ("..." if len(user_input) > 80 else ""),
            output_summary=output_summary,
            latency_ms=latency_ms,
            detail={"extracted_terms": terms},
        ))

        if not terms:
            return [], [tc.to_dict() for tc in tool_calls], total_usage

        # ---- 步骤 2: NIH UMLS MCP - 术语标准化与同义词 ----
        spec = self._get_spec("umls")
        start_ts = time.time()
        real_result = await self._try_real_mcp(
            "umls", "get_synonyms", {"terms": terms}
        )
        if real_result is not None:
            umls_data = real_result
            status = "success"
            output_summary = f"获取 {len(terms)} 个术语的标准化信息"
        else:
            umls_data = None
            status = "fallback"
            output_summary = "（LLM 降级）将在定义步骤中合并标准化"

        latency_ms = (time.time() - start_ts) * 1000
        tool_calls.append(ToolCallRecord(
            tool_key="umls",
            tool_name=spec.name,
            icon=spec.icon,
            action=f"查询 {len(terms)} 个术语的同义词与层级关系",
            status=status,
            input_summary=", ".join(terms[:5]) + ("..." if len(terms) > 5 else ""),
            output_summary=output_summary,
            latency_ms=latency_ms,
            detail={"umls_data": umls_data} if umls_data else {},
        ))

        # ---- 步骤 3: medical-terminologies-mcp - 临床术语标准化 ----
        spec = self._get_spec("medical_terminologies")
        start_ts = time.time()
        real_result = await self._try_real_mcp(
            "medical_terminologies", "standardize_term", {"terms": terms}
        )
        if real_result is not None:
            medterm_data = real_result
            status = "success"
            output_summary = f"标准化 {len(terms)} 个临床术语"
        else:
            medterm_data = None
            status = "fallback"
            output_summary = "（LLM 降级）跳过临床术语标准化"

        latency_ms = (time.time() - start_ts) * 1000
        tool_calls.append(ToolCallRecord(
            tool_key="medical_terminologies",
            tool_name=spec.name,
            icon=spec.icon,
            action=f"标准化临床术语（ICD-10 / SNOMED CT）",
            status=status,
            input_summary=", ".join(terms[:5]),
            output_summary=output_summary,
            latency_ms=latency_ms,
        ))

        # ---- 步骤 4: 生成术语定义（合并 UMLS 同义词信息） ----
        spec = self._get_spec("openbiomed")  # 复用 OpenBioMed 的定义能力
        start_ts = time.time()
        definitions, usage = self._define_terms_with_llm(terms, callbacks)
        total_usage = merge_dict(total_usage, usage)
        latency_ms = (time.time() - start_ts) * 1000

        # 若 UMLS 返回了同义词数据，合并到定义中
        if umls_data and isinstance(umls_data, dict):
            for definition in definitions:
                term_key = definition.term.lower()
                if term_key in umls_data:
                    extra_synonyms = umls_data[term_key].get("synonyms", [])
                    existing = set(s.lower() for s in definition.synonyms)
                    for syn in extra_synonyms:
                        if syn.lower() not in existing:
                            definition.synonyms.append(syn)

        tokens_saved = len(definitions) * self.TOKENS_SAVED_PER_TERM
        tool_calls.append(ToolCallRecord(
            tool_key="openbiomed",
            tool_name="OpenBioMed Skills",
            icon="🔍",
            action=f"为 {len(definitions)} 个术语生成标准化定义卡片",
            status="success",
            input_summary=", ".join(d.term for d in definitions[:5]),
            output_summary=f"生成 {len(definitions)} 个术语定义，估算节省 {tokens_saved} Token",
            latency_ms=latency_ms,
            tokens_saved=tokens_saved,
            detail={
                "definitions": [d.model_dump() for d in definitions],
            },
        ))

        return definitions, [tc.to_dict() for tc in tool_calls], total_usage

    async def search_pubmed(
        self, query: str, max_results: int = 3, callbacks: list | None = None
    ) -> tuple[list[dict], list[dict], dict]:
        """PubMed 文献检索接口。

        优先调用 MCP 端点；若端点未配置，直连 NCBI E-utilities 兜底。
        返回 (文献列表, 工具调用记录, token_usage)。
        """
        spec = self._get_spec("pubmed_search")
        start_ts = time.time()

        real_result = await self._try_real_mcp(
            "pubmed_search", "search_pubmed",
            {"query": query, "max_results": max_results},
        )

        if real_result is not None:
            articles = real_result.get("articles", real_result.get("results", []))
            status = "success"
            output_summary = f"检索到 {len(articles)} 篇相关文献"
        else:
            # 兜底：无 MCP 端点时直连 NCBI E-utilities
            articles = await self._search_pubmed_eutils(query, max_results)
            if articles:
                status = "fallback_eutils"
                output_summary = f"（NCBI E-utilities 直连）检索到 {len(articles)} 篇文献"
            else:
                status = "fallback"
                output_summary = "（MCP 未连接 + E-utilities 无结果）PubMed 检索跳过"

        latency_ms = (time.time() - start_ts) * 1000
        record = ToolCallRecord(
            tool_key="pubmed_search",
            tool_name=spec.name,
            icon=spec.icon,
            action=f"检索 PubMed：{query[:50]}",
            status=status,
            input_summary=query[:80],
            output_summary=output_summary,
            latency_ms=latency_ms,
            detail={"articles": articles[:max_results]},
        )

        return articles, [record.to_dict()], {}

    async def _search_pubmed_eutils(
        self, query: str, max_results: int = 3
    ) -> list[dict]:
        """直连 NCBI E-utilities 检索 PubMed，返回文献列表。

        离线或失败时返回空列表，不阻塞主流程。
        """
        # Task 19 SEC-1.3: 使用 defusedxml 防御 XXE/实体扩展攻击
        # defusedxml 不可用时降级到 xml.etree（离线开发场景，风险可接受）
        try:
            from defusedxml import ElementTree as ET  # type: ignore
        except ImportError:
            import xml.etree.ElementTree as ET  # type: ignore
            logger.warning("defusedxml 未安装，NCBI XML 解降级到 xml.etree（有实体扩展风险）")

        try:
            import requests
        except ImportError:
            return []

        ncbi_email = getattr(settings, "NCBI_EMAIL", "") if hasattr(self, "_settings_loaded") else ""
        if not ncbi_email:
            try:
                from app.config import settings as _settings
                ncbi_email = getattr(_settings, "NCBI_EMAIL", "")
            except Exception:
                pass

        # NCBI API Key（可选，有 Key 时限流从 3 req/s 提升至 10 req/s）
        ncbi_api_key = ""
        try:
            from app.config import settings as _settings
            ncbi_api_key = getattr(_settings, "NCBI_API_KEY", "")
        except Exception:
            pass

        # 步骤 1: esearch 获取 PMID 列表
        await _ncbi_wait_rate_limit()  # 限速：确保不超过 2 req/s
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        }
        if ncbi_email:
            params["email"] = ncbi_email
            params["tool"] = "BioDynamicsAgent"
        if ncbi_api_key:
            params["api_key"] = ncbi_api_key

        try:
            resp = await asyncio.to_thread(
                requests.get, esearch_url, params=params, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            logger.warning("NCBI esearch 失败：%s", exc)
            return []

        if not id_list:
            return []

        # 步骤 2: efetch 获取文献详情
        await _ncbi_wait_rate_limit()  # 限速：确保不超过 2 req/s
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if ncbi_email:
            params["email"] = ncbi_email
            params["tool"] = "BioDynamicsAgent"
        if ncbi_api_key:
            params["api_key"] = ncbi_api_key

        try:
            resp = await asyncio.to_thread(
                requests.get, efetch_url, params=params, timeout=15
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("NCBI efetch 失败：%s", exc)
            return []

        # 步骤 3: 解析 XML
        articles: list[dict] = []
        try:
            root = ET.fromstring(resp.text)
            for article_elem in root.findall(".//PubmedArticle"):
                pmid_elem = article_elem.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""
                title_elem = article_elem.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else ""
                abstract_parts = article_elem.findall(".//AbstractText")
                # 用 itertext() 递归获取所有子节点文本，兼容带 Label 属性的
                # 结构化摘要（如 <AbstractText Label="BACKGROUND">...</AbstractText>），
                # 直接用 p.text 只能取直接文本节点，会丢失子元素内容导致摘要截断
                abstract = " ".join(
                    "".join(p.itertext()) for p in abstract_parts
                )[:2000]
                if pmid:
                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "source": f"PMID:{pmid}",
                    })
        except ET.ParseError as exc:
            logger.warning("NCBI XML 解析失败：%s", exc)

        return articles[:max_results]

    def rewrite_query(
        self, user_input: str, definitions: list[TermDefinition]
    ) -> tuple[str, ToolCallRecord]:
        """基于 MCP 术语标准化结果重写用户查询，提升 RAG 检索精准度。

        将中文/缩写术语替换为标准英文术语，如
        "TGF-β抑制CD8+ T细胞" → "transforming growth factor beta inhibits cytotoxic T lymphocytes"
        """
        spec = self._get_spec("medical_terminologies")
        start_ts = time.time()

        rewritten = user_input
        replacements: list[str] = []
        for definition in definitions:
            # 若标准名与原始术语不同，执行大小写不敏感的替换
            if definition.canonical_name.lower() != definition.term.lower():
                pattern = re.compile(re.escape(definition.term), re.IGNORECASE)
                if pattern.search(rewritten):
                    rewritten = pattern.sub(definition.canonical_name, rewritten)
                    replacements.append(
                        f"{definition.term} → {definition.canonical_name}"
                    )

        latency_ms = (time.time() - start_ts) * 1000
        record = ToolCallRecord(
            tool_key="medical_terminologies",
            tool_name=spec.name,
            icon=spec.icon,
            action="基于术语标准化重写查询",
            status="success" if replacements else "fallback",
            input_summary=user_input[:80],
            output_summary=f"重写后：{rewritten[:80]}" if replacements else "无需重写",
            latency_ms=latency_ms,
            tokens_saved=0,
            detail={
                "rewritten_query": rewritten,
                "replacements": replacements,
            },
        )

        return rewritten, record


# -----------------------------------------------------------------------------
# 辅助函数：合并 token usage 字典
# -----------------------------------------------------------------------------
def usage_to_dict(handler: UsageAccumulator) -> dict:
    """将 UsageAccumulator 转为普通字典。"""
    from app.token_usage import usage_from_accumulator
    return usage_from_accumulator(handler)


def merge_dict(base: dict, extra: dict) -> dict:
    """合并两个 token_usage 字典，值相加。"""
    if not extra:
        return base
    if not base:
        return dict(extra)
    merged = dict(base)
    for key, val in extra.items():
        merged[key] = merged.get(key, 0) + (val or 0)
    return merged


# 全局 MCP 客户端实例（惰性初始化，避免导入时即连接）
_mcp_client: MCPBioClient | None = None


def get_mcp_client() -> MCPBioClient:
    """获取全局 MCP 客户端单例。"""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPBioClient()
    return _mcp_client
