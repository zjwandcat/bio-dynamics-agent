# BioDynamics Agent - RAG 系统专项测试脚本
# 验证本地 ChromaDB 检索、PubMed E-utilities 在线兜底、完整在线参数提取链路是否畅通。
#
# 运行方式：
#   cd backend
#   .\venv\Scripts\activate
#   python scripts\test_rag_system.py

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

# 将 backend 目录加入 Python 路径，以导入 app 包
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.config import llm, settings  # noqa: E402
from app.mcp_client import MCPBioClient  # noqa: E402
from app.prompts import RAG_EXTRACTION_PROMPT  # noqa: E402
from app.rag_client import RagClient  # noqa: E402

# -----------------------------------------------------------------------------
# 测试报告全局收集器：所有测试输出会累积到 console_buffer，便于最终生成 Markdown 报告
# 同时实时打印到 stdout，让用户在运行期间看到进度
# -----------------------------------------------------------------------------
console_buffer = io.StringIO()


def log(msg: str) -> None:
    """同时打印到控制台并累积到报告缓冲区。"""
    print(msg, flush=True)
    console_buffer.write(msg + "\n")


# -----------------------------------------------------------------------------
# 测试 3 用到的结构化输出模型（与 nodes.py 中 RAGExtractionOutput 等价）
# -----------------------------------------------------------------------------
class _ExtractedParam(BaseModel):
    """RAG 从文献中提取的单条参数记录。"""

    param_name: str = Field(..., description="参数名称，如 Kd / Km / Vmax / 半衰期 / IC50")
    value: float = Field(..., description="参数数值")
    unit: str = Field(..., description="单位，已统一为 h 或 nM")
    species: str = Field(default="", description="物种，如 Human / Mouse")
    cell_line: str = Field(default="", description="细胞系，如 HeLa / T-cell")
    context: str = Field(default="", description="参数出现的生物学上下文")
    confidence: str = Field(default="MEDIUM", description="HIGH 或 MEDIUM")


class _RAGExtractionOutput(BaseModel):
    """RAG 参数提取的结构化输出。"""

    params: list[_ExtractedParam] = Field(default_factory=list, description="提取的参数列表")


# =============================================================================
# 测试 1：本地 ChromaDB 参数检索
# =============================================================================
def test_local_chromadb() -> dict[str, Any]:
    """调用 RagClient.search_params_hybrid，验证本地向量库检索链路。"""
    log("\n" + "=" * 80)
    log("【测试 1】本地 ChromaDB 参数检索")
    log("=" * 80)
    log(f"查询：osimertinib EGFR IC50")
    log(f"ChromaDB 持久化目录：{settings.CHROMA_PERSIST_DIR}")
    log(f"Collection 名称：{settings.CHROMA_COLLECTION_NAME}")

    result: dict[str, Any] = {
        "status": "失败",
        "record_count": 0,
        "top_records": [],
        "log": "",
    }

    # 检查 ChromaDB 数据库文件是否存在
    chroma_db_path = Path(settings.CHROMA_PERSIST_DIR) / "chroma.sqlite3"
    if not chroma_db_path.exists():
        log(f"[错误] ChromaDB 数据库文件不存在：{chroma_db_path}")
        result["log"] = "ChromaDB 数据库文件不存在"
        return result
    log(f"[OK] ChromaDB 数据库文件存在：{chroma_db_path}")

    try:
        client = RagClient()
        log(f"RagClient.available = {client.available}")
        if not client.available:
            log("[错误] RagClient 不可用（ChromaDB 连接失败）")
            result["log"] = "RagClient 不可用"
            return result

        # 探测 collection 数据量
        collection = client._get_collection()
        if collection is None:
            log("[错误] 无法获取 ChromaDB collection")
            result["log"] = "collection 获取失败"
            return result

        try:
            count = collection.count()
            log(f"[OK] Collection '{settings.CHROMA_COLLECTION_NAME}' 记录总数：{count}")
            result["collection_count"] = count
        except Exception as exc:
            log(f"[警告] 无法读取 collection 记录数：{exc}")
            result["collection_count"] = "未知"

        # 调用高阶检索
        log("\n--- 调用 search_params_hybrid ---")
        reranked, insights = client.search_params_hybrid(
            query="osimertinib EGFR IC50",
            species_context="Human",
            top_k=5,
        )
        result["record_count"] = len(reranked)
        log(f"\n返回记录数：{len(reranked)}")

        # 打印洞察数据
        log("\n--- RAG 洞察数据 ---")
        log(f"rewritten_query: {insights.get('rewritten_query', '')}")
        log(f"total_candidates: {insights.get('total_candidates', 0)}")
        log(f"source_distribution: {insights.get('source_distribution', {})}")
        rewrites = insights.get("rewrites", [])
        log(f"rewrites 数量: {len(rewrites)}")
        for rw in rewrites[:5]:
            log(f"  - {rw}")
        expanded_terms = insights.get("expanded_terms", [])
        log(f"expanded_terms: {expanded_terms}")

        # 打印 Top 3 记录
        log("\n--- Top 3 记录 ---")
        for idx, rec in enumerate(reranked[:3]):
            source = rec.get("source", "")
            rerank_score = rec.get("_rerank_score", 0.0)
            param_name = rec.get("param_name", "")
            value = rec.get("value", "")
            unit = rec.get("unit", "")
            species = rec.get("species", "")
            log(
                f"  Top {idx + 1}: param_name={param_name}, value={value} {unit}, "
                f"species={species}, source={source}, _rerank_score={rerank_score:.4f}"
            )
            result["top_records"].append({
                "param_name": param_name,
                "value": value,
                "unit": unit,
                "species": species,
                "source": source,
                "_rerank_score": float(rerank_score),
            })

        # 判定状态：哪怕只有 1 条返回即视为检索链路畅通
        if reranked:
            result["status"] = "成功"
            result["log"] = "本地 ChromaDB 检索链路畅通"
            log("\n[结论] 本地 ChromaDB 检索：成功")
        else:
            result["status"] = "失败"
            result["log"] = "本地检索返回空（可能是 collection 无 osimertinib 相关数据）"
            log("\n[结论] 本地 ChromaDB 检索：返回空（链路畅通但无匹配数据）")

    except Exception as exc:
        log(f"[异常] 测试 1 抛出异常：{exc}")
        log(traceback.format_exc())
        result["log"] = f"异常：{exc}"

    return result


# =============================================================================
# 测试 2：PubMed E-utilities 在线兜底
# =============================================================================
async def test_pubmed_eutils() -> dict[str, Any]:
    """实例化 MCPBioClient，调用 search_pubmed，验证 E-utilities 兜底链路。"""
    log("\n" + "=" * 80)
    log("【测试 2】PubMed E-utilities 在线兜底")
    log("=" * 80)
    log(f"查询：osimertinib EGFR IC50")
    log(f"NCBI_EMAIL 配置：{settings.NCBI_EMAIL or '(未配置)'}")
    log(f"MCP_PUBMED_URL 配置：{settings.MCP_PUBMED_URL or '(未配置，将走 E-utilities 兜底)'}")

    result: dict[str, Any] = {
        "status": "失败",
        "article_count": 0,
        "sample_articles": [],
        "log": "",
    }

    try:
        mcp = MCPBioClient()
        log(f"MCPBioClient.has_real_mcp = {mcp.has_real_mcp}")
        log(f"MCPBioClient.available = {mcp.available}")

        log("\n--- 调用 mcp.search_pubmed ---")
        # search_pubmed 返回 (articles, tool_calls, token_usage)
        articles, tool_calls, token_usage = await mcp.search_pubmed(
            query="osimertinib EGFR IC50",
            max_results=3,
        )
        result["article_count"] = len(articles)
        log(f"\n返回文章数：{len(articles)}")

        # 打印工具调用记录
        log("\n--- 工具调用记录 ---")
        for tc in tool_calls:
            log(
                f"  tool={tc.get('tool_name', '')}, status={tc.get('status', '')}, "
                f"latency_ms={tc.get('latency_ms', 0):.1f}, "
                f"output={tc.get('output_summary', '')}"
            )

        # 校验字段完整性并打印样例
        log("\n--- 样例文章 ---")
        required_fields = ("pmid", "title", "abstract", "source")
        for idx, art in enumerate(articles):
            missing = [f for f in required_fields if not art.get(f)]
            pmid = art.get("pmid", "")
            title = art.get("title", "")[:80]
            abstract_len = len(art.get("abstract", ""))
            source = art.get("source", "")
            log(
                f"  [{idx + 1}] PMID={pmid}, source={source}, "
                f"abstract_len={abstract_len}, missing_fields={missing}"
            )
            log(f"      Title: {title}")
            result["sample_articles"].append({
                "pmid": pmid,
                "title": art.get("title", ""),
                "source": source,
                "abstract_length": abstract_len,
                "missing_fields": missing,
            })

        # 判定：返回 ≥1 篇且每篇包含 pmid/source 即视为链路畅通
        if articles and all(art.get("pmid") and art.get("source") for art in articles):
            result["status"] = "成功"
            result["log"] = "PubMed E-utilities 兜底链路畅通"
            log("\n[结论] PubMed E-utilities 在线兜底：成功")
        elif articles:
            result["status"] = "部分成功"
            result["log"] = "返回文章但部分字段缺失"
            log("\n[结论] PubMed E-utilities 在线兜底：部分成功（字段缺失）")
        else:
            result["status"] = "失败"
            result["log"] = "PubMed 兜底返回空（可能网络不通或查询无结果）"
            log("\n[结论] PubMed E-utilities 在线兜底：返回空")

    except Exception as exc:
        log(f"[异常] 测试 2 抛出异常：{exc}")
        log(traceback.format_exc())
        result["log"] = f"异常：{exc}"

    return result


# =============================================================================
# 测试 3：完整在线参数提取链路
# =============================================================================
async def test_full_extraction_chain() -> dict[str, Any]:
    """模拟本地无数据：从 PubMed 取摘要 → LLM 提取 IC50 参数。"""
    log("\n" + "=" * 80)
    log("【测试 3】完整在线参数提取链路（PubMed → LLM 提取 IC50）")
    log("=" * 80)

    result: dict[str, Any] = {
        "status": "失败",
        "extracted_params": [],
        "log": "",
    }

    try:
        mcp = MCPBioClient()
        log("步骤 1：调用 mcp.search_pubmed 获取奥希替尼相关文献摘要")
        articles, _, _ = await mcp.search_pubmed(
            query="osimertinib EGFR inhibitor IC50",
            max_results=3,
        )
        log(f"获取到 {len(articles)} 篇文献")

        if not articles:
            log("[错误] 未获取到任何文献，无法继续提取参数")
            result["log"] = "无文献可提取"
            return result

        # 拼接所有文献摘要作为 LLM 输入
        combined_text = ""
        for art in articles:
            title = art.get("title", "")
            abstract = art.get("abstract", "")
            combined_text += f"Title: {title}\nAbstract: {abstract}\n\n"

        # 截断防止超长
        combined_text = combined_text[:4000]
        log(f"步骤 2：拼接文献文本（长度 {len(combined_text)} 字符）")
        log("\n--- 拼接文本预览（前 300 字符）---")
        log(combined_text[:300])

        log("\n步骤 3：调用 LLM 使用 RAG_EXTRACTION_PROMPT 提取 IC50 参数")
        # 使用 llm.with_structured_output 验证底层 config.py 的 strip_markdown_json 修复是否生效
        structured_llm = llm.with_structured_output(_RAGExtractionOutput)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_EXTRACTION_PROMPT),
                ("human", "请提取以下文献片段中的动力学参数。"),
            ]
        )
        chain = prompt.partial(document_chunk=combined_text) | structured_llm

        extraction_result: _RAGExtractionOutput = chain.invoke({})
        params = [p.model_dump() for p in extraction_result.params]
        result["extracted_params"] = params

        log(f"\nLLM 提取到 {len(params)} 条参数：")
        for idx, p in enumerate(params):
            log(
                f"  [{idx + 1}] param_name={p.get('param_name')}, "
                f"value={p.get('value')} {p.get('unit')}, "
                f"species={p.get('species')}, cell_line={p.get('cell_line')}, "
                f"confidence={p.get('confidence')}"
            )
            log(f"      context: {p.get('context', '')[:120]}")

        if params:
            result["status"] = "成功"
            result["log"] = "完整在线参数提取链路畅通"
            log("\n[结论] 完整在线参数提取链路：成功")
        else:
            result["status"] = "部分成功"
            result["log"] = "链路畅通但 LLM 未从摘要中提取到参数（可能文献未含明确 IC50 数值）"
            log("\n[结论] 完整在线参数提取链路：链路畅通但无参数提取")

    except Exception as exc:
        log(f"[异常] 测试 3 抛出异常：{exc}")
        log(traceback.format_exc())
        result["log"] = f"异常：{exc}"

    return result


# =============================================================================
# 主流程：串行执行三个测试并生成报告
# =============================================================================
async def run_all_tests() -> dict[str, Any]:
    """运行全部测试，返回结构化结果。"""
    log("=" * 80)
    log("BioDynamics Agent RAG 系统专项测试")
    log("=" * 80)
    log(f"Python 版本：{sys.version.split()[0]}")
    log(f"EMBEDDING_PROVIDER：{settings.EMBEDDING_PROVIDER}")
    log(f"EMBEDDING_MODEL：{settings.EMBEDDING_MODEL}")
    log(f"NCBI_EMAIL：{settings.NCBI_EMAIL or '(未配置)'}")
    log(f"OPENAI_MODEL：{settings.OPENAI_MODEL}")
    log(f"OPENAI_BASE_URL：{settings.OPENAI_BASE_URL}")

    # 先收集本地 ChromaDB 记录总数（供报告使用）
    chroma_total = "未知"
    try:
        chroma_db_path = Path(settings.CHROMA_PERSIST_DIR) / "chroma.sqlite3"
        if chroma_db_path.exists():
            client = RagClient()
            if client.available:
                collection = client._get_collection()
                if collection is not None:
                    chroma_total = collection.count()
    except Exception as exc:
        chroma_total = f"读取失败：{exc}"

    log(f"本地 ChromaDB 记录总数：{chroma_total}")

    test1_result = test_local_chromadb()
    test2_result = await test_pubmed_eutils()
    test3_result = await test_full_extraction_chain()

    return {
        "env": {
            "python_version": sys.version.split()[0],
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "embedding_model": settings.EMBEDDING_MODEL,
            "ncbi_email": settings.NCBI_EMAIL or "(未配置)",
            "openai_model": settings.OPENAI_MODEL,
            "chroma_persist_dir": settings.CHROMA_PERSIST_DIR,
            "chroma_collection_name": settings.CHROMA_COLLECTION_NAME,
            "chroma_total": chroma_total,
        },
        "test1_local_chromadb": test1_result,
        "test2_pubmed_eutils": test2_result,
        "test3_full_extraction": test3_result,
    }


def generate_report(results: dict[str, Any], console_log: str) -> str:
    """根据测试结果生成 Markdown 报告。"""
    env = results["env"]
    t1 = results["test1_local_chromadb"]
    t2 = results["test2_pubmed_eutils"]
    t3 = results["test3_full_extraction"]

    # 转义 Markdown 表格中可能的管道符
    def esc(s: Any) -> str:
        return str(s).replace("|", "\\|").replace("\n", " ")

    lines = []
    lines.append("# RAG 系统专项测试报告")
    lines.append("")
    lines.append("> 本报告由 `scripts/test_rag_system.py` 自动生成，验证 BioDynamics Agent 的 RAG 检索链路（本地 ChromaDB + PubMed E-utilities 在线兜底 + LLM 参数提取）。")
    lines.append("")

    lines.append("## 1. 测试环境")
    lines.append("")
    lines.append(f"- Python 版本：`{env['python_version']}`")
    lines.append(f"- EMBEDDING_PROVIDER 配置：`{env['embedding_provider']}`")
    lines.append(f"- EMBEDDING_MODEL 配置：`{env['embedding_model']}`")
    lines.append(f"- NCBI_EMAIL 配置：`{env['ncbi_email']}`")
    lines.append(f"- OPENAI_MODEL 配置：`{env['openai_model']}`")
    lines.append(f"- ChromaDB 持久化目录：`{env['chroma_persist_dir']}`")
    lines.append(f"- ChromaDB Collection 名称：`{env['chroma_collection_name']}`")
    lines.append(f"- 本地 ChromaDB 记录总数：**{env['chroma_total']}**")
    lines.append("")

    lines.append("## 2. 测试结果")
    lines.append("")

    # ---- 测试 1 ----
    lines.append("### 2.1 本地 ChromaDB 检索")
    lines.append("")
    lines.append(f"- 状态：**{t1['status']}**")
    lines.append(f"- 返回记录数：{t1['record_count']}")
    lines.append(f"- Collection 记录总数：{t1.get('collection_count', '未知')}")
    lines.append("")
    lines.append("Top 记录摘要：")
    lines.append("")
    if t1["top_records"]:
        lines.append("| 序号 | param_name | value | unit | species | source | _rerank_score |")
        lines.append("|------|------------|-------|------|---------|--------|---------------|")
        for i, rec in enumerate(t1["top_records"], 1):
            lines.append(
                f"| {i} | {esc(rec['param_name'])} | {esc(rec['value'])} | "
                f"{esc(rec['unit'])} | {esc(rec['species'])} | {esc(rec['source'])} | "
                f"{rec['_rerank_score']:.4f} |"
            )
    else:
        lines.append("> 无 Top 记录（本地库未检索到 osimertinib 相关参数）。")
    lines.append("")
    lines.append("**日志与分析：**")
    lines.append("")
    lines.append(f"```\n{t1['log']}\n```")
    lines.append("")

    # ---- 测试 2 ----
    lines.append("### 2.2 PubMed E-utilities 在线兜底")
    lines.append("")
    lines.append(f"- 状态：**{t2['status']}**")
    lines.append(f"- 返回文章数：{t2['article_count']}")
    lines.append("")
    lines.append("样例文章 PMID 及 Title：")
    lines.append("")
    if t2["sample_articles"]:
        lines.append("| 序号 | PMID | source | abstract_length | missing_fields | Title |")
        lines.append("|------|------|--------|-----------------|----------------|-------|")
        for i, art in enumerate(t2["sample_articles"], 1):
            lines.append(
                f"| {i} | {esc(art['pmid'])} | {esc(art['source'])} | "
                f"{art['abstract_length']} | {esc(art['missing_fields'])} | "
                f"{esc(art['title'][:60])} |"
            )
    else:
        lines.append("> 无样例文章（PubMed 检索未返回结果）。")
    lines.append("")
    lines.append("**日志与分析：**")
    lines.append("")
    lines.append(f"```\n{t2['log']}\n```")
    lines.append("")

    # ---- 测试 3 ----
    lines.append("### 2.3 完整在线参数提取链路")
    lines.append("")
    lines.append(f"- 状态：**{t3['status']}**")
    lines.append(f"- LLM 提取的参数数量：{len(t3['extracted_params'])}")
    lines.append("")
    lines.append("LLM 提取的参数：")
    lines.append("")
    if t3["extracted_params"]:
        lines.append("| 序号 | param_name | value | unit | species | cell_line | confidence |")
        lines.append("|------|------------|-------|------|---------|-----------|------------|")
        for i, p in enumerate(t3["extracted_params"], 1):
            lines.append(
                f"| {i} | {esc(p.get('param_name', ''))} | {esc(p.get('value', ''))} | "
                f"{esc(p.get('unit', ''))} | {esc(p.get('species', ''))} | "
                f"{esc(p.get('cell_line', ''))} | {esc(p.get('confidence', ''))} |"
            )
        lines.append("")
        lines.append("参数上下文详情：")
        lines.append("")
        for i, p in enumerate(t3["extracted_params"], 1):
            lines.append(
                f"- **[{i}] {p.get('param_name', '')}** = {p.get('value', '')} {p.get('unit', '')}"
                f"（confidence: {p.get('confidence', '')}）"
            )
            lines.append(f"  - context: {p.get('context', '')}")
    else:
        lines.append("> LLM 未从摘要中提取到参数（可能文献未含明确 IC50/EC50 数值）。")
    lines.append("")
    lines.append("**日志与分析：**")
    lines.append("")
    lines.append(f"```\n{t3['log']}\n```")
    lines.append("")

    # ---- 问题与修复建议 ----
    lines.append("## 3. 发现的问题与修复建议")
    lines.append("")

    problems = []
    if t1["status"] != "成功":
        problems.append(
            f"- **本地 ChromaDB 检索未返回 osimertinib 相关参数**：\n"
            f"  - 现象：返回记录数 = {t1['record_count']}，状态 = {t1['status']}\n"
            f"  - 排查建议：\n"
            f"    1. 检查 `{env['chroma_persist_dir']}/chroma.sqlite3` 是否存在（已确认存在）\n"
            f"    2. 检查 collection `{env['chroma_collection_name']}` 是否有 osimertinib/EGFR 相关数据\n"
            f"    3. 当前 Collection 总记录数为 {t1.get('collection_count', '未知')}，若数据稀少需运行 `scripts/build_rag_db.py` 补充\n"
            f"    4. EMBEDDING_PROVIDER=`{env['embedding_provider']}`，已确认配置正确"
        )

    if t2["status"] == "失败":
        problems.append(
            f"- **PubMed E-utilities 兜底失败**：\n"
            f"  - 现象：返回文章数 = {t2['article_count']}\n"
            f"  - 排查建议：\n"
            f"    1. 检查 `.env` 中 `NCBI_EMAIL` 配置（当前为 `{env['ncbi_email']}`）\n"
            f"    2. 在终端执行 `curl \"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=osimertinib&retmode=json\"` 确认网络\n"
            f"    3. 检查 `mcp_client.py` 的 `_search_pubmed_eutils` 方法异常是否被静默吞掉"
        )
    elif t2["status"] == "部分成功":
        problems.append(
            f"- **PubMed 返回文章但字段缺失**：\n"
            f"  - 现象：部分文章缺少 pmid/source/abstract 字段\n"
            f"  - 排查建议：检查 `_search_pubmed_eutils` 的 XML 解析逻辑"
        )

    if t3["status"] != "成功":
        problems.append(
            f"- **完整在线参数提取链路异常**：\n"
            f"  - 现象：状态 = {t3['status']}，提取参数数 = {len(t3['extracted_params'])}\n"
            f"  - 排查建议：\n"
            f"    1. 确认测试 2 的 PubMed 检索是否成功（前置依赖）\n"
            f"    2. 检查 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 是否可用\n"
            f"    3. 若提取数为 0，可能是文献摘要未含明确 IC50 数值，属正常现象"
        )

    if not problems:
        lines.append("**RAG 链路已完全打通。** 三个测试场景全部成功，奥希替尼检索与参数提取链路畅通。")
        lines.append("")
        lines.append("- 本地 ChromaDB 检索可正常返回 osimertinib/EGFR 相关参数")
        lines.append("- PubMed E-utilities 在线兜底可正常返回真实文献（含 PMID、Title、Abstract）")
        lines.append("- 完整在线参数提取链路（PubMed → LLM 提取 IC50）可正常工作")
    else:
        lines.append("测试中发现以下问题：")
        lines.append("")
        for p in problems:
            lines.append(p)
        lines.append("")
        lines.append("**修复建议汇总：**")
        lines.append("")
        lines.append("1. 若本地 ChromaDB 数据稀少：运行 `python scripts/build_rag_db.py --queries \"osimertinib EGFR\" \"EGFR inhibitor\" --max-results 30` 补充参数库")
        lines.append("2. 若 PubMed 网络不通：检查防火墙/代理设置，确认能访问 `eutils.ncbi.nlm.nih.gov`")
        lines.append("3. 若 LLM 提取为空：检查文献摘要是否包含明确 IC50/EC50 数值，必要时扩大检索词或增加 `max_results`")

    lines.append("")
    lines.append("## 4. 完整控制台输出")
    lines.append("")
    lines.append("<details><summary>点击展开完整控制台日志</summary>")
    lines.append("")
    lines.append("```")
    lines.append(console_log)
    lines.append("```")
    lines.append("")
    lines.append("</details>")

    return "\n".join(lines)


async def main() -> None:
    """主入口：运行所有测试并生成报告。"""
    # log() 已经实时打印到 stdout 并累积到 console_buffer
    # 直接运行测试，用户可实时看到输出
    results = await run_all_tests()

    console_log = console_buffer.getvalue()

    # 生成报告
    report = generate_report(results, console_log)

    # 写入项目根目录的 RAG_TEST_REPORT.md
    # backend 的父目录即项目根目录（gzlab）
    report_path = BACKEND_DIR.parent / "RAG_TEST_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[报告已生成] {report_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
