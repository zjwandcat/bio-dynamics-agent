# BioDynamics Agent - 全局配置模块
# 负责从环境变量加载配置，并初始化全局 LLM 客户端。

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)

# 加载 .env 文件（如果存在）
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    """应用配置，所有敏感信息均来自环境变量。"""

    # OpenAI 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # 备用 LLM 配置
    BACKUP_API_KEY: str = os.getenv("BACKUP_API_KEY", "")
    BACKUP_BASE_URL: str = os.getenv("BACKUP_BASE_URL", "")
    BACKUP_MODEL: str = os.getenv("BACKUP_MODEL", "")

    # 服务配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # CORS 配置
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # ChromaDB 向量库配置（本地持久化，无需 Docker）
    _chroma_persist_dir_raw: str = os.getenv(
        "CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "vector_db")
    )
    CHROMA_PERSIST_DIR: str = str(
        Path(_chroma_persist_dir_raw)
        if Path(_chroma_persist_dir_raw).is_absolute()
        else BASE_DIR / _chroma_persist_dir_raw
    )
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "biodynamics_params")

    # v2 升级：四路 RAG 拆分（Mechanism / Parameter / Experiment / Evidence）
    # 留空时自动以默认名创建；运行 scripts/seed_collections.py 从历史 collection 灌入。
    CHROMA_COLLECTION_MECHANISM: str = os.getenv("CHROMA_COLLECTION_MECHANISM", "biodynamics_mechanism")
    CHROMA_COLLECTION_PARAMETER: str = os.getenv("CHROMA_COLLECTION_PARAMETER", "biodynamics_parameter")
    CHROMA_COLLECTION_EXPERIMENT: str = os.getenv("CHROMA_COLLECTION_EXPERIMENT", "biodynamics_experiment")
    CHROMA_COLLECTION_EVIDENCE: str = os.getenv("CHROMA_COLLECTION_EVIDENCE", "biodynamics_evidence")

    # Embedding 模型配置
    # EMBEDDING_PROVIDER: openai（默认，调用云端 API） | local（使用 sentence-transformers 本地模型）| openrouter
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    # 默认模型根据 provider 自动选择，local 模式下使用轻量本地模型，openrouter 模式需显式指定
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
        if EMBEDDING_PROVIDER.lower() == "local"
        else "text-embedding-3-small",
    )
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")

    # OpenRouter 专用配置：用于 Embedding / Rerank 模型，与主 LLM 解耦
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_EMBEDDING_MODEL: str = os.getenv(
        "OPENROUTER_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    )
    # 兼容旧版单模型配置；新版支持逗号分隔多模型，如：
    # OPENROUTER_RERANK_MODELS=cohere/rerank-4-pro,nvidia/llama-nemotron-rerank-vl-1b-v2:free
    # 默认优先级：cohere/rerank-4-pro 优先，免费模型兜底
    OPENROUTER_RERANK_MODEL: str = os.getenv("OPENROUTER_RERANK_MODEL", "cohere/rerank-4-pro")
    OPENROUTER_RERANK_MODELS: list[str] = [
        m.strip()
        for m in os.getenv(
            "OPENROUTER_RERANK_MODELS",
            os.getenv("OPENROUTER_RERANK_MODEL", "cohere/rerank-4-pro,nvidia/llama-nemotron-rerank-vl-1b-v2:free"),
        ).split(",")
        if m.strip()
    ]

    # SiliconFlow 专用配置：用于 Embedding / Rerank 模型，与主 LLM 解耦
    SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_BASE_URL: str = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    SILICONFLOW_EMBEDDING_MODEL: str = os.getenv(
        "SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3"
    )
    SILICONFLOW_RERANK_MODEL: str = os.getenv(
        "SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
    )
    SILICONFLOW_RERANK_MODELS: list[str] = [
        m.strip()
        for m in os.getenv("SILICONFLOW_RERANK_MODELS", os.getenv("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")).split(",")
        if m.strip()
    ]

    # 讯飞 MaaS 专用配置：用于 Embedding / Rerank 模型，与主 LLM 独立
    # key 通常为 appid:api_secret 格式，整体作为 Bearer Token 使用
    XFYUN_MAAS_API_KEY: str = os.getenv("XFYUN_MAAS_API_KEY", "")
    XFYUN_MAAS_EMBEDDING_BASE_URL: str = os.getenv(
        "XFYUN_MAAS_EMBEDDING_BASE_URL", "https://maas-api.cn-huabei-1.xf-yun.com/v2"
    )
    XFYUN_MAAS_RERANK_BASE_URL: str = os.getenv(
        "XFYUN_MAAS_RERANK_BASE_URL", "https://maas-api.cn-huabei-1.xf-yun.com/v2"
    )
    XFYUN_MAAS_EMBEDDING_MODEL: str = os.getenv(
        "XFYUN_MAAS_EMBEDDING_MODEL", "xop3qwen8bembedding"
    )
    XFYUN_MAAS_RERANK_MODEL: str = os.getenv(
        "XFYUN_MAAS_RERANK_MODEL", "xop3qwen8breranker"
    )
    XFYUN_MAAS_RERANK_MODELS: list[str] = [
        m.strip()
        for m in os.getenv("XFYUN_MAAS_RERANK_MODELS", os.getenv("XFYUN_MAAS_RERANK_MODEL", "xop3qwen8breranker")).split(",")
        if m.strip()
    ]

    # Rerank 策略：rule（启发式）| model（调用任意 /rerank API）| hybrid（融合）
    # 旧值 openrouter 仍兼容，会被映射为 model
    # 默认启用 model，RAG 启动后自动调用 rerank 模型提升排序精度
    RERANK_PROVIDER: str = os.getenv("RERANK_PROVIDER", "model")
    # Rerank 多提供商候选列表，按优先级排列。
    # 默认优先级：讯飞 MaaS > OpenRouter > SiliconFlow
    RERANK_PROVIDERS: list[str] = [
        p.strip()
        for p in os.getenv("RERANK_PROVIDERS", "xfyun,openrouter,siliconflow").split(",")
        if p.strip()
    ]
    # Rerank 模型选择模式：auto（按 RERANK_PROVIDERS 顺序优先可用）| llm（主 LLM 在可用模型中选择）
    RERANK_SELECTION_MODE: str = os.getenv("RERANK_SELECTION_MODE", "auto")

    # PubMed E-utilities 联系邮箱
    NCBI_EMAIL: str = os.getenv("NCBI_EMAIL", "")
    # NCBI API Key（可选，申请后限流从 3 req/s 提升至 10 req/s）
    NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", "")

    # MCP（Model Context Protocol）工具端点配置
    # 留空则自动降级为 LLM 内部知识完成术语查询，不影响主流程
    MCP_OPENBIOMED_URL: str = os.getenv("MCP_OPENBIOMED_URL", "")
    MCP_MEDTERM_URL: str = os.getenv("MCP_MEDTERM_URL", "")
    MCP_PUBMED_URL: str = os.getenv("MCP_PUBMED_URL", "")
    MCP_UMLS_URL: str = os.getenv("MCP_UMLS_URL", "")
    # MCP 总开关：设为 false 可完全跳过 node0 术语查询节点
    MCP_ENABLED: bool = os.getenv("MCP_ENABLED", "true").lower() == "true"

    # 在线数据库自动补充：当本地 ChromaDB 检索命中不足时，自动查询 KEGG/Reactome/UniProt/ChEMBL
    RAG_ONLINE_FALLBACK: bool = os.getenv("RAG_ONLINE_FALLBACK", "true").lower() == "true"
    RAG_ONLINE_FALLBACK_THRESHOLD: float = float(os.getenv("RAG_ONLINE_FALLBACK_THRESHOLD", "0.3"))
    # 在线回退单次查询超时（秒），避免外部 API 慢响应阻塞 workflow
    RAG_ONLINE_QUERY_TIMEOUT: float = float(os.getenv("RAG_ONLINE_QUERY_TIMEOUT", "10.0"))
    # 在线回退工作流总时长上限（秒），超过则触发熔断降级
    RAG_ONLINE_TOTAL_BUDGET: float = float(os.getenv("RAG_ONLINE_TOTAL_BUDGET", "600.0"))

    # 沙箱执行配置：确定性求解与资源限制
    # 默认 LSODA 求解器（确定性），禁止随机噪声除非 SDE 模式显式开启
    SANDBOX_TIMEOUT: int = int(os.getenv("SANDBOX_TIMEOUT", "60"))
    SANDBOX_MAX_STEP_RATIO: float = float(os.getenv("SANDBOX_MAX_STEP_RATIO", "0.01"))  # T_END 的比例
    SANDBOX_AUDIT_LOG: bool = os.getenv("SANDBOX_AUDIT_LOG", "true").lower() == "true"
    SANDBOX_AUDIT_LOG_DIR: str = os.getenv("SANDBOX_AUDIT_LOG_DIR", "data/sandbox_logs")

    # 系统降级模式开关：full | rag_only | template_only
    # - full: RAG + SBML + Sandbox 全部启用（默认）
    # - rag_only: SBML 失败时仅用 RAG 参数（自动降级，无需手动设置）
    # - template_only: RAG 严重缺失时使用模板默认参数（自动降级）
    DEGRADATION_MODE: str = os.getenv("DEGRADATION_MODE", "full").lower()

    # 监控指标配置：prometheus | log | off
    # - prometheus: 集成 prometheus_client，暴露 /metrics 端点
    # - log: 轻量结构化日志记录器（默认，零依赖）
    # - off: 关闭监控
    METRICS_BACKEND: str = os.getenv("METRICS_BACKEND", "log").lower()
    METRICS_LOG_DIR: str = os.getenv("METRICS_LOG_DIR", "data/metrics")

    # v3 工作流版本开关："v3" (默认，Supervisor-Worker 动态编排) | "v2" | "v1"
    WORKFLOW_VERSION: str = os.getenv("WORKFLOW_VERSION", "v3").lower()

    # =============================================================================
    # v4 迁移 Feature Flags（Phase 1）
    # 详见 BioDynamics_v4_Migration_Plan.md
    # 所有 flag 默认 false，保证 v3 行为完全不受影响
    # =============================================================================
    # Phase 1: Ontology Agent + Pathway Graph
    V4_ONTOLOGY_AGENT_ENABLED: bool = os.getenv(
        "V4_ONTOLOGY_AGENT_ENABLED", "false"
    ).lower() == "true"
    V4_PATHWAY_GRAPH_ENABLED: bool = os.getenv(
        "V4_PATHWAY_GRAPH_ENABLED", "false"
    ).lower() == "true"

    # Phase 2: Reaction IR v2 + Adapter Pattern
    # V4_REACTION_IR_ENABLED: 控制 v4 Reaction IR v2 的生成
    #   - false（默认）：worker_ode 跳过 v4 路径，完全走 v3 network_json
    #   - true：调用 Reaction Builder 生成 v4_reaction_ir 写入 state
    # V4_REACTION_IR_ADAPTER_ENABLED: 控制 v4→v3 的 Adapter 同步
    #   - false（默认）：v4_reaction_ir 生成但不同步到 network_json
    #   - true：v4_reaction_ir 生成后，通过 v4_to_v3 Adapter 同步写入 network_json
    # 铁律：两个 flag 均为 false 时，系统行为与 v3 完全一致
    V4_REACTION_IR_ENABLED: bool = os.getenv(
        "V4_REACTION_IR_ENABLED", "false"
    ).lower() == "true"
    V4_REACTION_IR_ADAPTER_ENABLED: bool = os.getenv(
        "V4_REACTION_IR_ADAPTER_ENABLED", "false"
    ).lower() == "true"

    # Phase 3: Pathway Graph + ODE Template v2
    # V4_PATHWAY_GRAPH_ENABLED: 控制 v4 Pathway Graph 的构建
    #   - false（默认）：跳过 Pathway Graph 构建，state.v4_pathway_graph 保持 None
    #   - true：调用 PathwayGraphBuilder 构建 v4_pathway_graph 写入 state
    # V4_ODE_TEMPLATE_V2_ENABLED: 控制 v4 ODE Template 的渲染
    #   - false（默认）：worker_ode 跳过 v4 路径，仍走 v3 ode_templates/
    #   - true：调用 ODERendererV2 从 ReactionIRv2 + PathwayGraph 渲染 v4 ODE 代码
    # 铁律：两个 flag 均为 false 时，系统行为与 v3 完全一致
    # 依赖关系：V4_ODE_TEMPLATE_V2_ENABLED=true 时建议同时开启
    #           V4_PATHWAY_GRAPH_ENABLED（用于提取 temporal/DDE 信息）
    # 注意：V4_PATHWAY_GRAPH_ENABLED 已在 Phase 1 段定义（第 200 行），此处不重复
    V4_ODE_TEMPLATE_V2_ENABLED: bool = os.getenv(
        "V4_ODE_TEMPLATE_V2_ENABLED", "false"
    ).lower() == "true"

    # Phase 4: Pathway Planner + Specialist + Cross-talk Coordinator
    # V4_PATHWAY_PLANNER_ENABLED: 控制 v4 Pathway Planner 的通路识别
    #   - false（默认）：跳过通路识别，state.v4_pathway_class 保持 None
    #   - true：调用 Pathway Planner 输出 v4_pathway_class + 预识别 cross-talk edges
    # 铁律：flag=false 时系统行为与 v3 完全一致（不识别通路、不写 v4_pathway_class）
    # 依赖关系：建议与 V4_PATHWAY_GRAPH_ENABLED 配合使用（Pathway Specialist 需通路图）
    V4_PATHWAY_PLANNER_ENABLED: bool = os.getenv(
        "V4_PATHWAY_PLANNER_ENABLED", "false"
    ).lower() == "true"

    # V4_PATHWAY_SPECIALIST_ENABLED: 控制 v4 Pathway Specialist 的执行
    #   - false（默认）：跳过 Specialist 模块加载与应用，state.v4_reaction_ir 不被 Specialist 修改
    #   - true：根据 pathway_class 调用对应 Specialist，输出通路特异 Reaction IR 片段
    # 铁律：flag=false 时 Specialist 不执行，v3 行为完全不受影响
    V4_PATHWAY_SPECIALIST_ENABLED: bool = os.getenv(
        "V4_PATHWAY_SPECIALIST_ENABLED", "false"
    ).lower() == "true"

    # V4_CROSSTALK_COORDINATOR_ENABLED: 控制 v4 Cross-talk Coordinator 的执行
    #   - false（默认）：跳过 cross-talk 协调，state.v4_crosstalk_edges 等保持空
    #   - true：多通路场景下调用 Coordinator 合并 shared species + cross-talk edges
    # 铁律：flag=false 时 Coordinator 不执行，v3 行为完全不受影响
    # 依赖关系：建议与 V4_PATHWAY_SPECIALIST_ENABLED 配合使用（Coordinator 消费 Specialist 输出）
    # 职责边界：不修改 Specialist 内部 Reaction；不生成 ODE；不做 SBML 验证
    V4_CROSSTALK_COORDINATOR_ENABLED: bool = os.getenv(
        "V4_CROSSTALK_COORDINATOR_ENABLED", "false"
    ).lower() == "true"

    # =============================================================================
    # Phase 5 Feature Flags（SBML Grounding + Validation Pyramid）
    # =============================================================================
    # V4_SBML_GROUNDER_ENABLED: 控制 v4 SBML Grounder Agent 的执行
    # 职责：建立 ODE ↔ Reaction ↔ SBML ↔ Parameter ↔ PMID 五级映射链
    # 铁律：flag=false 时 Grounder 不执行，v3 LLM 解析行为不变
    V4_SBML_GROUNDER_ENABLED: bool = os.getenv(
        "V4_SBML_GROUNDER_ENABLED", "false"
    ).lower() == "true"

    # V4_VALIDATION_PYRAMID_ENABLED: 控制 v4 Validation Pyramid（5 层）的执行
    # 职责：Level 1 internal / Level 2 SBML / Level 3 cross-pathway / Level 4 benchmark / Level 5 hypothesis
    # 铁律：flag=false 时 Pyramid 不执行，v3 model_consistency_validator + sbml_validator 行为不变
    V4_VALIDATION_PYRAMID_ENABLED: bool = os.getenv(
        "V4_VALIDATION_PYRAMID_ENABLED", "false"
    ).lower() == "true"

    # V4_CALIBRATION_AGENT_ENABLED: 控制 v4 Calibration Agent 的执行
    # 职责：用 BioModels reference 或用户实验数据拟合参数，输出置信区间
    # 铁律：flag=false 时 Calibration 不执行，v3 参数估计行为不变
    V4_CALIBRATION_AGENT_ENABLED: bool = os.getenv(
        "V4_CALIBRATION_AGENT_ENABLED", "false"
    ).lower() == "true"

    # V4_HYPOTHESIS_AGENT_ENABLED: 控制 v4 Hypothesis Agent（P6）的执行
    # 职责：基于 pathway graph + Reaction IR 生成假设列表（v4_hypothesis_list）
    # 铁律：flag=false 时 P6 不执行，state.v4_hypothesis_list 保持空
    #       Level 5 Hypothesis Validation 自动 skipped（pass=True，不阻塞）
    # 依赖关系：被 Level 5 Hypothesis Validation 消费（validation_v2/level5_hypothesis.py）
    V4_HYPOTHESIS_AGENT_ENABLED: bool = os.getenv(
        "V4_HYPOTHESIS_AGENT_ENABLED", "false"
    ).lower() == "true"


# =============================================================================
# 依赖隔离策略（try-import 模板）—— Phase 5 前置（SubTask 5.0.2）
# =============================================================================
# 设计原则：所有可选科学计算依赖必须 try-import，失败时降级到标准库或简化方法，
# 并记录 warning，不阻塞主流水线。
#
# 依赖矩阵：
# - roadrunner  : SBML 仿真（Level 2 Track A）  → 不可用降级到 Track B 结构相似度
# - lmfit       : 参数校准（Calibration Agent） → 不可用降级到 scipy.optimize.least_squares
# - SALib       : Sobol/Morris 全局灵敏度       → 不可用仅运行 local sensitivity
# - lxml        : SBML XML 解析（sbml_parser_v2）→ 不可用降级到 xml.etree.ElementTree
# - chromadb    : 向量库（已在 P1 中处理）       → 不可用降级到内存检索
#
# 使用模式（在消费模块中）：
#   from app.config import ROADRUNNER_AVAILABLE
#   if ROADRUNNER_AVAILABLE:
#       import roadrunner
#       # Track A: 真实 SBML 仿真
#   else:
#       # Track B: 结构相似度评分
# =============================================================================

# roadrunner: SBML 仿真器（Level 2 Track A 依赖）
try:
    import roadrunner  # type: ignore
    ROADRUNNER_AVAILABLE = True
    ROADRUNNER_VERSION = getattr(roadrunner, "__version__", "unknown")
except ImportError:
    ROADRUNNER_AVAILABLE = False
    ROADRUNNER_VERSION = None
    logger.warning(
        "roadrunner 未安装：Level 2 SBML Validation 将降级到 Track B 结构相似度评分。"
        "安装命令：pip install python-roadrunner"
    )

# lmfit: 参数校准（Calibration Agent 依赖）
try:
    import lmfit  # type: ignore
    LMFIT_AVAILABLE = True
    LMFIT_VERSION = getattr(lmfit, "__version__", "unknown")
except ImportError:
    LMFIT_AVAILABLE = False
    LMFIT_VERSION = None
    logger.warning(
        "lmfit 未安装：Calibration Agent 将降级到 scipy.optimize.least_squares。"
        "安装命令：pip install lmfit"
    )

# SALib: 全局灵敏度分析（Sobol/Morris 依赖）
try:
    import SALib  # type: ignore
    SALIB_AVAILABLE = True
    SALIB_VERSION = getattr(SALib, "__version__", "unknown")
except ImportError:
    SALIB_AVAILABLE = False
    SALIB_VERSION = None
    logger.warning(
        "SALib 未安装：Sensitivity Analysis 将仅运行 local sensitivity（forward difference）。"
        "安装命令：pip install SALib"
    )

# lxml: SBML XML 解析（sbml_parser_v2 依赖）
try:
    import lxml  # type: ignore
    from lxml import etree as lxml_etree  # type: ignore
    LXML_AVAILABLE = True
    LXML_VERSION = getattr(lxml, "__version__", "unknown")
except ImportError:
    LXML_AVAILABLE = False
    LXML_VERSION = None
    # 不打印 warning：xml.etree.ElementTree 作为标准库后备已足够（lxml 是优化，非必需）
    logger.info(
        "lxml 未安装：sbml_parser_v2 将使用 xml.etree.ElementTree（标准库）解析 SBML。"
        "如需更严格的 XML 校验，可安装：pip install lxml"
    )


settings = Settings()


# 如果环境变量未提供 API Key，则使用占位符，避免模块导入时因空字符串触发 OpenAI 校验错误。
# 运行真实请求前必须在 .env 中配置有效的 OPENAI_API_KEY。
if not settings.OPENAI_API_KEY:
    settings.OPENAI_API_KEY = "sk-placeholder-please-set-openai-api-key"


def strip_markdown_json(text: str) -> str:
    """剥离 LLM 返回的 markdown 代码块标记。

    BigModel (glm-5.1 / glm-4.7-flash) 在 structured output 与普通调用中均可能返回
    ```json ... ``` / ```python ... ``` 包裹的内容，而 OpenAI SDK 的 model_validate_json
    不剥离 markdown 标记直接抛 ValidationError。此函数提取纯内容字符串。
    统一实现，供 _StructuredOutputRunnable / _safe_json_parse / _strip_markdown_code_blocks 复用。
    """
    if not isinstance(text, str):
        return text
    # 匹配 ```lang ... ``` 或 ``` ... ```（lang 为任意语言标签，非贪婪取第一个代码块）
    match = re.search(r"```(?:\w+)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text.strip()


class _StructuredOutputRunnable(Runnable):
    """自定义 structured output Runnable，处理 BigModel markdown 包裹的 JSON。

    绕过 ChatOpenAI.with_structured_output 内部的 model_validate_json 解析
    （该解析不剥离 markdown 代码块，在返回前就抛 ValidationError），改为：
    1. 调用底层 ChatOpenAI.invoke 获取原始 AIMessage
    2. 用 strip_markdown_json 剥离 markdown 标记
    3. 用 schema.model_validate_json 解析为 Pydantic 模型

    主备切换在 parse 层面也生效：primary 的 invoke 或 parse 失败时尝试 backup。
    """

    BACKUP_DELAY_SECONDS: float = 0.5

    def __init__(
        self, primary: Runnable, schema: Any, backup: Runnable | None = None
    ) -> None:
        self._primary = primary
        self._backup = backup
        self._schema = schema

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        try:
            response = self._primary.invoke(input, config=config, **kwargs)
            return self._parse_response(response)
        except Exception:
            if self._backup is None:
                raise
            time.sleep(self.BACKUP_DELAY_SECONDS)
            response = self._backup.invoke(input, config=config, **kwargs)
            return self._parse_response(response)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        try:
            response = await self._primary.ainvoke(input, config=config, **kwargs)
            return self._parse_response(response)
        except Exception:
            if self._backup is None:
                raise
            time.sleep(self.BACKUP_DELAY_SECONDS)
            response = await self._backup.ainvoke(input, config=config, **kwargs)
            return self._parse_response(response)

    def _parse_response(self, response: Any) -> Any:
        """从 AIMessage 提取文本、剥离 markdown、解析为 Pydantic 模型。

        处理三类常见不匹配：
        1. markdown 代码块包裹 → strip_markdown_json 剥离
        2. LLM 按 prompt 的 Output Format 返回 JSON 数组，但 schema 期望对象
           （如 RAG_EXTRACTION_PROMPT 要求输出数组，RAGExtractionOutput 期望 {"params": [...]}）
           → 自动找到 schema 中第一个 list 类型字段，把数组包装进去
        3. 文本中混有非 JSON 内容 → 正则提取最外层 JSON
        """
        text = response.content if hasattr(response, "content") else str(response)
        cleaned = strip_markdown_json(text)

        # 优先尝试直接 model_validate_json（LLM 按 schema 输出对象时直接成功）
        try:
            return self._schema.model_validate_json(cleaned)
        except Exception:
            pass

        # 解析为 Python 对象以处理结构不匹配
        parsed: Any = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # 兜底：正则提取最外层 JSON 对象或数组
            match = re.search(r"[\[{][\s\S]*[\]}]", cleaned)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = None
            if parsed is None:
                raise ValueError(f"无法从 LLM 响应中解析 JSON：{cleaned[:200]}")

        # LLM 返回数组但 schema 期望对象时，自动包装到第一个 list 类型字段
        if isinstance(parsed, list):
            for field_name, field_info in self._schema.model_fields.items():
                annotation = field_info.annotation
                if getattr(annotation, "__origin__", None) is list:
                    parsed = {field_name: parsed}
                    break

        return self._schema.model_validate(parsed)


class FallbackLLM(Runnable):
    """主 LLM 调用失败时自动切换到备用 LLM 的包装器。"""

    # 主备切换前短暂等待，避免同一 provider 因瞬时限流导致两个 key 连续 burst
    BACKUP_DELAY_SECONDS: float = 0.5

    def __init__(self, primary: Runnable, backup: Runnable | None = None):
        self.primary = primary
        self.backup = backup

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        try:
            return self.primary.invoke(input, config=config, **kwargs)
        except Exception:
            if self.backup is None:
                raise
            time.sleep(self.BACKUP_DELAY_SECONDS)
            return self.backup.invoke(input, config=config, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        try:
            return await self.primary.ainvoke(input, config=config, **kwargs)
        except Exception:
            if self.backup is None:
                raise
            time.sleep(self.BACKUP_DELAY_SECONDS)
            return await self.backup.ainvoke(input, config=config, **kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        """返回自定义 structured output Runnable，统一处理 BigModel markdown 包裹的 JSON。

        不使用 ChatOpenAI.with_structured_output（其内部 model_validate_json 不剥离
        markdown 标记会抛 ValidationError），改为 _StructuredOutputRunnable 直接
        invoke 底层 ChatOpenAI 取原始 AIMessage 后手动清洗解析。
        """
        return _StructuredOutputRunnable(self.primary, schema, self.backup)


# -----------------------------------------------------------------------------
# OpenRouter Embedding / Rerank 客户端封装
# -----------------------------------------------------------------------------
class OpenRouterEmbeddings(Embeddings):
    """OpenRouter Embedding API 的 LangChain 兼容封装。

    OpenRouter 提供与 OpenAI 兼容的 /embeddings 端点，但模型名与 OpenAI 不同
    （如 nvidia/llama-nemotron-embed-vl-1b-v2:free）。本类直接通过 requests 调用
    POST {base_url}/embeddings，返回与 OpenAIEmbeddings 一致的 list[float] 格式。
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _call(self, texts: list[str]) -> list[list[float]]:
        """统一发起 embedding 请求并解析响应。"""
        if not self.api_key:
            raise ValueError("OpenRouter API Key 未配置，无法获取 embedding")
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": texts}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            # OpenRouter 返回 data[*].index + embedding，按 index 排序保证顺序
            items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in items]
        except Exception as exc:
            logger.error("OpenRouter embedding 请求失败 (model=%s): %s", self.model, exc)
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量文档 embedding。"""
        return self._call(texts)

    def embed_query(self, text: str) -> list[float]:
        """单条查询 embedding。"""
        result = self._call([text])
        return result[0]


class OpenRouterRerankClient:
    """OpenRouter Rerank API 封装。

    OpenRouter 提供 /rerank 端点，支持 cohere/rerank-4-pro 与
    nvidia/llama-nemotron-rerank-vl-1b-v2:free 等模型。输入 query + documents，
    返回按相关性排序的 (index, relevance_score) 列表。
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """调用 OpenRouter /rerank 接口。

        返回 [{"index": int, "relevance_score": float, "document": {"text": str}}, ...]
        按 relevance_score 降序排列。失败时抛出异常，调用方负责降级。
        """
        if not self.api_key:
            raise ValueError("OpenRouter API Key 未配置，无法调用 rerank")
        if not documents:
            return []
        url = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return list(data.get("results", []))

    def health_check(self) -> bool:
        """轻量连通性探测。"""
        try:
            self.rerank("health", ["ok"], top_n=1)
            return True
        except Exception as exc:
            logger.debug("OpenRouter rerank 健康检查失败 (%s): %s", self.model, exc)
            return False


class SiliconFlowEmbeddings(Embeddings):
    """SiliconFlow Embedding API 的 LangChain 兼容封装。

    SiliconFlow 提供 OpenAI 兼容的 /embeddings 端点，模型如 BAAI/bge-m3。
    默认 base_url: https://api.siliconflow.cn/v1
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _call(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("SiliconFlow API Key 未配置，无法获取 embedding")
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": texts}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in items]
        except Exception as exc:
            logger.error("SiliconFlow embedding 请求失败 (model=%s): %s", self.model, exc)
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts)

    def embed_query(self, text: str) -> list[float]:
        result = self._call([text])
        return result[0]


class SiliconFlowRerankClient:
    """SiliconFlow Rerank API 封装。

    端点：POST {base_url}/rerank，支持 BAAI/bge-reranker-v2-m3 等模型。
    返回结构与 OpenRouter / Cohere 兼容：results[*].index + relevance_score。
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.api_key:
            raise ValueError("SiliconFlow API Key 未配置，无法调用 rerank")
        if not documents:
            return []
        url = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return list(data.get("results", []))

    def health_check(self) -> bool:
        """轻量连通性探测：发一条空文档列表的 rerank 请求会报错，改为发单条。"""
        try:
            self.rerank("health", ["ok"], top_n=1)
            return True
        except Exception as exc:
            logger.debug("SiliconFlow rerank 健康检查失败 (%s): %s", self.model, exc)
            return False


class XfyunMaasEmbeddings(Embeddings):
    """讯飞 MaaS Embedding API 的 LangChain 兼容封装。

    讯飞 MaaS 提供 OpenAI 兼容的 /embeddings 端点，模型如 Qwen3-Embedding-8B
    （modelId: xop3qwen8bembedding）。默认端点：
    https://maas-api.cn-huabei-1.xf-yun.com/v2/embeddings
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://maas-api.cn-huabei-1.xf-yun.com/v2",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _call(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("讯飞 MaaS API Key 未配置，无法获取 embedding")
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": texts}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in items]
        except Exception as exc:
            logger.error("讯飞 MaaS embedding 请求失败 (model=%s): %s", self.model, exc)
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._call(texts)

    def embed_query(self, text: str) -> list[float]:
        result = self._call([text])
        return result[0]


class XfyunMaasRerankClient:
    """讯飞 MaaS Rerank API 封装。

    端点：POST {base_url}/rerank，支持 Qwen3-Reranker-8B（modelId: xop3qwen8breranker）。
    默认端点：https://maas-api.cn-huabei-1.xf-yun.com/v2/rerank
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://maas-api.cn-huabei-1.xf-yun.com/v2",
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.api_key:
            raise ValueError("讯飞 MaaS API Key 未配置，无法调用 rerank")
        if not documents:
            return []
        url = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        # 兼容两种常见返回结构：results 数组直接位于根或 data 字段下
        results = data.get("results") or data.get("data", [])
        return list(results)

    def health_check(self) -> bool:
        """轻量连通性探测。"""
        try:
            self.rerank("health", ["ok"], top_n=1)
            return True
        except Exception as exc:
            logger.debug("讯飞 MaaS rerank 健康检查失败 (%s): %s", self.model, exc)
            return False


# -----------------------------------------------------------------------------
# 多提供商 Rerank 管理器
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RerankCandidate:
    """单个 rerank 模型候选。"""

    provider: str           # "openrouter" | "siliconflow" | "xfyun"
    model: str              # 完整模型名
    client: Any             # 对应 rerank 客户端实例
    display_name: str       # 用于前端展示，如 "OpenRouter/cohere-rerank-4-pro"


class RerankManager:
    """管理多个 rerank 提供商/模型，支持连通性探测、优先级回退、LLM 选择。

    使用策略：
    1. 初始化时根据 settings 注册所有候选（OpenRouter、SiliconFlow）。
    2. health_check() 探测每个候选是否可连通，返回可用列表。
    3. rerank() 默认按 RERANK_PROVIDERS 顺序选择第一个可用候选；若
       RERANK_SELECTION_MODE=llm，则调用主 LLM 在可用候选中为当前 query 选择最佳模型。
    4. 任何候选失败时自动尝试下一个，全部失败则抛出异常（调用方回退到 rule）。
    """

    def __init__(self) -> None:
        self.candidates: list[RerankCandidate] = []
        self._available: list[RerankCandidate] = []
        self._build_candidates()

    def _build_candidates(self) -> None:
        """按 settings 构建候选列表。"""
        # OpenRouter
        if settings.OPENROUTER_API_KEY:
            for model in settings.OPENROUTER_RERANK_MODELS:
                client = OpenRouterRerankClient(
                    api_key=settings.OPENROUTER_API_KEY,
                    model=model,
                    base_url=settings.OPENROUTER_BASE_URL,
                )
                safe_model = model.replace("/", "-")
                self.candidates.append(
                    RerankCandidate(
                        provider="openrouter",
                        model=model,
                        client=client,
                        display_name=f"OpenRouter/{safe_model}",
                    )
                )
        # SiliconFlow
        if settings.SILICONFLOW_API_KEY:
            for model in settings.SILICONFLOW_RERANK_MODELS:
                client = SiliconFlowRerankClient(
                    api_key=settings.SILICONFLOW_API_KEY,
                    model=model,
                    base_url=settings.SILICONFLOW_BASE_URL,
                )
                safe_model = model.replace("/", "-")
                self.candidates.append(
                    RerankCandidate(
                        provider="siliconflow",
                        model=model,
                        client=client,
                        display_name=f"SiliconFlow/{safe_model}",
                    )
                )
        # 讯飞 MaaS
        if settings.XFYUN_MAAS_API_KEY:
            for model in settings.XFYUN_MAAS_RERANK_MODELS:
                client = XfyunMaasRerankClient(
                    api_key=settings.XFYUN_MAAS_API_KEY,
                    model=model,
                    base_url=settings.XFYUN_MAAS_RERANK_BASE_URL,
                )
                safe_model = model.replace("/", "-")
                self.candidates.append(
                    RerankCandidate(
                        provider="xfyun",
                        model=model,
                        client=client,
                        display_name=f"XfyunMaas/{safe_model}",
                    )
                )

    def health_check(self) -> list[dict[str, Any]]:
        """探测所有候选连通性，返回带 status 的元信息列表。"""
        reports: list[dict[str, Any]] = []
        available: list[RerankCandidate] = []
        for cand in self.candidates:
            try:
                ok = cand.client.health_check()
            except Exception as exc:
                logger.debug("Rerank 健康检查异常 %s: %s", cand.display_name, exc)
                ok = False
            reports.append({
                "provider": cand.provider,
                "model": cand.model,
                "display_name": cand.display_name,
                "available": ok,
            })
            if ok:
                available.append(cand)
        self._available = available
        return reports

    @property
    def available_candidates(self) -> list[RerankCandidate]:
        """返回最近一次 health_check() 后判定为可用的候选。"""
        return self._available

    def _select_candidate(self, query: str) -> RerankCandidate | None:
        """根据配置策略选择一个可用候选。"""
        available = self._available or self.candidates
        if not available:
            return None

        # auto 模式：按 RERANK_PROVIDERS 顺序选择第一个可用候选
        if settings.RERANK_SELECTION_MODE != "llm":
            priority = {p: i for i, p in enumerate(settings.RERANK_PROVIDERS)}
            ordered = sorted(available, key=lambda c: priority.get(c.provider, 999))
            return ordered[0]

        # llm 模式：让主 LLM 根据 query 在可用候选中选择最合适的
        return self._select_with_llm(query, available)

    def _select_with_llm(self, query: str, available: list[RerankCandidate]) -> RerankCandidate | None:
        """调用主 LLM 在可用 rerank 模型中选择最适合当前 query 的模型。"""
        if not available:
            return None
        if len(available) == 1:
            return available[0]

        options = "\n".join(
            f"{i+1}. {c.display_name} ({c.provider})"
            for i, c in enumerate(available)
        )
        prompt = f"""你是模型路由专家。请根据以下查询，从可用的 rerank 模型中选择最适合的一个。

查询：{query}

可用 rerank 模型：
{options}

请直接返回所选模型的序号（1-{len(available)}），不要解释。"""
        try:
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            # 提取第一个数字
            match = re.search(r"\d+", text)
            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(available):
                    logger.info("LLM 选择 rerank 模型: %s", available[idx].display_name)
                    return available[idx]
        except Exception as exc:
            logger.warning("LLM 选择 rerank 模型失败，降级到 auto 策略：%s", exc)
        # 失败时回退到 auto
        priority = {p: i for i, p in enumerate(settings.RERANK_PROVIDERS)}
        ordered = sorted(available, key=lambda c: priority.get(c.provider, 999))
        return ordered[0]

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> tuple[list[dict[str, Any]], RerankCandidate | None]:
        """执行 rerank，返回 (results, 实际使用的候选)。

        按策略选择候选，若失败则尝试下一个可用候选，全部失败抛出异常。
        """
        if not documents:
            return [], None

        # 确保可用性列表已刷新（若未调用 health_check，至少用所有候选）
        candidates = list(self._available) if self._available else list(self.candidates)
        if not candidates:
            raise RuntimeError("没有配置任何 rerank 候选模型")

        # 根据策略选出首选候选
        primary = self._select_candidate(query)
        if primary is None:
            raise RuntimeError("没有可用的 rerank 候选模型")

        # 把首选放最前面，其余按原顺序作为 fallback
        ordered = [primary] + [c for c in candidates if c is not primary]
        last_error: Exception | None = None
        for cand in ordered:
            try:
                results = cand.client.rerank(query=query, documents=documents, top_n=top_n)
                logger.debug("Rerank 成功: %s, 返回 %d 条", cand.display_name, len(results))
                return results, cand
            except Exception as exc:
                logger.warning("Rerank 候选 %s 失败: %s", cand.display_name, exc)
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("所有 rerank 候选均失败")


# OpenRouter 要求免费模型携带 HTTP-Referer 与 X-Title 头部以识别调用来源。
# 当 LLM 使用 OpenRouter 端点时自动注入，避免免费模型因缺少头部被拒绝。
def _make_openrouter_headers(base_url: str) -> dict[str, str] | None:
    if "openrouter" in base_url.lower():
        return {
            "HTTP-Referer": settings.FRONTEND_URL,
            "X-Title": "BioDynamics Agent",
        }
    return None


# 全局主 LLM 实例（max_retries=2，让 OpenAI 客户端在限流时自动指数退避重试）
_primary_llm = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    model=settings.OPENAI_MODEL,
    temperature=0.2,
    max_retries=2,
    default_headers=_make_openrouter_headers(settings.OPENAI_BASE_URL),
)

# 全局备用 LLM 实例（仅在配置完整时初始化）
_backup_llm: ChatOpenAI | None = None
if settings.BACKUP_API_KEY and settings.BACKUP_BASE_URL and settings.BACKUP_MODEL:
    _backup_llm = ChatOpenAI(
        api_key=settings.BACKUP_API_KEY,
        base_url=settings.BACKUP_BASE_URL,
        model=settings.BACKUP_MODEL,
        temperature=0.2,
        max_retries=2,
        default_headers=_make_openrouter_headers(settings.BACKUP_BASE_URL),
    )

# 供所有 LangGraph 节点复用的带故障转移 LLM
llm: FallbackLLM = FallbackLLM(_primary_llm, _backup_llm)

# 全局 Embedding 模型实例，供 RAG 向量检索复用。
# 支持五种模式：
# 1. openai：调用 OpenAI 兼容云端 Embedding API（默认）。
# 2. local：使用 sentence-transformers 本地模型，避免 API 余额/网络问题。
# 3. openrouter：调用 OpenRouter /embeddings 端点。
# 4. siliconflow：调用 SiliconFlow /embeddings 端点，如 BAAI/bge-m3。
# 5. xfyun：调用讯飞 MaaS /embeddings 端点，如 Qwen3-Embedding-8B。
_embedding_provider = settings.EMBEDDING_PROVIDER.lower()
if _embedding_provider == "local":
    from app.local_embeddings import LocalEmbeddings

    embedding_model: Embeddings = LocalEmbeddings(model_name=settings.EMBEDDING_MODEL)
elif _embedding_provider == "openrouter":
    embedding_model = OpenRouterEmbeddings(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.OPENROUTER_EMBEDDING_MODEL,
        base_url=settings.OPENROUTER_BASE_URL,
    )
elif _embedding_provider == "siliconflow":
    embedding_model = SiliconFlowEmbeddings(
        api_key=settings.SILICONFLOW_API_KEY,
        model=settings.SILICONFLOW_EMBEDDING_MODEL,
        base_url=settings.SILICONFLOW_BASE_URL,
    )
elif _embedding_provider == "xfyun":
    embedding_model = XfyunMaasEmbeddings(
        api_key=settings.XFYUN_MAAS_API_KEY,
        model=settings.XFYUN_MAAS_EMBEDDING_MODEL,
        base_url=settings.XFYUN_MAAS_EMBEDDING_BASE_URL,
    )
else:
    embedding_model = OpenAIEmbeddings(
        api_key=settings.EMBEDDING_API_KEY or settings.OPENAI_API_KEY,
        base_url=settings.EMBEDDING_BASE_URL or settings.OPENAI_BASE_URL,
        model=settings.EMBEDDING_MODEL,
    )

# 全局 Rerank 管理器实例（当 RERANK_PROVIDER=model/hybrid/openrouter 时使用）。
# 失败时 RagClient.rerank_results 会降级到启发式重排，避免阻塞主流程。
# 兼容旧版 openrouter_rerank_client 变量名，但实际指向 RerankManager。
openrouter_rerank_client: RerankManager | None = None
rerank_manager: RerankManager | None = None
if settings.RERANK_PROVIDER in ("openrouter", "model", "hybrid"):
    rerank_manager = RerankManager()
    openrouter_rerank_client = rerank_manager
