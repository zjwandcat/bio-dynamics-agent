# BioDynamics Agent - 全局配置模块
# 负责从环境变量加载配置，并初始化全局 LLM 客户端。

import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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

    # Embedding 模型配置
    # EMBEDDING_PROVIDER: openai（默认，调用云端 API） | local（使用 sentence-transformers 本地模型）
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    # 默认模型根据 provider 自动选择，local 模式下使用轻量本地模型
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
        if EMBEDDING_PROVIDER.lower() == "local"
        else "text-embedding-3-small",
    )
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")

    # PubMed E-utilities 联系邮箱
    NCBI_EMAIL: str = os.getenv("NCBI_EMAIL", "")


settings = Settings()


# 如果环境变量未提供 API Key，则使用占位符，避免模块导入时因空字符串触发 OpenAI 校验错误。
# 运行真实请求前必须在 .env 中配置有效的 OPENAI_API_KEY。
if not settings.OPENAI_API_KEY:
    settings.OPENAI_API_KEY = "sk-placeholder-please-set-openai-api-key"


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

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "FallbackLLM":
        primary_structured = self.primary.with_structured_output(schema, **kwargs)
        backup_structured = (
            self.backup.with_structured_output(schema, **kwargs)
            if self.backup is not None
            else None
        )
        return FallbackLLM(primary_structured, backup_structured)


# 全局主 LLM 实例（max_retries=2，让 OpenAI 客户端在限流时自动指数退避重试）
_primary_llm = ChatOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    model=settings.OPENAI_MODEL,
    temperature=0.2,
    max_retries=2,
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
    )

# 供所有 LangGraph 节点复用的带故障转移 LLM
llm: FallbackLLM = FallbackLLM(_primary_llm, _backup_llm)

# 全局 Embedding 模型实例，供 RAG 向量检索复用。
# 支持两种模式：
# 1. openai：调用 OpenAI 兼容云端 Embedding API（默认）。
# 2. local：使用 sentence-transformers 本地模型，避免 API 余额/网络问题。
if settings.EMBEDDING_PROVIDER.lower() == "local":
    from app.local_embeddings import LocalEmbeddings

    embedding_model: Embeddings = LocalEmbeddings(model_name=settings.EMBEDDING_MODEL)
else:
    embedding_model = OpenAIEmbeddings(
        api_key=settings.EMBEDDING_API_KEY or settings.OPENAI_API_KEY,
        base_url=settings.EMBEDDING_BASE_URL or settings.OPENAI_BASE_URL,
        model=settings.EMBEDDING_MODEL,
    )
