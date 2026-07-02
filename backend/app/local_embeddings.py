# BioDynamics Agent - 本地 Embedding 模型封装
# 当云端 Embedding API（如 OpenAI/BigModel）不可用或余额不足时，
# 自动降级到 sentence-transformers 本地模型，保证 RAG 流水线可离线运行。

from typing import Any

from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """基于 sentence-transformers 的本地向量模型，兼容 LangChain Embeddings 接口。"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs: Any):
        # 延迟导入，避免在仅使用云端 Embedding 时引入 torch/sentence-transformers
        import os

        # 国内网络环境下优先使用镜像站下载模型；setdefault 避免覆盖用户显式配置
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from sentence_transformers import SentenceTransformer

        # local_files_only=True 优先使用已缓存模型，避免 HuggingFace 在线检查 adapter_config 时超时
        load_kwargs = {"local_files_only": True, **kwargs}
        self.model = SentenceTransformer(model_name, **load_kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档列表。"""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """编码单条查询。"""
        embedding = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()[0]
