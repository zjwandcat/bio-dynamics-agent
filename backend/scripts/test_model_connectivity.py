#!/usr/bin/env python3
"""大模型连通性探测脚本。

双击 test_models.bat 时调用本脚本，依次检测：
1. 主 LLM / 备用 LLM（轻量调用）
2. Embedding 模型（单条文本 embedding）
3. 所有配置的 Rerank 候选（单文档 rerank）

输出格式为表格，便于快速排查哪个供应商/模型不可用。
"""

import sys
from pathlib import Path

# 将 backend 加入路径，确保能 import app.config
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import (
    OpenRouterEmbeddings,
    OpenRouterRerankClient,
    SiliconFlowEmbeddings,
    SiliconFlowRerankClient,
    XfyunMaasEmbeddings,
    XfyunMaasRerankClient,
    embedding_model,
    llm,
    rerank_manager,
    settings,
)


def _ok(name: str) -> None:
    print(f"  [OK]   {name}")


def _fail(name: str, reason: str) -> None:
    print(f"  [FAIL] {name}: {reason}")


def _section(title: str) -> None:
    print(f"\n{title}")
    print("-" * 60)


def test_llm(provider_name: str, model_name: str, client) -> None:
    """轻量测试 LLM 是否可调用。"""
    try:
        response = client.invoke("reply only 'pong'")
        text = response.content if hasattr(response, "content") else str(response)
        if text and len(text) > 0:
            _ok(f"{provider_name} / {model_name}")
        else:
            _fail(f"{provider_name} / {model_name}", "返回空响应")
    except Exception as exc:
        _fail(f"{provider_name} / {model_name}", str(exc))


def _test_embedding_client(provider: str, model: str, client) -> None:
    """测试指定 embedding 客户端。"""
    try:
        vec = client.embed_query("connectivity test")
        dim = len(vec)
        _ok(f"Embedding / {provider} / {model} (dim={dim})")
    except Exception as exc:
        _fail(f"Embedding / {provider} / {model}", str(exc))


def test_embedding() -> None:
    """测试当前配置的 Embedding 模型。"""
    provider = settings.EMBEDDING_PROVIDER
    if provider == "siliconflow":
        model = settings.SILICONFLOW_EMBEDDING_MODEL
    elif provider == "openrouter":
        model = settings.OPENROUTER_EMBEDDING_MODEL
    elif provider == "xfyun":
        model = settings.XFYUN_MAAS_EMBEDDING_MODEL
    else:
        model = settings.EMBEDDING_MODEL
    _test_embedding_client(provider, model, embedding_model)


def test_all_configured_embeddings() -> None:
    """测试所有配置了 API Key 的 Embedding 模型（不限于当前 provider）。"""
    _section("Embedding (All Configured)")

    # OpenRouter embedding
    if settings.OPENROUTER_API_KEY:
        client = OpenRouterEmbeddings(
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.OPENROUTER_EMBEDDING_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
        )
        _test_embedding_client("OpenRouter", settings.OPENROUTER_EMBEDDING_MODEL, client)
    else:
        print("  (OpenRouter API Key 未配置，跳过)")

    # SiliconFlow embedding
    if settings.SILICONFLOW_API_KEY:
        client = SiliconFlowEmbeddings(
            api_key=settings.SILICONFLOW_API_KEY,
            model=settings.SILICONFLOW_EMBEDDING_MODEL,
            base_url=settings.SILICONFLOW_BASE_URL,
        )
        _test_embedding_client("SiliconFlow", settings.SILICONFLOW_EMBEDDING_MODEL, client)
    else:
        print("  (SiliconFlow API Key 未配置，跳过)")

    # 讯飞 MaaS embedding
    if settings.XFYUN_MAAS_API_KEY:
        client = XfyunMaasEmbeddings(
            api_key=settings.XFYUN_MAAS_API_KEY,
            model=settings.XFYUN_MAAS_EMBEDDING_MODEL,
            base_url=settings.XFYUN_MAAS_EMBEDDING_BASE_URL,
        )
        _test_embedding_client("XfyunMaas", settings.XFYUN_MAAS_EMBEDDING_MODEL, client)
    else:
        print("  (讯飞 MaaS API Key 未配置，跳过)")


def _test_rerank_client(provider: str, model: str, client) -> None:
    """测试指定 rerank 客户端。"""
    try:
        client.rerank("health", ["ok"], top_n=1)
        _ok(f"Rerank / {provider} / {model}")
    except Exception as exc:
        _fail(f"Rerank / {provider} / {model}", str(exc))


def test_rerank_candidates() -> None:
    """测试所有配置了 API Key 的 Rerank 候选（不依赖 RERANK_PROVIDER 开关）。"""
    # OpenRouter rerank models
    if settings.OPENROUTER_API_KEY:
        for model in settings.OPENROUTER_RERANK_MODELS:
            client = OpenRouterRerankClient(
                api_key=settings.OPENROUTER_API_KEY,
                model=model,
                base_url=settings.OPENROUTER_BASE_URL,
            )
            _test_rerank_client("OpenRouter", model, client)
    else:
        print("  (OpenRouter API Key 未配置，跳过)")

    # SiliconFlow rerank models
    if settings.SILICONFLOW_API_KEY:
        for model in settings.SILICONFLOW_RERANK_MODELS:
            client = SiliconFlowRerankClient(
                api_key=settings.SILICONFLOW_API_KEY,
                model=model,
                base_url=settings.SILICONFLOW_BASE_URL,
            )
            _test_rerank_client("SiliconFlow", model, client)
    else:
        print("  (SiliconFlow API Key 未配置，跳过)")

    # 讯飞 MaaS rerank models
    if settings.XFYUN_MAAS_API_KEY:
        for model in settings.XFYUN_MAAS_RERANK_MODELS:
            client = XfyunMaasRerankClient(
                api_key=settings.XFYUN_MAAS_API_KEY,
                model=model,
                base_url=settings.XFYUN_MAAS_RERANK_BASE_URL,
            )
            _test_rerank_client("XfyunMaas", model, client)
    else:
        print("  (讯飞 MaaS API Key 未配置，跳过)")


def main() -> int:
    print("=" * 60)
    print("BioDynamics Agent - 大模型连通性探测")
    print("=" * 60)

    _section("LLM")
    test_llm("Primary LLM", settings.OPENAI_MODEL, llm.primary)
    if llm.backup is not None:
        test_llm("Backup LLM", settings.BACKUP_MODEL or "backup", llm.backup)
    else:
        print("  (未配置备用 LLM)")

    _section("Embedding (Current)")
    test_embedding()

    test_all_configured_embeddings()

    _section("Rerank Candidates (All Configured)")
    test_rerank_candidates()

    print("\n" + "=" * 60)
    print("探测完成。若上方出现 [FAIL]，请检查对应 API Key 与网络。")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
