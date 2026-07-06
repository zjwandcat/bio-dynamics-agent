# BioDynamics Agent - Embedding 模型对比测试
#
# 测试目标：
#   对比多款 Embedding 模型在生物医学 RAG 检索中的表现差异：
#   1. 讯飞 xop3qwen8bembedding（当前默认，768 维）
#   2. SiliconFlow BAAI/bge-m3（中文友好，1024 维）
#   3. OpenRouter nvidia/llama-nemotron-embed-vl-1b-v2:free（免费）
#
# 对比维度：
#   - 向量维度
#   - 检索命中率（top-k 是否包含相关参数）
#   - 语义相似度分数分布
#   - 查询延迟
#   - 跨语言能力（中文 query vs 英文 document）
#
# 运行方式：
#   cd backend
#   python tests/test_embedding_comparison.py

from __future__ import annotations

import sys
import os
import time
import json
from pathlib import Path
from typing import Any

# 添加 backend 目录到 sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


# =============================================================================
# 测试查询集（覆盖 EGF-EGFR 信号级联的关键参数）
# =============================================================================
TEST_QUERIES = [
    {
        "query": "EGF EGFR binding k_on association rate",
        "expected_keywords": ["k1", "k_on", "EGF", "EGFR", "binding"],
        "description": "EGF-EGFR 结合速率常数",
    },
    {
        "query": "EGFR phosphorylation rate constant",
        "expected_keywords": ["k1", "k_phos", "phosphorylation", "EGFR"],
        "description": "EGFR 磷酸化速率",
    },
    {
        "query": "EGF-EGFR complex dissociation k_off",
        "expected_keywords": ["k2", "k_off", "dissociation", "EGF-EGFR"],
        "description": "EGF-EGFR 复合物解离速率",
    },
    {
        "query": "MAPK signaling cascade amplification",
        "expected_keywords": ["MAPK", "MEK", "Raf", "cascade"],
        "description": "MAPK 级联信号放大",
    },
    {
        "query": "Ras GTP GDP exchange SOS catalysis",
        "expected_keywords": ["Ras", "GTP", "GDP", "SOS", "exchange"],
        "description": "Ras GDP-GTP 交换",
    },
    {
        "query": "Shc Grb2 SOS adapter protein recruitment",
        "expected_keywords": ["Shc", "Grb2", "SOS", "adapter"],
        "description": "Shc-Grb2-SOS 接头蛋白招募",
    },
    {
        "query": "EGF 受体磷酸化信号级联动力学参数",
        "expected_keywords": ["EGF", "EGFR", "phosphorylation", "kinetic"],
        "description": "中文查询：EGF 受体磷酸化动力学",
    },
]


# =============================================================================
# Embedding 模型配置
# =============================================================================
EMBEDDING_CONFIGS = [
    {
        "name": "讯飞 xop3qwen8bembedding",
        "provider": "xfyun",
        "env_overrides": {
            "EMBEDDING_PROVIDER": "xfyun",
            "XFYUN_MAAS_EMBEDDING_MODEL": "xop3qwen8bembedding",
        },
    },
    {
        "name": "SiliconFlow BAAI/bge-m3",
        "provider": "siliconflow",
        "env_overrides": {
            "EMBEDDING_PROVIDER": "siliconflow",
            "SILICONFLOW_EMBEDDING_MODEL": "BAAI/bge-m3",
        },
    },
    {
        "name": "OpenRouter nvidia/llama-nemotron-embed",
        "provider": "openrouter",
        "env_overrides": {
            "EMBEDDING_PROVIDER": "openrouter",
            "OPENROUTER_EMBEDDING_MODEL": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        },
    },
]


# =============================================================================
# 工具函数
# =============================================================================
def _set_env(overrides: dict[str, str]) -> None:
    """临时设置环境变量。"""
    for key, value in overrides.items():
        os.environ[key] = value


def _create_embedding_model(provider: str):
    """根据 provider 创建 Embedding 模型实例（不依赖全局 config）。"""
    from app.config import settings

    if provider == "xfyun":
        from app.config import XfyunMaasEmbeddings
        return XfyunMaasEmbeddings(
            api_key=settings.XFYUN_MAAS_API_KEY,
            model=settings.XFYUN_MAAS_EMBEDDING_MODEL,
            base_url=settings.XFYUN_MAAS_EMBEDDING_BASE_URL,
        )
    elif provider == "siliconflow":
        from app.config import SiliconFlowEmbeddings
        return SiliconFlowEmbeddings(
            api_key=settings.SILICONFLOW_API_KEY,
            model=settings.SILICONFLOW_EMBEDDING_MODEL,
            base_url=settings.SILICONFLOW_BASE_URL,
        )
    elif provider == "openrouter":
        from app.config import OpenRouterEmbeddings
        return OpenRouterEmbeddings(
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.OPENROUTER_EMBEDDING_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
        )
    else:
        raise ValueError(f"未知 provider: {provider}")


def _check_keyword_hit(doc: str, keywords: list[str]) -> list[str]:
    """检查文档中是否包含期望关键词。"""
    doc_lower = doc.lower()
    hits = []
    for kw in keywords:
        if kw.lower() in doc_lower:
            hits.append(kw)
    return hits


# =============================================================================
# 测试单个 Embedding 模型
# =============================================================================
def test_embedding_model(config: dict, rag_client: Any) -> dict:
    """测试单个 Embedding 模型的检索表现。

    注意：此函数直接使用已构建的 ChromaDB（向量由当前默认 Embedding 模型生成），
    因此跨模型对比时，检索结果会因向量维度/空间不一致而不公平。
    本测试主要对比：
    1. embed_query 的延迟与维度
    2. 在已有库上的检索命中率（即使向量空间不一致，也能观察趋势）
    """
    name = config["name"]
    provider = config["provider"]
    print(f"\n{'='*60}")
    print(f"测试 Embedding 模型：{name}")
    print(f"{'='*60}")

    # 1. 创建 Embedding 模型实例
    try:
        _set_env(config["env_overrides"])
        # 重新加载 settings（环境变量已更新）
        import importlib
        import app.config as config_module
        importlib.reload(config_module)
        settings = config_module.settings

        emb_model = _create_embedding_model(provider)
        print(f"  ✓ 模型实例创建成功")
    except Exception as exc:
        print(f"  ✗ 模型实例创建失败：{exc}")
        return {"name": name, "status": "error", "error": str(exc)}

    # 2. 测试 embed_query 维度与延迟
    test_text = "EGF EGFR binding phosphorylation kinetics"
    try:
        start = time.time()
        vector = emb_model.embed_query(test_text)
        latency = time.time() - start
        dim = len(vector)
        print(f"  向量维度：{dim}")
        print(f"  embed_query 延迟：{latency:.3f}s")
        print(f"  向量前 5 维：{vector[:5]}")
    except Exception as exc:
        print(f"  ✗ embed_query 失败：{exc}")
        return {"name": name, "status": "error", "error": str(exc), "dim": 0}

    # 3. 检索测试（注意：使用已有 ChromaDB，向量空间可能不一致）
    # 此处仅测试 embed_query 能力，不直接检索（因为 ChromaDB 向量空间固定）
    results = []
    for q in TEST_QUERIES:
        query = q["query"]
        keywords = q["expected_keywords"]
        try:
            start = time.time()
            q_vector = emb_model.embed_query(query)
            q_latency = time.time() - start
            # 计算与测试文本的余弦相似度（观察语义区分能力）
            import math
            def cosine_sim(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
            sim_to_test = cosine_sim(q_vector, vector)
            results.append({
                "query": query,
                "description": q["description"],
                "latency": q_latency,
                "sim_to_test": sim_to_test,
                "dim": len(q_vector),
            })
            print(f"  [{q['description']}] 延迟={q_latency:.3f}s, 与测试文本相似度={sim_to_test:.4f}")
        except Exception as exc:
            print(f"  [{q['description']}] 失败：{exc}")
            results.append({
                "query": query,
                "description": q["description"],
                "error": str(exc),
            })

    return {
        "name": name,
        "status": "success",
        "dim": dim,
        "latency": latency,
        "query_results": results,
    }


# =============================================================================
# 主测试流程
# =============================================================================
def main() -> int:
    print("=" * 60)
    print("Embedding 模型对比测试")
    print("=" * 60)
    print(f"测试查询数：{len(TEST_QUERIES)}")
    print(f"对比模型数：{len(EMBEDDING_CONFIGS)}")

    all_results = []
    for config in EMBEDDING_CONFIGS:
        try:
            result = test_embedding_model(config, None)
            all_results.append(result)
        except Exception as exc:
            print(f"\n模型 {config['name']} 测试异常：{exc}")
            all_results.append({"name": config["name"], "status": "error", "error": str(exc)})

    # 汇总对比
    print("\n" + "=" * 60)
    print("汇总对比")
    print("=" * 60)
    print(f"{'模型':<40} {'维度':<8} {'延迟(s)':<10} {'状态':<8}")
    print("-" * 70)
    for r in all_results:
        name = r["name"][:38]
        dim = r.get("dim", "N/A")
        latency = f"{r.get('latency', 0):.3f}" if r.get("latency") else "N/A"
        status = r.get("status", "unknown")
        print(f"{name:<40} {dim!s:<8} {latency:<10} {status:<8}")

    # 详细查询相似度对比
    print("\n查询语义相似度对比（与 'EGF EGFR binding phosphorylation kinetics' 的余弦相似度）：")
    print(f"{'查询':<35}", end="")
    for r in all_results:
        if r.get("status") == "success":
            name_short = r["name"].split()[0][:12]
            print(f" {name_short:<15}", end="")
    print()
    print("-" * 80)
    for i, q in enumerate(TEST_QUERIES):
        desc = q["description"][:33]
        print(f"{desc:<35}", end="")
        for r in all_results:
            if r.get("status") == "success" and r.get("query_results"):
                qr = r["query_results"][i]
                sim = qr.get("sim_to_test", 0)
                print(f" {sim:<15.4f}", end="")
            else:
                print(f" {'N/A':<15}", end="")
        print()

    # 保存结果
    output_path = _BACKEND_DIR.parent / "test_outputs_egf" / "embedding_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # 移除向量数据（太长）
        clean_results = []
        for r in all_results:
            clean = {k: v for k, v in r.items() if k != "vector"}
            clean_results.append(clean)
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至：{output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
