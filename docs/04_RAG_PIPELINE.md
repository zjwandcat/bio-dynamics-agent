# RAG Pipeline

## 目标

RAG 为机制、参数、实验和证据提供可追溯候选；它不是 ODE 的最终裁决器。参数进入
`state["parameters"]` 后，N6 和模板才会将其转成方程中的数值。

## 检索链

```text
user input / edge query
  -> optional query rewrite
  -> Chroma semantic search
  -> BM25 keyword search
  -> merge + deduplicate + hybrid score
  -> provider rerank
  -> source authority / species specificity weighting
  -> selected candidates
  -> online fallback when hit rate is low
  -> state parameters / evidence
```

主要实现：`backend/app/rag_client.py::RagClient`、
`backend/app/rag_collections.py`、`backend/app/local_embeddings.py`。

## 四路 collection

| collection | 内容 | 主要消费者 |
|---|---|---|
| `biodynamics_mechanism` | 机制/通路片段 | N3 mechanism RAG |
| `biodynamics_parameter` | Kd/Km/Vmax/速率等 | N5 parameter RAG |
| `biodynamics_experiment` | 实验协议和 readout | N9 |
| `biodynamics_evidence` | 文献/证据 | N10、SA evidence |

旧 `biodynamics_params` collection 仍存在兼容逻辑。持久化目录是
`backend/data/vector_db/`，属于可重建运行资产，不是稳定源码。

## 数据源与构建脚本

| 数据源 | 运行入口 | 说明 |
|---|---|---|
| BioModels/SBML | `scripts/fetch_rag_data.py`, `biomodels_client.py` | raw/processed SBML 和参数 |
| PubMed | `mcp_client.py`, `nodes_v2.py` N10 fallback | 需要网络和 NCBI 设置 |
| KEGG/Reactome/UniProt/ChEMBL | `bio_db_client.py` | 在线 fallback |
| Chroma embedding | `scripts/embed_data.py`, `seed_collections.py` | 本地向量化/四路迁移 |
| 增量更新 | `scripts/update_vector_db.py`, `extend_rag_db.py` | 会修改 `backend/data` |

## 降级路径

- `MCP_ENABLED=false`：跳过术语 MCP，主流程继续。
- Chroma 不可用：降级到内存检索/简化路径。
- 本地命中率低于 `RAG_ONLINE_FALLBACK_THRESHOLD`：查询在线数据库，受 timeout/budget 限制。
- Rerank provider 按 `RERANK_PROVIDERS` 顺序尝试，失败后继续 fallback。
- Fast mode 不做完整检索，使用 `FAST_MODE_ESTIMATED` 参数占位，不能当科学证据。

## 参数 contract

N5 输出每条边的参数对象通常包括：

```json
{
  "edge_key": "Source->Target",
  "param_name": "kd",
  "value": 5.0,
  "unit": "nM",
  "source": "PMID:... or fallback marker",
  "confidence": "HIGH|MEDIUM|LOW",
  "is_fallback": false
}
```

`source`、`confidence`、`is_fallback` 会影响 report、SA provenance 和 benchmark 解释。
无命中时必须标记 fallback，不能把估算值伪装成文献参数。

## 维护入口

| 问题 | 文件 |
|---|---|
| query rewrite/混合评分 | `rag_client.py::rewrite_query`, `hybrid_search` |
| 参数选择/单位归一化 | `rag_client.py::normalize_param`, `nodes_v2.py::n5_parameter_rag` |
| collection 名称/懒单例 | `config.py`, `rag_collections.py` |
| rerank provider | `config.py::RerankManager`, provider client classes |
| 在线 fallback | `rag_client.py`, `bio_db_client.py` |
| 实验规则 | `scientific_alignment/experiment_planner.py`, `knowledge/experiments/` |
| 证据融合 | `scientific_alignment/evidence_fuser.py`, `evidence_ranker.py` |

## 已知边界

RAG “返回候选”不等于“返回正确通路参数”。参数质量需要以 pathway benchmark、
BioModels 对比和 provenance 检查确认；不要仅凭 `rag_hit_rate` 或 HTTP 200 判定成功。
