# Configuration Reference

配置真相源是 `backend/app/config.py::Settings`。它固定加载 `backend/.env`，并在 import
时构造全局 LLM、embedding 和 rerank client；修改 `.env` 后需要重启进程。

`backend/.env.example` 只列常用项，不是完整清单。不要读取、输出或提交真实 `.env`。

## LLM 与服务

| Key | 默认值 | 作用 |
|---|---|---|
| `OPENAI_API_KEY` | empty -> runtime placeholder | 主 OpenAI-compatible credential |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 主 LLM endpoint |
| `OPENAI_MODEL` | `gpt-4o` | 主模型 |
| `BACKUP_API_KEY/BASE_URL/MODEL` | empty | 备用 LLM，配置完整才初始化 |
| `HOST` | `0.0.0.0` | backend bind host |
| `PORT` | `8000` | backend port |
| `FRONTEND_URL` | `http://localhost:3000` | CORS allow origin、OpenRouter header |
| `LOG_LEVEL` | `INFO` | Python log level |
| `LOG_JSON` | `true` | JSON structured logging |

主/备 ChatOpenAI 当前代码固定 `temperature=0.2`、`max_retries=0`，不是 env 配置。

## Chroma 与 embedding

| Key | 默认值 |
|---|---|
| `CHROMA_PERSIST_DIR` | `backend/data/vector_db` |
| `CHROMA_COLLECTION_NAME` | `biodynamics_params` |
| `CHROMA_COLLECTION_MECHANISM` | `biodynamics_mechanism` |
| `CHROMA_COLLECTION_PARAMETER` | `biodynamics_parameter` |
| `CHROMA_COLLECTION_EXPERIMENT` | `biodynamics_experiment` |
| `CHROMA_COLLECTION_EVIDENCE` | `biodynamics_evidence` |
| `EMBEDDING_PROVIDER` | `openai`，支持 `local/openrouter/siliconflow/xfyun` |
| `EMBEDDING_MODEL` | openai: `text-embedding-3-small`; local: MiniLM |
| `EMBEDDING_BASE_URL/API_KEY` | empty，回退主 LLM provider |

## Rerank providers

| Key | 默认值 |
|---|---|
| `RERANK_PROVIDER` | `model`，兼容 `rule/hybrid/openrouter` |
| `RERANK_PROVIDERS` | `xfyun,openrouter,siliconflow` |
| `RERANK_SELECTION_MODE` | `auto` |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `OPENROUTER_API_KEY` | empty |
| `OPENROUTER_EMBEDDING_MODEL` | `nvidia/llama-nemotron-embed-vl-1b-v2:free` |
| `OPENROUTER_RERANK_MODEL` | `cohere/rerank-4-pro`，legacy alias |
| `OPENROUTER_RERANK_MODELS` | Cohere pro + NVIDIA free fallback |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1` |
| `SILICONFLOW_API_KEY` | empty |
| `SILICONFLOW_EMBEDDING_MODEL` | `BAAI/bge-m3` |
| `SILICONFLOW_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3`，legacy alias |
| `SILICONFLOW_RERANK_MODELS` | `BAAI/bge-reranker-v2-m3` |
| `XFYUN_MAAS_*_BASE_URL` | `https://maas-api.cn-huabei-1.xf-yun.com/v2` |
| `XFYUN_MAAS_API_KEY` | empty |
| `XFYUN_MAAS_EMBEDDING_MODEL` | `xop3qwen8bembedding` |
| `XFYUN_MAAS_RERANK_MODEL` | `xop3qwen8breranker`，legacy alias |
| `XFYUN_MAAS_RERANK_MODELS` | `xop3qwen8breranker` |

每个 provider 还需要对应 `*_API_KEY`。不得把 key 写进文档/fixture。

## 外部数据与 RAG

| Key | 默认值 | 说明 |
|---|---|---|
| `NCBI_EMAIL` / `NCBI_API_KEY` | empty | PubMed E-utilities contact/rate limit |
| `MCP_ENABLED` | `true` | MCP terminology master switch |
| `MCP_OPENBIOMED_URL` | empty | OpenBioMed MCP |
| `MCP_MEDTERM_URL` | empty | medical terminology MCP |
| `MCP_PUBMED_URL` | empty | PubMed MCP |
| `MCP_UMLS_URL` | empty | UMLS MCP |
| `RAG_ONLINE_FALLBACK` | `true` | 低命中时外部数据库补充 |
| `RAG_ONLINE_FALLBACK_THRESHOLD` | `0.3` | trigger threshold |
| `RAG_ONLINE_QUERY_TIMEOUT` | `10.0s` | 单查询 timeout |
| `RAG_ONLINE_TOTAL_BUDGET` | `600.0s` | workflow online budget |

`main.py` 还会无条件设置 `NO_PROXY=*`，该行为当前不是 env switch。

## Sandbox / solver / metrics

| Key | 默认值 | 说明 |
|---|---|---|
| `SANDBOX_TIMEOUT` | `60s` | subprocess timeout |
| `SANDBOX_MAX_STEP_RATIO` | `0.01` | max step / T_END |
| `SANDBOX_AUDIT_LOG` | `true` | audit log switch |
| `SANDBOX_AUDIT_LOG_DIR` | `data/sandbox_logs` | 相对 backend cwd |
| `DEGRADATION_MODE` | `full` | `full/rag_only/template_only` |
| `METRICS_BACKEND` | `log` | `prometheus/log/off` |
| `METRICS_LOG_DIR` | `data/metrics` | metrics path |

其他硬编码运行参数：chat `recursion_limit=50`；clarification timeout 120s；dynamic
router fail-safe 默认 30s；sandbox retries 为 Fast 1、Standard/Manual 3。

## V4 coarse flags

| Key | 默认值 | 覆盖范围 |
|---|---|---|
| `V4_SCIENTIFIC_LAYER_ENABLED` | false | ontology、IR、graph、ODE v2、planner、specialist、crosstalk |
| `V4_VALIDATION_ENABLED` | false | SBML、pyramid、calibration |
| `V4_HYPOTHESIS_ENABLED` | false | hypothesis、dynamic routing |

## V4 fine flags

全部默认 false，除非另述：

- `V4_ONTOLOGY_AGENT_ENABLED`
- `V4_PATHWAY_GRAPH_ENABLED`
- `V4_REACTION_IR_ENABLED`
- `V4_REACTION_IR_ADAPTER_ENABLED`
- `V4_ODE_TEMPLATE_V2_ENABLED`
- `V4_PATHWAY_PLANNER_ENABLED`
- `V4_PATHWAY_SPECIALIST_ENABLED`
- `V4_CROSSTALK_COORDINATOR_ENABLED`
- `V4_SPECIALIST_KG_FEEDBACK_ENABLED`
- `V4_SPECIALIST_KG_WRITEBACK_MODE=none`，可用 `mode_a/mode_b/both`
- `V4_SBML_GROUNDER_ENABLED`
- `V4_VALIDATION_PYRAMID_ENABLED`
- `V4_CALIBRATION_AGENT_ENABLED`
- `V4_HYPOTHESIS_AGENT_ENABLED`
- `V4_DYNAMIC_ROUTING_ENABLED`

解析优先级在 `Settings._resolve_v4_flag`：显式 fine env > coarse on > fine attribute。

## Scientific Alignment

总开关：`V4_SCIENTIFIC_ALIGNMENT_ENABLED=false`。子开关均默认 false：

`SA_MECHANISM_GRAPH`, `SA_PARAMETER_PRIOR`, `SA_BIOMODELS_ORACLE`,
`SA_EVIDENCE_FUSION`, `SA_SEVEN_AXIS`, `SA_LOOP_TERMINATION`, `SA_CANONICAL`,
`SA_CONSISTENCY_CHECKER`, `SA_PARAMETER_CONFIDENCE`, `SA_SCIENTIFIC_CRITIC`,
`SA_MULTI_DIM_CONFIDENCE`, `SA_SPRINT1_GROUND_TRUTH`,
`SA_SPRINT2_EVIDENCE_RENDERER`, `SA_SPRINT3_CONSISTENCY_GATE`,
`SA_SPRINT4_EXPERIMENT_RULE_ENGINE`, `SA_SPRINT5_PROVENANCE_EXPLAINABILITY`,
`SA_BIOMODELS_CALIBRATION`。

Reviewer/RAG flags：

- `V4_SCIENTIFIC_REVIEWER_ENABLED=false`
- `V4_SEQUENTIAL_RETRIEVER=true`（重要例外：默认开启）
- `V4_LEGACY_CRITIC=false`
- `V4_LEGACY_SEVEN_AXIS=false`
- `RAG_LEGACY_PARALLEL=false`

## Benchmark

| Key | 默认值 | 含义 |
|---|---|---|
| `BENCHMARK_REAL_ORCHESTRATOR` | false | 使用真实 LangGraph/SA chain |
| `BENCHMARK_LEGACY_SYNTHETIC` | false | 显式允许 deprecated synthetic path |

## Dead / misleading configuration

`WORKFLOW_VERSION=v3` 存在于 Settings，但当前 `/api/chat` 没有用它选择图。不要通过
修改该 key 期望切换 v1/v2/v3。

## Frontend

`NEXT_PUBLIC_API_BASE` 在 `frontend/lib/api.ts` 读取，默认
`http://localhost:8000`。Next public env 通常 build-time 内联；Docker 远程部署必须验证
构建产物中的实际值。`ControlBar.tsx` 当前还有独立硬编码，应视为已知问题。
