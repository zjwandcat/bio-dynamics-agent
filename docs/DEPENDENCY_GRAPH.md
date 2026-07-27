# Dependency Graph

## Runtime dependency graph

```text
FastAPI main
  -> Settings / providers
  -> compiled_workflow_v3
      -> BioDynamicsState
      -> v1 compatibility nodes
      -> v2 N0-N11 nodes
      -> V4 hooks
          -> ontology
          -> pathway planner / specialists / crosstalk
          -> Reaction IR / Pathway Graph / ODE Renderer
          -> SBML / calibration / sensitivity / validation / hypothesis
      -> sandbox / solver
  -> SSE adapter
  -> frontend SSE parser
  -> Zustand store
  -> UI components
```

## Scientific dependency graph

```text
Ontology + Terminology
  -> Mechanism / Entity normalization
  -> Reaction / Knowledge Graph
      -> RAG query context
      -> Pathway recognition / Specialist
      -> Reaction IR
  -> Parameter selection + provenance
  -> ODE template selection / rendering
  -> Solver / time series
  -> Metrics
      -> Validation Pyramid
      -> Experiment planning
      -> Evidence retrieval
      -> Scientific report
      -> Scientific Alignment / Benchmark
```

## Blast radius matrix

| 修改模块 | 直接消费者 | 可能破坏 |
|---|---|---|
| `config.py` | 全 backend imports | 服务启动、providers、flags、benchmark mode |
| `state.py` | LangGraph 全节点、main SSE | reducer、跨请求污染、V4 state |
| N1 entity | N2/N4/RAG/ontology | 所有下游 node naming |
| N2 mechanism | KG、template selector、specialist | topology、template、report |
| N5 parameter | N6、solver、provenance | dynamics、units、confidence |
| Reaction IR | PathwayGraph、V4 renderer、validation | mechanism conservation |
| Specialist | KG writeback、V4 ODE、benchmark | 单通路和 cross-talk |
| v3 template | sandbox、metrics、validator | 使用该 template 的多通路 |
| v4 template/renderer | Mode B sandbox | mapped pathway 的数值结果 |
| sandbox | 所有 simulations | security、timeout、CSV/image、retries |
| metrics | validation、report、benchmark | peak/order/score/claims |
| validation | UI、acceptance、benchmark | pass/fail semantics |
| SSE event | frontend store | UI state、progress、errors |
| pathway ID mapping | REST、SA、specialist、frontend | routing and fixture lookup |
| canonical YAML | consistency、benchmark、report | scientific expected behavior |

## Optional dependency branches

| Dependency | 主路径 | 缺失时 |
|---|---|---|
| RoadRunner | SBML time-series validation | structural similarity Track B |
| lmfit | calibration | scipy least squares |
| SALib | Sobol/Morris | local sensitivity only |
| lxml | strict/faster SBML XML | stdlib ElementTree |
| jitcdde | DDE pathways | ODE approximation |
| chromadb | persistent retrieval | in-memory/simplified fallback |

## External service graph

```text
LLM: OpenAI-compatible primary -> optional backup
Embedding: openai | local | openrouter | siliconflow | xfyun
Rerank: xfyun -> openrouter -> siliconflow -> heuristic fallback
Evidence: local Chroma -> PubMed / KEGG / Reactome / UniProt / ChEMBL
SBML: local raw files <-> BioModels API
Terminology: MCP OpenBioMed / UMLS / MedTerm / PubMed
```

## Change strategy

从上游改动时扩大测试范围。entity/mechanism/state/config 是高 blast radius；单个
pathway specialist/YAML 是较窄边界；report CSS 是低科学 blast radius，但 SSE/store
仍可能是共享边界。
