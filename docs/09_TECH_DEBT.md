# Technical Debt

## P0：影响真实性或可复现性

| Debt | 原因 | 偿还标准 |
|---|---|---|
| Validation soft gate | 失败后仍能生成完成报告 | 定义阻断/重试策略；测试失败不会被呈现为 pass |
| 10 pathway quantitative failures | 动态、守恒、爆炸未稳定 | 每通路真实 benchmark + regression，禁止 synthetic pass |
| BioModels ID 多来源不一致 | specialist/canonical/golden 独立维护 | 单一 registry + schema validation + ID provenance |
| tests 被 ignore | 验证资产不属于版本库 | 测试和 fixture 纳入 Git，clean clone 可运行 |
| working tree 大量未提交源码 | HEAD 不能代表运行版本 | 分批审计/提交，不丢用户改动，建立 release baseline |

## P1：增加维护成本

| Debt | 表现 | 建议 |
|---|---|---|
| `main.py`, `graph_v3.py`, `nodes_v2.py`, `config.py` 巨型文件 | ownership 和契约集中 | 先补契约测试，再按 API/graph/nodes/providers 拆分 |
| v1/v2/v3/v4 命名并存 | 版本含义和生产路径不直观 | 明确 active/legacy/experimental，移除死 version flag |
| V4 flat + grouped state | 每次写入需要双同步 | 选单一 canonical state，adapter 仅在边界存在 |
| 多套 pathway key | 大小写/后缀/别名不一致 | central PathwayId enum + mapping validation |
| 多套 validation status | pass/skipped/degraded/partial/fail | 定义统一三态/五态 schema，不丢方法和证据 |
| Prompt 内联分散 | 难查 active/deprecated | 按 `PROMPT_INDEX.md` 建 registry 或 metadata |
| 模板路由重复 | Prompt、selector、renderer 各有规则 | selector 为唯一真相源，Prompt 只建议不裁决 |
| API base 分散 | hardcoded localhost | 统一 `frontend/lib/api.ts` 或同源 proxy |

## P2：运行与部署债务

- `NO_PROXY=*` import side effect 应变为显式配置。
- Docker/CI/本机 Python、Node 版本不一致，应建立支持矩阵。
- Compose 缺 healthcheck、reverse proxy、TLS、resource limit。
- frontend public env 需要 build-time 注入或 runtime config。
- demo scripts 和实际 routes 已漂移，需要契约测试。
- 向量库、SBML、benchmark 数据缺版本 manifest/checksum。
- 运行目录中混有源码、cache、日志、corrupted DB backup 和导出文件。

## P3：文档与治理债务

- 旧 `ARCHITECTURE.md` 描述 v2，容易误导生产路径探索。
- README 的 benchmark 状态与当前磁盘报告冲突，缺 `measured_at`/commit/flags。
- Scientific claims 缺统一证据等级和有效期。
- 设计决策散在注释、审计报告和 Prompt 中，需持续维护 `DESIGN_DECISIONS.md`。

## 偿债顺序

1. 固化可复现状态：tests、fixtures、data manifest、Git baseline。
2. 修真实性：validation gate、benchmark、BioModels registry。
3. 收敛 contract：pathway ID、state、status、SSE error schema。
4. 最后拆文件/重构目录；在前 3 步前做大重构会丢失科学行为基线。
