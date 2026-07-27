# Known Problems

本文记录已由当前代码或磁盘证据确认的问题，不等同于未来路线图。状态可能随工作树
变化；修复后应附测试/benchmark 证据并更新本表。

## 科学与 benchmark

| ID | Problem | Evidence | Likely ownership | 状态 |
|---|---|---|---|---|
| KP-01 | 当前磁盘 10 通路 `12_check_report.json` 均 `overall_passed=false` | `backend/data/sa_logs/all_10_pathways/*/12_check_report.json` | 全 pipeline | open，未在本次重跑 |
| KP-02 | EGFR peak time 失败 | `EGFR_RTK/12_check_report.json` | EGFR specialist、parameter、time scale、template | open |
| KP-03 | MAPK peak/order/adaptation 失败 | `MAPK_ERK/12_check_report.json` | MAPK feedback/DUSP kinetics、template | open |
| KP-04 | PI3K mass conservation + dynamics 失败 | `PI3K_AKT_mTOR/12_check_report.json` | reaction topology、ODE source/sink | open |
| KP-05 | TGF-beta、WNT、Cell Cycle 数值爆炸 | 对应 `12_check_report.json` | initial conditions、units、template kinetics、solver | open |
| KP-06 | p53/NF-kB oscillation、steady state、ordering 失败 | 对应报告 | delay/feedback template、DDE fallback | open |
| KP-07 | canonical、specialist comments、golden fixtures 中部分 BioModels ID 口径不一致 | `knowledge/canonical`, `pathways/specialists`, `benchmarks/golden` | scientific data governance | open |
| KP-08 | Pipeline 完成可能被误解为科学通过 | report 总会生成；validation 是软门 | validation/report contract | open |

## 架构与契约

| ID | Problem | Evidence | 影响 |
|---|---|---|---|
| KP-09 | `WORKFLOW_VERSION` 是死配置 | `main.py` 固定导入/运行 `compiled_workflow_v3` | 改 env 不切版本 |
| KP-10 | Validation Pyramid 在 report 后运行且为软门 | `graph_v3.py::build_workflow_v3` | 失败不会自动回到 simulator |
| KP-11 | `v4_validation_report` 可能发两次 | `main.py::_v3_event_stream` + `_emit_worker_outputs` | 前端必须以后一次覆盖 |
| KP-12 | workflow exception SSE data 是 object，前端 error case 主要按 string 处理 | `main.py` error branch、`frontend/lib/store.ts` | UI 可能显示“未知错误” |
| KP-13 | Dynamic router Agent 输出主要用于 dispatch，不能假设替代主 worker | `agent_orchestration/dynamic_router.py`、图 hook 位置 | 容易错误修改非生产路径 |
| KP-14 | V4 state 有 flat/grouped 双表示 | `state.py` | 单边写入会产生状态漂移 |

## API、前端与部署

| ID | Problem | Evidence | 影响 |
|---|---|---|---|
| KP-15 | 前端引用未实现的 `/api/v4/sbml/import`、`/api/v4/biomodels/{id}` | `SbmlUpload.tsx`, `BioModelsFetcher.tsx`, 后端 route 列表 | 高级输入模式失败 |
| KP-16 | `ControlBar.tsx` 硬编码 `http://localhost:8000` | component source | 绕过统一 API base |
| KP-17 | Compose runtime 注入 `NEXT_PUBLIC_API_BASE`，Next public env 常在 build 时内联 | Docker/Next build behavior | 远程部署可能请求浏览器 localhost |
| KP-18 | `scripts/demo_benchmark.sh` 检查不存在的 `/health` | script + FastAPI routes | 误判并重复启动 8000 |
| KP-19 | `scripts/demo.sh` 使用 `/workspace?pathway=...`，该路由目前 redirect `/advanced` | route source | demo query 不再生效 |
| KP-20 | `main.py` 强制 `NO_PROXY=*` | import side effect | 必须代理的外部请求失败 |
| KP-21 | CORS 只允许一个 `FRONTEND_URL` | `main.py` middleware | 仅适合单前端/研究部署 |

## 测试与版本资产

| ID | Problem | Evidence | 影响 |
|---|---|---|---|
| KP-22 | `.gitignore` 排除全部 backend/frontend/verification tests | `.gitignore` | clean clone/CI 缺验证资产 |
| KP-23 | Makefile/CI 又引用被 ignore 的 tests | `Makefile`, `.github/workflows/ci.yml` | 本机存在不代表远端存在 |
| KP-24 | scientific regression workflow 可能使用多余 repo directory prefix | `.github/workflows/scientific-regression.yml` | CI cwd 错误 |
| KP-25 | Playwright 自动启动 frontend，不启动真实 backend | `frontend/playwright.config.ts` | API E2E 需额外服务/mock |
| KP-26 | 工作树与 HEAD/远端不一致且存在大量 untracked 运行/源码资产 | `git status` | reset/pull 可能丢实现 |

## 安全边界

| ID | Problem | Evidence | 影响 |
|---|---|---|---|
| KP-27 | sandbox 是临时目录 + subprocess + denylist/AST/timeout，不是容器隔离 | `sandbox.py` | LLM code execution 仍是高风险边界 |
| KP-28 | optional dependency 降级改变科学算法 | `config.py` try-import matrix | 同一代码不同机器结果不同 |

## 关闭问题的要求

关闭科学问题必须提供：复现输入、commit/worktree 标识、flags、依赖版本、修复前后
结构化 benchmark、CSV/metrics 和 regression test。仅截图、报告 markdown、HTTP 200 或
“没有异常”不足以关闭问题。
