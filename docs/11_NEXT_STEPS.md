# Next Steps

以下顺序来自当前逆向结果，强调可复现性和科学真实性；不是新增功能路线图。

## Phase 0：建立可复现基线

1. 将 `backend/tests/`、`verification/`、`frontend/__tests__/`、`frontend/e2e/` 中需要的
   测试纳入版本控制，验证 clean clone。
2. 盘点 259 项工作树变更，区分源码、数据、日志、cache、诊断脚本；不丢用户改动地
   建立可审计提交序列。
3. 为 raw SBML、canonical、golden、vector DB 建 manifest：版本、来源、checksum、构建命令。
4. 重新运行 10 pathway real benchmark，记录 commit、flags、Python/Node、optional deps 和 provider。
5. 用新结果刷新 `PROJECT_STATE.json`，不要沿用 README 或旧 report 数字。

## Phase 1：修真实性阻塞项

1. 将 Validation Pyramid 从软门升级为明确的 fail/retry/stop contract。
2. 建立统一 BioModels registry，消除 canonical/specialist/golden ID 冲突。
3. 按失败类型排序修通路：先 numerical explosion/mass conservation，再 peak/order/oscillation，最后 report/evidence。
4. 为每个通路增加 negative benchmark，防止“全部平线也通过”。
5. 统一 pass/partial/fail/skipped/degraded schema，前后端展示不丢方法和证据。

## Phase 2：收敛架构 contract

1. 让 `WORKFLOW_VERSION` 真正选择版本或删除死配置。
2. 统一 pathway identifier，自动校验所有 mapping/YAML。
3. 选定 V4 canonical state，逐步移除 flat/grouped 双写。
4. 将 SSE schema 类型化，修复 error payload 和 double validation emission。
5. 统一 API base，补齐或移除未实现的 SBML/BioModels frontend endpoints。

## Phase 3：降低维护成本

1. 在测试保护下拆分 `main.py`：app/routes/chat SSE/benchmark SSE/SA service。
2. 拆分 `graph_v3.py`：planning/HITL/workers/hooks/build。
3. 拆分 `nodes_v2.py`：entity/mechanism/RAG/ODE/features/evidence/report。
4. 将 Prompt 变为带 owner、consumer、status、schema 的 registry。
5. 将 template routing 收敛到 selector/renderer，不再由 Prompt 和注释复制规则。

## Phase 4：部署与安全

1. 把 `NO_PROXY` 改为显式配置。
2. 明确 sandbox threat model，生产运行采用容器/低权限/资源限制。
3. 对齐 Python/Node 支持矩阵，并在 CI 验证 Docker 版本。
4. 增加 backend healthcheck、compose health dependency、reverse proxy/TLS 指南。

## 成功指标

- clean clone 可以安装并运行所有 required tests。
- 10 pathway benchmark 的结果可由同一 manifest 重复得到。
- 任一 failed pathway 可由 mapping 直接定位到 retriever/IR/template/solver/validation/report。
- AI 首先读取 `PROJECT_STATE.json` 和本组文档，不需扫描 runtime artifacts。
- 新功能不能降低既有 pathway 的真实 benchmark 状态。
