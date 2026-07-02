# BioDynamics Agent

> 把生物医学研究者的**定性假说**（Qualitative Hypothesis）一键转化为**定量 ODE 仿真**（Ordinary Differential Equation, 常微分方程），并直接给出可视化预测与机理论证 —— 全栈 AI 智能体，覆盖 Web 前后端 + LangGraph 工作流 + RAG 知识注入 + 沙箱安全执行。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-green)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688)](https://fastapi.tiangolo.com/)

---

## 目录

- [项目简介](#项目简介)
- [它能解决什么问题](#它能解决什么问题)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [前置环境](#前置环境)
  - [一键启动（Windows）](#一键启动windows)
  - [手动启动](#手动启动)
  - [环境变量配置](#环境变量配置)
- [使用指南](#使用指南)
- [LangGraph 工作流](#langgraph-工作流)
- [RAG 知识库与 SBML 复用](#rag-知识库与-sbml-复用)
- [安全沙箱说明](#安全沙箱说明)
- [API 接口](#api-接口)
- [离线 RAG 知识库构建](#离线-rag-知识库构建)
- [常见问题](#常见问题)
- [开发规范](#开发规范)
- [许可证](#许可证)
- [致谢](#致谢)

---

## 项目简介

**BioDynamics Agent** 是一个面向**计算系统生物学**与**转化医学**的端到端 AI 智能体平台。研究人员只需在 Web 聊天框中输入一段自然语言描述的**生物机制假说**（例如 *“TGF-β 抑制 CD8⁺ T 细胞活性，且该抑制强度随时间累积”*），后端 LangGraph 工作流会自动完成以下闭环：

1. **机制解析**（LLM 结构化抽取生物实体与相互作用）
2. **知识检索**（向量库 RAG + SBML 复用，命中真实动力学参数）
3. **数学建模**（自动生成基于 Hill 方程的 ODE Python 代码）
4. **安全沙箱执行**（subprocess 隔离运行 + 静态安全扫描 + 生物学常识校验）
5. **审计纠错**（失败自动重试 ≤ 3 次，错误反馈回流到建模节点）
6. **报告生成**（Markdown 预测报告 + 时间序列图 Base64 回传）

整个过程通过 **SSE（Server-Sent Events）** 实时把节点执行状态、Token 消耗、最终代码、图表与报告推送到前端。

---

## 它能解决什么问题

生物医学研究长期存在 **"定性假说 ↔ 定量模型"鸿沟**：

| 痛点 | BioDynamics Agent 的解决方案 |
| --- | --- |
| 实验生物学家不懂 ODE / Hill 方程 / SciPy，无法把假说变成可执行代码 | 自然语言输入 → LLM 自动生成可执行的 `scipy.integrate.solve_ivp` 仿真代码 |
| 手动写 ODE 容易给出脱离文献的参数（Kd、Km、半衰期等） | 内置 RAG 知识库（ChromaDB），自动从 PubMed 文献里抽取真实动力学参数 |
| 已有 SBML 模型无法直接复用为新假说 | `SBML_PARSER_PROMPT` 解析 SBML 文本，提取可复用网络拓扑 |
| LLM 生成的代码可能存在 `os.system`、`eval`、网络访问等危险调用 | 沙箱静态黑名单扫描（`os/sys/subprocess/socket/...`） |
| 仿真失败（语法错、NaN、负浓度）需人工排查 | 审计员节点自动重试 ≤ 3 次，按错误类型给出自然语言修改建议 |
| 仿真结果缺乏机理论证 | 报告节点自动撰写 Markdown 预测报告 + 湿实验验证建议 |

### 典型应用场景

- **免疫治疗机制探索**：TGF-β / PD-1 / CD8⁺ T 细胞相互作用建模
- **肿瘤生长与药物响应**：信号通路 + 给药动力学联合仿真
- **传染病动力学**：宿主-病原体耦合 ODE 预测
- **教学与科普**：把教科书上的通路图变成可交互的时间序列图

---

## 技术栈

### 前端（[frontend/](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/frontend)）

| 类别 | 技术 | 用途 |
| --- | --- | --- |
| 框架 | **Next.js 16**（App Router + Turbopack） | React 19 全栈框架 |
| UI 库 | **Shadcn UI** + **Tailwind CSS 4** + **Radix UI** | 暗色全屏聊天界面 |
| 动效 | **Framer Motion** | 消息进入 / 状态切换动画 |
| 渲染 | **react-markdown** + **react-syntax-highlighter** + **remark-gfm** | Markdown / 代码高亮 / 表格 |
| 图标 | **lucide-react** | UI 图标 |
| 类型 | **TypeScript 5** | 全量静态类型 |

### 后端（[backend/](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend)）

| 类别 | 技术 | 用途 |
| --- | --- | --- |
| Web 框架 | **FastAPI 0.139** + **Uvicorn 0.49** | REST + SSE 流式接口 |
| Agent 编排 | **LangGraph 1.2** + **LangChain 1.3** | 6 节点状态机工作流 |
| 检查点 | **MemorySaver**（in-memory checkpoint） | 多轮对话短期记忆 |
| LLM | **OpenAI 兼容 API**（默认 `gpt-4o`，兼容智谱 GLM / DeepSeek 等） | 机制解析、代码生成、审计、报告 |
| Embedding | **OpenAI text-embedding-3-small**（兼容本地 `local_embeddings.py`） | RAG 向量化 |
| 向量库 | **ChromaDB 1.5**（本地持久化） | 动力学参数语义检索 |
| 数值计算 | **NumPy 2.5** + **SciPy 1.18**（`solve_ivp`）+ **Matplotlib 3.11** | ODE 求解与可视化 |
| HTTP | **requests** + **httpx** | PubMed E-utilities / OpenAI 客户端 |
| 沙箱 | **subprocess + tempfile + 静态黑名单** | 隔离执行 LLM 生成的代码 |
| 配置 | **pydantic-settings** + **python-dotenv** | `.env` 环境变量加载 |

> 离线 RAG 知识库构建脚本默认使用 `requests` + `xml.etree.ElementTree` 解析 PubMed XML，**不依赖 biopython**（Python 3.14 兼容性更稳）。

---

## 项目结构

```text
bio-dynamics-agent/
├── README.md                     # 本文件
├── LICENSE                       # MIT 开源协议
├── start-dev.bat                 # Windows 一键启动脚本
│
├── backend/                      # Python 后端
│   ├── .env.example              # 环境变量示例（拷贝为 .env 后填值）
│   ├── requirements.txt          # 依赖清单
│   ├── debug_e2e.py              # 端到端调试脚本
│   ├── app/                      # 应用代码
│   │   ├── main.py               # FastAPI 入口（SSE /api/chat）
│   │   ├── graph.py              # LangGraph 状态机组装
│   │   ├── nodes.py              # 6 个节点实现（解析 / RAG / 生成 / 执行 / 审计 / 报告）
│   │   ├── state.py              # BioDynamicsState 状态定义
│   │   ├── schemas.py            # Pydantic 请求 / 响应模型
│   │   ├── prompts.py            # 所有 System Prompt（含 RAG / SBML 模板）
│   │   ├── config.py             # 全局配置 + LLM / Embedding 客户端
│   │   ├── rag_client.py         # ChromaDB 持久化 + 单位归一化 + 语义检索
│   │   ├── sbml_parser.py        # SBML → 网络拓扑结构化抽取
│   │   ├── sandbox.py            # subprocess 沙箱 + 静态安全扫描 + 常识校验
│   │   ├── local_embeddings.py   # 离线 Embedding 兜底
│   │   └── token_usage.py        # Token 累加与归一化
│   ├── scripts/                  # 离线工具脚本
│   │   ├── fetch_rag_data.py     # 从 PubMed 抓取文献片段
│   │   ├── embed_data.py         # 文献 → 参数结构化抽取
│   │   ├── build_rag_db.py       # 一键构建 RAG 知识库
│   │   └── update_vector_db.py   # 增量更新 ChromaDB
│   └── data/                     # （首次运行后生成）原始文献 / 缓存
│
└── frontend/                     # Next.js 前端
    ├── package.json
    ├── tsconfig.json
    ├── components.json           # Shadcn UI 配置
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx              # 全屏暗色聊天主页
    │   └── globals.css
    └── components/
        ├── chat/
        │   ├── ChatMessage.tsx   # 消息气泡（Markdown / 图表 / Token）
        │   └── ChatInput.tsx     # 输入框（Enter 发送 / Shift+Enter 换行）
        └── ui/                   # Shadcn UI 原子组件
```

---

## 快速开始

### 前置环境

- **Python** ≥ 3.11（推荐 3.12，已在 3.14 验证可跑）
- **Node.js** ≥ 20
- **npm** ≥ 10（或 pnpm / yarn / bun）
- 一份 **OpenAI 兼容的 API Key**（官方 OpenAI、智谱 BigModel、DeepSeek、月之暗面等任一）
- （可选）**Qdrant** / **ChromaDB**：本项目默认走 **ChromaDB 本地持久化**，无需 Docker

### 一键启动（Windows）

```powershell
# 1. 进入项目根目录
cd c:\Users\27553\Desktop\gzlab\bio-dynamics-agent

# 2. 拷贝并填写环境变量
copy backend\.env.example backend\.env
# 用任意编辑器打开 backend\.env，填入 OPENAI_API_KEY 等

# 3. 双击或命令行执行
.\start-dev.bat
```

脚本会自动：检测 `backend\venv\`、后台拉起后端 Uvicorn（端口 8000）和前端 Next.js（端口 3000）、等待 9 秒后自动打开浏览器。

### 手动启动

#### 1. 启动后端

```powershell
cd backend

# 创建虚拟环境（首次）
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt

# 启动 FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

后端启动后访问 [http://localhost:8000](http://localhost:8000) 应返回：

```json
{"status":"ok","service":"BioDynamics Agent"}
```

#### 2. 启动前端

```powershell
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端访问 [http://localhost:3000](http://localhost:3000)。

### 环境变量配置

复制 [backend/.env.example](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/.env.example) 为 `backend/.env` 并填入真实值：

```ini
# —— LLM（OpenAI 兼容）——
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1     # 或 https://open.bigmodel.cn/api/paas/v4
OPENAI_MODEL=gpt-4o                            # 或 glm-4-plus、deepseek-chat 等

# —— 服务 ——
HOST=0.0.0.0
PORT=8000
FRONTEND_URL=http://localhost:3000

# —— ChromaDB（默认本地 ./chroma_db，无需 Docker）——
# 若想切到 Qdrant，参见 app/config.py 中的预留开关

# —— Embedding（默认复用 OPENAI_*；可独立覆盖）——
EMBEDDING_MODEL=text-embedding-3-small

# —— PubMed E-utilities 联系邮箱（NCBI 要求，用于离线建库）——
NCBI_EMAIL=your-email@example.com
```

---

## 使用指南

1. 打开 [http://localhost:3000](http://localhost:3000)
2. 在底部输入框用自然语言描述生物机制，例如：
   > *“TGF-β 抑制 CD8⁺ T 细胞活性，抑制强度随时间累积；当 TGF-β 浓度达到 5 nM 时，T 细胞完全失活。”*
3. 点击 **发送**（或按 Enter；Shift+Enter 换行）
4. 前端会按节点推送状态：
   - `正在解析生物网络...`
   - `正在检索文献参数...`
   - `正在生成 ODE 仿真代码...`
   - `正在执行仿真代码...`
   - `正在审计执行结果...`（如失败会自动重试）
   - `正在生成预测报告...`
5. 最终消息包含：
   - **ODE 代码块**（可一键复制）
   - **仿真结果图表**（Base64 PNG，可右键保存）
   - **Markdown 预测报告**（趋势、机制、实验验证建议）
   - **Token 消耗**（每条 Agent 消息右上角）
6. 顶部按钮：
   - **🔄 重建知识库**：后台触发 RAG 增量更新
   - **🗑 清空**：清空当前 `thread_id` 的短期记忆

---

## LangGraph 工作流

`app/graph.py` 把 6 个节点串成完整状态机：

```text
START
  ↓
[node1_parse_network]   机制解析（LLM 结构化输出 → JSON）
  ↓  (若 need_human_review 则 END)
[node1_5_rag_search]    RAG 检索 + SBML 复用（注入到 Node 2 上下文）
  ↓
[node2_generate_code]   生成 ODE Python 代码（优先使用 RAG 真实参数）
  ↓
[node3_execute_sandbox] 沙箱执行（静态安全扫描 + subprocess）
  ↓
[node4_audit_and_correct] 审计（success / retry / failed）
  ↓  (retry 且 retry_count < 3 → 回到 node2；否则 → report)
[node5_generate_report] 生成 Markdown 报告
  ↓
END
```

**关键设计**：

- **检查点**：`MemorySaver` 让多轮对话可恢复，`/api/chat/clear-memory` 可手动清除
- **审计回灌**：失败时把 `stdout_stderr` 注入 Node 2 的 `error_feedback` 槽位，触发自动纠错
- **重试上限**：3 次，超过则终止并返回 `failure_report`

---

## RAG 知识库与 SBML 复用

### RAG（ChromaDB）

- **离线建库**：`python scripts/build_rag_db.py` 会依次：
  1. 用 `requests` 调用 PubMed E-utilities 抓取文献摘要
  2. 用 `RAG_EXTRACTION_PROMPT` 从摘要中抽取动力学参数（Kd / Km / Vmax / 半衰期等）
  3. 自动做**单位归一化**（时间统一 h，浓度统一 nM）
  4. Embedding 后写入本地 `chroma_db/`
- **运行时检索**：`node1_5_rag_search` 把用户网络 JSON 编码为查询，从 ChromaDB 拉 Top-K 参数
- **参数决策**：`RAG_DECISION_PROMPT` 让 LLM 判断哪些参数可信、哪些需要 fallback 到估算
- **优先级**：Node 2 优先使用 RAG 真实参数（注释 `# 来源：RAG`），缺失才估算（注释 `# 估算值`）

### SBML 复用

`app/sbml_parser.py` 把已有 SBML 模型文本通过 `SBML_PARSER_PROMPT` 抽取为网络节点 / 边 JSON，Node 2 优先基于该网络生成方程，避免从零构建。

---

## 安全沙箱说明

`app/sandbox.py` 用 `tempfile` + `subprocess` 在隔离目录执行 LLM 生成的代码，并把 stdout/stderr 捕获回来。**两道防线**：

1. **静态黑名单扫描**（在执行前）
   - 禁止导入：`os, sys, subprocess, socket, pathlib, shutil, urllib, requests, http, ftplib, smtplib, email, pickle, ctypes, multiprocessing, threading, asyncio`
   - 禁止内建：`eval, exec, compile, open, input, __import__`
2. **生物学常识校验**（在执行后）
   - 解析代码输出的 `BIO_CHECK: <species> = <value>`
   - 若出现 `负值 / NaN / Inf` 立即判定为失败，触发 Node 4 重试

---

## API 接口

| Method | Path | 说明 |
| --- | --- | --- |
| `GET`  | `/` | 健康检查 |
| `POST` | `/api/chat` | **核心**：SSE 流式对话（`ChatRequest`：`message`, `thread_id`） |
| `POST` | `/api/chat/clear-memory` | 清空 `thread_id` 短期记忆 |
| `POST` | `/api/admin/update-vector-db` | 后台触发 ChromaDB 增量更新 |

`/api/chat` 的 SSE 事件类型（`data:` 行 JSON 解码后 `type` 字段）：

| `type` | 含义 |
| --- | --- |
| `node_status` | 节点状态变更（含 `node`, `status`, `detail`） |
| `token_usage` | 单次 LLM 调用的 Token 消耗 |
| `code` | Node 2 生成的 Python 代码 |
| `image` | Node 3 仿真输出 PNG（Base64） |
| `report` | Node 5 Markdown 报告 |
| `error` | 失败 / 重试 / 终态 |
| `done` | 工作流结束 |

---

## 离线 RAG 知识库构建

```powershell
cd backend
.\venv\Scripts\Activate.ps1

# 一键：抓文献 + 抽参数 + 入库（首次约 10-30 分钟）
python scripts/build_rag_db.py

# 增量更新（已有库时使用）
curl -X POST http://localhost:8000/api/admin/update-vector-db
```

前置：`.env` 中 `OPENAI_API_KEY` 与 `NCBI_EMAIL` 已填写。

---

## 常见问题

**Q1. 前端能连上后端吗？**
默认前端通过 `http://localhost:8000` 访问后端；若后端端口变更，请同步修改 `frontend/app/page.tsx` 顶部的 `API_BASE`。

**Q2. 报错 `ModuleNotFoundError: No module named 'app'`？**
在 `backend/` 目录下启动 Uvicorn，确保 `app/` 是 `cwd` 的子目录。

**Q3. biopython 装不上？**
本项目已**完全避开 biopython**。离线建库脚本使用 `requests` + `xml.etree.ElementTree`，兼容 Python 3.14。

**Q4. ChromaDB 与 Qdrant 怎么选？**
- 默认 ChromaDB：本地持久化、无需 Docker、开箱即用
- Qdrant：需要 Docker 启动 `qdrant/qdrant`，或在 [Qdrant Cloud](https://cloud.qdrant.io/) 申请免费集群；切换方式见 `app/config.py` 注释

**Q5. 怎样替换 LLM 供应商？**
`.env` 中把 `OPENAI_BASE_URL` 改成目标供应商的 OpenAI 兼容端点（智谱、DeepSeek、火山、Azure OpenAI 等），`OPENAI_MODEL` 改成对应模型名即可。`gpt-4o` 推荐；本地模型可用 Ollama（base_url 指向 `http://localhost:11434/v1`）。

---

## 开发规范

- **Python**：[PEP 8](https://peps.python.org/pep-0008/) + [PEP 20](https://peps.python.org/pep-0020/)，中文详细注释优先
- **TypeScript**：ESLint + Next.js 默认规范
- **提交规范**：建议使用 Conventional Commits（`feat:`, `fix:`, `docs:`, `chore:`）
- **目录约定**：
  - RAG 相关模块位于 [`backend/app/rag_client.py`](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/rag_client.py) 与 [`backend/app/sbml_parser.py`](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/sbml_parser.py)
  - 离线脚本位于 [`backend/scripts/`](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/scripts)
  - LangGraph Prompt 集中于 [`backend/app/prompts.py`](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/prompts.py)

---

## 许可证

本项目以 **MIT License** 开源，详见 [LICENSE](./LICENSE) 文件。

```text
MIT License

Copyright (c) 2026 BioDynamics Agent Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**第三方依赖**（详见各 `package.json` 与 `requirements.txt`）：所有前端 / 后端依赖均沿用各自上游的开源协议（MIT / Apache-2.0 / BSD 等），本项目不修改其许可。

---

## 致谢

- [LangGraph](https://langchain-ai.github.io/langgraph/) / [LangChain](https://python.langchain.com/) — Agent 编排基石
- [FastAPI](https://fastapi.tiangolo.com/) — 高性能 Python Web 框架
- [Next.js](https://nextjs.org/) + [Shadcn UI](https://ui.shadcn.com/) — 现代前端栈
- [ChromaDB](https://www.trychroma.com/) — 本地优先的向量数据库
- [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) — 公开生物医学文献接口
- [OpenAI](https://openai.com/) / [智谱 BigModel](https://open.bigmodel.cn/) — LLM 与 Embedding 服务

---

**⭐ 如果这个项目对你有帮助，欢迎 Star / Fork / 提 PR！**
