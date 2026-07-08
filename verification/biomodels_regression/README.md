# BioModels 回归测试 — `biomodels_regression/`

从 EBI BioModels 数据库下载 30 个参考 SBML 模型，用 **roadrunner**（黄金标准）
重新仿真，并与 **BioDynamics v4** 的仿真输出比较。

## 文件清单

| 文件 | 作用 |
|------|------|
| `conftest.py` | pytest fixture：SBML 缓存目录、HTTP session、roadrunner 引擎、30 条用例元数据、metrics 写入器 |
| `test_biomodels_regression.py` | 30 条参数化回归用例 + 聚合报告生成 |
| `biomodels_config.yaml` | （历史）YAML 配置，保留作参考 |

## 比较指标

每条用例均计算并断言以下 6 项指标：

| 指标 | 说明 | 默认阈值 |
|------|------|----------|
| RMSE | 均方根误差（共有物种时间序列拼接后） | < `rmse_threshold`（每模型独立） |
| Pearson | 皮尔逊相关系数 | > `pearson_threshold`（每模型独立） |
| Peak Time | 主导物种达峰时间差 | < 2.0 min |
| Peak Height | 主导物种峰值高度相对误差 | < 30% |
| AUC | 曲线下面积相对误差（梯形法，归一化） | < 25% |
| Steady State | 稳态浓度相对误差（t=120min） | < 20% |

附加结构性断言：v4 物种数与参考物种数差值 ≤ 2。

## 通路覆盖（10 条核心通路）

`EGFR_RTK` · `MAPK_ERK` · `PI3K_AKT_mTOR` · `p53` · `APOPTOSIS` ·
`CELL_CYCLE` · `JAK_STAT` · `NF_KB` · `WNT` · `TGF_BETA`

## 30 个真实 BioModels ID

包含但不限于任务要求的 5 个锚点：
`BIOMD0000000010` (Schoeberl MAPK)、`BIOMD0000000012` (Markevich 双磷酸化)、
`BIOMD0000000007` (Kholodenko EGFR)、`BIOMD0000000056` (Tyson/Novak 细胞周期变体)、
`BIOMD0000000152`（见 `conftest.py` 完整清单）。

完整 30 条见 `conftest.py` 中 `BIOMODELS_ENTRIES`。

## 运行方式

```bash
# 仅收集（不执行），验证用例可枚举
pytest verification/biomodels_regression/ --collect-only -q

# 运行完整回归（需 roadrunner + 网络 + v4 仿真服务）
BIODYNAMICS_API_URL=http://localhost:8000 \
  pytest verification/biomodels_regression/ -v

# 仅运行报告聚合（轻量，不下载/仿真）
pytest verification/biomodels_regression/test_biomodels_regression.py::test_generate_biomodels_report
```

## 输出产物

写入 `verification/reports/biomodels_regression/`：

- `metrics.jsonl` — 每用例逐条 metrics（JSON Lines）
- `report.json` — 汇总 pass/fail 矩阵
- `report.md` — Markdown 表格报告
- `report.html` — HTML 报告（含样式化表格）

## 依赖

- `roadrunner`（`pip install roadrunner`）— 优先仿真器
- `python-libsbml`（`pip install python-libsbml`）— 后备 SBML 解析
- `scipy` / `numpy` — 指标计算与后备积分
- `requests` — BioModels API 下载
- `PyYAML` — 历史配置读取

缺失任一依赖时，相关用例会优雅 skip，不会导致收集失败。

## 执行约束

- 每个回归用例标记 `@pytest.mark.benchmark`（长时运行，默认不在快速 CI 跑）
- 无网络 / 无 roadrunner / 无 v4 服务时自动 skip
- SBML 持久缓存在 `backend/data/raw/biomodels/`，跨会话复用
- 已知 v4 P0 bug（FM-001/002/003）以独立 skip 用例文档化
