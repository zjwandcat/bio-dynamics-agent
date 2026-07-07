# EGF-EGFR 仿真预测报告生化建模审计（Biochemical Modeling Audit）

> **审计对象**：用户提供的 EGF-EGFR 仿真预测报告（含文字与 0–120 min 曲线图）  
> **模型**：BIOMD0000000205（Brightman & Fell, 2000）  
> **仿真代码路径**：沙箱临时 CSV `C:\Users\27553\AppData\Local\Temp\bio_dynamics_csv_ynogcdrl\simulation.csv`  
> **代码库**：`c:\Users\27553\Desktop\gzlab\bio-dynamics-agent`  
> **审计日期**：2026-07-05

---

## 1. Executive Summary

**是否可信：NO**

该仿真报告在多个维度上与 BIOMD0000000205 的 SBML 语义不一致，且报告文本存在系统性单位误读。主要错误类型（Top 3）：

1. **Time unit misinterpretation（A）**：仿真运行在 **分钟（min）** 尺度，但报告把所有时间指标当成 **小时（h）** 解读，错误放大倍率为 **×60**。
2. **State variable aggregation error（B）**：Agent 将 BIOMD 中显式的受体-复合物状态（如 `EGF-pEGFR-2`、`EGF-pEGFR-2-pShc-Grb2-SOS`）压缩为 `ERBB1`、`phosphorylated_EGFR`、`GRB2` 等聚合变量，丢失了 free/bound、total/phosphorylated、single/double phosphorylated 的语义区分。
3. **SBML topology mapping error（D）**：关键复合物形成反应缺失（`pShc_Grb2 = 0`、`EGF_EGFR_complex` 与 `Shc_Grb2_SOS` 全程不变），且模型 ID `BIOMD0000000205` 被当作物种加入 ODE，导致质量守恒被严重破坏。

---

## 2. Critical Failures Table

| Category | Error | Impact |
|----------|-------|--------|
| **A. Time unit misinterpretation** | 图片 X 轴为 `Time (min)`，终点 120 min；报告文本写成 "120 小时"、"43 小时"、"93.5 小时"；表格表头 `(h)` 硬编码。 | 所有时间指标数量级错误 ×60；AUC 单位错误；生物学时序结论失真。 |
| **B. State variable aggregation** | Agent 用 `ERBB1` 表示 EGFR，用 `phosphorylated_EGFR` 表示 pEGFR，但 BIOMD 中真实状态为 `EGF`、`EGFR`、`EGF-EGFR`、`EGF-EGFR-2`、`EGF-pEGFR-2` 等；缺失 `ppMEK`、`ppERK` 双磷酸化状态。 | 受体总量被误当游离受体，磷酸化产物与游离态无法区分；级联放大指标不可信。 |
| **D. SBML topology mapping** | 缺失 `EGF-pEGFR-2-Shc`、`EGF-pEGFR-2-pShc-Grb2-SOS`、`Raf-RasGTP`、`pRaf-MEK`、`ppMEK-ERK` 等关键复合物；`pShc_Grb2 = 0`；`BIOMD0000000205` 作为物种出现。 | 信号级联断链；ERBB1/GRB2 凭空增长 4546×/368×；违反质量守恒。 |
| **C. Kinetics type mismatch** | BIOMD 使用 mass-action binding（`uM⁻¹s⁻¹`）+ 一级催化（`s⁻¹`）；Agent 模板使用 Michaelis-Menten 磷酸化 + generic activation（`k_cat * src`），未导入 BIOMD 真实参数。 | 动力学速率、稳态、峰值时间全部偏离模型官方定义。 |
| **E. Post-processing hallucination** | `simulation_duration_h` 存储分钟值；`half-life` 用于瞬态信号蛋白；`fold_change` 基于错误基线；`AUC` 单位未换算。 | 表格中半衰期、AUC、倍数变化均无生理意义。 |

---

## 3. Time-scale Diagnostic

### 3.1 官方模型时间单位

BIOMD0000000205 全部动力学参数使用 **秒（s）** 作为时间单位：

- 结合速率：`100 uM⁻¹s⁻¹`（EGF + EGFR → EGF-EGFR）
- 解离速率：`0.0038 s⁻¹`
- 自磷酸化：`2.014 s⁻¹`（EGF-EGFR-2 → EGF-pEGFR-2）
- Ras 激活：`0.1434 s⁻¹`
- Raf/MEK/ERK 级联：均为 `s⁻¹` 或 `uM⁻¹s⁻¹`

**结论：模型官方时间单位为 s；Agent 使用 min 作为仿真单位是合理的（120 min = 7200 s），但报告错误地把 min 读成 h。**

### 3.2 预期 vs 实际时间尺度

| 事件 | BIOMD 文献预期 | Agent 报告 | 偏差 |
|------|----------------|------------|------|
| EGF-EGFR 结合 | 秒级（k_on = 100 uM⁻¹s⁻¹，[EGF]≈0.008 nM） | 未显式输出复合物 | N/A |
| pEGFR 峰值 | **5–10 min** | "120 小时"（实际 CSV 120 min） | 单位错误，且峰值极低（0.017 nM） |
| Ras_GTP 峰值 | 早期（min 级） | "43 小时"（实际 42.94 min） | 单位错误 |
| pERK 峰值 | 60–120 min | "120 小时"（实际 120 min） | 单位错误，数值伪影 |
| 仿真终点 | 120 min 合理 | 被说成 120 h | **×60 错误** |

### 3.3 时间单位错误链路

1. `template_selector.py` 对 `Signaling_Cascade_Phos` 返回 `(120.0, 300, "min")`（正确）。
2. `feature_extractor.py` 把 `t[-1] - t[0] = 120.0` 存入 `simulation_duration_h`，字段名暗示小时。
3. `standard.md.j2` 表头硬编码 `"达峰时间 (h)"`、`"半衰期 (h)"`，未读取 `time_unit`。
4. `N11_REPORT_FILL_PROMPT` Few-shot 示例把 `peak_time: 12.0` 解释为 "12 小时"，诱导 LLM 沿用小时单位。

**错误放大倍率：×60（min → hour）**

---

## 4. Structural Issues

### 4.1 状态变量语义错误（State Variable Aggregation Error）

Agent 输出物种与 BIOMD 真实物种语义对比：

| Agent 物种 | Agent 语义 | BIOMD 对应真实状态 | 问题 |
|------------|------------|--------------------|------|
| `Epidermal_Growth_Factor` | 游离 EGF | `EGF` | 初始条件应为 0.008 nM，实际 0.1 nM |
| `ERBB1` | EGFR 总量 | `EGFR`（游离受体） | 被当成可积累物种，从 0.1 → 454.65 nM |
| `EGF_EGFR_complex` | 复合物 | `EGF-EGFR`、`EGF-EGFR-2` | 初始 0.1 nM 且全程不变，未参与反应 |
| `phosphorylated_EGFR` | pEGFR | `EGF-pEGFR-2` | 峰值仅 0.017 nM，远低于 BIOMD 预期 |
| `phosphorylated_Shc` | pShc | `pShc`（经 `EGF-pEGFR-2-pShc` 释放） | 未与复合物状态关联 |
| `GRB2` | Grb2 总量 | `Grb2` | 从 0.1 → 36.8 nM，违反质量守恒 |
| `Shc_Grb2_SOS` | 三元复合物 | `EGF-pEGFR-2-pShc-Grb2-SOS` | 全程 0.1 nM 不变 |
| `pShc_Grb2` | pShc-Grb2 | 无直接对应 | **始终为 0，断链** |
| `phosphorylated_MEK` | pMEK | `pMEK`/`ppMEK` | 缺失 `ppMEK` 双磷酸化状态 |
| `pERK` | pERK | `pERK`/`ppERK` | 缺失 `ppERK` 双磷酸化状态 |
| `BIOMD0000000205` | 模型 ID | **非物种** | 被错误加入 ODE |

### 4.2 网络结构断链（Broken Edges）

Agent 网络中处于游离或为零的关键节点：

- `EGF_EGFR_complex`：初始 0.1 nM，全程 0.1 nM，未参与 EGF-EGFR 结合反应。
- `Shc_Grb2_SOS`：初始 0.1 nM，全程 0.1 nM，未招募 SOS 到膜上。
- `pShc_Grb2`：**始终为 0**，说明 pShc → Grb2 绑定反应未渲染。
- `SOS1`：从 0.1 衰减到 0.01，未参与下游 Ras 激活。
- `Ras_GDP`：全程 0.1 nM，未因生成 `Ras_GTP` 而减少，违反质量守恒。
- `ERK`、`MAP2K1`：全程 0.1 nM，未因生成 `pERK`/`phosphorylated_MEK` 而减少。

### 4.3 缺失反应（Missing Reactions）

与 BIOMD0000000205 官方反应相比，Agent 缺失或错误表示的反应：

| BIOMD 反应 | 含义 | Agent 状态 |
|------------|------|------------|
| `EGF + EGFR ⇌ EGF-EGFR` | 配体-受体结合 | 未正确注入初始条件，复合物游离 |
| `2 EGF-EGFR ⇌ EGF-EGFR-2` | 受体二聚化 | 未建模 |
| `EGF-EGFR-2 → EGF-pEGFR-2` | 自磷酸化 | 简化为 `phosphorylated_EGFR` |
| `EGF-pEGFR-2 + Shc ⇌ EGF-pEGFR-2-Shc` | Shc 招募 | 未建模 |
| `EGF-pEGFR-2-Shc → EGF-pEGFR-2 + pShc` | Shc 磷酸化 | 简化为 `phosphorylated_Shc` |
| `EGF-pEGFR-2-pShc + Grb2 ⇌ EGF-pEGFR-2-pShc-Grb2` | Grb2 招募 | `pShc_Grb2 = 0` |
| `EGF-pEGFR-2-pShc-Grb2 + SOS ⇌ EGF-pEGFR-2-pShc-Grb2-SOS` | SOS 招募 | 未建模 |
| `Raf + RasGTP ⇌ Raf-RasGTP → pRaf + RasGTP` | Ras 激活 Raf | 简化为 `phosphorylated_Raf` |
| `pRaf + MEK ⇌ pRaf-MEK → pRaf + pMEK` | MEK 磷酸化 | 简化为 `phosphorylated_MEK` |
| `pMEK + MEK ⇌ pMEK-MEK → pMEK + ppMEK` | MEK 双磷酸化 | **缺失 `ppMEK`** |
| `ppMEK + ERK ⇌ ppMEK-ERK → ppMEK + pERK` | ERK 磷酸化 | 简化为 `pERK` |
| `ppMEK + pERK ⇌ ppMEK-pERK → ppMEK + ppERK` | ERK 双磷酸化 | **缺失 `ppERK`** |

### 4.4 质量守恒违规

从 CSV 计算：

| 物种 | 初始 (nM) | 峰值 (nM) | 倍数变化 | 问题 |
|------|----------|----------|---------|------|
| `ERBB1` | 0.1 | **454.65** | 4546× | 游离受体不可能被凭空合成 |
| `GRB2` | 0.1 | **36.80** | 368× | 适配蛋白总量应保持守恒 |
| `pERK` | 0 | 13.85 | ∞ | 底物 `ERK` 始终 0.1，未减少 |
| `phosphorylated_MEK` | 0 | 4.40 | ∞ | 底物 `MAP2K1` 始终 0.1，未减少 |

---

## 5. Kinetics Type Audit

### 5.1 BIOMD 官方动力学类型

BIOMD0000000205 基于 **mass-action kinetics**：

- 二分子结合：`rate = k_on * [A] * [B]`，单位 `uM⁻¹s⁻¹`
- 一分子转化/解离：`rate = k * [A]`，单位 `s⁻¹`
- 无 Michaelis-Menten 酶饱和项、无 Hill 系数、无基因表达延迟。

### 5.2 Agent 模板动力学类型

`Signaling_Cascade_Phos.j2` 使用：

- `binding`：mass-action（正确方向）。
- `phosphorylation`：**Michaelis-Menten** `v = k_phos * [enzyme] * [S] / (Km + [S])`，与 BIOMD 的 mass-action 复合物机制不同。
- `activation`（generic fallback）：`d[target]/dt += k_cat * [source] - k_degr * [target]`，这是 **基因表达/表型动力学的近似形式**，不适用于秒-分钟级的酶促信号转导。
- `exchange`：`d[RasGTP]/dt += k_exchange * [RasGDP]`，未质量守恒地消耗 RasGDP。

**结论**：Agent 将酶促信号级联错误地部分建模为 gene-expression-like activation，引入了不合理的持续积累和延迟。

---

## 6. Post-processing Metrics Audit

| 指标 | 是否适用 | 问题 | 正确替代 |
|------|----------|------|----------|
| **AUC** | ⚠️ 部分适用 | 单位依赖时间单位；当前 `AUC = ∫y dt_min` 但被表格标为 `(h)`，数值未换算 | 统一为 `nM·min` 或 `nM·s`，并在表头明确标注 |
| **Half-life** | ❌ 不适用 | 信号蛋白（pEGFR、pERK）是瞬态激活-去磷酸化过程，不存在单指数衰减；SOS1 的 "34.7 h" 是分钟数据被 log-linear 拟合后的伪影 | 对该模型不使用 half-life；改用 peak time、decay time to 50% peak |
| **Fold-change** | ⚠️ 需谨慎 | 当前基于初始值 0.1 nM 计算，但初始值本身错误 | 使用正确初始浓度（EGF=0.008，EGFR=0.3）重新计算；对磷酸化产物使用 [pX]/([X]+[pX]) 比例 |
| **Steady-state** | ❌ 不适用 | 120 min 内 pERK 仍在上升，未达稳态 | 标注为 "120 min 终值"，而非稳态 |

---

## 7. Root Cause Analysis

按审计框架的 6 类归因：

| 类别 | 具体根因 | 涉及文件/位置 |
|------|----------|---------------|
| **A. time unit misinterpretation** | `feature_extractor.py` 把分钟时长存入 `simulation_duration_h`；`standard.md.j2` 硬编码 `(h)`；N11 prompt 示例以小时解释 `peak_time` | `feature_extractor.py:281`；`standard.md.j2:11`；`prompts_v2.py:295-302` |
| **B. state variable aggregation error** | `_unique_species_from_edges` 从节点/边提取的名称被简化为 `ERBB1`、`phosphorylated_EGFR` 等聚合变量，未保留 BIOMD 的 free/bound/complex 区分 | `nodes_v2.py:1436-1472` |
| **C. kinetics type mismatch** | `Signaling_Cascade_Phos.j2` 对 phosphorylation 使用 Michaelis-Menten，对 activation 使用 gene-expression-like 线性增长，偏离 BIOMD mass-action | `Signaling_Cascade_Phos.j2:105-134`；`_mechanism_phosphorylation.j2` |
| **D. SBML topology mapping error** | 未从 BIOMD0000000205.json 读取真实反应网络；未建模关键复合物；模型 ID 被当作物种 | `nodes_v2.py:1604-1721`；`BIOMD0000000205.json` 未被 N5 直接读取 |
| **E. post-processing hallucination** | `simulation_duration_h` 字段名误导；`half-life` 算法未判断模型类型；报告 LLM 基于错误表头生成解释 | `feature_extractor.py:91-135`；`standard.md.j2` |
| **F. solver instability / integration artifact** | 未见数值发散或 NaN；主要问题非求解器引起 | N/A |

---

## 8. Fix Recommendations

### 8.1 修复 Agent Prompt

1. **在 N11_REPORT_FILL_PROMPT 中注入 `time_unit`**：
   - 修改 Few-shot 示例，根据 `time_unit` 动态解释 `peak_time`。
   - 强制要求 LLM 在机制分析中显式声明 "所有时间单位为 {time_unit}"。

2. **增加状态变量语义约束**：
   - 要求 LLM 区分 `free receptor`、`ligand-receptor complex`、`phosphorylated complex`、`double-phosphorylated kinase`。
   - 禁止将模型 ID、通路名称等非物种文本写入物种列表。

3. **增加后处理指标约束**：
   - 明确 `half-life` 仅适用于衰减型物种（如药物清除），不适用于瞬态磷酸化蛋白。
   - `fold-change` 必须基于正确的初始浓度，并区分 total vs phosphorylated fraction。

### 8.2 修复 Simulation Pipeline

| 优先级 | 修复点 | 方案 |
|--------|--------|------|
| **P0** | 时间单位透传 | `feature_extractor.extract` 接收 `time_unit`，输出 `simulation_duration` + `time_unit`；`standard.md.j2` 根据 `time_unit` 动态渲染表头；`ReportRenderer.render` 传递 `time_unit` |
| **P0** | 初始条件映射 | `_parse_initial_conditions` 后，用 `_raw_to_ode` 把 `EGF`→`Epidermal_Growth_Factor`、`EGFR`→`ERBB1` 映射，再匹配 `y0`；优先使用 BIOMD 官方初始浓度 |
| **P0** | SBML ground truth 注入 | 当用户输入含 BIOMD*/MODEL* ID 时，N5 直接读取 `backend/data/processed/{model_id}.json` 的参数与初始浓度；RAG 仅作补充 |
| **P1** | 复合物网络建模 | 从 BIOMD JSON 解析完整反应网络，保留 `EGF-pEGFR-2-Shc`、`*-Grb2-SOS`、`*-RasGDP`、`*-MEK`、`*-ERK` 等复合物；禁止把复合物简化为单一磷酸化状态 |
| **P1** | 质量守恒校验 | 在 `n6_ode_generator` 渲染后增加守恒检查：总受体、总 Grb2、总 MEK、总 ERK 等在积分过程中应基本守恒（相对漂移 < 5%） |
| **P1** | 双磷酸化状态 | Raf/MEK/ERK 级联必须显式建模 `MEK ↔ pMEK ↔ ppMEK` 和 `ERK ↔ pERK ↔ ppERK` |
| **P2** | 单位换算 | 读取 SBML 参数时，根据 `uM_1_s_1`/`sec_1` 统一换算为当前 `time_unit` 下的一致单位 |
| **P2** | 节点过滤 | `_unique_species_from_edges` 对 `reaction_equation` 提取的 token 增加白名单校验，过滤模型 ID、空字符串、非物种 token |

---

## 9. 附录 A：原始 CSV 关键指标（已校准时间单位为 min）

> 数据来源：`C:\Users\27553\AppData\Local\Temp\bio_dynamics_csv_ynogcdrl\simulation.csv`

| 物种 | 初始 (nM) | 峰值 (nM) | 达峰时间 (min) | 稳态 (nM) | 备注 |
|------|----------|----------|---------------|----------|------|
| Epidermal_Growth_Factor | 0.100 | 0.100 | 0.000 | 0.100 | 应为 0.008 nM |
| ERBB1 | 0.100 | **454.650** | **120.000** | 448.929 | 质量守恒严重违规 |
| EGF_EGFR_complex | 0.100 | 0.100 | 0.000 | 0.100 | 未参与反应 |
| phosphorylated_EGFR | 0.000 | 0.017 | 120.000 | 0.017 | 峰值异常低 |
| SHC1 | 0.100 | 0.100 | 0.000 | 0.100 | 未参与反应 |
| phosphorylated_Shc | 0.000 | 0.394 | 120.000 | 0.379 | 未连接复合物 |
| GRB2 | 0.100 | 36.799 | 120.000 | 34.170 | 质量守恒严重违规 |
| SOS1 | 0.100 | 0.100 | 0.000 | 0.010 | 仅衰减 |
| Shc_Grb2_SOS | 0.100 | 0.100 | 0.000 | 0.100 | 未参与反应 |
| Ras_GDP | 0.100 | 0.100 | 0.000 | 0.100 | 未因 RasGTP 生成而减少 |
| Ras_GTP | 0.100 | 0.303 | 42.943 | 0.177 | 初始值错误（应为 0） |
| RAF1 | 0.100 | 0.100 | 0.000 | 0.100 | 未参与反应 |
| phosphorylated_Raf | 0.000 | 1.110 | 93.512 | 1.068 | 初始值错误 |
| MAP2K1 | 0.100 | 0.100 | 0.000 | 0.100 | 未因 pMEK 生成而减少 |
| phosphorylated_MEK | 0.000 | 4.400 | 120.000 | 4.285 | 缺失 ppMEK |
| ERK | 0.100 | 0.100 | 0.000 | 0.100 | 未因 pERK 生成而减少 |
| pERK | 0.000 | 13.852 | 120.000 | 12.871 | 缺失 ppERK |
| BIOMD0000000205 | 0.100 | 0.100 | 0.000 | 0.100 | **模型 ID 被当作物种** |
| pShc_Grb2 | 0.000 | 0.000 | 0.000 | 0.000 | 断链 |

---

## 10. 附录 B：BIOMD0000000205 官方关键参数摘录

| 反应 | 速率常数 | 单位 | 初始浓度 |
|------|----------|------|----------|
| EGF + EGFR → EGF-EGFR | k1 = 100 | uM⁻¹s⁻¹ | EGF = 0.0081967 nM |
| EGF-EGFR → EGF + EGFR | k2 = 0.0038 | s⁻¹ | EGFR = 0.3 nM |
| 2 EGF-EGFR → EGF-EGFR-2 | k1 = 10 | uM⁻¹s⁻¹ | Shc = 1.0 nM |
| EGF-EGFR-2 → EGF-pEGFR-2 | k1 = 2.014 | s⁻¹ | Grb2 = 1.0 nM |
| EGF-pEGFR-2 + Shc → EGF-pEGFR-2-Shc | k1 = 90 | uM⁻¹s⁻¹ | SOS = 0.3 nM |
| EGF-pEGFR-2-Shc → EGF-pEGFR-2 + pShc | k1 = 4.481 | s⁻¹ | RasGDP = 0.15 nM |
| Raf + RasGTP → Raf-RasGTP | k1 = 1.754 | uM⁻¹s⁻¹ | Raf = 0.15 nM |
| Raf-RasGTP → pRaf + RasGTP | k1 = 0.7624 | s⁻¹ | MEK = 1.0 nM |
| pRaf + MEK → pRaf-MEK | k1 = 4.0 | uM⁻¹s⁻¹ | ERK = 1.0 nM |
| pRaf-MEK → pRaf + pMEK | k1 = 3.5 | s⁻¹ | |
| pRaf + pMEK → pRaf-pMEK | k1 = 4.0 | uM⁻¹s⁻¹ | |
| pRaf-pMEK → pRaf + ppMEK | k1 = 2.9 | s⁻¹ | |
| ppMEK + ERK → ppMEK-ERK | k1 = 3.0 | uM⁻¹s⁻¹ | |
| ppMEK-ERK → ppMEK + pERK | k1 = 16.0 | s⁻¹ | |
| ppMEK + pERK → ppMEK-pERK | k1 = 3.0 | uM⁻¹s⁻¹ | |
| ppMEK-pERK → ppMEK + ppERK | k1 = 5.7 | s⁻¹ | |

---

## 11. 审计结论

该 EGF-EGFR 仿真报告 **不可采信**。除了用户已经发现的时间单位错误外，还存在更根本的生化建模错误：

1. **初始条件未按 BIOMD 官方值注入**（EGF/EGFR 应为 0.008/0.3 nM，实际为 0.1 nM）。
2. **状态变量聚合错误**，丢失了 BIOMD 中 free/bound/complex 的显式区分。
3. **网络拓扑映射错误**，关键复合物未建模，模型 ID 被当作物种，导致质量守恒被破坏。
4. **动力学类型不匹配**，部分反应被错误建模为 gene-expression-like activation。
5. **后处理指标误用**，half-life、fold-change、AUC 均不适合当前模型或未正确换算。

**建议**：在修复上述 P0/P1 问题后，使用 BIOMD0000000205 官方参数和完整反应网络重新生成 ODE，并将仿真时长明确标注为分钟（min），再生成报告。

---

*审计人：Biochemical Modeling Auditor（AI 辅助）*  
*报告版本：v2.0（整合生化建模审计框架）*  
*结论：当前报告不可直接采信，需结构性修复后重新仿真。*
