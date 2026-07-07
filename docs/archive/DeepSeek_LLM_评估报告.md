# DeepSeek LLM 接入评估报告

> **评估日期**：2026-07-05
> **评估目标**：为 BioDynamics Agent 选择主用 LLM，测试 DeepSeek 三个版本的适用性
> **测试 API Key**：`sk-3f6b4c2f...8775`
> **测试端点**：`https://api.deepseek.com`

---

## 一、测试设计

### 1.1 测试题目

使用贴近 agent 真实节点任务（N1 NER + N2 模板规划 + N6 ODE 生成）的综合题：

> **用户假说**：肿瘤细胞分泌的 TGF-β 会抑制 CD8+ T 细胞活性。我想测试一种 TGF-β 抑制剂，看给药后 CD8+ T 细胞的动态恢复。
>
> **要求输出**：JSON 含 entities / edges / template / ode_equations / initial_conditions / reasoning

### 1.2 评估维度

| 维度 | 权重 | 说明 |
|---|---|---|
| 实体抽取完整性 | 高 | 是否抽全肿瘤细胞/TGF-β/CD8+/抑制剂四个实体 |
| 边方向与交互类型 | 高 | inhibition / binding / activation 是否准确 |
| ODE 方程数学正确性 | 高 | 是否含药物-靶点耦合项、Emax 抑制项 |
| 模板选择合理性 | 中 | PKPD vs Simple_Inhibition 的取舍 |
| JSON 结构化可用性 | 高 | agent 下游能否直接解析 |
| 响应延迟 | 中 | agent 7 节点累积延迟 |
| Token 消耗 | 低 | 成本考量 |

---

## 二、三版本测试结果

### 2.1 汇总对比

| 版本 | 延迟 | Token | JSON 解析 | 思考链长度 | 实体数 | 边数 | 方程数 |
|---|---|---|---|---|---|---|---|
| **v4-flash 非思考** | 5.72s | 754 | ✅ | 0 | 3 | 2 | 3 |
| **v4-flash 思考** | 14.92s | 1679 | ✅ | 1501 字 | 4 | 3 | 4 |
| **v4-pro 思考** | 34.36s | 2671 | ✅ | 4345 字 | 4 | 3 | 3 |

### 2.2 逐项质量评估

#### v4-flash 非思考（5.72s）

- **实体抽取**：⚠️ 漏抽"肿瘤细胞"，只抽了 TGFb / CD8_Tcell / TGFb_inhibitor
- **边**：2 条（缺肿瘤细胞→TGFb 的分泌边）
- **模板**：PKPD_OneCompartment ✅
- **ODE 方程**：含 Emax 抑制项 `(1 - TGFb/(EC50+TGFb))` ✅，含药物消除项 ✅
- **数学正确性**：方程形式合理，但因漏实体导致系统不完整
- **评价**：速度快，但实体抽取不全，N1 节点漏实体会导致下游 KG 错误

#### v4-flash 思考（14.92s）⭐ 推荐

- **实体抽取**：✅ 4 个全抽到（Tumor_Cell / TGF_beta / CD8_T_Cell / TGF_beta_Inhibitor）
- **边**：✅ 3 条全对（肿瘤→TGFb activation / TGFb→CD8 inhibition / inhibitor→TGFb binding）
- **模板**：PKPD_OneCompartment ✅
- **ODE 方程**：4 条，含分泌+降解+结合+消除+Emax 抑制，耦合完整 ✅
- **思考链**：明确推理了"肿瘤细胞分泌 TGFβ 应作为 activation 边"
- **评价**：实体/边全对，方程完整且耦合正确，延迟 15s 可接受

#### v4-pro 思考（34.36s）

- **实体抽取**：4 个全抽到，但 CD8_T_activity 类型标记为 "protein"（应为 cell）⚠️
- **边**：3 条，但 **Drug→TGFb 误判为 inhibition**（前两版正确标为 binding）❌
- **模板**：Simple_Inhibition（与 v4-flash 思考版选 PKPD 不同）
- **ODE 方程**：3 条，形式合理但药物消除项简化
- **思考链**：4345 字，反复斟酌但最终 binding 判断反而出错
- **评价**：延迟 34s 过高，思考链冗长但对交互类型判断反而不如 flash 思考版准确

---

## 三、结论与推荐

### 3.1 最终方案（已采纳）

| 角色 | 供应商 | 模型 | 触发时机 |
|---|---|---|---|
| **主 LLM** | OpenRouter | `nvidia/nemotron-3-ultra-550b-a55b:free` | 日常高频调用（免费额度 50 次/天） |
| **备用 LLM** | DeepSeek | `deepseek-v4-flash`（思考模式默认开启） | OpenRouter 429 限流 / 主用失败 / 免费额度耗尽时自动切换 |

### 3.2 选型理由

1. **v4-flash 思考版综合质量最佳**：在实体抽取、边类型判断、方程完整性三维度全对，是唯一一个零错误的版本，作为质量兜底备用最合适
2. **v4-pro 思考版不适合本 agent**：延迟 34s 过高（7 节点累积 4 分钟+），且 binding/inhibition 判断反而不如 flash 思考版准确，性价比低
3. **v4-flash 非思考版备选**：速度最快（5.7s）但漏抽肿瘤细胞，对 N1 实体抽取节点有风险
4. **思考模式默认开启**：DeepSeek API 默认 `thinking=enabled`，config.py 的 ChatOpenAI 不传 thinking 参数即默认思考模式，无需改代码
5. **reasoning_content 自动隔离**：LangChain ChatOpenAI 只读 content 字段，reasoning_content 不参与上下文拼接，符合 DeepSeek 多轮对话规范
6. **供应商级容灾**：主用 OpenRouter 与备用 DeepSeek 为不同供应商，单点故障不影响整体可用性

### 3.3 接入方式

DeepSeek 兼容 OpenAI API 格式，**无需改 config.py**，仅改 .env 即可。当前 .env 配置：

```env
# 主 LLM（OpenRouter）
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free

# 备用 LLM（DeepSeek v4-flash 思考版，不同供应商级容灾）
BACKUP_API_KEY=sk-3f6b4c2fd8fa4f95bb4b24c41fd08775
BACKUP_BASE_URL=https://api.deepseek.com
BACKUP_MODEL=deepseek-v4-flash
```

### 3.4 主备切换验证

实测 FallbackLLM 切换链路正常：
- 主用 OpenRouter 当前免费额度已耗尽（429）
- FallbackLLM 自动切换备用 DeepSeek v4-flash
- DeepSeek 返回 "OK"，切换成功
- 供应商级备用确认：`openrouter` 主 + `deepseek` 备 = True

### 3.5 并发与限速

| 模型 | 并发限制 | 备注 |
|---|---|---|
| deepseek-v4-flash | 2500 | 充足，agent 单线程足够 |
| deepseek-v4-pro | 500 | 若后续需要可切换 |

---

## 四、测试原始数据

详细测试结果（含三个版本的完整回答、思考链、Token 统计）已保存于：
`backend/deepseek_test_results.json`

测试脚本：`backend/test_deepseek.py`
