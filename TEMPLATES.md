# BioDynamics Agent v2 — Templates Reference

> 本文件是 v2 全部 9 个 Jinja2 模板的权威参考：8 个 ODE 模板 + 1 个报告模板。每个模板列出文件名、用途、必填变量、Python 代码骨架、调用示例。
> 上层架构见 [ARCHITECTURE.md](ARCHITECTURE.md)；安装与运行见 [README.md](README.md)。

---

## 目录

1. [约定](#约定)
2. [ODE 模板](#ode-模板)
   - [Simple_Inhibition.j2](#simple_inhibitionj2)
   - [Simple_Activation.j2](#simple_activationj2)
   - [Cascade_Inhibition.j2](#cascade_inhibitionj2)
   - [Cascade_Activation.j2](#cascade_activationj2)
   - [PKPD_OneCompartment.j2](#pkpd_onecompartmentj2)
   - [PKPD_TwoCompartment.j2](#pkpd_twocompartmentj2)
   - [DoseSweep.j2](#dosesweepj2)
   - [Combination.j2](#combinationj2)
3. [Report Template](#report-template)
4. [调用流程（伪代码）](#调用流程伪代码)

---

## 约定

* **目录**：`backend/app/ode_templates/*.j2`（ODE 模板），`backend/app/report_templates/*.j2`（报告模板）
* **加载器**：[`app/ode_templates/__init__.py::render_template`](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates/__init__.py)，使用 `jinja2.Environment(undefined=StrictUndefined)`
* **调用入口**：`render_template("Simple_Inhibition", {"species_names": [...], ...}) → str`
* **强制尾部**：所有 ODE 模板必须在末尾 emit 三件套
  ```python
  np.savetxt("simulation.csv", data, delimiter=",", header=header, comments="")
  fig.savefig("simulation.png", dpi=120)
  print(f"BIO_CHECK: {sp} = ...")  # 每个 species 一行
  ```
  沙箱 cwd=tempfile，CSV/PNG 落盘后由 `execute_simulation_code_v2` 读取并返回 `simulation_csv_path` / `image_base64`
* **变量命名**：模板变量必须与 `network_relations` 字典键一致；`n.eval_points` 默认 200，`t_end` 单位 h，`Kd` 单位 nM

---

## ODE 模板

### Simple_Inhibition.j2

**用途**：单条抑制边（inhibitor → target）的双物种时间序列仿真。最常见的入门模板。

**必填变量**：

| 变量 | 类型 | 说明 |
| --- | --- | --- |
| `species_names` | `list[str]` | 例 `["Inhibitor", "Target"]` |
| `inhibitor` | `str` | 抑制物名，渲染为字符串字面量 |
| `target` | `str` | 被抑制物种名 |
| `kd` | `float` | Hill 抑制常数（nM） |
| `n_hill` | `float` | Hill 系数，推荐 1~3 |
| `degradation` | `float` | target 降解速率（1/h） |
| `production` | `float` | target 基础生成速率（nM/h） |
| `t_end` | `float` | 仿真时长（h） |
| `n_eval` | `int` | 采样点数 |
| `y0` | `list[float]` | 初值 `[inhibitor0, target0]` |

**ODE 形式**：
```
dI/dt = -0.1 * I
dT/dt = production * (Kd^n / (Kd^n + I^n + 1e-12)) - degradation * T
```

**代码骨架**：
```python
def _ode(t, y):
    inhibitor, target = y
    inhibition = KD ** N_HILL / (KD ** N_HILL + inhibitor ** N_HILL + 1e-12)
    d_inhibitor = -0.1 * inhibitor
    d_target = PROD_RATE * inhibition - DEGRADATION * target
    return [d_inhibitor, d_target]
```

**调用示例**：
```python
from app.ode_templates import render_template
code = render_template("Simple_Inhibition", {
    "species_names": ["TGF_beta", "CD8_T"],
    "inhibitor": "TGF_beta",
    "target": "CD8_T",
    "kd": 5.0,
    "n_hill": 2.0,
    "degradation": 0.05,
    "production": 1.0,
    "t_end": 48.0,
    "n_eval": 200,
    "y0": [5.0, 10.0],
})
```

---

### Simple_Activation.j2

**用途**：单条激活边（activator → target）。

**必填变量**：与 Simple_Inhibition 几乎一致，把 `inhibitor` 换成 `activator`。

**ODE 形式**：
```
dA/dt = -0.1 * A
dT/dt = production * (A^n / (Kd^n + A^n + 1e-12)) - degradation * T
```

**调用示例**：
```python
code = render_template("Simple_Activation", {
    "species_names": ["IL2", "T_cell"],
    "activator": "IL2",
    "target": "T_cell",
    "kd": 1.0, "n_hill": 1.5,
    "degradation": 0.05, "production": 1.0,
    "t_end": 24.0, "n_eval": 200,
    "y0": [0.5, 0.0],
})
```

---

### Cascade_Inhibition.j2

**用途**：N 个物种组成的多步抑制级联；N4 KG 输出 edges 列表驱动。

**必填变量**：

| 变量 | 类型 | 说明 |
| --- | --- | --- |
| `species_names` | `list[str]` | 全部物种名（按 N4 KG nodes 顺序） |
| `edges_json` | `str` (JSON) | `[{"source": "A", "target": "B", "interaction": "inhibition"}, ...]` |
| `params_json` | `str` (JSON) | `{species_name: {"kd": 1.0, "n": 2.0, "production": 1.0, "degradation": 0.1}}` |
| `t_end`, `n_eval`, `y0` | 同上 | |

**ODE 形式**：对每个 species 叠加基础降解 + 每条入边的 Hill 调控项。

**调用示例**：
```python
import json
code = render_template("Cascade_Inhibition", {
    "species_names": ["TGF_beta", "SMAD3", "CD8_T", "IFNG"],
    "edges_json": json.dumps([
        {"source": "TGF_beta", "target": "SMAD3", "interaction": "activation"},
        {"source": "SMAD3", "target": "CD8_T", "interaction": "inhibition"},
        {"source": "CD8_T", "target": "IFNG", "interaction": "activation"},
    ]),
    "params_json": json.dumps({
        "SMAD3": {"kd": 5.0, "n": 2.0, "production": 1.0, "degradation": 0.1},
        "CD8_T": {"kd": 2.0, "n": 2.0, "production": 1.0, "degradation": 0.05},
        "IFNG":  {"kd": 1.0, "n": 1.5, "production": 2.0, "degradation": 0.1},
    }),
    "t_end": 48.0, "n_eval": 300,
    "y0": [5.0, 0.0, 1.0, 0.0],
})
```

---

### Cascade_Activation.j2

**用途**：与 Cascade_Inhibition 对称，全部为激活边。

**必填变量**：同 Cascade_Inhibition，但 `edges[].interaction` 限定为 `"activation"`。

**调用示例**：同 Cascade_Inhibition，把 `inhibition` → `activation`。

---

### PKPD_OneCompartment.j2

**用途**：1-房室 PK + Emax 药效模型。IV 推注（bolus）场景。

**必填变量**：

| 变量 | 类型 | 说明 |
| --- | --- | --- |
| `species_names` | `list[str]` | 例 `["Drug_conc", "Target"]` |
| `drug_name` | `str` | 药物名（用于图例） |
| `dose` | `float` | 初始浓度（nM） |
| `k10` | `float` | 消除速率常数（1/h） |
| `target` | `str` | 药效作用靶点 |
| `ec50` | `float` | 半效应浓度（nM） |
| `emax` | `float` | 最大效应（0~1） |
| `gamma` | `float` | Hill 系数 |
| `t_end`, `n_eval`, `y0` | 同上 | |

**ODE 形式**：
```
dD/dt = -k10 * D
dT/dt = 0.5 * (1 - Emax * D^γ / (EC50^γ + D^γ + 1e-12)) - 0.05 * T
```

**图形**：2 子图（上：药物浓度 vs 时间；下：靶点响应 vs 时间）。

**调用示例**：
```python
code = render_template("PKPD_OneCompartment", {
    "species_names": ["Galunisertib", "SMAD3_p"],
    "drug_name": "Galunisertib",
    "dose": 100.0,
    "k10": 0.1,
    "target": "SMAD3_p",
    "ec50": 51.0,
    "emax": 0.8,
    "gamma": 1.5,
    "t_end": 72.0, "n_eval": 300,
    "y0": [100.0, 5.0],
})
```

---

### PKPD_TwoCompartment.j2

**用途**：2-房室 PK + Emax。中央室 + 外周室交换。

**必填变量**：

| 变量 | 类型 | 说明 |
| --- | --- | --- |
| `species_names` | `list[str]` | 例 `["Drug_central", "Drug_peripheral", "Target"]` |
| `drug_name` | `str` | 药物名 |
| `dose` | `float` | 中央室初始浓度 |
| `k10` | `float` | 中央室消除 |
| `k12` | `float` | 中央→外周 |
| `k21` | `float` | 外周→中央 |
| `target`, `ec50`, `emax`, `gamma` | 同 PKPD_OneCompartment | |
| `t_end`, `n_eval`, `y0` | 同上 | |

**ODE 形式**：
```
dCc/dt = -k10*Cc - k12*Cc + k21*Cp
dCp/dt =  k12*Cc - k21*Cp
dT/dt  = 0.5 * (1 - Emax * Cc^γ / (EC50^γ + Cc^γ + 1e-12)) - 0.05 * T
```

**调用示例**：
```python
code = render_template("PKPD_TwoCompartment", {
    "species_names": ["Drug_c", "Drug_p", "Biomarker"],
    "drug_name": "CompoundX",
    "dose": 200.0,
    "k10": 0.05, "k12": 0.1, "k21": 0.08,
    "target": "Biomarker",
    "ec50": 30.0, "emax": 0.9, "gamma": 2.0,
    "t_end": 96.0, "n_eval": 300,
    "y0": [200.0, 0.0, 10.0],
})
```

---

### DoseSweep.j2

**用途**：在 PKPD_OneCompartment 基础上额外跑剂量扫描，输出 IC50 / IC90 / HED + 浓度-效应曲线。

**必填变量**：

| 变量 | 类型 | 说明 |
| --- | --- | --- |
| `species_names`, `drug_name`, `dose`, `k10`, `target`, `ec50`, `emax`, `gamma`, `t_end`, `n_eval`, `y0` | 同 PKPD_OneCompartment | |
| `conc_min_factor` | `float` | 扫描下限倍数（相对 EC50），如 0.1 |
| `conc_max_factor` | `float` | 扫描上限倍数，如 100 |
| `n_points` | `int` | 扫描点数，如 30 |

**stdout 标记**（沙箱解析）：

```
DOSE_RESPONSE: {"concentrations": [...], "effects": [...], "drug_name": "..."}
IC50: <float>
IC90: <float>
HED: <float>
```

前端 `DoseResponseCurve.tsx` 用 `DOSE_RESPONSE` 绘 Sigmoid Emax 曲线 + IC50/IC90/HED 标线。

**调用示例**：
```python
code = render_template("DoseSweep", {
    "species_names": ["Drug_conc", "Target"],
    "drug_name": "Galunisertib",
    "dose": 100.0, "k10": 0.1,
    "target": "Target",
    "ec50": 51.0, "emax": 0.8, "gamma": 1.5,
    "t_end": 48.0, "n_eval": 200,
    "y0": [100.0, 5.0],
    "conc_min_factor": 0.1, "conc_max_factor": 100.0, "n_points": 30,
})
```

---

### Combination.j2

**用途**：联合用药。三组仿真（A 单药 / B 单药 / 联合）+ Chou-Talalay CI。

**必填变量**：

| 变量 | 类型 | 说明 |
| --- | --- | --- |
| `drugs_json` | `str` (JSON) | `[{"name": "DrugA", "dose": ..., "ec50": ..., "emax": ..., "gamma": ..., "target": "..."}, ...]` |
| `t_end`, `n_eval`, `y0` | 同上 | |

**stdout 标记**（每个 fa 一行）：

```
COMBO_CI: fa=0.5, CI=0.842
COMBO_CI: fa=0.75, CI=0.791
COMBO_CI: fa=0.9, CI=0.755
```

**判定**：CI < 0.8 → 协同；0.8~1.2 → 叠加；> 1.2 → 拮抗。

**调用示例**：
```python
import json
code = render_template("Combination", {
    "drugs_json": json.dumps([
        {"name": "DrugA", "dose": 50.0, "ec50": 30.0, "emax": 0.7, "gamma": 1.5, "target": "Target"},
        {"name": "DrugB", "dose": 20.0, "ec50": 10.0, "emax": 0.6, "gamma": 2.0, "target": "Target"},
    ]),
    "t_end": 72.0, "n_eval": 300,
    "y0": [10.0],
})
```

---

## Report Template

### `standard.md.j2`

**用途**：拼装最终 Markdown 报告。N11 LLM 仅填充 4 个 JSON 字段，Python 用本模板渲染。

**渲染变量**：

| 变量 | 类型 | 必填 | 来源 |
| --- | --- | --- | --- |
| `llm` | `dict` | ✓ | LLM 填充：`{mechanism_analysis, simulation_interpretation, discussion, limitations}` |
| `metrics` | `dict` | ✓ | N8 输出 `{species, overall, combo}` |
| `evidence` | `list[dict]` | ✓ | N10 输出 paper_evidence |
| `experiments` | `list[dict]` | ✓ | N9 输出 experiment_protocols |
| `knowledge_graph` | `dict` | ✓ | N4 输出（用于底部元信息） |
| `confidence` | `float` | ✓ | N8 输出 |

**模板骨架**（[文件](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/report_templates/standard.md.j2)）：

```markdown
# 仿真预测报告

## 1. 机制分析
{{ llm.mechanism_analysis }}

## 2. 仿真结果解读
{{ llm.simulation_interpretation }}

### 物种指标
| 物种 | 峰值 (nM) | 达峰时间 (h) | 倍数变化 | 稳态 (nM) | 半衰期 (h) | AUC |
|------|----------|--------------|----------|-----------|------------|-----|
{% for sp, m in metrics.species.items() %}
| {{ sp }} | {{ "%.3f"|format(m.peak) }} | ... |
{% endfor %}

{% if metrics.combo %}
### 联合用药指标
...
{% endif %}

## 3. 实验验证建议
{% for ex in experiments %}- **{{ ex.name }}**：靶点 {{ ex.target }}，细胞系 {{ ex.cell_line }}（来源：PMID {{ ex.pmid }}）
{% endfor %}

## 4. 文献证据
{% for ev in evidence %}- {{ ev.title }}（PMID: {{ ev.pmid }}，图 {{ ev.figure_ref | default('-') }}，细胞系 {{ ev.cell_line | default('-') }}）
{% endfor %}

## 5. 讨论
{{ llm.discussion }}

## 6. 局限性
{{ llm.limitations }}

---
*置信度：{{ "%.2f"|format(confidence) }}；节点版本：v2；工作流：{{ knowledge_graph.node_count or 0 }} 节点 / {{ knowledge_graph.edge_count or 0 }} 边*
```

**调用示例**：
```python
from app.report_renderer import ReportRenderer
renderer = ReportRenderer(template_dir="app/report_templates")
md = renderer.render(
    llm_filled={
        "mechanism_analysis": "TGF-β 通过 SMAD3 抑制 CD8+ T 细胞激活...",
        "simulation_interpretation": "在 12.3h 达到峰值，fold change 2.5...",
        "discussion": "本结果与文献 PMID:12345678 报道一致...",
        "limitations": "参数来自估计；未考虑时间延迟。",
    },
    metrics={
        "species": {"TGF_beta": {"peak": 5.0, "peak_time": 0.0, "fold_change": 1.0, "steady_state": 4.8, "half_life": 24.0, "auc": 240.0}},
        "overall": {},
        "combo": {},
    },
    evidence=[{"pmid": "12345678", "title": "TGF-β pathway in CD8", "figure_ref": "Fig.2", "cell_line": "Jurkat"}],
    experiments=[{"name": "Western blot", "target": "pSMAD2", "cell_line": "Jurkat", "pmid": "12345678"}],
    knowledge_graph={"node_count": 4, "edge_count": 3},
    confidence=0.85,
)
```

---

## 调用流程（伪代码）

```python
# N6 ode_generator 的核心调用链
from app.ode_templates import render_template
from app.rule_engine import RuleEngine

# 1) LLM 输出定性 network_relations
network_relations: dict = llm.invoke(NODE6_RELATION_PROMPT, ...)

# 2) Rule Engine 校验
rule_engine = RuleEngine()
result: RuleResult = rule_engine.check(network_relations, parameters)
if not result.ok and state["retry_count"] < 1:
    return {"rule_violations": [...], "retry_count": state["retry_count"] + 1}

# 3) 模板渲染（注入 network_relations + parameters + 拓扑变量）
vars = {
    "network_relations": network_relations,
    "parameters": state["parameters"],
    "knowledge_graph": state["knowledge_graph"],
    "mechanism": state["mechanism"],
    "t_end": 48.0, "n_eval": 200, "y0": [...],
    # 模板特有变量
    "species_names": [...], "edges_json": json.dumps([...]), "params_json": json.dumps({...}),
    # PKPD
    "drug_name": "Galunisertib", "dose": 100.0, "k10": 0.1, "ec50": 51.0, "emax": 0.8, "gamma": 1.5,
    # 联合
    "drugs_json": json.dumps([{...}, {...}]),
    # 剂量扫描
    "conc_min_factor": 0.1, "conc_max_factor": 100.0, "n_points": 30,
}
python_code = render_template(state["mechanism"]["template"], vars)

# 4) 沙箱执行
result = execute_simulation_code_v2(python_code)
# result.simulation_csv_path → N8 读取
# result.error_class        → 路由判断
# result.image_base64       → SSE image_ready 事件
```

---

**反馈与修订**：如模板变量与代码不一致，请以代码为准并提交 PR。
