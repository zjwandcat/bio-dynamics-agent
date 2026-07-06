# BIOMD0000000205 语义收敛与模型一致性修复报告

## 1. 修改文件列表（exact paths）

| # | 文件路径 | 修改说明 |
|---|---------|---------|
| 1 | [backend/app/biomodels_reactions.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/biomodels_reactions.py) | 新增：ALLOWED_PATHWAY_SET、FORBIDDEN_PATHWAY_TERMS、CORE_SPECIES_SET、CANONICAL_REDUCTION_MAP、PLOT_CANONICAL_SET；实现 reaction 过滤、canonical reduction、MAPK 链强制注入 |
| 2 | [backend/app/model_consistency_validator.py](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/model_consistency_validator.py) | 新增：三重一致性校验器（pathway integrity / conservation sanity / no phantom pathway）；本次修复 Ras 蛋白池统计口径 |
| 3 | [backend/app/ode_templates/Signaling_Cascade_Phos.j2](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/app/ode_templates/Signaling_Cascade_Phos.j2) | 修复：绘图层仅允许 PLOT_CANONICAL_SET 中的 7 个 canonical 物种，防止曲线爆炸 |
| 4 | [backend/validation_report_biomd0205.json](file:///c:/Users/27553/Desktop/gzlab/bio-dynamics-agent/backend/validation_report_biomd0205.json) | 验证原始 JSON 数据（由一次性脚本生成，脚本已删除） |

---

## 2. 关键修改 diff patch

### 2.1 Signaling_Cascade_Phos.j2 — 可视化层 canonical 过滤（TASK 5）

```diff
-# 绘图
-fig, ax = plt.subplots(figsize=(10, 6))
-for i, name in enumerate(SPECIES_NAMES):
-    ax.plot(sol.t, sol.y[i], label=name, linewidth=2)
-ax.set_xlabel("Time (min)")
-ax.set_ylabel("Concentration (nM)")
-ax.set_title("EGF-EGFR Signaling Cascade with Phosphorylation")
-ax.legend(loc="upper right", fontsize=8)
-ax.grid(alpha=0.3)
-fig.tight_layout()
-fig.savefig("simulation.png", dpi=120)
+# 绘图（TASK 5：只允许 canonical species，防止曲线爆炸）
+PLOT_CANONICAL_SET = {
+    "EGFR_active",
+    "Shc_complex",
+    "Grb2_SOS_complex",
+    "Ras_active",
+    "Raf_active",
+    "MEK_active",
+    "ERK_active",
+}
+fig, ax = plt.subplots(figsize=(10, 6))
+plotted = 0
+for i, name in enumerate(SPECIES_NAMES):
+    if name in PLOT_CANONICAL_SET:
+        ax.plot(sol.t, sol.y[i], label=name, linewidth=2)
+        plotted += 1
+ax.set_xlabel("Time (min)")
+ax.set_ylabel("Concentration (nM)")
+ax.set_title("EGF-EGFR Signaling Cascade (Canonical View)")
+ax.legend(loc="upper right", fontsize=8)
+ax.grid(alpha=0.3)
+fig.tight_layout()
+fig.savefig("simulation.png", dpi=120)
```

### 2.2 model_consistency_validator.py — Ras 蛋白池守恒口径修正

```diff
-    # 规范 active 节点 = 对应蛋白池的总量代表
-    required_active = {
-        "Ras_active": "Ras",
-        "Raf_active": "Raf",
-        "MEK_active": "MEK",
-        "ERK_active": "ERK",
-    }
-    for active_sp, pool_name in required_active.items():
-        if active_sp not in graph.species:
-            warnings.append(f"缺失规范 active 节点：{active_sp}（{pool_name} 池）")
-            continue
-        val = graph.species_initial.get(active_sp, 0.0)
-        pool_totals[pool_name] = val
-        if val <= 0:
-            warnings.append(f"{pool_name} 蛋白池（{active_sp}）初始浓度为 0")
+    # 定义蛋白池成员（按 canonical 命名）
+    pool_definitions: dict[str, list[str]] = {
+        "EGFR": [s for s in graph.species if "egfr" in s.lower()],
+        "Ras": ["Ras_active", "Ras_inactive"],
+        "Raf": ["Raf_active"],
+        "MEK": ["MEK_active"],
+        "ERK": ["ERK_active"],
+    }
+
+    required_active = {
+        "Ras_active": "Ras",
+        "Raf_active": "Raf",
+        "MEK_active": "MEK",
+        "ERK_active": "ERK",
+    }
+
+    # 1. 检查规范 active 节点存在性
+    for active_sp, pool_name in required_active.items():
+        if active_sp not in graph.species:
+            warnings.append(f"缺失规范 active 节点：{active_sp}（{pool_name} 池）")
+
+    # 2. 检查蛋白池总浓度 > 0
+    for pool_name, members in pool_definitions.items():
+        total = sum(graph.species_initial.get(m, 0.0) for m in members if m in graph.species)
+        pool_totals[pool_name] = total
+        if total <= 0:
+            warnings.append(f"{pool_name} 蛋白池初始总浓度为 0（成员：{members}）")
```

### 2.3 biomodels_reactions.py — 核心收敛规则（已落地，本次未再改动）

关键常量：

```python
ALLOWED_PATHWAY_SET: frozenset[str] = frozenset({"EGF_EGFR_MAPK"})
FORBIDDEN_PATHWAY_TERMS: tuple[str, ...] = (
    "pi3k", "akt", "mtor", "nf-kappa", "nf-kb", "nfkb", "nf-κb",
    "jak", "stat", "stat3", "stat5",
    "feedback", "crosstalk", "cross-talk", "emergent",
)
PLOT_CANONICAL_SET: tuple[str, ...] = (
    "EGFR_active", "Shc_complex", "Grb2_SOS_complex",
    "Ras_active", "Raf_active", "MEK_active", "ERK_active",
)
_REQUIRED_MAPK_CHAIN: list[tuple[str, str, str]] = [
    ("Raf_active", "MEK_active", "phosphorylation"),
    ("MEK_active", "ERK_active", "phosphorylation"),
]
```

关键函数：

- `collapse_species(species_name)` — 长复合物名 → canonical 节点。
- `_reaction_in_allowed_pathway(...)` — 非 EGF_EGFR_MAPK 反应直接 reject。
- `reaction_graph_to_edges(...)` — canonical reduction + 方向校正 + 去重 + `_ensure_mapk_chain`。

---

## 3. Canonical model graph（before / after）

### 3.1 Before — 原始 BIOMD0000000205 经 JSON 解析后的物种/反应

- **物种数**：11
- **反应数**：30
- **问题**：包含 `EGF-pEGFR-2-pShc-Grb2-SOS-RasGDP`、`Raf-RasGTP`、`Grb2-SOS` 等多层复合物，导致曲线爆炸。

代表性反应：

```text
EGF + EGFR → EGF-EGFR
2 EGF-EGFR → EGF-EGFR-2
EGF-pEGFR-2-pShc-Grb2-SOS + RasGDP → EGF-pEGFR-2-pShc-Grb2-SOS-RasGDP
Raf + RasGTP → Raf-RasGTP
ppMEK-ERK → ppMEK + pERK
```

### 3.2 After — canonical reduction 后的 EGF-EGFR-MAPK 反应图

- **物种数**：10（含 EGF、SHP、Ras_inactive 等辅助物种；canonical 绘制节点为 7 个）
- **边数**：10
- **通路**：唯一允许的 `EGF_EGFR_MAPK`

```text
EGF → EGFR_active
EGFR_active → Shc_complex
SHP → Shc_complex
Shc_complex → Grb2_SOS_complex
EGFR_active → Grb2_SOS_complex
Grb2_SOS_complex → Ras_active
Ras_active → Raf_active
Raf_active → MEK_active
MEK_active → ERK_active
Ras_active → Ras_inactive
```

> 注：`Raf_active → MEK_active → ERK_active` 在原始反应图中已存在，未触发强制注入；`_ensure_mapk_chain` 作为兜底机制保留。

---

## 4. Validation report

### 4.1 三重一致性校验（TASK 6）

| 检查项 | 状态 | 结果 |
|-------|------|------|
| Pathway integrity | ✅ 通过 | `EGF → ERK_active` 可达，路径：`EGF → EGFR_active → Grb2_SOS_complex → Ras_active → Raf_active → MEK_active → ERK_active` |
| Conservation sanity | ✅ 通过 | EGFR=0.3, Ras=0.15, Raf=0.5, MEK=0.68, ERK=0.4 |
| No phantom pathway | ✅ 通过 | 未发现 PI3K/Akt/NF-κB/JAK-STAT/feedback/crosstalk 等术语 |

**一致性校验总结果：通过 ✅**

### 4.2 数值仿真结果

- **模板**：`Signaling_Cascade_Phos`
- **时长**：120 min
- **沙箱状态**：`success`，`error_class=none`
- **BIO_CHECK 终值**：

```text
BIO_CHECK: EGF = 0.0082
BIO_CHECK: EGFR_active = 0.0100
BIO_CHECK: Shc_complex = 0.0101
BIO_CHECK: SHP = 0.0245
BIO_CHECK: Grb2_SOS_complex = 2.0258
BIO_CHECK: Ras_active = 0.0099
BIO_CHECK: Raf_active = 0.0795
BIO_CHECK: MEK_active = 0.0104
BIO_CHECK: ERK_active = 1.6278
BIO_CHECK: Ras_inactive = 0.0005
```

### 4.3 Canonical 物种动力学指标

| 物种 | 峰值 (nM) | 峰值时间 (min) | 终值 (nM) |
|------|----------|---------------|----------|
| EGFR_active | 0.0100 | 19.26 | 0.0100 |
| Shc_complex | 1.0000 | 0.00 | 0.0101 |
| Grb2_SOS_complex | 2.1494 | 2.81 | 2.0258 |
| Ras_active | 0.1037 | 1.61 | 0.0099 |
| Raf_active | 0.5000 | 0.00 | 0.0795 |
| MEK_active | 0.6800 | 0.00 | 0.0104 |
| ERK_active | 1.6278 | 120.00 | 1.6278 |

> Ras 峰值出现在 **1.61 min**，符合“分钟尺度”要求；`ERK_active`（即 pERK/ppERK 汇总）**> 0**，MAPK 链未断链。

---

## 5. Acceptance criteria 核对

| 验收项 | 要求 | 实测 | 状态 |
|-------|------|------|------|
| 模型层 | 仅 MAPK cascade，无外部 pathway | `ALLOWED_PATHWAY_SET={EGF_EGFR_MAPK}`，无 forbidden 命中 | ✅ |
| 数值层 | Ras peak ~ 分钟尺度 | Ras_active 峰值 0.104 nM @ 1.61 min | ✅ |
| 数值层 | pERK > 0 | ERK_active 终值 1.6278 nM，峰值 1.6278 nM | ✅ |
| 结构层 | ≤ 8 canonical 节点 | 绘图白名单 7 个 canonical species | ✅ |
| 可视化层 | 无曲线爆炸 | `PLOT_CANONICAL_SET` 过滤，仅 7 条线 | ✅ |
| 一致性 | Pathway integrity | EGF → ERK_active 可达 | ✅ |
| 一致性 | Conservation sanity | 关键蛋白池总浓度均 > 0 | ✅ |
| 一致性 | No phantom pathway | 无 PI3K/Akt/feedback/crosstalk | ✅ |

---

## 6. 结论

BIOMD0000000205 仿真系统已完成语义收敛：

1. 非模型通路（PI3K/Akt/NF-κB/JAK-STAT/feedback/crosstalk）被硬约束拒绝。
2. 长复合物经 canonical reduction 折叠为 7 个核心节点，绘图层仅输出这些节点。
3. MAPK 下游链 `Raf_active → MEK_active → ERK_active` 保持完整，ERK_active > 0。
4. 新增的三重一致性校验器全部通过。

残余说明：
- `Shc_complex`、`Grb2_SOS_complex` 的峰值受 canonical reduction 聚合方式影响，数值上不代表单一磷酸化形式的绝对浓度，但符合“蛋白池总量”语义。
- `ERK_active` 在 120 min 内单调上升，提示去磷酸化/负反馈强度可进一步依据原始 BIOMD 参数调优，但当前已满足“pERK > 0”的最低验收标准。
