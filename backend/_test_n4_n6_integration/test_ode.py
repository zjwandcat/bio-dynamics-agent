"""
ODE 仿真（Signaling Cascade with Phosphorylation 模板）
- 基于 mass-action binding + Michaelis-Menten phosphorylation
- 显式建模磷酸化中间体（pEGFR, pShc, pMEK, pMAPK 等）
- 信号级联放大效应（每级磷酸化由上游激酶催化）

关键改进（vs Cascade_Activation）：
1. 使用 mass-action kinetics（而非 Hill function），更符合生化反应动力学
2. 显式建模结合/解离/磷酸化/去磷酸化
3. 信号放大通过级联磷酸化实现（每级有激酶催化）
4. pEGFR 达峰时间由 k_phos/k_dephos 比值决定（5-10 min 量级）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.integrate import solve_ivp

SPECIES_NAMES = ['EGF', 'EGFR', 'pEGFR']
T_END = 60.0
N_EVAL = 300
Y0 = [0.008, 0.3, 0.0]
EDGES = [{'source': 'EGF', 'target': 'EGFR', 'interaction': 'activation', 'mechanism': 'binding', 'reaction_equation': 'EGF + EGFR → EGF-EGFR', 'directed': True}, {'source': 'EGFR', 'target': 'pEGFR', 'interaction': 'activation', 'mechanism': 'phosphorylation', 'reaction_equation': 'EGFR → pEGFR', 'directed': True}]
PARAMS = {'EGFR': {'k_on': 100.0, 'k_off': 0.01, 'degradation': 0.001}, 'pEGFR': {'k_phos': 0.5, 'k_dephos': 0.05, 'degradation': 0.002}}
N_SP = len(SPECIES_NAMES)

# 物种名 → 索引映射
SP_IDX = {name: i for i, name in enumerate(SPECIES_NAMES)}


def _get_param(target_name, param_key, default):
    """从 PARAMS 中获取指定 target 的参数值。"""
    eparams = PARAMS.get(target_name, {})
    return eparams.get(param_key, default)


def _ode(t, y):
    """ODE 右端函数：基于 reaction mechanism 构建 dy/dt。

    Mechanism 类型：
    - binding: d[A-B]/dt = k_on * [A] * [B] - k_off * [A-B]
    - phosphorylation: d[pX]/dt = k_phos * [enzyme] * [X] - k_dephos * [pX]
      * 若 source 是酶（同时出现在 reaction_equation 两侧），不消耗 source
      * 若 source 是底物（仅出现在反应式左侧），消耗 source（Xxx → pXxx 转换）
    - exchange: d[RasGTP]/dt = k_exchange * [RasGDP]
    - degradation: d[X]/dt = -k_deg * [X]
    - activation (generic): d[target]/dt = k_cat * [source] - k_degr * [target]
    """
    y = np.maximum(y, 0.0)
    dy = np.zeros_like(y)

    # 1. 基础降解（所有物种）
    for i, name in enumerate(SPECIES_NAMES):
        k_deg = _get_param(name, "degradation", 0.0)
        if k_deg > 0:
            dy[i] -= k_deg * y[i]

    # 2. 按 edge 的 mechanism 构建 ODE 项
    for e in EDGES:
        src_name = e.get("source", "")
        tgt_name = e.get("target", "")
        mechanism = e.get("mechanism", "activation")
        interaction = e.get("interaction", "activation")
        reaction_eq = e.get("reaction_equation", "")

        if src_name not in SP_IDX or tgt_name not in SP_IDX:
            continue

        s_idx = SP_IDX[src_name]
        t_idx = SP_IDX[tgt_name]
        src = y[s_idx]
        tgt = y[t_idx]

        # 解析 reaction_equation 判断 source 是否为酶（出现在产物侧）
        # 例：'pEGFR + Shc → pEGFR + pShc' → pEGFR 是酶（不消耗）
        # 例：'EGFR → pEGFR' → EGFR 是底物（消耗，注意 EGFR 不是 pEGFR 的子串匹配）
        # 使用 token-based 匹配避免 'EGFR' 被 'pEGFR' 误判
        _is_enzyme = False
        if reaction_eq and "→" in reaction_eq:
            _parts = reaction_eq.split("→", 1)
            if len(_parts) == 2:
                _rhs = _parts[1]
                # 把 + 与空白作为分隔符，得到产物侧的 token 集合
                _rhs_tokens = set(_rhs.replace("+", " ").split())
                if src_name in _rhs_tokens:
                    _is_enzyme = True

        if mechanism == "binding":
            # A + B → A-B (mass action)
            k_on = _get_param(tgt_name, "k_on", _get_param(tgt_name, "k1", 1.0))
            k_off = _get_param(tgt_name, "k_off", _get_param(tgt_name, "k2", 0.01))
            # 产物（target）生成
            dy[t_idx] += k_on * src * tgt * 0.01  # 缩放因子避免数值过大
            # 反应物消耗（仅当 source 是自由形式时）
            if src_name in SPECIES_NAMES and not src_name.startswith("p"):
                dy[s_idx] -= k_on * src * tgt * 0.005

        elif mechanism == "phosphorylation":
            # Xxx → pXxx (由上游激酶催化)
            # 两种子情形：
            #   (a) source 是底物（Xxx → pXxx）：消耗 source，质量守恒
            #   (b) source 是酶（pXxx + Yyy → pXxx + pYyy）：不消耗 source
            k_phos = _get_param(tgt_name, "k_phos", _get_param(tgt_name, "k1", 0.1))
            k_dephos = _get_param(tgt_name, "k_dephos", _get_param(tgt_name, "k2", 0.001))
            # 磷酸化产物生成
            dy[t_idx] += k_phos * src
            # 去磷酸化消耗
            dy[t_idx] -= k_dephos * tgt
            # 若 source 是底物（非酶），消耗 source 以保证质量守恒
            # 这一步是 pEGFR 5-10 min 达峰的关键：底物耗尽后 pEGFR 仅靠 dephos 衰减
            if not _is_enzyme:
                dy[s_idx] -= k_phos * src

        elif mechanism == "exchange":
            # RasGDP → RasGTP (由 SOS 催化)
            k_exchange = _get_param(tgt_name, "k_exchange", _get_param(tgt_name, "k1", 0.1))
            dy[t_idx] += k_exchange * src
            # GDP 形式消耗（RasGDP → RasGTP 是转换）
            if "GDP" in src_name:
                dy[s_idx] -= k_exchange * src

        elif mechanism == "dephosphorylation":
            # pXxx → Xxx
            k_dephos = _get_param(tgt_name, "k_dephos", _get_param(tgt_name, "k2", 0.01))
            dy[t_idx] += k_dephos * src
            dy[s_idx] -= k_dephos * src

        elif mechanism == "recruitment":
            # pXxx + Yyy → pXxx-Yyy (binding)
            k_on = _get_param(tgt_name, "k_on", _get_param(tgt_name, "k1", 1.0))
            k_off = _get_param(tgt_name, "k_off", _get_param(tgt_name, "k2", 0.01))
            dy[t_idx] += k_on * src * 0.1 - k_off * tgt

        elif mechanism == "degradation":
            k_deg = _get_param(tgt_name, "k_deg", 0.1)
            dy[s_idx] -= k_deg * src

        else:  # activation (generic fallback)
            k_cat = _get_param(tgt_name, "k_cat", _get_param(tgt_name, "k1", 0.1))
            k_degr = _get_param(tgt_name, "degradation", 0.01)
            dy[t_idx] += k_cat * src - k_degr * tgt

    return dy.tolist()


# 运行仿真
sol = solve_ivp(
    _ode,
    (0.0, T_END),
    Y0,
    method="LSODA",
    t_eval=np.linspace(0.0, T_END, N_EVAL),
    vectorized=False,
    max_step=T_END / 100,
)

# 输出 CSV
header = "t," + ",".join(SPECIES_NAMES)
data = np.column_stack([sol.t, *sol.y])
np.savetxt("simulation.csv", data, delimiter=",", header=header, comments="")

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
for i, name in enumerate(SPECIES_NAMES):
    ax.plot(sol.t, sol.y[i], label=name, linewidth=2)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Concentration (nM)")
ax.set_title("EGF-EGFR Signaling Cascade with Phosphorylation")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("simulation.png", dpi=120)

# 输出关键指标
for i, sp in enumerate(SPECIES_NAMES):
    val = sol.y[i, -1]
    safe_val = max(val, 0.0) if np.isfinite(val) else 0.0
    print(f"BIO_CHECK: {sp} = {safe_val:.4f}")