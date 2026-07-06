"""
ODE 仿真（Cascade Activation 模板）
- 多步激活级联
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.integrate import solve_ivp

SPECIES_NAMES = ['EGF', 'EGFR', 'SHC1', 'GRB2', 'SOS1', 'HRAS', 'MAPK1']
T_END = 48.0
N_EVAL = 200
Y0 = [0.008, 0.3, 10.0, 10.0, 10.0, 10.0, 10.0]
EDGES = [{'source': 'EGF', 'target': 'EGFR', 'interaction': 'activation', 'directed': True}, {'source': 'EGFR', 'target': 'SHC1', 'interaction': 'activation', 'directed': True}, {'source': 'SHC1', 'target': 'GRB2', 'interaction': 'activation', 'directed': True}, {'source': 'GRB2', 'target': 'SOS1', 'interaction': 'activation', 'directed': True}, {'source': 'SOS1', 'target': 'HRAS', 'interaction': 'activation', 'directed': True}, {'source': 'HRAS', 'target': 'MAPK1', 'interaction': 'activation', 'directed': True}]
PARAMS = {'EGFR': {'kd': 0.3, 'n': 2.0, 'production': 1.0, 'degradation': 0.1}, 'SHC1': {'kd': 0.0, 'n': 2.0, 'production': 1.0, 'degradation': 0.1}, 'GRB2': {'kd': 0.0, 'n': 2.0, 'production': 1.0, 'degradation': 0.1}, 'SOS1': {'kd': 0.0, 'n': 2.0, 'production': 1.0, 'degradation': 0.1}, 'HRAS': {'kd': 0.3, 'n': 2.0, 'production': 1.0, 'degradation': 0.1}, 'MAPK1': {'kd': 10.0, 'n': 2.0, 'production': 1.0, 'degradation': 0.1}}
N_SP = len(SPECIES_NAMES)

def _hill_act(x, kd, n):
    if x <= 0:
        return 0.0
    return (x ** n) / (kd ** n + x ** n + 1e-12)

def _ode(t, y):
    y = np.maximum(y, 0.0)
    dy = np.zeros_like(y)
    for i in range(N_SP):
        deg = PARAMS.get(SPECIES_NAMES[i], {}).get("degradation", 0.1)
        dy[i] = -deg * y[i]
    for e in EDGES:
        s_idx = SPECIES_NAMES.index(e["source"])
        t_idx = SPECIES_NAMES.index(e["target"])
        eparams = PARAMS.get(SPECIES_NAMES[t_idx], {})
        kd = eparams.get("kd", eparams.get("Kd", 1.0))
        n = eparams.get("n", eparams.get("hill", 2.0))
        prod = eparams.get("production", 1.0)
        if e["interaction"] == "activation":
            factor = _hill_act(y[s_idx], kd, n)
        else:
            factor = 0.0
        dy[t_idx] += prod * factor
    return dy.tolist()

sol = solve_ivp(
    _ode,
    (0.0, T_END),
    Y0,
    method="LSODA",
    t_eval=np.linspace(0.0, T_END, N_EVAL),
    vectorized=False,
)

header = "t," + ",".join(SPECIES_NAMES)
data = np.column_stack([sol.t, *sol.y])
np.savetxt("simulation.csv", data, delimiter=",", header=header, comments="")

fig, ax = plt.subplots(figsize=(9, 5))
for i, name in enumerate(SPECIES_NAMES):
    ax.plot(sol.t, sol.y[i], label=name, linewidth=2)
ax.set_xlabel("Time (h)")
ax.set_ylabel("Concentration (nM)")
ax.set_title("Cascade Activation Simulation")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("simulation.png", dpi=120)

for i, sp in enumerate(SPECIES_NAMES):
    val = sol.y[i, -1]
    safe_val = max(val, 0.0) if np.isfinite(val) else 0.0
    print(f"BIO_CHECK: {sp} = {safe_val:.4f}")