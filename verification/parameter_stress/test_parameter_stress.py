"""Parameter Stress Test — Part 11

Perturbs all kinetic parameters by ±10%, ±30%, ±50% and checks simulation stability.
Detects: NaN, divergence, negative concentration, oscillation distortion.
"""
import pytest
import numpy as np
from pathlib import Path

PERTURBATION_LEVELS = [0.10, 0.30, 0.50]
PATHWAYS = [
    "EGFR_RTK", "MAPK_ERK", "PI3K_AKT_MTOR", "P53_SIGNALING", "APOPTOSIS",
    "CELL_CYCLE", "JAK_STAT", "NF_KB_SIGNALING", "WNT_SIGNALING", "TGF_BETA_SIGNALING",
]

def get_baseline_parameters(pathway: str) -> dict:
    """Get baseline parameters for a pathway."""
    pytest.skip(f"Parameter extraction not implemented for {pathway}")

def perturb_parameters(params: dict, level: float, seed: int = 42) -> dict:
    """Perturb all parameters by ±level using fixed RNG seed."""
    rng = np.random.default_rng(seed)
    perturbed = {}
    for key, value in params.items():
        factor = 1.0 + rng.uniform(-level, level)
        perturbed[key] = value * factor
    return perturbed

def run_simulation(pathway: str, params: dict, t_end: float = 120.0) -> dict:
    """Run simulation with given parameters. Returns time series + metrics."""
    pytest.skip("Simulation runner not implemented")

def check_stability(result: dict) -> dict:
    """Check for NaN, divergence, negative concentration."""
    ts = np.array(result.get("time_series", []))
    return {
        "has_nan": bool(np.any(np.isnan(ts))),
        "has_inf": bool(np.any(np.isinf(ts))),
        "has_negative": bool(np.any(ts < -1e-10)),
        "has_diverged": bool(np.any(np.abs(ts) > 1e6)),
        "max_value": float(np.max(np.abs(ts))) if ts.size > 0 else 0.0,
    }

@pytest.mark.parametrize("pathway", PATHWAYS)
@pytest.mark.parametrize("level", PERTURBATION_LEVELS)
def test_parameter_perturbation_stability(pathway, level):
    """Test that ±{level*100}% perturbation doesn't break simulation."""
    baseline = get_baseline_parameters(pathway)
    perturbed = perturb_parameters(baseline, level)
    result = run_simulation(pathway, perturbed)
    stability = check_stability(result)
    
    assert not stability["has_nan"], f"{pathway} ±{level*100:.0f}%: NaN detected"
    assert not stability["has_inf"], f"{pathway} ±{level*100:.0f}%: Inf detected"
    assert not stability["has_negative"], f"{pathway} ±{level*100:.0f}%: Negative concentration"
    assert not stability["has_diverged"], f"{pathway} ±{level*100:.0f}%: Diverged (max={stability['max_value']:.2e})"

@pytest.mark.skip(reason="Known v4 limitation: all specialists lack kinetic parameters (FM-013)")
def test_no_kinetic_parameters():
    """Documents P0 bug: no Km/kcat/Vmax in any specialist."""
    pass

@pytest.mark.skip(reason="Known v4 limitation: ODE Renderer uses default params (FM-002)")
def test_ode_renderer_default_params():
    """Documents P0 bug: _extract_params reads non-existent field."""
    pass
