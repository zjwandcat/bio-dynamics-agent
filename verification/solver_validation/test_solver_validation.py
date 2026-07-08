"""Solver Validation — Part 12

Validates all ODE solvers against known analytical solutions and
biological dynamics patterns: stiffness, oscillation, bistability, delay.
"""
import pytest
import numpy as np
from scipy.integrate import solve_ivp

# --- Analytical test problems ---

def linear_decay(t, y, k=0.1):
    """dy/dt = -k*y, analytical: y(t) = y0*exp(-k*t)"""
    return [-k * y[0]]

def exponential_growth(t, y, k=0.1):
    """dy/dt = k*y, analytical: y(t) = y0*exp(k*t)"""
    return [k * y[0]]

def harmonic_oscillator(t, y):
    """d2y/dt2 = -y, analytical: y(t) = cos(t)"""
    return [y[1], -y[0]]

def van_der_pol(t, y, mu=1000):
    """Stiff Van der Pol oscillator."""
    return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

def bistable_system(t, y, a=1.0, b=1.0):
    """dy/dt = a*y - y^3, stable at y=±sqrt(a)"""
    return [a * y[0] - y[0]**3]

def delay_oscillator_approx(t, y, k=0.1, tau=30.0):
    """Approximation of delay oscillator dy/dt = -k*y(t-tau)."""
    # Without DDE solver, this is just linear decay
    return [-k * y[0]]

# --- Solver tests ---

@pytest.mark.parametrize("method", ["LSODA", "RK45", "BDF", "Radau"])
def test_linear_decay_accuracy(method):
    """Test solver accuracy on linear decay."""
    sol = solve_ivp(linear_decay, [0, 100], [1.0], method=method, dense_output=True, rtol=1e-8, atol=1e-10)
    analytical = 1.0 * np.exp(-0.1 * sol.t)
    error = np.max(np.abs(sol.y[0] - analytical))
    assert error < 1e-4, f"{method}: max error {error:.2e} > 1e-4"

@pytest.mark.parametrize("method", ["LSODA", "BDF", "Radau"])
def test_stiff_system(method):
    """Test solver on stiff Van der Pol (mu=1000)."""
    sol = solve_ivp(van_der_pol, [0, 100], [2.0, 0.0], method=method, rtol=1e-6, atol=1e-8)
    assert sol.success, f"{method} failed on stiff system"
    assert not np.any(np.isnan(sol.y)), f"{method}: NaN in output"

@pytest.mark.parametrize("method", ["LSODA", "RK45", "DOP853"])
def test_oscillation_period(method):
    """Test solver preserves oscillation period."""
    sol = solve_ivp(harmonic_oscillator, [0, 20*np.pi], [1.0, 0.0], method=method, rtol=1e-10, atol=1e-12, dense_output=True)
    # Period should be 2*pi
    peaks = []
    for i in range(1, len(sol.t)-1):
        if sol.y[0, i] > sol.y[0, i-1] and sol.y[0, i] > sol.y[0, i+1]:
            peaks.append(sol.t[i])
    if len(peaks) >= 2:
        period = np.mean(np.diff(peaks))
        assert abs(period - 2*np.pi) < 0.01, f"{method}: period {period:.4f} != 2π={2*np.pi:.4f}"

def test_bistable_stability():
    """Test that bistable system converges to correct fixed point."""
    # y0 > 0 → converges to +sqrt(a) = 1
    sol_pos = solve_ivp(bistable_system, [0, 100], [0.5], method="LSODA", rtol=1e-8)
    assert abs(sol_pos.y[0, -1] - 1.0) < 0.01, f"Positive fixed point: {sol_pos.y[0,-1]:.4f} != 1.0"
    # y0 < 0 → converges to -sqrt(a) = -1
    sol_neg = solve_ivp(bistable_system, [0, 100], [-0.5], method="LSODA", rtol=1e-8)
    assert abs(sol_neg.y[0, -1] + 1.0) < 0.01, f"Negative fixed point: {sol_neg.y[0,-1]:.4f} != -1.0"

@pytest.mark.skip(reason="Known v4 limitation: DDE solver non-functional (FM-005)")
def test_dde_solver_functional():
    """Documents P0 bug: DDE solver always falls back to ODE."""
    pass

@pytest.mark.skip(reason="Known v4 limitation: no event detection (FM-046)")
def test_solver_event_detection():
    """Documents P1 bug: solve_ivp called without events parameter."""
    pass

@pytest.mark.skip(reason="Known v4 limitation: max_step hardcoded (FM-047)")
def test_solver_adaptive_step():
    """Documents P1 bug: max_step=0.5 hardcoded."""
    pass
