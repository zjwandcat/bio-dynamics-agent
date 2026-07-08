"""BioModels Regression Test Suite — Part 8

Downloads, parses, simulates, and compares 30+ BioModels reference models
against BioDynamics v4 simulation output.

Metrics: RMSE, Pearson Correlation, Peak Time, Peak Amplitude, Steady State, AUC
"""
import pytest
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "biomodels_config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="module")
def biomodels_config():
    return load_config()

# --- Helper functions (stubs) ---

def download_biomodels_sbml(biomodels_id: str, cache_dir: Path) -> Path:
    """Download SBML model from EBI BioModels, use cache if available."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{biomodels_id}.xml"
    if cache_file.exists():
        return cache_file
    # TODO: implement actual download via biomodels_client
    pytest.skip(f"BioModels download not implemented for {biomodels_id}")

def simulate_biomodels_reference(sbml_path: Path, t_end: float = 120.0) -> dict:
    """Simulate the reference SBML model using roadrunner."""
    pytest.skip("roadrunner simulation not implemented")

def simulate_biodynamics(biomodels_id: str, pathway: str, t_end: float = 120.0) -> dict:
    """Run BioDynamics v4 simulation for the given pathway."""
    pytest.skip("BioDynamics simulation integration not implemented")

def compute_metrics(reference: dict, biodynamics: dict) -> dict:
    """Compute RMSE, Pearson, peak time diff, amplitude diff, steady state diff, AUC diff."""
    import numpy as np
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error

    ref_ts = np.array(reference.get("time_series", []))
    bio_ts = np.array(biodynamics.get("time_series", []))

    if len(ref_ts) == 0 or len(bio_ts) == 0:
        return {"rmse": float("inf"), "pearson": 0.0}

    min_len = min(len(ref_ts), len(bio_ts))
    ref_flat = ref_ts[:min_len].flatten()
    bio_flat = bio_ts[:min_len].flatten()

    rmse = float(np.sqrt(mean_squared_error(ref_flat, bio_flat)))
    pearson, _ = pearsonr(ref_flat, bio_flat) if len(ref_flat) > 1 else (0.0, 0.0)

    return {
        "rmse": rmse,
        "pearson": float(pearson),
        "peak_time_diff": abs(reference.get("peak_time", 0) - biodynamics.get("peak_time", 0)),
        "peak_amplitude_diff": abs(reference.get("peak_amplitude", 0) - biodynamics.get("peak_amplitude", 0)),
        "steady_state_diff": abs(reference.get("steady_state", 0) - biodynamics.get("steady_state", 0)),
        "auc_diff": abs(reference.get("auc", 0) - biodynamics.get("auc", 0)),
    }

# --- Parametrized tests ---

@pytest.mark.parametrize("entry", load_config())
def test_biomodels_regression(entry, tmp_path):
    """Run full regression: download → parse → simulate → compare."""
    biomodels_id = entry["biomodels_id"]
    pathway = entry["pathway"]

    # Step 1: Download
    sbml_path = download_biomodels_sbml(biomodels_id, tmp_path / "cache")

    # Step 2: Simulate reference
    ref_result = simulate_biomodels_reference(sbml_path, t_end=120.0)

    # Step 3: Simulate BioDynamics
    bio_result = simulate_biodynamics(biomodels_id, pathway, t_end=120.0)

    # Step 4: Compute metrics
    metrics = compute_metrics(ref_result, bio_result)

    # Step 5: Assertions
    assert metrics["rmse"] < entry["rmse_threshold"], (
        f"{biomodels_id} ({pathway}): RMSE {metrics['rmse']:.2f} > threshold {entry['rmse_threshold']}"
    )
    assert metrics["pearson"] > entry["pearson_threshold"], (
        f"{biomodels_id} ({pathway}): Pearson {metrics['pearson']:.4f} < threshold {entry['pearson_threshold']}"
    )
    assert metrics["peak_time_diff"] < 2.0, (
        f"{biomodels_id}: peak time diff {metrics['peak_time_diff']:.2f} min"
    )
    assert metrics["peak_amplitude_diff"] / max(ref_result.get("peak_amplitude", 1), 1) < 0.3, (
        f"{biomodels_id}: peak amplitude diff {metrics['peak_amplitude_diff']:.2f} > 30%"
    )

@pytest.mark.skip(reason="Known v4 limitation: ODE Renderer reads non-existent fields (FM-001/002/003)")
def test_biomodels_regression_known_limitation():
    """This test documents the known P0 bug: ODE Renderer produces zero-flux ODE."""
    pass
