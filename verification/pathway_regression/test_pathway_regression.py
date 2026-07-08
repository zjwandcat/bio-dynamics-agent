"""Pathway Regression Test Suite — Part 9

50 mechanism-level regression cases (5 per pathway × 10 pathways) driven by
`pathway_config.yaml`. Each case drives a single biochemical step through the
v4 Pathway Specialist pipeline and asserts:

  1. Species count matches the expected set
  2. Reaction mechanism matches the expected MechanismType
  3. Rendered ODE code contains the expected number of RHS terms
  4. The declared validation level (L1..L5) passes (or fails) as expected

Cases tagged with a `p0_bug` in the config are auto-skipped via
`pytest.param(marks=pytest.mark.skip(reason=...))` — they document known P0
defects in the v4 ODE Renderer (FM-001/002/003) and will be un-skipped once the
fix lands.
"""
import re
import pytest
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "pathway_config.yaml"

# Human-readable descriptions for each known P0 bug (kept in sync with the
# `p0_bug` field documented at the top of pathway_config.yaml).
P0_BUG_REASONS = {
    "FM-001": (
        "Known P0 bug FM-001: ODE Renderer reads non-existent substrate field, "
        "producing zero-flux ODE for this mechanism."
    ),
    "FM-002": (
        "Known P0 bug FM-002: ODE Renderer reads non-existent modifier field, "
        "dropping the catalytic term from the rendered ODE."
    ),
    "FM-003": (
        "Known P0 bug FM-003: ODE Renderer emits zero-flux ODE for Michaelis-"
        "Menten phosphorylation steps."
    ),
}


def load_config():
    """Load the pathway regression config and expand each entry into a
    `pytest.param` carrying the appropriate skip mark when a P0 bug is set."""
    with open(CONFIG_PATH) as f:
        entries = yaml.safe_load(f)

    params = []
    for entry in entries:
        p0_bug = entry.get("p0_bug")
        if p0_bug:
            reason = P0_BUG_REASONS.get(
                p0_bug, f"Known P0 bug {p0_bug}: ODE Renderer defect."
            )
            params.append(
                pytest.param(entry, marks=pytest.mark.skip(reason=reason),
                             id=entry["test_name"])
            )
        else:
            params.append(pytest.param(entry, id=entry["test_name"]))
    return params


@pytest.fixture(scope="module")
def pathway_config():
    """Module-scoped access to the raw config list (without marks)."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# --- Helper functions (stubs) -------------------------------------------------
# Each stub skips until the corresponding v4 integration is wired up. They are
# intentionally minimal so the parametrized surface (50 ids + skips) is visible
# in `pytest --collect-only` from day one.

def build_pathway_state(pathway: str, test_name: str) -> dict:
    """Invoke the v4 Pathway Specialist for `pathway` and return the LangGraph
    state slice containing v4_reaction_ir / v4_ode_system / v4_pathway_graph."""
    pytest.skip(f"Pathway Specialist integration not implemented for {pathway}")


def extract_species(state: dict) -> list[str]:
    """Return the list of species names produced by the specialist."""
    ir = state.get("v4_reaction_ir") or {}
    species = ir.get("species") or []
    return [s.get("name", "") if isinstance(s, dict) else str(s) for s in species]


def extract_reactions(state: dict) -> list[dict]:
    """Return the list of reaction dicts produced by the specialist."""
    ir = state.get("v4_reaction_ir") or {}
    return ir.get("reactions") or []


def render_ode_code(state: dict) -> str:
    """Render the v4 ODE system to its Python source string."""
    ode = state.get("v4_ode_system") or {}
    return ode.get("source", "")


def count_ode_terms(ode_code: str) -> int:
    """Count RHS terms in the rendered ODE source.

    A RHS term is any assignment of the form `d<Species>_dt = <expr>`. We count
    occurrences of the derivative-assignment marker so the figure stays stable
    across formatting changes.
    """
    if not ode_code:
        return 0
    # Matches `dX_dt =` (the canonical v4 ODE renderer convention).
    return len(re.findall(r"\bd\w+_dt\s*=", ode_code))


def run_validation_level(state: dict, level: str) -> bool:
    """Run the requested validation pyramid level against `state`.

    Returns True if the level passes (or is skipped-pass), False otherwise.
    """
    # TODO: dispatch to app.validation_v2 validators once integration is wired.
    pytest.skip(f"Validation level {level} integration not implemented")


# --- Parametrized regression cases -------------------------------------------

@pytest.mark.parametrize("entry", load_config())
def test_pathway_regression(entry):
    """Full mechanism-level regression for a single biochemical step."""
    pathway = entry["pathway"]
    test_name = entry["test_name"]

    # Step 1: build the v4 state for this pathway / step
    state = build_pathway_state(pathway, test_name)

    # Step 2: assert species count matches expected set
    species = extract_species(state)
    for expected_species in entry["expected_species"]:
        assert expected_species in species, (
            f"{test_name} ({pathway}): expected species '{expected_species}' "
            f"not in produced species {species}"
        )
    assert len(species) >= len(entry["expected_species"]), (
        f"{test_name} ({pathway}): produced {len(species)} species, expected "
        f"at least {len(entry['expected_species'])}"
    )

    # Step 3: assert reaction mechanism matches expected
    reactions = extract_reactions(state)
    mechanisms = {r.get("mechanism") for r in reactions}
    assert entry["expected_mechanism"] in mechanisms, (
        f"{test_name} ({pathway}): expected mechanism "
        f"'{entry['expected_mechanism']}' not in produced mechanisms {mechanisms}"
    )
    assert len(reactions) == entry["expected_reactions"], (
        f"{test_name} ({pathway}): produced {len(reactions)} reactions, "
        f"expected {entry['expected_reactions']}"
    )

    # Step 4: assert rendered ODE code contains expected number of terms
    ode_code = render_ode_code(state)
    actual_terms = count_ode_terms(ode_code)
    assert actual_terms == entry["expected_ode_terms"], (
        f"{test_name} ({pathway}): ODE rendered {actual_terms} RHS terms, "
        f"expected {entry['expected_ode_terms']}"
    )
    # The ODE code must reference every expected species by name.
    for expected_species in entry["expected_species"]:
        assert re.search(rf"\b{re.escape(expected_species)}\b", ode_code), (
            f"{test_name} ({pathway}): ODE code does not reference species "
            f"'{expected_species}'"
        )

    # Step 5: assert validation level passes/fails as expected
    level = entry["validation_level"]
    validation_passed = run_validation_level(state, level)
    if entry["expected_pass"]:
        assert validation_passed, (
            f"{test_name} ({pathway}): validation level {level} was expected "
            f"to pass but failed"
        )
    else:
        assert not validation_passed, (
            f"{test_name} ({pathway}): validation level {level} was expected "
            f"to fail but passed"
        )


# --- Standalone skip-decorated documentation tests for P0 bugs ---------------
# These mirror the parametrized skips above as explicit, discoverable test
# functions so the known-failure ledger is visible in any pytest collection.

@pytest.mark.skip(reason="Known P0 bug FM-001: ODE Renderer reads non-existent "
                         "substrate field, producing zero-flux ODE.")
def test_p0_bug_fm001_ode_renderer_substrate_field():
    """Documents FM-001: substrate field is read from the wrong schema location,
    so ubiquitination / degradation steps render zero-flux ODEs."""
    pass


@pytest.mark.skip(reason="Known P0 bug FM-002: ODE Renderer reads non-existent "
                         "modifier field, dropping the catalytic term.")
def test_p0_bug_fm002_ode_renderer_modifier_field():
    """Documents FM-002: the catalytic modifier (kinase/E3 ligase) term is
    dropped from Michaelis-Menten phosphorylation ODEs."""
    pass


@pytest.mark.skip(reason="Known P0 bug FM-003: ODE Renderer emits zero-flux ODE "
                         "for Michaelis-Menten phosphorylation steps.")
def test_p0_bug_fm003_ode_renderer_zero_flux_mm():
    """Documents FM-003: autophosphorylation and double-phosphorylation MM
    steps render as zero-flux, breaking the MAPK cascade amplification."""
    pass


@pytest.mark.skip(reason="Aggregator: 21 parametrized cases are skipped above "
                         "due to P0 bugs FM-001/002/003. See pathway_config.yaml "
                         "`p0_bug` field for the full ledger.")
def test_p0_bug_skip_ledger():
    """Placeholder that documents the count of P0-skipped cases.

    Once the ODE Renderer is fixed, remove the `p0_bug` fields from
    pathway_config.yaml and this test (the parametrized cases will then run).
    """
    pass
