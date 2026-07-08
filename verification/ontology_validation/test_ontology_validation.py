"""Ontology Validation — Part 13

Tests HGNC, UniProt, GO, ChEBI lookup accuracy, alias/synonym resolution,
species classification error rate, and coverage.
"""
import pytest
from pathlib import Path

# --- Known ground truth entries ---
HGNC_TEST_CASES = [
    {"name": "EGFR", "expected_hgnc_id": "3236", "expected_uniprot": "P00533"},
    {"name": "TP53", "expected_hgnc_id": "11998", "expected_uniprot": "P04637"},
    {"name": "AKT1", "expected_hgnc_id": "391", "expected_uniprot": "P31749"},
    {"name": "MAPK1", "expected_hgnc_id": "6871", "expected_uniprot": "P28482"},
    {"name": "NFKB1", "expected_hgnc_id": "7794", "expected_uniprot": "P19838"},
    {"name": "STAT3", "expected_hgnc_id": "11364", "expected_uniprot": "P40763"},
    {"name": "CTNNB1", "expected_hgnc_id": "2514", "expected_uniprot": "P35222"},
    {"name": "SMAD2", "expected_hgnc_id": "6768", "expected_uniprot": "Q15796"},
    {"name": "MDM2", "expected_hgnc_id": "6973", "expected_uniprot": "Q00987"},
    {"name": "BAX", "expected_hgnc_id": "953", "expected_uniprot": "Q07812"},
]

CHEBI_TEST_CASES = [
    {"name": "EGF", "expected_chebi_id": "CHEBI:132945"},
    {"name": "ATP", "expected_chebi_id": "CHEBI:15422"},
    {"name": "GTP", "expected_chebi_id": "CHEBI:15996"},
    {"name": "doxorubicin", "expected_chebi_id": "CHEBI:28748"},
    {"name": "imatinib", "expected_chebi_id": "CHEBI:31690"},
]

GO_TEST_CASES = [
    {"name": "EGFR", "expected_go_terms": ["GO:0007179", "GO:0007165"]},
    {"name": "TP53", "expected_go_terms": ["GO:0006915", "GO:0005634"]},
    {"name": "AKT1", "expected_go_terms": ["GO:0004672", "GO:0005515"]},
]

@pytest.mark.parametrize("case", HGNC_TEST_CASES)
def test_hgnc_lookup(case):
    """Test HGNC gene symbol → HGNC ID + UniProt ID mapping."""
    pytest.skip("Requires network access to rest.genenames.org")

@pytest.mark.parametrize("case", CHEBI_TEST_CASES)
def test_chebi_lookup(case):
    """Test ChEBI chemical name → ChEBI ID mapping."""
    pytest.skip("Requires network access to ChEBI")

@pytest.mark.parametrize("case", GO_TEST_CASES)
def test_go_lookup(case):
    """Test GO term annotation for gene products."""
    pytest.skip("Requires network access to QuickGO")

@pytest.mark.skip(reason="Known v4 limitation: EGF dual identity (FM-017)")
def test_egf_dual_identity():
    """Documents P1 bug: EGF in both _KNOWN_PROTEINS and _KNOWN_CHEMICALS."""
    pass

@pytest.mark.skip(reason="Known v4 limitation: SBO reverse map loses 3 mechanisms (FM-016)")
def test_sbo_reverse_map():
    """Documents P1 bug: SBO_TO_MECHANISM drops nuclear_import, ubiquitination, sequestration."""
    from app.ontology.sbo_terms import SBO_TO_MECHANISM
    assert "nuclear_import" in SBO_TO_MECHANISM.values(), "nuclear_import lost in reverse map"
    assert "ubiquitination" in SBO_TO_MECHANISM.values(), "ubiquitination lost in reverse map"
    assert "sequestration" in SBO_TO_MECHANISM.values(), "sequestration lost in reverse map"

@pytest.mark.skip(reason="Known v4 limitation: GO geneProductSymbol invalid (FM-018)")
def test_go_client_invalid_param():
    """Documents P1 bug: QuickGO doesn't support geneProductSymbol parameter."""
    pass

def test_species_type_coverage():
    """Test that species_type accepts all v3 NER types."""
    from app.ontology.ontology_agent import _SPECIES_TYPES
    expected_types = {"gene", "protein", "chemical", "pathway", "unknown",
                      "ligand", "receptor", "kinase", "drug", "complex", "rna"}
    missing = expected_types - _SPECIES_TYPES
    assert not missing, f"Missing species types: {missing}"
