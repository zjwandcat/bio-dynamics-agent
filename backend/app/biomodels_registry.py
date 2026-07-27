"""Verified BioModels references for the ten supported pathways.

The model metadata below was checked against the EBI BioModels JSON API on
2026-07-21.  Runtime code should use this module instead of copying model IDs
into pathway initializers and specialists.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BioModelsReference:
    model_id: str
    name: str
    pmid: str


BIOMODELS_REGISTRY: dict[str, BioModelsReference] = {
    "EGFR_RTK": BioModelsReference(
        "BIOMD0000000048",
        "Kholodenko1999 - EGFR signaling",
        "10514507",
    ),
    "MAPK_ERK": BioModelsReference(
        "BIOMD0000000010",
        "Kholodenko2000 - Ultrasensitivity and negative feedback in MAPK",
        "10712587",
    ),
    "PI3K_AKT_mTOR": BioModelsReference(
        "BIOMD0000000262",
        "Fujita2010 - Akt signalling under EGF stimulation",
        "20664065",
    ),
    "p53": BioModelsReference(
        "BIOMD0000000252",
        "Hunziker2010 - p53 stress-specific response",
        "20624280",
    ),
    "APOPTOSIS": BioModelsReference(
        "BIOMD0000000102",
        "Legewie2006 - Apoptosis wild type",
        "16978046",
    ),
    "CELL_CYCLE": BioModelsReference(
        "BIOMD0000000056",
        "Chen2004 - Cell cycle regulation",
        "15169868",
    ),
    "JAK_STAT": BioModelsReference(
        "BIOMD0000000347",
        "Bachmann2011 - JAK2/STAT5 feedback control",
        "21772264",
    ),
    "NF_KB": BioModelsReference(
        "BIOMD0000000140",
        "Hoffmann2002 - Wild-type IkB/NF-kB signalling",
        "12424381",
    ),
    "WNT": BioModelsReference(
        "BIOMD0000000658",
        "Lee2003 - Roles of APC and Axin in Wnt pathway",
        "14551908",
    ),
    "TGF_BETA": BioModelsReference(
        "BIOMD0000000342",
        "Zi2011 - TGF-beta pathway",
        "21613981",
    ),
}


_PATHWAY_ALIASES: dict[str, str] = {
    "p53_signaling": "p53",
    "Apoptosis": "APOPTOSIS",
    "Cell_Cycle": "CELL_CYCLE",
    "NF_kB": "NF_KB",
    "Wnt": "WNT",
    "TGF_beta": "TGF_BETA",
    "EGFR": "EGFR_RTK",
    "MAPK": "MAPK_ERK",
    "PI3K": "PI3K_AKT_mTOR",
    "CellCycle": "CELL_CYCLE",
    "JAKSTAT": "JAK_STAT",
    "NFKB": "NF_KB",
    "TGFB": "TGF_BETA",
}


def canonical_pathway_id(pathway: str) -> str:
    """Return the registry key for a runtime, fixture, or frontend alias."""
    key = _PATHWAY_ALIASES.get(pathway, pathway)
    if key not in BIOMODELS_REGISTRY:
        raise KeyError(f"No verified BioModels reference for pathway {pathway!r}")
    return key


def get_biomodels_reference(pathway: str) -> BioModelsReference:
    return BIOMODELS_REGISTRY[canonical_pathway_id(pathway)]


def get_biomodels_id(pathway: str) -> str:
    return get_biomodels_reference(pathway).model_id


__all__ = [
    "BIOMODELS_REGISTRY",
    "BioModelsReference",
    "canonical_pathway_id",
    "get_biomodels_id",
    "get_biomodels_reference",
]
