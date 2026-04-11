"""
vardiag — Variance Alignment Diagnostic

Public API for:
- diagnose
- diagnose_cv
- scan

The package provides a lightweight, label-aware, model-free diagnostic for
assessing whether variance-based feature filtering is likely safe or harmful
before model training.
"""

from importlib.metadata import PackageNotFoundError, version

from .core import DiagnosticResult, ScanReport, diagnose, diagnose_cv, scan
from .metrics import (
    OverlapRow,
    alpha_prime,
    bh_fdr,
    classify_zone,
    compute_overlap_curve,
    decoupling_index,
    eta2_features,
    eta_enrichment,
    f_di,
    pca_alignment,
    rank_features,
    vsa_mannwhitney,
)

try:
    __version__ = version("vardiag")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "Mirza Shoaib and collaborators"

from . import data  # noqa: E402

__all__ = [
    "diagnose",
    "diagnose_cv",
    "scan",
    "DiagnosticResult",
    "ScanReport",
    "eta2_features",
    "eta_enrichment",
    "vsa_mannwhitney",
    "alpha_prime",
    "pca_alignment",
    "f_di",
    "classify_zone",
    "decoupling_index",
    "compute_overlap_curve",
    "rank_features",
    "bh_fdr",
    "OverlapRow",
]
