"""
vardiag — Variance Alignment Diagnostic for Multi-Omics Feature Selection
==========================================================================

A lightweight, model-free, label-aware diagnostic framework that identifies
whether variance-based feature filtering is safe or harmful before model
training — in under 1 second.

Quick start
-----------
>>> from vardiag import diagnose
>>> result = diagnose(X_train, y_train, k_pct=10)
>>> print(result.zone)          # GREEN_SAFE | RED_HARMFUL | YELLOW_INCONCLUSIVE
>>> print(result.summary())

Full DI analysis (requires SHAP):
>>> from vardiag import scan
>>> report = scan(X_train, y_train, shap_importance=shap_dict)
>>> print(report.summary())

Cross-validation safe version:
>>> from vardiag import diagnose_cv
>>> result = diagnose_cv(X, y, cv_folds=train_indices_per_fold)

Reference
---------
"When Variance Misleads: A Diagnostic Framework for Feature Selection
in Multi-Omics Data"  (2026)
"""

from .core import (
    diagnose,
    diagnose_cv,
    scan,
    DiagnosticResult,
    ScanReport,
)

from .metrics import (
    # VAD primitives (operate on numpy arrays)
    eta2_features,
    eta_enrichment,
    vsa_mannwhitney,
    alpha_prime,
    pca_alignment,
    f_di,
    classify_zone,
    # DI primitives
    decoupling_index,
    compute_overlap_curve,
    rank_features,
    bh_fdr,
    OverlapRow,
)

__version__ = "0.1.0"
__author__  = "VAD authors"

from . import data  # noqa: E402  — bundled synthetic manuscript views

__all__ = [
    # High-level API
    "diagnose",
    "diagnose_cv",
    "scan",
    "DiagnosticResult",
    "ScanReport",
    # Low-level metrics
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
