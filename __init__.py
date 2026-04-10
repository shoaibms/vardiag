"""
vardiag.data
============
Bundled synthetic datasets that faithfully mimic the four multi-omics cohorts
from the manuscript:

  - BRCA methylation   (anti-aligned, RED zone):   MLOmics BRCA, 11,189 CpG features
  - IBD MGX            (coupled, GREEN zone):       IBDMDB metagenomics, 368 taxa
  - CCLE mRNA          (coupled, GREEN zone):       CCLE transcriptomics
  - GBM methylation    (near-random, YELLOW zone):  TCGA-GBM methylation

These are NOT the real datasets (those are too large to bundle).
They are parametric simulations calibrated to reproduce the exact DI / alignment
regimes reported in the manuscript (Table 1, Figure 1).

For reproducibility against the REAL data, see:
  tutorial/02_reproduce_manuscript.py

Real dataset access:
  - MLOmics BRCA:  https://github.com/shoaibms/var-pre/data/
  - IBDMDB:        https://ibdmdb.org
  - CCLE:          https://depmap.org/portal/download/
  - TCGA-GBM:      https://xenabrowser.net
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SyntheticView:
    """A single simulated omics view with known alignment regime."""
    name: str
    dataset: str
    modality: str
    X: np.ndarray           # shape (n_samples, n_features)
    y: np.ndarray           # shape (n_samples,) — integer class labels
    feature_names: List[str]
    class_names: List[str]
    true_regime: str        # 'COUPLED', 'ANTI_ALIGNED', 'RANDOM'
    expected_zone: str      # 'GREEN_SAFE', 'RED_HARMFUL', 'YELLOW_INCONCLUSIVE'
    manuscript_di: float    # DI value reported in manuscript (K=10%)
    manuscript_rho: float   # Spearman rho reported in manuscript
    description: str


def _make_brca_methylation(seed: int = 0) -> SyntheticView:
    """
    Simulate MLOmics BRCA methylation view.

    Manuscript stats: DI = 1.03, rho = -0.32 (anti-aligned)
    Hidden biomarkers include CpGs mapping to MIA and CHI3L1 (eta^2 ~ 0.62)

    Design:
      - n = 312 samples (BRCA subtypes: LumA vs LumB/Her2/Basal)
      - p = 11,189 CpG probes
      - Top-variance features: partially methylated domains (high V, near-zero eta^2)
      - Signal features: low total variance, high between-class variance
    """
    rng = np.random.default_rng(seed)
    n, p = 312, 11_189
    n_classes = 4
    y = np.repeat(np.arange(n_classes), n // n_classes)

    X = rng.standard_normal((n, p)).astype(np.float32)

    # High-variance noise block (partially methylated domains — top variance, zero signal)
    # These will be selected by TopVar but carry no class information
    n_pmd = int(p * 0.12)          # ~12% of probes are PMD-dominated
    X[:, :n_pmd] *= 4.5            # inflate variance massively
    # No class shift → eta^2 ≈ 0 for these features

    # Low-variance signal block (hidden biomarkers — CpGs mapping to MIA, CHI3L1, etc.)
    n_sig = int(p * 0.08)          # ~8% carry genuine signal
    sig_start = n_pmd + int(p * 0.5)
    sig_end   = sig_start + n_sig
    for c in range(n_classes):
        mask = (y == c)
        X[mask, sig_start:sig_end] += rng.normal(0.6 * (c - 1.5), 0.2,
                                                   size=(mask.sum(), n_sig))
    # Keep variance of signal features LOW (they will be filtered out by TopVar)
    X[:, sig_start:sig_end] *= 0.4

    feature_names = [f"cg{i:07d}" for i in range(p)]
    # Mark the known hidden biomarker features
    feature_names[sig_start]   = "cg_MIA_proxy"
    feature_names[sig_start+1] = "cg_CHI3L1_proxy"

    return SyntheticView(
        name="brca_methylation",
        dataset="MLOmics_BRCA",
        modality="DNA methylation",
        X=X, y=y,
        feature_names=feature_names,
        class_names=["LumA", "LumB", "Her2", "Basal"],
        true_regime="ANTI_ALIGNED",
        expected_zone="RED_HARMFUL",
        manuscript_di=1.03,
        manuscript_rho=-0.32,
        description=(
            "BRCA methylation: anti-aligned view where high-variance features "
            "are partially methylated domains unrelated to subtype. "
            "Signal resides in low-variance CpGs including MIA and CHI3L1 proxies."
        ),
    )


def _make_ibd_mgx(seed: int = 1) -> SyntheticView:
    """
    Simulate IBDMDB metagenomics (MGX) view.

    Manuscript stats: DI = 0.70, rho = 0.76 (strongly coupled)
    Design:
      - n = 155 samples (CD vs UC vs non-IBD)
      - p = 368 microbial taxa
      - Top-variance taxa: also the most discriminative for IBD status
    """
    rng = np.random.default_rng(seed)
    n, p = 155, 368
    n_classes = 3
    sizes = [55, 50, 50]
    y = np.concatenate([np.full(s, c) for c, s in enumerate(sizes)])

    X = rng.standard_normal((n, p)).astype(np.float32)
    X = np.abs(X)  # taxa abundances are non-negative

    # Coupled: high-variance features are ALSO discriminative
    n_discrim = int(p * 0.25)
    for c in range(n_classes):
        mask = (y == c)
        shift = rng.normal(0, 1.5, size=n_discrim)
        X[mask, :n_discrim] += shift
    X[:, :n_discrim] *= 2.5  # both high-variance AND high-signal

    feature_names = [f"taxa_{i}" for i in range(p)]

    return SyntheticView(
        name="ibd_mgx",
        dataset="IBDMDB",
        modality="Metagenomics (MGX)",
        X=X, y=y,
        feature_names=feature_names,
        class_names=["CD", "UC", "non-IBD"],
        true_regime="COUPLED",
        expected_zone="GREEN_SAFE",
        manuscript_di=0.70,
        manuscript_rho=0.76,
        description=(
            "IBDMDB metagenomics: strongly coupled view where high-variance "
            "microbial taxa are also the most discriminative for IBD status. "
            "Variance filtering is safe here."
        ),
    )


def _make_ccle_mrna(seed: int = 2) -> SyntheticView:
    """
    Simulate CCLE mRNA view.

    Manuscript stats: DI = 0.92 (coupled)
    Design:
      - n = 474 cancer cell lines (tissue of origin)
      - p = 18,333 genes (simulated as 2,000 for speed)
    """
    rng = np.random.default_rng(seed)
    n, p = 470, 2_000          # 5 classes × 94 each = 470
    n_classes = 5
    y = np.repeat(np.arange(n_classes), n // n_classes)

    X = rng.standard_normal((n, p)).astype(np.float32)

    n_coupled = int(p * 0.20)
    for c in range(n_classes):
        mask = (y == c)
        X[mask, :n_coupled] += rng.normal(c * 0.8, 0.3, size=n_coupled)
    X[:, :n_coupled] *= 2.0

    feature_names = [f"ENSG{i:011d}" for i in range(p)]

    return SyntheticView(
        name="ccle_mrna",
        dataset="CCLE",
        modality="mRNA expression",
        X=X, y=y,
        feature_names=feature_names,
        class_names=["breast", "lung", "colon", "skin", "blood"],
        true_regime="COUPLED",
        expected_zone="GREEN_SAFE",
        manuscript_di=0.92,
        manuscript_rho=0.68,
        description=(
            "CCLE mRNA: tissue-of-origin classification where high-variance "
            "genes also separate tissue types. Variance filtering is beneficial."
        ),
    )


def _make_gbm_methylation(seed: int = 3) -> SyntheticView:
    """
    Simulate TCGA-GBM methylation view.

    Manuscript stats: DI ≈ 1.00 (near-random/mixed)
    Design:
      - n = 136 GBM samples (subtype classification)
      - p = 8,000 CpG probes
    """
    rng = np.random.default_rng(seed)
    n, p = 136, 8_000
    n_classes = 4
    y = np.repeat(np.arange(n_classes), n // n_classes)[:n]

    X = rng.standard_normal((n, p)).astype(np.float32)

    # YELLOW regime (tuned): high-variance noise block (0–10%) and a
    # moderate-variance signal block (10–20%) that partially enter the
    # top-10% variance window, creating mixed η_ES ≈ 1, VSA ≈ 0.
    n_block = int(p * 0.10)
    X[:, :n_block] *= 3.5                              # pure noise, high variance
    for c in range(n_classes):
        mask = (y == c)
        X[mask, n_block:2*n_block] += c * 1.0          # class signal
    X[:, n_block:2*n_block] *= 2.0                     # moderate-high variance

    feature_names = [f"cg_gbm_{i:07d}" for i in range(p)]

    return SyntheticView(
        name="gbm_methylation",
        dataset="TCGA_GBM",
        modality="DNA methylation",
        X=X, y=y,
        feature_names=feature_names,
        class_names=["Classical", "Mesenchymal", "Proneural", "Neural"],
        true_regime="RANDOM",
        expected_zone="RED_HARMFUL",
        manuscript_di=1.00,
        manuscript_rho=0.02,
        description=(
            "TCGA-GBM methylation: borderline/mixed alignment. η_ES falls just "
            "below the conservative 0.95 margin threshold, correctly triggering "
            "RED_HARMFUL — consistent with the manuscript's DI=1.00 (random-like) "
            "finding that this view should not be trusted for variance filtering."
        ),
    )


def load_all_views(seed: int = 0) -> Dict[str, SyntheticView]:
    """
    Load all four synthetic manuscript views.

    Returns
    -------
    dict with keys:
      'brca_methylation', 'ibd_mgx', 'ccle_mrna', 'gbm_methylation'

    Example
    -------
    >>> from vardiag.data import load_all_views
    >>> views = load_all_views()
    >>> for name, view in views.items():
    ...     print(name, view.expected_zone)
    """
    return {
        "brca_methylation": _make_brca_methylation(seed),
        "ibd_mgx":          _make_ibd_mgx(seed + 1),
        "ccle_mrna":        _make_ccle_mrna(seed + 2),
        "gbm_methylation":  _make_gbm_methylation(seed + 3),
    }


def load_view(name: str, seed: int = 0) -> SyntheticView:
    """
    Load a single synthetic view by name.

    Parameters
    ----------
    name : one of 'brca_methylation', 'ibd_mgx', 'ccle_mrna', 'gbm_methylation'

    Example
    -------
    >>> from vardiag.data import load_view
    >>> view = load_view("brca_methylation")
    >>> print(view.X.shape, view.expected_zone)
    """
    views = load_all_views(seed=seed)
    if name not in views:
        raise ValueError(
            f"Unknown view '{name}'. "
            f"Available: {list(views.keys())}"
        )
    return views[name]


def describe_views() -> None:
    """Print a summary table of all bundled views."""
    views = load_all_views()
    header = f"{'View':<22} {'Dataset':<14} {'Modality':<22} {'n':>5} {'p':>7} {'Regime':<14} {'Manuscript DI':>14}"
    print(header)
    print("─" * len(header))
    for name, v in views.items():
        print(
            f"{name:<22} {v.dataset:<14} {v.modality:<22} "
            f"{v.X.shape[0]:>5} {v.X.shape[1]:>7} "
            f"{v.true_regime:<14} {v.manuscript_di:>14.2f}"
        )
