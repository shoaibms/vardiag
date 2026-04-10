# Changelog

All notable changes to **vardiag** will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-04-10

### Added

**Core package**
- `vardiag.metrics` — all mathematical primitives operating on NumPy arrays with no IO dependencies:
  - `eta2_features()` — per-feature variance decomposition (V_total, V_between, η²)
  - `eta_enrichment()` — signal enrichment ratio η_ES
  - `vsa_mannwhitney()` — Variance-Signal Alignment via Mann-Whitney U
  - `alpha_prime()` — Spearman(V_total, η²), K-free monotone alignment
  - `pca_alignment()` — PCLA and SAS multivariate alignment metrics
  - `f_di()` — supervision-free Decoupling Index analogue
  - `classify_zone()` — GREEN / RED / YELLOW zone assignment with optional CI intervals
  - `decoupling_index()`, `compute_overlap_curve()` — full DI curve computation
  - `bh_fdr()` — Benjamini-Hochberg FDR correction

**Public API**
- `diagnose(X, y)` — full VAD in under one second, returns `DiagnosticResult`
- `diagnose_cv(X, y, cv_folds)` — leakage-free CV version, fold-by-fold aggregation
- `scan(X, y, shap_importance)` — full DI curve + hidden biomarker analysis, returns `ScanReport`

**Validation**
- `vardiag.validation` — strict, early input validation for all public functions with actionable error messages and `UserWarning` for soft issues (NaN columns, imbalanced classes, constant features, duplicate feature names)

**CLI**
- `vardiag run` — run VAD from the command line on `.npy`, `.csv`, `.tsv`, or `.txt` files
- `vardiag info` — check installed dependencies and version

**Bundled data**
- `vardiag.data` — four synthetic views calibrated to manuscript cohorts:
  - `brca_methylation` (MLOmics BRCA, 312 × 11,189, RED_HARMFUL)
  - `ibd_mgx` (IBDMDB, 155 × 368, GREEN_SAFE)
  - `ccle_mrna` (CCLE, 470 × 2,000, GREEN_SAFE)
  - `gbm_methylation` (TCGA-GBM, 136 × 8,000, RED_HARMFUL)

**Tutorial**
- `tutorial/01_full_tutorial.py` — 10-step walkthrough reproducing all key manuscript findings

**Tests**
- 107 tests across 11 test classes, 93% coverage
- Coverage gate enforced at 90% on core modules
- CI matrix: Python 3.9 – 3.13

**Package infrastructure**
- `pyproject.toml` with `[full]`, `[plot]`, and `[dev]` extras
- GitHub Actions CI workflow
- `CITATION.cff`
- `CONTRIBUTING.md`
- `LICENSE` (MIT)
- Pure-NumPy fallbacks for all metrics (scipy and scikit-learn optional)

### Notes

- Classification tasks only. Regression support planned for v0.2.
- Real dataset loading helpers planned for v0.2 (currently documented in tutorial Step 10).

---

## Planned

### [0.2.0]
- `diagnose_cv` uncertainty outputs (per-fold SD, CI intervals)
- Real dataset download helpers for all four manuscript cohorts
- Expanded SHAP input types (ndarray, pd.Series)
- Additional TSV and pandas-header CLI tests
- Optional `to_json()` / `to_frame()` on result objects
