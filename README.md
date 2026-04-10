# vardiag — Variance Alignment Diagnostic

[![CI](https://github.com/shoaibms/vardiag/actions/workflows/ci.yml/badge.svg)](https://github.com/shoaibms/vardiag/actions)
[![PyPI](https://img.shields.io/pypi/v/vardiag)](https://pypi.org/project/vardiag/)
[![Python](https://img.shields.io/pypi/pyversions/vardiag)](https://pypi.org/project/vardiag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
**vardiag** tells you — typically in under one second, without training any model — whether
variance-based feature filtering is safe or harmful for your omics dataset.

```python
from vardiag import diagnose
result = diagnose(X_train, y_train, k_pct=10)
print(result.zone)      # GREEN_SAFE | RED_HARMFUL | YELLOW_INCONCLUSIVE
print(result.summary())
```

---

## The Problem

Variance filtering — retaining the most variable features — is the *de facto*
default in single-cell toolkits (Seurat, scanpy), DNA methylation pipelines, and
multi-omics integration workflows. Its core assumption is almost never tested:

> *High-variance features enrich for biological signal.*

Across 14 omics views from four public cohorts, this assumption ranges from
**strongly correct (ρ = 0.93)** to **completely wrong (ρ = −0.32)**. In the worst
case (BRCA methylation), variance filtering degrades prediction accuracy by
**16.2 percentage points** relative to a random baseline, while systematically
excluding prognostically validated biomarkers including *MIA* and *CHI3L1*.

**vardiag** formalises and detects this problem before you commit to any model.

---

## Installation

```bash
# Minimal — NumPy only
pip install vardiag

# Recommended — adds scipy and scikit-learn for the full metric suite
pip install "vardiag[full]"

# Development install from source
git clone https://github.com/shoaibms/vardiag
cd vardiag
pip install -e ".[dev]"
```

Check your environment after installing:

```bash
vardiag info
```

---

## Dependencies

| Package | Status | Role |
|---|---|---|
| `numpy >= 1.23` | **Required** | All computations |
| `scipy >= 1.9` | Recommended | Exact Mann-Whitney U and Spearman ρ |
| `scikit-learn >= 1.2` | Recommended | PCA for PCLA and SAS metrics |
| `pandas >= 1.5` | Optional | Named-column CSV loading in CLI |
| `matplotlib >= 3.6` | Optional | Plotting |

Without scipy and scikit-learn, vardiag falls back to pure-NumPy
approximations for all metrics. Install the full suite with:

```bash
pip install "vardiag[full]"
```

---

## Quick Start

```python
import numpy as np
from vardiag import diagnose

# Load your pre-normalised feature matrix and class labels
X = np.load("methylation_train.npy")   # shape: (n_samples, n_features)
y = np.load("labels_train.npy")        # shape: (n_samples,)

result = diagnose(X, y, k_pct=10)
print(result.summary())
```

**Example output:**

```
============================================================
  VAD Diagnostic Report  (K = 10%)
============================================================
  Dataset    : 312 samples × 11189 features  (4 classes)
  Computed in: 0.43 s

  Zone       : 🔴  RED_HARMFUL

  Metrics
    η_ES     : +0.285  (>1 = enriched for signal)
    VSA      : -0.045  (>0 = aligned)
    α'       : -0.230  (Spearman V vs η²)
    PCLA     : +0.024  (PCA-weighted η²)
    SAS      : -0.191  (spectral alignment)
    F-DI     : +1.045  (<1 = coupled)

  Decision rule
    ⚠️  Variance filtering likely harmful. Use importance-guided
    selection (e.g. SHAP) or include all features.
============================================================
```

---

## Zone Interpretation

| Zone | Condition | Meaning | Action |
|---|---|---|---|
| 🟢 **GREEN_SAFE** | η_ES > 1.05 and VSA > 0 | High-var features carry above-average signal | Proceed with variance filtering |
| 🔴 **RED_HARMFUL** | η_ES < 0.95 and VSA < 0 | High-var features depleted of signal | Use importance-guided selection (e.g. SHAP) |
| 🟡 **YELLOW_INCONCLUSIVE** | Mixed signals | Ambiguous alignment | Run a small pilot ablation |

---

## API

### `diagnose(X, y, k_pct=10)`

Run the VAD on a single (X, y) view. Returns a `DiagnosticResult`.

```python
result = diagnose(
    X,                     # ndarray (n_samples, n_features)
    y,                     # ndarray (n_samples,) class labels
    k_pct=10,              # feature budget as % of total (default 10)
    n_pca_components=30,   # for PCLA/SAS metrics (default 30)
    random_state=0,        # for PCA reproducibility (default 0)
    margin=0.05,           # conservative threshold margin (default 0.05)
)

result.zone          # str: 'GREEN_SAFE' | 'RED_HARMFUL' | 'YELLOW_INCONCLUSIVE'
result.eta_es        # float: signal enrichment ratio
result.vsa           # float: variance-signal alignment
result.alpha_prime   # float: Spearman(V_total, η²)
result.pcla          # float: PCA-weighted signal alignment
result.sas           # float: spectral alignment score
result.f_di          # float: supervision-free decoupling index
result.elapsed_s     # float: wall-clock time in seconds
result.summary()     # formatted multi-line string
result.to_dict()     # flat dict — use as a pandas DataFrame row
```

### `diagnose_cv(X, y, cv_folds, k_pct=10)`

Leakage-free version for use inside ML pipelines. Computes VAD
fold-by-fold on training data only, then aggregates.

```python
from sklearn.model_selection import StratifiedKFold
from vardiag import diagnose_cv

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
train_folds = [tr for tr, _ in skf.split(X, y)]

result = diagnose_cv(X, y, cv_folds=train_folds, k_pct=10)
print(result.summary())
```

### `scan(X, y, shap_importance, feature_names=None, k_pcts=[1,5,10,20])`

Full DI curve and hidden biomarker analysis. Requires pre-computed SHAP scores.

```python
import shap
from vardiag import scan

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_train)
shap_imp  = dict(zip(feature_names, np.abs(shap_vals).mean(axis=0).tolist()))

report = scan(
    X_train, y_train,
    shap_importance=shap_imp,
    feature_names=feature_names,
    k_pcts=[1, 5, 10, 20],
)
print(report.summary())
# Prints: DI curve table, hidden biomarker fraction, gene-level Jaccard
```

### Command-line interface

```bash
# Run on CSV files
vardiag run --X features.csv --y labels.csv --k 10

# Save result as JSON
vardiag run --X features.npy --y labels.npy --k 10 --out result.json

# With a feature names file (one name per line)
vardiag run --X features.csv --y labels.csv --features gene_names.txt

# Check installed dependencies
vardiag info
```

Supported file formats: `.npy`, `.csv`, `.tsv`, `.txt`

---

## Metrics Reference

| Symbol | Name | Formula | Interpretation |
|---|---|---|---|
| **η_ES** | Signal enrichment ratio | mean(η² in TopVar) / mean(η² all) | > 1: signal concentrated in high-var features |
| **VSA** | Variance-Signal Alignment | AUROC(η²; TopVar vs Rest) − 0.5 | > 0: high-var features rank higher on signal |
| **α'** | Monotone alignment | Spearman(V_total, η²) | > 0: variance and signal co-vary |
| **PCLA** | PCA label alignment | Σ(λ_k / Σλ) · η²(PC_k) | Higher: PCA variance structure tracks labels |
| **SAS** | Spectral alignment | Spearman(λ_k, η²(PC_k)) | > 0: higher-eigenvalue PCs carry more signal |
| **F-DI** | Supervision-free DI | 1 − J̃(TopVar, Topη²) | < 1: coupled; > 1: anti-aligned |
| **DI** | Decoupling Index | 1 − J̃(TopVar, TopSHAP) | Requires SHAP. < 1: coupled; > 1: anti-aligned |

The signal fraction η² = V_between / V_total (ANOVA-style decomposition).
V_between is the between-class variance; V_within is the within-class variance.

---

## Input Requirements

| Requirement | Details |
|---|---|
| `X` shape | 2-D, `(n_samples, n_features)` |
| `X` dtype | Any numeric type; internally coerced to `float32` |
| `y` shape | 1-D, `(n_samples,)` |
| Minimum samples | 4 |
| Minimum features | 2 |
| Minimum classes | 2 |
| Task type | Classification only (v0.1) |
| NaN values | Handled by mean-imputation with a `UserWarning` |
| Constant features | Handled gracefully with a `UserWarning` |
| Normalisation | Apply your own normalisation before calling `diagnose` |

For leakage-free use in ML workflows, always pass training-split data
or use `diagnose_cv`.

---

## Tutorial: Reproducing the Manuscript Results

A complete 10-step tutorial is included that reproduces all key findings
on bundled synthetic views calibrated to the real cohorts.

```bash
python tutorial/01_full_tutorial.py
```

Expected runtime: under 30 seconds.

| Step | Content |
|---|---|
| 1 | Installation and dependency check |
| 2 | The core problem — when variance filtering fails |
| 3 | The Decoupling Index (DI) — formalising misalignment |
| 4 | Signal fraction η² — the mechanistic explanation |
| 5 | The VAD diagnostic — GREEN / RED / YELLOW zones |
| 6 | Full sweep across K% values (manuscript Figure 1 equivalent) |
| 7 | Hidden biomarkers — features excluded by variance filtering |
| 8 | Cross-view summary table (manuscript Table 1 equivalent) |
| 9 | Leakage-free CV integration |
| 10 | Instructions for loading the real manuscript datasets |

**Reproduced zones (Step 8):**

```
View                  Zone           η_ES    VSA   F-DI   ms DI   ms ρ
brca_methylation      RED_HARMFUL    0.29  -0.04   1.05    1.03  -0.32
ibd_mgx               GREEN_SAFE     4.88   0.50   0.11    0.70   0.76
ccle_mrna             GREEN_SAFE     5.09   0.49   0.27    0.92   0.68
gbm_methylation       RED_HARMFUL    0.94  -0.01   0.25    1.00   0.02
```

---

## Bundled Synthetic Views

```python
from vardiag.data import load_all_views, load_view, describe_views

describe_views()          # print overview table of all four views

view = load_view("brca_methylation")
print(view.X.shape)       # (312, 11189)
print(view.expected_zone) # RED_HARMFUL
print(view.manuscript_di) # 1.03
print(view.description)   # plain-text explanation of the view design

views = load_all_views()  # dict of all four views
```

| View name | Dataset | Modality | n | p | Zone |
|---|---|---|---|---|---|
| `brca_methylation` | MLOmics BRCA | DNA methylation | 312 | 11,189 | RED |
| `ibd_mgx` | IBDMDB | Metagenomics | 155 | 368 | GREEN |
| `ccle_mrna` | CCLE | mRNA expression | 470 | 2,000 | GREEN |
| `gbm_methylation` | TCGA-GBM | DNA methylation | 136 | 8,000 | RED |

---

## Real Datasets

The manuscript analyses four public cohorts. Loading instructions:

**MLOmics BRCA** (methylation, mRNA, miRNA, CNV)
Source: [Yang et al. 2025, Sci. Data 12:913](https://github.com/shoaibms/var-pre)

```python
X = np.load("mlomics_brca_methylation_X.npy")  # (312, 11189)
y = np.load("mlomics_brca_y.npy")              # (312,)
result = diagnose(X, y, k_pct=10)              # Expected: RED_HARMFUL
```

**IBDMDB** (MGX metagenomics, MBX metabolomics)
Source: [Lloyd-Price et al. 2019, Nature 569:655](https://ibdmdb.org)

```python
result = diagnose(X_mgx, y_ibd, k_pct=10)     # Expected: GREEN_SAFE
```

**CCLE** (mRNA, CNV, proteomics)
Source: [depmap.org/portal/download/](https://depmap.org/portal/download/)

**TCGA-GBM** (mRNA, methylation, CNV)
Source: [xenabrowser.net](https://xenabrowser.net)

Full preprocessing pipeline: [github.com/shoaibms/var-pre/code/01_bundles/](https://github.com/shoaibms/var-pre)

---

## Benchmarking Results

| Result | Value |
|---|---|
| Variance–importance alignment range | ρ = −0.32 to +0.93 across 14 views |
| Max accuracy loss from variance filtering | −16.2 pp vs random baseline |
| SHAP outperforms variance filtering | 27/28 model–view combinations |
| Hidden biomarker fraction | Up to 25.9% of features (mean 17.9%) |
| VAD computation time | < 1 second per view |

---

## Running the Tests

```bash
# Full test suite with coverage
pytest vardiag/tests/ -v --cov=vardiag --cov-report=term-missing

# Quick run
pytest vardiag/tests/ -q
```

Expected: **105 tests passing**, 93% coverage, 0 RuntimeWarnings.

---

## Project Structure

```
vardiag/
├── vardiag/
│   ├── __init__.py       public API surface
│   ├── metrics.py        all mathematical primitives (IO-free)
│   ├── core.py           diagnose / diagnose_cv / scan
│   ├── validation.py     strict input validation
│   ├── data.py           bundled synthetic manuscript views
│   ├── cli.py            command-line interface
│   └── tests/
│       └── test_vardiag.py   105 tests
├── tutorial/
│   └── 01_full_tutorial.py   10-step manuscript walkthrough
├── examples/
│   └── quickstart.py         6 quick examples
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## Citation

```bibtex
@article{vardiag2026,
  title   = {When Variance Misleads: A Diagnostic Framework for Feature
             Selection in Multi-Omics Data},
  year    = {2026},
  journal = {(under review)},
  url     = {https://github.com/shoaibms/var-pre},
}
```

---

## Links

| | |
|---|---|
| Package | https://github.com/shoaibms/vardiag |
| Manuscript and analysis code | https://github.com/shoaibms/var-pre |
| Bug reports | https://github.com/shoaibms/vardiag/issues |

---

## License

MIT © 2026
