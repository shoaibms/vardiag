# vardiag — Variance Alignment Diagnostic

[![CI](https://github.com/shoaibms/vardiag/actions/workflows/ci.yml/badge.svg)](https://github.com/shoaibms/vardiag/actions)
[![PyPI](https://img.shields.io/pypi/v/vardiag)](https://pypi.org/project/vardiag/)
[![Python](https://img.shields.io/pypi/pyversions/vardiag)](https://pypi.org/project/vardiag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/shoaibms/vardiag/blob/main/LICENSE)

**vardiag** is a lightweight Python package for the Variance Alignment Diagnostic
(VAD): a label-aware, model-free check for whether variance-based feature filtering
is likely safe, harmful, or inconclusive for a given labelled high-dimensional dataset.

```python
from vardiag import diagnose
from vardiag.data import load_view

view = load_view("brca_methylation")
result = diagnose(view.X, view.y, k_pct=10)

print(result.zone)      # RED_HARMFUL
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
| `numpy >= 1.23` | **Required** | Core computations |
| `scipy >= 1.9` | Recommended | Exact Mann–Whitney U and Spearman ρ |
| `scikit-learn >= 1.2` | Recommended | PCA for PCLA and SAS |
| `pandas >= 1.5` | Optional | CSV/TSV loading with headers in the CLI |
| `matplotlib >= 3.6` | Optional | Plotting examples |

Without scipy and scikit-learn, vardiag falls back to pure-NumPy approximations
for all metrics. Install the full suite with:

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
  Dataset    : 312 samples x 11189 features  (4 classes)
  Computed in: 0.43 s

  Zone       : RED_HARMFUL

  Metrics
    eta_ES   : +0.285  (>1 = enriched for signal)
    VSA      : -0.045  (>0 = aligned)
    alpha'   : -0.230  (Spearman V vs eta2)
    PCLA     : +0.024  (PCA-weighted eta2)
    SAS      : -0.191  (spectral alignment)
    F-DI     : +1.045  (<1 = coupled)

  Decision rule
    Variance filtering likely harmful. Use importance-guided
    selection (e.g. SHAP) or include all features.
============================================================
```

---

## Zone Interpretation

The analytical decision anchors from the manuscript are:

- **η_ES = 1**: break-even point for signal enrichment in the high-variance tail
- **VSA = 0**: no variance–signal alignment

By default, the package applies a **conservative margin** of `margin=0.05` around
these thresholds for zone assignment:

- GREEN requires `η_ES > 1.05` and `VSA > 0`
- RED requires `η_ES < 0.95` and `VSA < 0`
- everything else is YELLOW

| Zone | Condition (default `margin=0.05`) | Meaning | Action |
|---|---|---|---|
| **GREEN_SAFE** | `η_ES > 1.05` and `VSA > 0` | High-var features carry above-average class signal | Proceed with variance filtering |
| **RED_HARMFUL** | `η_ES < 0.95` and `VSA < 0` | High-var features depleted of signal | Use importance-guided selection (e.g. SHAP) |
| **YELLOW_INCONCLUSIVE** | Otherwise | Near-boundary or mixed evidence | Run a small pilot ablation |

The analytical threshold from the manuscript is η_ES = 1.0. The ±0.05 margin
creates the 1.05 / 0.95 boundaries shown above. Pass `margin=0.0` to use the
exact manuscript thresholds without conservatism.

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
result.alpha_prime   # float: Spearman(V_total, eta2)
result.pcla          # float: PCA-weighted signal alignment
result.sas           # float: spectral alignment score
result.f_di          # float: supervision-free decoupling index
result.elapsed_s     # float: wall-clock time in seconds
result.summary()     # formatted multi-line string
result.to_dict()     # flat dict — use as a pandas DataFrame row
```

### `diagnose_cv(X, y, cv_folds, k_pct=10)`

Leakage-safe wrapper for cross-validation workflows. The diagnostic is computed
on each training fold only, then aggregated across folds by taking the **mean**
of numeric metrics (`eta_es`, `vsa`, `alpha_prime`, `pcla`, `sas`, `f_di`,
`elapsed_s`). The reported `zone` is the zone obtained by applying the standard
zone rule to the aggregated mean metrics.

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

For multiclass problems, aggregate SHAP values across classes before passing
to `scan()`:

```python
import numpy as np
import shap
from vardiag import scan


def mean_abs_shap_importance(shap_values) -> np.ndarray:
    """
    Convert SHAP outputs to one feature-importance vector.

    Supports:
    - list[n_classes] of (n_samples, n_features)   — TreeExplainer multiclass
    - ndarray (n_classes, n_samples, n_features)   — stacked multiclass
    - ndarray (n_samples, n_features)              — binary / regression
    """
    if isinstance(shap_values, list):
        arr = np.stack(shap_values, axis=0)
        return np.abs(arr).mean(axis=(0, 1))
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return np.abs(arr).mean(axis=(0, 1))
    if arr.ndim == 2:
        return np.abs(arr).mean(axis=0)
    raise ValueError(f"Unsupported SHAP shape: {arr.shape}")


explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap_imp    = mean_abs_shap_importance(shap_values)

report = scan(
    X_train,
    y_train,
    shap_importance=dict(zip(feature_names, shap_imp.tolist())),
    feature_names=feature_names,
    k_pcts=[1, 5, 10, 20],
)
print(report.summary())
# Prints: VAD report, DI curve table, hidden biomarker fraction, Jaccard
```

A complete copy-pasteable example with a standalone demo is in
`examples/scan_multiclass_shap.py`.

### Auxiliary metrics

Only **η_ES** and **VSA** drive the default GREEN / RED / YELLOW zone rule.
The other metrics are complementary diagnostics, not zone drivers:

- **α'** (Spearman of V_total vs η²) checks monotone global agreement between
  variance ranking and signal ranking. Useful for confirming zone assignments
  on borderline YELLOW cases where η_ES and VSA point in the same direction
  but are close to the threshold.
- **PCLA** measures whether the leading PCA directions preferentially capture
  class-discriminative signal (eigenvalue-weighted η²).
- **SAS** checks whether larger-eigenvalue PCs are more label-aligned
  (Spearman of explained variance vs η²_PC).
- **F-DI** is a model-free companion to the SHAP-supervised DI. It substitutes
  η²-ranked features for SHAP-ranked features and requires no model training.
  Use `DI` (from `scan()`) when supervised importance is available; use `F-DI`
  as a lightweight pre-training signal.

### Command-line interface

```bash
# CSV
vardiag run --X features.csv --y labels.csv --k 10

# TSV + line-delimited text labels
vardiag run --X features.tsv --y labels.txt --k 10

# NumPy arrays, save JSON output
vardiag run --X features.npy --y labels.npy --k 10 --out result.json

# Check installed dependencies
vardiag info
```

Supported matrix formats: `.npy`, `.csv`, `.tsv`, whitespace-delimited `.txt`
Supported label formats: `.npy`, `.csv`, `.tsv`, one-label-per-line `.txt`

---

## Metrics Reference

| Symbol | Name | Formula | Interpretation |
|---|---|---|---|
| **η_ES** | Signal enrichment ratio | mean(η² in TopVar) / mean(η² all) | > 1: signal concentrated in high-var features |
| **VSA** | Variance-Signal Alignment | AUROC(η²; TopVar vs Rest) − 0.5 | > 0: high-var features rank higher on signal |
| **α'** | Monotone alignment | Spearman(V_total, η²) | > 0: variance and signal co-vary |
| **PCLA** | PCA label alignment | Σ(λ_k / Σλ) · η²(PC_k) | Higher: PCA variance structure tracks labels |
| **SAS** | Spectral alignment | Spearman(λ_k, η²(PC_k)) | > 0: higher-eigenvalue PCs carry more signal |
| **F-DI** | Supervision-free DI | 1 − J(TopVar, Topη²) | < 1: coupled; > 1: anti-aligned |
| **DI** | Decoupling Index | 1 − J(TopVar, TopSHAP) | Requires SHAP. < 1: coupled; > 1: anti-aligned |

The signal fraction η² = V_between / V_total (ANOVA-style decomposition).
V_between is the between-class variance; V_total is the total feature variance.

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

## Self-contained example using bundled views

```python
from vardiag import diagnose
from vardiag.data import load_view

for name in ["brca_methylation", "ibd_mgx", "ccle_mrna", "gbm_methylation"]:
    view = load_view(name)
    result = diagnose(view.X, view.y, k_pct=10)
    print(
        f"{name:18s}  zone={result.zone:20s}  "
        f"eta_es={result.eta_es:6.3f}  vsa={result.vsa:6.3f}"
    )
```

Run `python examples/quickstart.py` for the full six-example walkthrough.

---

## Tutorial: Demonstrating the Manuscript Alignment Regimes

A complete 10-step tutorial **demonstrates the key VAD alignment regimes** using
bundled synthetic views calibrated to the manuscript cohorts.

```bash
python tutorial/01_full_tutorial.py
```

Runtime depends on machine and installed extras.

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
View                  Zone             eta_ES   VSA    F-DI   ms DI   ms rho
brca_methylation      RED_HARMFUL       0.29  -0.04   1.05    1.03   -0.32
ibd_mgx               GREEN_SAFE        4.88   0.50   0.11    0.70    0.76
ccle_mrna             GREEN_SAFE        5.09   0.49   0.27    0.92    0.68
gbm_methylation       RED_HARMFUL       0.29  -0.05   1.04    1.00    0.02
```

> **Note on `gbm_methylation`:** The manuscript reports a DI of 1.00 for TCGA-GBM
> methylation, indicating chance-level overlap between variance-ranked and
> SHAP-ranked features — variance filtering provides a random sample of important
> features. The package's VAD consistently assigns RED_HARMFUL (η_ES ≈ 0.29):
> high-variance features carry only ~29% of the average class signal, making them
> substantially depleted of discriminative information. **DI and VAD are
> complementary, not contradictory**: DI says the selected features overlap randomly
> with SHAP-important features; VAD says those same high-variance features have low
> intrinsic class signal. Both consistently indicate that variance filtering is
> unreliable for this view. TCGA-GBM methylation is treated as a sensitivity
> analysis in the manuscript because the underlying subtype signal is weaker than
> in the primary cohorts.

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
| `gbm_methylation` | TCGA-GBM | DNA methylation | 136 | 8,000 | RED* |

*See tutorial Step 8 note above on DI vs VAD assignment for `gbm_methylation`.

These views are procedurally generated — no external data files are bundled.
Real manuscript DI and ρ reference values are stored as metadata on each view
(`manuscript_di`, `manuscript_rho`) for comparison.

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

Full preprocessing pipeline: [github.com/shoaibms/var-pre](https://github.com/shoaibms/var-pre)

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

The test suite collects **107 tests** across 11 test classes, with coverage
enforced above 90% on core modules. Run `pytest --co -q | tail -1` to verify the
current count before any release.

---

## Project Structure

```
vardiag/
├── vardiag/
│   ├── __init__.py          public API surface
│   ├── metrics.py           all mathematical primitives (IO-free)
│   ├── core.py              diagnose / diagnose_cv / scan
│   ├── validation.py        strict input validation
│   ├── data.py              bundled synthetic manuscript views
│   ├── cli.py               command-line interface
│   ├── py.typed             PEP 561 type marker
│   └── tests/
│       └── test_vardiag.py  107 tests
├── tutorial/
│   └── 01_full_tutorial.py  10-step manuscript walkthrough
├── examples/
│   ├── quickstart.py            self-contained bundled-view examples
│   └── scan_multiclass_shap.py  correct multiclass SHAP integration
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CITATION.cff
└── LICENSE
```

---

## Citation

```bibtex
@article{vardiag2026,
  title   = {When Variance Misleads: A Diagnostic Framework for Feature
             Selection in Multi-Omics Data},
  author  = {Shoaib, Mirza and others},
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
