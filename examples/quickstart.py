#!/usr/bin/env python3
"""
vardiag quickstart example
===========================
Demonstrates the full VAD workflow on two synthetic multi-omics-style datasets:

  1. Coupled view    — high-variance features carry class signal → GREEN_SAFE
  2. Decoupled view  — high-variance features are pure noise     → RED_HARMFUL

Run:
    python examples/quickstart.py
"""

import numpy as np
import sys

print("=" * 60)
print("  vardiag — Variance Alignment Diagnostic  Quickstart")
print("=" * 60)

# ── 1. Import ────────────────────────────────────────────────────────────────
from vardiag import diagnose, diagnose_cv, scan

# ── 2. Synthetic data generators ─────────────────────────────────────────────

def make_coupled(n=200, p=1000, seed=0):
    """
    COUPLED: top-variance features also carry class-discriminative signal.
    Mirrors e.g. CCLE mRNA (DI = 0.92 in paper).
    """
    rng = np.random.default_rng(seed)
    y   = np.array([0] * (n // 2) + [1] * (n // 2))
    X   = rng.standard_normal((n, p)).astype(np.float32)
    # First 100 features: high variance AND high signal
    X[:, :100]      *= 3.0          # inflate variance
    X[:n//2, :100]  -= 3.0          # class 0 shifted down
    X[n//2:, :100]  += 3.0          # class 1 shifted up
    return X, y, [f"gene_{i}" for i in range(p)]


def make_decoupled(n=200, p=1000, seed=1):
    """
    DECOUPLED: top-variance features are inflated noise; signal lives
    in low-variance features. Mirrors e.g. MLOmics methylation (DI = 1.03).
    """
    rng = np.random.default_rng(seed)
    y   = np.array([0] * (n // 2) + [1] * (n // 2))
    X   = rng.standard_normal((n, p)).astype(np.float32)
    # First 100: high variance but NO class signal
    X[:, :100] *= 10.0
    # Features 100–200: low variance but discriminative
    X[:n//2, 100:200] -= 1.5
    X[n//2:, 100:200] += 1.5
    return X, y, [f"cpg_{i}" for i in range(p)]


# ── 3. Basic diagnose() ───────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("  Example 1: Coupled view (variance ≈ signal)")
print("─" * 60)
X_c, y_c, names_c = make_coupled()
result_c = diagnose(X_c, y_c, k_pct=10)
print(result_c.summary())

print("\n" + "─" * 60)
print("  Example 2: Decoupled view (variance ≠ signal)")
print("─" * 60)
X_d, y_d, names_d = make_decoupled()
result_d = diagnose(X_d, y_d, k_pct=10)
print(result_d.summary())


# ── 4. Multiple K% values ─────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Example 3: Varying K% budget (decoupled view)")
print("─" * 60)
print(f"\n  {'K%':>4}  {'η_ES':>7}  {'VSA':>7}  {'Zone'}")
print("  " + "-" * 44)
for k in [1, 5, 10, 20]:
    r = diagnose(X_d, y_d, k_pct=k)
    print(f"  {k:>4}  {r.eta_es:>7.3f}  {r.vsa:>7.3f}  {r.zone}")


# ── 5. CV-safe version ────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Example 4: Leakage-free CV version")
print("─" * 60)
try:
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_folds = [tr for tr, _ in skf.split(X_d, y_d)]
    r_cv = diagnose_cv(X_d, y_d, cv_folds=train_folds, k_pct=10)
    print(f"\n  CV-aggregated zone : {r_cv.zone}")
    print(f"  η_ES (mean)        : {r_cv.eta_es:.3f}")
    print(f"  VSA  (mean)        : {r_cv.vsa:.3f}")
    print(f"  Elapsed            : {r_cv.elapsed_s:.2f} s")
except ImportError:
    print("  (scikit-learn not installed — skipping CV example)")


# ── 6. Full scan() with mock SHAP ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Example 5: Full scan() with SHAP importance")
print("─" * 60)
rng = np.random.default_rng(42)

# Mock SHAP: correct signal features (100–200) get high importance
n_feat = len(names_d)
shap_imp = {}
for i, name in enumerate(names_d):
    if 100 <= i < 200:
        shap_imp[name] = float(rng.random() * 0.5 + 0.5)   # high: true signal
    else:
        shap_imp[name] = float(rng.random() * 0.1)           # low: noise

report = scan(
    X_d, y_d,
    shap_importance=shap_imp,
    feature_names=names_d,
    k_pcts=[1, 5, 10, 20],
    primary_k=10,
)
print(report.summary())


# ── 7. to_dict() for DataFrame integration ───────────────────────────────────
print("\n" + "─" * 60)
print("  Example 6: Export to DataFrame")
print("─" * 60)
try:
    import pandas as pd
    rows = []
    for label, result in [("coupled_mRNA", result_c), ("decoupled_methylation", result_d)]:
        row = result.to_dict()
        row["view"] = label
        rows.append(row)
    df = pd.DataFrame(rows)[["view", "zone", "eta_es", "vsa", "alpha_prime", "pcla"]]
    print("\n" + df.to_string(index=False))
except ImportError:
    print("  (pandas not installed — skipping DataFrame example)")

print("\n" + "=" * 60)
print("  Quickstart complete.")
print("=" * 60)
