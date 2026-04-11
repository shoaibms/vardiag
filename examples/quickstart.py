#!/usr/bin/env python3
"""
vardiag quickstart
==================
Self-contained demonstration using the bundled synthetic manuscript views.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

print("=" * 72)
print("  vardiag — Quickstart")
print("=" * 72)

from vardiag import diagnose, diagnose_cv, scan
from vardiag.data import load_all_views


# ── 1. Run VAD on all four bundled views ─────────────────────────────────────

print("\n" + "─" * 72)
print("  Example 1: All four bundled manuscript views")
print("─" * 72)

views = load_all_views()

print(f"\n  {'View':<20}  {'Zone':<22}  {'eta_ES':>7}  {'VSA':>7}  {'ms DI':>7}")
print("  " + "─" * 68)
for name, view in views.items():
    result = diagnose(view.X, view.y, k_pct=10)
    print(
        f"  {name:<20}  {result.zone:<22}  "
        f"{result.eta_es:>7.3f}  {result.vsa:>7.3f}  "
        f"{view.manuscript_di:>7.2f}"
    )


# ── 2. Detailed report for the most interesting view ─────────────────────────

print("\n" + "─" * 72)
print("  Example 2: Full diagnostic report — BRCA methylation")
print("─" * 72)

view = views["brca_methylation"]
result = diagnose(view.X, view.y, k_pct=10)
print(result.summary())


# ── 3. Multiple K% values ─────────────────────────────────────────────────────

print("\n" + "─" * 72)
print("  Example 3: Varying K% budget (BRCA methylation)")
print("─" * 72)

print(f"\n  {'K%':>4}  {'eta_ES':>7}  {'VSA':>7}  {'Zone'}")
print("  " + "─" * 48)
for k in [1, 5, 10, 20]:
    r = diagnose(view.X, view.y, k_pct=k)
    print(f"  {k:>4}  {r.eta_es:>7.3f}  {r.vsa:>7.3f}  {r.zone}")


# ── 4. Leakage-free CV version ────────────────────────────────────────────────

print("\n" + "─" * 72)
print("  Example 4: Leakage-free CV version")
print("─" * 72)

try:
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_folds = [tr for tr, _ in skf.split(view.X, view.y)]
    r_cv = diagnose_cv(view.X, view.y, cv_folds=train_folds, k_pct=10)
    print(f"\n  CV-aggregated zone : {r_cv.zone}")
    print(f"  eta_ES (mean)      : {r_cv.eta_es:.3f}")
    print(f"  VSA  (mean)        : {r_cv.vsa:.3f}")
    print(f"  Elapsed            : {r_cv.elapsed_s:.2f} s")
except ImportError:
    print("  (scikit-learn not installed — skipping CV example)")
    print('  Install with: pip install "vardiag[full]"')


# ── 5. Full scan() with mock SHAP ─────────────────────────────────────────────

print("\n" + "─" * 72)
print("  Example 5: Full scan() with mock SHAP importance")
print("─" * 72)

rng = np.random.default_rng(42)
feature_names = view.feature_names
n_feat = len(feature_names)

# Mock: randomly assign importance — in practice use a fitted model + SHAP
shap_imp = {name: float(rng.random()) for name in feature_names}

report = scan(
    view.X,
    view.y,
    shap_importance=shap_imp,
    feature_names=feature_names,
    k_pcts=[1, 5, 10, 20],
    primary_k=10,
)
print(report.summary())


# ── 6. Export to dict / DataFrame ─────────────────────────────────────────────

print("\n" + "─" * 72)
print("  Example 6: Export results to DataFrame")
print("─" * 72)

try:
    import pandas as pd
    rows = []
    for name, v in views.items():
        r = diagnose(v.X, v.y, k_pct=10)
        row = r.to_dict()
        row["view"] = name
        rows.append(row)
    df = pd.DataFrame(rows)[["view", "zone", "eta_es", "vsa", "alpha_prime", "f_di"]]
    print("\n" + df.to_string(index=False))
except ImportError:
    print("  (pandas not installed — skipping DataFrame example)")
    print('  Install with: pip install "vardiag[full]"')


print("\n" + "=" * 72)
print("  Quickstart complete.")
print("=" * 72)
