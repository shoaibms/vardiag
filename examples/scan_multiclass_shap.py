#!/usr/bin/env python3
"""
Example: vardiag scan() with multiclass SHAP aggregation.

This script shows the correct way to compute SHAP importance for
multiclass problems and pass it to vardiag.scan().

Run:
    pip install shap scikit-learn
    python examples/scan_multiclass_shap.py
"""

from __future__ import annotations

import numpy as np

try:
    import shap
except ImportError as e:
    raise ImportError(
        "This example requires shap. Install it with: pip install shap"
    ) from e

from vardiag import scan


def mean_abs_shap_importance(shap_values) -> np.ndarray:
    """
    Convert SHAP outputs to a single per-feature importance vector.

    Handles all three common output shapes from shap.TreeExplainer:

    - list[n_classes] of (n_samples, n_features)   — TreeExplainer multiclass
    - ndarray (n_classes, n_samples, n_features)    — stacked multiclass
    - ndarray (n_samples, n_features)               — binary or regression

    Returns
    -------
    ndarray, shape (n_features,)
    """
    if isinstance(shap_values, list):
        # multiclass: list of per-class arrays
        arr = np.stack(shap_values, axis=0)          # (n_classes, n_samples, n_features)
        return np.abs(arr).mean(axis=(0, 1))         # mean over classes and samples

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return np.abs(arr).mean(axis=(0, 1))         # (n_classes, n_samples, n_features)
    if arr.ndim == 2:
        return np.abs(arr).mean(axis=0)              # (n_samples, n_features)

    raise ValueError(
        f"Unsupported SHAP shape: {arr.shape}. "
        "Expected 2-D (n_samples, n_features) or 3-D (n_classes, n_samples, n_features)."
    )


def build_scan_report(model, X_train, y_train, feature_names):
    """
    Compute a vardiag ScanReport from a fitted model.

    Parameters
    ----------
    model        : fitted tree-based estimator (RandomForest, XGBoost, etc.)
    X_train      : ndarray (n_samples, n_features)
    y_train      : ndarray (n_samples,)
    feature_names: list[str], length n_features

    Returns
    -------
    ScanReport
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap_imp = mean_abs_shap_importance(shap_values)

    shap_importance = dict(zip(feature_names, shap_imp.tolist()))

    return scan(
        X_train,
        y_train,
        shap_importance=shap_importance,
        feature_names=feature_names,
        k_pcts=[1, 5, 10, 20],
        primary_k=10,
    )


# ---------------------------------------------------------------------------
# Standalone demo — runs on synthetic data without any external dataset
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as e:
        raise ImportError(
            "The demo requires scikit-learn. "
            'Install it with: pip install "vardiag[full]"'
        ) from e

    print("Building synthetic demo data...")
    rng = np.random.default_rng(42)
    n, p, n_classes = 200, 100, 3

    # Coupled: first 20 features are both high-variance and discriminative
    X_demo = rng.standard_normal((n, p)).astype("float32")
    y_demo = np.repeat(np.arange(n_classes), n // n_classes)
    for c in range(n_classes):
        X_demo[y_demo == c, :20] += c * 2.0
    X_demo[:, :20] *= 2.0

    feat_names = [f"gene_{i}" for i in range(p)]

    print(f"Fitting RandomForest on {n} samples × {p} features ({n_classes} classes)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_demo, y_demo)

    print("Computing SHAP values and running scan()...\n")
    report = build_scan_report(model, X_demo, y_demo, feat_names)
    print(report.summary())
