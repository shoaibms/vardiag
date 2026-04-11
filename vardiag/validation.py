"""
vardiag.validation
==================
Strict, early input validation for all public API functions.
Fails fast with clear, actionable error messages.
"""
from __future__ import annotations

import warnings
from typing import Mapping, Optional, Sequence

import numpy as np


def validate_xy(
    X: np.ndarray,
    y: np.ndarray,
    k_pct: int = 10,
    n_pca_components: int = 30,
    caller: str = "diagnose",
) -> tuple:
    """
    Validate and coerce (X, y) inputs for diagnose() and diagnose_cv().

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features), float32
    y : np.ndarray, shape (n_samples,)
    """
    # --- coerce types ---
    try:
        X = np.asarray(X, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise TypeError(f"[{caller}] X could not be converted to a numeric array: {e}") from e

    try:
        y = np.asarray(y)
    except (TypeError, ValueError) as e:
        raise TypeError(f"[{caller}] y could not be converted to an array: {e}") from e

    # --- dimensionality ---
    if X.ndim != 2:
        raise ValueError(
            f"[{caller}] X must be 2-D (n_samples, n_features), "
            f"got shape {X.shape}. "
            f"If X has shape (n_features,), pass X.reshape(1, -1) for a single sample."
        )
    if y.ndim != 1:
        raise ValueError(
            f"[{caller}] y must be 1-D (n_samples,), got shape {y.shape}."
        )

    n_samples, n_features = X.shape

    # --- size ---
    if n_samples != y.shape[0]:
        raise ValueError(
            f"[{caller}] X and y must have the same number of samples. "
            f"Got X.shape[0]={n_samples}, len(y)={y.shape[0]}."
        )
    if n_samples < 4:
        raise ValueError(
            f"[{caller}] Need at least 4 samples, got {n_samples}."
        )
    if n_features < 2:
        raise ValueError(
            f"[{caller}] Need at least 2 features, got {n_features}. "
            f"VAD is not meaningful on single-feature data."
        )

    # --- class labels ---
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError(
            f"[{caller}] VAD requires at least 2 classes, "
            f"got y with only 1 unique value: {classes[0]}."
        )

    # class balance warning (not an error)
    counts = np.array([(y == c).sum() for c in classes])
    min_count = int(counts.min())
    if min_count < 3:
        warnings.warn(
            f"[{caller}] Minority class has only {min_count} sample(s). "
            f"VAD metrics may be unreliable. Results should be interpreted cautiously.",
            UserWarning,
            stacklevel=3,
        )

    # --- k_pct ---
    if not isinstance(k_pct, (int, float)) or not (0 < k_pct < 100):
        raise ValueError(
            f"[{caller}] k_pct must be a number in the range (0, 100), got {k_pct!r}."
        )

    # --- pca components ---
    if n_pca_components < 2:
        raise ValueError(
            f"[{caller}] n_pca_components must be >= 2, got {n_pca_components}."
        )

    # --- NaN/Inf content warnings ---
    nan_cols = int(np.any(~np.isfinite(X), axis=0).sum())
    if nan_cols > 0:
        frac = nan_cols / n_features
        warnings.warn(
            f"[{caller}] X contains NaN or Inf values in {nan_cols}/{n_features} "
            f"features ({frac:.1%}). These are handled by mean-imputation internally, "
            f"but consider inspecting your preprocessing.",
            UserWarning,
            stacklevel=3,
        )

    # --- constant features warning ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_var = np.nanvar(X, axis=0, ddof=0)
    n_const = int((col_var == 0).sum())
    if n_const > 0:
        warnings.warn(
            f"[{caller}] {n_const}/{n_features} features have zero variance (constant). "
            f"These will always rank last by variance and will not affect η_ES or VSA.",
            UserWarning,
            stacklevel=3,
        )

    return X, y


def validate_cv_folds(
    cv_folds: Sequence,
    n_samples: int,
    caller: str = "diagnose_cv",
) -> list:
    """
    Validate cv_folds: each element must be a 1-D array of valid training indices.
    """
    if not hasattr(cv_folds, "__iter__"):
        raise TypeError(f"[{caller}] cv_folds must be iterable, got {type(cv_folds)}.")

    folds = list(cv_folds)
    if len(folds) == 0:
        raise ValueError(f"[{caller}] cv_folds is empty — need at least 1 fold.")

    validated = []
    for i, fold in enumerate(folds):
        try:
            fold_arr = np.asarray(fold, dtype=int).ravel()
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"[{caller}] cv_folds[{i}] could not be converted to integer indices: {e}"
            ) from e

        if fold_arr.ndim != 1:
            raise ValueError(
                f"[{caller}] cv_folds[{i}] must be 1-D, got shape {fold_arr.shape}."
            )
        if fold_arr.size == 0:
            raise ValueError(f"[{caller}] cv_folds[{i}] is empty.")
        if fold_arr.size < 4:
            warnings.warn(
                f"[{caller}] cv_folds[{i}] has only {fold_arr.size} training samples. "
                f"VAD metrics may be unreliable for small training splits.",
                UserWarning,
                stacklevel=3,
            )
        out_of_range = (fold_arr < 0) | (fold_arr >= n_samples)
        if out_of_range.any():
            bad = fold_arr[out_of_range][:5].tolist()
            raise ValueError(
                f"[{caller}] cv_folds[{i}] contains {out_of_range.sum()} out-of-range "
                f"indices (n_samples={n_samples}). Examples: {bad}."
            )
        validated.append(fold_arr)

    return validated


def validate_scan_inputs(
    X: np.ndarray,
    y: np.ndarray,
    shap_importance: Mapping,
    feature_names: Optional[Sequence],
    k_pct: int,
    caller: str = "scan",
) -> tuple:
    """
    Validate all inputs for scan(). Returns (X, y, feature_names).
    """
    X, y = validate_xy(X, y, k_pct=k_pct, caller=caller)
    n_samples, n_features = X.shape

    # --- feature_names ---
    if feature_names is not None:
        feature_names = list(feature_names)
        if len(feature_names) != n_features:
            raise ValueError(
                f"[{caller}] len(feature_names)={len(feature_names)} does not match "
                f"X.shape[1]={n_features}."
            )
        if len(set(feature_names)) != len(feature_names):
            dupes = [f for f in feature_names if feature_names.count(f) > 1]
            warnings.warn(
                f"[{caller}] feature_names contains {len(dupes)} duplicate name(s). "
                f"Examples: {list(set(dupes))[:5]}. DI calculations use set operations "
                f"and duplicates will be deduplicated.",
                UserWarning,
                stacklevel=3,
            )
    else:
        feature_names = [f"f{i}" for i in range(n_features)]

    # --- shap_importance ---
    if not isinstance(shap_importance, Mapping):
        raise TypeError(
            f"[{caller}] shap_importance must be a dict {{feature_name: float}}, "
            f"got {type(shap_importance)}."
        )

    shap_keys = set(shap_importance.keys())
    feat_set  = set(feature_names)
    overlap   = shap_keys & feat_set
    overlap_frac = len(overlap) / max(len(feat_set), 1)

    if len(overlap) == 0:
        raise ValueError(
            f"[{caller}] shap_importance keys and feature_names share NO common names. "
            f"Example shap key: {next(iter(shap_keys), 'N/A')}. "
            f"Example feature name: {feature_names[0] if feature_names else 'N/A'}. "
            f"These must match."
        )
    if overlap_frac < 0.5:
        warnings.warn(
            f"[{caller}] Only {overlap_frac:.0%} of feature_names are present in "
            f"shap_importance ({len(overlap)}/{len(feat_set)}). "
            f"DI calculation will be restricted to the overlapping set.",
            UserWarning,
            stacklevel=3,
        )

    return X, y, feature_names
