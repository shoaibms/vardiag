"""
vardiag.core
============
High-level public API for the Variance Alignment Diagnostic (VAD).

Typical usage
-------------
>>> import numpy as np
>>> from vardiag import diagnose
>>> X, y = load_my_omics_data()            # (n_samples, n_features), labels
>>> result = diagnose(X, y, k_pct=10)
>>> print(result.zone)                     # 'RED_HARMFUL' | 'GREEN_SAFE' | 'YELLOW_INCONCLUSIVE'
>>> print(result.summary())

For downstream DI analysis (requires SHAP importance):
>>> from vardiag import scan
>>> report = scan(X, y, shap_importance=shap_dict, k_pcts=[1, 5, 10, 20])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .metrics import (
    eta2_features,
    eta_enrichment,
    vsa_mannwhitney,
    alpha_prime,
    pca_alignment,
    f_di,
    classify_zone,
    compute_overlap_curve,
    rank_features,
    OverlapRow,
)
from .validation import validate_xy, validate_cv_folds, validate_scan_inputs


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticResult:
    """
    Output of ``diagnose(X, y)``.

    Attributes
    ----------
    zone : str
        "GREEN_SAFE", "RED_HARMFUL", or "YELLOW_INCONCLUSIVE"
    eta_es : float
        Signal enrichment ratio: mean(η² in TopVar(K%)) / mean(η² overall).
        > 1 means high-variance features carry above-average signal.
    vsa : float
        Variance-Signal Alignment (Mann–Whitney AUROC − 0.5). Range [−0.5, +0.5].
    alpha_prime : float
        Spearman(V_total, η²). K-free monotone alignment.
    pcla : float
        PCA-weighted signal alignment (eigenvalue-weighted η² of PCs).
    sas : float
        Spectral alignment score: Spearman(explained_variance, η²_PC).
    f_di : float
        Supervision-free DI analogue (uses η² ranking, no model training).
    k_pct : int
        Feature budget used (%).
    n_features : int
        Total number of features.
    n_samples : int
        Number of training samples.
    n_classes : int
        Number of unique classes.
    elapsed_s : float
        Wall-clock time in seconds.
    """
    zone: str
    eta_es: float
    vsa: float
    alpha_prime: float
    pcla: float
    sas: float
    f_di: float
    k_pct: int
    n_features: int
    n_samples: int
    n_classes: int
    elapsed_s: float

    def summary(self) -> str:
        """Human-readable one-page summary."""
        zone_emoji = {"GREEN_SAFE": "✅", "RED_HARMFUL": "🔴", "YELLOW_INCONCLUSIVE": "⚠️"}.get(self.zone, "❓")
        lines = [
            "=" * 60,
            f"  VAD Diagnostic Report  (K = {self.k_pct}%)",
            "=" * 60,
            f"  Dataset    : {self.n_samples} samples × {self.n_features} features  ({self.n_classes} classes)",
            f"  Computed in: {self.elapsed_s:.2f} s",
            "",
            f"  Zone       : {zone_emoji}  {self.zone}",
            "",
            "  Metrics",
            f"    η_ES     : {self.eta_es:+.3f}  (>1 = enriched for signal)",
            f"    VSA      : {self.vsa:+.3f}  (>0 = aligned)",
            f"    α'       : {self.alpha_prime:+.3f}  (Spearman V vs η²)",
            f"    PCLA     : {self.pcla:+.3f}  (PCA-weighted η²)",
            f"    SAS      : {self.sas:+.3f}  (spectral alignment)",
            f"    F-DI     : {self.f_di:+.3f}  (<1 = coupled)",
            "",
            "  Decision rule",
        ]
        if self.zone == "GREEN_SAFE":
            lines.append("    Proceed with variance filtering. Signal is concentrated")
            lines.append("    in high-variance features for this view.")
        elif self.zone == "RED_HARMFUL":
            lines.append("    ⚠️  Variance filtering likely harmful. Use importance-guided")
            lines.append("    selection (e.g. SHAP) or include all features.")
        else:
            lines.append("    Uncertain. Run a small pilot ablation comparing TopVar vs")
            lines.append("    Random selection before committing.")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        """Flat dict representation suitable for a DataFrame row."""
        return {
            "zone": self.zone,
            "eta_es": self.eta_es,
            "vsa": self.vsa,
            "alpha_prime": self.alpha_prime,
            "pcla": self.pcla,
            "sas": self.sas,
            "f_di": self.f_di,
            "k_pct": self.k_pct,
            "n_features": self.n_features,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "elapsed_s": self.elapsed_s,
        }


@dataclass
class ScanReport:
    """
    Output of ``scan(X, y, shap_importance, k_pcts)`` —
    full DI curve + VAD diagnostics at multiple feature budgets.
    """
    vad: DiagnosticResult                  # VAD at primary_k
    di_curve: List[OverlapRow]             # DI(K) across k_pcts
    hidden_biomarker_fraction: float       # fraction of features: low-var but high-importance
    gene_level_jaccard: float              # overlap between TopVar and TopSHAP feature sets
    k_pcts: List[float]
    primary_k: int

    def summary(self) -> str:
        lines = [self.vad.summary(), ""]
        lines.append("  DI Curve")
        lines.append(f"  {'K%':>4}  {'J':>6}  {'J_rand':>7}  {'DI':>6}")
        lines.append("  " + "-" * 28)
        for row in self.di_curve:
            lines.append(f"  {row.k_pct:>4.0f}  {row.J:>6.3f}  {row.J_rand:>7.3f}  {row.DI:>6.3f}")
        lines.append("")
        lines.append(f"  Hidden biomarkers : {self.hidden_biomarker_fraction:.1%} of features")
        lines.append(f"  Gene-level Jaccard: {self.gene_level_jaccard:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Primary public functions
# ---------------------------------------------------------------------------

def diagnose(
    X: np.ndarray,
    y: np.ndarray,
    k_pct: int = 10,
    n_pca_components: int = 30,
    random_state: int = 0,
    margin: float = 0.05,
) -> DiagnosticResult:
    """
    Run the Variance Alignment Diagnostic on a single (X, y) view.

    Executes in < 1 second on typical omics datasets.
    All computation is on the data as provided — no cross-validation split is
    applied here. For leakage-free use in ML pipelines, pass only training-set
    data (see ``diagnose_cv``).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix (pre-normalised, as it will be used for modelling).
    y : ndarray, shape (n_samples,)
        Class labels.
    k_pct : int
        Feature budget as percent of total features (default 10).
    n_pca_components : int
        Number of PCs for PCLA/SAS computation (default 30).
    random_state : int
        Seed for PCA randomised SVD solver.
    margin : float
        Conservative margin on η_ES for zone classification (default 0.05).

    Returns
    -------
    DiagnosticResult
    """
    t0 = time.perf_counter()

    X, y = validate_xy(X, y, k_pct=k_pct,
                       n_pca_components=n_pca_components,
                       caller="diagnose")

    n_samples, n_features = X.shape
    n_classes = int(np.unique(y).size)
    v_total, v_between, eta2 = eta2_features(X, y)

    # VAD metrics
    eta_es, _, _ = eta_enrichment(eta2, v_total, k_pct=k_pct)
    vsa           = vsa_mannwhitney(eta2, v_total, k_pct=k_pct)
    ap            = alpha_prime(v_total, eta2)
    fdi           = f_di(eta2, v_total, k_pct=k_pct)
    pca_out       = pca_alignment(X, y,
                                  n_components=n_pca_components,
                                  random_state=random_state)
    sas  = pca_out.get("sas",  float("nan"))
    pcla = pca_out.get("pcla", float("nan"))

    zone = classify_zone(eta_es, vsa, margin=margin)
    elapsed = time.perf_counter() - t0

    return DiagnosticResult(
        zone=zone,
        eta_es=float(eta_es),
        vsa=float(vsa),
        alpha_prime=float(ap),
        pcla=float(pcla),
        sas=float(sas),
        f_di=float(fdi),
        k_pct=int(k_pct),
        n_features=int(n_features),
        n_samples=int(n_samples),
        n_classes=int(n_classes),
        elapsed_s=float(elapsed),
    )


def scan(
    X: np.ndarray,
    y: np.ndarray,
    shap_importance: Mapping[str, float],
    feature_names: Optional[Sequence[str]] = None,
    k_pcts: Iterable[float] = (1, 5, 10, 20),
    primary_k: int = 10,
    n_pca_components: int = 30,
    random_state: int = 0,
) -> ScanReport:
    """
    Full VAD + DI analysis: requires pre-computed SHAP importance scores.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,)
    shap_importance : dict  {feature_name: mean_abs_shap}
    feature_names : optional list of feature names (len = n_features).
        Defaults to ["f0", "f1", ...] if not provided.
    k_pcts : K% values for DI curve (default [1, 5, 10, 20])
    primary_k : K% for primary VAD zone (default 10)
    n_pca_components : for PCLA/SAS
    random_state : for PCA

    Returns
    -------
    ScanReport
    """
    X, y, feature_names = validate_scan_inputs(
        X, y, shap_importance, feature_names, k_pct=primary_k, caller="scan"
    )
    k_pcts_list = list(k_pcts)
    n_samples, n_features = X.shape

    # VAD diagnostic at primary_k
    vad_result = diagnose(X, y, k_pct=primary_k,
                          n_pca_components=n_pca_components,
                          random_state=random_state)

    # Variance ranking
    v_total, _, eta2 = eta2_features(X, y)
    var_scores = dict(zip(feature_names, v_total.tolist()))
    ranked_by_var = rank_features(var_scores, ascending=False)

    # Importance ranking from SHAP
    ranked_by_shap = rank_features(shap_importance, ascending=False)
    # Restrict to features present in both
    common = set(ranked_by_var) & set(ranked_by_shap)
    ranked_by_var_common  = [f for f in ranked_by_var  if f in common]
    ranked_by_shap_common = [f for f in ranked_by_shap if f in common]

    # DI curve
    di_curve = compute_overlap_curve(ranked_by_var_common, ranked_by_shap_common, k_pcts_list)

    # Hidden biomarker fraction at primary_k
    p = n_features
    top_n = max(1, int(p * float(primary_k) / 100.0))
    top_var_idx  = set(np.argpartition(-np.where(np.isfinite(v_total), v_total, -np.inf),
                                        top_n - 1)[:top_n].tolist())
    shap_arr     = np.array([shap_importance.get(f, 0.0) for f in feature_names], dtype=float)
    top_shap_idx = set(np.argpartition(-np.where(np.isfinite(shap_arr), shap_arr, -np.inf),
                                        top_n - 1)[:top_n].tolist())
    # Q4: low variance (not in top_var) but high importance (in top_shap)
    hidden = top_shap_idx - top_var_idx
    hidden_fraction = float(len(hidden)) / float(p) if p > 0 else float("nan")

    # Gene-level Jaccard between TopVar and TopSHAP
    topvar_set  = set(ranked_by_var_common[:top_n])
    topshap_set = set(ranked_by_shap_common[:top_n])
    union_n = len(topvar_set | topshap_set)
    gene_jaccard = (len(topvar_set & topshap_set) / union_n) if union_n > 0 else 0.0

    return ScanReport(
        vad=vad_result,
        di_curve=di_curve,
        hidden_biomarker_fraction=hidden_fraction,
        gene_level_jaccard=float(gene_jaccard),
        k_pcts=k_pcts_list,
        primary_k=primary_k,
    )


def diagnose_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: Sequence[np.ndarray],
    k_pct: int = 10,
    n_pca_components: int = 30,
    random_state: int = 0,
) -> DiagnosticResult:
    """
    Leakage-free VAD: compute metrics fold-by-fold on training splits only,
    then aggregate.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,)
    cv_folds : list of 1-D arrays of training indices per fold
        e.g. from sklearn.model_selection.StratifiedKFold
    k_pct, n_pca_components, random_state : as in ``diagnose``

    Returns
    -------
    DiagnosticResult  (averaged across folds)
    """
    t0 = time.perf_counter()
    X, y = validate_xy(X, y, k_pct=k_pct,
                       n_pca_components=n_pca_components,
                       caller="diagnose_cv")
    cv_folds = validate_cv_folds(cv_folds, n_samples=X.shape[0], caller="diagnose_cv")

    fold_metrics: List[Dict[str, float]] = []
    for fold_idx, train_idx in enumerate(cv_folds):
        Xtr = X[train_idx, :]
        ytr = y[train_idx]
        if len(np.unique(ytr)) < 2:
            continue
        r = diagnose(Xtr, ytr, k_pct=k_pct,
                     n_pca_components=n_pca_components,
                     random_state=random_state + fold_idx)
        fold_metrics.append({
            "eta_es": r.eta_es,
            "vsa": r.vsa,
            "alpha_prime": r.alpha_prime,
            "pcla": r.pcla,
            "sas": r.sas,
            "f_di": r.f_di,
        })

    if not fold_metrics:
        raise ValueError("No valid folds (need at least 2 classes per fold).")

    def _mean(key: str) -> float:
        vals = [m[key] for m in fold_metrics if np.isfinite(m[key])]
        return float(np.mean(vals)) if vals else float("nan")

    eta_es_mean = _mean("eta_es")
    vsa_mean    = _mean("vsa")
    zone = classify_zone(eta_es_mean, vsa_mean)
    elapsed = time.perf_counter() - t0

    return DiagnosticResult(
        zone=zone,
        eta_es=eta_es_mean,
        vsa=vsa_mean,
        alpha_prime=_mean("alpha_prime"),
        pcla=_mean("pcla"),
        sas=_mean("sas"),
        f_di=_mean("f_di"),
        k_pct=int(k_pct),
        n_features=int(X.shape[1]),
        n_samples=int(X.shape[0]),
        n_classes=int(np.unique(y).size),
        elapsed_s=float(elapsed),
    )
