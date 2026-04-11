"""
vardiag.metrics
===============
Core, IO-free computational primitives for the Variance Alignment Diagnostic (VAD)
and the Decoupling Index (DI).

All functions operate on plain NumPy arrays (X, y) and have no project-specific
dependencies. scipy and sklearn are optional; pure-NumPy fallbacks are provided.

Reference
---------
"When Variance Misleads: A Diagnostic Framework for Feature Selection
in Multi-Omics Data"  (manuscript, 2026)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import warnings

import numpy as np

try:
    from scipy import stats as _stats
except Exception:  # pragma: no cover
    _stats = None

try:
    from sklearn.decomposition import PCA as _PCA
except Exception:  # pragma: no cover
    _PCA = None

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float32, copy=False)
    return x


def _nanmean0(X: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(X, axis=axis)
    return np.where(np.isfinite(m), m, 0.0)


def _nanvar0(X: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanvar(X, axis=axis, ddof=0)
    return np.where(np.isfinite(v), v, 0.0)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    if _stats is not None:
        r, _ = _stats.spearmanr(a[m], b[m])
        return float(r) if np.isfinite(r) else float("nan")
    # pure-NumPy fallback (approximate for ties)
    ra = a[m].argsort().argsort().astype(float)
    rb = b[m].argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float("nan") if denom <= 0 else float((ra * rb).sum() / denom)


def _mean_impute(X: np.ndarray) -> np.ndarray:
    X = _as_float(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(X.dtype, copy=False)
    out = X.copy()
    mask = ~np.isfinite(out)
    if mask.any():
        out[mask] = np.take(col_means, np.where(mask)[1])
    return out


# ---------------------------------------------------------------------------
# VAD primitives
# ---------------------------------------------------------------------------

def eta2_features(
    X: np.ndarray,
    y: np.ndarray,
    eps: float = _EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-feature variance decomposition.

    Computes for each feature j:
      - V_total_j   : total variance  (nanvar, ddof=0)
      - V_between_j : between-class variance  (law of total variance)
      - eta2_j      : signal fraction  = V_between_j / (V_total_j + eps), clipped [0,1]

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)  — class labels
    eps : float, numerical stability guard

    Returns
    -------
    v_total, v_between, eta2  — each shape (n_features,)
    """
    X = _as_float(X)
    y = np.asarray(y)
    n, p = X.shape
    if n == 0 or p == 0:
        z = np.zeros(p, dtype=float)
        return z.copy(), z.copy(), z.copy()

    mu = _nanmean0(X, axis=0)
    v_total = _nanvar0(X, axis=0)

    classes, counts = np.unique(y, return_counts=True)
    v_between = np.zeros(p, dtype=np.float64)
    for c, cnt in zip(classes, counts):
        if cnt <= 0:
            continue
        w = float(cnt) / float(n)
        mu_c = _nanmean0(X[y == c, :], axis=0)
        diff = (mu_c - mu).astype(np.float64, copy=False)
        v_between += w * diff * diff

    eta2 = v_between / (v_total.astype(np.float64) + float(eps))
    eta2 = np.where(np.isfinite(eta2), eta2, 0.0)
    eta2 = np.clip(eta2, 0.0, 1.0)
    v_between = np.where(np.isfinite(v_between), v_between, 0.0)

    return v_total.astype(np.float64), v_between, eta2


def eta_enrichment(
    eta2: np.ndarray,
    v_total: np.ndarray,
    k_pct: int = 10,
    eps: float = _EPS,
) -> Tuple[float, float, float]:
    """
    Signal enrichment ratio (eta_ES).

    eta_ES(K) = mean(eta^2 in TopVar(K%)) / mean(eta^2 overall)

    eta_ES > 1 : high-variance features are enriched for signal  → safe to filter
    eta_ES < 1 : high-variance features are depleted of signal   → harmful to filter

    Returns
    -------
    eta_es, eta_topvar, eta_all
    """
    eta2 = np.asarray(eta2, dtype=float)
    v_total = np.asarray(v_total, dtype=float)
    p = int(eta2.size)
    if p == 0:
        return float("nan"), float("nan"), float("nan")

    top_n = max(1, int(p * float(k_pct) / 100.0))
    v = np.where(np.isfinite(v_total), v_total, -np.inf)
    top_idx = np.arange(p) if top_n >= p else np.argpartition(-v, top_n - 1)[:top_n]

    valid = eta2[np.isfinite(eta2)]
    eta_all = float(np.mean(valid)) if valid.size else 0.0
    eta_top = float(np.mean(eta2[top_idx])) if top_idx.size else float("nan")
    eta_es  = float(eta_top / (eta_all + float(eps)))

    return eta_es, eta_top, eta_all


def vsa_mannwhitney(
    eta2: np.ndarray,
    v_total: np.ndarray,
    k_pct: int = 10,
) -> float:
    """
    Variance-Signal Alignment (VSA).

    VSA(K) = AUROC(eta^2; TopVar(K%) vs Rest) - 0.5

    Computed via Mann–Whitney U.
    Range: [-0.5, +0.5].  VSA > 0 → signal enriched in high-variance tail.

    Parameters
    ----------
    eta2    : per-feature signal fraction (from eta2_features)
    v_total : per-feature total variance
    k_pct   : feature budget as % of total features
    """
    eta2 = np.asarray(eta2, dtype=float)
    v_total = np.asarray(v_total, dtype=float)
    p = int(eta2.size)
    if p < 3:
        return float("nan")

    top_n = max(1, int(p * float(k_pct) / 100.0))
    v = np.where(np.isfinite(v_total), v_total, -np.inf)
    if top_n >= p:
        return 0.0

    top_idx = np.argpartition(-v, top_n - 1)[:top_n]
    mask_top = np.zeros(p, dtype=bool)
    mask_top[top_idx] = True

    x1 = eta2[mask_top & np.isfinite(eta2)]
    x0 = eta2[(~mask_top) & np.isfinite(eta2)]
    if x1.size == 0 or x0.size == 0:
        return float("nan")

    if _stats is not None:
        u, _ = _stats.mannwhitneyu(x1, x0, alternative="two-sided")
        return float(u / (x1.size * x0.size) - 0.5)

    # pure-NumPy fallback
    x = np.concatenate([x1, x0])
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    r1 = ranks[: x1.size].sum()
    u  = r1 - x1.size * (x1.size + 1) / 2.0
    return float(u / (x1.size * x0.size) - 0.5)


def alpha_prime(v_total: np.ndarray, eta2: np.ndarray) -> float:
    """
    K-free monotone alignment: Spearman(V_total, eta^2).

    Positive → variance and signal co-vary.  Negative → anti-aligned.
    """
    return _safe_spearman(np.asarray(v_total, dtype=float),
                          np.asarray(eta2, dtype=float))


def eta2_1d(z: np.ndarray, y: np.ndarray, eps: float = _EPS) -> float:
    """eta^2 for a single feature vector z against class labels y."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y)
    n = z.size
    if n == 0:
        return float("nan")
    mu = float(np.nanmean(z))
    v_total = float(np.nanvar(z, ddof=0))
    if not np.isfinite(v_total) or v_total <= 0:
        return 0.0
    v_between = 0.0
    for c, cnt in zip(*np.unique(y, return_counts=True)):
        if cnt <= 0:
            continue
        w = float(cnt) / float(n)
        mu_c = float(np.nanmean(z[y == c]))
        v_between += w * (mu_c - mu) ** 2
    eta2 = v_between / (v_total + float(eps))
    return float(min(1.0, max(0.0, eta2))) if np.isfinite(eta2) else 0.0


def pca_alignment(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 30,
    random_state: int = 0,
) -> Dict[str, float]:
    """
    PCA-based multivariate alignment diagnostics.

    Returns
    -------
    dict with keys:
      sas  : Spearman(explained_variance, eta2_per_PC)
             — does variance-capture by PCs correlate with label-discriminability?
      pcla : sum(normalised_eigenvalue * eta2_PC)   — eigenvalue-weighted signal alignment
    """
    if _PCA is None:
        return {"sas": float("nan"), "pcla": float("nan")}

    X = _mean_impute(X)
    y = np.asarray(y)
    n, p = X.shape
    m = int(min(max(1, n_components), max(1, n - 1), max(1, p)))
    if m < 2:
        return {"sas": float("nan"), "pcla": float("nan")}

    pca = _PCA(n_components=m, svd_solver="randomized", random_state=int(random_state))
    Z = pca.fit_transform(X)
    lambdas = np.where(np.isfinite(pca.explained_variance_),
                       pca.explained_variance_, 0.0).astype(float)

    eta2_pcs = np.array([eta2_1d(Z[:, k], y) for k in range(Z.shape[1])], dtype=float)
    sas = _safe_spearman(lambdas, eta2_pcs)
    w   = lambdas / (float(lambdas.sum()) + _EPS)
    pcla = float(np.sum(w * eta2_pcs))

    return {
        "sas":  float(sas) if np.isfinite(sas) else float("nan"),
        "pcla": float(pcla),
    }


def f_di(
    eta2: np.ndarray,
    v_total: np.ndarray,
    k_pct: int = 10,
    eps: float = _EPS,
) -> float:
    """
    F-DI(K) — supervision-free DI analogue using η² ranking instead of SHAP.

    Mirrors the original DI formula using eta^2-ranked sets in place of
    importance-ranked sets. No model training required.

    F-DI > 1 → anti-aligned (low overlap between TopVar and TopEta2)
    F-DI < 1 → coupled
    """
    eta2 = np.asarray(eta2, dtype=float)
    v_total = np.asarray(v_total, dtype=float)
    p = int(eta2.size)
    if p < 3:
        return float("nan")

    top_n = max(1, int(p * float(k_pct) / 100.0))
    if top_n >= p:
        return 0.0

    v = np.where(np.isfinite(v_total), v_total, -np.inf)
    e = np.where(np.isfinite(eta2), eta2, -np.inf)
    idx_var = set(np.argpartition(-v, top_n - 1)[:top_n].tolist())
    idx_eta = set(np.argpartition(-e, top_n - 1)[:top_n].tolist())

    union = len(idx_var | idx_eta)
    if union == 0:
        return float("nan")
    j_obs  = len(idx_var & idx_eta) / union
    q      = float(top_n) / float(p)
    j_rand = q / (2.0 - q)
    tilde_j = (j_obs - j_rand) / (1.0 - j_rand + float(eps))
    return float(1.0 - tilde_j)


def classify_zone(
    eta_es: float,
    vsa: float,
    eta_es_lo: float = float("nan"),
    eta_es_hi: float = float("nan"),
    vsa_lo: float = float("nan"),
    vsa_hi: float = float("nan"),
    margin: float = 0.05,
) -> str:
    """
    Assign a VAD risk zone.

    Returns one of:
      "GREEN_SAFE"          — variance filtering predicted safe
      "RED_HARMFUL"         — variance filtering predicted harmful
      "YELLOW_INCONCLUSIVE" — pilot ablation recommended

    Decision rules
    --------------
    If 95% CI intervals are provided (eta_es_lo/hi, vsa_lo/hi):
        RED    : eta_es_hi < 1.0 AND vsa_hi < 0.0
        GREEN  : eta_es_lo > 1.0 AND vsa_lo > 0.0
        YELLOW : otherwise

    Point-estimate fallback (with 5% margin on eta_ES):
        RED    : eta_es < 0.95 AND vsa < 0.0
        GREEN  : eta_es > 1.05 AND vsa > 0.0
        YELLOW : otherwise
    """
    has_ci = all(math.isfinite(v) for v in [eta_es_lo, eta_es_hi, vsa_lo, vsa_hi])
    if has_ci:
        if eta_es_hi < 1.0 and vsa_hi < 0.0:
            return "RED_HARMFUL"
        if eta_es_lo > 1.0 and vsa_lo > 0.0:
            return "GREEN_SAFE"
        return "YELLOW_INCONCLUSIVE"

    if not (math.isfinite(eta_es) and math.isfinite(vsa)):
        return "YELLOW_INCONCLUSIVE"
    if eta_es < (1.0 - margin) and vsa < 0.0:
        return "RED_HARMFUL"
    if eta_es > (1.0 + margin) and vsa > 0.0:
        return "GREEN_SAFE"
    return "YELLOW_INCONCLUSIVE"


# ---------------------------------------------------------------------------
# DI primitives  (from decoupling_metrics.py)
# ---------------------------------------------------------------------------

def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard overlap of two feature-name sets. Returns 1.0 if both empty."""
    if not a and not b:
        return 1.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def j_rand(q: float) -> float:
    """Expected Jaccard overlap of two independent random top-q sets (large-N limit)."""
    q = float(q)
    if q <= 0: return 0.0
    if q >= 1: return 1.0
    return q / (2.0 - q)


def decoupling_index(J: float, q: float) -> float:
    """
    DI(K) = 1 - J̃(K)

    where J̃ = (J - J_rand) / (1 - J_rand) rescales so random overlap → 0,
    perfect overlap → 1, giving:
      DI ≈ 0   : coupled (variance enriches important features)
      DI ≈ 1   : random-like overlap
      DI > 1   : anti-aligned (variance depletes important features)
    """
    Jr    = j_rand(q)
    denom = 1.0 - Jr
    j_tilde = (float(J) - Jr) / denom if denom > 0 else 0.0
    return 1.0 - j_tilde


@dataclass(frozen=True)
class OverlapRow:
    """Single point on a DI(K) curve."""
    k_pct: float
    k: int
    q: float
    J: float
    J_rand: float
    dJ: float
    J_tilde: float
    DI: float


def compute_overlap_curve(
    ranked_by_variance:   Sequence[str],
    ranked_by_importance: Sequence[str],
    k_pcts: Iterable[float],
) -> List[OverlapRow]:
    """
    DI curve across K% thresholds.

    Parameters
    ----------
    ranked_by_variance   : feature names sorted by variance, best first
    ranked_by_importance : feature names sorted by importance (e.g. SHAP), best first
    k_pcts               : iterable of K values (percentages, e.g. [1, 5, 10, 20])

    Returns
    -------
    List[OverlapRow]
    """
    n = min(len(ranked_by_variance), len(ranked_by_importance))
    if n <= 0:
        return []
    out: List[OverlapRow] = []
    for k_pct in k_pcts:
        q = float(k_pct) / 100.0
        k = max(1, min(int(round(q * n)), n))
        A = set(ranked_by_variance[:k])
        B = set(ranked_by_importance[:k])
        J  = jaccard(A, B)
        Jr = j_rand(q)
        dJ = J - Jr
        denom  = 1.0 - Jr
        j_tilde = (J - Jr) / denom if denom > 0 else 0.0
        DI      = 1.0 - j_tilde
        out.append(OverlapRow(k_pct=float(k_pct), k=k, q=q,
                              J=J, J_rand=Jr, dJ=dJ, J_tilde=j_tilde, DI=DI))
    return out


def rank_features(scores: Mapping[str, float], ascending: bool = False) -> List[str]:
    """Sort feature names by score. Default: descending (highest importance / variance first)."""
    return sorted(scores.keys(), key=lambda k: (float(scores[k]) * (-1 if not ascending else 1), str(k)))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction. Returns q-values aligned to input order."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out
