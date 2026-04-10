"""
vardiag — complete test suite
==============================
11 test classes covering:

  TestEta2Features        — variance decomposition correctness & edge cases
  TestMetricsPrimitives   — eta_enrichment, VSA, alpha', f_di, DI curve
  TestClassifyZone        — zone assignment (all branches + margin boundaries)
  TestValidation          — strict input validation (all error & warning paths)
  TestDiagnose            — high-level API correctness + performance
  TestDiagnoseCV          — leakage-free CV version
  TestScan                — full DI pipeline
  TestDataModule          — bundled synthetic manuscript views
  TestCLI                 — command-line interface (npy + csv + missing file)
  TestNumericalRobustness — NaNs, constants, imbalance, wide matrices
  TestBhFdr               — multiple testing correction
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

from vardiag import (
    diagnose, diagnose_cv, scan,
    DiagnosticResult, ScanReport,
    eta2_features, eta_enrichment, vsa_mannwhitney,
    alpha_prime, f_di, classify_zone,
    decoupling_index, compute_overlap_curve, bh_fdr,
)
from vardiag.validation import validate_xy, validate_cv_folds, validate_scan_inputs
from vardiag.data import load_all_views, load_view, describe_views


# ── shared fixtures ───────────────────────────────────────────────────────────

def _coupled(seed=0, n=200, p=500):
    rng = np.random.default_rng(seed)
    y = np.array([0]*(n//2) + [1]*(n//2))
    X = rng.standard_normal((n, p)).astype(np.float32)
    X[:, :50] *= 2.0
    X[:n//2, :50] -= 3.0
    X[n//2:, :50] += 3.0
    return X, y


def _decoupled(seed=1, n=200, p=500):
    rng = np.random.default_rng(seed)
    y = np.array([0]*(n//2) + [1]*(n//2))
    X = rng.standard_normal((n, p)).astype(np.float32)
    X[:, :50] *= 10.0
    X[:n//2, 50:100] -= 1.5
    X[n//2:, 50:100] += 1.5
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
class TestEta2Features:

    def test_output_shapes(self):
        X, y = _coupled()
        vt, vb, eta2 = eta2_features(X, y)
        assert vt.shape == vb.shape == eta2.shape == (X.shape[1],)

    def test_eta2_bounded(self):
        _, _, eta2 = eta2_features(*_coupled())
        assert np.all(eta2 >= 0.0) and np.all(eta2 <= 1.0)

    def test_perfect_separation(self):
        X = np.array([[0.], [0.], [10.], [10.]])
        y = np.array([0, 0, 1, 1])
        assert eta2_features(X, y)[2][0] > 0.9

    def test_no_signal_near_zero(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 1)).astype(np.float32)
        y = np.array([0]*100 + [1]*100)
        assert eta2_features(X, y)[2][0] < 0.05

    def test_nan_column_no_crash(self):
        X, y = _coupled(p=10)
        X[:, 3] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, eta2 = eta2_features(X, y)
        assert eta2[3] == 0.0 and np.isfinite(eta2[3])

    def test_all_nan_matrix(self):
        X = np.full((20, 5), np.nan, dtype=np.float32)
        y = np.array([0]*10 + [1]*10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vt, vb, eta2 = eta2_features(X, y)
        assert np.all(np.isfinite(eta2)) and np.all(eta2 == 0.0)

    def test_empty_input(self):
        vt, vb, eta2 = eta2_features(np.zeros((0, 5), np.float32), np.array([], int))
        assert vt.shape == (5,) and np.all(vt == 0.0)

    def test_constant_column(self):
        X = np.ones((50, 3), dtype=np.float32)
        y = np.array([0]*25 + [1]*25)
        _, _, eta2 = eta2_features(X, y)
        assert np.all(np.isfinite(eta2))


# ══════════════════════════════════════════════════════════════════════════════
class TestMetricsPrimitives:

    def test_eta_es_coupled_gt1(self):
        X, y = _coupled()
        vt, _, eta2 = eta2_features(X, y)
        assert eta_enrichment(eta2, vt, k_pct=10)[0] > 1.0

    def test_eta_es_decoupled_lt1(self):
        X, y = _decoupled()
        vt, _, eta2 = eta2_features(X, y)
        assert eta_enrichment(eta2, vt, k_pct=10)[0] < 1.0

    def test_eta_enrichment_all_finite(self):
        X, y = _coupled()
        vt, _, eta2 = eta2_features(X, y)
        assert all(math.isfinite(v) for v in eta_enrichment(eta2, vt))

    def test_vsa_coupled_positive(self):
        X, y = _coupled()
        vt, _, eta2 = eta2_features(X, y)
        assert vsa_mannwhitney(eta2, vt) > 0.0

    def test_vsa_decoupled_negative(self):
        X, y = _decoupled()
        vt, _, eta2 = eta2_features(X, y)
        assert vsa_mannwhitney(eta2, vt) < 0.0

    def test_vsa_range(self):
        X, y = _coupled()
        vt, _, eta2 = eta2_features(X, y)
        assert -0.5 <= vsa_mannwhitney(eta2, vt) <= 0.5

    def test_vsa_too_few_features_nan(self):
        assert math.isnan(vsa_mannwhitney(np.array([0.5, 0.1]), np.array([1.0, 2.0])))

    def test_alpha_prime_coupled_positive(self):
        X, y = _coupled()
        vt, _, eta2 = eta2_features(X, y)
        assert alpha_prime(vt, eta2) > 0.0

    def test_f_di_coupled_lt1(self):
        X, y = _coupled()
        vt, _, eta2 = eta2_features(X, y)
        assert f_di(eta2, vt) < 1.0

    def test_f_di_decoupled_gt1(self):
        X, y = _decoupled()
        vt, _, eta2 = eta2_features(X, y)
        assert f_di(eta2, vt) > 1.0

    def test_di_random_overlap(self):
        q = 0.1
        assert abs(decoupling_index(q / (2 - q), q) - 1.0) < 1e-9

    def test_di_perfect_overlap_lt1(self):
        assert decoupling_index(1.0, 0.1) < 1.0

    def test_di_zero_overlap_gt1(self):
        assert decoupling_index(0.0, 0.1) > 1.0

    def test_overlap_curve_identical_lists(self):
        feats = [f"g{i}" for i in range(100)]
        curve = compute_overlap_curve(feats, feats, [5, 10, 20])
        assert len(curve) == 3
        assert all(r.DI < 1.0 for r in curve)

    def test_overlap_curve_disjoint_lists(self):
        a = [f"a{i}" for i in range(100)]
        b = [f"b{i}" for i in range(100)]
        assert compute_overlap_curve(a, b, [10])[0].DI > 1.0


# ══════════════════════════════════════════════════════════════════════════════
class TestClassifyZone:

    def test_green_point(self):
        assert classify_zone(1.5,  0.2)  == "GREEN_SAFE"

    def test_red_point(self):
        assert classify_zone(0.7, -0.1)  == "RED_HARMFUL"

    def test_yellow_ambiguous(self):
        assert classify_zone(1.0,  0.0)  == "YELLOW_INCONCLUSIVE"

    def test_yellow_nan(self):
        assert classify_zone(float("nan"), float("nan")) == "YELLOW_INCONCLUSIVE"

    def test_green_by_ci(self):
        assert classify_zone(1.4, 0.3, 1.1, 1.8, 0.1, 0.45) == "GREEN_SAFE"

    def test_red_by_ci(self):
        assert classify_zone(0.8, -0.1, 0.6, 0.9, -0.3, -0.05) == "RED_HARMFUL"

    def test_yellow_by_ci_mixed(self):
        assert classify_zone(1.0, 0.0, 0.8, 1.2, -0.1, 0.1) == "YELLOW_INCONCLUSIVE"

    def test_margin_below_is_red(self):
        assert classify_zone(0.94, -0.01) == "RED_HARMFUL"

    def test_margin_above_is_green(self):
        assert classify_zone(1.06, 0.01)  == "GREEN_SAFE"

    def test_margin_inside_is_yellow(self):
        assert classify_zone(1.0, 0.5)   == "YELLOW_INCONCLUSIVE"


# ══════════════════════════════════════════════════════════════════════════════
class TestValidation:

    def test_rejects_1d_X(self):
        with pytest.raises(ValueError, match="2-D"):
            validate_xy(np.zeros(100), np.zeros(100, dtype=int))

    def test_rejects_2d_y(self):
        with pytest.raises(ValueError, match="1-D"):
            validate_xy(np.zeros((100, 10)), np.zeros((100, 1), dtype=int))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same number of samples"):
            validate_xy(np.zeros((100, 10)), np.zeros(50, dtype=int))

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 4 samples"):
            validate_xy(np.zeros((3, 10)), np.array([0, 1, 0]))

    def test_rejects_single_feature(self):
        with pytest.raises(ValueError, match="at least 2 features"):
            validate_xy(np.zeros((20, 1)), np.array([0]*10 + [1]*10))

    def test_rejects_one_class(self):
        with pytest.raises(ValueError, match="2 classes"):
            validate_xy(np.zeros((20, 5)), np.zeros(20, dtype=int))

    def test_rejects_k_pct_zero(self):
        with pytest.raises(ValueError, match="k_pct"):
            validate_xy(np.zeros((20, 5)), np.array([0]*10+[1]*10), k_pct=0)

    def test_rejects_k_pct_100(self):
        with pytest.raises(ValueError, match="k_pct"):
            validate_xy(np.zeros((20, 5)), np.array([0]*10+[1]*10), k_pct=100)

    def test_warns_imbalanced_classes(self):
        X = np.zeros((30, 5), dtype=np.float32)
        y = np.array([0]*28 + [1]*2)
        with pytest.warns(UserWarning, match="Minority class"):
            validate_xy(X, y)

    def test_warns_nan_columns(self):
        X = np.zeros((20, 5), dtype=np.float32)
        X[:, 2] = np.nan
        y = np.array([0]*10 + [1]*10)
        with pytest.warns(UserWarning, match="NaN or Inf"):
            validate_xy(X, y)

    def test_warns_constant_features(self):
        X = np.ones((20, 5), dtype=np.float32)
        y = np.array([0]*10 + [1]*10)
        with pytest.warns(UserWarning, match="zero variance"):
            validate_xy(X, y)

    def test_valid_input_passes_and_coerces(self):
        X, y = _coupled()
        X_v, y_v = validate_xy(X, y)
        assert X_v.dtype == np.float32
        assert X_v.shape == X.shape

    def test_rejects_empty_cv_folds(self):
        with pytest.raises(ValueError, match="empty"):
            validate_cv_folds([], n_samples=100)

    def test_rejects_empty_fold_array(self):
        with pytest.raises(ValueError, match="empty"):
            validate_cv_folds([np.array([])], n_samples=100)

    def test_rejects_out_of_range_fold_indices(self):
        with pytest.raises(ValueError, match="out-of-range"):
            validate_cv_folds([np.array([0, 1, 999])], n_samples=100)

    def test_valid_folds_pass(self):
        result = validate_cv_folds([np.arange(50), np.arange(50, 100)], n_samples=100)
        assert len(result) == 2

    def test_rejects_wrong_feature_names_length(self):
        X, y = _coupled(p=20)
        with pytest.raises(ValueError, match="feature_names"):
            validate_scan_inputs(X, y, {"a": 1.0}, feature_names=["x"]*10, k_pct=10)

    def test_rejects_no_shap_overlap(self):
        X, y = _coupled(p=20)
        names = [f"f{i}" for i in range(20)]
        shap  = {f"g{i}": 1.0 for i in range(20)}
        with pytest.raises(ValueError, match="NO common names"):
            validate_scan_inputs(X, y, shap, feature_names=names, k_pct=10)

    def test_warns_low_shap_overlap(self):
        X, y = _coupled(p=20)
        names = [f"f{i}" for i in range(20)]
        shap  = {f"f{i}": 1.0 for i in range(5)}
        shap.update({f"g{i}": 1.0 for i in range(15)})
        with pytest.warns(UserWarning, match="25%"):
            validate_scan_inputs(X, y, shap, feature_names=names, k_pct=10)

    def test_warns_duplicate_feature_names(self):
        X, y = _coupled(p=10)
        names = [f"f{i%5}" for i in range(10)]
        shap  = {f"f{i%5}": float(i) for i in range(10)}
        with pytest.warns(UserWarning, match="duplicate"):
            validate_scan_inputs(X, y, shap, feature_names=names, k_pct=10)

    def test_rejects_non_mapping_shap(self):
        X, y = _coupled(p=10)
        with pytest.raises(TypeError, match="dict"):
            validate_scan_inputs(X, y, shap_importance=[1.0]*10,
                                 feature_names=None, k_pct=10)


# ══════════════════════════════════════════════════════════════════════════════
class TestDiagnose:

    def test_returns_diagnostic_result(self):
        assert isinstance(diagnose(*_coupled()), DiagnosticResult)

    def test_coupled_is_green(self):
        r = diagnose(*_coupled(n=200, p=500))
        assert r.zone == "GREEN_SAFE"

    def test_decoupled_is_red(self):
        r = diagnose(*_decoupled(n=200, p=500))
        assert r.zone == "RED_HARMFUL"

    def test_completes_under_5s(self):
        import time
        X, y = _coupled(n=200, p=500)
        t0 = time.perf_counter()
        diagnose(X, y)
        assert time.perf_counter() - t0 < 5.0

    def test_fields_populated(self):
        X, y = _coupled()
        r = diagnose(X, y)
        assert r.n_samples == 200 and r.n_features == 500
        assert r.n_classes == 2 and r.k_pct == 10
        assert r.elapsed_s >= 0.0

    def test_all_metrics_finite(self):
        r = diagnose(*_coupled())
        for f in ["eta_es","vsa","alpha_prime","pcla","sas","f_di"]:
            assert math.isfinite(getattr(r, f)), f"{f} not finite"

    def test_summary_contains_zone(self):
        r = diagnose(*_coupled())
        assert r.zone in r.summary()

    def test_to_dict_has_all_keys(self):
        d = diagnose(*_coupled()).to_dict()
        for k in ["zone","eta_es","vsa","alpha_prime","pcla","sas",
                  "f_di","k_pct","n_features","n_samples","elapsed_s"]:
            assert k in d

    def test_rejects_1d_X(self):
        with pytest.raises(ValueError):
            diagnose(np.zeros(100), np.zeros(100, dtype=int))

    def test_rejects_mismatched_samples(self):
        with pytest.raises(ValueError):
            diagnose(np.zeros((100,10)), np.zeros(50, dtype=int))

    def test_rejects_one_class(self):
        with pytest.raises(ValueError):
            diagnose(np.zeros((20,10)), np.zeros(20, dtype=int))

    def test_all_k_pcts_run(self):
        X, y = _decoupled()
        for k in [1, 5, 10, 20]:
            r = diagnose(X, y, k_pct=k)
            assert r.zone in ("RED_HARMFUL","GREEN_SAFE","YELLOW_INCONCLUSIVE")

    def test_multiclass(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((150, 200)).astype(np.float32)
        y = np.array([0]*50 + [1]*50 + [2]*50)
        assert diagnose(X, y).n_classes == 3

    def test_float64_input_accepted(self):
        X, y = _coupled()
        assert isinstance(diagnose(X.astype(np.float64), y), DiagnosticResult)

    def test_list_input_accepted(self):
        X, y = _coupled(n=50, p=50)
        assert isinstance(diagnose(X.tolist(), y.tolist()), DiagnosticResult)


# ══════════════════════════════════════════════════════════════════════════════
class TestDiagnoseCV:

    def _folds(self, n, k=5):
        idx = np.arange(n)
        parts = np.array_split(idx, k)
        return [np.concatenate([parts[j] for j in range(k) if j != i])
                for i in range(k)]

    def test_returns_result(self):
        X, y = _coupled(n=100, p=200)
        assert isinstance(diagnose_cv(X, y, cv_folds=self._folds(100)), DiagnosticResult)

    def test_zone_matches_single(self):
        X, y = _decoupled(n=200, p=300)
        assert diagnose(X, y).zone == diagnose_cv(X, y, self._folds(200)).zone

    def test_rejects_empty_folds(self):
        X, y = _coupled()
        with pytest.raises(ValueError):
            diagnose_cv(X, y, cv_folds=[])

    def test_rejects_out_of_range_indices(self):
        X, y = _coupled(n=100, p=50)
        with pytest.raises(ValueError, match="out-of-range"):
            diagnose_cv(X, y, cv_folds=[np.array([0,1,9999])])

    def test_sklearn_folds(self):
        pytest.importorskip("sklearn")
        from sklearn.model_selection import StratifiedKFold
        X, y = _coupled(n=100, p=200)
        folds = [tr for tr, _ in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)]
        r = diagnose_cv(X, y, cv_folds=folds)
        assert r.zone in ("GREEN_SAFE","YELLOW_INCONCLUSIVE","RED_HARMFUL")


# ══════════════════════════════════════════════════════════════════════════════
class TestScan:

    def _shap(self, names, rng):
        return {n: float(rng.random()) for n in names}

    def test_returns_scan_report(self):
        rng = np.random.default_rng(0)
        X, y = _coupled(n=100, p=200)
        names = [f"g{i}" for i in range(200)]
        r = scan(X, y, shap_importance=self._shap(names, rng), feature_names=names)
        assert isinstance(r, ScanReport)

    def test_di_curve_length(self):
        rng = np.random.default_rng(1)
        X, y = _coupled(n=100, p=200)
        names = [f"g{i}" for i in range(200)]
        r = scan(X, y, shap_importance=self._shap(names, rng),
                 feature_names=names, k_pcts=[1,5,10,20])
        assert len(r.di_curve) == 4

    def test_hidden_fraction_bounded(self):
        rng = np.random.default_rng(2)
        X, y = _coupled(n=100, p=200)
        names = [f"g{i}" for i in range(200)]
        r = scan(X, y, shap_importance=self._shap(names, rng), feature_names=names)
        assert 0.0 <= r.hidden_biomarker_fraction <= 1.0

    def test_jaccard_bounded(self):
        rng = np.random.default_rng(3)
        X, y = _coupled(n=100, p=200)
        names = [f"g{i}" for i in range(200)]
        r = scan(X, y, shap_importance=self._shap(names, rng), feature_names=names)
        assert 0.0 <= r.gene_level_jaccard <= 1.0

    def test_summary_has_di_curve(self):
        rng = np.random.default_rng(4)
        X, y = _coupled(n=100, p=200)
        names = [f"g{i}" for i in range(200)]
        assert "DI Curve" in scan(X, y, shap_importance=self._shap(names, rng),
                                   feature_names=names).summary()

    def test_rejects_no_shap_overlap(self):
        X, y = _coupled(p=20)
        names = [f"f{i}" for i in range(20)]
        with pytest.raises(ValueError, match="NO common names"):
            scan(X, y, shap_importance={f"z{i}": 1.0 for i in range(20)},
                 feature_names=names)

    def test_auto_feature_names(self):
        rng = np.random.default_rng(5)
        X, y = _coupled(n=80, p=100)
        shap = {f"f{i}": float(rng.random()) for i in range(100)}
        assert isinstance(scan(X, y, shap_importance=shap), ScanReport)


# ══════════════════════════════════════════════════════════════════════════════
class TestDataModule:

    def test_load_all_views_count(self):
        assert len(load_all_views(seed=42)) == 4

    def test_all_views_correct_zone(self):
        for name, view in load_all_views(seed=42).items():
            r = diagnose(view.X, view.y, k_pct=10)
            assert r.zone == view.expected_zone, \
                f"{name}: expected {view.expected_zone}, got {r.zone}"

    def test_brca_shape(self):
        v = load_view("brca_methylation")
        assert v.X.shape == (312, 11_189) and v.y.shape == (312,)

    def test_ibd_shape(self):
        assert load_view("ibd_mgx").X.shape == (155, 368)

    def test_unknown_view_raises(self):
        with pytest.raises(ValueError, match="Unknown view"):
            load_view("nonexistent")

    def test_describe_runs(self, capsys):
        describe_views()
        out = capsys.readouterr().out
        assert "brca_methylation" in out and "ANTI_ALIGNED" in out

    def test_feature_names_length_matches(self):
        v = load_view("brca_methylation")
        assert len(v.feature_names) == v.X.shape[1]

    def test_hidden_biomarker_names_present(self):
        v = load_view("brca_methylation")
        assert "cg_MIA_proxy" in v.feature_names
        assert "cg_CHI3L1_proxy" in v.feature_names


# ══════════════════════════════════════════════════════════════════════════════
class TestCLI:

    def test_info_command(self):
        from vardiag.cli import build_parser
        import io, contextlib
        args = build_parser().parse_args(["info"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            args.func(args)
        assert "vardiag" in buf.getvalue()

    def test_run_npy(self, tmp_path):
        X, y = _coupled(n=60, p=100)
        xp, yp, op = tmp_path/"X.npy", tmp_path/"y.npy", tmp_path/"out.json"
        np.save(xp, X); np.save(yp, y)
        from vardiag.cli import build_parser
        import io, contextlib
        args = build_parser().parse_args(
            ["run","--X",str(xp),"--y",str(yp),"--k","10","--out",str(op)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            args.func(args)
        assert "Zone" in buf.getvalue()
        d = json.loads(op.read_text())
        assert d["zone"] in ("GREEN_SAFE","RED_HARMFUL","YELLOW_INCONCLUSIVE")

    def test_run_csv(self, tmp_path):
        X, y = _coupled(n=40, p=50)
        xp, yp = tmp_path/"X.csv", tmp_path/"y.csv"
        # Write without index/header so row counts stay consistent
        np.savetxt(xp, X, delimiter=",", fmt="%.6f")
        np.savetxt(yp, y.reshape(-1, 1), delimiter=",", fmt="%d")
        from vardiag.cli import build_parser
        import io, contextlib
        args = build_parser().parse_args(["run","--X",str(xp),"--y",str(yp)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            args.func(args)
        assert "Zone" in buf.getvalue()

    def test_run_missing_file_exits(self, tmp_path):
        yp = tmp_path/"y.npy"
        np.save(yp, np.array([0]*10+[1]*10))
        from vardiag.cli import build_parser, cmd_run
        args = build_parser().parse_args(
            ["run","--X",str(tmp_path/"nope.npy"),"--y",str(yp)])
        with pytest.raises(SystemExit):
            cmd_run(args)

    def test_run_tsv(self, tmp_path):
        """TSV path exercises the pandas-header fallback branch."""
        X, y = _coupled(n=40, p=50)
        xp, yp = tmp_path/"X.tsv", tmp_path/"y.csv"
        np.savetxt(xp, X, delimiter="\t", fmt="%.6f")
        np.savetxt(yp, y.reshape(-1, 1), delimiter=",", fmt="%d")
        from vardiag.cli import build_parser
        import io, contextlib
        args = build_parser().parse_args(["run","--X",str(xp),"--y",str(yp)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            args.func(args)
        assert "Zone" in buf.getvalue()

    def test_run_feature_names_file(self, tmp_path):
        """--features file loader path."""
        X, y = _coupled(n=60, p=100)
        xp = tmp_path/"X.npy"
        yp = tmp_path/"y.npy"
        fp = tmp_path/"features.txt"
        np.save(xp, X)
        np.save(yp, y)
        fp.write_text("\n".join(f"gene_{i}" for i in range(100)))
        from vardiag.cli import build_parser
        import io, contextlib
        args = build_parser().parse_args(
            ["run","--X",str(xp),"--y",str(yp),"--features",str(fp)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            args.func(args)
        out = buf.getvalue()
        assert "Zone" in out
        assert "Feature names: 100 provided" in out


# ══════════════════════════════════════════════════════════════════════════════
class TestNumericalRobustness:

    def test_all_nan_X_runs(self):
        X = np.full((20, 10), np.nan, dtype=np.float32)
        y = np.array([0]*10 + [1]*10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = diagnose(X, y)
        assert r.zone in ("GREEN_SAFE","RED_HARMFUL","YELLOW_INCONCLUSIVE")

    def test_constant_features_runs(self):
        X = np.ones((40, 20), dtype=np.float32)
        y = np.array([0]*20 + [1]*20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = diagnose(X, y)
        assert isinstance(r, DiagnosticResult)

    def test_heavily_imbalanced(self):
        rng = np.random.default_rng(9)
        X = rng.standard_normal((100, 50)).astype(np.float32)
        y = np.array([0]*95 + [1]*5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert isinstance(diagnose(X, y), DiagnosticResult)

    def test_many_classes(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((100, 50)).astype(np.float32)
        y = np.repeat(np.arange(10), 10)
        assert diagnose(X, y).n_classes == 10

    def test_wide_matrix_p_gt_n(self):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((30, 5000)).astype(np.float32)
        y = np.array([0]*15 + [1]*15)
        assert isinstance(diagnose(X, y), DiagnosticResult)

    def test_no_runtime_warnings_on_clean_data(self):
        X, y = _coupled()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diagnose(X, y)
        rw = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(rw) == 0, f"Unexpected RuntimeWarnings: {[str(x.message) for x in rw]}"


# ══════════════════════════════════════════════════════════════════════════════
class TestBhFdr:

    def test_shape(self):
        p = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        assert bh_fdr(p).shape == p.shape

    def test_bounded(self):
        q = bh_fdr(np.array([0.001, 0.01, 0.05, 0.1, 0.5]))
        assert np.all(q >= 0.0) and np.all(q <= 1.0)

    def test_monotone(self):
        q = bh_fdr(np.array([0.001, 0.01, 0.05, 0.1, 0.5]))
        assert np.all(np.diff(q) >= -1e-12)

    def test_single_value(self):
        assert abs(float(bh_fdr(np.array([0.03]))[0]) - 0.03) < 1e-9

    def test_empty(self):
        assert bh_fdr(np.array([])).size == 0

    def test_all_ones(self):
        assert np.allclose(bh_fdr(np.ones(10)), 1.0)
