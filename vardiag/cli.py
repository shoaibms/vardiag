"""
vardiag.cli
===========

Command-line interface for the Variance Alignment Diagnostic.

Examples
--------
    vardiag run --X features.csv --y labels.csv --k 10
    vardiag run --X features.tsv --y labels.txt --k 10 --out result.json
    vardiag run --X features.npy --y labels.npy
    vardiag info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


def _load_matrix(path: str):
    import numpy as np

    p = Path(path)
    if not p.exists():
        _fail(f"File not found: {path}")

    suffix = p.suffix.lower()

    try:
        if suffix == ".npy":
            arr = np.load(p, allow_pickle=False)
            return np.asarray(arr, dtype=np.float32), None

        if suffix == ".csv":
            try:
                arr = np.loadtxt(p, delimiter=",", dtype=np.float32)
                return np.asarray(arr, dtype=np.float32), None
            except ValueError:
                import pandas as pd
                df = pd.read_csv(p)
                return df.to_numpy(dtype=np.float32), list(df.columns)

        if suffix == ".tsv":
            try:
                arr = np.loadtxt(p, delimiter="\t", dtype=np.float32)
                return np.asarray(arr, dtype=np.float32), None
            except ValueError:
                import pandas as pd
                df = pd.read_csv(p, sep="\t")
                return df.to_numpy(dtype=np.float32), list(df.columns)

        if suffix == ".txt":
            # whitespace-delimited numeric matrix, no header support
            arr = np.loadtxt(p, delimiter=None, dtype=np.float32)
            return np.asarray(arr, dtype=np.float32), None

        _fail(
            f"Unsupported matrix format: {suffix}. "
            "Supported: .npy, .csv, .tsv, .txt"
        )

    except SystemExit:
        raise
    except Exception as e:
        _fail(f"Could not load matrix from {path}: {e}")


def _load_labels(path: str):
    import numpy as np

    p = Path(path)
    if not p.exists():
        _fail(f"File not found: {path}")

    suffix = p.suffix.lower()

    try:
        if suffix == ".npy":
            return np.load(p, allow_pickle=False)

        if suffix == ".csv":
            try:
                return np.loadtxt(p, dtype=int, delimiter=",")
            except ValueError:
                return np.loadtxt(p, dtype=str, delimiter=",")

        if suffix == ".tsv":
            try:
                return np.loadtxt(p, dtype=int, delimiter="\t")
            except ValueError:
                return np.loadtxt(p, dtype=str, delimiter="\t")

        if suffix == ".txt":
            lines = [
                line.strip()
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            try:
                return np.asarray([int(x) for x in lines])
            except ValueError:
                return np.asarray(lines, dtype=str)

        _fail(
            f"Unsupported label format: {suffix}. "
            "Supported: .npy, .csv, .tsv, .txt"
        )

    except SystemExit:
        raise
    except Exception as e:
        _fail(f"Could not load labels from {path}: {e}")


def _load_feature_names(path: str) -> list:
    p = Path(path)
    if not p.exists():
        _fail(f"Feature names file not found: {path}")
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def cmd_run(args: argparse.Namespace) -> None:
    from .core import diagnose

    X, inferred_names = _load_matrix(args.X)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    y = _load_labels(args.y)

    feature_names = None
    if args.features:
        feature_names = _load_feature_names(args.features)
        if len(feature_names) != X.shape[1]:
            _fail(
                f"Feature-name count mismatch: got {len(feature_names)} names "
                f"but X has {X.shape[1]} features. "
                f"The file must contain exactly one name per line per feature column."
            )
    elif inferred_names is not None:
        feature_names = inferred_names

    print(f"\nLoaded X: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Loaded y: {len(y)} labels  ({len(set(y.tolist()))} classes)")
    if feature_names:
        print(f"Feature names: {len(feature_names)} provided")
    print(f"Running VAD at K = {args.k}%...\n")

    try:
        result = diagnose(
            X,
            y,
            k_pct=args.k,
            n_pca_components=args.pca_components,
            random_state=args.seed,
            margin=args.margin,
        )
    except (ValueError, TypeError) as e:
        _fail(str(e))

    print(result.summary())

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResult written to: {args.out}")


def cmd_info(_args: argparse.Namespace) -> None:
    import importlib

    from . import __version__

    print(f"\nvardiag v{__version__}")
    print("─" * 40)

    deps = {
        "numpy":      "required",
        "scipy":      "optional — exact Mann-Whitney U and Spearman",
        "sklearn":    "optional — PCA for PCLA/SAS metrics",
        "pandas":     "optional — CSV/TSV files with headers",
        "matplotlib": "optional — plotting examples",
    }

    for pkg, note in deps.items():
        mod_name = "sklearn" if pkg == "sklearn" else pkg
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed")
            status = f"OK  {ver}"
        except ImportError:
            status = "NOT INSTALLED"
        print(f"  {pkg:<14} {status:<28}  ({note})")

    print()
    print('Tip: install the full dependency set with: pip install "vardiag[full]"')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vardiag",
        description=(
            "Variance Alignment Diagnostic for feature-filter safety assessment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vardiag run --X features.csv --y labels.csv
  vardiag run --X features.tsv --y labels.txt --k 5 --out result.json
  vardiag run --X features.npy --y labels.npy --margin 0.05
  vardiag info
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run",
        help="Run the VAD diagnostic on one matrix/label pair.",
    )
    run_p.add_argument(
        "--X", required=True,
        help="Feature matrix: .npy, .csv, .tsv, or whitespace-delimited .txt",
    )
    run_p.add_argument(
        "--y", required=True,
        help="Labels: .npy, .csv, .tsv, or one-label-per-line .txt",
    )
    run_p.add_argument(
        "--features", default=None,
        help="Optional text file with one feature name per line.",
    )
    run_p.add_argument(
        "--k", type=int, default=10,
        help="Feature budget percentage K (default: 10)",
    )
    run_p.add_argument(
        "--pca-components", type=int, default=30,
        help="Number of PCA components for PCLA/SAS (default: 30)",
    )
    run_p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for PCA (default: 0)",
    )
    run_p.add_argument(
        "--margin", type=float, default=0.05,
        help="Conservative decision margin around eta_ES = 1 (default: 0.05)",
    )
    run_p.add_argument(
        "--out", default=None,
        help="Optional JSON output path",
    )
    run_p.set_defaults(func=cmd_run)

    info_p = sub.add_parser("info", help="Show version and dependency status.")
    info_p.set_defaults(func=cmd_info)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
