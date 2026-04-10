"""
vardiag.cli
===========
Command-line interface for the Variance Alignment Diagnostic.

Usage
-----
    vardiag run --X features.csv --y labels.csv --k 10
    vardiag run --X features.npy --y labels.npy --k 10 --out result.json
    vardiag run --X features.csv --y labels.csv --features gene_names.txt
    vardiag info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _load_matrix(path: str) -> "np.ndarray":
    import numpy as np
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    suffix = "".join(p.suffixes).lower()
    try:
        if suffix == ".npy":
            return np.load(path, allow_pickle=False), None
        elif suffix in (".csv", ".tsv", ".txt"):
            delim = "\t" if suffix == ".tsv" else ","
            # Try numpy first (headerless CSV — most common for numeric matrices)
            try:
                arr = np.loadtxt(path, delimiter=delim, dtype=np.float32)
                return arr, None
            except ValueError:
                pass
            # Fallback: pandas (handles named-column CSVs with string headers)
            try:
                import pandas as pd
                df = pd.read_csv(path, sep=delim)
                return df.values.astype(np.float32), list(df.columns)
            except Exception as e:
                raise ValueError(f"Could not parse {path} as numeric CSV: {e}") from e
        else:
            print(f"[ERROR] Unsupported file format: {suffix}. "
                  f"Supported: .npy, .csv, .tsv, .txt", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not load {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _load_labels(path: str) -> "np.ndarray":
    import numpy as np
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    suffix = "".join(p.suffixes).lower()
    try:
        if suffix == ".npy":
            return np.load(path, allow_pickle=False)
        else:
            # Try numeric first, then string labels
            try:
                return np.loadtxt(path, dtype=int, delimiter=",")
            except ValueError:
                return np.loadtxt(path, dtype=str, delimiter=",")
    except Exception as e:
        print(f"[ERROR] Could not load labels from {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _load_feature_names(path: str) -> list:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Feature names file not found: {path}", file=sys.stderr)
        sys.exit(1)
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the VAD diagnostic on X and y files."""
    import numpy as np
    from .core import diagnose

    # Load X
    result = _load_matrix(args.X)
    if isinstance(result, tuple):
        X, inferred_names = result
    else:
        X, inferred_names = result, None

    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Load y
    y = _load_labels(args.y)

    # Load feature names
    feature_names = None
    if args.features:
        feature_names = _load_feature_names(args.features)
    elif inferred_names is not None:
        feature_names = inferred_names

    print(f"\nLoaded X: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Loaded y: {len(y)} labels  ({len(set(y.tolist()))} classes)")
    if feature_names:
        print(f"Feature names: {len(feature_names)} provided")
    print(f"Running VAD at K = {args.k}%...\n")

    try:
        result = diagnose(X, y, k_pct=args.k,
                          n_pca_components=args.pca_components,
                          random_state=args.seed)
    except (ValueError, TypeError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(result.summary())

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResult written to: {args.out}")


def cmd_info(_args: argparse.Namespace) -> None:
    """Print package version and dependency information."""
    from . import __version__
    import importlib

    print(f"\nvardiag v{__version__}")
    print("─" * 40)

    deps = {
        "numpy":        "required",
        "scipy":        "optional — provides exact Mann-Whitney U and Spearman",
        "sklearn":      "optional — provides PCA for PCLA/SAS metrics",
        "pandas":       "optional — enables CSV loading with headers in CLI",
        "matplotlib":   "optional — for plotting",
    }
    for pkg, note in deps.items():
        mod_name = "sklearn" if pkg == "sklearn" else pkg
        try:
            mod = importlib.import_module(mod_name if mod_name != "sklearn"
                                          else "sklearn")
            ver = getattr(mod, "__version__", "installed")
            status = f"✅  {ver}"
        except ImportError:
            status = "❌  not installed"
        print(f"  {pkg:<14} {status:<20}  ({note})")

    print()
    print("Tip: Install full dependencies with:")
    print('  pip install "vardiag[full]"')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vardiag",
        description="Variance Alignment Diagnostic for multi-omics feature selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vardiag run --X features.csv --y labels.csv
  vardiag run --X features.csv --y labels.csv --k 5 --out result.json
  vardiag run --X data.npy --y labels.npy --features gene_names.txt --k 10
  vardiag info
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = sub.add_parser("run", help="Run the VAD diagnostic on your data.")
    run_p.add_argument("--X", required=True,
                       help="Feature matrix file (.npy, .csv, .tsv). "
                            "Rows = samples, columns = features.")
    run_p.add_argument("--y", required=True,
                       help="Class labels file (.npy or .csv/.txt, one label per line).")
    run_p.add_argument("--features", default=None,
                       help="Optional: text file with one feature name per line.")
    run_p.add_argument("--k", type=int, default=10,
                       help="Feature budget K%% (default: 10).")
    run_p.add_argument("--pca-components", type=int, default=30,
                       help="Number of PCA components for PCLA/SAS (default: 30).")
    run_p.add_argument("--seed", type=int, default=0,
                       help="Random seed for PCA (default: 0).")
    run_p.add_argument("--out", default=None,
                       help="Optional: save result as JSON to this path.")
    run_p.set_defaults(func=cmd_run)

    # --- info ---
    info_p = sub.add_parser("info", help="Show version and dependency status.")
    info_p.set_defaults(func=cmd_info)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
