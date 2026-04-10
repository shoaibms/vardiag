# Contributing to vardiag

## Development setup

```bash
git clone https://github.com/your-lab/vardiag
cd vardiag
pip install -e ".[dev]"
```

## Running tests

```bash
pytest vardiag/tests/ -v
```

To verify the minimal install (NumPy only):

```bash
pip install -e "."   # without [dev]
python -c "from vardiag import diagnose; print('OK')"
```

## Package structure

| File | Role |
|---|---|
| `vardiag/metrics.py` | All mathematical primitives — operates on raw NumPy arrays, no IO |
| `vardiag/core.py` | Public API: `diagnose()`, `diagnose_cv()`, `scan()` |
| `vardiag/__init__.py` | Public surface — only import from here |
| `vardiag/tests/test_vardiag.py` | Full test suite (37 tests) |
| `examples/quickstart.py` | End-to-end demonstration |

## Design principles

1. **`metrics.py` must stay IO-free.** All functions take `(X, y)` numpy arrays
   and return floats or arrays. No file paths, no pandas, no project-specific logic.

2. **scipy and sklearn are optional.** Every function has a pure-NumPy fallback.
   The package must be installable with `pip install vardiag` (numpy-only) and
   still run `diagnose()` correctly.

3. **No leakage by design.** `diagnose()` documents that users should pass
   training-split data. `diagnose_cv()` enforces this by construction.

4. **All stochastic operations are seeded.** PCA uses `random_state` throughout.

## Adding a new metric

1. Implement the function in `vardiag/metrics.py` with a numpy-only fallback.
2. Export it from `vardiag/__init__.py`.
3. Add at least: a correctness test on synthetic data, a bounds test, and an
   edge-case test (empty input, NaNs).
4. Document the metric in README.md's metrics table.

## Submitting changes

Please open an issue first for any non-trivial change. Then:

1. Fork → feature branch → PR against `main`
2. All 37 existing tests must pass
3. New features must include tests
4. Update README if the public API changes
