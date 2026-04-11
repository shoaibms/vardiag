# Contributing to vardiag

Thank you for your interest in contributing.

---

## Getting started

```bash
git clone https://github.com/shoaibms/vardiag
cd vardiag
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest
```

The full suite must pass at ≥ 90% coverage before any pull request is merged.

## Reporting bugs

Please open an issue at https://github.com/shoaibms/vardiag/issues and include:

- the exact command or code that triggered the problem
- the full error traceback
- your Python version and the output of `vardiag info`

## Suggesting changes

Open an issue to discuss the change before submitting a pull request.
For changes that affect the mathematical definitions of any metric, please
reference the relevant section of the manuscript.

## Code style

- Follow PEP 8.
- All public functions must have a NumPy-style docstring.
- New metrics must have corresponding tests in `vardiag/tests/test_vardiag.py`.

## License

By contributing you agree that your contributions will be licensed under the
MIT licence that covers this project.
