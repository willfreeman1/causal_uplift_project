"""Step 0: import every required package and load Hillstrom."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

PACKAGES = (
    "numpy",
    "pandas",
    "matplotlib",
    "sklearn",
    "lightgbm",
    "shap",
    "econml",
    "causalml",
    "sklift",
    "ortools",
)

# pip name when it differs from the import name
_PIP_NAME = {"sklearn": "scikit-learn", "sklift": "scikit-uplift"}


def _version(import_name: str) -> str:
    pip_name = _PIP_NAME.get(import_name, import_name)
    try:
        return importlib.metadata.version(pip_name)
    except importlib.metadata.PackageNotFoundError:
        mod = importlib.import_module(import_name)
        return str(getattr(mod, "__version__", "unknown"))


def main() -> None:
    print(f"Python {sys.version}")
    for name in PACKAGES:
        importlib.import_module(name)
        print(f"  {name}: {_version(name)}")

    from causal_uplift.data import load_hillstrom

    df = load_hillstrom()
    print(f"hillstrom: {len(df):,} rows x {df.shape[1]} columns")
    print(f"  columns: {list(df.columns)}")
    print("  arm mix:")
    print(df["segment"].value_counts(normalize=True).sort_index().to_string())


if __name__ == "__main__":
    main()
