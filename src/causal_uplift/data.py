"""Load, encode, and split the Hillstrom MineThatData email experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from causal_uplift.economics import RANDOM_STATE, TRAIN_FRACTION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "hillstrom.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

EXPECTED_COLUMNS = (
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
)

NUMERIC_FEATURES = ("recency", "history", "mens", "womens", "newbie")
CATEGORICAL_FEATURES = ("zip_code", "channel")
OUTCOMES = ("visit", "conversion", "spend")


def load_hillstrom(path: Path | None = None) -> pd.DataFrame:
    """Read the committed CSV. Does not split or encode."""
    csv_path = Path(path) if path is not None else RAW_PATH
    df = pd.read_csv(csv_path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Hillstrom CSV missing columns: {missing}")
    return df


def encode_features(
    df: pd.DataFrame,
    reference_columns: pd.Index | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Numeric features plus one-hot zip_code and channel.

    `history_segment` is dropped: it is a binning of `history`.
    Dummy columns keep the original spelling, including Hillstrom's
    'Surburban' typo.

    If `reference_columns` is provided, reindex to that exact schema and
    fill unseen categories with zero. This keeps train and score matrices
    shape-compatible even when category support differs across slices.
    """
    numeric = df.loc[:, list(NUMERIC_FEATURES)].astype(float)
    dummies = pd.get_dummies(
        df.loc[:, list(CATEGORICAL_FEATURES)],
        drop_first=False,
        dtype=float,
    )
    X = pd.concat([numeric, dummies], axis=1)
    if reference_columns is not None:
        X = X.reindex(columns=list(reference_columns), fill_value=0.0)
    return X


def split_build_grade(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Half the rows to build models, half to grade. Stratified by arm."""
    build, grade = train_test_split(
        df,
        train_size=train_fraction,
        stratify=df["segment"],
        random_state=random_state,
    )
    return build.reset_index(drop=True), grade.reset_index(drop=True)


def load_and_split(
    path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_build_grade(load_hillstrom(path))
