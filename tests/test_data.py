"""Sanity checks for load, encode, and the 50/50 stratified split."""

from causal_uplift.data import encode_features, load_and_split, load_hillstrom
from causal_uplift.economics import ARMS, RANDOM_STATE, TRAIN_FRACTION


def test_hillstrom_shape_and_columns():
    df = load_hillstrom()
    assert len(df) == 64_000
    assert set(ARMS) <= set(df["segment"].unique())
    share = df["segment"].value_counts(normalize=True)
    assert (share - 1 / 3).abs().max() < 0.01


def test_encode_features_is_numeric():
    df = load_hillstrom()
    X = encode_features(df.head(20))
    assert X.shape[0] == 20
    assert "history" in X.columns
    assert "history_segment" not in X.columns
    assert any(c.startswith("zip_code_") for c in X.columns)
    assert X.isna().sum().sum() == 0


def test_encode_features_aligns_to_reference_schema():
    df = load_hillstrom()
    build = df[df["zip_code"] != "Rural"].head(1000).reset_index(drop=True)
    grade = df.head(1000).reset_index(drop=True)

    X_build = encode_features(build)
    X_grade = encode_features(grade, reference_columns=X_build.columns)

    assert list(X_grade.columns) == list(X_build.columns)
    assert X_grade.shape[1] == X_build.shape[1]


def test_split_is_half_and_stratified():
    build, grade = load_and_split()
    n = len(build) + len(grade)
    assert n == 64_000
    assert abs(len(build) / n - TRAIN_FRACTION) < 0.001
    build_share = build["segment"].value_counts(normalize=True).sort_index()
    grade_share = grade["segment"].value_counts(normalize=True).sort_index()
    assert (build_share - grade_share).abs().max() < 0.01


def test_split_is_reproducible():
    a_build, a_grade = load_and_split()
    b_build, b_grade = load_and_split()
    assert a_build.equals(b_build)
    assert a_grade.equals(b_grade)
    assert RANDOM_STATE == 42
