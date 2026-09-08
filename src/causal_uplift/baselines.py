"""Policies that do not use per-person treatment effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from causal_uplift.data import encode_features
from causal_uplift.economics import (
    ARM_CONTROL,
    ARM_MENS,
    ARM_WOMENS,
    COSTS,
    PRIMARY_OUTCOME,
    RANDOM_STATE,
    budget_for_n,
)


def naive_propensity_policy(
    build: pd.DataFrame,
    grade: pd.DataFrame,
    y_col: str = PRIMARY_OUTCOME,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Treat likely spenders with the best offer the remaining budget can buy.

    Trained on the build half, ignoring which email was sent. This is the floor
    to beat: it funds people who would buy anyway.
    """
    model = LGBMRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=40,
        random_state=random_state,
        verbose=-1,
    )
    X_build = encode_features(build)
    X_grade = encode_features(grade, reference_columns=X_build.columns)
    model.fit(X_build, build[y_col].to_numpy(dtype=float))
    predicted = model.predict(X_grade)

    order = np.argsort(-predicted)
    policy = np.full(len(grade), ARM_CONTROL, dtype=object)
    remaining = budget_for_n(len(grade))
    best, cheap = ARM_MENS, ARM_WOMENS
    for i in order:
        if remaining >= COSTS[best]:
            policy[i] = best
            remaining -= COSTS[best]
        elif remaining >= COSTS[cheap]:
            policy[i] = cheap
            remaining -= COSTS[cheap]
        else:
            break
    return policy


def policy_cost(policy: np.ndarray) -> float:
    return float(sum(COSTS[a] for a in policy))


def policy_mix(policy: np.ndarray) -> pd.Series:
    return pd.Series(policy).value_counts().sort_index()
