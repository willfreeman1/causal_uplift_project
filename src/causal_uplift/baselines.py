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


def cheap_until_budget(
    n: int,
    budget: float | None = None,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Send the cheap women's email until the budget is gone. Same pot as the models."""
    remaining = float(budget if budget is not None else budget_for_n(n))
    policy = np.full(n, ARM_CONTROL, dtype=object)
    rng = np.random.default_rng(random_state)
    for i in rng.permutation(n):
        cost = COSTS[ARM_WOMENS]
        if cost > remaining + 1e-9:
            break
        policy[i] = ARM_WOMENS
        remaining -= cost
    return policy


def random_until_budget(
    n: int,
    budget: float | None = None,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Walk people in random order; pick men's or women's at random if it still fits."""
    remaining = float(budget if budget is not None else budget_for_n(n))
    policy = np.full(n, ARM_CONTROL, dtype=object)
    rng = np.random.default_rng(random_state)
    paid = [ARM_WOMENS, ARM_MENS]
    for i in rng.permutation(n):
        rng.shuffle(paid)
        first, second = paid
        if COSTS[first] <= remaining + 1e-9:
            policy[i] = first
            remaining -= COSTS[first]
        elif COSTS[second] <= remaining + 1e-9:
            policy[i] = second
            remaining -= COSTS[second]
        if remaining + 1e-9 < min(COSTS[ARM_WOMENS], COSTS[ARM_MENS]):
            break
    return policy


def policy_cost(policy: np.ndarray) -> float:
    return float(sum(COSTS[a] for a in policy))


def policy_mix(policy: np.ndarray) -> pd.Series:
    return pd.Series(policy).value_counts().sort_index()
