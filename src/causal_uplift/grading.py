"""Honest off-policy policy-value evaluation on the randomized holdout.

A policy is just a choice of one arm per customer. We never grade it with a
CATE model's own predictions. Headline numbers come from IPW and from
cross-fitted AIPW on the grade half, using known random-assignment
probabilities by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold

from causal_uplift.data import encode_features
from causal_uplift.economics import (
    ARM_CONTROL,
    ARM_MENS,
    ARM_WOMENS,
    ARMS,
    COSTS,
    DESIGN_PROPENSITY,
    PRIMARY_OUTCOME,
    RANDOM_STATE,
)

N_BOOTSTRAP = 500
N_FOLDS = 5


def _as_policy(policy: np.ndarray | pd.Series, n: int) -> np.ndarray:
    pi = np.asarray(policy, dtype=object)
    if pi.shape != (n,):
        raise ValueError(f"policy length {pi.shape} != {n}")
    unknown = set(pi) - set(ARMS)
    if unknown:
        raise ValueError(f"policy contains unknown arms: {unknown}")
    return pi


def arm_propensity(df: pd.DataFrame) -> dict[str, float]:
    """Realized assignment shares. On Hillstrom these are ~1/3."""
    return {str(k): float(v) for k, v in df["segment"].value_counts(normalize=True).to_dict().items()}


def design_propensity() -> dict[str, float]:
    """Known assignment probabilities from the experiment design."""
    return {arm: float(p) for arm, p in DESIGN_PROPENSITY.items()}


def constant_policy(n: int, arm: str) -> np.ndarray:
    if arm not in ARMS:
        raise ValueError(arm)
    return np.full(n, arm, dtype=object)


def random_policy(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(RANDOM_STATE)
    return rng.choice(np.array(ARMS, dtype=object), size=n)


def ipw_scores(
    df: pd.DataFrame,
    policy: np.ndarray | pd.Series,
    y_col: str = PRIMARY_OUTCOME,
    propensity: dict[str, float] | None = None,
) -> np.ndarray:
    """Per-row influence scores for incremental value vs always-control.

    V(π) = E[Y(π) − cost(π) − Y(0)]. Treat-nobody is therefore exactly 0.
    Using realized arm shares makes 'always this email' match ATE − cost.
    """
    n = len(df)
    pi = _as_policy(policy, n)
    A = df["segment"].to_numpy()
    y = df[y_col].to_numpy(dtype=float)
    p = propensity or arm_propensity(df)
    p_a = np.array([p[a] for a in A])
    p_0 = p[ARM_CONTROL]
    cost = np.array([COSTS[a] for a in pi], dtype=float)

    match = (A == pi).astype(float)
    control = (A == ARM_CONTROL).astype(float)
    return match / p_a * (y - cost) - control / p_0 * y


def crossfit_outcome_mu(
    df: pd.DataFrame,
    y_col: str = PRIMARY_OUTCOME,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    """Out-of-fold E[Y | X, arm] for each arm. Fit only on the grade set."""
    X = encode_features(df).to_numpy()
    y = df[y_col].to_numpy(dtype=float)
    A = df["segment"].to_numpy()
    mu = {arm: np.zeros(len(df), dtype=float) for arm in ARMS}
    min_arm_count = int(pd.Series(A).value_counts().min())
    if min_arm_count < 2:
        for arm in ARMS:
            in_arm = A == arm
            fill = y[in_arm].mean() if in_arm.any() else y.mean()
            mu[arm][:] = fill
        return mu

    effective_folds = max(2, min(n_folds, min_arm_count))
    folds = StratifiedKFold(
        n_splits=effective_folds, shuffle=True, random_state=random_state
    )

    for train_idx, test_idx in folds.split(X, A):
        for arm in ARMS:
            in_arm = A[train_idx] == arm
            if in_arm.sum() < 20:
                mu[arm][test_idx] = y[train_idx][in_arm].mean() if in_arm.any() else y[train_idx].mean()
                continue
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
            model.fit(X[train_idx][in_arm], y[train_idx][in_arm])
            mu[arm][test_idx] = model.predict(X[test_idx])
    return mu


def aipw_scores(
    df: pd.DataFrame,
    policy: np.ndarray | pd.Series,
    mu: dict[str, np.ndarray],
    y_col: str = PRIMARY_OUTCOME,
    propensity: dict[str, float] | None = None,
) -> np.ndarray:
    """Doubly-robust incremental-value scores. Honest if either mu or p is right."""
    n = len(df)
    pi = _as_policy(policy, n)
    A = df["segment"].to_numpy()
    y = df[y_col].to_numpy(dtype=float)
    p = propensity or arm_propensity(df)
    p_a = np.array([p[a] for a in A])
    p_0 = p[ARM_CONTROL]
    cost = np.array([COSTS[a] for a in pi], dtype=float)

    mu_pi = np.fromiter((mu[a][i] for i, a in enumerate(pi)), dtype=float, count=n)
    mu_0 = mu[ARM_CONTROL]
    match = (A == pi).astype(float)
    control = (A == ARM_CONTROL).astype(float)
    return (
        mu_pi
        - cost
        - mu_0
        + match / p_a * (y - mu_pi)
        - control / p_0 * (y - mu_0)
    )


def bootstrap_mean_ci(
    scores: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    n = len(scores)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = scores[rng.integers(0, n, n)].mean()
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(scores.mean()), float(lo), float(hi)


@dataclass(frozen=True)
class PolicyValue:
    name: str
    n: int
    ipw: float
    ipw_ci_low: float
    ipw_ci_high: float
    aipw: float | None
    aipw_ci_low: float | None
    aipw_ci_high: float | None

    def as_dict(self) -> dict:
        return {
            "policy": self.name,
            "n": self.n,
            "ipw": self.ipw,
            "ipw_ci_low": self.ipw_ci_low,
            "ipw_ci_high": self.ipw_ci_high,
            "aipw": self.aipw,
            "aipw_ci_low": self.aipw_ci_low,
            "aipw_ci_high": self.aipw_ci_high,
        }


class PolicyEvaluator:
    """Fit the grade-set outcome model once, then score many policies."""

    def __init__(
        self,
        df: pd.DataFrame,
        y_col: str = PRIMARY_OUTCOME,
        fit_outcome: bool = True,
        use_design_propensity: bool = True,
        n_folds: int = N_FOLDS,
        n_boot: int = N_BOOTSTRAP,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.df = df
        self.y_col = y_col
        self.n_boot = n_boot
        self.random_state = random_state
        self.propensity = (
            design_propensity() if use_design_propensity else arm_propensity(df)
        )
        self.mu = (
            crossfit_outcome_mu(df, y_col=y_col, n_folds=n_folds, random_state=random_state)
            if fit_outcome
            else None
        )

    def evaluate(self, policy: np.ndarray | pd.Series, name: str = "policy") -> PolicyValue:
        ipw = ipw_scores(self.df, policy, y_col=self.y_col, propensity=self.propensity)
        ipw_mean, ipw_lo, ipw_hi = bootstrap_mean_ci(
            ipw, n_boot=self.n_boot, random_state=self.random_state
        )
        aipw_mean = aipw_lo = aipw_hi = None
        if self.mu is not None:
            aipw = aipw_scores(
                self.df, policy, self.mu, y_col=self.y_col, propensity=self.propensity
            )
            aipw_mean, aipw_lo, aipw_hi = bootstrap_mean_ci(
                aipw, n_boot=self.n_boot, random_state=self.random_state
            )
        return PolicyValue(
            name=name,
            n=len(self.df),
            ipw=ipw_mean,
            ipw_ci_low=ipw_lo,
            ipw_ci_high=ipw_hi,
            aipw=aipw_mean,
            aipw_ci_low=aipw_lo,
            aipw_ci_high=aipw_hi,
        )


def ate_minus_cost(df: pd.DataFrame, arm: str, y_col: str = PRIMARY_OUTCOME) -> float:
    treated = df.loc[df["segment"] == arm, y_col]
    control = df.loc[df["segment"] == ARM_CONTROL, y_col]
    return float(treated.mean() - control.mean() - COSTS[arm])


def sanity_policies(n: int, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    rng = rng or np.random.default_rng(RANDOM_STATE)
    return {
        "treat_nobody": constant_policy(n, ARM_CONTROL),
        "random": random_policy(n, rng),
        "always_womens": constant_policy(n, ARM_WOMENS),
        "always_mens": constant_policy(n, ARM_MENS),
    }


def run_sanity_checks(df: pd.DataFrame, fit_outcome: bool = True) -> pd.DataFrame:
    evaluator = PolicyEvaluator(df, fit_outcome=fit_outcome)
    rows = [
        evaluator.evaluate(policy, name=name).as_dict()
        for name, policy in sanity_policies(len(df)).items()
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from causal_uplift.data import load_and_split

    _, grade = load_and_split()
    table = run_sanity_checks(grade)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")
    print(table.to_string(index=False))
    print("\nATE minus cost on the grade set:")
    from causal_uplift.economics import ARM_MENS, ARM_WOMENS

    print(f"  always_mens   expected {ate_minus_cost(grade, ARM_MENS): .4f}")
    print(f"  always_womens expected {ate_minus_cost(grade, ARM_WOMENS): .4f}")
