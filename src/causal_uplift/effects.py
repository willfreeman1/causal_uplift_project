"""Per-person extra spend from each email vs nothing (CATE / uplift).

Train on the build half only. Predict on the grade half only.
Headline comparison is whether average predicted effects recover the
known average effects — not a sum of the model's own guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression

from causal_uplift.data import encode_features
from causal_uplift.economics import (
    ARM_CONTROL,
    ARM_MENS,
    ARM_WOMENS,
    ARMS,
    PRIMARY_OUTCOME,
    RANDOM_STATE,
)

TREATMENT_ARMS = (ARM_WOMENS, ARM_MENS)


def _regressor(random_state: int = RANDOM_STATE) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=40,
        random_state=random_state,
        verbose=-1,
    )


def _X(df: pd.DataFrame) -> pd.DataFrame:
    return encode_features(df)


def _y(df: pd.DataFrame, y_col: str) -> np.ndarray:
    return df[y_col].to_numpy(dtype=float)


@dataclass
class CATEResult:
    name: str
    cate_womens: np.ndarray
    cate_mens: np.ndarray
    cate_womens_ci_low: np.ndarray | None = None
    cate_womens_ci_high: np.ndarray | None = None
    cate_mens_ci_low: np.ndarray | None = None
    cate_mens_ci_high: np.ndarray | None = None
    extras: dict = field(default_factory=dict)

    def mean_cates(self) -> dict[str, float]:
        return {
            ARM_WOMENS: float(self.cate_womens.mean()),
            ARM_MENS: float(self.cate_mens.mean()),
        }

    def negative_share(self) -> dict[str, float]:
        return {
            ARM_WOMENS: float((self.cate_womens < 0).mean()),
            ARM_MENS: float((self.cate_mens < 0).mean()),
        }


def _arm_matrix(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Customer features plus a one-hot for a *counterfactual* arm."""
    base = _X(df)
    dummies = {f"arm_{a}": np.zeros(len(df), dtype=float) for a in ARMS}
    dummies[f"arm_{arm}"] = np.ones(len(df), dtype=float)
    return pd.concat([base.reset_index(drop=True), pd.DataFrame(dummies)], axis=1)


def fit_s_learner(
    build: pd.DataFrame,
    grade: pd.DataFrame,
    y_col: str = PRIMARY_OUTCOME,
    random_state: int = RANDOM_STATE,
) -> CATEResult:
    """One spending model; arm is an input. Effect = predict(arm) − predict(none)."""
    X_build = _X(build).reset_index(drop=True)
    actual = pd.get_dummies(build["segment"], prefix="arm", dtype=float)
    for a in ARMS:
        col = f"arm_{a}"
        if col not in actual.columns:
            actual[col] = 0.0
    actual = actual[[f"arm_{a}" for a in ARMS]].reset_index(drop=True)
    model = _regressor(random_state)
    train_matrix = pd.concat([X_build, actual], axis=1)
    model.fit(train_matrix, _y(build, y_col))

    mu0 = model.predict(_arm_matrix(grade, ARM_CONTROL).reindex(columns=train_matrix.columns, fill_value=0.0))
    mu_w = model.predict(_arm_matrix(grade, ARM_WOMENS).reindex(columns=train_matrix.columns, fill_value=0.0))
    mu_m = model.predict(_arm_matrix(grade, ARM_MENS).reindex(columns=train_matrix.columns, fill_value=0.0))
    return CATEResult(
        name="s_learner",
        cate_womens=mu_w - mu0,
        cate_mens=mu_m - mu0,
    )


def fit_t_learner(
    build: pd.DataFrame,
    grade: pd.DataFrame,
    y_col: str = PRIMARY_OUTCOME,
    random_state: int = RANDOM_STATE,
) -> CATEResult:
    """One spending model per arm. Effect = mu_arm(x) − mu_control(x)."""
    models = {}
    X_build = _X(build)
    X_grade = _X(grade).reindex(columns=X_build.columns, fill_value=0.0)
    y = _y(build, y_col)
    for arm in ARMS:
        mask = build["segment"].to_numpy() == arm
        model = _regressor(random_state)
        model.fit(X_build.loc[mask], y[mask])
        models[arm] = model
    mu0 = models[ARM_CONTROL].predict(X_grade)
    return CATEResult(
        name="t_learner",
        cate_womens=models[ARM_WOMENS].predict(X_grade) - mu0,
        cate_mens=models[ARM_MENS].predict(X_grade) - mu0,
    )


def _fit_x_binary(
    build: pd.DataFrame,
    grade: pd.DataFrame,
    treated_arm: str,
    y_col: str,
    random_state: int,
) -> np.ndarray:
    """Künzel X-learner for one email vs nothing."""
    sub = build[build["segment"].isin([treated_arm, ARM_CONTROL])]
    X = _X(sub)
    y = _y(sub, y_col)
    treated = sub["segment"].to_numpy() == treated_arm
    control = ~treated

    mu0 = _regressor(random_state)
    mu1 = _regressor(random_state)
    mu0.fit(X.loc[control], y[control])
    mu1.fit(X.loc[treated], y[treated])

    d1 = y[treated] - mu0.predict(X.loc[treated])
    d0 = mu1.predict(X.loc[control]) - y[control]
    tau1 = _regressor(random_state)
    tau0 = _regressor(random_state)
    tau1.fit(X.loc[treated], d1)
    tau0.fit(X.loc[control], d0)

    # τ = e τ0 + (1−e) τ1, e = P(treated). RCT: e ≈ 1/2 in this subset.
    e = float(treated.mean())
    Xg = _X(grade).reindex(columns=X.columns, fill_value=0.0)
    return e * tau0.predict(Xg) + (1.0 - e) * tau1.predict(Xg)


def fit_x_learner(
    build: pd.DataFrame,
    grade: pd.DataFrame,
    y_col: str = PRIMARY_OUTCOME,
    random_state: int = RANDOM_STATE,
) -> CATEResult:
    """Two binary X-learners: women's vs nothing, and men's vs nothing."""
    return CATEResult(
        name="x_learner",
        cate_womens=_fit_x_binary(build, grade, ARM_WOMENS, y_col, random_state),
        cate_mens=_fit_x_binary(build, grade, ARM_MENS, y_col, random_state),
    )


def _fit_forest_binary(
    build: pd.DataFrame,
    grade: pd.DataFrame,
    treated_arm: str,
    y_col: str,
    random_state: int,
    n_estimators: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from econml.dml import CausalForestDML

    sub = build[build["segment"].isin([treated_arm, ARM_CONTROL])]
    Xb_df = _X(sub)
    Xb = Xb_df.to_numpy()
    yb = _y(sub, y_col)
    t = (sub["segment"].to_numpy() == treated_arm).astype(int)
    Xg = _X(grade).reindex(columns=Xb_df.columns, fill_value=0.0).to_numpy()

    est = CausalForestDML(
        model_y=_regressor(random_state),
        model_t=LogisticRegression(max_iter=500),
        discrete_treatment=True,
        n_estimators=n_estimators,
        min_samples_leaf=50,
        max_depth=6,
        random_state=random_state,
        n_jobs=1,
    )
    est.fit(yb, t, X=Xb)
    cate = np.asarray(est.effect(Xg), dtype=float).reshape(-1)
    lo, hi = est.effect_interval(Xg, alpha=0.05)
    return cate, np.asarray(lo, dtype=float).reshape(-1), np.asarray(hi, dtype=float).reshape(-1)


def fit_causal_forest(
    build: pd.DataFrame,
    grade: pd.DataFrame,
    y_col: str = PRIMARY_OUTCOME,
    random_state: int = RANDOM_STATE,
    n_estimators: int = 80,
) -> CATEResult:
    """Causal forest vs control, separately for each email. Includes 95% CIs."""
    w, w_lo, w_hi = _fit_forest_binary(build, grade, ARM_WOMENS, y_col, random_state, n_estimators)
    m, m_lo, m_hi = _fit_forest_binary(build, grade, ARM_MENS, y_col, random_state, n_estimators)
    return CATEResult(
        name="causal_forest",
        cate_womens=w,
        cate_mens=m,
        cate_womens_ci_low=w_lo,
        cate_womens_ci_high=w_hi,
        cate_mens_ci_low=m_lo,
        cate_mens_ci_high=m_hi,
    )


def fit_all_cate_models(
    build: pd.DataFrame,
    grade: pd.DataFrame,
    y_col: str = PRIMARY_OUTCOME,
    forest_estimators: int = 80,
    random_state: int = RANDOM_STATE,
) -> dict[str, CATEResult]:
    return {
        "s_learner": fit_s_learner(build, grade, y_col, random_state),
        "t_learner": fit_t_learner(build, grade, y_col, random_state),
        "x_learner": fit_x_learner(build, grade, y_col, random_state),
        "causal_forest": fit_causal_forest(
            build, grade, y_col, random_state, n_estimators=forest_estimators
        ),
    }


def measured_ate(df: pd.DataFrame, y_col: str = PRIMARY_OUTCOME) -> dict[str, float]:
    control = df.loc[df["segment"] == ARM_CONTROL, y_col]
    return {
        arm: float(df.loc[df["segment"] == arm, y_col].mean() - control.mean())
        for arm in TREATMENT_ARMS
    }


def recovery_table(
    results: dict[str, CATEResult],
    df: pd.DataFrame,
    y_col: str = PRIMARY_OUTCOME,
) -> pd.DataFrame:
    """Average predicted extra spend vs the experiment's measured average."""
    ate = measured_ate(df, y_col)
    rows = []
    for name, res in results.items():
        means = res.mean_cates()
        neg = res.negative_share()
        row: dict = {"method": name}
        for arm, short in ((ARM_WOMENS, "womens"), (ARM_MENS, "mens")):
            row[f"mean_cate_{short}"] = means[arm]
            row[f"ate_{short}"] = ate[arm]
            row[f"mean_minus_ate_{short}"] = means[arm] - ate[arm]
            row[f"pct_negative_{short}"] = 100.0 * neg[arm]
        if res.cate_mens_ci_low is not None:
            row["pct_mens_ci_above_zero"] = 100.0 * float((res.cate_mens_ci_low > 0).mean())
            row["pct_womens_ci_above_zero"] = 100.0 * float((res.cate_womens_ci_low > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_effects(
    forest_estimators: int = 80,
) -> tuple[dict[str, CATEResult], pd.DataFrame]:
    from causal_uplift.data import load_and_split

    build, grade = load_and_split()
    results = fit_all_cate_models(build, grade, forest_estimators=forest_estimators)
    table = recovery_table(results, grade)
    return results, table


if __name__ == "__main__":
    from causal_uplift.data import load_and_split

    build, grade = load_and_split()
    results, table = run_effects()
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")
    print("Primary outcome: spend ($ extra vs no email)")
    print(table.to_string(index=False))
    visit = {
        "s_learner": fit_s_learner(build, grade, y_col="visit"),
        "t_learner": fit_t_learner(build, grade, y_col="visit"),
        "x_learner": fit_x_learner(build, grade, y_col="visit"),
    }
    print("\nSupporting outcome: visit (probability-point extra vs no email)")
    print(recovery_table(visit, grade, y_col="visit").to_string(index=False))
