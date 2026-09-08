"""Gains-from-targeting (Qini) curves on the randomized holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score, qini_curve

from causal_uplift.economics import ARM_CONTROL, ARM_MENS, ARM_WOMENS, PRIMARY_OUTCOME
from causal_uplift.grading import ipw_scores


def binary_subset(df: pd.DataFrame, treated_arm: str) -> pd.DataFrame:
    return df[df["segment"].isin([treated_arm, ARM_CONTROL])].copy()


def arm_qini(
    df: pd.DataFrame,
    cate: np.ndarray,
    treated_arm: str,
    y_col: str = PRIMARY_OUTCOME,
) -> dict:
    """One email vs nothing. `cate` must align with `df` (full grade set)."""
    mask = df["segment"].isin([treated_arm, ARM_CONTROL]).to_numpy()
    y = df.loc[mask, y_col].to_numpy(dtype=float)
    t = (df.loc[mask, "segment"].to_numpy() == treated_arm).astype(int)
    uplift = np.asarray(cate)[mask]
    x, qini = qini_curve(y, uplift, t)
    auc = float(qini_auc_score(y, uplift, t))
    return {"x": np.asarray(x, dtype=float), "qini": np.asarray(qini, dtype=float), "auc": auc}


def cumulative_ipw(
    df: pd.DataFrame,
    scores: np.ndarray,
    arm_if_treated: np.ndarray,
    n_points: int = 21,
    y_col: str = PRIMARY_OUTCOME,
) -> pd.DataFrame:
    """Sort by score; at each fraction, treat the top slice with their chosen email."""
    n = len(df)
    order = np.argsort(-np.asarray(scores, dtype=float))
    fractions = np.linspace(0.0, 1.0, n_points)
    rows = []
    for frac in fractions:
        k = int(round(frac * n))
        policy = np.full(n, ARM_CONTROL, dtype=object)
        if k > 0:
            top = order[:k]
            policy[top] = np.asarray(arm_if_treated, dtype=object)[top]
        value = float(ipw_scores(df, policy, y_col=y_col).mean())
        rows.append({"fraction": frac, "n_treated": int((policy != ARM_CONTROL).sum()), "ipw": value})
    return pd.DataFrame(rows)


def permute_qini_auc(
    df: pd.DataFrame,
    cate: np.ndarray,
    treated_arm: str,
    y_col: str,
    n_perm: int = 50,
    random_state: int = 42,
) -> dict:
    """Shuffle scores vs outcomes. Real AUC should sit above most shuffled AUCs if signal is real."""
    observed = arm_qini(df, cate, treated_arm, y_col=y_col)["auc"]
    rng = np.random.default_rng(random_state)
    shuffled = []
    for _ in range(n_perm):
        perm = rng.permutation(cate)
        shuffled.append(arm_qini(df, perm, treated_arm, y_col=y_col)["auc"])
    shuffled = np.asarray(shuffled, dtype=float)
    return {
        "observed_auc": float(observed),
        "perm_mean": float(shuffled.mean()),
        "perm_95": float(np.quantile(shuffled, 0.95)),
        "share_perm_below_observed": float((shuffled < observed).mean()),
    }
