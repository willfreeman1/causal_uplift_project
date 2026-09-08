"""Look-before-modeling checks: balance, average effects, response pattern."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_uplift.data import FIGURES_DIR, NUMERIC_FEATURES, load_hillstrom
from causal_uplift.economics import ARM_CONTROL, ARM_MENS, ARM_WOMENS
from causal_uplift.plots import (
    save_arm_mix,
    save_ate_overview,
    save_feature_balance,
    save_visit_by_subgroup,
)

TREATMENT_ARMS = (ARM_WOMENS, ARM_MENS)
OUTCOMES = ("visit", "conversion", "spend")
Z_95 = 1.96


def arm_mix(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["segment"].value_counts().rename("n")
    share = df["segment"].value_counts(normalize=True).rename("share")
    return pd.concat([counts, share], axis=1).sort_index()


def feature_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of each numeric feature by arm, plus standardized mean difference vs control."""
    control = df[df["segment"] == ARM_CONTROL]
    rows: list[dict] = []
    for col in NUMERIC_FEATURES:
        ctrl_mean = float(control[col].mean())
        ctrl_var = float(control[col].var(ddof=1))
        row: dict = {"feature": col, ARM_CONTROL: ctrl_mean}
        for arm in TREATMENT_ARMS:
            treated = df[df["segment"] == arm]
            t_mean = float(treated[col].mean())
            pooled = np.sqrt((float(treated[col].var(ddof=1)) + ctrl_var) / 2)
            smd = (t_mean - ctrl_mean) / pooled if pooled > 0 else 0.0
            row[arm] = t_mean
            row[f"smd_{arm}"] = smd
        rows.append(row)
    return pd.DataFrame(rows)


def category_balance(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df.groupby("segment")[column]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .sort_index()
    )


def _mean_diff_ci(treated: pd.Series, control: pd.Series) -> tuple[float, float, float, float]:
    n_t, n_c = len(treated), len(control)
    ate = float(treated.mean() - control.mean())
    se = float(np.sqrt(treated.var(ddof=1) / n_t + control.var(ddof=1) / n_c))
    return ate, se, ate - Z_95 * se, ate + Z_95 * se


def average_effects(df: pd.DataFrame) -> pd.DataFrame:
    control = df[df["segment"] == ARM_CONTROL]
    rows: list[dict] = []
    for arm in TREATMENT_ARMS:
        treated = df[df["segment"] == arm]
        for outcome in OUTCOMES:
            ate, se, lo, hi = _mean_diff_ci(treated[outcome], control[outcome])
            rows.append(
                {
                    "arm": arm,
                    "outcome": outcome,
                    "ate": ate,
                    "se": se,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_treated": len(treated),
                    "n_control": len(control),
                }
            )
    return pd.DataFrame(rows)


def visit_effect_by_group(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Visit ATE vs control, sliced by a binary or categorical column."""
    rows: list[dict] = []
    for value, sub in df.groupby(column, observed=True):
        control = sub[sub["segment"] == ARM_CONTROL]
        if len(control) == 0:
            continue
        for arm in TREATMENT_ARMS:
            treated = sub[sub["segment"] == arm]
            if len(treated) == 0:
                continue
            ate, se, lo, hi = _mean_diff_ci(treated["visit"], control["visit"])
            rows.append(
                {
                    "group": column,
                    "value": value,
                    "arm": arm,
                    "ate": ate,
                    "se": se,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": len(sub),
                }
            )
    return pd.DataFrame(rows)


def visit_response_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """The slices PLAN.md section 5 calls out, plus a few more."""
    pieces = [
        visit_effect_by_group(df, "womens"),
        visit_effect_by_group(df, "mens"),
        visit_effect_by_group(df, "newbie"),
        visit_effect_by_group(df, "zip_code"),
        visit_effect_by_group(df, "channel"),
    ]
    recent = df.copy()
    recent["recent"] = (recent["recency"] <= 3).astype(int)
    pieces.append(visit_effect_by_group(recent, "recent"))
    return pd.concat(pieces, ignore_index=True)


def run_eda(df: pd.DataFrame | None = None, figures_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Compute the section-5 checks and write figures."""
    if df is None:
        df = load_hillstrom()
    out_dir = Path(figures_dir) if figures_dir is not None else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    mix = arm_mix(df)
    balance = feature_balance(df)
    effects = average_effects(df)
    pattern = visit_response_pattern(df)

    save_arm_mix(mix, out_dir / "arm_mix.png")
    save_feature_balance(balance, out_dir / "feature_balance.png")
    save_ate_overview(effects, out_dir / "ate_overview.png")
    save_visit_by_subgroup(pattern, out_dir / "visit_by_subgroup.png")

    return {
        "arm_mix": mix,
        "feature_balance": balance,
        "average_effects": effects,
        "visit_response_pattern": pattern,
    }


if __name__ == "__main__":
    results = run_eda()
    print(results["arm_mix"])
    print(results["average_effects"].to_string(index=False))
