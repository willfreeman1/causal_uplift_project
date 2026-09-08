"""Made-up customers with known extra-spend rules, including sleeping dogs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_uplift.allocation import allocate_from_result, greedy_allocate
from causal_uplift.data import split_build_grade
from causal_uplift.economics import (
    ARM_CONTROL,
    ARM_MENS,
    ARM_WOMENS,
    ARMS,
    COSTS,
    RANDOM_STATE,
    budget_for_n,
)
from causal_uplift.effects import fit_t_learner
from causal_uplift.grading import PolicyEvaluator


def make_synthetic(n: int = 40_000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Known extra spend for each email, plus a group that is hurt by contact."""
    rng = np.random.default_rng(random_state)
    recency = rng.integers(1, 13, n)
    history = rng.lognormal(mean=5.0, sigma=0.8, size=n)
    newbie = rng.binomial(1, 0.4, n)
    womens_buyer = rng.binomial(1, 0.5, n)
    zip_code = rng.choice(["Urban", "Surburban", "Rural"], n)
    channel = rng.choice(["Web", "Phone", "Multichannel"], n)

    # Groups the models can actually see (functions of features, not hidden labels).
    sleeping = ((newbie == 1) & (recency >= 9)).astype(int)
    expensive_only = ((womens_buyer == 0) & (history > np.quantile(history, 0.82))).astype(int)

    # True extra dollars vs nothing.
    tau_w = np.where(
        sleeping == 1,
        -1.8,
        np.where(womens_buyer == 1, 1.4, np.where(expensive_only == 1, 0.05, 0.25)),
    )
    # Men's vpd on the expensive-only group is higher than women's vpd on
    # women's-item buyers, so the perfect plan uses both emails.
    tau_m = np.where(
        sleeping == 1,
        -1.5,
        np.where(expensive_only == 1, 6.0, np.where(womens_buyer == 1, 1.5, 0.9)),
    )

    y0 = 0.4 + 0.002 * history + 0.2 * newbie
    segment = rng.choice(np.array(ARMS, dtype=object), size=n)
    tau_obs = np.where(segment == ARM_WOMENS, tau_w, np.where(segment == ARM_MENS, tau_m, 0.0))
    spend = np.clip(y0 + tau_obs + rng.normal(0.0, 0.8, n), 0.0, None)

    df = pd.DataFrame(
        {
            "recency": recency,
            "history_segment": "2) $100 - $200",
            "history": history,
            "mens": (1 - womens_buyer),
            "womens": womens_buyer,
            "zip_code": zip_code,
            "newbie": newbie,
            "channel": channel,
            "segment": segment,
            "visit": (spend > 0.8).astype(int),
            "conversion": (spend > 1.5).astype(int),
            "spend": spend,
            "true_tau_w": tau_w,
            "true_tau_m": tau_m,
            "sleeping": sleeping,
            "expensive_only": expensive_only,
        }
    )
    return df


def true_policy_value(df: pd.DataFrame, policy: np.ndarray) -> float:
    """Exact extra profit from the known extras, no noise."""
    tau_w = df["true_tau_w"].to_numpy()
    tau_m = df["true_tau_m"].to_numpy()
    value = np.zeros(len(df))
    pi = np.asarray(policy, dtype=object)
    mens = pi == ARM_MENS
    womens = pi == ARM_WOMENS
    value[womens] = tau_w[womens] - COSTS[ARM_WOMENS]
    value[mens] = tau_m[mens] - COSTS[ARM_MENS]
    return float(value.mean())


def perfect_policy(df: pd.DataFrame, budget: float) -> np.ndarray:
    return greedy_allocate(df["true_tau_w"].to_numpy(), df["true_tau_m"].to_numpy(), budget)


def run_simulation(n: int = 40_000, random_state: int = RANDOM_STATE) -> dict:
    df = make_synthetic(n=n, random_state=random_state)
    build, grade = split_build_grade(df)
    budget = budget_for_n(len(grade))

    perfect = perfect_policy(grade, budget)
    v_perfect = true_policy_value(grade, perfect)

    cate = fit_t_learner(build, grade)
    pipeline = allocate_from_result(cate, budget)
    v_true_pipeline = true_policy_value(grade, pipeline)
    capture = v_true_pipeline / v_perfect if v_perfect > 0 else np.nan

    evaluator = PolicyEvaluator(grade, fit_outcome=True, n_folds=3, n_boot=200)
    graded = evaluator.evaluate(pipeline, name="pipeline")

    dogs = grade["sleeping"].to_numpy() == 1
    dogs_untreated = float((pipeline[dogs] == ARM_CONTROL).mean()) if dogs.any() else np.nan
    pred_tau_w_dogs = float(cate.cate_womens[dogs].mean()) if dogs.any() else np.nan
    pred_tau_m_dogs = float(cate.cate_mens[dogs].mean()) if dogs.any() else np.nan

    return {
        "n_grade": len(grade),
        "v_perfect": v_perfect,
        "v_true_pipeline": v_true_pipeline,
        "capture": float(capture),
        "aipw": graded.aipw,
        "aipw_ci_low": graded.aipw_ci_low,
        "aipw_ci_high": graded.aipw_ci_high,
        "ipw": graded.ipw,
        "dogs_untreated": dogs_untreated,
        "pred_tau_w_dogs": pred_tau_w_dogs,
        "pred_tau_m_dogs": pred_tau_m_dogs,
        "n_pipeline_womens": int((pipeline == ARM_WOMENS).sum()),
        "n_pipeline_mens": int((pipeline == ARM_MENS).sum()),
        "n_perfect_womens": int((perfect == ARM_WOMENS).sum()),
        "n_perfect_mens": int((perfect == ARM_MENS).sum()),
    }


if __name__ == "__main__":
    out = run_simulation()
    print("Simulation (known rules)")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nHeadline: captured {100 * out['capture']:.1f}% of perfect extra profit.")
    print(
        f"Honest AIPW {out['aipw']:.3f} vs true pipeline value {out['v_true_pipeline']:.3f} "
        f"(CI {out['aipw_ci_low']:.3f} to {out['aipw_ci_high']:.3f})."
    )
    print(f"Sleeping dogs left alone: {100 * out['dogs_untreated']:.1f}%.")
