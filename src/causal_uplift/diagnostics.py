"""Holdout diagnostics: Qini, leave-alone check, heterogeneity, half-budget."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_uplift.allocation import (
    allocate_from_result,
    allocation_summary,
    best_vpd_and_arm,
)
from causal_uplift.curves import arm_qini, cumulative_ipw, permute_qini_auc
from causal_uplift.data import FIGURES_DIR, load_and_split
from causal_uplift.economics import (
    ARM_CONTROL,
    ARM_MENS,
    ARM_WOMENS,
    PRIMARY_OUTCOME,
    budget_for_n,
)
from causal_uplift.effects import fit_causal_forest, fit_t_learner
from causal_uplift.grading import PolicyEvaluator
from causal_uplift.plots import save_cumulative_ipw, save_mix_full_vs_half, save_qini_curves


def leave_alone_check(df: pd.DataFrame, policy: np.ndarray, y_col: str = PRIMARY_OUTCOME) -> pd.DataFrame:
    """Among people the plan skipped, did getting an email anyway help or hurt?"""
    skipped = df.loc[np.asarray(policy) == ARM_CONTROL].copy()
    control = skipped[skipped["segment"] == ARM_CONTROL]
    rows = []
    for arm in (ARM_WOMENS, ARM_MENS):
        treated = skipped[skipped["segment"] == arm]
        if len(treated) == 0 or len(control) == 0:
            continue
        ate = float(treated[y_col].mean() - control[y_col].mean())
        se = float(
            np.sqrt(
                treated[y_col].var(ddof=1) / len(treated)
                + control[y_col].var(ddof=1) / len(control)
            )
        )
        rows.append(
            {
                "arm": arm,
                "outcome": y_col,
                "n_treated": len(treated),
                "n_control": len(control),
                "ate": ate,
                "se": se,
                "ci_low": ate - 1.96 * se,
                "ci_high": ate + 1.96 * se,
            }
        )
    return pd.DataFrame(rows)


def run_diagnostics(figures_dir: Path | None = None) -> dict:
    out = Path(figures_dir) if figures_dir is not None else FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)

    build, grade = load_and_split()
    budget = budget_for_n(len(grade))
    t_cate = fit_t_learner(build, grade)
    forest = fit_causal_forest(build, grade)
    policy = allocate_from_result(t_cate, budget)
    policy_half = allocate_from_result(t_cate, budget / 2)

    evaluator = PolicyEvaluator(grade, fit_outcome=True, n_folds=3, n_boot=250)
    full_grade = evaluator.evaluate(policy, name="t_full")
    half_grade = evaluator.evaluate(policy_half, name="t_half")

    visit_qini = {
        "Women's email": arm_qini(grade, t_cate.cate_womens, ARM_WOMENS, y_col="visit"),
        "Men's email": arm_qini(grade, t_cate.cate_mens, ARM_MENS, y_col="visit"),
    }
    save_qini_curves(visit_qini, out / "qini_visit.png", "Gains from targeting — visits")

    scores, best_arm = best_vpd_and_arm(t_cate.cate_womens, t_cate.cate_mens)
    overall = cumulative_ipw(grade, scores, best_arm)
    save_cumulative_ipw(overall, out / "cumulative_ipw.png")

    skipped_spend = leave_alone_check(grade, policy, y_col=PRIMARY_OUTCOME)
    skipped_visit = leave_alone_check(grade, policy, y_col="visit")

    perm_visit_mens = permute_qini_auc(grade, t_cate.cate_mens, ARM_MENS, y_col="visit", n_perm=40)
    forest_ci = {
        "pct_mens_ci_above_zero": 100.0 * float((forest.cate_mens_ci_low > 0).mean()),
        "pct_womens_ci_above_zero": 100.0 * float((forest.cate_womens_ci_low > 0).mean()),
    }

    mix_full = allocation_summary(policy, budget)
    mix_half = allocation_summary(policy_half, budget / 2)
    save_mix_full_vs_half(mix_full, mix_half, out / "mix_full_vs_half.png")

    mix_table = pd.DataFrame(
        [
            {
                "budget": "full",
                "dollars": budget,
                "n_womens": mix_full["n_womens"],
                "n_mens": mix_full["n_mens"],
                "n_control": mix_full["n_control"],
                "aipw": full_grade.aipw,
                "aipw_ci_low": full_grade.aipw_ci_low,
                "aipw_ci_high": full_grade.aipw_ci_high,
            },
            {
                "budget": "half",
                "dollars": budget / 2,
                "n_womens": mix_half["n_womens"],
                "n_mens": mix_half["n_mens"],
                "n_control": mix_half["n_control"],
                "aipw": half_grade.aipw,
                "aipw_ci_low": half_grade.aipw_ci_low,
                "aipw_ci_high": half_grade.aipw_ci_high,
            },
        ]
    )

    return {
        "mix_table": mix_table,
        "skipped_spend": skipped_spend,
        "skipped_visit": skipped_visit,
        "perm_visit_mens": perm_visit_mens,
        "forest_ci": forest_ci,
        "visit_qini_auc": {k: v["auc"] for k, v in visit_qini.items()},
        "overall_ipw": overall,
    }


if __name__ == "__main__":
    out = run_diagnostics()
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")
    print("Offer mix, full vs half budget")
    print(out["mix_table"].to_string(index=False))
    print("\nLeave-alone group: extra spend if they got an email anyway")
    print(out["skipped_spend"].to_string(index=False))
    print("\nLeave-alone group: extra visits if they got an email anyway")
    print(out["skipped_visit"].to_string(index=False))
    print("\nForest: share with CATE CI entirely above 0")
    print(out["forest_ci"])
    print("\nPermutation test (men's email, visits):", out["perm_visit_mens"])
    print("\nVisit Qini AUC:", out["visit_qini_auc"])
