"""Build the store-file metrics and the money chart. No fake customers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from causal_uplift.allocation import allocate_from_result, allocation_summary
from causal_uplift.baselines import (
    cheap_until_budget,
    naive_propensity_policy,
    random_until_budget,
)
from causal_uplift.data import FIGURES_DIR, PROJECT_ROOT, load_and_split
from causal_uplift.economics import (
    ARM_CONTROL,
    ARM_MENS,
    ARM_WOMENS,
    budget_for_n,
)
from causal_uplift.effects import fit_all_cate_models
from causal_uplift.grading import PolicyEvaluator, constant_policy
from causal_uplift.plots import save_money_chart

METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"

POLICY_LABELS = {
    "treat_nobody": "Send nobody",
    "random": "Random emails, $2,400",
    "cheap": "Cheap email, $2,400",
    "naive": "Naive: likely buyers, expensive email",
    "uplift_s_learner": "Combined model (S)",
    "uplift_t_learner": "Separate models (T)",
    "uplift_x_learner": "Cross model (X)",
    "uplift_causal_forest": "Causal forest",
}

CHART_ORDER = [
    "treat_nobody",
    "random",
    "cheap",
    "naive",
    "uplift_s_learner",
    "uplift_t_learner",
    "uplift_x_learner",
    "uplift_causal_forest",
]


def _product_profile(grade: pd.DataFrame, policy) -> dict:
    g = grade.copy()
    g["policy"] = policy
    rows = []
    for arm, name in (
        (ARM_CONTROL, "no_email"),
        (ARM_WOMENS, "womens_email"),
        (ARM_MENS, "mens_email"),
    ):
        sub = g[g["policy"] == arm]
        rows.append(
            {
                "arm": name,
                "n": int(len(sub)),
                "share_womens_buyers": float(sub["womens"].mean()) if len(sub) else None,
                "share_newbie": float(sub["newbie"].mean()) if len(sub) else None,
                "mean_recency": float(sub["recency"].mean()) if len(sub) else None,
                "mean_history": float(sub["history"].mean()) if len(sub) else None,
            }
        )
    return {"t_learner_plan": rows}


def write_report(
    figures_dir: Path | None = None,
    metrics_path: Path | None = None,
    forest_estimators: int = 80,
) -> pd.DataFrame:
    figures = Path(figures_dir) if figures_dir is not None else FIGURES_DIR
    figures.mkdir(parents=True, exist_ok=True)
    out_json = Path(metrics_path) if metrics_path is not None else METRICS_PATH
    out_json.parent.mkdir(parents=True, exist_ok=True)

    build, grade = load_and_split()
    budget = budget_for_n(len(grade))
    cates = fit_all_cate_models(build, grade, forest_estimators=forest_estimators)
    evaluator = PolicyEvaluator(grade, fit_outcome=True, n_folds=3, n_boot=300)

    policies = {
        "treat_nobody": constant_policy(len(grade), ARM_CONTROL),
        "random": random_until_budget(len(grade), budget),
        "cheap": cheap_until_budget(len(grade), budget),
        "naive": naive_propensity_policy(build, grade),
    }
    for name, cate in cates.items():
        policies[f"uplift_{name}"] = allocate_from_result(cate, budget)

    t_policy = policies["uplift_t_learner"]
    t_half = allocate_from_result(cates["t_learner"], budget / 2)

    rows = []
    for name in CHART_ORDER:
        policy = policies[name]
        scored = evaluator.evaluate(policy, name=name).as_dict()
        mix = allocation_summary(policy, budget)
        scored.update(
            {
                "label": POLICY_LABELS[name],
                "n_womens": mix["n_womens"],
                "n_mens": mix["n_mens"],
                "n_control": mix["n_control"],
                "spent": mix["spent"],
                "budget_constrained": name != "treat_nobody",
            }
        )
        rows.append(scored)
    table = pd.DataFrame(rows)

    half_scored = evaluator.evaluate(t_half, name="t_half").as_dict()
    half_mix = allocation_summary(t_half, budget / 2)

    save_money_chart(table, figures / "money_chart.png")

    payload = {
        "n_grade": int(len(grade)),
        "budget": float(budget),
        "policies": table.drop(columns=["label"], errors="ignore").to_dict(orient="records"),
        "product": _product_profile(grade, t_policy),
        "half_budget_t_learner": {
            "dollars": float(budget / 2),
            "n_womens": half_mix["n_womens"],
            "n_mens": half_mix["n_mens"],
            "n_control": half_mix["n_control"],
            "aipw": half_scored["aipw"],
            "aipw_ci_low": half_scored["aipw_ci_low"],
            "aipw_ci_high": half_scored["aipw_ci_high"],
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return table


if __name__ == "__main__":
    table = write_report()
    pd.set_option("display.width", 140)
    print(table[["policy", "label", "aipw", "aipw_ci_low", "aipw_ci_high"]].to_string(index=False))
    print(f"\nWrote {METRICS_PATH} and {FIGURES_DIR / 'money_chart.png'}")
