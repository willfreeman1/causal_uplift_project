"""Budgeted send list: one email per person, one shared pot of money.

Expected profit of an email = estimated extra spend minus send cost.
Drop any pairing with profit <= 0 (never poke someone the model thinks
the email would not help). Rank the rest by profit per dollar and walk
down the list.

With one offer per person and a single budget this greedy rule is the
fractional-knapsack optimum only when fractional sends are allowed.
Our policy choices are 0/1 (send or not), so greedy is a fast heuristic.
On this dataset it is close to the exact 0-1 solution; tests check an
exact match when only one email is eligible, and a small gap otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_uplift.baselines import policy_cost, policy_mix
from causal_uplift.economics import (
    ARM_CONTROL,
    ARM_MENS,
    ARM_WOMENS,
    COSTS,
    budget_for_n,
)
from causal_uplift.effects import CATEResult, TREATMENT_ARMS


def _cates_for_arm(result: CATEResult, arm: str) -> np.ndarray:
    if arm == ARM_WOMENS:
        return result.cate_womens
    if arm == ARM_MENS:
        return result.cate_mens
    raise ValueError(arm)


def greedy_allocate(
    cate_womens: np.ndarray,
    cate_mens: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Assign each person at most one email, spending at most `budget`.

    Rank (person, email) pairs by (extra spend − cost) / cost. Skip a pair
    if the person is already assigned, profit is not positive, or the email
    no longer fits. Do not stop at the first skip — leftover cents can still
    buy the cheaper email further down the list.
    """
    n = len(cate_womens)
    if len(cate_mens) != n:
        raise ValueError("CATE vectors must be the same length")
    cates = {ARM_WOMENS: np.asarray(cate_womens, dtype=float), ARM_MENS: np.asarray(cate_mens, dtype=float)}

    pairs: list[tuple[float, int, str]] = []
    for i in range(n):
        for arm in TREATMENT_ARMS:
            profit = cates[arm][i] - COSTS[arm]
            if profit <= 0:
                continue
            vpd = profit / COSTS[arm]
            pairs.append((vpd, i, arm))
    pairs.sort(key=lambda row: row[0], reverse=True)

    policy = np.full(n, ARM_CONTROL, dtype=object)
    assigned = np.zeros(n, dtype=bool)
    remaining = float(budget)
    for _vpd, i, arm in pairs:
        if assigned[i]:
            continue
        cost = COSTS[arm]
        if cost > remaining:
            continue
        policy[i] = arm
        assigned[i] = True
        remaining -= cost
    return policy


def allocate_from_result(result: CATEResult, budget: float) -> np.ndarray:
    return greedy_allocate(result.cate_womens, result.cate_mens, budget)


def best_vpd_and_arm(
    cate_womens: np.ndarray,
    cate_mens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unconstrained best email per person by profit-per-dollar (or nothing)."""
    n = len(cate_womens)
    vpd = np.zeros(n, dtype=float)
    arm = np.full(n, ARM_CONTROL, dtype=object)
    cates = {ARM_WOMENS: np.asarray(cate_womens, dtype=float), ARM_MENS: np.asarray(cate_mens, dtype=float)}
    for i in range(n):
        best = 0.0
        chosen = ARM_CONTROL
        for a in TREATMENT_ARMS:
            profit = cates[a][i] - COSTS[a]
            if profit > 0:
                score = profit / COSTS[a]
                if score > best:
                    best = score
                    chosen = a
        vpd[i] = best
        arm[i] = chosen
    return vpd, arm


def ilp_allocate(
    cate_womens: np.ndarray,
    cate_mens: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Exact 0-1 assignment via OR-Tools. For small slices only."""
    from ortools.linear_solver import pywraplp

    n = len(cate_womens)
    cates = {ARM_WOMENS: np.asarray(cate_womens, dtype=float), ARM_MENS: np.asarray(cate_mens, dtype=float)}
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("No OR-Tools MIP solver (SCIP/CBC) available")

    x = {}
    for i in range(n):
        for arm in TREATMENT_ARMS:
            profit = cates[arm][i] - COSTS[arm]
            if profit <= 0:
                continue
            x[i, arm] = solver.BoolVar(f"x_{i}_{arm}")

    for i in range(n):
        vars_i = [x[i, arm] for arm in TREATMENT_ARMS if (i, arm) in x]
        if vars_i:
            solver.Add(sum(vars_i) <= 1)

    solver.Add(
        sum(COSTS[arm] * var for (i, arm), var in x.items()) <= budget
    )
    solver.Maximize(
        sum((cates[arm][i] - COSTS[arm]) * var for (i, arm), var in x.items())
    )
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"ILP failed with status {status}")

    policy = np.full(n, ARM_CONTROL, dtype=object)
    for (i, arm), var in x.items():
        if var.solution_value() > 0.5:
            policy[i] = arm
    return policy


def allocation_summary(policy: np.ndarray, budget: float) -> dict:
    mix = policy_mix(policy)
    return {
        "n": len(policy),
        "n_control": int((policy == ARM_CONTROL).sum()),
        "n_womens": int((policy == ARM_WOMENS).sum()),
        "n_mens": int((policy == ARM_MENS).sum()),
        "spent": policy_cost(policy),
        "budget": float(budget),
        "mix": mix,
    }


def run_allocation(forest_estimators: int = 80) -> pd.DataFrame:
    """Fit CATE models, build greedy plans, grade them honestly on the holdout."""
    from causal_uplift.baselines import naive_propensity_policy
    from causal_uplift.data import load_and_split
    from causal_uplift.effects import fit_all_cate_models
    from causal_uplift.grading import PolicyEvaluator, constant_policy

    build, grade = load_and_split()
    budget = budget_for_n(len(grade))
    results = fit_all_cate_models(build, grade, forest_estimators=forest_estimators)
    evaluator = PolicyEvaluator(grade, fit_outcome=True, n_folds=3, n_boot=300)

    policies = {
        "treat_nobody": constant_policy(len(grade), ARM_CONTROL),
        "naive": naive_propensity_policy(build, grade),
    }
    for name, cate in results.items():
        policies[f"uplift_{name}"] = allocate_from_result(cate, budget)

    rows = []
    for name, policy in policies.items():
        grade_row = evaluator.evaluate(policy, name=name).as_dict()
        summary = allocation_summary(policy, budget)
        grade_row.update(
            {
                "n_womens": summary["n_womens"],
                "n_mens": summary["n_mens"],
                "n_control": summary["n_control"],
                "spent": summary["spent"],
            }
        )
        rows.append(grade_row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = run_allocation()
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x: .4f}")
    cols = [
        "policy",
        "n_womens",
        "n_mens",
        "n_control",
        "spent",
        "ipw",
        "ipw_ci_low",
        "ipw_ci_high",
        "aipw",
        "aipw_ci_low",
        "aipw_ci_high",
    ]
    print(table[cols].to_string(index=False))
