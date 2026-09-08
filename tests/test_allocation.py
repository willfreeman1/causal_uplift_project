"""Greedy allocation respects the budget and matches the exact solver on a slice."""

import numpy as np

from causal_uplift.allocation import greedy_allocate, ilp_allocate
from causal_uplift.baselines import policy_cost
from causal_uplift.data import load_and_split
from causal_uplift.economics import ARM_CONTROL, ARM_MENS, ARM_WOMENS, COSTS
from causal_uplift.effects import fit_t_learner


def test_never_assigns_nonpositive_profit():
    cate_w = np.array([0.10, 2.00, 0.00, -0.50])
    cate_m = np.array([0.40, 0.40, 1.00, 3.00])
    # person 0 women's profit 0.10-0.20 < 0; men's 0.40-0.50 < 0 -> nobody
    # person 1 women's 1.80; men's -0.10 -> women's only
    # person 2 women's -0.20; men's 0.50 -> men's
    # person 3 both negative CATE for women; men's 2.50
    policy = greedy_allocate(cate_w, cate_m, budget=10.0)
    assert policy[0] == ARM_CONTROL
    assert policy[1] == ARM_WOMENS
    assert policy[2] == ARM_MENS
    assert policy[3] == ARM_MENS


def test_budget_is_respected():
    rng = np.random.default_rng(0)
    n = 40
    cate_w = rng.normal(0.4, 0.3, n)
    cate_m = rng.normal(0.8, 0.4, n)
    budget = 3.0
    policy = greedy_allocate(cate_w, cate_m, budget)
    assert policy_cost(policy) <= budget + 1e-9
    assert (policy == ARM_CONTROL).sum() + (policy == ARM_WOMENS).sum() + (
        policy == ARM_MENS
    ).sum() == n


def test_greedy_matches_ilp_when_only_one_email_is_eligible():
    """Same cost for every item → ranking by profit-per-dollar is exact."""
    rng = np.random.default_rng(1)
    n = 20
    cate_w = rng.normal(0.6, 0.3, n)
    cate_m = np.full(n, -1.0)  # men's email never profitable
    budget = 1.6  # at most eight women's emails
    greedy = greedy_allocate(cate_w, cate_m, budget)
    exact = ilp_allocate(cate_w, cate_m, budget)
    assert np.array_equal(greedy, exact)


def test_greedy_is_close_to_ilp_on_a_small_two_email_slice():
    """0-1 knapsack can beat greedy by a hair; the gap should be tiny."""
    rng = np.random.default_rng(1)
    n = 25
    cate_w = rng.normal(0.5, 0.4, n)
    cate_m = rng.normal(0.9, 0.5, n)
    budget = 4.0
    greedy = greedy_allocate(cate_w, cate_m, budget)
    exact = ilp_allocate(cate_w, cate_m, budget)

    def objective(policy):
        total = 0.0
        for i, arm in enumerate(policy):
            if arm == ARM_WOMENS:
                total += cate_w[i] - COSTS[arm]
            elif arm == ARM_MENS:
                total += cate_m[i] - COSTS[arm]
        return total

    g, e = objective(greedy), objective(exact)
    assert e + 1e-9 >= g
    assert (e - g) / max(e, 1e-9) < 0.03
    assert policy_cost(greedy) <= budget + 1e-9
    assert policy_cost(exact) <= budget + 1e-9


def test_t_learner_plan_uses_the_cheap_and_expensive_emails():
    build, grade = load_and_split()
    cate = fit_t_learner(build, grade)
    from causal_uplift.economics import budget_for_n
    from causal_uplift.allocation import allocate_from_result

    policy = allocate_from_result(cate, budget_for_n(len(grade)))
    assert (policy == ARM_WOMENS).any()
    assert (policy == ARM_MENS).any()
    assert (policy == ARM_CONTROL).mean() > 0.5
    assert policy_cost(policy) <= budget_for_n(len(grade)) + 1e-9
