"""Naive spend-propensity targeting, graded honestly."""

from causal_uplift.baselines import (
    cheap_until_budget,
    naive_propensity_policy,
    policy_cost,
    policy_mix,
    random_until_budget,
)
from causal_uplift.data import load_and_split
from causal_uplift.economics import ARM_CONTROL, ARM_MENS, ARM_WOMENS, COSTS, budget_for_n
from causal_uplift.grading import PolicyEvaluator


def test_naive_policy_respects_budget():
    build, grade = load_and_split()
    policy = naive_propensity_policy(build, grade)
    budget = budget_for_n(len(grade))
    spent = policy_cost(policy)
    assert spent <= budget + 1e-9
    # Scarcity: we should not be able to treat everyone with Men's.
    assert (policy == ARM_MENS).mean() < 0.25
    assert (policy == ARM_CONTROL).any()


def test_naive_policy_is_gradable():
    build, grade = load_and_split()
    policy = naive_propensity_policy(build, grade)
    result = PolicyEvaluator(grade, fit_outcome=False, n_boot=200).evaluate(
        policy, name="naive"
    )
    mix = policy_mix(policy)
    assert result.n == len(grade)
    assert ARM_MENS in mix.index
    # A finite number with a CI — not a circular Σ predicted uplift.
    assert result.ipw_ci_low < result.ipw_ci_high


def test_cheap_and_random_respect_the_same_budget():
    _, grade = load_and_split()
    n = len(grade)
    budget = budget_for_n(n)
    cheap = cheap_until_budget(n, budget)
    random = random_until_budget(n, budget)

    assert policy_cost(cheap) <= budget + 1e-9
    assert policy_cost(random) <= budget + 1e-9
    assert (cheap == ARM_WOMENS).sum() == round(budget / COSTS[ARM_WOMENS])
    assert (cheap == ARM_MENS).sum() == 0
    assert (random == ARM_WOMENS).any()
    assert (random == ARM_MENS).any()
    assert (random == ARM_CONTROL).any()
