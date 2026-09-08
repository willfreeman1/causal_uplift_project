"""Naive spend-propensity targeting, graded honestly."""

from causal_uplift.baselines import naive_propensity_policy, policy_cost, policy_mix
from causal_uplift.data import load_and_split
from causal_uplift.economics import ARM_CONTROL, ARM_MENS, budget_for_n
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
