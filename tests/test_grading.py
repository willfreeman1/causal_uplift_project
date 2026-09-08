"""Step 2 checkpoint: honest grades of simple plans, no CATE model involved."""

import numpy as np

from causal_uplift.data import load_and_split
from causal_uplift.economics import ARM_CONTROL, ARM_MENS, ARM_WOMENS, COSTS
from causal_uplift.grading import (
    PolicyEvaluator,
    design_propensity,
    ate_minus_cost,
    constant_policy,
    ipw_scores,
    random_policy,
)


def _grade():
    _, grade = load_and_split()
    return grade


def test_treat_nobody_ipw_is_exactly_zero():
    grade = _grade()
    scores = ipw_scores(grade, constant_policy(len(grade), ARM_CONTROL))
    assert abs(float(scores.mean())) < 1e-12
    assert np.allclose(scores, 0.0)


def test_always_mens_ipw_matches_ate_minus_cost():
    grade = _grade()
    expected = ate_minus_cost(grade, ARM_MENS)
    scores = ipw_scores(grade, constant_policy(len(grade), ARM_MENS))
    assert abs(float(scores.mean()) - expected) < 1e-10


def test_always_womens_ipw_matches_ate_minus_cost():
    grade = _grade()
    expected = ate_minus_cost(grade, ARM_WOMENS)
    scores = ipw_scores(grade, constant_policy(len(grade), ARM_WOMENS))
    assert abs(float(scores.mean()) - expected) < 1e-10


def test_random_policy_returns_finite_estimate_and_ci():
    grade = _grade()
    evaluator = PolicyEvaluator(grade, fit_outcome=False, n_boot=400)
    result = evaluator.evaluate(random_policy(len(grade)), name="random")
    assert result.ipw == result.ipw
    assert result.ipw_ci_low < result.ipw_ci_high
    assert result.ipw_ci_low <= result.ipw <= result.ipw_ci_high


def test_aipw_treat_nobody_is_zero_and_always_mens_tracks_ipw():
    grade = _grade()
    evaluator = PolicyEvaluator(grade, fit_outcome=True, n_boot=200, n_folds=3)
    nobody = evaluator.evaluate(constant_policy(len(grade), ARM_CONTROL), name="nobody")
    mens = evaluator.evaluate(constant_policy(len(grade), ARM_MENS), name="mens")

    assert nobody.aipw is not None and abs(nobody.aipw) < 1e-12
    assert mens.aipw is not None
    assert mens.aipw_ci_low is not None
    assert mens.aipw_ci_high is not None
    expected = ate_minus_cost(grade, ARM_MENS)
    # AIPW should land near the known ATE − cost (IPW is exact here).
    assert abs(mens.aipw - expected) < 0.15
    assert mens.aipw_ci_low <= expected <= mens.aipw_ci_high
    assert mens.ipw_ci_low <= expected <= mens.ipw_ci_high
    # Men's email is profitable on average after its $0.50 cost.
    assert expected > 0
    assert COSTS[ARM_MENS] == 0.50


def test_policy_evaluator_reduces_folds_on_small_samples():
    grade = _grade().groupby("segment", group_keys=False).head(3).reset_index(drop=True)
    evaluator = PolicyEvaluator(grade, fit_outcome=True, n_folds=5, n_boot=50)
    result = evaluator.evaluate(constant_policy(len(grade), ARM_CONTROL), name="nobody")
    assert result.n == len(grade)
    assert result.aipw is not None


def test_policy_evaluator_defaults_to_design_propensity():
    grade = _grade()
    evaluator = PolicyEvaluator(grade, fit_outcome=False, n_boot=50)
    assert evaluator.propensity == design_propensity()
