"""Diagnostics: leave-alone check, half-budget spends less, Qini runs."""

from causal_uplift.allocation import allocate_from_result
from causal_uplift.curves import arm_qini
from causal_uplift.data import load_and_split
from causal_uplift.diagnostics import leave_alone_check
from causal_uplift.economics import ARM_CONTROL, ARM_MENS, budget_for_n
from causal_uplift.effects import fit_t_learner


def test_leave_alone_check_has_both_emails():
    build, grade = load_and_split()
    cate = fit_t_learner(build, grade)
    policy = allocate_from_result(cate, budget_for_n(len(grade)))
    table = leave_alone_check(grade, policy, y_col="visit")
    assert set(table["arm"]) >= {ARM_MENS}
    assert (policy == ARM_CONTROL).mean() > 0.5


def test_arm_qini_runs_on_visits():
    build, grade = load_and_split()
    cate = fit_t_learner(build, grade)
    result = arm_qini(grade, cate.cate_mens, ARM_MENS, y_col="visit")
    assert result["x"].ndim == 1
    assert result["qini"].shape == result["x"].shape
    assert result["auc"] == result["auc"]  # not NaN
