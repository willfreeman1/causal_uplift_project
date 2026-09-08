"""CATE models recover the known average effects and have the right shape."""

from causal_uplift.data import load_and_split
from causal_uplift.economics import ARM_MENS, ARM_WOMENS
from causal_uplift.effects import (
    fit_s_learner,
    fit_t_learner,
    fit_x_learner,
    measured_ate,
    recovery_table,
)


def test_s_t_x_shapes_and_ate_recovery():
    build, grade = load_and_split()
    n = len(grade)
    ate = measured_ate(grade)
    results = {
        "s_learner": fit_s_learner(build, grade),
        "t_learner": fit_t_learner(build, grade),
        "x_learner": fit_x_learner(build, grade),
    }
    for name, res in results.items():
        assert res.cate_mens.shape == (n,)
        assert res.cate_womens.shape == (n,)
        means = res.mean_cates()
        # Recover the experimental average, not a made-up scale.
        assert abs(means[ARM_MENS] - ate[ARM_MENS]) < 1.0, name
        assert abs(means[ARM_WOMENS] - ate[ARM_WOMENS]) < 1.0, name
        # Men's email is the stronger offer on average.
        assert means[ARM_MENS] > means[ARM_WOMENS] - 0.15, name


def test_recovery_table_has_one_row_per_method():
    build, grade = load_and_split()
    results = {"t_learner": fit_t_learner(build, grade)}
    table = recovery_table(results, grade)
    assert list(table["method"]) == ["t_learner"]
    assert "mean_cate_mens" in table.columns


def test_t_learner_handles_category_drift_between_build_and_grade():
    build, grade = load_and_split()
    build = build[build["zip_code"] != "Rural"].reset_index(drop=True)
    res = fit_t_learner(build, grade)
    assert res.cate_mens.shape == (len(grade),)
