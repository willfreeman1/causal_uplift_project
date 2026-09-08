"""Known-DGP simulation: capture fraction, sleeping dogs skipped, AIPW near truth."""

from causal_uplift.simulate import make_synthetic, run_simulation, true_policy_value, perfect_policy
from causal_uplift.economics import ARM_CONTROL, ARM_MENS, ARM_WOMENS, budget_for_n


def test_sleeping_dogs_exist_and_perfect_plan_skips_them():
    df = make_synthetic(n=8_000, random_state=0)
    assert df["sleeping"].mean() > 0.05
    grade = df  # use all rows as the "grade" list for the perfect plan
    perfect = perfect_policy(grade, budget_for_n(len(grade)))
    dogs = grade["sleeping"] == 1
    assert (perfect[dogs] == ARM_CONTROL).mean() > 0.95
    assert (perfect == ARM_MENS).sum() > 0
    assert (perfect == ARM_WOMENS).sum() > 0
    assert true_policy_value(grade, perfect) > 0


def test_pipeline_captures_most_of_perfect_and_skips_dogs():
    out = run_simulation(n=8_000, random_state=0)
    assert out["v_perfect"] > 0
    assert 0.5 < out["capture"] <= 1.05
    assert out["dogs_untreated"] > 0.8
    assert out["pred_tau_w_dogs"] < 0
    assert out["pred_tau_m_dogs"] < 0
    # Honest grade should land near the known true value of the pipeline plan.
    assert out["aipw_ci_low"] - 0.05 <= out["v_true_pipeline"] <= out["aipw_ci_high"] + 0.05
    assert abs(out["aipw"] - out["v_true_pipeline"]) < 0.15
