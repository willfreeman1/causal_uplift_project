"""Reproduce PLAN.md section 5 numbers on the full Hillstrom file."""

from causal_uplift.eda import average_effects, feature_balance, visit_response_pattern
from causal_uplift.data import load_hillstrom
from causal_uplift.economics import ARM_MENS, ARM_WOMENS


def test_ates_match_plan_section_5():
    effects = average_effects(load_hillstrom()).set_index(["arm", "outcome"])["ate"]
    assert abs(effects.loc[(ARM_MENS, "visit")] - 0.0766) < 0.002
    assert abs(effects.loc[(ARM_MENS, "conversion")] - 0.0068) < 0.001
    assert abs(effects.loc[(ARM_MENS, "spend")] - 0.77) < 0.02
    assert abs(effects.loc[(ARM_WOMENS, "visit")] - 0.0452) < 0.002
    assert abs(effects.loc[(ARM_WOMENS, "conversion")] - 0.0031) < 0.001
    assert abs(effects.loc[(ARM_WOMENS, "spend")] - 0.42) < 0.02


def test_covariates_are_balanced():
    smd = feature_balance(load_hillstrom())
    smd_cols = [c for c in smd.columns if c.startswith("smd_")]
    assert smd[smd_cols].abs().to_numpy().max() < 0.05


def test_womens_email_only_competes_for_womens_buyers():
    pattern = visit_response_pattern(load_hillstrom())
    w = pattern[pattern["group"] == "womens"]

    def ate(value, arm):
        return float(w[(w["value"] == value) & (w["arm"] == arm)]["ate"].iloc[0])

    # Non-buyers: women's email is far behind men's.
    assert ate(0, ARM_WOMENS) < 0.02
    assert ate(0, ARM_MENS) > 0.05
    # Women's-item buyers: the two emails are close.
    assert abs(ate(1, ARM_WOMENS) - ate(1, ARM_MENS)) < 0.02
