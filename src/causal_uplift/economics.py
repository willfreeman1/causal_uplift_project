"""Costs, budget, split, and outcome — one source of truth.

Every script that needs these numbers should import them from here.
See PLAN.md section 5 for the reasoning behind each choice.
"""

from __future__ import annotations

# Arm labels as they appear in Hillstrom `segment`.
ARM_CONTROL = "No E-Mail"
ARM_WOMENS = "Womens E-Mail"
ARM_MENS = "Mens E-Mail"
ARMS = (ARM_CONTROL, ARM_WOMENS, ARM_MENS)
DESIGN_PROPENSITY = {
    ARM_CONTROL: 1.0 / 3.0,
    ARM_WOMENS: 1.0 / 3.0,
    ARM_MENS: 1.0 / 3.0,
}

COSTS = {
    ARM_CONTROL: 0.0,
    ARM_WOMENS: 0.20,
    ARM_MENS: 0.50,
}

# Budget = this fraction of (Men's email cost × number of grade-set customers).
BUDGET_FRACTION_OF_TREAT_ALL_MENS = 0.15

PRIMARY_OUTCOME = "spend"
SUPPORTING_OUTCOMES = ("visit", "conversion")

TRAIN_FRACTION = 0.50
RANDOM_STATE = 42


def budget_for_n(n_customers: int) -> float:
    """Total dollars available on a set of `n_customers` grade-set rows."""
    return BUDGET_FRACTION_OF_TREAT_ALL_MENS * COSTS[ARM_MENS] * n_customers
