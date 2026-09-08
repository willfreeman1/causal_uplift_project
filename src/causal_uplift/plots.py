"""Figures. EDA plots live here now; the money chart comes later."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from causal_uplift.economics import ARM_MENS, ARM_WOMENS

ARM_COLORS = {
    ARM_WOMENS: "#6B5B95",
    ARM_MENS: "#2E86AB",
}

OUTCOME_LABELS = {
    "visit": "Visit rate (pp)",
    "conversion": "Purchase rate (pp)",
    "spend": "Spend ($)",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def save_arm_mix(mix: pd.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    order = mix.sort_index()
    ax.bar(order.index.astype(str), order["share"], color="#4A4A4A")
    ax.axhline(1 / 3, color="#888888", linestyle="--", linewidth=1, label="1/3")
    ax.set_ylabel("Share of customers")
    ax.set_ylim(0, 0.45)
    ax.legend(frameon=False)
    ax.set_title("Assignment mix — should be equal thirds")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_feature_balance(balance: pd.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    smd_cols = [c for c in balance.columns if c.startswith("smd_")]
    x = range(len(balance))
    width = 0.35
    for i, col in enumerate(smd_cols):
        arm = col.replace("smd_", "")
        offset = (i - 0.5) * width
        ax.bar(
            [xi + offset for xi in x],
            balance[col],
            width=width,
            label=arm,
            color=ARM_COLORS.get(arm, "#888888"),
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axhline(0.1, color="#AAAAAA", linestyle=":", linewidth=1)
    ax.axhline(-0.1, color="#AAAAAA", linestyle=":", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(balance["feature"], rotation=20, ha="right")
    ax.set_ylabel("Standardized mean difference vs control")
    ax.set_title("Covariate balance — |SMD| << 0.1 means randomization worked")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_ate_overview(effects: pd.DataFrame, path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=False)
    for ax, outcome in zip(axes, ("visit", "conversion", "spend")):
        sub = effects[effects["outcome"] == outcome]
        y = range(len(sub))
        ate = sub["ate"].to_numpy()
        if outcome != "spend":
            ate = ate * 100
            lo = (sub["ci_low"].to_numpy()) * 100
            hi = (sub["ci_high"].to_numpy()) * 100
        else:
            lo = sub["ci_low"].to_numpy()
            hi = sub["ci_high"].to_numpy()
        for yi, val, low, high, arm in zip(y, ate, lo, hi, sub["arm"]):
            color = ARM_COLORS[arm]
            ax.errorbar(
                [val],
                [yi],
                xerr=[[val - low], [high - val]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=4,
            )
        ax.axvline(0, color="#888888", linewidth=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels(sub["arm"])
        ax.set_xlabel(OUTCOME_LABELS[outcome])
        ax.set_title(outcome)
    fig.suptitle("Average effect vs no email (95% CI)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_visit_by_subgroup(pattern: pd.DataFrame, path: Path) -> None:
    """Visit ATE for women's-item buyers vs everyone else — the pattern that matters."""
    _style()
    sub = pattern[(pattern["group"] == "womens")].copy()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = {0: "Did not buy women's items last year", 1: "Bought women's items last year"}
    x_positions = []
    x_labels = []
    i = 0
    for value in (0, 1):
        slice_ = sub[sub["value"] == value]
        for arm in (ARM_WOMENS, ARM_MENS):
            row = slice_[slice_["arm"] == arm].iloc[0]
            ax.bar(
                i,
                row["ate"] * 100,
                color=ARM_COLORS[arm],
                label=arm if value == 0 else None,
                yerr=1.96 * row["se"] * 100,
                capsize=4,
            )
            x_positions.append(i)
            x_labels.append(f"{labels[value]}\n{arm}")
            i += 1
        i += 0.4
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Visit-rate lift vs no email (pp)")
    ax.set_title("Women's email only competes among last-year women's-item buyers")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_qini_curves(curves: dict, path: Path, title: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, payload in curves.items():
        ax.plot(payload["x"], payload["qini"], label=f"{name} (AUC {payload['auc']:.3f})")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Fraction of customers targeted (ranked by estimated extra)")
    ax.set_ylabel("Cumulative Qini (vs random ranking)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_cumulative_ipw(curve: pd.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["fraction"], curve["ipw"], color="#2E86AB")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Fraction treated, ranked by profit per dollar")
    ax.set_ylabel("Honest extra profit per person ($)")
    ax.set_title("Overall plan: cumulative honest value")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_mix_full_vs_half(full: dict, half: dict, path: Path) -> None:
    _style()
    labels = ["Women's email", "Men's email", "No email"]
    full_n = [full["n_womens"], full["n_mens"], full["n_control"]]
    half_n = [half["n_womens"], half["n_mens"], half["n_control"]]
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], full_n, width, label="Full budget", color="#2E86AB")
    ax.bar([i + width / 2 for i in x], half_n, width, label="Half budget", color="#6B5B95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Customers")
    ax.set_title("Offer mix at full vs half budget")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_money_chart(table: pd.DataFrame, path: Path) -> None:
    """Bar chart of extra profit vs sending nothing, with 95% ranges."""
    _style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(table))
    y = table["aipw"].to_numpy(dtype=float)
    lo = table["aipw_ci_low"].to_numpy(dtype=float)
    hi = table["aipw_ci_high"].to_numpy(dtype=float)
    ax.bar(list(x), y, color="#2E86AB", width=0.7)
    ax.errorbar(
        list(x),
        y,
        yerr=[y - lo, hi - y],
        fmt="none",
        ecolor="#333333",
        capsize=4,
        linewidth=1.2,
    )
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(table["label"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("Extra profit vs sending nothing ($ per person)")
    ax.set_title("Send lists scored on held-out random emails (not the model's guesses)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
