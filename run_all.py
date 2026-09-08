"""Rebuild figures and metrics from the committed Hillstrom CSV. No fake file."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_uplift.data import FIGURES_DIR
from causal_uplift.diagnostics import run_diagnostics
from causal_uplift.eda import run_eda
from causal_uplift.report import write_report


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("1/4 Look at the data...")
    run_eda()
    print("2/4 Score send lists and write money chart...")
    write_report()
    print("3/4 Holdout checks (ranking, skipped people, half budget)...")
    run_diagnostics()
    print("4/4 Tests...")
    subprocess.check_call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    print(f"Done. Figures in {FIGURES_DIR}")
    print(f"Numbers in {ROOT / 'results' / 'metrics.json'}")


if __name__ == "__main__":
    main()
