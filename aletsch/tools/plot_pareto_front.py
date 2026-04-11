#!/usr/bin/env python3
"""
Draw the Pareto front for the Aletsch 2-objective Optuna study (Step 6).

The study minimizes two objectives simultaneously:
  - cost_usurf   : mean STD between modelled and observed DEMs
  - cost_velsurf : RMSE between modelled and observed surface speeds (2020)

Run from the aletsch/ root (the script defaults to the SQLite database
written in the cwd by Step 6):

    python tools/plot_pareto_front.py
    python tools/plot_pareto_front.py --storage sqlite:///optuna_2obj.db \
                                      --study  aletsch_2obj \
                                      --out    pareto_front.png --show

The script prints the list of Pareto-optimal trials (number, objective
values, parameters), and saves a scatter plot of all trials with the
Pareto front highlighted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OBJ_NAMES = ("cost_usurf", "cost_velsurf")
OBJ_LABELS = {
    "cost_usurf":   "DEM misfit  (mean STD of Δusurf, m)",
    "cost_velsurf": "Speed misfit  (RMSE of |v|, m/yr)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--storage", default="sqlite:///optuna_2obj.db",
                   help="Optuna storage URL (default: sqlite:///optuna_2obj.db)")
    p.add_argument("--study", default="aletsch_2obj",
                   help="Optuna study name (default: aletsch_2obj)")
    p.add_argument("--out", default="pareto_front.png",
                   help="Output image file (default: pareto_front.png)")
    p.add_argument("--show", action="store_true",
                   help="Also show the figure interactively")
    return p.parse_args()


def is_pareto_optimal(points: np.ndarray) -> np.ndarray:
    """Return a boolean mask over *points* (n, m) marking non-dominated rows.

    A point is Pareto-optimal if no other point is <= in all objectives and
    strictly < in at least one (we are minimizing).
    """
    n = points.shape[0]
    is_opt = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_opt[i]:
            continue
        # dominated by any other point?
        dominated = np.any(
            np.all(points <= points[i], axis=1) &
            np.any(points <  points[i], axis=1)
        )
        if dominated:
            is_opt[i] = False
    return is_opt


def main() -> int:
    args = parse_args()

    try:
        import optuna
    except ImportError:
        print("ERROR: optuna is required (pip install optuna).", file=sys.stderr)
        return 1

    storage = args.storage
    # If a bare path was given, convert it to a sqlite URL
    if "://" not in storage:
        storage = f"sqlite:///{storage}"

    # Helpful check for the default relative path
    if storage.startswith("sqlite:///") and not storage.startswith("sqlite:////"):
        db_rel = storage[len("sqlite:///"):]
        if not Path(db_rel).exists():
            print(f"ERROR: storage file not found: {db_rel}\n"
                  f"Run step 6 first, e.g.\n"
                  f"  igm_run -m +experiment=params_step6 \\\n"
                  f"          hydra/sweeper=igm_optuna \\\n"
                  f"          hydra.sweeper.optuna_config=optuna_2obj_params.yaml",
                  file=sys.stderr)
            return 1

    study = optuna.load_study(study_name=args.study, storage=storage)

    # Collect completed trials with 2 values
    trials = [t for t in study.trials
              if t.state == optuna.trial.TrialState.COMPLETE
              and t.values is not None and len(t.values) == 2]

    if not trials:
        print("No completed trials found in study.", file=sys.stderr)
        return 1

    values = np.array([t.values for t in trials])          # (n, 2)
    numbers = np.array([t.number for t in trials])

    pareto_mask = is_pareto_optimal(values)
    pareto_vals = values[pareto_mask]
    pareto_numbers = numbers[pareto_mask]
    # sort Pareto points by first objective for a clean staircase line
    order = np.argsort(pareto_vals[:, 0])
    pareto_vals = pareto_vals[order]
    pareto_numbers = pareto_numbers[order]

    # ---- Print summary ----
    print(f"\nStudy:   {args.study}")
    print(f"Storage: {storage}")
    print(f"Completed trials: {len(trials)}")
    print(f"Pareto-optimal trials: {len(pareto_vals)}\n")

    print(f"{'trial':>6}  {OBJ_NAMES[0]:>12}  {OBJ_NAMES[1]:>14}   params")
    print("-" * 78)
    for num, v in zip(pareto_numbers, pareto_vals):
        params = next(t.params for t in trials if t.number == num)
        params_str = ", ".join(f"{k}={v:.4g}" for k, v in params.items())
        print(f"{num:6d}  {v[0]:12.3f}  {v[1]:14.3f}   {params_str}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7.5, 6))

    ax.scatter(values[:, 0], values[:, 1],
               s=28, c="lightgray", edgecolor="gray", linewidth=0.4,
               label=f"all trials (n={len(trials)})", zorder=2)

    ax.scatter(pareto_vals[:, 0], pareto_vals[:, 1],
               s=70, c="crimson", edgecolor="black", linewidth=0.6,
               label=f"Pareto front (n={len(pareto_vals)})", zorder=4)

    # staircase line through the Pareto front
    if len(pareto_vals) >= 2:
        ax.plot(pareto_vals[:, 0], pareto_vals[:, 1],
                color="crimson", linewidth=1.2, alpha=0.7, zorder=3)

    # annotate Pareto trials
    for num, v in zip(pareto_numbers, pareto_vals):
        ax.annotate(str(num), xy=v, xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color="crimson")

    ax.set_xlabel(OBJ_LABELS[OBJ_NAMES[0]])
    ax.set_ylabel(OBJ_LABELS[OBJ_NAMES[1]])
    ax.set_title(f"Pareto front — {args.study}")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    out = Path(args.out)
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved to: {out.resolve()}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
