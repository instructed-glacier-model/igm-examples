"""
Analyze and plot results from an Optuna study stored in optuna.db.

Usage:
    python optuna_analysis.py                           # defaults: optuna.db, study_name from optuna_params.yaml
    python optuna_analysis.py --storage sqlite:///optuna.db --study synthetic
    python optuna_analysis.py --config my_config.yaml   # read storage/study_name from config

Produces a folder `optuna_plots/` with PNG figures.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_study(storage, study_name):
    import optuna

    study = optuna.load_study(study_name=study_name, storage=storage)
    return study


def get_completed_trials(study):
    import optuna

    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def short_name(param_name):
    """Shorten Hydra paths for plot labels.

    'processes.iceflow.physics.init_slidingco' -> 'init_slidingco'
    'processes.smb_simple.array.1.3'           -> 'smb_simple.array.1.3'
    """
    parts = param_name.split(".")
    # Walk from the end: keep trailing numeric segments attached
    idx = len(parts) - 1
    while idx > 0 and parts[idx].replace("-", "").isdigit():
        idx -= 1
    # Include at least the parent if the label is a known module element
    if idx > 1 and parts[idx - 1] in ("array", "list"):
        idx -= 1
    return ".".join(parts[idx:])


def plot_objective_history(study, obj_names, out_dir, is_multi):
    """Plot objective values over trials."""
    trials = get_completed_trials(study)
    if not trials:
        return

    numbers = [t.number for t in trials]

    if is_multi:
        n_obj = len(obj_names)
        fig, axes = plt.subplots(n_obj, 1, figsize=(10, 4 * n_obj), sharex=True)
        if n_obj == 1:
            axes = [axes]
        for i, (ax, name) in enumerate(zip(axes, obj_names)):
            vals = [t.values[i] for t in trials]
            ax.scatter(numbers, vals, s=10, alpha=0.6)
            # running best (cumulative min)
            running_best = np.minimum.accumulate(vals)
            ax.plot(numbers, running_best, color="red", linewidth=1.5, label="running best")
            ax.set_ylabel(name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Trial")
        fig.suptitle("Objective history", fontsize=14)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        vals = [t.value for t in trials]
        ax.scatter(numbers, vals, s=10, alpha=0.6)
        running_best = np.minimum.accumulate(vals)
        ax.plot(numbers, running_best, color="red", linewidth=1.5, label="running best")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Objective value")
        ax.set_title("Objective history")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "objective_history.png", dpi=150)
    plt.close(fig)


def plot_pareto_front(study, obj_names, out_dir):
    """Plot 2D Pareto front (multi-objective with 2 objectives)."""
    trials = get_completed_trials(study)
    if not trials or len(obj_names) < 2:
        return

    best_trials = study.best_trials
    best_numbers = {t.number for t in best_trials}

    vals_0 = [t.values[0] for t in trials]
    vals_1 = [t.values[1] for t in trials]

    pareto_0 = [t.values[0] for t in trials if t.number in best_numbers]
    pareto_1 = [t.values[1] for t in trials if t.number in best_numbers]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(vals_0, vals_1, s=15, alpha=0.4, label="all trials", color="C0")
    ax.scatter(pareto_0, pareto_1, s=50, alpha=0.9, label="Pareto front", color="red", edgecolors="black", zorder=5)

    # Connect Pareto points sorted by first objective
    if pareto_0:
        order = np.argsort(pareto_0)
        ax.plot(np.array(pareto_0)[order], np.array(pareto_1)[order], color="red", linewidth=1, alpha=0.5)

    ax.set_xlabel(obj_names[0])
    ax.set_ylabel(obj_names[1])
    ax.set_title("Pareto front")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_front.png", dpi=150)
    plt.close(fig)

    # If 3+ objectives, plot pairwise
    if len(obj_names) > 2:
        from itertools import combinations

        pairs = list(combinations(range(len(obj_names)), 2))
        fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6))
        if len(pairs) == 1:
            axes = [axes]
        for ax, (i, j) in zip(axes, pairs):
            vi = [t.values[i] for t in trials]
            vj = [t.values[j] for t in trials]
            pi = [t.values[i] for t in trials if t.number in best_numbers]
            pj = [t.values[j] for t in trials if t.number in best_numbers]
            ax.scatter(vi, vj, s=15, alpha=0.4, color="C0")
            ax.scatter(pi, pj, s=50, alpha=0.9, color="red", edgecolors="black", zorder=5)
            ax.set_xlabel(obj_names[i])
            ax.set_ylabel(obj_names[j])
            ax.grid(True, alpha=0.3)
        fig.suptitle("Pareto front (pairwise)", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / "pareto_front_pairwise.png", dpi=150)
        plt.close(fig)


def plot_param_vs_objective(study, obj_names, out_dir, is_multi):
    """Scatter plot of each parameter vs each objective."""
    trials = get_completed_trials(study)
    if not trials:
        return

    param_names = list(trials[0].params.keys())
    if not param_names:
        return

    if is_multi:
        objectives = [(name, [t.values[i] for t in trials]) for i, name in enumerate(obj_names)]
    else:
        objectives = [("objective", [t.value for t in trials])]

    n_params = len(param_names)
    n_obj = len(objectives)

    fig, axes = plt.subplots(n_obj, n_params, figsize=(5 * n_params, 4 * n_obj), squeeze=False)

    for row, (obj_label, obj_vals) in enumerate(objectives):
        for col, pname in enumerate(param_names):
            ax = axes[row, col]
            pvals = [t.params[pname] for t in trials]
            sc = ax.scatter(pvals, obj_vals, s=12, alpha=0.5, c=range(len(trials)), cmap="viridis")
            ax.set_xlabel(short_name(pname))
            ax.set_ylabel(obj_label)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Parameters vs objectives (color = trial order)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "param_vs_objective.png", dpi=150)
    plt.close(fig)


def plot_parallel_coordinate(study, obj_names, out_dir, is_multi):
    """Parallel coordinate plot: all params + objectives on normalized axes."""
    trials = get_completed_trials(study)
    if not trials:
        return

    param_names = list(trials[0].params.keys())
    if not param_names:
        return

    # Build columns: params + objectives
    columns = []
    labels = []
    for pname in param_names:
        columns.append([t.params[pname] for t in trials])
        labels.append(short_name(pname))

    if is_multi:
        for i, name in enumerate(obj_names):
            columns.append([t.values[i] for t in trials])
            labels.append(name)
        # Color by sum of normalized objectives
        obj_arrays = [np.array(columns[len(param_names) + i]) for i in range(len(obj_names))]
        color_vals = np.zeros(len(trials))
        for arr in obj_arrays:
            rng = arr.max() - arr.min()
            if rng > 0:
                color_vals += (arr - arr.min()) / rng
    else:
        columns.append([t.value for t in trials])
        labels.append("objective")
        color_vals = np.array(columns[-1])

    # Normalize each column to [0, 1]
    normed = []
    for col in columns:
        arr = np.array(col, dtype=float)
        rng = arr.max() - arr.min()
        if rng > 0:
            normed.append((arr - arr.min()) / rng)
        else:
            normed.append(np.zeros_like(arr))

    fig, ax = plt.subplots(figsize=(3 + 2 * len(labels), 6))
    x = np.arange(len(labels))

    # Sort by color to draw best on top
    order = np.argsort(-color_vals)
    cmap = plt.cm.viridis_r

    c_min, c_max = color_vals.min(), color_vals.max()
    c_rng = c_max - c_min if c_max > c_min else 1.0

    for idx in order:
        y = [normed[c][idx] for c in range(len(labels))]
        color = cmap((color_vals[idx] - c_min) / c_rng)
        ax.plot(x, y, color=color, alpha=0.3, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Normalized value")
    ax.set_title("Parallel coordinate plot (darker = better)")
    ax.grid(True, alpha=0.3, axis="x")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(c_min, c_max))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="sum of normalized objectives" if is_multi else "objective")

    fig.tight_layout()
    fig.savefig(out_dir / "parallel_coordinate.png", dpi=150)
    plt.close(fig)


def plot_param_importance(study, obj_names, out_dir, is_multi):
    """Parameter importance using Optuna's fANOVA (if available)."""
    try:
        import optuna

        trials = get_completed_trials(study)
        if len(trials) < 10:
            print("  Skipping parameter importance (need >= 10 completed trials)")
            return

        if is_multi:
            n_obj = len(obj_names)
            fig, axes = plt.subplots(1, n_obj, figsize=(6 * n_obj, 5))
            if n_obj == 1:
                axes = [axes]
            for i, (ax, name) in enumerate(zip(axes, obj_names)):
                try:
                    importance = optuna.importance.get_param_importances(study, target=lambda t: t.values[i])
                except Exception:
                    continue
                short_labels = [short_name(k) for k in importance.keys()]
                vals = list(importance.values())
                ax.barh(short_labels, vals, color="C0")
                ax.set_xlabel("Importance")
                ax.set_title(name)
                ax.grid(True, alpha=0.3, axis="x")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            try:
                importance = optuna.importance.get_param_importances(study)
            except Exception:
                plt.close(fig)
                return
            short_labels = [short_name(k) for k in importance.keys()]
            vals = list(importance.values())
            ax.barh(short_labels, vals, color="C0")
            ax.set_xlabel("Importance")
            ax.set_title("Parameter importance")
            ax.grid(True, alpha=0.3, axis="x")

        fig.tight_layout()
        fig.savefig(out_dir / "param_importance.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("  Skipping parameter importance (requires scikit-learn)")


def print_best_trials(study, obj_names, is_multi):
    """Print best trial(s) with their parameters."""
    if is_multi:
        best = study.best_trials
        print(f"\n{'='*60}")
        print(f"  {len(best)} Pareto-optimal trial(s)")
        print(f"{'='*60}")
        for t in best:
            named = {n: f"{v:.4g}" for n, v in zip(obj_names, t.values)}
            print(f"\n  Trial {t.number}:")
            print(f"    Objectives: {named}")
            for pname, pval in t.params.items():
                print(f"    {short_name(pname):>25s} = {pval}")
    else:
        best = study.best_trial
        print(f"\n{'='*60}")
        print(f"  Best trial: #{best.number}  (value = {best.value:.4g})")
        print(f"{'='*60}")
        for pname, pval in best.params.items():
            print(f"    {short_name(pname):>25s} = {pval}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna optimization results")
    parser.add_argument("--config", default="optuna_params.yaml", help="Optuna config file (to read storage/study_name)")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (overrides config)")
    parser.add_argument("--study", default=None, help="Study name (overrides config)")
    parser.add_argument("--out-dir", default="optuna_plots", help="Output directory for plots")
    args = parser.parse_args()

    # Read storage/study_name from config if not given on command line
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    storage = args.storage or cfg.get("storage", "sqlite:///optuna.db")
    study_name = args.study or cfg.get("study_name", "study")

    # Detect objective names
    obj_names = []
    if "objectives" in cfg:
        obj_names = [o["name"] for o in cfg["objectives"]]
    is_multi = len(obj_names) > 1

    if not obj_names:
        obj_names = ["objective"]

    print(f"Loading study '{study_name}' from {storage}")
    study = load_study(storage, study_name)

    n_complete = len(get_completed_trials(study))
    n_total = len(study.trials)
    print(f"  {n_complete} completed / {n_total} total trials")
    print(f"  Objectives: {obj_names}")

    if n_complete == 0:
        print("No completed trials found. Nothing to plot.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print("\nGenerating plots...")

    print("  - objective_history.png")
    plot_objective_history(study, obj_names, out_dir, is_multi)

    if is_multi:
        print("  - pareto_front.png")
        plot_pareto_front(study, obj_names, out_dir)

    print("  - param_vs_objective.png")
    plot_param_vs_objective(study, obj_names, out_dir, is_multi)

    print("  - parallel_coordinate.png")
    plot_parallel_coordinate(study, obj_names, out_dir, is_multi)

    print("  - param_importance.png")
    plot_param_importance(study, obj_names, out_dir, is_multi)

    print_best_trials(study, obj_names, is_multi)

    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
