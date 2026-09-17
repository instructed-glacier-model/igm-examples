#!/usr/bin/env python3

"""Diagnostics for the Step 2 Optuna study.

Two figures:

* `optuna_cost_vs_parameters.png` — the CTS RMSE against each search parameter, which
  shows which parameters the misfit is actually sensitive to. A flat cloud means the
  observations do not constrain that parameter.
* `optuna_parameter_pairs.png` — the sampled parameter pairs, coloured by cost, which
  shows where the sampler concentrated and whether the optimum sits against a bound
  (in which case the bound should be widened in `optuna/optuna_cts.yaml`).

Run from the example root, after the Step 2 sweep:

    python tools/plot_optuna.py --storage sqlite:///optuna_cts.db --study dronbreen_cts

For an interactive alternative:  optuna-dashboard sqlite:///optuna_cts.db
"""

import argparse
import itertools
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna

# Short axis labels for the parameters searched in optuna/optuna_cts.yaml.
LABELS = {
    "processes.clim_carra_ela.T_ela": "Temperature above ELA (°C)",
    "processes.clim_carra_ela.ela": "ELA (m a.s.l.)",
    "processes.enthalpy.thermal.basal_heat_flux_ref": "Geothermal flux (W m$^{-2}$)",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--storage", default="sqlite:///optuna_cts.db")
    p.add_argument("--study", default="dronbreen_cts")
    p.add_argument("--out", default="plots", help="Output folder")

    return p.parse_args()


def load_trials(storage, study_name):
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and np.isfinite(t.value)
    ]
    if not trials:
        raise SystemExit(
            f"Study '{study_name}' has no completed trial. Run the Step 2 sweep first."
        )

    names = sorted(trials[0].params)
    params = {n: np.array([t.params[n] for t in trials]) for n in names}
    cost = np.array([t.value for t in trials])

    return params, cost


def label(name):
    return LABELS.get(name, name.split(".")[-1])


def plain_label(name):
    """Axis label without mathtext, for printing to a terminal."""

    return label(name).replace("$^{-2}$", "-2")


def plot_cost_vs_parameters(params, cost, out_path):
    fig, axes = plt.subplots(1, len(params), figsize=(4.2 * len(params), 3.8), squeeze=False)

    best = int(np.argmin(cost))
    for ax, (name, values) in zip(axes[0], params.items()):
        ax.scatter(values, cost, s=22, c="#4a7fb5", edgecolor="none", alpha=0.8)
        ax.scatter(values[best], cost[best], s=90, marker="*", c="#c0392b", zorder=3,
                   label=f"best: {cost[best]:.1f} m")
        ax.set(xlabel=label(name), ylabel="CTS RMSE (m)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_pairs(params, cost, out_path):
    pairs = list(itertools.combinations(params, 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 3.8), squeeze=False)

    best = int(np.argmin(cost))
    for ax, (a, b) in zip(axes[0], pairs):
        # Colour scale from the data: hardwired limits hide the structure of a study
        # whose costs land outside them.
        sc = ax.scatter(params[a], params[b], c=cost, s=45, cmap="viridis_r",
                        edgecolor="none")
        ax.scatter(params[a][best], params[b][best], s=140, marker="*", c="#c0392b",
                   edgecolor="k", linewidth=0.5, zorder=3)
        ax.set(xlabel=label(a), ylabel=label(b))
        fig.colorbar(sc, ax=ax, label="CTS RMSE (m)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    params, cost = load_trials(args.storage, args.study)
    os.makedirs(args.out, exist_ok=True)

    best = int(np.argmin(cost))
    print(f"{len(cost)} completed trials, best CTS RMSE {cost[best]:.2f} m")
    for name, values in params.items():
        print(f"  {plain_label(name):28s} {values[best]:.3f}")

    for filename, plot in [
        ("optuna_cost_vs_parameters.png", plot_cost_vs_parameters),
        ("optuna_parameter_pairs.png", plot_parameter_pairs),
    ]:
        path = os.path.join(args.out, filename)
        plot(params, cost, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
