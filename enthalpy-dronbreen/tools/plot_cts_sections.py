#!/usr/bin/env python3

"""Plot the modelled thermal structure along the GPR lines, against the observed CTS.

For each radar line, draws a vertical section through the glacier: temperate ice in
red and cold ice in blue, with the radar-observed cold-temperate transition surface
laid over them as a dashed black line. The modelled CTS is the boundary between the two
fills, so where the dashed line runs through blue the model made too little temperate
ice, and where it runs through red, too much.

Transects are labelled T1, T2, ... in order of how much temperate ice the radar found,
so T1 is the most temperate line of the survey.

Run from the example root, e.g.:

    python tools/plot_cts_sections.py --run outputs/2026-09-16/17-45-54
    python tools/plot_cts_sections.py --run outputs/<stamp> --all --time 2022

After a Step 2 sweep, plot the best trial instead of naming its directory by hand:

    python tools/plot_cts_sections.py --best-of sqlite:///optuna_cts.db
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.cts import (
    best_trial_run,
    cts_elevation,
    enthalpy_grid,
    layer_elevations,
    load_cts_profiles,
    representative_profile,
    profile_rmse,
    sample_columns,
)
from tools.plotting import STYLE, draw_section, section_key


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="Hydra run directory holding output.nc")
    src.add_argument(
        "--best-of",
        metavar="STORAGE",
        help="Optuna storage URL; plots the lowest-cost trial of the study",
    )
    p.add_argument("--study", default="dronbreen_cts", help="Study name for --best-of")
    p.add_argument(
        "--sweep-dir", default="multirun/optuna_cts", help="Where trial dirs live"
    )
    p.add_argument(
        "--line",
        help="Radar line to plot, by label (T3) or full id (default: one that crosses "
             "a transition and fits about as well as the survey median)",
    )
    p.add_argument("--all", action="store_true", help="Plot every radar line")
    p.add_argument(
        "--time", type=float, help="Model year to plot (default: the last saved one)"
    )
    p.add_argument("--data-dir", default="data/cts_gpr", help="GPR observation folder")
    p.add_argument("--glacier", default="dronbreen")
    p.add_argument("--year", type=int, default=2022, help="Survey year")
    p.add_argument("--out", help="Output folder (default: <run>/plots)")

    return p.parse_args()


def section_figure(profile, model, year, out_path):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.1, 2.0))

        draw_section(ax, profile, model, legend=False)
        # Upper left: on these transects the ice rises towards the right, so a legend
        # in the usual upper-right corner sits on top of the glacier.
        ax.legend(*section_key(), loc="upper left", frameon=False,
                  handletextpad=0.5, labelspacing=0.28, borderaxespad=0.2,
                  fontsize=6.2)

        rmse = profile_rmse(model["cts"], profile["cts_obs"])
        title = f"{profile['label']}: {profile['name']} — {year:.0f}"
        if np.isfinite(rmse):
            title += f", CTS RMSE {rmse:.1f} m"
        ax.set_title(title, color="#52514e")

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    return rmse


def main():
    args = parse_args()

    run = os.path.abspath(
        args.run
        if args.run
        else best_trial_run(args.best_of, args.study, args.sweep_dir)
    )
    cfg = OmegaConf.load(os.path.join(run, ".hydra", "config.yaml"))
    nz, vert_spacing = enthalpy_grid(cfg)

    ds = xr.open_dataset(os.path.join(run, "output.nc"))
    snapshot = ds.isel(time=-1) if args.time is None else ds.sel(time=args.time, method="nearest")
    year = float(snapshot["time"])

    x, y = ds["x"].values, ds["y"].values
    profiles = load_cts_profiles(args.data_dir, args.glacier, args.year)

    def sample(profile):
        east, north = profile["easting"], profile["northing"]
        column = lambda name: sample_columns(snapshot[name].values, x, y, east, north)

        E, E_pmp = column("E"), column("E_pmp")
        topg, thk, usurf = column("topg"), column("thk"), column("usurf")

        return {
            "E": E,
            "E_pmp": E_pmp,
            "thk": thk,
            "usurf": usurf,
            "topg": topg,
            "z": layer_elevations(topg, thk, nz, vert_spacing),
            "cts": cts_elevation(E, E_pmp, topg, thk, vert_spacing),
        }

    models = {p["label"]: sample(p) for p in profiles}

    if not args.all:
        if args.line:
            wanted = [p for p in profiles if args.line in (p["name"], p["label"])]
            if not wanted:
                raise SystemExit(f"No radar line named {args.line}")
            profiles = wanted
        else:
            # Not the most contrasted transect: that tends to be the worst-fitting one,
            # which would misrepresent the model. See tools/cts.representative_profile.
            misfits = [
                profile_rmse(models[p["label"]]["cts"], p["cts_obs"]) for p in profiles
            ]
            profiles = [representative_profile(profiles, misfits)]

    out_dir = args.out or os.path.join(run, "plots")
    os.makedirs(out_dir, exist_ok=True)

    for profile in profiles:
        model = models[profile["label"]]
        out_path = os.path.join(
            out_dir, f"cts_section_{profile['label']}_{year:.0f}.png"
        )
        rmse = section_figure(profile, model, year, out_path)
        print(f"{profile['label']:4s} {profile['name']:34s} CTS RMSE {rmse:6.1f} m  "
              f"-> {out_path}")


if __name__ == "__main__":
    main()
