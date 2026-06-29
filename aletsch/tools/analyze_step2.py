#!/usr/bin/env python3
"""
Analyze a single-parameter sweep: plot velocity RMSE, thickness RMSE
(validation against thkobs), and volume as a function of the swept parameter.

Usage:
    python analyze_step2.py <results_dir> --param <hydra_param_name>

Examples:
    python analyze_step2.py multirun/2026-03-12/11-00-00 --param processes.iceflow.physics.sliding.tau_ref
    python analyze_step2.py multirun/2026-03-12/12-00-00 --param processes.iceflow.physics.viscosity.arrhenius
"""

import argparse
import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import re
import yaml


def parse_param(nc_path, results_dir, param_name):
    """Extract a parameter value from directory name or Hydra overrides."""
    rel_path = os.path.relpath(nc_path, results_dir)

    short_name = param_name.split(".")[-1]
    for pattern in [re.escape(param_name) + r'=([0-9.e+-]+)',
                    short_name + r'_([0-9.e+-]+)',
                    short_name + r'=([0-9.e+-]+)']:
        m = re.search(pattern, rel_path)
        if m:
            return float(m.group(1))

    nc_dir = os.path.dirname(nc_path)
    overrides_file = os.path.join(nc_dir, ".hydra", "overrides.yaml")
    if os.path.exists(overrides_file):
        with open(overrides_file) as f:
            overrides = yaml.safe_load(f)
        for override in overrides:
            m = re.search(re.escape(param_name) + r'=([0-9.e+-]+)', override)
            if m:
                return float(m.group(1))

    return None


def load_thkobs(input_path, ds_out):
    """Load thkobs from input.nc, cropped to match output grid."""
    input_ds = xr.open_dataset(input_path)
    if "thkobs" not in input_ds:
        input_ds.close()
        return None, None

    x_out, y_out = ds_out.x.values, ds_out.y.values
    thkobs_full = input_ds["thkobs"].sel(
        x=slice(x_out.min(), x_out.max()),
        y=slice(y_out.min(), y_out.max()),
    ).values
    input_ds.close()

    thk_mask = ~np.isnan(thkobs_full)
    if not thk_mask.any():
        return None, None
    return thkobs_full, thk_mask


def main():
    parser = argparse.ArgumentParser(description="Plot single-parameter sweep results")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--param", required=True,
                        help="Full Hydra parameter name (e.g. processes.iceflow.physics.sliding.tau_ref)")
    args = parser.parse_args()

    param_name = args.param
    short_name = param_name.split(".")[-1]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # tools/ lives one level below the example root, where data/ resides
    input_path = os.path.normpath(os.path.join(script_dir, "..", "data", "input.nc"))

    nc_files = sorted(glob.glob(os.path.join(args.results_dir, "**/geology-optimized.nc"), recursive=True))
    if not nc_files:
        print(f"No geology-optimized.nc found in {args.results_dir}")
        return

    # Load thkobs once using first output to determine crop
    ds0 = xr.open_dataset(nc_files[0])
    thkobs, thk_mask = load_thkobs(input_path, ds0)
    ds0.close()

    if thkobs is None:
        print("WARNING: No thickness observations found in input.nc")

    params, vel_rmses, thk_rmses, volumes = [], [], [], []

    for nc_path in nc_files:
        val = parse_param(nc_path, args.results_dir, param_name)
        if val is None:
            rel = os.path.relpath(nc_path, args.results_dir)
            print(f"  [skip] Cannot parse {short_name} from: {rel}")
            continue

        ds = xr.open_dataset(nc_path)
        thk = ds["thk"].values
        velsurf_mag = ds["velsurf_mag"].values
        velsurfobs_mag = ds["velsurfobs_mag"].values
        icemask = ds["icemask"].values
        dx = float(ds.x.values[1] - ds.x.values[0])
        ds.close()

        mask_vel = icemask > 0.5
        vel_rmse = np.sqrt(np.nanmean((velsurf_mag[mask_vel] - velsurfobs_mag[mask_vel]) ** 2))

        if thkobs is not None:
            thk_rmse = np.sqrt(np.nanmean((thk[thk_mask] - thkobs[thk_mask]) ** 2))
        else:
            thk_rmse = np.nan

        vol = np.sum(thk) * dx ** 2 * 1e-9

        params.append(val)
        vel_rmses.append(vel_rmse)
        thk_rmses.append(thk_rmse)
        volumes.append(vol)

        print(f"  {short_name}={val:g}: vel_rmse={vel_rmse:.2f} m/yr, thk_rmse={thk_rmse:.1f} m, vol={vol:.1f} km3")

    if len(params) < 2:
        print("Not enough results to plot.")
        return

    idx = np.argsort(params)
    params = np.array(params)[idx]
    vel_rmses = np.array(vel_rmses)[idx]
    thk_rmses = np.array(thk_rmses)[idx]
    volumes = np.array(volumes)[idx]

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(params, vel_rmses, "o-", color="steelblue", markersize=8, linewidth=2)
    for i, p in enumerate(params):
        ax.annotate(f"  {p:g}", (params[i], vel_rmses[i]), fontsize=9)
    ax.set_xlabel(short_name, fontsize=12)
    ax.set_ylabel("Velocity RMSE (m/yr)", fontsize=12)
    ax.set_title("Velocity misfit", fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(params, thk_rmses, "s-", color="orangered", markersize=8, linewidth=2)
    for i, p in enumerate(params):
        ax.annotate(f"  {p:g}", (params[i], thk_rmses[i]), fontsize=9)
    ax.set_xlabel(short_name, fontsize=12)
    ax.set_ylabel("Thickness RMSE (m)", fontsize=12)
    ax.set_title("Thickness misfit (validation against thkobs)", fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(params, volumes, "D-", color="seagreen", markersize=8, linewidth=2)
    for i, p in enumerate(params):
        ax.annotate(f"  {p:g}", (params[i], volumes[i]), fontsize=9)
    ax.set_xlabel(short_name, fontsize=12)
    ax.set_ylabel("Ice volume (km$^3$)", fontsize=12)
    ax.set_title("Volume", fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Parameter sweep: {short_name}", fontsize=14, y=1.02)
    fig.tight_layout()
    out_png = os.path.join(args.results_dir, f"sweep_{short_name}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")

    # --- Summary table ---
    print("\n" + "=" * 50)
    print(f"{short_name:>12} {'vel_rmse':>10} {'thk_rmse':>10} {'volume':>10}")
    print("=" * 50)
    for i in range(len(params)):
        print(f"{params[i]:>12g} {vel_rmses[i]:>10.2f} {thk_rmses[i]:>10.1f} {volumes[i]:>10.1f}")


if __name__ == "__main__":
    main()
