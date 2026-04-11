#!/usr/bin/env python3
"""
Spatial misfit maps for an Aletsch IGM run.

Two rows are produced from the saved NetCDF output:

  (a) DEM misfit  : modelled minus observed surface elevation at each of the
                    seven historical observation years
                    (1880, 1926, 1957, 1980, 1999, 2009, 2017),
                    masked to ice-covered pixels (modelled thk > 1 m).

  (b) Speed (2020): observed surface speed | modelled surface speed |
                    model − obs difference, masked to pixels where the InSAR
                    velocity field is valid and the observed speed is above
                    1 m/yr.

The DEM and difference panels share a symmetric red/blue diverging colormap
so the structure of the residuals is immediately readable; the two
absolute-speed panels share a common viridis colorbar so they can be
compared by eye.

Run from the aletsch/ root (the script reads data/past_surf.nc and
data/input.nc by default):

    python tools/plot_misfit_maps.py --run <path/to/run/dir>
    python tools/plot_misfit_maps.py --run multirun/2026-04-11/_optuna/114 --show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

OBS_YEARS = [1880, 1926, 1957, 1980, 1999, 2009, 2017]
VEL_THR = 1.0          # m/yr — same as track_velsurf_obs default
THK_MASK_THR = 1.0     # m   — modelled ice mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run", required=True,
                   help="Directory holding output.nc from an IGM run "
                        "(e.g. outputs/<timestamp> or multirun/<date>/<trial>)")
    p.add_argument("--past-surf", default="data/past_surf.nc",
                   help="NetCDF with surf_<year> observed DEMs")
    p.add_argument("--input", default="data/input.nc",
                   help="NetCDF with uvelsurfobs / vvelsurfobs")
    p.add_argument("--out", default="misfit_maps.png",
                   help="Output PNG (default: misfit_maps.png)")
    p.add_argument("--show", action="store_true",
                   help="Also open the figure interactively")
    return p.parse_args()


def load_run(run_dir: Path):
    nc_path = run_dir / "output.nc"
    if not nc_path.exists():
        sys.exit(
            f"ERROR: {nc_path} not found.\n"
            f"Pass --run pointing to an IGM run directory that contains an "
            f"output.nc file (e.g. outputs/<timestamp> or "
            f"multirun/<date>/<trial>)."
        )
    nc = Dataset(nc_path)
    t = np.asarray(nc.variables["time"][:])
    x = np.asarray(nc.variables["x"][:])
    y = np.asarray(nc.variables["y"][:])
    usurf = np.asarray(nc.variables["usurf"][:])           # (T, ny, nx)
    thk = np.asarray(nc.variables["thk"][:])
    uvel = np.asarray(nc.variables["uvelsurf"][:])
    vvel = np.asarray(nc.variables["vvelsurf"][:])
    nc.close()
    return t, x, y, usurf, thk, uvel, vvel


def load_obs_dems(past_surf_path: Path):
    nc = Dataset(past_surf_path)
    obs = {y: np.squeeze(np.asarray(nc.variables[f"surf_{y}"][:])) for y in OBS_YEARS}
    nc.close()
    return obs


def load_obs_vel(input_path: Path):
    nc = Dataset(input_path)
    uo = np.squeeze(np.asarray(nc.variables["uvelsurfobs"][:]))
    vo = np.squeeze(np.asarray(nc.variables["vvelsurfobs"][:]))
    nc.close()
    return uo, vo


def nearest_time_index(t: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(t - target)))


def main() -> int:
    args = parse_args()
    run = Path(args.run)
    t, x, y, usurf, thk, uvel, vvel = load_run(run)
    obs_dems = load_obs_dems(Path(args.past_surf))
    uo, vo = load_obs_vel(Path(args.input))

    extent = (x.min(), x.max(), y.min(), y.max())
    dem_vmax = 80.0   # symmetric (m)
    vel_vmax = 80.0   # symmetric (m/yr)

    # ---- DEM panels ----
    n_dem = len(OBS_YEARS)
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(
        2, n_dem,
        height_ratios=[1.0, 1.05],
        hspace=0.32, wspace=0.18,
    )

    dem_axes = []
    for i, yr in enumerate(OBS_YEARS):
        ax = fig.add_subplot(gs[0, i])
        dem_axes.append(ax)

        k = nearest_time_index(t, yr)
        diff = usurf[k] - obs_dems[yr]
        diff = np.where(thk[k] > THK_MASK_THR, diff, np.nan)

        im = ax.imshow(diff, origin="lower", extent=extent,
                       cmap="RdBu_r", vmin=-dem_vmax, vmax=dem_vmax,
                       interpolation="nearest")
        # ice outline
        ax.contour(thk[k] > THK_MASK_THR, levels=[0.5],
                   extent=extent, colors="k", linewidths=0.6, alpha=0.7)

        valid = np.isfinite(diff)
        if valid.any():
            mean = np.nanmean(diff)
            std = np.nanstd(diff)
            ax.set_title(f"{yr}\nmean {mean:+.1f} m   std {std:.1f} m",
                         fontsize=10)
        else:
            ax.set_title(f"{yr}\n(no ice)", fontsize=10)

        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("DEM misfit\nmodel − obs (m)", fontsize=11)

    cbar_ax = fig.add_axes([0.92, 0.55, 0.012, 0.32])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Δusurf (m)", fontsize=10)

    # ---- Speed panels (2020): observed | modelled | difference ----
    # Split the bottom row into three equal-width subplots; n_dem = 7 columns,
    # so use a small inner grid spec to get exactly three equal-width axes.
    inner = gs[1, :].subgridspec(1, 3, wspace=0.22)
    ax_o = fig.add_subplot(inner[0, 0])
    ax_m = fig.add_subplot(inner[0, 1])
    ax_v = fig.add_subplot(inner[0, 2])

    k_end = nearest_time_index(t, 2020.0)
    speed_mod = np.sqrt(uvel[k_end] ** 2 + vvel[k_end] ** 2)
    speed_obs = np.sqrt(uo ** 2 + vo ** 2)

    valid = (
        np.isfinite(uo)
        & np.isfinite(vo)
        & (speed_obs >= VEL_THR)
        & (thk[k_end] > THK_MASK_THR)
    )

    diff_v = np.where(valid, speed_mod - speed_obs, np.nan)
    obs_view = np.where(valid, speed_obs, np.nan)
    mod_view = np.where(valid, speed_mod, np.nan)

    # Common color scale for the two absolute-speed panels so they can be
    # compared by eye. Set the upper bound from the observed field so the
    # modelled panel uses the same dynamic range, even if the model is faster.
    speed_vmax = float(np.nanpercentile(speed_obs[valid], 99)) if valid.any() else 200.0
    speed_vmax = max(50.0, np.ceil(speed_vmax / 10.0) * 10.0)

    def _decorate(ax):
        ax.contour(thk[k_end] > THK_MASK_THR, levels=[0.5],
                   extent=extent, colors="w", linewidths=0.6, alpha=0.7)
        ax.set_xticks([]); ax.set_yticks([])

    # observed speed
    im_o = ax_o.imshow(obs_view, origin="lower", extent=extent,
                       cmap="viridis", vmin=0, vmax=speed_vmax,
                       interpolation="nearest")
    _decorate(ax_o)
    ax_o.set_title("Observed surface speed (2020)", fontsize=11)
    cb_o = fig.colorbar(im_o, ax=ax_o, fraction=0.04, pad=0.02)
    cb_o.set_label("|u_s|  (m yr$^{-1}$)", fontsize=10)

    # modelled speed (same colormap & limits)
    im_m = ax_m.imshow(mod_view, origin="lower", extent=extent,
                       cmap="viridis", vmin=0, vmax=speed_vmax,
                       interpolation="nearest")
    _decorate(ax_m)
    ax_m.set_title("Modelled surface speed (2020)", fontsize=11)
    cb_m = fig.colorbar(im_m, ax=ax_m, fraction=0.04, pad=0.02)
    cb_m.set_label("|u_s|  (m yr$^{-1}$)", fontsize=10)

    # speed misfit (model − obs)
    im_v = ax_v.imshow(diff_v, origin="lower", extent=extent,
                       cmap="RdBu_r", vmin=-vel_vmax, vmax=vel_vmax,
                       interpolation="nearest")
    ax_v.contour(thk[k_end] > THK_MASK_THR, levels=[0.5],
                 extent=extent, colors="k", linewidths=0.6, alpha=0.7)
    ax_v.set_xticks([]); ax_v.set_yticks([])

    diff_finite = diff_v[np.isfinite(diff_v)]
    if diff_finite.size:
        rmse = float(np.sqrt(np.mean(diff_finite ** 2)))
        bias = float(np.mean(diff_finite))
        ax_v.set_title(
            f"Speed misfit at 2020\nRMSE {rmse:.1f}   bias {bias:+.1f}  (m yr$^{{-1}}$)",
            fontsize=11,
        )
    else:
        ax_v.set_title("Speed misfit at 2020\n(no valid pixels)", fontsize=11)
    cb_v = fig.colorbar(im_v, ax=ax_v, fraction=0.04, pad=0.02)
    cb_v.set_label("Δ|u_s|  (m yr$^{-1}$)", fontsize=10)

    fig.suptitle(
        f"Aletsch — spatial misfits  ({run.name})",
        fontsize=13, y=0.995,
    )

    out = Path(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {out.resolve()}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
