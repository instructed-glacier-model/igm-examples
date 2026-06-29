#!/usr/bin/env python3
"""
Compare the surface-velocity fields of two IGM runs.

Intended for Step 6: validate the on-line retrained iceflow solver against the
reference identity-mapping solver. Run A is the "test" (e.g. the on-line solver),
run B is the "reference"; the script reports error metrics of (test - reference)
over the glaciated area and saves a 3-panel figure (reference | test | difference).

Usage:
    python tools/compare_velocity.py <test_run> <reference_run> [options]

Examples:
    python tools/compare_velocity.py outputs/step6_online outputs/step6_reference
    python tools/compare_velocity.py outputs/step6_online outputs/step6_reference \
        --time 2000 --out step6_velocity_compare.png --show

Each <run> is a directory containing an output NetCDF (output.nc / output_reference.nc)
or a direct path to such a file. Requires `velsurf_mag` (and, if available,
`uvelsurf`/`vvelsurf` for the vector error and `thk` for the ice mask).
"""
import argparse
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_nc(path):
    """Resolve a run directory (or file) to an output NetCDF path."""
    if os.path.isfile(path):
        return path
    for cand in ("output.nc", "output_reference.nc"):
        p = os.path.join(path, cand)
        if os.path.exists(p):
            return p
    ncs = [f for f in sorted(os.listdir(path))
           if f.endswith(".nc") and not f.endswith("_ts.nc")]
    if ncs:
        return os.path.join(path, ncs[0])
    raise FileNotFoundError(f"No output NetCDF found in '{path}'")


def surf_speed(ds, t):
    """Surface speed at time t (use velsurf_mag, else build it from u/v)."""
    sel = ds.sel(time=t, method="nearest")
    if "velsurf_mag" in ds:
        return np.asarray(sel["velsurf_mag"].values, dtype=float)
    if "uvelsurf" in ds and "vvelsurf" in ds:
        u = np.asarray(sel["uvelsurf"].values, dtype=float)
        v = np.asarray(sel["vvelsurf"].values, dtype=float)
        return np.hypot(u, v)
    raise KeyError("No velsurf_mag or uvelsurf/vvelsurf in dataset")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test_run", help="run dir/file of the TEST solution (e.g. on-line solver)")
    ap.add_argument("reference_run", help="run dir/file of the REFERENCE solution (identity mapping)")
    ap.add_argument("--time", type=float, default=None, help="time to compare (default: last common save)")
    ap.add_argument("--out", default="compare_velocity.png", help="output figure path")
    ap.add_argument("--label-test", default="on-line solver")
    ap.add_argument("--label-ref", default="reference (identity)")
    ap.add_argument("--show", action="store_true", help="also display the figure")
    args = ap.parse_args()

    fa, fb = find_nc(args.test_run), find_nc(args.reference_run)
    da, db = xr.open_dataset(fa), xr.open_dataset(fb)

    # common comparison time
    ta, tb = da["time"].values, db["time"].values
    t = args.time if args.time is not None else float(min(ta.max(), tb.max()))
    ta_sel, tb_sel = float(da.sel(time=t, method="nearest")["time"]), \
                     float(db.sel(time=t, method="nearest")["time"])
    print(f"test      : {fa}  (t={ta_sel:g})")
    print(f"reference : {fb}  (t={tb_sel:g})")

    va, vb = surf_speed(da, t), surf_speed(db, t)
    if va.shape != vb.shape:
        raise ValueError(f"grid mismatch: test {va.shape} vs reference {vb.shape}")

    # ice mask: glaciated in the reference (fall back to test, else any finite)
    if "thk" in db:
        mask = np.asarray(db.sel(time=t, method="nearest")["thk"].values, dtype=float) > 1.0
    elif "thk" in da:
        mask = np.asarray(da.sel(time=t, method="nearest")["thk"].values, dtype=float) > 1.0
    else:
        mask = np.isfinite(va) & np.isfinite(vb)
    mask &= np.isfinite(va) & np.isfinite(vb)

    diff = va - vb
    d = diff[mask]
    rmse = float(np.sqrt(np.mean(d ** 2)))
    mae = float(np.mean(np.abs(d)))
    maxd = float(np.max(np.abs(d)))
    mean_ref = float(np.mean(vb[mask]))
    rel = 100.0 * rmse / mean_ref if mean_ref else float("nan")

    print(f"\nSurface-speed comparison over {int(mask.sum())} glaciated cells (m/yr):")
    print(f"  mean speed   test={np.mean(va[mask]):8.2f}   reference={mean_ref:8.2f}")
    print(f"  RMSE(test-ref) = {rmse:7.2f}   ({rel:.1f}% of mean reference speed)")
    print(f"  MAE            = {mae:7.2f}")
    print(f"  max |diff|     = {maxd:7.2f}")

    # vector RMSE if components available in both
    if all(k in da for k in ("uvelsurf", "vvelsurf")) and \
       all(k in db for k in ("uvelsurf", "vvelsurf")):
        ua = np.asarray(da.sel(time=t, method="nearest")["uvelsurf"].values, float)
        vva = np.asarray(da.sel(time=t, method="nearest")["vvelsurf"].values, float)
        ub = np.asarray(db.sel(time=t, method="nearest")["uvelsurf"].values, float)
        vvb = np.asarray(db.sel(time=t, method="nearest")["vvelsurf"].values, float)
        vec = np.sqrt((ua - ub) ** 2 + (vva - vvb) ** 2)[mask]
        print(f"  vector RMSE    = {float(np.sqrt(np.mean(vec ** 2))):7.2f}")

    # figure: reference | test | difference
    speed = np.where(mask, vb, np.nan)
    vmax = np.nanpercentile(np.where(mask, np.maximum(va, vb), np.nan), 99)
    dabs = np.nanpercentile(np.abs(np.where(mask, diff, np.nan)), 99)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, field, title, cmap, vlim in (
        (ax[0], np.where(mask, vb, np.nan), f"{args.label_ref}", "viridis", (0, vmax)),
        (ax[1], np.where(mask, va, np.nan), f"{args.label_test}", "viridis", (0, vmax)),
        (ax[2], np.where(mask, diff, np.nan), "test − reference", "RdBu_r", (-dabs, dabs)),
    ):
        im = a.imshow(field, origin="lower", cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        a.set_title(title)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, label="m/yr")
    fig.suptitle(f"Surface speed @ t={ta_sel:g}  —  RMSE={rmse:.2f} m/yr ({rel:.1f}%)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
