#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Minimal inversion-accuracy diagnostic.

At the end of a thickness inversion this output module compares the inverted
state against the synthetic "truth" stored in the input file:

  - thickness : state.thk        vs  state.true_thk
  - velocity  : |state.uvelsurf, state.vvelsurf|  vs  |obs|

It prints the thickness and surface-speed RMSE (over the ice mask) to screen
and saves two comparison figures (reference / modelled / difference / residual
histogram), mimicking the speed_magnitude_comparison.png and
thickness_comparison.png produced by the full ``compare_latest_inversion.py``
diagnostic script.
"""

import numpy as np
import matplotlib.pyplot as plt


def _np(state, name):
    """Fetch a state field as a 2D numpy array (or None if absent)."""
    if not hasattr(state, name):
        return None
    return np.asarray(getattr(state, name), dtype=float)


def _rmse(diff, mask):
    d = diff[mask]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, 0
    return float(np.sqrt(np.mean(d ** 2))), int(d.size)


def _comparison_figure(fname, ref, mod, mask, extent, titles, field_label,
                       diff_label, cmap):
    """Reference / modelled / difference maps + residual histogram (2x2)."""
    ref_m = np.where(mask, ref, np.nan)
    mod_m = np.where(mask, mod, np.nan)
    diff = np.where(mask, mod - ref, np.nan)
    res = diff[np.isfinite(diff)]
    rmse = float(np.sqrt(np.mean(res ** 2))) if res.size else np.nan
    bias = float(np.mean(res)) if res.size else np.nan

    vmin = np.nanmin([np.nanmin(ref_m), np.nanmin(mod_m)])
    vmax = np.nanmax([np.nanmax(ref_m), np.nanmax(mod_m)])
    dmax = np.nanmax(np.abs(diff)) if res.size else 1.0

    fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    panels = [
        (ax[0, 0], ref_m, titles[0], cmap, vmin, vmax, field_label),
        (ax[0, 1], mod_m, titles[1], cmap, vmin, vmax, field_label),
        (ax[1, 0], diff, "Difference\nRMSE=%.2f, Bias=%.2f" % (rmse, bias),
         "RdBu_r", -dmax, dmax, diff_label),
    ]
    for a, data, title, cm, lo, hi, label in panels:
        im = a.imshow(data, origin="lower", extent=extent, cmap=cm,
                      vmin=lo, vmax=hi, aspect="equal")
        a.set_title(title)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, label=label)

    ax[1, 1].hist(res, bins=40)
    ax[1, 1].axvline(0.0, color="k", ls="--", lw=1)
    ax[1, 1].set_title("Residuals (modelled - reference)")
    ax[1, 1].set_xlabel(diff_label)
    ax[1, 1].set_ylabel("count")

    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(" inverse_accuracy: wrote figure to %s" % fname)


def initialize(cfg, state):
    pass


def run(cfg, state):
    thk = _np(state, "thk")
    true_thk = _np(state, "true_thk")
    u, v = _np(state, "uvelsurf"), _np(state, "vvelsurf")
    uobs, vobs = _np(state, "uvelsurfobs"), _np(state, "vvelsurfobs")
    icemask = _np(state, "icemask")

    if any(a is None for a in (thk, true_thk, u, v, uobs, vobs)):
        print(" inverse_accuracy: missing required fields, skipping diagnostic")
        return

    # Ice mask: where there is (true) ice.
    ice = true_thk > 0
    if icemask is not None:
        ice = ice & (icemask > 0.5)

    # --- thickness ---
    thk_diff = thk - true_thk
    thk_rmse, n_thk = _rmse(thk_diff, ice)

    # --- surface speed ---
    speed_mod = np.hypot(u, v)
    speed_obs = np.hypot(uobs, vobs)
    vel_diff = speed_mod - speed_obs
    vel_mask = ice & np.isfinite(speed_obs) & np.isfinite(speed_mod)
    vel_rmse, n_vel = _rmse(vel_diff, vel_mask)

    print("")
    print(" === Inversion accuracy (over ice mask) ===")
    print(" Thickness RMSE     : %8.2f m      (n=%d)" % (thk_rmse, n_thk))
    print(" Surface speed RMSE : %8.2f m/yr   (n=%d)" % (vel_rmse, n_vel))
    print("")

    # --- comparison figures ---
    x, y = _np(state, "x"), _np(state, "y")
    if x is not None and y is not None:
        extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    else:
        extent = None

    cfg_ia = cfg.outputs.inverse_accuracy
    _comparison_figure(
        cfg_ia.speed_plot, speed_obs, speed_mod, vel_mask, extent,
        ("Observed |v|", "Modelled |v|"),
        "Surface speed [m/yr]", "Speed error [m/yr]", "turbo",
    )
    _comparison_figure(
        cfg_ia.thickness_plot, true_thk, thk, ice, extent,
        ("True thickness", "Inverted thickness"),
        "Ice thickness [m]", "Thickness error [m]", "cividis",
    )
