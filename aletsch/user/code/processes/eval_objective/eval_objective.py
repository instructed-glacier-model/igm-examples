#!/usr/bin/env python3
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Compute misfit scores at the end of a time_relaxation run.

Exposes three RMSEs as ``state.score`` for IGM's Hydra-Optuna sweeper:

- ``rmse_divflux_minus_amb`` : |∇·q − amb| over icemask (m/yr). Drops to
  zero at steady state; equivalent to modelled dhdt matching dhdt_obs.
- ``rmse_vel`` : modelled vs observed surface speed (m/yr), restricted to
  icemask cells with finite, positive velocity observations.
- ``rmse_thk`` : modelled ice thickness vs ``thkobs`` (m), restricted to
  cells where ``thkobs`` is finite (e.g. radar-profile lines).

Only ``finalize`` produces the scores; ``initialize``/``update`` are no-ops
because ``time_relaxation`` runs the whole forward loop inside its own
``initialize`` and leaves the final fields in ``state``.
"""

import numpy as np
import tensorflow as tf


def initialize(cfg, state):
    pass


def update(cfg, state):
    pass


def finalize(cfg, state):
    mask = state.icemask.numpy() > 0.5

    resid_df = (state.divflux.numpy() - state.amb.numpy())[mask]
    rmse_df = float(np.sqrt(np.mean(resid_df ** 2))) if resid_df.size else float("nan")

    if hasattr(state, "velsurf_magobs"):
        vmod = tf.sqrt(state.uvelsurf ** 2 + state.vvelsurf ** 2).numpy()
        vobs = state.velsurf_magobs.numpy()
        sel_v = mask & np.isfinite(vobs) & (vobs > 0.0)
        if sel_v.any():
            rmse_v = float(np.sqrt(np.mean((vmod[sel_v] - vobs[sel_v]) ** 2)))
        else:
            rmse_v = float("nan")
    else:
        rmse_v = float("nan")

    if hasattr(state, "thkobs"):
        thk = state.thk.numpy()
        tobs = state.thkobs.numpy()
        sel_t = np.isfinite(tobs) & (tobs >= 0.0)
        if sel_t.any():
            rmse_t = float(np.sqrt(np.mean((thk[sel_t] - tobs[sel_t]) ** 2)))
        else:
            rmse_t = float("nan")
    else:
        rmse_t = float("nan")

    # NaN scores confuse Optuna / NSGA sorting — replace with a large
    # penalty so a broken trial is dominated but not filtered out silently.
    penalty = float(cfg.processes.eval_objective.nan_penalty)
    rmse_df = rmse_df if np.isfinite(rmse_df) else penalty
    rmse_v  = rmse_v  if np.isfinite(rmse_v)  else penalty
    rmse_t  = rmse_t  if np.isfinite(rmse_t)  else penalty

    state.score = {
        "rmse_divflux_minus_amb": rmse_df,
        "rmse_vel": rmse_v,
        "rmse_thk": rmse_t,
    }

    print(f"[eval_objective] scores: "
          f"rmse_div-amb={rmse_df:.3f}  "
          f"rmse_vel={rmse_v:.2f}  "
          f"rmse_thk={rmse_t:.2f}")
