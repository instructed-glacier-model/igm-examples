#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""

Compare modelled surface speeds to the observed surface velocity field at
the final time step of the simulation, and write the RMSE to
``state.score['cost_velsurf']`` so it can be used as an objective by the
Optuna sweeper.

The misfit is computed in the same spirit as the ``misfit_velsurf`` cost
term of IGM's ``data_assimilation`` module: only pixels where both
observed velocity components are valid (non-NaN), where the observed
speed is above a minimum threshold, and where ice is present are kept;
modelled and observed **speed magnitudes** are then compared via RMSE.

Inputs (from state, typically loaded by the ``local`` inputs module from
``input.nc`` and produced by ``iceflow``):
  - state.uvelsurf, state.vvelsurf       (modelled surface velocity)
  - state.uvelsurfobs, state.vvelsurfobs (observed surface velocity)
  - state.thk                            (ice thickness)

Output:
  - state.score['cost_velsurf']          (float, RMSE in m/yr)
"""

import numpy as np


def initialize(cfg, state):
    pass


def update(cfg, state):
    pass


def finalize(cfg, state):
    required = ("uvelsurf", "vvelsurf", "uvelsurfobs", "vvelsurfobs", "thk")
    for name in required:
        if not hasattr(state, name):
            print(f" track_velsurf_obs: missing state.{name}, skipping velocity score")
            return

    threshold = float(cfg.processes.track_velsurf_obs.velsurfobs_thr)

    uv = np.asarray(state.uvelsurf)
    vv = np.asarray(state.vvelsurf)
    uvobs = np.asarray(state.uvelsurfobs)
    vvobs = np.asarray(state.vvelsurfobs)
    thk = np.asarray(state.thk)

    speed_mod = np.sqrt(uv * uv + vv * vv)
    speed_obs = np.sqrt(uvobs * uvobs + vvobs * vvobs)

    mask = (
        ~np.isnan(uvobs)
        & ~np.isnan(vvobs)
        & (speed_obs >= threshold)
        & (thk > 1)
    )

    n = int(np.sum(mask))
    if n == 0:
        print(" track_velsurf_obs: no valid velocity observations, skipping score")
        return

    diff = speed_mod[mask] - speed_obs[mask]
    rmse = float(np.sqrt(np.mean(diff * diff)))

    print(" Final surface speed RMSE (mod vs obs)  : %8.2f m/yr  (n=%d pixels)"
          % (rmse, n))

    if not hasattr(state, "score"):
        state.score = {}
    state.score["cost_velsurf"] = rmse
