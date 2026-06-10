#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""

Track observation of Aletsch, compare to existing dem

==============================================================================

Input: ---
Output: ----
"""

# Import the most important libraries
import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from netCDF4 import Dataset


def initialize(cfg,state):

    nc = Dataset(os.path.join(state.original_cwd, "data", 'past_surf.nc'), "r" )

    for y in [1880,1926,1957,1980,1999,2009,2017]:
        vars(state)['surf_'+str(y)] = np.squeeze( nc.variables['surf_'+str(y)] ).astype("float32")
    for v in nc.variables:
        if v not in ['x','y']:
            vars(state)[v] = tf.Variable(np.squeeze( nc.variables[v] ).astype("float32"))
    nc.close()

    state.usurf = vars(state)['surf_'+str(int(cfg.processes.time.start))]
    # Clamp to non-negative thickness: the observed 1880 surface can sit a few
    # metres below the bedrock estimate, giving spurious negative thk. The new
    # dahunet emulator builds log((grad_s)^3 * thk^3 + ...) features, so a
    # negative thk makes the argument of log() negative -> NaN velocities.
    state.thk   = tf.maximum(state.usurf - state.topg, 0.0)
    state.usurf = state.topg + state.thk

    state.track_stds = []
    state.obs_years = [1880,1926,1957,1980,1999,2009,2017]
    state.obs_next_idx = 0
    state.t_prev = float(cfg.processes.time.start)

def update(cfg,state):

    while state.obs_next_idx < len(state.obs_years):
        y = state.obs_years[state.obs_next_idx]
        if state.t < y:
            break
        state.obs_next_idx += 1

        diff = (state.usurf-vars(state)['surf_'+str(y)]).numpy()
        diff = diff[state.thk>1]
        mean  = np.mean(diff)
        std   = np.std(diff)
        vol   = np.sum(state.thk) * (state.dx ** 2) / 10 ** 9
        print(" Check modelled vs observed surface at time : %8.0f (obs year %d) ; Mean discr. : %8.2f  ;  Std : %8.2f |  Ice volume : %8.2f " \
                % (state.t, y, mean, std, vol) )

        state.track_stds.append(std)


def finalize(cfg,state):
    if hasattr(state, 'track_stds') and len(state.track_stds) > 0:
        mean_std = float(np.mean(state.track_stds))
        print(" Mean STD across all observation years : %8.2f " % mean_std)
        if not hasattr(state, 'score'):
            state.score = {}
        state.score['cost_usurf'] = mean_std
