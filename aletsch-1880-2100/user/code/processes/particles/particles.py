#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf
from netCDF4 import Dataset
import igm 

# take over the official functions
update     = igm.processes.particles.particles.update
# take over the official functions
finalize   = igm.processes.particles.particles.finalize

# define a new initialize function using the official one
def initialize(cfg, state):
    igm.processes.particles.particles.initialize(cfg, state)

    # load the seeding map 
    nc = Dataset( os.path.join(state.original_cwd, "data", 'seeding.nc'), "r" ) 
    state.seeding = np.squeeze( nc.variables["seeding"] ).astype("float32") 
    nc.close()

# customize the seeding_particles function
def seeding_particles(cfg, state):
    """
    User seeding particles in the glacier
    """
 
    I = (state.thk>1)&(state.seeding>0.5)&state.gridseed  # here you may redefine how you want to seed particles

    state.nparticle["x"] = state.X[I] - state.x[0]  # x position of the particle
    state.nparticle["y"] = state.Y[I] - state.y[0]  # y position of the particle
    state.nparticle["z"] = state.usurf[I]           # z position of the particle

    state.nparticle["t"] = tf.ones_like(state.nparticle["x"]) * state.t
    state.nparticle["r"] = (state.nparticle["z"] - state.topg[I]) / state.thk[I]
    state.nparticle["r"] = tf.where(state.thk[I] == 0, tf.ones_like(state.nparticle["r"]), state.nparticle["r"])

    if "weight" in cfg.processes.particles.fields:
        state.nparticle["weight"] = tf.ones_like(state.nparticle["x"])
    if "englt" in cfg.processes.particles.fields:
        state.nparticle["englt"] = tf.zeros_like(state.nparticle["x"])
    if "velmag" in cfg.processes.particles.fields:
        state.nparticle["velmag"] = tf.zeros_like(state.nparticle["x"])


# override the official seeding_particles function
igm.processes.particles.update_particles.seeding_particles_user = seeding_particles
