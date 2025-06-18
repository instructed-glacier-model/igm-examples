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
    state.nparticle_x = state.X[I] - state.x[0]  # x position of the particle
    state.nparticle_y = state.Y[I] - state.y[0]  # y position of the particle
    state.nparticle_z = state.usurf[I]  # z position of the particle
    state.nparticle_r = tf.ones_like(state.X[I])  # relative position in the ice column
    state.nparticle_w = tf.ones_like(state.X[I])  # weight of the particle
    state.nparticle_t = (tf.ones_like(state.X[I])*state.t)  # "date of birth" of the particle (useful to compute its age)
    state.nparticle_englt = tf.zeros_like(state.X[I])  # time spent by the particle burried in the glacier
    state.nparticle_thk = state.thk[I]  # ice thickness at position of the particle
    state.nparticle_topg = state.topg[I]  # z position of the bedrock under the particle

# override the official seeding_particles function
igm.processes.particles.update_tf.seeding_particles = seeding_particles
