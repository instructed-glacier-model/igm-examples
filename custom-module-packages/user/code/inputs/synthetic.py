#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from igm.inputs.complete_data import complete_data

from synthetic_package import create_bedrock, get_coordinates

def run(cfg, state):

    length_x = 100  # 10 km ?
    length_y = 200  # 20 km ?
    resolution = 100  # m
    
    # Create the X and Y coordinates for the grid
    x, y, X, Y = get_coordinates(length_x, length_y, resolution)
    
    # Create bedrock
    topg = create_bedrock(X, Y)

    # Initialize ice thickness to be 0 everywhere
    thk = np.zeros_like(topg)

    state.x = tf.constant(x.astype("float32"))
    state.y = tf.constant(y.astype("float32"))

    state.topg = tf.Variable(topg.astype("float32"))
    state.thk = tf.Variable(thk.astype("float32"))

    complete_data(state)