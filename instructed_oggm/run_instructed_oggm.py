
"""
    This script shows a simple example of using the "instruction" to run the IGM model
    within the OGGM framework using the IGM_Model2D class. The IGM model is called at
    each time step to compute the ice velocity and the ice thickness evolution. The
    ice flow model is called through the python wrapper of the IGM package.
    
    Code written by: Julien Jehl, Fabien Maussion, and Guillaume Jouvet
"""

### general processes 
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr 

### processes related to oggm & igm
from oggm import cfg, utils, workflow, tasks
from oggm.cfg import G, SEC_IN_YEAR, SEC_IN_DAY
from oggm.core.massbalance import LinearMassBalance
from igm.instructed_oggm import IGM_Model2D

# Choice of a glacier (RGI v7, matching the maintained OGGM prepro used by IGM's
# oggm_shop input module — see igm/inputs/oggm_shop/oggm_util.py).
rgi_ids = ["RGI2000-v7.0-G-11-02596"]  # Aletsch

cfg.initialize(logging_level="WARNING")
cfg.PATHS["working_dir"] = utils.gettempdir(dirname="OGGM-Distrib", reset=True)
# Current IGM/OGGM prepro directories (the old `igm_v1` exp is no longer hosted).
base_url = "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/exps/igm_v4"

gdirs = workflow.init_glacier_directories(
    rgi_ids, prepro_base_url=base_url, from_prepro_level=3, prepro_border=40,
    prepro_rgi_version="70G",
)
gdir = gdirs[0]

with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
    ds = ds.load()

# igm_v4 ships Millan (2022) ice thickness, not the Farinotti consensus.
thick = ds.millan_ice_thickness.where(~ds.millan_ice_thickness.isnull(), 0)
bed  = ds.topo - thick
mask = ds.glacier_mask.data == 1
topo = ds.topo

# Define SMB model
mb = LinearMassBalance(ela_h=2900.0)

# Define IGM model
sdmodel = IGM_Model2D( bed.data, init_ice_thick=thick.data, dx=gdir.grid.dx,
                    mb_model=mb, y0=0, mb_filter=mask, x=ds.x, y=ds.y)

# Run the model
ods = sdmodel.run_until_and_store(100, grid=gdir.grid, print_stdout="My run")

########################

# Plot the results
for i in range(0,ods.ice_thickness.shape[0],10):
    plt.imshow(ods.ice_thickness[i, :])
    plt.colorbar()
    plt.savefig("snapshot" + str(i) + ".png")
    plt.close()
    print(i)
