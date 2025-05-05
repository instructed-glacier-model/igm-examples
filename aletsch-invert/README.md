
# Overview

This setup allows testing of the simplest optimization scheme for inverting an emulated ice flow model. The goal is to find the optimal ice thickness that best explains observational data while remaining consistent with the ice flow emulator, demonstrated here for the case of the Aletsch Glacier.

## Former set-up

- params_1.yaml : Opitmize thk to fit ice thickness and ice velocity
- params_2.yaml : Optimize thk, slidingco, usurf to fit ice thickness, ice velocity, usurf, ...

## Ongoing (improved) setup (5 May 2025)

The inversion scheme remains exploratory and not yet robust. This section gathers a series of notes and preliminary tests, with expected outcomes over the coming month.

- params_arh.yaml : optimize the Arhenius param only to fit surface velocities.
- params_sli.yaml : optimize the sliding coeff only  to fit surface velocities.
- params_thk.yaml : optimize  ice thickness to fit surface velocities.
- params_both.yaml : optimize both sliding coeff and ice thicknes to fit surface velocities and ice thicknes profiles

Here are the list of important parameters, and justification for changing them.

### Emulator Training Strategy

- **Pretrained Emulator vs. Training from scratch:**  
  The ultimate goal is to eliminate the use of a pretrained emulator, thereby avoiding potential dependency of results on the initial training. It was found feasible to retrain the emulator from scratch, provided that:
  - The network is lightweight (e.g., `nb_layers: 6`, `nb_out_filter: 16`)
  - A small number of vertical discretization points is used (`Nz = 2–8`)
  
  Increasing `Nz` impedes convergence, likely due to an identified (though complex) cause. Moreover, given the anisotropic nature of the vertical discretization, using `Nz > 10` adds little value.  
  **Recommendation:** Use `numerics.Nz: 2`, `emulator.pretrained: false`, and a light neural network. Allow a training phase prior to optimization by setting `emulator.warm_up_it: 0` and a large number of initial iterations (`emulator.nbit_init: 1000`) with a strong learning rate (`emulator.lr_init: 0.001`).

### Optimization Loop Design

- **Avoiding Interference Between Loss Terms:**  
  Simultaneously optimizing both (i) the retraining of the emulator to enforce physics and (ii) the reduction of data misfit has produced undesirable results, likely due to internal feedbacks.  
  **Recommendation:** Alternate between phases:
  - Multiple iterations focused on misfit reduction
  - Followed by multiple iterations focused on energy minimization (Blatter–Pattyn)
  
  Suggested parameters:  
  - `emulator.retrain_freq: 100`  
  - `emulator.nbit: 500`  
  This means retraining is performed for 500 steps every 100 outer optimization iterations.

### Obstacle Constraints

- **Constraint Strategy:**  
  Use `processes.data_assimilation.optimization.obstacle_constraint: ['reproject']` instead of the default `['reproject', 'penalty']`.  
  This applies a strong reprojection at each time step but skips the penalty term, which has shown to potentially introduce instability without providing clear benefits in initial tests.

### Parameter Initialization

- **Sliding Coefficient and Arrhenius Factor:**  
  When optimizing parameters like `slidingco` or `arrhenius`, it seems safer to start from values that underestimate velocities.  
  **Recommendation:**  
  - Use high values for `slidingco` (e.g., `init_slidingco: 0.1`)  
  - Use low values for `arrhenius` (e.g., `init_arrhenius: 10`)  
  This encourages the emulator to converge monotonically toward a more dynamic ice flow regime.

### Learning Rate and Normalization

- **Disable Learning Rate Decay:**  
  To avoid unintended slowdowns, disable learning rate decay:
  - `iceflow.emulator.lr_decay: 1.0`
  - `data_assimilation.optimization.step_size_decay: 1.0`

- **Perturbation during Training:**  
  Set `data_assimilation.optimization.pertubate: True` to allow retraining not only on the current input but also on perturbed versions.  
  For instance, if optimizing `slidingco`, this setting trains the emulator on `slidingco - ε`, `slidingco`, and `slidingco + ε`, enhancing its sensitivity to control parameter changes.

### Anisotropic Smoothing and Convexity Weight

- **Anisotropic Smoothing (`smooth_anisotropy_factor`)**:  
  Current default (`0.2`) causes chessboard artifacts in ice thickness.  
  **Recommendation:** Set to `1.0` to deactivate anisotropic smoothing.

- **Convexity Weight (`convexity_weight`)**:  
  This parameter is highly empirical and unintuitive, particularly in the absence of data.  
  **Recommendation:** Deactivated by setting to `0`.
  
###  Normalization Fix (`fix_opti_normalization_issue`)
  A normalization inconsistency exists between sum-based and mean-based cost terms.  
  **Recommendation:** Set to `true` to enforce consistent normalization (means).  
  **Note:** This will require increasing regularization parameters:
  - Thickness: `~10³`
  - Sliding coefficient: `~10¹⁰`

###  Sliding Coefficient Transformation (`log_slidingco`)
  Optimization will operate on the square root of the scaled `slidingco` instead of the raw value.  
  This implicitly enforces positivity and improves stability.  
  **Recommendation:** Enabled by setting to `true`.  
  **Note:** This affects the scaling (around `1e−6`) of `slidingco`.
  
# References (data)

@article{millan2019mapping,
  title={Mapping surface flow velocity of glaciers at regional scale using a multiple sensors approach},
  author={Millan, Romain and Mouginot, J{\'e}r{\'e}mie and Rabatel, Antoine and Jeong, Seongsu and Cusicanqui, Diego and Derkacheva, Anna and Chekki, Mondher},
  journal={Remote Sensing},
  volume={11},
  number={21},
  pages={2498},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}

@article{grab2021ice,
  title={Ice thickness distribution of all Swiss glaciers based on extended ground-penetrating radar data and glaciological modeling},
  author={Grab, Melchior and Mattea, Enrico and Bauder, Andreas and Huss, Matthias and Rabenstein, Lasse and Hodel, Elias and Linsbauer, Andreas and Langhammer, Lisbeth and Schmid, Lino and Church, Gregory and others},
  journal={Journal of Glaciology},
  volume={67},
  number={266},
  pages={1074--1092},
  year={2021},
  publisher={Cambridge University Press}
}

@article{linsbauer2021new,
  title={The new Swiss Glacier Inventory SGI2016: From a topographical to a glaciological dataset},
  author={Linsbauer, Andreas and Huss, Matthias and Hodel, Elias and Bauder, Andreas and Fischer, Mauro and Weidmann, Yvo and B{\"a}rtschi, Hans and Schmassmann, Emanuel},
  journal={Frontiers in Earth Science},
  pages={774},
  year={2021},
  publisher={Frontiers}
}


