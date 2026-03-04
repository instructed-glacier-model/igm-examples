# Aletsch Glacier — Inverse Modeling and Data Assimilation

**Note:** This example is outdated and under revision. Data assimilation in IGM is work in progress.

This example demonstrates IGM's data assimilation module, which finds optimal ice thickness, surface elevation, and/or basal sliding parameters that best explain observational data while remaining consistent with the ice flow emulator. Applied here to the Great Aletsch Glacier.

**Tip:** Prefix any command with `TF_CPP_MIN_LOG_LEVEL=3` to suppress verbose TensorFlow messages.

## Configurations

### `params_1.yaml` — Optimize ice thickness only

Inverts for ice thickness (`thk`) by fitting observed surface velocities, ice thickness profiles, and the glacier mask. This is the simplest inversion setup.

```bash
igm_run +experiment=params_1
```

- **Control variables**: `thk`
- **Cost components**: `velsurf`, `icemask`, `thk`
- **Iterations**: 1000

### `params_2.yaml` — Joint optimization of thickness, sliding, and surface

Simultaneously optimizes ice thickness, basal sliding coefficient, and surface elevation. Adds surface elevation misfit and flux divergence constraints for a more complete inversion.

```bash
igm_run +experiment=params_2
```

- **Control variables**: `thk`, `slidingco`, `usurf`
- **Cost components**: `velsurf`, `icemask`, `thk`, `usurf`, `divfluxfcz`
- **Iterations**: 1000

## How it works

The `data_assimilation` module minimizes a cost function that balances:
- **Data fitting**: misfit between modelled and observed surface velocities, ice thickness, and surface elevation
- **Regularization**: enforces spatial smoothness and physical plausibility of the optimized fields
- **Constraints**: non-negative ice thickness, glacier mask boundaries

The optimization uses the ADAM optimizer with adaptive learning rate decay. The ice flow emulator is retrained during optimization to maintain physical consistency.

## Outputs

Results are saved in a timestamped subfolder under `outputs/`. Key output files:
- `geology-optimized.nc` — optimized ice thickness, sliding coefficient, and surface elevation
- `optimize.nc` — full iteration history
- `convergence.png` — convergence plot of cost functions
- `costs.dat` — cost function values per iteration
- `rms_std_vol.dat` — RMS error, standard deviation, and ice volume evolution
- `vtp/*.vtp` — 3D visualization files for ParaView

## Data

Input data (`input.nc`) includes observed surface velocities (Millan et al., 2019), ice thickness profiles (Grab et al., 2021), and glacier outlines (Linsbauer et al., 2021).

## References

- Millan, R. et al. (2019). Mapping surface flow velocity of glaciers at regional scale using a multiple sensors approach. *Remote Sensing*, 11(21), 2498.
- Grab, M. et al. (2021). Ice thickness distribution of all Swiss glaciers based on extended ground-penetrating radar data and glaciological modeling. *Journal of Glaciology*, 67(266), 1074-1092.
- Linsbauer, A. et al. (2021). The new Swiss Glacier Inventory SGI2016: From a topographical to a glaciological dataset. *Frontiers in Earth Science*, 774.
