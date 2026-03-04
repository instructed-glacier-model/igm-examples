# Aletsch Glacier — Step-by-Step Tutorial

This example provides a progressive, educational tutorial for modeling the Great Aletsch Glacier (Switzerland) with IGM. Each step builds on the previous one, introducing new features and increasing complexity.

**Tip:** Prefix any command with `TF_CPP_MIN_LOG_LEVEL=3` to suppress verbose TensorFlow messages.

## Steps

### Step 1: Simple ELA-based SMB (`params_step1.yaml`)

The simplest setup: an advance-retreat simulation using the built-in `smb_simple` module with time-varying Equilibrium Line Altitudes (ELA). Runs from 1900 to 2000.

```bash
igm_run +experiment=params_step1
```

Results are saved in a timestamped subfolder under `outputs/` (e.g., `outputs/2025-01-15_14-30-00/`). The main output file is `output.nc`, which can be visualized with `ncview`, the NetCDF viewer extension in VS Code, or any NetCDF-compatible tool.

### Step 2: Custom user-defined SMB module (`params_step2.yaml`)

Replaces `smb_simple` with a custom user module (`mysmb`) that computes SMB using a sinusoidal ELA signal. Demonstrates how to write and plug in your own process module (see `user/code/processes/mysmb.py`).

```bash
igm_run +experiment=params_step2
```

Results are saved in a timestamped subfolder under `outputs/`. Open `output.nc` with `ncview` or any NetCDF viewer.

### Step 3: Accumulation/melt SMB model (`params_step3.yaml`)

Uses realistic climate data (1880-2020) with custom modules:
- `clim_aletsch` — loads daily temperature and precipitation data
- `smb_accmelt` — temperature-index mass balance model
- `track_usurf_obs` — compares modelled vs observed surface elevations at 7 historical dates (1880, 1926, 1957, 1980, 1999, 2009, 2017)

```bash
igm_run +experiment=params_step3
```

Results are saved in a timestamped subfolder under `outputs/`. Open `output.nc` with `ncview` or any NetCDF viewer.

### Step 4: With particle tracking (`params_step4.yaml`)

Same as Step 3, but adds particle tracking to visualize ice flow paths using a custom seeding map.

```bash
igm_run +experiment=params_step4
```

Results are saved in a timestamped subfolder under `outputs/`. Open `output.nc` with `ncview` or any NetCDF viewer.

### Step 5: Optuna parameter optimization (`params_step5.yaml`)

Uses the Optuna sweeper to automatically find optimal `weight_accumulation` and `weight_ablation` parameters that minimize the misfit between modelled and observed DEMs (mean STD from `track_usurf_obs`).

```bash
# Single run (no optimization):
igm_run +experiment=params_step5

# With Optuna optimization (50 trials, 4 in parallel):
igm_run -m +experiment=params_step5 hydra/sweeper=igm_optuna
```

Optimization results are stored in `multiruns/`. Each trial output is in a separate subfolder. The Optuna study is persisted in an SQLite database (`optuna.db`). You can visualize the optimization results interactively with:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna.db
```

This opens a web dashboard where you can explore trial histories, parameter importances, and Pareto fronts.

**Optimal parameters found:**

The best trial achieved a cost of `cost_usurf = 30.60` (mean STD between modelled and observed surface elevations across all 7 observation years), with parameters:
- `weight_accumulation = 1.062`
- `weight_ablation = 1.304`

These optimized values are used in Steps 3, 4, and 5 as the default parameters. They were selected because they minimize the overall misfit between the modelled glacier surface and the 7 observed DEMs spanning 1880-2017, providing the best fit to the historical record of the Great Aletsch Glacier.

## Custom Modules

| Module | Description |
|--------|-------------|
| `mysmb` | Simple custom SMB with sinusoidal ELA (Step 2) |
| `clim_aletsch` | Climate data loader for Aletsch (Steps 3-5) |
| `smb_accmelt` | Temperature-index accumulation/melt model (Steps 3-5) |
| `track_usurf_obs` | Compares modelled vs observed surface; computes optimization score (Steps 3-5) |
| `particles` | Custom particle seeding from spatial map (Step 4) |

## Data

Input data is automatically downloaded on first run. It includes:
- `input.nc` — bedrock topography, ice thickness, surface velocities
- `temp_prec.dat` — daily temperature and precipitation (1880-2100)
- `past_surf.nc` — observed surface elevations at 7 dates
- `massbalance.nc` — snow redistribution and direct radiation maps
- `mbparameter.dat` — mass balance parameters
- `bassin.nc` — basin masks
- `seeding.nc` — particle seeding map
