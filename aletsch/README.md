# Aletsch Glacier — Step-by-Step Tutorial

This example provides a progressive, educational tutorial for modeling the Great Aletsch Glacier (Switzerland) with IGM. Each step builds on the previous one, introducing new features and increasing complexity.

**Tip:** Prefix any command with `TF_CPP_MIN_LOG_LEVEL=3` to suppress verbose TensorFlow messages.

## Steps

### Step 0: Fully offline (pretrained) emulator (`params_step0.yaml`)

The same simple ELA-based advance-retreat simulation as Step 1 (built-in `smb_simple`, 1900-2000), but run with a **fully offline iceflow emulator**. Instead of retraining the neural-network emulator on-the-fly during the simulation, this step loads a pretrained network (`dahunet_mini.keras`) and uses it as-is — no retraining ever happens.

This is configured in the `iceflow` block via the `unified` method with `mapping: network`: the warm-up and initialisation iteration counts are disabled (`nbit_warmup: -1`, `nbit_init: -1`, `nbit: -1`), while `network.pretrained: true` loads the shipped weights. Because there is no in-the-loop training, this is typically **much quicker** than the on-the-fly emulator used in the other steps, at the cost of relying entirely on the generality of the pretrained network.

```bash
igm_run +experiment=params_step0
```

Results are saved in a timestamped subfolder under `outputs/`. Open `output.nc` with `ncview` or any NetCDF viewer.

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
igm_run -m +experiment=params_step5 \
        hydra/sweeper=igm_optuna \
        hydra.sweeper.optuna_config=optuna_1obj_params.yaml
```

The single-objective optimization settings (sampler, parameter ranges, study name, ...) live in `optuna_1obj_params.yaml`. Optimization results are stored in `multiruns/`, with each trial output in a separate subfolder. The Optuna study is persisted in an SQLite database (`optuna.db`). You can visualize the optimization results interactively with:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna.db
```

This opens a web dashboard where you can explore trial histories, parameter importances, and Pareto fronts.

**Optimal parameters found:**

The best trial achieved a cost of `cost_usurf = 30.60` (mean STD between modelled and observed surface elevations across all 7 observation years), with parameters:
- `weight_accumulation = 1.062`
- `weight_ablation = 1.304`

These optimized values are used in Steps 3, 4, 5, and 6 as the default parameters. They were selected because they minimize the overall misfit between the modelled glacier surface and the 7 observed DEMs spanning 1880-2017, providing the best fit to the historical record of the Great Aletsch Glacier.

### Step 6: Multi-objective Optuna optimization (`params_step6.yaml`)

Same model as Step 5, but optimizes **two** objectives simultaneously:

- `cost_usurf`  — mean STD between modelled and observed DEMs (as in Step 5)
- `cost_velsurf` — RMSE between modelled and observed surface **speeds** at the last time step (2020), evaluated where the InSAR velocity field is available

The velocity score is computed by a new dedicated user module, `track_velsurf_obs`, which compares modelled and observed surface speeds at the final time step following the spirit of the `misfit_velsurf` cost term of IGM's `data_assimilation` module (threshold on observed speed, valid pixels only).

A dedicated config `optuna_2obj_params.yaml` declares the two objectives and switches the sampler from **TPE** (used in Step 5) to **NSGA-II**, a genetic algorithm well suited for Pareto-front exploration. It adds `tau_ref` as a third free parameter, sweeping it over the physically meaningful range `[0.08, 0.50] MPa` (the Weertman reference velocity is fixed to `u_ref = 35 m/yr` so that `tau_ref` *is* the basal shear stress at a typical Aletsch trunk speed).

```bash
# Single run (no optimization):
igm_run +experiment=params_step6

# With NSGA-II multi-objective optimization (200 trials, 4 in parallel):
igm_run -m +experiment=params_step6 \
        hydra/sweeper=igm_optuna \
        hydra.sweeper.optuna_config=optuna_2obj_params.yaml
```

The study is persisted in `optuna_2obj.db` (study name `aletsch_2obj`) so you can visualize it with `optuna-dashboard sqlite:///optuna_2obj.db`.

**Pareto front plot.** A standalone script in `tools/` reads the SQLite database and draws the Pareto front:

```bash
python tools/plot_pareto_front.py
# or with explicit arguments:
python tools/plot_pareto_front.py --storage sqlite:///optuna_2obj.db \
                                  --study   aletsch_2obj \
                                  --out     pareto_front.png --show
```

It prints all Pareto-optimal trials (number, both objective values, parameters) and saves a scatter plot with the non-dominated set highlighted, so you can pick the trade-off that best fits your modelling goal.

**Inspecting the spatial misfits of a chosen trial.** Each Step 6 trial — both single runs and trials produced by `-m hydra/sweeper=igm_optuna` — writes an `output.nc` (decadal snapshots) and an `output_ts.nc` time series in its working directory. To compare modelled vs.\ observed fields in space for a chosen trial, point `tools/plot_misfit_maps.py` at that directory:

```bash
# single run (saved under outputs/<timestamp>):
python tools/plot_misfit_maps.py --run outputs/<timestamp>

# a Pareto-optimal trial from the optimization (saved under multirun/...):
python tools/plot_misfit_maps.py --run multirun/<date>/<trial_number>
```

For a higher-resolution replay (annual snapshots) of one specific best-trial parameter set, copy `experiment/params_step6.yaml` to a custom file, bake in the chosen `(weight_accumulation, weight_ablation, init_slidingco)` values, and set `processes.time.save: 1.0`.

### Step 7: Inverting for ice thickness from surface velocities (`params_step7.yaml`)

A standalone **inversion** example: rather than running the glacier forward in time, this step uses IGM's `field_inversion` assimilation module to **invert for the spatially-varying ice thickness** (`thk`) that best reproduces a field of observed surface velocities.

The inversion minimises an objective combining:

- a **misfit** term (`velsurf`, Huber loss) between modelled (`uvelsurf`, `vvelsurf`) and observed (`uvelsurfobs`, `vvelsurfobs`) surface velocity components, and
- a **regularization** term on `thk` (a squared-Laplacian smoothness penalty referenced to the surface `usurf`),

while bounding the thickness to the physically plausible range `[0, 1000] m`. The forward velocities are produced by the `unified` iceflow emulator (here using the L-BFGS optimizer with a Hager-Zhang line search and the shipped pretrained `dahunet_mini.keras` network).

The input file (`inverse_250m.nc`) is **synthetic**: it was generated by running IGM forward with a known ("true") thickness field, recording the resulting surface velocities, and storing both. The recorded velocities act as synthetic observations for the inversion, and the true thickness field lets you validate the result — the `inverse_accuracy` output compares the inverted thickness against the truth. Note that `tau_ref` is fixed to the value used when generating the input file, since here we invert for thickness only, not the sliding parameter.

```bash
igm_run +experiment=params_step7
```

Results are saved in a timestamped subfolder under `outputs/`. The run writes the inverted fields to `output.nc` (open with `ncview` or any NetCDF viewer) alongside the `inverse_accuracy` diagnostics comparing inverted and true thickness. The simulation also produces two figures comparing thickness and surface velocity misfit.

## Visualization tools

Two standalone post-processing scripts live in `tools/` and are independent of IGM itself (they only depend on `numpy`, `matplotlib`, `netCDF4`, and `optuna`):

| Script | Purpose |
|--------|---------|
| `tools/plot_pareto_front.py` | Read the Optuna SQLite study and draw the Pareto front (default: `optuna_2obj.db` / study `aletsch_2obj`). |
| `tools/plot_misfit_maps.py` | From a saved IGM run (`output.nc`), plot model−obs DEM differences at the seven observation years and surface speeds (observed, modelled, difference) at the final time step. |

Both scripts accept `--help` for the full list of options.

## Custom Modules

| Module | Description |
|--------|-------------|
| `mysmb` | Simple custom SMB with sinusoidal ELA (Step 2) |
| `clim_aletsch` | Climate data loader for Aletsch (Steps 3-6) |
| `smb_accmelt` | Temperature-index accumulation/melt model (Steps 3-6) |
| `track_usurf_obs` | Compares modelled vs observed surface; writes `cost_usurf` to `state.score` (Steps 3-6) |
| `track_velsurf_obs` | Compares modelled vs observed surface speeds at the final time step; writes `cost_velsurf` to `state.score` (Step 6) |
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
