# Drønbreen — polythermal glacier with the enthalpy module

**Modelling the thermal structure of a Svalbard polythermal glacier, and calibrating it
against ground-penetrating radar.**

Drønbreen (Nordenskiöld Land, Svalbard) is a Scandinavian-type polythermal glacier: a
layer of temperate ice at the bed, overlain by cold ice. The boundary between the two —
the **cold–temperate transition surface (CTS)** — has been mapped along ten
ground-penetrating radar (GPR) lines, which makes this glacier a rare case where a
modelled thermal structure can be checked directly against observations.

The example runs in two steps:

1. **A thermal spin-up** on a fixed geometry, with the `enthalpy` module coupled to ice
   flow and to till hydrology.
2. **An Optuna calibration** of the surface thermal forcing, minimising the RMSE between
   the modelled and the observed CTS.

**Tip:** Prefix any command with `TF_CPP_MIN_LOG_LEVEL=3` to suppress verbose TensorFlow
messages. Results are written to a timestamped subfolder under `outputs/`, or to the
directory passed via `hydra.run.dir=`.

## Requirements

IGM, plus `optuna`, `pandas`, `scipy`, `netCDF4` and `xarray` for the calibration and
the post-processing scripts. `optuna-dashboard` is optional.

## Usage

### Step 1 — Forward modelling with the enthalpy module

```bash
igm_run +experiment=params_step1_spinup
```

A spin-up from 1900 to 2022 under a constant CARRA-derived climate, on a **fixed
geometry**: there is no `thk` process, so the ice surface never moves and only the
thermal state evolves. Roughly 2 minutes on a GPU.

122 years is more than enough: the modelled CTS settles within about 70 years of the
start, so the glacier is in thermal equilibrium well before 2022. Starting in 1600
instead — as one might expect a "spin-up" to — costs four times the compute and changes
the misfit by under 2%.

On a real polythermal glacier of this type, most temperate ice forms in the
accumulation area, where meltwater percolates into the firn and refreezes, releasing
latent heat. IGM does not model firn refreezing explicitly, so this example uses a
deliberate proxy: **a constant air temperature is imposed above the equilibrium line**.

```yaml
processes:
  clim_carra_ela:
    ela: 666.0         # m a.s.l.
    T_ela: 0.42        # °C imposed above it
```

These values, and the geothermal flux, come from the Step 2 calibration. They are a
tuned surface forcing, not a measured temperature.

Then plot the modelled thermal structure against the radar observations:

```bash
python tools/plot_cts_sections.py --run outputs/<date>/<time>          # one radar line
python tools/plot_cts_sections.py --run outputs/<date>/<time> --all    # every radar line
```

`plot_cts_sections.py` draws a vertical section along a radar line: modelled temperate
ice in red, modelled cold ice in blue, and the observed CTS over them as a dashed black
line. The modelled CTS is the boundary between the two fills, so where the dashed line
runs through blue the model made too little temperate ice, and through red, too much.
With no `--line` it picks a transect that both crosses a transition and fits about as
well as the survey median — several lines cross almost entirely cold or entirely
temperate ice, and the most sharply divided one happens to be the worst-fitting.

Transects are labelled `T1`–`T10` by how much temperate ice the radar found, so `T1` is
the most temperate line of the survey. Running `--all` is worth doing once: with the
shipped parameters the model reproduces about 89% of the observed temperate ice overall,
but that aggregate hides real differences between lines — two of them come out almost
entirely cold where the radar sees a clear temperate layer.

### Step 2 — Calibrating the thermal forcing with Optuna

```bash
igm_run -m \
    +experiment=params_step2_optuna \
    hydra/sweeper=igm_optuna \
    hydra.sweeper.optuna_config=optuna/optuna_cts.yaml
```

Each trial runs the Step 1 model with a new combination of parameters, extracts the
modelled CTS along every radar line, and returns the mean RMSE as `cost_cts`. Optuna
proposes the next combination with a Tree-structured Parzen Estimator.

One trial is a full Step 1 run, so budget a couple of minutes each; three run
concurrently, putting a 40-trial sweep at roughly half an hour on a single GPU. Set
`n_trials` and `n_jobs` in `optuna/optuna_cts.yaml` — they cannot be overridden on the
command line, because Hydra treats `optuna_config` as a path rather than a config group.

Inspect the result:

```bash
python tools/plot_optuna.py --storage sqlite:///optuna_cts.db --study dronbreen_cts
python tools/plot_cts_sections.py --best-of sqlite:///optuna_cts.db
optuna-dashboard sqlite:///optuna_cts.db     # optional, interactive
```

`plot_optuna.py` prints the best parameters and plots the cost against each of them.
`plot_cts_sections.py --best-of` finds the lowest-cost trial and draws its section, so
the best run can be inspected without looking up its trial number.

A 64-trial sweep reaches **18.8 m**, and those best-fit values are what
`params_step1_spinup.yaml` ships. Only the ELA is really constrained, though: of the 64
trials, 30 land within 1 m of the best misfit, and across that near-optimal set `ela`
spans just 20% of its search range while `T_ela` and the geothermal flux each span
over 70%. The two best trials tie at 18.81 m with `T_ela` of 1.44 vs 0.42 °C — the
radar fixes the altitude of the thermal transition and the total heat budget, but not
how the heat is split between refreezing above and geothermal flux below.

## Calibrated parameters

| Parameter | Range searched | Meaning |
|---|---|---|
| `processes.clim_carra_ela.T_ela` | −0.5 to 2 °C | Temperature imposed above the ELA — the refreezing proxy. Heat from above |
| `processes.clim_carra_ela.ela` | 650 to 720 m | Altitude above which it applies |
| `processes.enthalpy.thermal.basal_heat_flux_ref` | 0.030 to 0.080 W m⁻² | Geothermal flux — heat from below. Poorly known in Svalbard |

### How to tell whether a misfit is any good

`cost_cts` is a mean RMSE in metres, which means nothing on its own. Before trusting one,
ask what a model that knows nothing would score. That check matters here because 60% of
the radar points sit on cold ice, where the observation is really "there is no CTS here"
rather than a measured surface — if those dominated, a trivially cold glacier would score
well and the calibration would be measuring nothing.

| | `cost_cts` |
|---|---|
| All cold — CTS pinned to the bed | 37.4 m |
| All temperate | 89.9 m |
| Calibrated model | ≈19 m |

These numbers are for this setup; what matters is the ratio. Roughly halving the error of
the best null says the objective carries real information — had the calibrated run landed
near 37 m it would have been no better than assuming the glacier is frozen throughout.

Splitting the misfit by error mode goes one step further: it averages about 27 m where
temperate ice is observed against 14 m where the ice is observed cold, so the headline
number flatters the model on the quantity that matters most, the position of the CTS
where one exists.

## Key model parameters

| Parameter | Value | Note |
|---|---|---|
| `processes.enthalpy.numerics.Nz` | 30 | Thermal grid; must be fine enough to resolve the CTS |
| `processes.enthalpy.numerics.vert_spacing` | 4.0 | Packs layers towards the bed |
| `processes.enthalpy.thermal.basal_heat_flux_ref` | 0.044 W m⁻² | Calibrated in Step 2 but only weakly (see above); near the Svalbard estimate of 0.040 (Schäfer et al., 2012). IGM's default is 0.065 |
| `processes.iceflow.numerics.Nz` | 2 | MOLHO resolves the vertical profile analytically |
| `processes.iceflow.physics.sliding.tau_ref` | 0.325 MPa | Basal shear stress at `u_ref` |
| `processes.subglacial_hydrology.mode` | `till_storage` | Tulaczyk till hydrology (Bueler & van Pelt, 2015) |

The two vertical grids are independent: ice flow needs only two degrees of freedom per
column under MOLHO, while the thermal state needs thirty to place the CTS accurately.

`subglacial_hydrology` in `till_storage` mode requires `enthalpy` to run **before** it,
which is why the process order in the experiment files is not alphabetical.

## Data

| File | Contents |
|---|---|
| `data/input_dronbreen.nc` | 81 × 91 grid at 100 m, EPSG:32633. Surface DEM (Norwegian Polar Institute, 2024), bed from kriged GPR, ice mask, and CARRA 1990–2024 mean annual / mean summer air temperature |
| `data/cts_gpr/*.csv` | Ten GPR lines surveyed on 28–29 March 2022 (100 MHz), giving bed elevation and CTS elevation with uncertainty bounds |

Everything ships with the repository; nothing is downloaded.

The radar data are from Mannerfelt et al.,
<https://zenodo.org/doi/10.5281/zenodo.17882300>, trimmed here to the columns the example uses.
Please cite the original dataset if you use them. In those files `temperate_elevation`
is `bed_elevation + temperate`, where `temperate` is the thickness of the temperate
layer; note that `temperate_upper ≤ temperate ≤ temperate_lower`, the opposite of the
`thickness_*` convention in the same dataset.

## Files

```
enthalpy-dronbreen/
├── data/                 # input NetCDF and GPR observations
├── experiment/           # the two experiment configurations (+ common/ fragment)
├── optuna/               # sweeper configuration for Step 2
├── tools/                # post-processing scripts, and the shared CTS diagnostic
└── user/
    ├── code/processes/   # clim_carra_ela, track_cts_err
    └── conf/processes/   # their Hydra config stubs
```

`tools/cts.py` holds the one implementation of "where does the ice stop being cold?",
used both by the in-run scoring module and by the plotting scripts, so that the
calibration and the figures cannot drift apart.

## References

- Aschwanden, A., Bueler, E., Khroulev, C. & Blatter, H. (2012). An enthalpy formulation
  for glaciers and ice sheets. *Journal of Glaciology*, 58(209), 441–457.
  doi:[10.3189/2012JoG11J088](https://doi.org/10.3189/2012JoG11J088)
- Bueler, E. & van Pelt, W. (2015). Mass-conserving subglacial hydrology in the Parallel
  Ice Sheet Model version 0.6. *Geoscientific Model Development*, 8(6), 1613–1635.
  doi:[10.5194/gmd-8-1613-2015](https://doi.org/10.5194/gmd-8-1613-2015)
- Schäfer, M., Zwinger, T., Christoffersen, P., Gillet-Chaulet, F., Laakso, K.,
  Pettersson, R., Pohjola, V. A., Strozzi, T. & Moore, J. C. (2012). Sensitivity of
  basal conditions in an inverse model: Vestfonna ice cap, Nordaustlandet/Svalbard.
  *The Cryosphere*, 6, 771–783.
  doi:[10.5194/tc-6-771-2012](https://doi.org/10.5194/tc-6-771-2012)
- Mannerfelt, E. S. et al. (2025). Draft dataset publication from Mannerfelt et al.
  Zenodo. doi:[10.5281/zenodo.17882300](https://doi.org/10.5281/zenodo.17882300)
