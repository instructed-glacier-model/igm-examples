# Aletsch Glacier — IGM Tutorials

A progressive collection of tutorials for modelling the Great
Aletsch Glacier (Switzerland) with IGM. All tutorials share the same
`data/`, `user/` (custom modules) and `tools/` (analysis scripts) folders, and
every configuration lives in a single `experiment/` directory, run with
`igm_run +experiment=<name>`. Files are prefixed by part letter 
(`params_A_*`, `params_B_*`, …) so they list in tutorial order.

**Part A** demonstrates **standard forward modelling** — running the glacier
forward in time — built up in steps of increasing complexity (simple ELA SMB →
custom SMB module → realistic climate/SMB → particle tracking). Steps 1–4 all use
the fast **off-line trained iceflow emulator**; Step 5 repeats Step 1 with the
**on-line retrained iceflow solver** to contrast the two. **Parts B–E** then present four **alternative
strategies for data assimilation**, i.e. for constraining model parameters or
state from observations. Each has different strengths; they are summarised here:

| Part | Strategy | IGM module / tool | What it does |
|------|----------|-------------------|--------------|
| **B** | Hyperparameter tuning | Optuna sweeper | Searches a few scalar model parameters (SMB weights, sliding) that best reproduce the observed DEMs and velocities. |
| **C** | Control optimisation | `data_assimilation` | Inverts the ice thickness from surface velocities. |
| **D** | Control optimisation  | `field_inversion` | nverts the ice thickness from surface velocities. Newer; set to **replace `data_assimilation`** in the long term. |
| **E** | Time relaxation | `time_relaxation` | Nudges several fields jointly inside a transient forward run until model and observations are mutually consistent. |

**Prerequisites**

First install IGM and its dependencies by following the installation steps at
**https://igm-model.org/**. That already provides everything Parts A–E and the
`tools/` scripts need.  

**Tip:** prefix any command with `TF_CPP_MIN_LOG_LEVEL=3` to silence verbose
TensorFlow logging. Results are written to a timestamped subfolder under
`outputs/` (or to the directory you pass via `hydra.run.dir=`). Open `output.nc`
with `ncview`, the VS Code NetCDF viewer, or any NetCDF tool.

---

# Part A — Forward modelling (Steps 1–5)

The standard way to run IGM forward in time. Each step builds on the previous
one, introducing new features. 

### Step 1: Simple ELA-based SMB (`params_A_step1.yaml`)

The simplest setup: we simulate the evolution of the Aletsch Glacier over 100 years (1900–2000) using the built-in `smb` module (method `simple`) with a time-varying Equilibrium Line Altitude (ELA). For simplicity, Steps 1–4 use the off-line trained iceflow emulator (`common: iceflow_offline`), which requires no knowledge of IGM's iceflow machinery — see Step 5 for the emulator-vs-solver distinction.

```bash
igm_run +experiment=params_A_step1
```

The simulation takes between 5 seconds and 1 minute (depending on your set-up) and shows the evolving ice-thickness and speed fields live, thanks to the activated live dashboard.

### Step 2: Custom user-defined SMB module (`params_A_step2.yaml`) 

Replaces `smb` (in `simple` mode) with a custom user module (`mysmb`) that computes
the SMB from a sinusoidal ELA signal. It demonstrates how to write and plug in your
own process module (see `user/code/processes/mysmb.py`).

```bash
igm_run +experiment=params_A_step2
```

Besides the live dashboard, you can explore the simulation output by reading the NetCDF file written by IGM, e.g. with `ncview` or any other NetCDF viewer.

### Step 3: Realistic climate and accumulation/melt SMB (`params_A_step3.yaml`)

To increase the realism of the simulation, we now include realistic climate data (1880–2020) through custom modules:
- `clim_aletsch` — loads daily temperature and precipitation data
- `smb_accmelt` — temperature-index mass balance model
- `track_usurf_obs` — compares modelled vs observed surface elevations at 7 historical dates (1880, 1926, 1957, 1980, 1999, 2009, 2017)

Run it as follows (note that the new modules introduce a computational bottleneck related to the daily time-stepping of the climate and SMB modules, making the simulation a few times more expensive):

```bash
igm_run +experiment=params_A_step3
```

During the simulation, IGM reports the mismatch against the known observed DEMs; the goal is to minimise the standard deviation (STD) of that mismatch, which is achieved after parameter calibration (Part B).

### Step 4: With particle tracking (`params_A_step4.yaml`)

Same as Step 3, but adds particle tracking (simply by adding the `particles` module to the parameter file) to visualise ice-flow paths. Here we use a user-defined particle seeding (a custom user module) to highlight the two main regions responsible for debris production, which directly build the two legendary moraines of the Aletsch Glacier.

```bash
igm_run +experiment=params_A_step4
```

### Step 5: Off-line trained emulator vs on-line retrained solver (`params_A_step5.yaml`)

Identical to Step 1 (the same simple-ELA run, 1900–2000) **except for the
iceflow component**: it uses the **on-line retrained iceflow solver**
(`common: iceflow_online`) instead of the off-line trained emulator. The two differ in what the neural network actually does:

- **Off-line trained iceflow emulator** (Steps 1–4, `iceflow_offline`): a network
  pretrained beforehand and used **frozen** — it *emulates* the ice-flow physics
  learned offline from a large catalogue of modelled glaciers, with no training during the run. This is the easiest option for modelling "standard" mountain glaciers, as it avoids retraining a network or tuning training settings. It should, however, only be used for glaciers similar to those it was trained on: typical mountain-glacier geometries at 50–250 m horizontal resolution, and it currently supports only the 2-layer MOLHO vertical basis.
  
- **On-line retrained iceflow solver** (this step, and Parts B–E, `iceflow_online`): the network is trained from scratch at initialisation and **retrained on the fly** every few time steps to minimise the actual ice-flow energy — so it *solves* the physics for the current state, adapting to the evolving geometry and to whatever sliding/rheology you set. This is the most generic option and can handle essentially any ice flow, including applications far outside the off-line emulator's training domain — such as the Greenland or Antarctic ice sheets, marine-terminating or floating ice, surging glaciers, strongly hydrology-controlled sliding, or very different grid resolutions. It does, however, come with extra numerical parameters that must be tuned, plus additional numerical experiments to validate them (to assess the fidelity against the reference identity mapping).

```bash
igm_run +experiment=params_A_step5
```

---

# Part B — Data assimilation by hyperparameter tuning

The first data-assimilation strategy. Rather than inverting a spatial field, it
treats a few **scalar model parameters** as unknowns and uses the **Optuna**
sweeper to search the values that best reproduce observations. The forward model
is the realistic climate/SMB setup of Part A Steps 3–4 (`clim_aletsch` +
`smb_accmelt` + observation tracking), but with the on-line retrained iceflow
solver (Step 5), run from 1880 to 2020.

### Step 1: Single-objective optimization (`params_B_1obj.yaml`)

Finds the `weight_accumulation` and `weight_ablation` parameters that minimize
the misfit between modelled and observed DEMs (mean STD from `track_usurf_obs`).

```bash 
# Single run (no optimization):
igm_run +experiment=params_B_1obj

# With Optuna optimization (50 trials, 4 in parallel):
igm_run -m +experiment=params_B_1obj \
        hydra/sweeper=igm_optuna \
        hydra.sweeper.optuna_config=optuna/optuna_1obj_params.yaml
```

The single-objective optimization settings live in `optuna/optuna_1obj_params.yaml`.
Results are stored in `multirun/`, and the study is persisted in an SQLite
database (`optuna_1obj.db`). To explore the study interactively, install the Optuna
dashboard — a small web app that is **not bundled with IGM** — and point it at that
database:

```bash
pip install optuna-dashboard          # one-off install
optuna-dashboard sqlite:///optuna_1obj.db 
```

**Optimal parameters found:** the best trial achieved `cost_usurf = XXX` (mean
STD between modelled and observed surface elevations across all 7 observation
years) with `weight_accumulation = 1.84`, `weight_ablation = 2.01`. These
optimized values are used as the defaults in the realistic forward runs
(Part A, Steps 3–4) and as the baseline here.

### Step 2: Multi-objective optimization (`params_B_2obj.yaml`)

Same model as Step 1, but optimizes **two** objectives simultaneously:

- `cost_usurf`  — mean STD between modelled and observed DEMs (as in Step 1)
- `cost_velsurf` — RMSE between modelled and observed surface **speeds** at the last time step (2020), 
where the velocity field is available (computed by the `track_velsurf_obs` user module).

A dedicated config `optuna/optuna_2obj_params.yaml` declares the two objectives,
switches the sampler from **TPE** to **NSGA-II** (a genetic algorithm suited to
Pareto-front exploration), and adds `tau_ref` as a third free parameter swept
over `[0.08, 0.50] MPa` (with `u_ref = 100 m/yr`, so `tau_ref` *is* the basal
shear stress at a typical Aletsch trunk speed).

```bash
# Single run (no optimization):
igm_run +experiment=params_B_2obj

# With NSGA-II multi-objective optimization (200 trials, 4 in parallel):
igm_run -m +experiment=params_B_2obj \
        hydra/sweeper=igm_optuna \
        hydra.sweeper.optuna_config=optuna/optuna_2obj_params.yaml
```

The study is persisted in `optuna_2obj.db` (study `aletsch_2obj`).

**Pareto front plot.** A standalone script reads the SQLite database and draws
the Pareto front:

```bash
python tools/plot_pareto_front.py
# or with explicit arguments:
python tools/plot_pareto_front.py --storage sqlite:///optuna_2obj.db \
                                  --study   aletsch_2obj \
                                  --out     pareto_front.png --show
```

**Inspecting the spatial misfits of a chosen trial.** Each trial writes an
`output.nc` (decadal snapshots) and an `output_ts.nc` time series. To compare
modelled vs observed fields in space, point `tools/plot_misfit_maps.py` at that
directory:

```bash
# single run (saved under outputs/<timestamp>):
python tools/plot_misfit_maps.py --run outputs/<timestamp>

# a Pareto-optimal trial from the optimization:
python tools/plot_misfit_maps.py --run multirun/<date>/<trial_number>
```

---

# Part C — Ice-thickness data assimilation (`data_assimilation` module) 

Recovers the ice thickness from surface observations using IGM's
`data_assimilation` module. The inversion optimizes the ice thickness field so
that modelled surface velocities (Blatter–Pattyn physics) best match observed
velocities, while regularization keeps the solution physically plausible. This
is IGM's current inversion but `field_inversion` will be its successor.

Steps 1–3 use `params_C_offline.yaml` (off-line trained emulator); Step 4 uses
`params_C_online.yaml` (on-line retrained solver). Both share the same
data-assimilation setup:
- **Ice flow**: the `unified` iceflow — the off-line trained emulator in Steps 1–3 (`params_C_offline`), or the on-line retrained solver in Step 4 (`params_C_online`, retrained every 50 iterations)
- **Control variable**: ice thickness (`thk`)
- **Cost function**: surface velocity misfit (`velsurf`) + ice-mask penalty (`icemask`)
- **Optimizer**: ADAM with learning-rate decay, up to 1000 iterations
- **Regularization**: gradient penalty on `thk` (weight `regularization.thk`)

### Step 1 — Single inversion (off-line trained emulator)

```bash
igm_run +experiment=params_C_offline hydra.run.dir=outputs/DA_step1
```

This takes ~15 min on a GPU. Key result files in `outputs/DA_step1/`:
- `geology-optimized.nc` — final ice thickness, velocities and other fields
- `optimize.nc` — optimization history (iterations)


### Step 2 — L-curve analysis (velocity only)

The regularization weight controls the trade-off between fitting the data and
keeping the thickness smooth. The L-curve method plots data misfit against
solution roughness for several weights and looks for the "elbow". Sweep with
Hydra multirun, run in parallel via the joblib launcher (one-off
`pip install hydra-joblib-launcher`, not bundled with IGM):

```bash
igm_run -m +experiment=params_C_offline \
    assimilations.data_assimilation.regularization.thk=1,3,10,30,100,300,1000,3000,10000 \
    hydra/launcher=joblib hydra.launcher.n_jobs=3 \
    hydra.sweep.dir=outputs/DA_step2
```

```bash
python tools/analyze_step1.py outputs/DA_step2
```

This produces `lcurve_step1.png` (L-curve + misfit-vs-regularization). The elbow
(typically reg ≈ 10–30 for Aletsch) indicates the best balance.

### Step 3 — Sliding-coefficient sweep with thickness validation

When thickness observations (GPR) are available, they can validate the inversion
and constrain the sliding coefficient. Sweep `physics.sliding.tau_ref` with the
regularization weight fixed:

```bash
igm_run -m +experiment=params_C_offline \
    processes.iceflow.physics.sliding.tau_ref=0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.35,0.4 \
    hydra/launcher=joblib hydra.launcher.n_jobs=3 \
    hydra.sweep.dir=outputs/DA_step3
```

```bash
python tools/analyze_step2.py outputs/DA_step3 --param processes.iceflow.physics.sliding.tau_ref
```

Produces `sweep_tau_ref.png` (velocity RMSE, thickness RMSE vs GPR, and ice
volume as functions of `tau_ref`). The best coefficient minimizes thickness RMSE
while keeping velocity RMSE acceptable.
 
### Step 4 — Single inversion (on-line retrained solver)

Exactly the same inversion as Step 1, but the ice flow uses the **on-line
retrained iceflow solver** (`params_C_online`, `common: iceflow_online`) instead
of the frozen off-line emulator. Here `data_assimilation` keeps **retraining the
network during the inversion** (`retrain_iceflow_model: true`), so the ice-flow
physics is continually re-fitted to the thickness as it evolves — more faithful
than Step 1's frozen emulator, at the cost of being slower.

```bash
igm_run +experiment=params_C_online hydra.run.dir=outputs/DA_step4
```

---

# Part D — Field inversion (`field_inversion` module)

A newer, lighter inversion route that is set to replace `data_assimilation`
(Part C) in the long term. Instead of running the glacier forward in time, the
`field_inversion` module solves a single **bounded optimisation** for the
spatially-varying ice thickness (`thk`) that best reproduces observed surface
velocities. The objective combines a **misfit** term (`velsurf`, Huber loss
between modelled `uvelsurf/vvelsurf` and observed `uvelsurfobs/vvelsurfobs`) and
a **regularization** term on `thk` (squared-Laplacian smoothness referenced to
the surface `usurf`), with the thickness bounded to `[0, 1000] m`. Forward
velocities come from the off-line trained iceflow emulator (the shipped pretrained
`dahunet_mini.keras` network); the inversion itself uses an L-BFGS optimiser
(Hager–Zhang line search).

### Step 1 — Real-world inversion (`params_D_real.yaml`)

Inverts directly the **real Aletsch observations** stored in `data/input.nc`.

Key points of using real data:
- There is **no ground-truth thickness**, so inspect the result through the
  iterative `optimize.nc` and the final `output.nc`.
- Observed surface velocities cover only part of the glacier (~73% of cells are
  NaN); the misfit term automatically restricts the cost to finite observations
  intersected with `icemask`, so gaps are handled cleanly (`mask: icemask`).
- The basal sliding parameter is genuinely unknown, so `tau_ref` is set to the
  in-distribution default value (0.213 MPa at u_ref=100).

```bash
igm_run +experiment=params_D_real
```

Writes the inverted fields to `output.nc` and the optimization history to
`optimize.nc`.

### Step 2 — Synthetic validation benchmark (`params_D_synthetic.yaml`)

The same machinery applied to a **synthetic** input (`data/inverse_250m.nc`),
generated by running IGM forward with a known ("true") thickness field and
recording the resulting surface velocities. The recorded velocities act as
synthetic observations, and the true thickness lets you **validate** the result:
the `inverse_accuracy` output compares inverted vs true thickness. Here `tau_ref`
is fixed to the value used when generating the input file (0.15 MPa at u_ref=100),
since we invert for thickness only.

```bash
igm_run +experiment=params_D_synthetic
```

Writes the inverted fields to `output.nc` alongside `inverse_accuracy`
diagnostics, plus two figures comparing thickness and surface-velocity misfit.

> **Note:** `data_assimilation` (Part C), `field_inversion` (Part D) and
> `time_relaxation` (Part E) are complementary assimilation routes. Parts C and D
> recover a single field (`thk`); Part E nudges several fields jointly inside a
> transient forward run.

---

# Part E — Data assimilation by time relaxation (`time_relaxation` module)

*(forward time-relaxation data assimilation as used by Frank and al., 2026 ; modified after an original implementation by T. Frank)*

Instead of a one-shot inversion, the **`time_relaxation`** module integrates the
forward ice-flow model for **500 years** while a few **control fields are nudged
a little at every time step**, so the modelled glacier slowly relaxes onto the
observations. By the end, geometry, mass balance and velocity are mutually
consistent and match the data.

**The method.** `time_relaxation` is fully generic — a run is a list of
independent `steps`, each an orthogonal triple `(residual, update law, control)`.
A control field `C` is nudged so a residual `r` between a modelled quantity `M`
and a target `T` is driven toward zero. The whole inner loop runs inside the
module, which *replaces* the usual `time` module. 

**What this example fits.** Three steps run together:

| Control nudged | Driven to match | Step |
|---|---|---|
| `thk` (ice thickness) | flux divergence `divflux` → apparent mass balance `amb = smb − dhdt_obs` | `amb_thk` |
| `usurf` (surface elevation) | same shared AMB residual `(amb − divflux)` | `amb_usurf` |
| `tau_ref` (basal friction) | observed surface speed `velsurf_magobs` | `friction` |

The first two are the **apparent-mass-balance bed inversion** of *Frank & van
Pelt (2025)*: `thk` and `usurf` are perturbed jointly until the modelled flux
divergence equals the apparent mass balance. The third is a classic friction
inversion nudging `tau_ref` until modelled `velsurf_mag` matches the observed
surface speed; it runs on a slow cadence (every 50 yr, `cadence: 50.0`) through
to `end_time: 500`.

Two ice-flow back-ends are provided, mirroring Part C:

```bash
igm_run +experiment=params_E_offline   # fast: frozen off-line trained emulator (dahunet_mini)
igm_run +experiment=params_E_online    # robust: on-line retrained solver (from-scratch dahunet)
```

Both use the `unified` stack. The off-line emulator is fast but valid only near its
training regime (so `tau_ref` is clamped to a physical `[0.05, 0.5]` MPa); the
on-line solver is retrained on the fly, so it stays physically consistent as the
geometry and `tau_ref` evolve — more robust and a tighter velocity fit, but slower.
Inputs come from `data/input.nc` (geometry + observations: `thk`,
`usurf`, `dhdt`, `uvelsurfobs/vvelsurfobs`, `thkobs`, `icemask`). The run writes:

- `output.nc` — 11 snapshots (`t = 0, 50, …, 500 yr`) of `thk, usurf, tau_ref, velsurf_mag, velsurf_magobs, divflux, amb, dhdt, dhdt_obs, …`;
- `output_ts.nc` — area/volume time series; `misfits.csv` — per-save residual norms;
- `fit_dashboard/fit_t<TIME>.png` + `fit_evolution.gif` — a **live 8-panel fit dashboard** rendered each save (surface speed obs/model/residual + `tau_ref` map; apparent mass balance target/modelled/residual + live RMSE-convergence curve). Headless-safe (Agg); set `assimilations.time_relaxation.viz.show: true` for an interactive window.

The custom `eval_objective` module prints three fit scores at the end (and
exposes them as `state.score`): `rmse_divflux_minus_amb` (AMB fit, m/yr),
`rmse_vel` (surface-speed fit, m/yr) and `rmse_thk` (vs radar `thkobs`, m).
 
---

# Data (`data/`)

| File | Description |
|------|-------------|
| `input.nc` | Geometry + observations: bedrock topography, `thk`, `usurf`, `icemask`, surface velocities (`uvelsurfobs/vvelsurfobs`), `thkobs` (GPR), `dhdt`. Used by Parts A, B, C, D (Step 1) and E. |
| `inverse_250m.nc` | Synthetic benchmark with a known true thickness/velocity field (Part D, Step 2). |
| `temp_prec.dat` | Daily temperature and precipitation (1880–2100). |
| `past_surf.nc` | Observed surface elevations at 7 dates. |
| `massbalance.nc` | Snow redistribution and direct-radiation maps. |
| `mbparameter.dat` | Mass-balance parameters. |
| `bassin.nc` | Basin masks. |
| `seeding.nc` | Particle seeding map (Part A, Step 4). |

Velocity data from Millan et al. (2019), thickness from Grab et al. (2021),
outlines from Linsbauer et al. (2021).

---

# References

- Millan, R. et al. (2019). Mapping surface flow velocity of glaciers at regional scale using a multiple sensors approach. *Remote Sensing*, 11(21), 2498.
- Grab, M. et al. (2021). Ice thickness distribution of all Swiss glaciers based on extended ground-penetrating radar data and glaciological modeling. *Journal of Glaciology*, 67(266), 1074–1092.
- Linsbauer, A. et al. (2021). The new Swiss Glacier Inventory SGI2016: From a topographical to a glaciological dataset. *Frontiers in Earth Science*, 774.
- Frank, Thomas, et al. "Global glacier-free topography reveals a large potential for future lakes in presently ice-covered terrain." Nature Communications 17.1 (2026): 3985.
