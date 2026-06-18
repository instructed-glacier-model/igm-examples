# Bueler et al. (2005) test C — SIA dome benchmark

Analytical benchmark for an isothermal shallow-ice (SIA) ice sheet. A circular,
radially symmetric, **time-evolving** surface mass balance grows a circular ice
dome (Bueler test C, Table 2, p. 297). It is used to check whether IGM grows the
expected symmetric dome.

Test written by Alexander Jarosh based on Ed Bueler's work on SIA flow model
benchmarks from 2005.

**Tip:** Prefix any command with `TF_CPP_MIN_LOG_LEVEL=3` to suppress verbose
TensorFlow messages.

## Usage

```bash
conda activate igm19        # latest dev needs TF 2.19 / Keras 3
igm_run +experiment=params
```

Results are saved in a timestamped subfolder under `outputs/`. The main output
file is `output.nc`, which can be visualized with `ncview` or any NetCDF viewer.

A standalone comparison script against a mass-conserving SIA reference surface
lives in `compare_results/compare_results.py`
(reference data: `compare_results/SIA_data_bueler.nc`).

## Configuration (`params.yaml`)

- **Geometry**: 80×80 grid, 100 m resolution (`user/code/inputs/Bueler2005C.py`).
- **Forcing**: the analytical Bueler-C mass balance, recomputed every step in
  `user/code/processes/forcing_Bueler2005C.py` (sets `state.smb`).
- **Iceflow**: unified stack with the dahunet (SIA-informed) emulator, trained
  online from scratch. `A = 100 MPa⁻³ yr⁻¹` (`viscosity.arrhenius`), low
  sliding (`sliding.tau_ref = 0.01`) to approximate the no-slip EISMINT setup.
- **Time**: 0 → 1000 yr (the full benchmark runs ~15 000 yr; issues already
  appear in the first 1000 yr, so we stop there).

## Migration to the latest IGM `dev`

This example was updated to the current `dev` conventions (see
`igm-dev-to-main-upt.md`):

- run in the **`igm19`** conda env (TF 2.19 / Keras 3), not `igm` (TF 2.15);
- legacy `emulated` mode → **unified** iceflow with the **dahunet** emulator
  (NiceNet is deprecated);
- physics keys regrouped: `init_arrhenius` → `viscosity.arrhenius`,
  `init_slidingco` → `sliding.tau_ref`;
- `core.check_module_needs: false` — the forcing module sets `state.smb` only in
  `update()`, which the post-init module-needs checker would otherwise flag.

## References

- Bueler E, Lingle CS, Kallen-Brown JA, Covey DN, Bowman LN. Exact solutions and
  verification of numerical models for isothermal ice sheets. *Journal of
  Glaciology*. 2005;51(173):291-306. doi:10.3189/172756505781829449
- Jarosch, A. H., Schoof, C. G., and Anslow, F. S.: Restoring mass conservation
  to shallow ice flow models over complex terrain, *The Cryosphere*, 7, 229–240,
  doi:10.5194/tc-7-229-2013, 2013.

## Status / known issues (work in progress)

- First results showed a directional flow preference in IGM, demonstrated by this
  dome benchmark with time-evolving SMB (PR #22).
- Partially fixed by increasing the amount of training, but not in the strict
  no-slip condition — needs further investigation.
- Symmetry is perfectly preserved when using the solver.
- The SMB should be checked against the analytical solution and the SIA
  reference of Alexander Jarosh.
