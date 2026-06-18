# instructed_oggm (Experimental / Legacy)

Couples IGM with [OGGM](https://oggm.org/) (Open Global Glacier Model): OGGM
provides the glacier directory (bedrock, ice thickness, mask, grid) and a mass
balance model, and IGM's ice-flow model is driven step-by-step through the
`IGM_Model2D` wrapper (`igm.instructed_oggm.IGM_Model2D`, an OGGM `Model2D`
subclass). This is a **standalone Python script**, not a Hydra `igm_run`
experiment — so the param-file migration guide (`igm-dev-to-main-upt.md`) does
**not** apply here; the coupling is driven entirely from `run_instructed_oggm.py`.

Code written by Julien Jehl, Fabien Maussion, and Guillaume Jouvet.

## Usage

```bash
conda activate igm19        # latest dev needs TF 2.19 / Keras 3; also needs `oggm` installed
python run_instructed_oggm.py
```

## Status with the latest IGM `dev` (checked 2026-06-18, igm19 env)

- ✅ **IGM side is compatible.** `from igm.instructed_oggm import IGM_Model2D`
  imports cleanly, the `IGM_Model2D(...)` constructor signature still matches the
  call in `run_instructed_oggm.py`, and it uses the **legacy** iceflow stack
  (`method: emulated`, `state.slidingco`) which is unchanged on `dev`. OGGM 1.6.2
  is installed in the `igm19` env.
- ❌ **Blocked by OGGM data hosting, not by IGM.** The script fetches a bespoke
  prepro directory:
  ```
  https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/exps/igm_v1/
  ```
  which is **no longer reachable** (`init_glacier_directories` raises
  `InvalidParamsError: base url seems unreachable`). The standard OGGM 1.6 prepro
  is not a drop-in replacement: it is not published for `prepro_border=30`, and
  the elev-bands prepro does not ship the 2D `consensus_ice_thickness` /
  `glacier_mask` gridded fields this script reads.

### Reviving the example

Repoint the OGGM data source (`base_url` / `from_prepro_level` /
`prepro_border` in `run_instructed_oggm.py`) at a currently-hosted OGGM prepro
that provides, in `gridded_data`, the 2D fields the script consumes
(`consensus_ice_thickness`, `topo`, `glacier_mask`) — or build that gridded data
locally with the OGGM gridded-attributes + Farinotti-consensus thickness tasks.
Once the directory loads, the IGM coupling above is expected to run as-is.
