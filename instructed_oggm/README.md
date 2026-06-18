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

## Status with the latest IGM `dev` (verified 2026-06-18, igm19 env)

✅ **Runs end-to-end** (100-yr Aletsch run, writing `snapshot*.png`). Two fixes
were needed:

1. **OGGM data path (this script).** The old bespoke prepro URL
   (`.../oggm_v1.6/exps/igm_v1/`, RGI6, border 30) is no longer hosted. It now
   uses the same maintained prepro as IGM's `oggm_shop` input
   (`igm/inputs/oggm_shop/oggm_util.py`): RGI **v7** Aletsch
   (`RGI2000-v7.0-G-11-02596`) from
   `.../oggm_v1.6/exps/igm_v4`, `from_prepro_level=3`, `prepro_border=40`,
   `prepro_rgi_version="70G"`. That prepro ships **`millan_ice_thickness`**
   (not the Farinotti `consensus_ice_thickness`), so the script reads that.

2. **IGM core (`igm/instructed_oggm.py`).** `IGM_Model2D` builds its config with
   `load_yaml_recursive(igm/conf)`, which pulls in the whole conf tree including
   `assimilations/pretraining`. `iceflow.initialize/update` then saw
   `"pretraining" in cfg.assimilations` and **skipped iceflow entirely**
   (`State has no attribute 'ubar'`). The constructor now drops that
   `assimilations` branch so the forward iceflow actually runs. This edit lives
   in the IGM source tree, not in this example.

It uses the **legacy** iceflow stack (`method: emulated`, `state.slidingco`),
which is unchanged on `dev`. OGGM 1.6.2 is installed in the `igm19` env.

### Switching glacier

Edit `rgi_ids` in `run_instructed_oggm.py` to any RGI v7 glacier ID available in
the `igm_v4` prepro (keep `prepro_rgi_version="70G"` for individual glaciers).
