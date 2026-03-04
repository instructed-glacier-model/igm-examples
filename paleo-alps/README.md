# Last Glacial Maximum Glacier Modeling in the European Alps

Simulates paleoglacier dynamics in the European Alps around the Last Glacial Maximum (LGM, ~24 ka BP) using IGM. The glacier mass balance is driven by the EPICA Dome C ice core temperature reconstruction.

**Tip:** Prefix any command with `TF_CPP_MIN_LOG_LEVEL=3` to suppress verbose TensorFlow messages.

## Usage

```bash
igm_run +experiment=params
```

Results are saved in a timestamped subfolder under `outputs/`. The main output file is `output.nc`, which can be visualized with `ncview`, the NetCDF viewer extension in VS Code, or any NetCDF-compatible tool.

## Switching catchments

Five Alpine glacier catchments are available as GeoTIFF topography files in `data/`:

| Catchment | File |
|-----------|------|
| Linth (default) | `data/linth/topg-linth.tif` |
| Rhine | `data/rhine/topg-rhine.tif` |
| Lyon | `data/lyon/topg-lyon.tif` |
| Ticino | `data/ticino/topg-ticino.tif` |
| Jura | `data/jura/topg-jura.tif` |

To run a different catchment, copy the desired topography file to `data/topg.tif`:
```bash
cp data/rhine/topg-rhine.tif data/topg.tif
igm_run +experiment=params
```

**Note:** This example currently uses the `load_tif` input module, which will be deprecated in a future version of IGM. The recommended input module is `local`.

## Configuration (`params.yaml`)

- **Time period**: -30,000 to -25,000 years BP (5,000 years around the LGM)
- **Save interval**: every 100 years
- **Spatial coarsening**: factor 2 (to speed up computation)
- **Live 2D plotting**: enabled

Key SMB parameters (in `smb_signal`):
- `pdela`: present-day ELA (3000 m)
- `deladt`: ELA sensitivity to temperature (200 m/°C)
- `gradabl` / `gradacc`: ablation / accumulation gradients
- `maxacc`: maximum accumulation (1.0 m/y)

## Custom Modules

| Module | Description |
|--------|-------------|
| `smb_signal` | ELA-based mass balance driven by the EPICA temperature signal |

## Companion Scripts

Two standalone plotting scripts are provided in `user/code/processes/`:
- `plot-climate-forcing.py` — plots the EPICA temperature signal and corresponding ELA over time
- `plot-result.py` — plots maximum ice thickness map and glacier extent along a flowline

Run them after a simulation:
```bash
python user/code/processes/plot-result.py
python user/code/processes/plot-climate-forcing.py
```

## Data

- `topg.tif` — present-day bedrock topography with glaciers and lakes removed (Millan et al., 2022)
- `EDC_dD_temp_estim.tab` — EPICA Dome C temperature reconstruction (Jouzel & Masson-Delmotte, 2007)
- `*-flowline.dat` — flowline coordinates for each catchment (used by `plot-result.py`)

## References

- Millan, R., Mouginot, J., Rabatel, A., & Morlighem, M. (2022). Ice velocity and thickness of the world's glaciers. *Nature Geoscience*, 15(2), 124-129.
- Jouzel, J. & Masson-Delmotte, V. (2007). EPICA Dome C Ice Core 800KYr deuterium data and temperature estimates. PANGAEA. doi:10.1594/PANGAEA.683655
