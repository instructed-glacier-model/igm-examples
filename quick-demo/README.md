# Quick Demo — Model Any Glacier from its RGI ID

The simplest way to get started with IGM: model any glacier worldwide given its [Randolph Glacier Inventory](https://www.glims.org/RGI/) (RGI) ID. This example models the Great Aletsch Glacier (Switzerland) from 1800 to 2100 with a +4°C warming scenario.

**Tip:** Prefix any command with `TF_CPP_MIN_LOG_LEVEL=3` to suppress verbose TensorFlow messages.

## Usage

```bash
igm_run +experiment=params
```

To model a different glacier, override the RGI ID from the command line:
```bash
igm_run +experiment=params inputs.oggm_shop.RGI_ID="RGI2000-v7.0-G-11-01450"
```

Results are saved in a timestamped subfolder under `outputs/`. The main output file is `output.nc`, which can be visualized with `ncview`, the NetCDF viewer extension in VS Code, or any NetCDF-compatible tool.

The simulation pipeline is defined in `experiment/params.yaml` and proceeds as follows:

### Data download (`oggm_shop` input)

On first run, the `oggm_shop` module automatically downloads and caches data from [OGGM](https://oggm.org/) for the specified glacier:
- Surface DEM (Copernicus GLO-90)
- Ice thickness (Millan et al., 2022)
- Surface velocities (Millan et al., 2022)
- Glacier outline and mask (RGI v7)
- Monthly climate data (GSWP3_W5E5)
- Calibrated SMB parameters

All data is saved to `data/input.nc` at 100 m resolution. Subsequent runs reuse the cached data.

### Climate forcing (`clim_oggm` process)

Processes monthly temperature and precipitation from the OGGM climate dataset. A climate trend is applied on top of the historical record: **1900-2020**: no change (baseline) and **2020-2100**: linear warming up to +4°C, constant precipitation

### Surface mass balance (`smb_oggm` process)

Computes yearly SMB using a monthly temperature-index model calibrated by OGGM (Hugonnet et al., 2021). Accumulation occurs when temperature is below a threshold; melt is proportional to positive degree days.

### Outputs (`local` + `plot2d`)

`output.nc` contains all saved 2D fields (ice thickness, surface elevation, velocities, SMB, ...) at 10-year intervals. `plot2d` generates live 2D plots of velocity magnitude during the simulation

## Configuration (`params.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RGI_ID` | `RGI2000-v7.0-G-11-02596` | Aletsch Glacier |
| `time.start` / `end` | 1800 / 2100 | 300-year simulation |
| `time.save` | 10.0 | Save every 10 years |
| `iceflow.physics.init_slidingco` | 0.25 | Basal sliding coefficient |
| `clim_oggm.clim_trend_array` | +4°C by 2100 | Warming scenario |
| `plot2d.live` | true | Real-time 2D visualization |
