# IGM Examples

This repository contains several IGM examples to help you become familiar with IGM.

Each folder is organized to separate data, parameters, and user functions:
```
example-name/
├── data/                # Input data (NetCDF files, climate data)
├── experiment/          # Hydra YAML config files (params.yaml, ...)
└── user/
    ├── code/
    │   ├── inputs/      # Custom input modules
    │   └── processes/   # Custom process modules
    └── conf/
        ├── inputs/      # YAML configs for custom input modules
        └── processes/   # YAML configs for custom process modules
```

## Usage

After installing IGM, navigate to the example folder of interest, and run:
```bash
igm_run +experiment=params
```
This command will create a new directory named `outputs` to store the model results.

**Tip:** Prefix any command with `TF_CPP_MIN_LOG_LEVEL=3` to suppress verbose TensorFlow messages:
```bash
TF_CPP_MIN_LOG_LEVEL=3 igm_run +experiment=params
```

## Examples 

- **`aletsch`** — Step-by-step tutorial for modeling the Great Aletsch Glacier, progressively introducing custom SMB modules, realistic climate forcing, particle tracking, and Optuna parameter optimization.

- **`quick-demo`** — Model any glacier given an RGI ID, using OGGM-based climate forcing and SMB.

- **`synthetic`** — Step-by-step tutorial with synthetic bedrock: basic simulation, particle tracking, and Optuna multi-objective optimization.

- **`paleo-alps`** — Paleo glacier modeling for the European Alps around the last glacial maximum (LGM, ~24 ka BP).

## Deprecated Examples

The following examples are kept for reference but are deprecated:

- **`Bueler2005C`** — Analytical benchmark (Bueler et al., 2005, test C).
- **`instructed_oggm`** — Experimental/legacy OGGM coupling. May not work with current versions.

## Inversion / Data Assimilation

The former examples **`aletsch-invert`** was removed (outdated), data assimilation is work in progress. As a temporary solution, you may look at the repo dedicated to data assimilation examples in https://github.com/instructed-glacier-model/igm-examples-invert
 
