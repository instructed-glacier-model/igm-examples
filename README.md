# Overview

This repository contains several simple IGM examples to help you become familiar with IGM.

Each folder is organized to separate data, parameters, and user functions:
```
├── data
│   ├── ....
├── experiment 
│   └── params.yaml
└── user
    ├── code
    │   └── processes
    │       └── usermodule
    └── conf
        └── processes
            └── usermodule.yaml 
```

Usage: After installing IGM, navigate to the example folder of interest, and run:
```bash
igm_run +experiment=params
```
This command will create a new directory named `outputs` to store the model results.

You may run the following examples (listed in order of complexity):

- `quick-demo` offers a setup to model any glacier given an RGI ID, using OGGM-based climate forcing and SMB. It also provides an example to include a custom user-defined SMB module/parameterization.

- `aletsch-basic` provides a straightforward setup for an advance-retreat simulation of the largest glacier in the European Alps -- the Aletsch Glacier in Switzerland -- using a simple mass balance parameterization based on time-varying Equilibrium Line Altitudes (ELA).

- `aletsch-1880-2100` provides the setup to reproduce simulations of the Great Aletsch Glacier (Switzerland) for past and future conditions based on the CH2018 climate scenarios and an accumulation/melt model.

- `aletsch-invert` provide examples of data assimilation with IGM (Note: inverse modeling requires tuning parameters for each glacier). 

- `paleo-alps`  consists of a basic setup to run a paleo glacier model for the European Alps during paleo times with different catchments (Lyon, Ticino, Rhine, Linth glaciers) using IGM around the last glacial maximum (LGM, approximately 24 BP in the Alps).

- `synthetic` allows you to conduct simple numerical experiments with basic synthetic bedrock topographies.


