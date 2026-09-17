#!/bin/sh
# Remove everything the runs generate. The input data ships with the repository and is
# deliberately left alone.
rm -rf outputs multirun plots optuna_cts.db optimization_results.csv climate_ref.nc
