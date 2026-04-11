#!/usr/bin/env bash
# Remove all artifacts produced by IGM and the optimization sweeper.
set -e
rm -rf outputs multirun
rm -f  optimization_results.csv optuna.db optuna_2obj.db
rm -f  pareto_front.png misfit_maps.png
find . -name __pycache__ -prune -exec rm -rf {} +
