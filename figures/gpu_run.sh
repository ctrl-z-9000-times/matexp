#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i gpu_run.log)
exec 2>&1

rm -rf gpu_data
mkdir -p gpu_data

# MOD_FILES=(../mod/*.mod)
MOD_FILES=(../mod/Nav11.mod)
MOD_FILES+=(hh_markov.mod)

for file in $MOD_FILES; do
    python gpu_sim.py $file
done
python gpu_plot.py
