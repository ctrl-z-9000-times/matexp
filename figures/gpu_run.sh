#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i gpu_log.txt)
exec 2>&1

mkdir -p gpu_data

MOD_DIR=../mod
MOD_FILES=(
    "AMPA_13state.mod"
    "NMDA_10state.mod"
    "Nav11_6state.mod"
    "Kv11_4state.mod"
    "Kv11_6state.mod"
    "Kv11_11state.mod"
    "Kv11_13state.mod"
)

for file in "${MOD_FILES[@]}"; do
    python gpu_sim.py $MOD_DIR/$file
done
# python gpu_plot.py
