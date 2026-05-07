#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i gpu_log.txt)
exec 2>&1

lscpu
lspci
if [ -x "$(command -v nvidia-smi)" ]; then
    nvidia-smi
fi

echo password | sudo -S pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
pip install cupy-cuda12x

mkdir -p gpu_data

MOD_DIR=../mod
MOD_FILES=(
    "Nav11_6state.mod"
    "AMPA_13state.mod"
    "NMDA_10state.mod"
    "Kv11_4state.mod"
    "Kv11_6state.mod"
    "Kv11_11state.mod"
    "Kv11_13state.mod"
)

for file in "${MOD_FILES[@]}"; do
    python gpu_sim.py $MOD_DIR/$file
done
python gpu_plot.py
