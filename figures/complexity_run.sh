#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i complexity_log.txt)
exec 2>&1


rm -rf complexity_data
mkdir -p complexity_data

MOD_DIR=../mod
python complexity_sim.py $MOD_DIR/Nav11_6state.mod 1e-2 1e-3 1e-4

# python complexity_sim.py $MOD_DIR/AMPA_13state.mod 1e-2 1e-3 1e-4 1e-5
# python complexity_sim.py $MOD_DIR/NMDA_10state.mod 1e-2 1e-3 1e-4
# python complexity_sim.py $MOD_DIR/Nav11_6state.mod 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8
# python complexity_sim.py $MOD_DIR/Kv11_4state.mod 1e-2 1e-3 1e-4 1e-5
# python complexity_sim.py $MOD_DIR/Kv11_6state.mod 1e-2 1e-3 1e-4 1e-5
# python complexity_sim.py $MOD_DIR/Kv11_11state.mod 1e-2 1e-3 1e-4 1e-5
# python complexity_sim.py $MOD_DIR/Kv11_13state.mod 1e-2 1e-3 1e-4 1e-5

python complexity_plot.py

