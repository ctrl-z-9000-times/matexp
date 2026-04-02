#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i ap_log.txt)
exec 2>&1

time_steps=(
    .001
    .0125
    .025
    .0375
    .050
)

for dt in "${time_steps[@]}"; do
    python ap_sim.py matexp $dt
    python ap_sim.py sparse $dt
    python ap_sim.py approx32 $dt
done

max_errors=(
    .0001
    .001
    .01
    .1
)

for err in "${max_errors[@]}"; do
    python ap_sim.py approx64 .025 $err
done

python ap_plot.py
