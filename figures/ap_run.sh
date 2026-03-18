#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i ap_log.txt)
exec 2>&1

time_steps=(
    .0025
    .008
    .025
    .080
    .250
)

for dt in "${time_steps[@]}"; do
    python menon2009.py matexp $dt
    python menon2009.py sparse $dt
    python menon2009.py approx64 $dt
done

max_errors=(
    .0005
    .005
    .05
    .5
)

for err in "${max_errors[@]}"; do
    python menon2009.py approx32 .025 $err
done

python ap_plot.py
