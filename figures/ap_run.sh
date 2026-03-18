#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i ap_log.txt)
exec 2>&1

# rm -rf ap_traces

time_steps=(
    # .001
    # .010
    # .025
    # .050
    .100
    # .200
)

for dt in "${time_steps[@]}"; do
    # python menon2009.py matexp $dt
    # python menon2009.py sparse $dt
    python menon2009.py approx64 $dt
    # python menon2009.py approx32 $dt
done

max_errors=(
    .000001
    .0001
    .01
    1
    5
    10
    20
)

# for err in "${max_errors[@]}"; do
#     python menon2009.py approx .025 $err
# done

python ap_plot.py
