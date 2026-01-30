#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i hh_run.log)
exec 2>&1

# rm -rf hh_traces

time_steps=(
    .001
    .010
    .025
    .050
    .100
    .200
)

max_errors=(
    .000001
    .0001
    .01
    1
    5
    10
    20
)

for dt in "${time_steps[@]}"; do
    python hh_sim.py approx $dt
    python hh_sim.py sparse $dt
    python hh_sim.py matexp $dt
done

for err in "${max_errors[@]}"; do
    python hh_sim.py approx .025 $err
done

python hh_plot.py
