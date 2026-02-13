#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i err_run.log)
exec 2>&1

rm -rf err_data

CELLS=10000

SEED=$RANDOM

# python -c "import numpy; print('\n'.join(str(x) for x in numpy.geomspace(.001, 1, 10)))"
time_steps=(
    0.001
    0.0021544346900318843
    0.004641588833612777
    0.01
    0.021544346900318832
    0.046415888336127774
    0.1
    0.21544346900318823
    0.46415888336127775
    1.0
)

for dt in "${time_steps[@]}"; do
    python benchmark.py $SEED matexp $dt $CELLS
    python benchmark.py $SEED sparse $dt $CELLS
    python benchmark.py $SEED approx $dt $CELLS
done

python err_plot.py
