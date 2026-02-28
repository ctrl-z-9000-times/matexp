#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i f32_run.log)
exec 2>&1

rm -rf f32_data

CELLS=100000

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
    python benchmark.py "f32_data/matexp_$dt" $SEED matexp $dt $CELLS
    python benchmark.py "f32_data/approx64A_$dt" $SEED approx $dt $CELLS --error 1e-3
    python benchmark.py "f32_data/approx64B_$dt" $SEED approx $dt $CELLS --error 1e-4
    python benchmark.py "f32_data/approx64C_$dt" $SEED approx $dt $CELLS --error 1e-5
    python benchmark.py "f32_data/approx32A_$dt" $SEED approx $dt $CELLS --f32 --error 1e-3
    python benchmark.py "f32_data/approx32B_$dt" $SEED approx $dt $CELLS --f32 --error 1e-4
    python benchmark.py "f32_data/approx32C_$dt" $SEED approx $dt $CELLS --f32 --error 1e-5
done

python err_plot.py f32_data
