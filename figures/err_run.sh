#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i err_log.txt)
exec 2>&1

rm -rf err_data

CELLS=10000

SEED=$RANDOM

# python -c "import numpy; print('\n'.join(str(x) for x in numpy.geomspace(.001, 1, 10)))"

for dt in "(
    0.0021544346900318843
    0.004641588833612777
    0.021544346900318832
    0.046415888336127774
    0.21544346900318823
    0.46415888336127775
    1.0
)"; do
    # Don't compute matexp @ dt=.001, instead use dt=1
    python benchmark.py "err_data/matexp_$dt" $SEED matexp $dt $CELLS all
done

for dt in "(
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
)"; do
    python benchmark.py "err_data/sparse_$dt" $SEED sparse $dt $CELLS all
    python benchmark.py "err_data/approx_$dt" $SEED approx64 $dt $CELLS all
done

python err_plot.py err_data
