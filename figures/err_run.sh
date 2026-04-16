#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i err_log.txt)
exec 2>&1

CELLS=10000

SEED=$RANDOM

# python -c "import numpy; print('\n'.join(str(x) for x in numpy.geomspace(.001, 1, 10)))"

MECH=all

if [ -n "${RUNFAST:-}" ]; then
    MECH=dedup
fi

python benchmark.py "err_data/matexp_1.0" $SEED matexp 1.0 $CELLS $MECH &

for dt in 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
    python benchmark.py "err_data/sparse_$dt" $SEED sparse $dt $CELLS $MECH &
done

wait -f $(jobs -p)

for dt in 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0
do
    python benchmark.py "err_data/approx32a_$dt" $SEED approx32 $dt $CELLS $MECH
    python benchmark.py "err_data/approx32b_$dt" $SEED approx32 $dt $CELLS $MECH --error 1e-8
    python benchmark.py "err_data/approx64_$dt" $SEED approx64 $dt $CELLS $MECH --error 1e-8
done

python err_plot.py err_data
