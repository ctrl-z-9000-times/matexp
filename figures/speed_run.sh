#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i speed_log.txt)
exec 2>&1

rm -rf speed_data
mkdir -p speed_data

DT=.025
CELLS=1000
SEED=$RANDOM
CALI_CONFIG=runtime-report,calc.inclusive python benchmark.py /dev/null $SEED sparse $DT $CELLS dedup 2> speed_data/sparse
CALI_CONFIG=runtime-report,calc.inclusive python benchmark.py /dev/null $SEED matexp $DT $CELLS dedup 2> speed_data/matexp
CALI_CONFIG=runtime-report,calc.inclusive python benchmark.py /dev/null $SEED approx32 $DT $CELLS dedup --error 1e-3 2> speed_data/approx32
CALI_CONFIG=runtime-report,calc.inclusive python benchmark.py /dev/null $SEED approx64 $DT $CELLS dedup --error 1e-8 2> speed_data/approx64
python speed_plot.py $DT $CELLS
