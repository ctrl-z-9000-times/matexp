#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i speed_run.log)
exec 2>&1

rm -rf speed_data
mkdir -p speed_data

DT=.025
CELLS=1000
SEED=$RANDOM
CALI_CONFIG=runtime-report,calc.inclusive python benchmark.py $SEED sparse $DT $CELLS 2> speed_data/sparse
CALI_CONFIG=runtime-report,calc.inclusive python benchmark.py $SEED matexp $DT $CELLS 2> speed_data/matexp
CALI_CONFIG=runtime-report,calc.inclusive python benchmark.py $SEED approx $DT $CELLS 2> speed_data/approx
python speed_plot.py $DT
