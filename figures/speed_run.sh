#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i speed_log.txt)
exec 2>&1

lscpu
if [ -x "$(command -v lspci)" ]; then
	lspci
fi
if [ -x "$(command -v nvidia-smi)" ]; then
	nvidia-smi -L
fi

rm -rf speed_data
mkdir -p speed_data

DT=.025
CELLS=1000
SEED=$RANDOM
CALI_CONFIG=runtime-report,calc.inclusive python benchmark.py /dev/null $SEED sparse $DT $CELLS dedup 2> speed_data/sparse
CALI_CONFIG=runtime-report,calc.inclusive python benchmark.py /dev/null $SEED matexp $DT $CELLS dedup 2> speed_data/matexp
CALI_CONFIG=runtime-report,calc.inclusive python benchmark.py /dev/null $SEED approx $DT $CELLS dedup 2> speed_data/approx
python speed_plot.py $DT $CELLS
