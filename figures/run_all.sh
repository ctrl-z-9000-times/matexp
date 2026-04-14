#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i run_log.txt)
exec 2>&1

lscpu
lspci
nvidia-smi

python propagator_run.py

python logarithmic_transform.py

bash err_run.sh

bash ap_run.sh

bash speed_run.sh

bash gpu_run.sh

python speed_vs_accuracy.py

# Gather up the results
mkdir results
mv ./*.png results/
mv ./*_log.txt results/
mv -r ./*_data results/
mv -r propagator_out results/
tar -czf results.tar.gz results
