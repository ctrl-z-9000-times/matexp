#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i run_log.txt)
exec 2>&1

export NOSHOW=1

echo "RUNFAST DEBUG MODE FLAG: $RUNFAST"

lscpu
lspci
if [ -x "$(command -v nvidia-smi)" ]; then
	nvidia-smi
fi

python propagator_run.py

python logarithmic_transform.py

bash err_run.sh

bash ap_run.sh

bash speed_run.sh

bash gpu_run.sh

bash complexity_run.sh

# Gather up the results
mkdir results
mv $HOME/install_log.txt results/
mv ./*.png results/
mv ./*_log.txt results/
mv ./*_data results/
mv propagator_out results/
tar -czf results.tar.gz results
