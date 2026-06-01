#!/usr/bin/bash
set -ex

# Redirect stdout and stderr to a log file.
exec > >(tee -i run_log.txt)
exec 2>&1

export NOSHOW=1

echo "RUNFAST DEBUG MODE FLAG: $RUNFAST"

lscpu
if [ -x "$(command -v nvidia-smi)" ]; then
	nvidia-smi
fi

python propagator_run.py

python logarithmic_transform.py

bash err_run.sh

bash ap_run.sh

bash speed_run.sh

bash complexity_run.sh

bash gpu_run.sh

# Gather up the results
mkdir results
cp -rp $HOME/install_log.txt results/
cp -rp ./*.png results/
cp -rp ./*.jpg results/
cp -rp ./*_log.txt results/
cp -rp ./*_data results/
cp -rp ./*.csv results/
cp -rp propagator_out results/
tar -czf results.tar.gz results
