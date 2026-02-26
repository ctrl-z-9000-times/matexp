#!/usr/bin/bash
set -ex

python propagator_matrix.py

python logarithmic_transform.py

bash err_run.sh

bash speed_run.sh

bash gpu_run.sh

bash hh_run.sh

python speed_vs_accuracy.py
