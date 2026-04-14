#!/usr/bin/bash
set -ex
cd $HOME

# Install prerequisite software
echo password | sudo -S apt-get update && apt-get install -y \
	python-is-python3 python-dev-is-python3 \
	git cmake bison flex python3-dev libreadline-dev libx11-dev libxcomposite-dev

# Download the github repo's
git clone https://github.com/ctrl-z-9000-times/nrn.git
git clone https://github.com/ctrl-z-9000-times/matexp.git

# Download the PyPI dependencies.
pip install --upgrade pip setuptools wheel
# pip install --user -r nrn/nrn_requirements.txt

# Build NEURON (with nrnbuild.py)
mkdir nrn/build
cd nrn
git checkout nrnbuild
cd build
cmake .. \
	-DPYTHON_EXECUTABLE=$(which python3) \
	-DCMAKE_INSTALL_PREFIX=$HOME/.local/ \
	-DCMAKE_BUILD_TYPE=Release \
	-DNRN_ENABLE_RX3D=OFF \
	-DNRN_ENABLE_MPI=OFF \
	-DNRN_ENABLE_CORENEURON=ON \
	-DCORENRN_ENABLE_NMODL=ON \
	-DNMODL_ENABLE_PYTHON_BINDINGS=ON \
	-DNRN_ENABLE_TESTS=OFF
cmake --build . --parallel --target install

# Build NEURON
git checkout matexp3
cmake .. \
	-DPYTHON_EXECUTABLE=$(which python3) \
	-DCMAKE_INSTALL_PREFIX=$HOME/.local/ \
	-DCMAKE_BUILD_TYPE=Release \
	-DNRN_ENABLE_RX3D=OFF \
	-DNRN_ENABLE_MPI=OFF \
	-DNRN_ENABLE_CORENEURON=ON \
	-DCORENRN_ENABLE_NMODL=ON \
	-DNMODL_ENABLE_PYTHON_BINDINGS=ON \
	-DNRN_ENABLE_TESTS=OFF
cmake --build . --parallel --target install

# Install the matexp program from source
cd $HOME/matexp
pip install --user --editable .
