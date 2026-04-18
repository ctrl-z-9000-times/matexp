#!/usr/bin/bash
set -ex
cd $HOME

# Redirect stdout and stderr to a log file.
exec > >(tee -i install_log.txt)
exec 2>&1

export PATH=$PATH:$HOME/.local/bin

# Install prerequisite software
echo password | sudo -S apt-get update
# echo password | sudo -S apt-get upgrade -y
echo password | sudo -S apt-get install -y \
	python-is-python3 python-dev-is-python3 \
	git cmake bison flex python3-dev  \
	libx11-dev libxcomposite-dev libncurses-dev libreadline-dev \
	libopenmpi-dev openmpi-bin \
	tmux

# Download the github repo's
git clone https://github.com/ctrl-z-9000-times/nrn.git
git clone https://github.com/ctrl-z-9000-times/matexp.git
git clone https://github.com/LLNL/Caliper.git

# Download the PyPI dependencies.
echo password | sudo -S pip install --upgrade pip setuptools wheel
pip install matplotlib
# pip install --user -r nrn/nrn_requirements.txt

# Build Caliper (the profiling library)
cd $HOME/Caliper
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local/
make && make install

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
	-DNMODL_ENABLE_PYTHON_BINDINGS=ON \
	-DNRN_ENABLE_TESTS=OFF \
	-DNRN_ENABLE_PROFILING=ON -DNRN_PROFILER=caliper
cmake --build . --parallel --target install

# Install the matexp program from source
cd $HOME/matexp
pip install --user --editable .
