#!/usr/bin/bash
set -ex
cd $HOME

# Redirect stdout and stderr to a log file.
exec > >(tee -i install_log.txt)
exec 2>&1

export PATH=$PATH:$HOME/.local/bin
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc

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

# Update PIP? If pip was installed by the OS then this will fail
# echo password | sudo -S pip install --upgrade pip setuptools wheel

# Download the PyPI dependencies.
pip install matplotlib pytest find-libpython jinja2 cmcrameri
# pip install --user -r nrn/nrn_requirements.txt

# Install the matexp program from source
cd $HOME/matexp
pip install --user --editable .

# Build Caliper (the profiling library)
cd $HOME/Caliper
mkdir install
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/Caliper/install/
make && make install

# Build NEURON (with nrnbuild.py)
cd $HOME/nrn
mkdir build && cd build
git checkout nrnbuild
cmake .. \
	-DPYTHON_EXECUTABLE=$(which python3) \
	-DCMAKE_INSTALL_PREFIX=$HOME/.local/ \
	-DNRN_INSTALL_PYTHON_PREFIX=lib/python3.10/site-packages/neuron/ \
	-DNRN_ENABLE_RX3D=OFF \
	-DNRN_ENABLE_MPI=OFF \
	-DNRN_ENABLE_CORENEURON=ON \
	-DNMODL_ENABLE_PYTHON_BINDINGS=ON \
	-DNRN_ENABLE_TESTS=ON  -DCORENRN_ENABLE_UNIT_TESTS=ON \
	-DNRN_ENABLE_PROFILING=ON -DCORENRN_ENABLE_CALIPER_PROFILING=ON -DNRN_PROFILER=caliper \
	-DCMAKE_PREFIX_PATH=$HOME/Caliper/install/share/cmake/caliper/
cmake --build . --parallel --target install
nrnbuild.py --help

echo "INSTALL COMPLETE"
