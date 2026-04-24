set -ex

# Remove the freeware drivers
apt purge nvidia*
apt remove nvidia-*
rm /etc/apt/sources.list.d/cuda*
apt autoremove -y
apt autoclean -y
rm -rf /usr/local/cuda*

# Install the latest driver available
apt install nvidia-driver-590-server

reboot

# Install the CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install nvidia-cuda-toolkit

reboot
