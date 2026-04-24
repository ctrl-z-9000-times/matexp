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
