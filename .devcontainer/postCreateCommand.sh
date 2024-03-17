#!/bin/bash -eux

# activate venv
source ~/venv/bin/activate
pip install -r spectra/requirements.txt
echo /home/dev/workspace/spectra/ > /home/dev/venv/lib/python3.11/site-packages/module.pth

# install python dependency 
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install -r minimum_env_latest/requirements.txt 

# Install starship prompt (optional)
curl -fsSL https://starship.rs/install.sh -o starship_install.sh
echo "docker" | sudo --stdin sh starship_install.sh -y
rm starship_install.sh
mkdir -p ~/.config/fish && echo "starship init fish | source" >> ~/.config/fish/config.fish
echo "source ~/venv/bin/activate.fish" >> ~/.config/fish/config.fish
fish -c "set -U fish_greeting"

# alias
echo 'alias start-jupyter="jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token=''"' >> ~/.config/fish/config.fish
