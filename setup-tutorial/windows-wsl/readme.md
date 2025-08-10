# setup SPECTRA environment on Windows WSL

## prerequisites


you should have

1. your WSL installed on your Windows machine.
    - reference : https://learn.microsoft.com/ja-jp/windows/wsl/install
2. vscode installed on your Windows machine.
3. vscode could be launched from WSL terminal.
    - reference : https://learn.microsoft.com/ja-jp/windows/wsl/tutorials/wsl-vscode


## installation

steps

1. install `starship` for better terminal experience
2. install `miniconda` for python environment management
3. git clone SPECTRA repository
    - configure git user name and email
4. setup conda environment for SPECTRA
5. install recommended vscode extensions



### install starship

starship is a cross-shell prompt that can be used in WSL to enhance your terminal experience (by displaying rich information in your terminal prompt).

- reference: https://github.com/starship/starship

open your WSL terminal, run the following command

```bash
curl -sS https://starship.rs/install.sh | sh
eval "$(starship init bash)"
```

configure a preset for starship

```bash
mkdir -p ~/.starship && touch ~/.starship/starship.toml
starship preset no-nerd-font -o ~/.starship/starship.toml
```

you may find more presets in https://starship.rs/presets/


### install miniconda

miniconda is a minimal installer for conda, which is a package manager for Python and other languages.

- reference: https://www.anaconda.com/docs/getting-started/miniconda/install

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

### clone SPECTRA repository

first we configure git user name and email (replace "Your Name" and "Your Email" with your actual name and email address)

```bash
git config --global user.name "Your Name"
git config --global user.email "Your Email"
```

then we clone the SPECTRA repository (replace `/path/to/your/workspace/as/you/like` with your actual workspace path)

```bash
cd /path/you/want/to/place/spectra
git clone https://github.com/kouui/spectra-devcontainer.git spectra
```

### setup conda environment for SPECTRA

setup conda environment for SPECTRA

```bash
cd /path/to/spectra
conda create -n spectra.py3.11 python=3.11
# add spectra source code to the python environment, make sure you are in the /path/to/spectra directory
pwd >> ~/miniconda3/envs/spectra.py3.11/lib/python3.11/site-packages/module.pth
# activate the conda environment
conda activate spectra.py3.11

# check whether you are using the correct python interpreter
which python
# should be something like ~/miniconda3/envs/spectra.py3.11/bin/python

# install required packages
pip install -r spectra/requirements.txt
# if you need to install additional packages, you should always activate the conda environment first, otherwise the packages will be installed in the unexpected location.

```

### install recommended vscode extensions

you can install the recommended vscode extensions by running the following command in your WSL terminal

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.vscode-jupyter-cell-tags
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.flake8
code --install-extension github.copilot
code --install-extension github.copilot-chat
```

to use github copilot, you need to sign in with your GitHub account.

- reference: https://zenn.dev/muit_techblog/articles/38e5780fb6011a

## you workflow

after everything is set up, you can start playing with SPECTRA by the following steps.

```bash
# move to the directory of SPECTRA
cd /path/to/spectra
# activate the conda environment
conda activate spectra.py3.11
# open vscode in the current directory
code .
```

