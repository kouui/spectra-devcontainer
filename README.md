# SPECTRA

A spectroscopy and radiative transfer computation library supporting atomic physics calculations, statistical equilibrium solvers, radiative transfer (Feautrier method), and slab/cloud models for H, He, Ca II, O V, C III, Si III and more.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (package manager)

## Installation

```bash
git clone https://github.com/kouui/spectra.git
cd spectra
uv sync
```

## Running Notebooks

Notebooks are located in `notebooks/basic/` and cover topics such as radiative transfer, statistical equilibrium, and absorption profiles.

### Option A: VS Code (recommended)

Install the following VS Code extensions (recommended automatically when opening this project):

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

Then open any `.ipynb` file in VS Code and select the `.venv` Python interpreter as the kernel.

### Option B: JupyterLab in browser

```bash
uv sync --extra notebook
uv run jupyter lab
```

Then open `http://localhost:8888` in your browser.

## Running Examples

```bash
uv run test/examples/example.He.py
uv run test/examples/example.SE.py
```

## Development

```bash
uv sync --extra dev
uv run ruff check src/
uv run pyright src/
uv run pytest test/
```

## TODO

- [ ] Update `data/atom` folder. Prof Ichimoto may have the latest data file.
- [ ] Implement different kinds of Doppler velocity: relative to solar spectra, relative to observer.
