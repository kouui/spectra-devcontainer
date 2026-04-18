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

Notebooks are located in `notebooks/` and cover topics such as radiative transfer, statistical equilibrium, and absorption profiles.

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
uv run tests/examples/example.He.py
uv run tests/examples/example.SE.py
```

## Development

```bash
uv sync --extra dev
uv run ruff check src/
uv run pyright src/
uv run pytest tests/
```

## Conventions

### Custom modules

All new/experimental Python modules (`*.py`) should be placed in `src/spectra/Experimental/` first. Once the module is stable and reviewed, it can be promoted to the appropriate package (e.g., `Atomic/`, `Function/`).

### Local development

To avoid conflicts when syncing with the remote repository (`git pull`), use the following gitignored directories for local work:

| Purpose | Directory | Import example |
|---|---|---|
| Local notebooks | `notebooks_local/` | — |
| Local Python modules | `src/spectra/Experimental/local/` | `from spectra.Experimental.local import my_module` |

These folders are excluded from version control. When a local module is ready to be shared, move it from `local/` to `src/spectra/Experimental/`.

## TODO

- [ ] Update `data/atom` folder. Prof Ichimoto may have the latest data file.
- [ ] Implement different kinds of Doppler velocity: relative to solar spectra, relative to observer.
