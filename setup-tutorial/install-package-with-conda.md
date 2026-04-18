# Installing Spectra with Conda

> **Assumptions**
> - Your path to the `spectra` folder is `/path/to/spectra/`
> - Your path to Anaconda3 or Miniconda3 is `/path/to/anaconda3`

---

## Step 1: Create a New Conda Environment

**Linux / macOS:**

```bash
conda create -n spectra-py39 python=3.9
```

**Windows:**

```bash
conda create -n spectra-py39 -c conda-forge python=3.9
```

---

## Step 2: Make the Spectra Folder Visible to Python

Add the path to the `spectra` folder to the environment's `site-packages` by writing a `.pth` file.

**Linux / macOS:**

```bash
echo /path/to/spectra/ > /path/to/anaconda3/envs/spectra-py39/lib/python3.9/site-packages/module.pth
```

**Windows:**

```bash
echo /path/to/spectra/ > /path/to/anaconda3/envs/spectra-py39/Lib/site-packages/module.pth
```

---

## Step 3: Activate the Environment

```bash
conda activate spectra-py39
```

---

## Step 4: Install Required Packages

Install the minimum set of packages needed to run the source code:

```bash
pip install numba==0.53.1 numpy==1.20.3 scipy==1.6.3 debtcollector==2.2.0
```

---

## Daily Usage

After the initial setup, the only step required each time you want to run the code is **Step 3** — activating the environment:

```bash
conda activate spectra-py39
```

To return to the default (base) environment:

```bash
conda deactivate
```

---

## Examples

Example scripts are located at:

```
spectra/test/examples/example.*.py
```

---

## Optional / Additional Packages

### Plotting and Symbolic Math

```bash
pip install matplotlib sympy
```

### Jupyter Notebook Support

Reference: https://starpentagon.net/analytics/conda_env_jupyter_notebook/

```bash
conda install -c conda-forge notebook ipykernel
ipython kernel install --user --name spectra-py39
```
