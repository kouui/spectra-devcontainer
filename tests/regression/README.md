# Regression Tests

Numerical regression tests that verify the ruff/pyright refactoring (commits `99a453d`–`3ebf2f5`) did not change any computation results.

## How reference values were generated

1. **Baseline commit**: `a84128b` — the last commit before any ruff formatting or lint fixes were applied.

2. **Worktree creation**:
   ```bash
   git worktree add /tmp/spectra-ref a84128b
   cd /tmp/spectra-ref
   uv sync
   ```

3. **Generation script**: A Python script (`gen_reference.py`, not committed) imports all key modules and calls every public function with representative inputs, collecting 178 scalar/array outputs into a dict.

4. **Output**: Results are serialized to `reference_values.json` via `json.dump`. Arrays are converted to nested lists with `ndarray.tolist()` (lossless for float64).

5. **Cleanup**:
   ```bash
   git worktree remove /tmp/spectra-ref --force
   ```

## reference_values.json key format

Keys follow the pattern `Module.function_description`:

| Pattern | Example |
|---|---|
| Unit test | `BasicP.doppler_width_freq`, `LTELib.boltzmann_scalar` |
| Parametrized | `Hydrogen.CE_ni1_nj2_Te7000`, `Special.E1_x0.5` |
| E2E pipeline | `E2E.H_SE_Nh_Te.n_SE`, `E2E.H_CloudModel.tau_max` |

## Covered modules

| Module | Test file | Functions tested |
|---|---|---|
| `Atomic.BasicP` | `test_reg_BasicP.py` | `wave_to_freq_`, `freq_to_wave_`, `dop_vel_to_shift_`, `doppler_width_`, `damping_const_a_`, `refractive_index_in_air_`, `air_to_vacuum_`, `vacuum_to_air_` |
| `Atomic.LTELib` | `test_reg_LTELib.py` | `boltzmann_distribution_`, `saha_distribution_`, `planck_cm_`, `planck_hz_`, `einsteinA_to_einsteinBs_cm_`, `LTE_ratio_` |
| `Atomic.Collision` | `test_reg_Collision.py` | `Cij_to_Cji_` |
| `Atomic.Hydrogen` | `test_reg_Hydrogen.py` | `gaunt_factor_`, `absorption_oscillator_strength_`, `einstein_A_coefficient_`, `CE_rate_coe_`, `CI_rate_coe_`, `PI_cross_section_cm_`, `ratio_Etran_to_Eionize_`, `Rki_spon_rate_coe_`, `collisional_broadening_LinearStark_` |
| `Atomic.ContinuumOpacity` | `test_reg_ContinuumOpacity.py` | `thomson_scattering_`, `hydrogenic_bf_cross_sec_n_`, `HI_bf_LTE_cross_sec_`, `HI_rayleigh_cross_sec_`, `Hminus_cross_sec_`, `HI_ff_cross_sec_`, `gaunt_factor_ff_` |
| `Atomic.SEsolver` | `test_reg_SEsolver.py` | `solve_SE_`, `set_matrixC_`, `set_matrixR_` |
| `RadiativeTransfer.Profile` | `test_reg_Profile.py` | `voigt_`, `gaussian_`, `hf_` |
| `RadiativeTransfer.Feautrier` | `test_reg_Feautrier.py` | `formal_improved_RH_` |
| `RadiativeTransfer.Thin` | `test_reg_Thin.py` | `relative_flux_` |
| `Math.Special` | `test_reg_Math.py` | `E0_`, `E1_`, `E2_`, `E3_` |
| `Math.Integrate` | `test_reg_Math.py` | `trapze_`, `simpson_` |
| `Math.GaussLeg` | `test_reg_Math.py` | `gauss_quad_coe_` |
| `Util.MeshUtil` | `test_reg_MeshUtil.py` | `make_full_line_mesh_` |
| SE pipeline (H, He) | `test_reg_e2e_SE.py` | `cal_SE_with_Nh_Te_`, `cal_SE_with_Ne_Te_` |
| Cloud model (H) | `test_reg_e2e_CloudModel.py` | `SE_to_slab_0D_` |

## How to regenerate reference values

After an intentional algorithm/constant change, regenerate both snapshots from
current code:

```bash
# reference_values.json  (unit + e2e SE + cloud model keys)
uv run python scripts/gen_reference.py
# atom_reference_values.json  (8 canonical atom loads)
uv run python scripts/gen_atom_reference.py
```

`gen_reference.py` runs the regression suite in record mode (`REGEN_REFS=1`):
each `assert_close(actual, ref[key])` writes `actual` back under `key` (see
`conftest.py`), so the golden file is rebuilt from the exact inputs the tests
already define -- there is no separate input table to keep in sync. A second
normal pass then verifies the regenerated file. Review the resulting diff before
committing.

## Running

```bash
uv run --extra dev python -m pytest tests/regression/ -v
```
