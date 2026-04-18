# Refactor Plan: Separating `Vd_sun` and `Vd_obs`

## Problem statement

The codebase currently has a single field `Atmosphere.Vd` that is used
with two physically different meanings:

1. **The atom's velocity in the sun's rest frame** (loaded from FAL as
   a vertical flow velocity).
2. **The atom's line-of-sight velocity relative to the observer** (used
   in `SELib` / `ExSpectrum` / `ExLTELib` when applying
   `Δλ = λ₀ · Vd / c`).

These are **not the same quantity**. Mixing them is only correct in
the degenerate case of disk-center observation with a stationary
observer, and even then only up to a sign convention.

## Physical argument for the separation

Statistical equilibrium (SE) describes level populations in the
**atom's rest frame**, driven by the radiation field the atom sees
locally. That radiation field (`backRad`, `PI_intensity`, …) is
specified in the **sun's rest frame** (photosphere-frame continuum,
neighbouring plasma radiation, etc.).

To compute `Jbar = ∫ I(λ) · φ(λ - λ₀) dλ` correctly, the absorption
profile `φ` (naturally in atom-rest-frame) must be expressed on the
same wavelength axis as `backRad`. The shift needed is exactly the
atom's velocity **relative to the sun**, projected onto the line of
sight — call this `Vd_sun`.

The observer's velocity relative to the sun (`Vd_obs`) contributes a
**constant** wavelength shift to every output photon. It does not
enter `Jbar`, it does not affect level populations, and it does not
couple to the radiative transfer solve. It is purely a post-processing
shift on the final spectrum's wavelength axis.

Hence the clean split:

```
  +-----------------+      +------------------+      +-------------+
  | Atmosphere      |      | Observation /    |      | Spectrum    |
  |   Vd_sun[z]     |      |   Spectrum meta  |      |  output     |
  |  (LOS proj of   |      |   Vd_obs         |      |  λ grid     |
  |  local flow in  |      |  (single scalar) |      |             |
  |  sun frame)     |      |                  |      |             |
  +--------+--------+      +---------+--------+      +------+------+
           |                         |                      ^
           v                         |                      |
  +-----------------+                |                      |
  | SELib           |                |                      |
  |  uses Vd_sun    |                |                      |
  |  for wm shift   |                |                      |
  |  + Jbar integ   |                |                      |
  +--------+--------+                |                      |
           |                         |                      |
           v                         |                      |
  +-----------------+                |                      |
  | CloudModel /    |                |                      |
  | RT solver       |                |                      |
  | (still in sun-  |                |                      |
  |  frame λ)       |                |                      |
  +--------+--------+                |                      |
           |                         |                      |
           v                         v                      |
  +----------------------------------------------------+    |
  |  apply_observer_shift_(spec_sun, Vd_obs)           |----+
  |    λ_obs[k] = λ_sun[k] * (1 + Vd_obs / c)          |
  +----------------------------------------------------+
```

## Current usages to change

The following table lists every place where the current `Vd` field is
touched, and the target state after the refactor.

```
  +------------------------------------------+-------------------------+
  | File:line                                | After refactor          |
  +------------------------------------------+-------------------------+
  | Struct/Atmosphere.py:31 (Atmosphere0D)   | rename Vd -> Vd_sun;    |
  | Struct/Atmosphere.py:57 (AtmosphereC1D)  | add docstring: LOS      |
  |                                          | projection, sun-frame,  |
  |                                          | sign convention         |
  +------------------------------------------+-------------------------+
  | Struct/Atmosphere.py:232,243 (init_VAL_) | Vd_sun = zeros_like(Vt) |
  |                                          | (unchanged semantics)   |
  +------------------------------------------+-------------------------+
  | Experimental/ExFAL.py:78,94,145          | store raw FAL vel into  |
  |                                          | V_vertical; project to  |
  |                                          | Vd_sun once μ (viewing  |
  |                                          | geometry) is known      |
  +------------------------------------------+-------------------------+
  | Function/SEquil/SELib.py:323              | Vd_sun = atmos.Vd_sun   |
  | Function/SEquil/SELib.py:362,531,604      | keep formula            |
  |                                          |   wm_cm + w0*Vd_sun/c   |
  |                                          | (it is the right thing  |
  |                                          | for Jbar integration    |
  |                                          | against sun-frame       |
  |                                          | backRad)                |
  +------------------------------------------+-------------------------+
  | Function/Icp/SELib.py:368,410,576,639     | same as SEquil/SELib    |
  |                                          | (these two files are    |
  |                                          | duplicates; consider    |
  |                                          | deduping)               |
  +------------------------------------------+-------------------------+
  | Experimental/ExSpectrum.py:44,71          | remove the in-loop      |
  |                                          | shift spec += w0*Vd/c;  |
  |                                          | apply Vd_obs once, at   |
  |                                          | the end, via            |
  |                                          | apply_observer_shift_   |
  +------------------------------------------+-------------------------+
  | Experimental/ExLTELib.py:252,276          | use Vd_sun for the      |
  |                                          | profile-shift inside    |
  |                                          | the RT integrand; apply |
  |                                          | Vd_obs at output only   |
  +------------------------------------------+-------------------------+
```

## Proposed design

### 1. `Atmosphere` struct

Rename and document the field. Optionally keep the raw vertical
velocity as a separate field so projection can be deferred:

```python
@_dataclass(**STRUCT_KWGS_UNFROZEN)
class AtmosphereC1D:
    ...
    # LOS-projected velocity of the local plasma in the sun's rest
    # frame, [cm/s]. Sign convention: positive = receding from the
    # observer (redshift). Used by SELib when interpolating backRad
    # (which is also sun-frame) at the shifted wavelength mesh.
    Vd_sun : T_ARRAY

    # (optional) raw vertical flow velocity from the atmosphere model
    # file, [cm/s], positive upwards. Kept around so that Vd_sun can
    # be recomputed if viewing geometry changes.
    V_vertical : T_ARRAY
    ...
```

### 2. `SELib` (both `SEquil/SELib.py` and `Icp/SELib.py`)

Read `Vd_sun` instead of `Vd`. The wavelength shift stays the same:

```python
Vd_sun = atmos.Vd_sun
...
wm_cm_shifted = wm_cm[:] + (w0 * Vd_sun / CST.c_)
I_cm_interp   = _numpy.interp(wm_cm_shifted, backRad[0,:], backRad[1,:])
```

The shifted mesh `wave_mesh_cm_shifted_all` returned from SELib is
still expressed in **sun-frame wavelengths** — this is the output
that downstream RT solvers should consume.

### 3. New high-level helper

Add a single function in `Experimental/ExSpectrum.py` (or a new module
`Function/Observation.py`) that applies the observer shift as a pure
post-processing step:

```python
def apply_observer_shift_(
    wl_sun : T_ARRAY,    # wavelength grid in sun-frame, [cm]
    Vd_obs : T_FLOAT,    # observer LOS velocity w.r.t. sun, [cm/s]
) -> T_ARRAY:
    """Return the same grid expressed in observer-frame wavelengths.

    `Vd_obs` sign convention: positive = observer receding from sun
    (so observed wavelengths are longer than sun-frame wavelengths).
    """
    return wl_sun * (1.0 + Vd_obs / CST.c_)
```

`Vd_obs` is a **single scalar** for a given observation, not a
per-depth-point array.

### 4. `ExSpectrum.init_spectrum_`

Remove the in-loop shift:

```python
# REMOVE:
# spectrum[bias:bias+nwave] += ( w0 * Vd / CST.c_ )
```

The returned `Spectrum.spectrum` is then a sun-frame grid. Callers
that want the observer-frame grid call `apply_observer_shift_` once
on the final output.

### 5. `ExFAL`

Stop silently identifying FAL `vel` with `Vd`. FAL's `vel` column is
a **vertical** flow velocity, not a LOS projection. Options:

(a) **Minimum change** — keep the current behaviour but require the
    caller to explicitly opt in, and rename in place:

    ```python
    V_vertical = vel   # [cm/s], positive upwards
    # At disk center (mu=1), Vd_sun = -V_vertical up to sign of the
    # observer's "positive-away" convention. Do NOT set Vd_sun here
    # without knowing mu. Leave it for the caller.
    atmos = AtmosphereC1D(..., V_vertical=V_vertical,
                              Vd_sun=_numpy.zeros_like(V_vertical))
    ```

(b) **Structural change** — add a `project_to_los_` helper:

    ```python
    def project_to_los_(atmos: AtmosphereC1D, mu: T_FLOAT) -> None:
        # In-place: atmos.Vd_sun := -mu * atmos.V_vertical
        # (sign: upflow at disk center => blueshift => Vd_sun < 0
        #  under "positive = receding" convention)
        atmos.Vd_sun[:] = -mu * atmos.V_vertical[:]
    ```

Option (b) is the cleaner long-term answer.

## Sign conventions (to pin down before coding)

Before touching any source file, write down and commit to:

1. **`Vd_sun` sign** — positive means the local plasma parcel is
   receding from the observer (LOS component, in the sun's rest
   frame). Equivalently, observed wavelength > sun-frame wavelength.

2. **`Vd_obs` sign** — positive means the observer is receding from
   the sun. Observed wavelength > sun-frame wavelength.

3. **FAL `vel` column** — verify with the FAL source which direction
   is positive (upflow vs downflow). The FAL atmosphere files
   conventionally list positive as upward, but confirm before coding.

4. **`dop_vel_to_shift_` in `Atomic/BasicP.py:46`** — currently uses
   `p0 * v / c`. Its docstring says `v` is "line of sight velocity"
   but does not specify the frame or sign. Update the docstring once
   the above are fixed.

The `w0 * Vd / c` formula in SELib is consistent with **case 1**
above (positive `Vd_sun` → redshifted wavelength mesh → samples
`backRad` at longer sun-frame wavelengths). Do not change the
arithmetic; just rename and document.

## Step-by-step execution order

Each step should compile and keep the test cases green before moving
to the next.

```
  Step 1:  Document sign conventions in refactor_doppler_velocity.md
           (this file) and commit.

  Step 2:  Rename `Atmosphere.Vd` -> `Atmosphere.Vd_sun` across the
           codebase (mechanical rename, no logic change).
           Touched files:
             - Struct/Atmosphere.py
             - Function/SEquil/SELib.py
             - Function/Icp/SELib.py
             - Experimental/ExFAL.py
             - Experimental/ExSpectrum.py
             - Experimental/ExLTELib.py
             - notebooks that reference atmos.Vd
           Semantics unchanged; SE still uses Vd_sun for wm shift.

  Step 3:  Add `apply_observer_shift_` helper. Do NOT yet call it.

  Step 4:  Remove the in-loop observer shift in ExSpectrum.
           Add `Vd_obs` as a parameter to a new wrapper
           `init_spectrum_observed_` that calls `init_spectrum_`
           and then `apply_observer_shift_` on the returned grid.
           Keep `init_spectrum_` sun-frame.

  Step 5:  (Optional) Split FAL loading:
             - add V_vertical to AtmosphereC1D
             - add project_to_los_(atmos, mu)
             - update ExFAL to store V_vertical instead of silently
               aliasing vel to Vd_sun

  Step 6:  (Optional) Deduplicate SEquil/SELib.py and Icp/SELib.py
           by extracting the shared body into a helper.
```

Steps 1-4 form a minimum viable refactor that fixes the semantic
confusion. Steps 5-6 are correctness and cleanliness improvements.

## Out of scope

- Continuum Doppler shift: `doppler_shift_continuum` in
  `Atmosphere` currently raises `NotImplementedError` in `SELib`.
  That branch is untouched by this refactor.
- Relativistic corrections: the whole codebase uses the
  non-relativistic `Δλ/λ ≈ v/c`. No change here.
- Frequency-dependent observer motion (e.g. earth rotation across
  an exposure): `Vd_obs` remains a single scalar per spectrum.

## Quick summary

- `Vd_sun` lives on `Atmosphere`, varies with depth, used by SELib /
  RT for the sun-frame wavelength shift of the absorption profile.
- `Vd_obs` is a single scalar, applied **once** at the very end to
  the output spectrum's wavelength axis. Never enters SELib.
- FAL's `vel` is vertical, not LOS — needs projection by `mu`.
- SELib's existing `wm_cm + w0*Vd/c` formula is physically correct
  **if** we interpret `Vd` as `Vd_sun`. The refactor is mostly a
  renaming and documentation exercise plus extracting the observer
  shift out of the SE inner loop.
