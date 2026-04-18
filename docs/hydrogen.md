# Theoretical Hydrogen Atom — Workflow

This document describes how `init_theoretical_hydrogen_atom_` in
`spectra_src/Struct/Atom.py` constructs a fully theoretical hydrogen
`Atom` struct without reading any data file from disk.

## 1. Entry point

```python
from spectra_src.Struct.Atom import init_theoretical_hydrogen_atom_

atom, waveMesh = init_theoretical_hydrogen_atom_(nLevel=8)
```

`nLevel` is the total number of levels **including** the continuum.
It must be `>= 3`. The bound states correspond to principal quantum
numbers `n = 1 .. nLevel-1` and the last entry is the H II continuum.

## 2. Level structure (analytic Rydberg formula)

Energies are computed from

```
E_n  = E_ionize - Ry / n^2                [erg]
g_n  = 2 n^2                              (statistical weight)
Ry   = R_H * c * h                        (Rydberg energy unit)
R_H  = 109677.59 cm^-1                    (hydrogen Rydberg constant)
```

For `nLevel = 8` the resulting level diagram looks like this:

```
  E [eV]
  13.60 +-- ----------------- continuum (stage=2, g=1)
        |      ^
  13.32 |    n=7 ---------------- g=98
  13.22 |    n=6 ---------------- g=72
  13.06 |    n=5 ---------------- g=50
  12.75 |    n=4 ---------------- g=32
  12.09 |    n=3 ---------------- g=18
        |      |   bound states
  10.20 |    n=2 ---------------- g=8     (stage=1)
        |      |
        |      |
   0.00 +-- n=1 (1s 2S 1/2) ----- g=2     (ground)
```

The `Level` numpy struct array is filled directly in memory; no
`.Level` ASCII file is read or written.

## 3. Data-source decision matrix

For every kind of atomic data, the theoretical path either computes
it immediately during construction, or defers it to runtime (SE solve):

```
  +------------+-----------------------------+--------------------------+
  |   Data     |   Where it is provided      |   Formula / source       |
  +------------+-----------------------------+--------------------------+
  | Level      | init_theoretical_hydrogen_  | E_n = E_ionize - Ry/n^2  |
  |            |   atom_ (this function)     |                          |
  +------------+-----------------------------+--------------------------+
  | Aji        | AtomIO.make_Atom_Line_      | Hydrogen.einstein_A_     |
  |  (Line)    |   (called with path=None)   |   coefficient_(ni,nj)    |
  +------------+-----------------------------+--------------------------+
  | PI alpha   | AtomIO.make_Atom_PI_        | DegenerateN.compute_PI_  |
  |  (Cont)    |   (path=None, HYDROGEN)     |   cross_section_         |
  +------------+-----------------------------+--------------------------+
  | CE rates   | deferred to SE solve        | Hydrogen.CE_rate_coe_    |
  |            |   (data_src_CE = CALCULATE) |                          |
  +------------+-----------------------------+--------------------------+
  | CI rates   | deferred to SE solve        | Hydrogen.CI_rate_coe_    |
  |            |   (data_src_CI = CALCULATE) |                          |
  +------------+-----------------------------+--------------------------+
  | RL         | empty (nRL = 0)             | -                        |
  +------------+-----------------------------+--------------------------+
```

`CE` and `CI` return empty `Te_table` / `Omega_table` arrays, and the
`data_source_*` flag is set to `CALCULATE`. At SE solve time,
`Function/SEquil/SELib._get_Cij_` takes the `CALCULATE` branch and
calls the hydrogen analytic rate coefficients directly.

## 4. Construction flow

```
                    init_theoretical_hydrogen_atom_(nLevel)
                                    |
                                    v
        +---------------------------+---------------------------+
        |  Build Level numpy array (Rydberg formula, in memory) |
        |  Build Level_info_table : tuple of (conf, term, J)    |
        +---------------------------+---------------------------+
                                    |
                                    v
             AtomIO.nLine_nCont_nTran_(Level["stage"])
                    -> nLine, nCont, nTran, _has_continuum
                                    |
                                    v
             AtomIO.prepare_idx_ctj_mapping_(...)
                    -> Line_idx_table, Line_ctj_table,
                       Cont_idx_table, Cont_ctj_table
                                    |
        +---------------------------+---------------------------+
        |                           |                           |
        v                           v                           v
  AtomIO.make_Atom_Cont_     AtomIO.make_Atom_Line_      AtomIO.make_Atom_RL_
     (nCont, ...)              (path=None, HYDROGEN)        (path=None)
        |                     -> Aji via analytic fn       -> empty
        |                           |                           |
        +---------------------------+---------------------------+
                                    |
                                    v
             AtomIO.make_Atom_CECI_(path=None, "CE", ...)
                 -> empty Omega table, data_src_CE = CALCULATE
                                    |
                                    v
             AtomIO.make_Atom_CECI_(path=None, "CI", ...)
                 -> empty Omega table, data_src_CI = CALCULATE
                                    |
                                    v
             WavelengthMesh.init_Wave_Mesh_(Cont, Line, RL.Coe)
                                    |
                                    v
             AtomIO.make_Atom_PI_(path=None, Level, Cont,
                                  waveMesh.Cont_mesh, HYDROGEN, ...)
                 -> alpha_interp via DegenerateN analytic fn
                                    |
                                    v
             Assemble Atom(...) dataclass and Photo_Ionization,
             Collisional_Transition, Radiative_Line, ATOMIC_DATA_SOURCE
                                    |
                                    v
                          return (atom, waveMesh)
```

## 5. What happens later, at SE solve time

The `atom` object carries `_atomic_data_source.CE == CALCULATE` and
`.CI == CALCULATE`. When the statistical equilibrium solver computes
the collisional rate matrix in `Function/SEquil/SELib.py`:

```
  _get_Cij_(Line, Cont, Te, atom_type, CE_*, CI_*)
       |
       |  data_src_CE == CALCULATE and atom_type == HYDROGEN
       v
  for k in range(nLine):
      Cij[k] = _Hydrogen.CE_rate_coe_(Line["ni"][k], Line["nj"][k], Te)

       |  data_src_CI == CALCULATE and atom_type == HYDROGEN
       v
  for k in range(nCont):
      Cij[k+nLine] = _Hydrogen.CI_rate_coe_(Cont["ni"][k], Te)
```

So the collisional rates are lazily computed per temperature, directly
from the analytic formulas in `spectra_src/Atomic/Hydrogen.py`.

## 6. Properties summary

```
  +-----------------------+--------------------------------------+
  | nLevel                | nLevel (user input, >= 3)            |
  | nLine                 | (nLevel-1)*(nLevel-2) / 2            |
  | nCont                 | nLevel - 1                           |
  | nTran                 | nLine + nCont                        |
  | nRL                   | 0                                    |
  | _atom_type            | E_ATOM.HYDROGEN                      |
  | _has_continuum        | True (when nLevel >= 2)              |
  | _atomic_data_source   | AJI=CALCULATE, CE=CALCULATE,         |
  |                       | CI=CALCULATE, PI=CALCULATE           |
  | files written         | none                                 |
  | files read            | none                                 |
  +-----------------------+--------------------------------------+
```

## 7. When to use this

- Quick SE or slab-model experiments where you do not want to manage
  `.conf` / `.Level` files on disk.
- Parameter sweeps over `nLevel` (e.g. convergence tests for the number
  of bound levels included in the model).
- Unit tests and notebooks that should be hermetic (no data files).

When you need experimentally measured atomic data (e.g. tabulated
Aji from NIST, tabulated CE Omega tables, tabulated PI cross
sections), use `init_Atom_` with a real `.conf` file instead.
