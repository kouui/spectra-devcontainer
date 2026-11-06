# rh_falc: FALC + 6-level hydrogen inputs for the MALI v2 end-to-end test

Input data of `tests/unittest/MALI_v2/test.falc_e2e.py`. The test pins the
frozen MALI v2 solver on the FALC atmosphere with the RH 6-level hydrogen atom.

## Files

- `H_6_CRD.atom`: verbatim copy of `Atoms/H_6_CRD.atom` from the RH
  distribution (6 levels incl. the proton, 10 lines, 5 continua, collision
  tables). The test parses it, so it must stay byte-identical to the RH file.
- `falc.npz`: the FALC model and the quantities derived from an RH run, all
  float64, 82 depth points ordered from the top of the atmosphere downwards:
  - `cmass` [g cm^-2], `T` [K], `ne` [cm^-3], `vturb` [cm s^-1]: columns of
    RH's `Atmos/FALC_82.atmos` (model C of Fontenla, Avrett & Loeser 1993, see the reference below).
  - `nH_file` (82, 5) and `np_file` (82) [cm^-3]: the hydrogen level
    populations (levels 1..5) and the proton density stored in that atmosphere
    file.
  - `n1s_rescale` (82): `sum_levels(n_RH) * 1e-6 / (sum(nH_file) + np_file)`,
    with `n_RH` the converged populations [m^-3] of the RH run below. It is
    the factor by which RH's chemical-equilibrium step (hydrogen partly into
    H2) rescales the file populations; the van der Waals damping uses
    `nH_file[:, 0] * n1s_rescale` as n(1s). Departs from 1 by at most 1.2e-3.
- `mali_v2_falc_all_reference.npz`: the pinned MALI v2 result (populations,
  LTE populations, iteration count, wavelength axis, and a deeper-converged
  fixed point), written by the test itself (see the test docstring for the
  regeneration switch).

## Provenance

The RH run that produced `n1s_rescale`: `rhf1d` on `Atmos/FALC_82.atmos` with
`Atoms/H_6_CRD.atom` as the only metal, ACTIVE, initial radiation
`ZERO_RADIATION`, and `Molecules/H2.molecule` PASSIVE with `LTE_POPULATIONS`;
abundances `Atoms/abundance.input`; keywords `NRAYS = 5`,
`N_MAX_SCATTER = 2`, `I_SUM = -1`, `ITER_LIMIT = 1e-6`, `N_MAX_ITER = 300`,
`NG_DELAY = 20`, `NG_ORDER = 2`, `NG_PERIOD = 5`, `STARTING_J = NEW_J`,
`OLD_BACKGROUND = FALSE`, `VMICRO_CHAR = 3.0` km/s, `VMACRO_TRESH = 0.0`,
`VACUUM_TO_AIR = FALSE`, `N_THREADS = 1`; it converged in 51 iterations.
The RH source tree used is https://github.com/han-uitenbroek/RH at commit
c95bdf66ff6bc1c6cea0ed04dc688100170e8ee6 with no tracked modifications. That
commit identifies the sources of the atom and atmosphere files. The `rhf1d`
binary was built separately and is newer than the run, so the commit is not a
record of the exact code that produced `n1s_rescale`.

Reference for the model: Fontenla, J. M., Avrett, E. H., & Loeser, R. 1993,
ApJ, 406, 319 (DOI 10.1086/172443; checked against the Crossref record). The
header of `FALC_82.atmos` itself only says "Model file from Oct 16, 1996,
extracted Jan 14 1997".

Reference for RH: Uitenbroek, H. 2001, ApJ, 557, 389
(DOI 10.1086/321659).

## Licence

No explicit licence file was found in the RH distribution used, so the
redistribution rights of the RH-derived files are not established.

- `H_6_CRD.atom` is a verbatim copy of an RH file.
- `falc.npz` is derived: its model columns come from RH's `FALC_82.atmos` and
  `n1s_rescale` comes from the output of an RH run (see Provenance).

Both are included only as small test fixtures, with attribution to their
author. Removal path if the RH authors object: delete this directory
(`tests/data/rh_falc/`), `tests/unittest/MALI_v2/_falc_setup.py` and
`tests/unittest/MALI_v2/test.falc_e2e.py`; no other test depends on them.
