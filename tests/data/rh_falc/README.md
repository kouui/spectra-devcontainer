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
    RH's `Atmos/FALC_82.atmos` (Fontenla, Avrett & Loeser 1993 model C).
  - `nH_file` (82, 5) and `np_file` (82) [cm^-3]: the hydrogen level
    populations (levels 1..5) and the proton density stored in that atmosphere
    file.
  - `n1s_rescale` (82): `sum_levels(n_RH) * 1e-6 / (sum(nH_file) + np_file)`,
    with `n_RH` the converged populations [m^-3] of the RH run below. It is
    the factor by which RH's chemical-equilibrium step (hydrogen partly into
    H2) rescales the file populations; the van der Waals damping uses
    `nH_file[:, 0] * n1s_rescale` as n(1s). Departs from 1 by at most 1.2e-3.
- `mali_v2_falc_all_reference.npz`: the pinned MALI v2 result, written by the
  test itself (see the test docstring for the regeneration switch).

## Provenance

The RH run that produced `n1s_rescale`: `rhf1d` on `Atmos/FALC_82.atmos` with
`Atoms/H_6_CRD.atom` as the only (ACTIVE) atom and `Molecules/H2.molecule`
PASSIVE, abundances `Atoms/abundance.input`; keywords `NRAYS = 5`,
`N_MAX_SCATTER = 2`, `I_SUM = -1`, `ITER_LIMIT = 1e-6`, `N_MAX_ITER = 300`,
`NG_DELAY = 20`, `NG_ORDER = 2`, `NG_PERIOD = 5`, `STARTING_J = NEW_J`,
`VMICRO_CHAR = 3.0` km/s, `VMACRO_TRESH = 0.0`, `VACUUM_TO_AIR = FALSE`,
`N_THREADS = 1`; it converged in 51 iterations. The RH checkout used is
https://github.com/han-uitenbroek/RH at commit
c95bdf66ff6bc1c6cea0ed04dc688100170e8ee6 (clean working tree).

Reference for RH: Uitenbroek, H. 2001, ApJ, 557, 389
(DOI 10.1086/321659).

## Licence

No explicit licence file was found in the RH distribution used. The two RH
files are included as small test fixtures with attribution to their author;
if the RH authors object, remove them and the test with them.
