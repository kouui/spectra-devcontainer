"""Rewrite the energy column of the theoretical hydrogen ``.Level`` files.

The historical ``data/atom/H/H*.Level`` files were generated with an
inconsistent Rydberg convention (some used R_inf for the level spacing while
hardcoding the proton-mass ionization literal). This script recomputes every
level energy from a single source of truth -- ``CST.E_Rydberg_H_`` (proton-mass
corrected) -- so that:

    bound level n : E = E_Rydberg_H_ / eV2erg_ * (1 - 1/n**2)   [eV]
    continuum     : E = E_Rydberg_H_ / eV2erg_                  [eV]

Only the trailing energy token of each level line is rewritten; all column
alignment and headers are preserved byte-for-byte (the historical files do not
match the current ``make_hydrogen_levels_`` template formatting, and normalising
that formatting is out of scope here).
"""

from __future__ import annotations

from pathlib import Path
import re

from spectra import Configurations as CFG
from spectra import Constants as CST

_RY_EV = CST.E_Rydberg_H_ / CST.eV2erg_

# A level line is 9 whitespace-separated fields ending in a %.7E float token.
_ERG = re.compile(r"(-?\d\.\d+E[+-]\d+)(\s*)$")


def _new_erg(fields: list[str]) -> float:
    stage = fields[7]
    if stage == "2":  # continuum
        return _RY_EV
    n = fields[3]
    if n == "1":  # ground state
        return 0.0
    return _RY_EV * (1.0 - 1.0 / int(n) ** 2)


def rewrite_(path: Path) -> int:
    n_changed = 0
    out = []
    for line in path.read_text().splitlines(keepends=True):
        fields = line.split()
        if len(fields) == 9 and _ERG.search(line):
            erg = _new_erg(fields)
            new_line = _ERG.sub(lambda m, e=erg: f"{e:.7E}{m.group(2)}", line)
            if new_line != line:
                n_changed += 1
            out.append(new_line)
        else:
            out.append(line)
    path.write_text("".join(out))
    return n_changed


def main() -> None:
    h_dir = CFG._ROOT_DIR / "data" / "atom" / "H"
    for path in sorted(h_dir.glob("H*.Level")):
        n = rewrite_(path)
        print(f"{path.name:14s}  {n} level energies rewritten")


if __name__ == "__main__":
    main()
