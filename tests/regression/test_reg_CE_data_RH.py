"""The excitation tables converted from RH atom files must reproduce the rates of the RH CE entries."""

import numpy as np
import pytest

from spectra import Constants as Cst
from spectra.Atomic import Collision
from spectra.ImportAll import CFG
from spectra.Struct import Atom

_CASES = {
    "He": ("data/conf/He.conf", "data/atom/He/He_10830_d3.RH.atom"),
    "Ca_I-II-III": ("data/conf/Ca_I-II-III.conf", "data/atom/Ca_I-II-III/CaI+II_45.atom.RH.20220802.txt"),
}


def _read_rh_collisions_(path):
    """Temperatures [K] and the CE/OMEGA entries in file order as (kind, values).

    CE values are converted to cm^3 s^-1 K^-1/2; OMEGA entries are only kept for their position.
    """
    temps = None
    entries = []
    for line in path.read_text().splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "TEMP" and temps is None:
            temps = np.array([float(w) for w in words[2 : 2 + int(words[1])]])
        elif words[0] in ("CE", "OMEGA"):
            assert temps is not None
            entries.append((words[0], np.array([float(w) for w in words[3 : 3 + temps.size]]) * 1.0e6))
    assert temps is not None
    return temps, entries


@pytest.mark.parametrize("name", list(_CASES))
def test_ce_rates_match_rh_ce(name):
    conf, rh_file = _CASES[name]
    atom = Atom.init_Atom_(str(CFG._ROOT_DIR / conf))
    temps, entries = _read_rh_collisions_(CFG._ROOT_DIR / rh_file)

    # Merged levels (e.g. 3P 0+1+2) rule out matching by level index; the converter writes the
    # entries in file order, one row each, so the entry k belongs to the transition k.
    assert atom.CE.Omega_table.shape == (len(entries), temps.size)
    # Omega is tabulated under the temperature rounded to 3 digits, but it was computed at the RH one.
    np.testing.assert_allclose(atom.CE.Te_table, temps, rtol=1e-2)

    deviations = []
    for k, (kind, ce) in enumerate(entries):
        if kind != "CE":
            continue
        c = atom.CE.Coe[k]
        rate = Collision.CE_rate_coe_(atom.CE.Omega_table[k] * c["f1"] / c["f2"], temps, c["gi"], c["dEij"])
        expected = ce * np.sqrt(temps) * np.exp(-c["dEij"] / (Cst.k_ * temps))
        deviations.append(rate / expected - 1.0)
    dev = np.concatenate(deviations)

    n_ce = sum(kind == "CE" for kind, _ in entries)
    assert n_ce > 0
    assert dev.size == n_ce * temps.size
    # Each Omega keeps the 4 significant digits of the original conversion, so a single value is only good
    # to 5e-4 and a wrong prefactor (1e-4) hides in that. It shows in the mean, which the rounding leaves
    # near zero because its scatter averages out over all values.
    assert np.abs(dev).max() < 6e-4
    assert abs(dev.mean()) < 3e-5
