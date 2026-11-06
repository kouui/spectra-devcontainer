"""The He excitation table must reproduce the rates of the RH atom file it was converted from."""

import numpy as np

from spectra import Constants as Cst
from spectra.Atomic import Collision
from spectra.ImportAll import CFG
from spectra.Struct import Atom

_RH_ATOM = CFG._ROOT_DIR / "data/atom/He/He_10830_d3.RH.atom"


def _read_rh_ce_(path):
    """RH "CE" rows keyed by (upper, lower) level index, in cm^3 s^-1 K^-1/2."""
    rows = {}
    for line in path.read_text().splitlines():
        words = line.split()
        if words and words[0] == "CE":
            rows[(int(words[1]), int(words[2]))] = np.array([float(w) for w in words[3:10]]) * 1.0e6
    return rows


def test_He_ce_rates_match_rh_ce():
    atom = Atom.init_Atom_(str(CFG._ROOT_DIR / "data/conf/He.conf"))
    Te = atom.CE.Te_table
    line_of = {(int(c["idxJ"]), int(c["idxI"])): k for k, c in enumerate(atom.CE.Coe)}

    deviations = []
    for (j, i), ce in _read_rh_ce_(_RH_ATOM).items():
        k = line_of[(j, i)]
        c = atom.CE.Coe[k]
        rate = Collision.CE_rate_coe_(atom.CE.Omega_table[k] * c["f1"] / c["f2"], Te, c["gi"], c["dEij"])
        expected = ce * np.sqrt(Te) * np.exp(-c["dEij"] / (Cst.k_ * Te))
        deviations.append(rate / expected - 1.0)
    dev = np.concatenate(deviations)

    assert dev.size == 190 * Te.size
    # Each Omega keeps the 4 significant digits of the original conversion, so a single value is only good
    # to 5e-4; a wrong prefactor (1e-4) hides in that. It shows in the mean, which rounding leaves near
    # zero (3e-6 here against a rounding scatter of 1.4e-4 over 1330 values).
    assert np.abs(dev).max() < 6e-4
    assert abs(dev.mean()) < 3e-5
