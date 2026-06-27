"""Regression tests for spectra.Atomic.Hydrogen"""

import pytest

from spectra import Constants as _CST
from spectra.Atomic import Hydrogen

from .conftest import assert_close


def _pi_edge_wave_(ni: int) -> float:
    """Bound-free threshold wavelength [cm] for level ``ni``.

    Derived from the Rydberg constant the code itself uses so the
    "at threshold" cases track the ionization edge exactly. A hardcoded
    wavelength sits on the cross-section discontinuity and would flip
    between 0 and the edge peak under any sub-angstrom change to the
    constant (e.g. the Balmer edge moves ~2 A between R_inf and R_H).
    """
    return _CST.h_ * _CST.c_ / (_CST.E_Rydberg_H_ / ni**2)


class TestGauntFactor:
    @pytest.mark.parametrize(
        "ni,x", [(1, 1.5), (1, 2.0), (1, 5.0), (2, 1.5), (2, 2.0), (2, 5.0), (3, 1.5), (3, 2.0), (3, 5.0)]
    )
    def test_gaunt_factor(self, ref, ni, x):
        assert_close(Hydrogen.gaunt_factor_(ni, x), ref[f"Hydrogen.gaunt_factor_ni{ni}_x{x}"])


class TestOscillatorStrength:
    @pytest.mark.parametrize("ni,nj", [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
    def test_fij(self, ref, ni, nj):
        assert_close(Hydrogen.absorption_oscillator_strength_(ni, nj), ref[f"Hydrogen.fij_ni{ni}_nj{nj}"])


class TestEinsteinA:
    @pytest.mark.parametrize("ni,nj", [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (1, 5)])
    def test_Aji(self, ref, ni, nj):
        assert_close(Hydrogen.einstein_A_coefficient_(ni, nj), ref[f"Hydrogen.Aji_ni{ni}_nj{nj}"])


class TestCERateCoe:
    @pytest.mark.parametrize(
        "ni,nj,Te",
        [
            (1, 2, 5000),
            (1, 2, 7000),
            (1, 2, 10000),
            (1, 3, 5000),
            (1, 3, 7000),
            (1, 3, 10000),
            (2, 3, 5000),
            (2, 3, 7000),
            (2, 3, 10000),
        ],
    )
    def test_CE(self, ref, ni, nj, Te):
        assert_close(Hydrogen.CE_rate_coe_(ni, nj, float(Te)), ref[f"Hydrogen.CE_ni{ni}_nj{nj}_Te{Te}"])


class TestCIRateCoe:
    @pytest.mark.parametrize(
        "ni,Te",
        [
            (1, 5000),
            (1, 7000),
            (1, 10000),
            (2, 5000),
            (2, 7000),
            (2, 10000),
            (3, 5000),
            (3, 7000),
            (3, 10000),
        ],
    )
    def test_CI(self, ref, ni, Te):
        assert_close(Hydrogen.CI_rate_coe_(ni, float(Te)), ref[f"Hydrogen.CI_ni{ni}_Te{Te}"])


class TestPICrossSection:
    def test_ni1_at_threshold(self, ref):
        assert_close(
            Hydrogen.PI_cross_section_cm_(1, _pi_edge_wave_(1), 1),
            ref["Hydrogen.PI_ni1_w912AA"],
        )

    def test_ni1_above_threshold(self, ref):
        assert_close(Hydrogen.PI_cross_section_cm_(1, 500e-8, 1), ref["Hydrogen.PI_ni1_w500AA"])

    def test_ni2_at_threshold(self, ref):
        assert_close(
            Hydrogen.PI_cross_section_cm_(2, _pi_edge_wave_(2), 1),
            ref["Hydrogen.PI_ni2_w3647AA"],
        )

    def test_below_threshold(self, ref):
        assert_close(Hydrogen.PI_cross_section_cm_(1, 2000e-8, 1), ref["Hydrogen.PI_ni1_w2000AA"])


class TestRatioEtranToEionize:
    def test_ratio(self, ref):
        assert_close(Hydrogen.ratio_Etran_to_Eionize_(1, 500e-8), ref["Hydrogen.ratio_Etran_ni1_w500AA"])


class TestRkiSponRateCoe:
    @pytest.mark.parametrize("ni", [1, 2, 3])
    def test_Rki_spon(self, ref, ni):
        assert_close(Hydrogen.Rki_spon_rate_coe_(ni, 7000.0), ref[f"Hydrogen.Rki_spon_ni{ni}_Te7000"])


class TestLinearStark:
    def test_broadening(self, ref):
        if "Hydrogen.LinearStark_n2_n3_Ne1e11" in ref:
            assert_close(
                Hydrogen.collisional_broadening_LinearStark_(2, 3, 1e11),
                ref["Hydrogen.LinearStark_n2_n3_Ne1e11"],
            )
