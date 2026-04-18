"""Regression tests for spectra.Atomic.PhotoIonize"""

import numpy as np

from spectra.Atomic import Hydrogen, PhotoIonize
from spectra.ImportAll import *

from .conftest import assert_close


class TestInterpolatePIIntensity:
    def test_interpolation(self, ref):
        backRad = np.load(CFG._ROOT_DIR / "data" / "intensity" / "atlas" / "QS" / "atlas_QS.20221118.npy")
        cont_mesh = np.array([[4000e-8, 4500e-8, 5000e-8],
                               [3000e-8, 3500e-8, 4000e-8]])
        result = PhotoIonize.interpolate_PI_intensity_(backRad, cont_mesh)
        assert_close(result, ref["PhotoIonize.interp_PI_intensity"])


class TestInterpolatePIAlpha:
    def test_interpolation(self, ref):
        alpha_table = np.array([[912e-8, 800e-8, 700e-8, 500e-8],
                                 [6.3e-18, 4.5e-18, 3.0e-18, 1.5e-18]])
        alpha_table_idxs = np.array([[0, 4]])
        cont_mesh = np.array([[912e-8, 800e-8, 700e-8, 600e-8, 500e-8]])
        result = PhotoIonize.interpolate_PI_alpha_(alpha_table, alpha_table_idxs, cont_mesh)
        assert_close(result, ref["PhotoIonize.interp_PI_alpha"])


class TestBoundFreeRadiativeTransitionCoefficient:
    def test_hydrogen_ni1(self, ref):
        wave = np.linspace(500e-8, 912e-8, 20)
        J = np.ones(20) * 1e-6
        alpha_bf = np.array([Hydrogen.PI_cross_section_cm_(1, w, 1) for w in wave])
        Rik, Rki_stim, Rki_spon = PhotoIonize.bound_free_radiative_transition_coefficient_(
            wave, J, alpha_bf, 7000.0, 0.01)
        assert_close(Rik, ref["PhotoIonize.bf_Rik"])
        assert_close(Rki_stim, ref["PhotoIonize.bf_Rki_stim"])
        assert_close(Rki_spon, ref["PhotoIonize.bf_Rki_spon"])
