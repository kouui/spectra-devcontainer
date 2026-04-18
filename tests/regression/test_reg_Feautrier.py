"""Regression tests for spectra.RadiativeTransfer.Feautrier"""

import numpy as np

from spectra.RadiativeTransfer import Feautrier

from .conftest import assert_close


class TestFormalImprovedRH:
    def test_const_source_symmetric(self, ref):
        ND = 20
        tau = np.logspace(-3, 2, ND)
        S = np.ones(ND) * 1.0
        j = Feautrier.formal_improved_RH_(tau, S, 1.0, 0.0, 0.0, 1.0, 0.0)
        assert_close(j, ref["Feautrier.const_source_symmetric"])

    def test_linear_source_diffusion(self, ref):
        ND = 20
        tau = np.logspace(-3, 2, ND)
        S = np.linspace(0.5, 2.0, ND)
        j = Feautrier.formal_improved_RH_(tau, S, 1.0, 0.0, 0.0, 0.0, S[-1])
        assert_close(j, ref["Feautrier.linear_source_diffusion"])
