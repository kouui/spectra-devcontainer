"""Regression tests for spectra.RadiativeTransfer.Thin"""

from spectra.ImportAll import *
from spectra.RadiativeTransfer import Thin

from .conftest import assert_close


class TestRelativeFlux:
    def test_scalar(self, ref):
        assert_close(
            Thin.relative_flux_(6.27e8, CST.c_ / 1216e-8, int(1e5)),
            ref["Thin.relative_flux_scalar"],
        )
