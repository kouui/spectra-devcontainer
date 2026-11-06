"""Regression tests for spectra.Struct.Atmosphere.init_VAL_"""

import pytest

from spectra.Struct import Atmosphere

from .conftest import assert_close

# tau5 is the trapezoidal integral of the continuum opacity, so it is pinned
# together with the depth grid and the thermodynamic columns it is built from.
_FIELDS = ("Z", "Te", "Ne", "Nh", "Vt", "tau5")


@pytest.mark.parametrize("val_type", ["A", "C", "F"])
def test_init_VAL_(ref, val_type):
    atmos = Atmosphere.init_VAL_(val_type)
    for name in _FIELDS:
        assert_close(getattr(atmos, name), ref[f"Atmosphere.VAL_{val_type}.{name}"], rtol=1e-12)


def test_init_VAL_unknown_type():
    with pytest.raises(NotImplementedError):
        Atmosphere.init_VAL_("B")
