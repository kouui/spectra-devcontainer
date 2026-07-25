"""Numba compilation of Feautrier.formal_improved_RH_.

CFG._IS_JIT is False by default, so the production module never compiles the
solver during a normal test run. These tests compile it explicitly: the IntEnum
branch, the NamedTuple return and the keyword defaults are all features that
work in the interpreter but can fail to lower in nopython mode.
"""

import numpy as np
import pytest

from spectra.Enums import E_FEAUTRIER_ORDER
from spectra.ImportAll import NB_NJIT_KWGS, nb_njit
from spectra.RadiativeTransfer import Feautrier

SECOND = E_FEAUTRIER_ORDER.SECOND
HERMITE = E_FEAUTRIER_ORDER.HERMITE

_jitted = nb_njit(**NB_NJIT_KWGS)(Feautrier.formal_improved_RH_)

_TAU = np.logspace(-4.0, 2.0, 41)
_S = np.linspace(0.5, 2.0, 41)
_ARGS = (_TAU, _S, 0.5, 0.0, 1.0, 0.0, 2.0)


@pytest.mark.parametrize("order", [SECOND, HERMITE])
@pytest.mark.parametrize("with_psi", [False, True])
def test_jit_matches_interpreted(order, with_psi):
    ref = Feautrier.formal_improved_RH_(*_ARGS, order, with_psi)
    got = _jitted(*_ARGS, order, with_psi)
    assert np.allclose(got.j, ref.j, rtol=1e-13, atol=0.0)
    assert got.I_emergent == pytest.approx(ref.I_emergent, rel=1e-13)
    assert got.Psi.shape == ref.Psi.shape
    assert np.allclose(got.Psi, ref.Psi, rtol=1e-13, atol=0.0)


def test_jit_accepts_keyword_arguments():
    ref = Feautrier.formal_improved_RH_(*_ARGS, HERMITE, True)
    got = _jitted(*_ARGS, order=HERMITE, with_psi=True)
    assert np.allclose(got.j, ref.j, rtol=1e-13, atol=0.0)


def test_jit_accepts_omitted_defaults():
    ref = Feautrier.formal_improved_RH_(*_ARGS)
    got = _jitted(*_ARGS)
    assert np.allclose(got.j, ref.j, rtol=1e-13, atol=0.0)
    assert got.Psi.shape == (0,)
