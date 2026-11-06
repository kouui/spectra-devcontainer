import unittest

import numpy as _numpy
from numpy import allclose as _ALLCLOSE

from spectra.ImportAll import *
from spectra.RadiativeTransfer import Profile as _Profile

voigt_ = _Profile.voigt_

_KWGS_CLOSE = {"rtol": 1.0e-05, "atol": 1.0e-20}


class Test_Profile(unittest.TestCase):
    def test_voigt_gaussian(self):

        x = _numpy.linspace(-3, 3, 61)
        a = 0
        gaussian = _Profile.gaussian_(x)
        voigt = _Profile.voigt_(a, x)
        self.assertTrue(_ALLCLOSE(voigt, gaussian, **_KWGS_CLOSE))  # type: ignore[arg-type]

    def test_voigt_hf(self):

        x = _numpy.linspace(-3, 3, 61)
        a = 0.1
        h, _f = _Profile.hf_(a, x)
        voigt = _Profile.voigt_(a, x)
        self.assertTrue(_ALLCLOSE(voigt, h, rtol=1.0e-03, atol=1.0e-20))

    def test_voigt_scalar_call_from_njit(self):

        # compiled kernels call voigt_ with scalars; cover both polynomial
        # branches (a < 0.01 and a >= 0.01)
        @nb_njit
        def _call_voigt(a, x):
            return voigt_(a, x)

        for a in (0.0, 1.0e-3, 0.1, 1.0):
            for x in _numpy.linspace(-5, 5, 21):
                self.assertEqual(_call_voigt(a, x), _Profile.voigt_(a, x))


if __name__ == "__main__":
    unittest.main()
