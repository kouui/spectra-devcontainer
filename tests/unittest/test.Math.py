import unittest

from numpy import allclose as _ALLCLOSE

from spectra.ImportAll import *
from spectra.Math import GaussLeg

_KWGS_CLOSE = {"rtol": 1.0e-05, "atol": 1.0e-20}


class Test_Gauss_Leg(unittest.TestCase):
    def test_gauss_quad_coe_odd_(self):

        a, b, n = -1, 1.0, 3
        xs, ws = GaussLeg.gauss_quad_coe_(a, b, n)
        xs_correct = (-0.7745966692414834, 0.0, +0.7745966692414834)
        ws_correct = (0.5555555555555556, 0.8888888888888888, 0.5555555555555556)

        self.assertTrue(_ALLCLOSE(xs, xs_correct, **_KWGS_CLOSE))  # type: ignore[arg-type]
        self.assertTrue(_ALLCLOSE(ws, ws_correct, **_KWGS_CLOSE))  # type: ignore[arg-type]

    def test_gauss_quad_coe_even_(self):

        a, b, n = -1, 1.0, 4
        xs, ws = GaussLeg.gauss_quad_coe_(a, b, n)
        xs_correct = (-0.8611363115940526, -0.3399810435848563, +0.3399810435848563, +0.8611363115940526)
        ws_correct = (0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538)

        self.assertTrue(_ALLCLOSE(xs, xs_correct, **_KWGS_CLOSE))  # type: ignore[arg-type]
        self.assertTrue(_ALLCLOSE(ws, ws_correct, **_KWGS_CLOSE))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
