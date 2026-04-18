"""Regression tests for spectra.Math.BasicM"""

from spectra.Math import BasicM

from .conftest import assert_close


class TestIsOdd:
    def test_odd(self, ref):
        assert_close(BasicM.is_odd_(3), ref["BasicM.is_odd_3"])
        assert_close(BasicM.is_odd_(1), ref["BasicM.is_odd_1"])

    def test_even(self, ref):
        assert_close(BasicM.is_odd_(4), ref["BasicM.is_odd_4"])
        assert_close(BasicM.is_odd_(0), ref["BasicM.is_odd_0"])
