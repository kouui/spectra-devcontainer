"""Regression tests for spectra.Util.RomanUtil"""

import pytest

from spectra.Util import RomanUtil


class TestIndexToRoman:
    def test_basic(self, ref):
        assert RomanUtil.index_to_roman_(1) == ref["RomanUtil.index_to_roman_1"]
        assert RomanUtil.index_to_roman_(2) == ref["RomanUtil.index_to_roman_2"]
        assert RomanUtil.index_to_roman_(3) == ref["RomanUtil.index_to_roman_3"]
        assert RomanUtil.index_to_roman_(10) == ref["RomanUtil.index_to_roman_10"]

    def test_invalid(self):
        with pytest.raises(ValueError):
            RomanUtil.index_to_roman_(0)


class TestRomanToIndex:
    def test_basic(self, ref):
        assert RomanUtil.roman_to_index_("I") == ref["RomanUtil.roman_to_index_I"]
        assert RomanUtil.roman_to_index_("II") == ref["RomanUtil.roman_to_index_II"]
        assert RomanUtil.roman_to_index_("X") == ref["RomanUtil.roman_to_index_X"]


class TestShiftRoman:
    def test_forward(self, ref):
        assert RomanUtil.shift_roman_("II", 1) == ref["RomanUtil.shift_roman_II_1"]

    def test_backward(self, ref):
        assert RomanUtil.shift_roman_("V", -2) == ref["RomanUtil.shift_roman_V_m2"]
