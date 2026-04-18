"""Regression tests for spectra.Atomic.BasicP"""

from spectra.Atomic import BasicP
from spectra.ImportAll import *

from .conftest import assert_close


class TestWaveFreqConversion:
    def test_wave_to_freq(self, ref):
        assert_close(BasicP.wave_to_freq_(5000e-8), ref["BasicP.wave_to_freq"])

    def test_freq_to_wave(self, ref):
        assert_close(BasicP.freq_to_wave_(CST.c_ / 5000e-8), ref["BasicP.freq_to_wave"])


class TestDopplerShift:
    def test_dop_vel_to_shift(self, ref):
        assert_close(BasicP.dop_vel_to_shift_(5000e-8, 1e6), ref["BasicP.dop_vel_to_shift"])


class TestDopplerWidth:
    def test_freq(self, ref):
        f_lya = CST.c_ / 1216e-8
        assert_close(BasicP.doppler_width_(f_lya, 7000.0, 5.0e5, 1.0), ref["BasicP.doppler_width_freq"])

    def test_wave(self, ref):
        assert_close(BasicP.doppler_width_(1216e-8, 7000.0, 5.0e5, 1.0), ref["BasicP.doppler_width_wave"])


class TestDampingConst:
    def test_damping_const_a(self, ref):
        f_lya = CST.c_ / 1216e-8
        dop_hz = BasicP.doppler_width_(f_lya, 7000.0, 5.0e5, 1.0)
        assert_close(BasicP.damping_const_a_(6.27e8, dop_hz), ref["BasicP.damping_const_a"])


class TestRefractiveIndex:
    def test_cm(self, ref):
        assert_close(BasicP.refractive_index_in_air_(5000e-8, "cm"), ref["BasicP.refractive_index_in_air_cm"])

    def test_um(self, ref):
        assert_close(BasicP.refractive_index_in_air_(0.5, "um"), ref["BasicP.refractive_index_in_air_um"])

    def test_nm(self, ref):
        assert_close(BasicP.refractive_index_in_air_(500.0, "nm"), ref["BasicP.refractive_index_in_air_nm"])

    def test_AA(self, ref):
        assert_close(BasicP.refractive_index_in_air_(5000.0, "AA"), ref["BasicP.refractive_index_in_air_AA"])


class TestAirVacuumConversion:
    def test_air_to_vacuum(self, ref):
        assert_close(BasicP.air_to_vacuum_(5000.0, "AA"), ref["BasicP.air_to_vacuum_AA"])

    def test_vacuum_to_air(self, ref):
        assert_close(BasicP.vacuum_to_air_(5000.0, "AA"), ref["BasicP.vacuum_to_air_AA"])
