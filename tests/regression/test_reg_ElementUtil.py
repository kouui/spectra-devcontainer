"""Regression tests for spectra.Util.ElementUtil"""

from spectra.Util import ElementUtil

from .conftest import assert_close


class TestSymToZ:
    def test_H(self, ref):
        assert ElementUtil.sym_to_z_("H") == ref["ElementUtil.sym_to_z_H"]

    def test_He(self, ref):
        assert ElementUtil.sym_to_z_("He") == ref["ElementUtil.sym_to_z_He"]

    def test_Ca(self, ref):
        assert ElementUtil.sym_to_z_("Ca") == ref["ElementUtil.sym_to_z_Ca"]

    def test_Fe(self, ref):
        assert ElementUtil.sym_to_z_("Fe") == ref["ElementUtil.sym_to_z_Fe"]


class TestSymToMass:
    def test_H(self, ref):
        assert_close(ElementUtil.sym_to_mass_("H"), ref["ElementUtil.sym_to_mass_H"])

    def test_He(self, ref):
        assert_close(ElementUtil.sym_to_mass_("He"), ref["ElementUtil.sym_to_mass_He"])

    def test_Ca(self, ref):
        assert_close(ElementUtil.sym_to_mass_("Ca"), ref["ElementUtil.sym_to_mass_Ca"])


class TestSymToAbun:
    def test_H(self, ref):
        assert_close(ElementUtil.sym_to_abun_("H"), ref["ElementUtil.sym_to_abun_H"])

    def test_He(self, ref):
        assert_close(ElementUtil.sym_to_abun_("He"), ref["ElementUtil.sym_to_abun_He"])


class TestFormatting:
    def test_format_sym(self, ref):
        assert ElementUtil.format_sym_("h") == ref["ElementUtil.format_sym_h"]
        assert ElementUtil.format_sym_("HE") == ref["ElementUtil.format_sym_HE"]

    def test_format_stage(self, ref):
        assert ElementUtil.format_stage_("ii") == ref["ElementUtil.format_stage_ii"]

    def test_format_ion(self, ref):
        assert ElementUtil.format_ion_("ca_ii") == ref["ElementUtil.format_ion_ca_ii"]


class TestIonConversion:
    def test_ion_to_sym_and_stage(self, ref):
        sym, stage = ElementUtil.ion_to_sym_and_stage_("ca_ii")
        expected = ref["ElementUtil.ion_to_sym_ca_ii"]
        assert sym == expected[0]
        assert stage == expected[1]

    def test_sym_and_stage_to_ion(self, ref):
        assert ElementUtil.sym_and_stage_to_ion_("Ca", "II") == ref["ElementUtil.sym_and_stage_to_ion"]


class TestIonizPotential:
    def test_Ca_II(self, ref):
        assert_close(ElementUtil.ion_to_ioniz_potential_("Ca_II"),
                     ref["ElementUtil.ion_to_ioniz_potential_Ca_II"])


class TestShiftIon:
    def test_shift(self, ref):
        assert ElementUtil.shfit_ion_("Ca_II", 1) == ref["ElementUtil.shfit_ion_Ca_II_1"]
