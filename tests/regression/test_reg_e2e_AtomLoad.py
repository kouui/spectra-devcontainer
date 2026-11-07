"""Regression tests for the AtomIO load pipeline.

For each of 8 canonical configurations, loads the atom via
``Atom.init_Atom_()`` (with the data-file paths read separately via
``AtomIO.read_conf_()``) and asserts that every field of the atom (the
wavelength mesh is read from ``atom._wave_mesh``) and every path matches
the reference snapshot in ``atom_reference_values.json``.

The snapshot was generated at commit ``a84128b`` (pre-refactoring) and
bit-exact equal to a snapshot generated on current ``main``, confirming
that the type-cleanup refactoring between the two was behaviour-preserving.
This test will fail on any subsequent change that alters the load output,
making it the safety net for the Stage 2 ``possibly-unbound`` cleanup.
"""

import pytest

from spectra.ImportAll import CFG
from spectra.Struct import Atom
from spectra.Util.AtomUtils import AtomIO as _AtomIO

from ._atom_serde import assert_atom_matches

# (conf path relative to repo root, readable name, is_hydrogen)
_CONFIGS: list[tuple[str, str, bool]] = [
    ("data/conf/H.conf", "H", True),
    ("data/conf/H6.conf", "H6", True),
    ("data/conf/H_theory.conf", "H_theory", True),
    ("data/conf/He.conf", "He", False),
    ("data/conf/He_I.conf", "He_I", False),
    ("data/conf/He_I_II.conf", "He_I_II", False),
    ("data/conf/Ca_II.conf", "Ca_II", False),
    ("data/conf/Ca_I-II-III.conf", "Ca_I-II-III", False),
]


# (RH atom file path relative to repo root, readable name)
_RH_ATOMS: list[tuple[str, str]] = [("tests/data/rh_falc/H_6_CRD.atom", "H_6_CRD_RH")]


@pytest.mark.parametrize(
    ("conf_rel", "name", "is_hydrogen"),
    _CONFIGS,
    ids=[cfg[1] for cfg in _CONFIGS],
)
def test_load_atom_matches_reference(conf_rel: str, name: str, is_hydrogen: bool, ref_atom):
    conf_path = str(CFG._ROOT_DIR / conf_rel)
    atom = Atom.init_Atom_(conf_path, is_hydrogen=is_hydrogen)
    path_dict = _AtomIO.read_conf_(conf_path)
    assert_atom_matches(atom, path_dict, name, ref_atom)


@pytest.mark.parametrize(("rh_rel", "name"), _RH_ATOMS, ids=[a[1] for a in _RH_ATOMS])
def test_load_RH_atom_matches_reference(rh_rel: str, name: str, ref_atom):
    rh_path = CFG._ROOT_DIR / rh_rel
    atom = Atom.init_Atom_from_RH_(rh_path)
    assert_atom_matches(atom, {"RH": str(rh_path)}, name, ref_atom)
