"""Tests of the RH atom file parser `Util.RH.AtomFile` (fast, no JIT needed).

Only the raw values and the rejection of unsupported or malformed input are
checked here; what the values become in an `Atom` is in `test.RH_Atom.py`.
"""

from pathlib import Path

import pytest

from spectra.Util.RH import AtomFile

RH_ATOM = Path(__file__).resolve().parents[1] / "data" / "rh_falc" / "H_6_CRD.atom"


@pytest.fixture(scope="module")
def atom_text():
    return RH_ATOM.read_text()


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "x.atom"
    p.write_text(text)
    return p


def test_parses_the_raw_values():
    rh = AtomFile.read_RH_atom_(RH_ATOM)
    assert rh.element == "H"
    assert rh.level_E_cm[1] == 82258.211  # cm^-1, unconverted
    assert rh.cont_alpha0_m2[0] == 6.152e-22  # m^2, unconverted
    assert rh.cont_lamb_min_nm[0] == 22.794  # nm, unconverted
    assert rh.line_symm[0] == "ASYMM"
    assert rh.CE_idx[0].tolist() == [1, 0]  # (upper, lower) as in the file
    assert rh.CI_idx[0].tolist() == [0, 5]  # (lower, upper) as in the file
    assert rh.CE_coe[0, 0] == 9.750e-16  # SI, unconverted


def test_unknown_keyword_raises_naming_it(atom_text, tmp_path):
    text = atom_text.replace("\nEND", "\n OHM      1  0    1.0e-16  1.0e-16\nEND")
    assert text != atom_text
    with pytest.raises(NotImplementedError, match="OHM"):
        AtomFile.read_RH_atom_(_write(tmp_path, text))


@pytest.mark.parametrize(
    ("old", "new", "name"),
    [
        ("HYDROGENIC       22.794", "EXPLICIT        22.794", "EXPLICIT"),
        ("VOIGT    100", "PRD_VOIGT 100", "PRD_VOIGT"),
        ("UNSOLD   1.000  0.000  1.000  0.000  4.70E+08", "BARKLEM  1.000  0.000  1.000  0.000  4.70E+08", "BARKLEM"),
        ("ASYMM  15.0", "RANDOM 15.0", "RANDOM"),
        ("    6       10       5        0", "    6       10       5        1", "fixed transitions"),
    ],
    ids=["continuum", "line-type", "vdW", "symmetry", "fixed"],
)
def test_unsupported_features_raise_naming_them(atom_text, tmp_path, old, new, name):
    assert old in atom_text
    with pytest.raises(NotImplementedError, match=name):
        AtomFile.read_RH_atom_(_write(tmp_path, atom_text.replace(old, new, 1)))


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        # RH accepts a 16th column (g_Lande_eff)
        ("4.70E+08  1.0E+00", "4.70E+08  1.0E+00  0.0", "columns"),
        # RH accepts level rows without levelNo
        ("2SE          '    0       0", "2SE          '    0", "levelNo"),
        # RH reads lamb_min <= 0 as "no limit"; here it is an error
        ("HYDROGENIC       22.794", "HYDROGENIC        0.000", "lamb_min"),
    ],
    ids=["16th-column", "no-levelNo", "lamb_min-zero"],
)
def test_input_that_RH_accepts_but_the_parser_does_not(atom_text, tmp_path, old, new, match):
    assert old in atom_text
    with pytest.raises(ValueError, match=match):
        AtomFile.read_RH_atom_(_write(tmp_path, atom_text.replace(old, new, 1)))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("  4   3  1.036E+00", "  4  -1  1.036E+00"),  # line, lower level negative: numpy would alias level 5
        ("  4   3  1.036E+00", "  6   3  1.036E+00"),  # line, upper level == nLevel
        ("  5   4      3.699E-21", "  5  -2      3.699E-21"),  # continuum, lower level negative
        ("  5   4      3.699E-21", " 17   4      3.699E-21"),  # continuum, upper level beyond the list
    ],
    ids=["line-negative", "line-too-large", "cont-negative", "cont-too-large"],
)
def test_level_index_outside_the_level_list_raises(atom_text, tmp_path, old, new):
    assert old in atom_text
    with pytest.raises(ValueError, match="outside"):
        AtomFile.read_RH_atom_(_write(tmp_path, atom_text.replace(old, new, 1)))


def test_file_without_end_raises(atom_text, tmp_path):
    with pytest.raises(ValueError, match="END"):
        AtomFile.read_RH_atom_(_write(tmp_path, atom_text.replace("END", "")))
