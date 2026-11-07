"""Tests of the array builders `Util.RH.AtomBuild` (fast, no JIT needed).

Each check is exercised on the function where it lives, from a parsed file.
The `Atom` built from the same functions is tested in `test.RH_Atom.py`.
"""

from pathlib import Path

import pytest

from spectra.Util.RH import AtomBuild, AtomFile

RH_ATOM = Path(__file__).resolve().parents[1] / "data" / "rh_falc" / "H_6_CRD.atom"


@pytest.fixture(scope="module")
def atom_text():
    return RH_ATOM.read_text()


def _parse(tmp_path: Path, text: str):
    p = tmp_path / "x.atom"
    p.write_text(text)
    return AtomFile.read_RH_atom_(p)


def _level_line_cont(tmp_path: Path, text: str):
    rh = _parse(tmp_path, text)
    Level, _ = AtomBuild.make_Level_(rh, 1)
    Line, _ = AtomBuild.make_Line_(rh, Level)
    Cont, _ = AtomBuild.make_Cont_(rh, Level)
    return rh, Level, Line, Cont


def _make_CE(rh, Level, Line):
    # the labels of the info tables are not read by `make_CECI_`, only passed through
    info = tuple((lab, "-", "-") for lab in rh.level_label)
    ctj = tuple((info[i], info[j]) for i, j in zip(Line["idxI"], Line["idxJ"], strict=True))
    return AtomBuild.make_CECI_("CE", Line, Level, info, ctj, rh.Te_table, rh.CE_idx, rh.CE_coe)


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "n"), [("H I 1S 2SE", 1), ("H I 12K 2KE", 12), ("Ca II 3D 2DE", 3)], ids=["1S", "12K", "CaII"]
)
def test_principal_n_from_label(label, n):
    assert AtomBuild.principal_n_from_label_(label) == n


def test_principal_n_without_a_number_raises():
    with pytest.raises(ValueError, match="principal quantum number"):
        AtomBuild.principal_n_from_label_("H I S 2SE")


def test_decreasing_stage_raises(atom_text, tmp_path):
    # swap the stages of the first and the last level: the levels of a stage must be contiguous
    text = atom_text.replace("2SE          '    0       0", "2SE          '    1       0").replace(
        "2HO          '    1       5", "2HO          '    0       5"
    )
    with pytest.raises(ValueError, match="stages"):
        AtomBuild.make_Level_(_parse(tmp_path, text), 1)


# ---------------------------------------------------------------------------
# lines and continua
# ---------------------------------------------------------------------------


def test_line_with_the_pair_in_the_wrong_order_raises(atom_text, tmp_path):
    # RH accepts j and i in either order; the builder requires (upper, lower)
    text = atom_text.replace("  1   0  4.162E-01", "  0   1  4.162E-01", 1)
    rh = _parse(tmp_path, text)
    Level, _ = AtomBuild.make_Level_(rh, 1)
    with pytest.raises(ValueError, match="upward"):
        AtomBuild.make_Line_(rh, Level)


def test_continuum_without_positive_ionization_energy_raises(atom_text, tmp_path):
    # level 4 (5G, E = 105290.508) moved above the proton: edge wavelength would be negative
    text = atom_text.replace("105290.508", "119290.508", 1)
    rh = _parse(tmp_path, text)
    Level, _ = AtomBuild.make_Level_(rh, 1)
    with pytest.raises(ValueError, match="non-positive ionization energy"):
        AtomBuild.make_Cont_(rh, Level)


def test_continuum_with_equal_energies_raises(atom_text, tmp_path):
    # zero ionization energy: infinite edge wavelength
    text = atom_text.replace("105290.508", "109677.617", 1)
    rh = _parse(tmp_path, text)
    Level, _ = AtomBuild.make_Level_(rh, 1)
    with pytest.raises(ValueError, match="non-positive ionization energy"):
        AtomBuild.make_Cont_(rh, Level)


def test_repeated_continuum_raises(atom_text, tmp_path):
    # the CI table is keyed by the (lower, upper) pair; a second row for it would fill nothing
    text = atom_text.replace("    6       10       5        0", "    6       10       6        0", 1).replace(
        "# Fixed Transitions", "  5   4      3.699E-21       20       HYDROGENIC      569.852\n# Fixed Transitions", 1
    )
    rh = _parse(tmp_path, text)
    Level, _ = AtomBuild.make_Level_(rh, 1)
    with pytest.raises(ValueError, match="listed twice"):
        AtomBuild.make_Cont_(rh, Level)


# ---------------------------------------------------------------------------
# collisions
# ---------------------------------------------------------------------------


def test_collision_row_without_a_listed_transition_raises(atom_text, tmp_path):
    text = atom_text.replace(" CE      4  3", " CE      5  3", 1)
    rh, Level, Line, _ = _level_line_cont(tmp_path, text)
    with pytest.raises(ValueError, match="CE row"):
        _make_CE(rh, Level, Line)


def test_repeated_CE_row_raises(atom_text, tmp_path):
    # RH sums repeated CE rows; the builder refuses them
    text = atom_text.replace(
        " CE      1  0",
        " CE      1  0    1.0e-16  1.0e-16  1.0e-16  1.0e-16  1.0e-16  1.0e-16\n CE      1  0",
        1,
    )
    rh, Level, Line, _ = _level_line_cont(tmp_path, text)
    with pytest.raises(ValueError, match="twice"):
        _make_CE(rh, Level, Line)
