# -------------------------------------------------------------------------------
# map the raw values of an RH atom file onto the arrays of an Atom
# -------------------------------------------------------------------------------
"""Turn an `AtomFile.RH_Atom_File` into the struct arrays and tables of an `Atom`.

An RH file is SI (energies in cm^-1); an Atom is CGS. Every value is converted
once, here, and stored in the unit the field comment of the dtype names. The
functions return arrays and tuples only: assembling the `Atom` itself is
`Struct.Atom.init_Atom_from_RH_`, and this module must not import `Struct`
(`Struct` imports `Util`).
"""

import re as _re

import numpy as _numpy

from ...ImportAll import *
from ..AtomUtils import AtomIO as _AtomIO
from . import AtomFile as _AtomFile

# f -> Aji factor [m^2 s^-1] of the frozen MALI v2 FALC case. RH's own expression
# (readatom.c: 2 pi (e / eps0) (e / m_e) / c with the values of its constant.h)
# gives 6.670253e-5, which is 9.2e-6 larger in relative terms. The value below is
# kept so that Aji matches that case; switching to RH's value belongs to the
# RH-parity work, which measures its effect on the residual against RH.
_RH_F_TO_AJI_SI = 6.670192e-5


def principal_n_from_label_(label: T_STR) -> T_INT:
    """Principal quantum number from an RH level label, e.g. ``'H I 1S 2SE'`` -> 1.

    The configuration is the third word and starts with n. RH's ``levelNo`` is
    only a running index.
    """
    words = label.split()
    m = _re.match(r"\d+", words[2]) if len(words) > 2 else None
    if m is None:
        raise ValueError(f"cannot read the principal quantum number from RH level label {label!r}")
    return int(m.group())


def full_profile_points_(Nlambda: T_INT, symm: T_STR) -> T_INT:
    """Points of the full (two-sided) line profile mesh for an RH ``Nlambda``.

    For SYMM RH counts one half of the profile; ASYMM counts the whole. The
    mesh builder needs an odd count, so an even one gets a point added.
    """
    n = Nlambda if symm == "ASYMM" else 2 * Nlambda - 1
    return n + 1 if n % 2 == 0 else n


def make_Level_(rh: _AtomFile.RH_Atom_File, Z: T_INT) -> T_TUPLE[T_ARRAY, T_CTJ_TABLE]:
    """Level struct array and the (label, "-", "-") info table of an RH atom.

    A level of stage Z + 1 (1-based) is the bare nucleus and gets ``n = 0``, as
    in `init_theoretical_hydrogen_atom_`: it has no principal quantum number,
    and the RH label of such a level (``'H I 6H 2HO'`` for the proton) is only
    a name. Every other level takes n from its label.
    """
    nLevel = rh.level_E_cm.size
    stage = rh.level_stage + 1
    if _numpy.any(_numpy.diff(stage) < 0):
        raise ValueError("RH atom file: level stages must not decrease along the level list")
    g_int = _numpy.rint(rh.level_g).astype(DT_NB_INT)
    if not _numpy.array_equal(g_int, rh.level_g):
        raise ValueError("RH atom file: statistical weights must be integers")

    Level = _numpy.zeros(nLevel, dtype=_AtomIO.LEVEL_DTYPE)
    Level["erg"][:] = rh.level_E_cm * CST.h_ * CST.c_
    Level["g"][:] = g_int
    Level["stage"][:] = stage
    Level["n"][:] = [
        0 if st == Z + 1 else principal_n_from_label_(lab) for st, lab in zip(stage, rh.level_label, strict=True)
    ]
    for s in _numpy.unique(stage):
        members = _numpy.flatnonzero(stage == s)
        Level["isGround"][members[_numpy.argmin(Level["erg"][members])]] = True

    # one tuple per level, as the (configuration, term, J) of the other readers;
    # an RH label is not split, so the whole label is the configuration
    Level_info_table: T_CTJ_TABLE = tuple((lab, "-", "-") for lab in rh.level_label)
    return Level, Level_info_table


def make_Line_(rh: _AtomFile.RH_Atom_File, Level: T_ARRAY) -> T_TUPLE[T_ARRAY, T_IDX_PAIR_TABLE]:
    """Line struct array of the lines RH lists, in file order, and their (lower, upper) pairs."""
    nLine = rh.line_i.size
    idx_table: T_IDX_PAIR_TABLE = tuple(zip(rh.line_i.tolist(), rh.line_j.tolist(), strict=True))
    for i, j in idx_table:
        if Level["stage"][i] != Level["stage"][j] or not rh.level_E_cm[j] > rh.level_E_cm[i]:
            raise ValueError(f"RH atom file: line ({j}, {i}) is not an upward transition within one stage")
    if len(set(idx_table)) != nLine:
        raise ValueError("RH atom file: a line is listed twice")

    Line = _numpy.zeros(nLine, dtype=_AtomIO.LINE_DTYPE)
    Line["idxI"][:] = rh.line_i
    Line["idxJ"][:] = rh.line_j
    for k, (i, j) in enumerate(idx_table):
        wl_cm = 1.0 / (rh.level_E_cm[j] - rh.level_E_cm[i])
        Line["AJI"][k] = rh.line_f[k] * (_RH_F_TO_AJI_SI / (wl_cm / CST.m2cm_) ** 2) * (rh.level_g[i] / rh.level_g[j])
    _AtomIO.finalize_Line_(Line, Level)
    return Line, idx_table


def make_Cont_(rh: _AtomFile.RH_Atom_File, Level: T_ARRAY) -> T_TUPLE[T_ARRAY, T_IDX_PAIR_TABLE]:
    """Cont struct array of the continua RH lists, in file order, and their (lower, upper) pairs.

    Checked before the array is built: a continuum ends on the ground level of
    the next stage, lies above its lower level (a non-positive ionization
    energy gives an infinite or negative edge wavelength, which would also slip
    past the ``lamb_min`` check of the mesh), and is listed once (the CI table
    is keyed by the pair).
    """
    idx_table: T_IDX_PAIR_TABLE = tuple(zip(rh.cont_i.tolist(), rh.cont_j.tolist(), strict=True))
    for i, j in idx_table:
        if Level["stage"][j] != Level["stage"][i] + 1 or not Level["isGround"][j]:
            raise ValueError(f"RH atom file: continuum ({j}, {i}) must end on the ground level of the next stage")
        if not Level["erg"][j] > Level["erg"][i]:
            raise ValueError(f"RH atom file: continuum ({j}, {i}) has a non-positive ionization energy")
    if len(set(idx_table)) != len(idx_table):
        raise ValueError("RH atom file: a continuum is listed twice")
    return _AtomIO.make_Atom_Cont_(len(idx_table), idx_table, Level), idx_table


def make_RL_Coe_(rh: _AtomFile.RH_Atom_File, Line: T_ARRAY) -> T_ARRAY:
    """One RL row per line, in the order of ``Line``, with the RH broadening in CGS."""
    nLine = Line.shape[0]
    Coe = _numpy.zeros(nLine, dtype=_AtomIO.RL_DTYPE)
    Coe["idxI"][:] = Line["idxI"]
    Coe["idxJ"][:] = Line["idxJ"]
    Coe["lineIndex"][:] = _numpy.arange(nLine)
    Coe["ProfileType"][:] = int(E_ABSORPTION_PROFILE_TYPE.VOIGT)  # the parser accepts VOIGT only
    Coe["qcore"][:] = rh.line_qcore
    Coe["qwing"][:] = rh.line_qwing
    Coe["nLambda"][:] = [full_profile_points_(int(n), s) for n, s in zip(rh.line_Nlambda, rh.line_symm, strict=True)]
    Coe["Grad"][:] = rh.line_Grad  # s^-1 in both
    Coe["vdW_type"][:] = [int(E_VDW_TYPE[t]) for t in rh.line_vdW_type]  # the parser accepts UNSOLD only
    Coe["cvdW"][:, :] = rh.line_cvdW  # dimensionless multipliers
    # A negative cStark carries Gamma / Ne in m^3 s^-1 and is converted to cm^3 s^-1
    # (the sign stays the flag); a positive one is a dimensionless multiplier.
    Coe["cStark"][:] = _numpy.where(rh.line_cStark < 0.0, rh.line_cStark * CST.m2cm_**3, rh.line_cStark)
    return Coe


def make_CECI_(
    tran_type: T_STR,
    Tran: T_ARRAY,
    Level: T_ARRAY,
    Level_info_table: T_CTJ_TABLE,
    Tran_ctj_table: T_CTJ_PAIR_TABLE,
    Te_table: T_ARRAY,
    idx_in_file: T_ARRAY,
    coe_in_file: T_ARRAY,
) -> T_TUPLE[
    T_ARRAY,
    T_ARRAY,
    T_ARRAY,
    T_E_COLLISIONAL_TRANSITION,
    T_E_COLLISIONAL_TRANSITION_SOURCE,
    T_E_COLLISIONAL_TRANSITION_FORMULA,
]:
    """CE or CI table of an RH atom file, in the layout of `AtomIO.make_Atom_CECI_`.

    Parameters
    ----------
    tran_type : T_STR
        "CE" (transitions are the lines) or "CI" (the continua)
    Te_table : T_ARRAY
        the file's TEMP grid, [K]
    idx_in_file : T_ARRAY
        (nRow, 2) level pairs as written in the file: (upper, lower) for CE,
        (lower, upper) for CI
    coe_in_file : T_ARRAY
        (nRow, nTemp) the file's coefficients, SI (m^3 ...)

    Returns
    -------
    Te_table, Omega_table, Coe, type, source, formula
        ``Omega_table`` is (nTran, nTemp) with the coefficients converted ONCE to
        CGS (x1e6); a transition the file has no row for stays 0. Row k belongs to
        ``Tran[k]``, so ``Coe["idxI"]`` is always the lower and ``Coe["idxJ"]``
        the upper level, whichever order the file uses. The formula is
        ``RH_CE`` / ``RH_CI`` (see `E_COLLISIONAL_TRANSITION_FORMULA`).
    """
    nTran = Tran.shape[0]
    _, _, Coe, _tt, _ts, _ = _AtomIO.make_Atom_CECI_(
        None, tran_type, nTran, Tran, Level, Level_info_table, Tran_ctj_table
    )
    row_of = {(int(Tran["idxI"][k]), int(Tran["idxJ"][k])): k for k in range(nTran)}

    Omega_table = _numpy.zeros((nTran, Te_table.size), dtype=DT_NB_FLOAT)
    filled = set()
    for (a, b), coe in zip(idx_in_file.tolist(), coe_in_file, strict=True):
        key = (b, a) if tran_type == "CE" else (a, b)
        if key not in row_of:
            raise ValueError(f"RH atom file: {tran_type} row ({a}, {b}) matches no listed transition")
        if key in filled:
            raise ValueError(f"RH atom file: {tran_type} row ({a}, {b}) is listed twice")
        filled.add(key)
        Omega_table[row_of[key], :] = coe * CST.m2cm_**3  # RH tabulates CE/CI with SI volumes (m^3); stored in cm^3

    formula = E_COLLISIONAL_TRANSITION_FORMULA.RH_CE if tran_type == "CE" else E_COLLISIONAL_TRANSITION_FORMULA.RH_CI
    return Te_table.copy(), Omega_table, Coe, _tt, _ts, formula
