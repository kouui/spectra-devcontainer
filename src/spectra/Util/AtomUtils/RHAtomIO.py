# -------------------------------------------------------------------------------
# reader of the RH (Uitenbroek) atom file format
# -------------------------------------------------------------------------------
"""Read an RH ``*.atom`` file into plain arrays and tuples.

Nothing is converted: energies stay in cm^-1, ``alpha0`` in m^2, ``lamb_min``
in nm, collision coefficients in the SI units of the file. Conversions and the
mapping onto `Struct.Atom.Atom` belong to the builder, so this module imports
no struct.

Only the subset of the format that the repository's RH hydrogen atom uses is
supported, and the import is stricter than RH on it. RH accepts an optional
16th line column (``g_Lande_eff``), level rows without ``levelNo``, line rows
with j and i in either order and several CE rows for one transition (summed).
This reader raises on the first two and on ``lamb_min <= 0``; the builder
(``Struct.Atom.init_Atom_from_RH_``) raises on the last two. Anything else
raises ``NotImplementedError`` naming it: guessing the meaning of an unknown
keyword would give a wrong atom without any error, and each new feature needs a
decision on how its data enters the solver.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as _numpy

from ...ImportAll import *

_N_LINE_COLUMNS = 15
_N_CONT_COLUMNS = 6
_SUPPORTED_LINE_TYPE = "VOIGT"
_SUPPORTED_SYMM = ("SYMM", "ASYMM")
_SUPPORTED_VDW = "UNSOLD"
_SUPPORTED_WAVEDEP = "HYDROGENIC"


class RH_Atom_File(NamedTuple):
    """Raw contents of an RH atom file; row order is the file order.

    Level indices in ``line_*``, ``cont_*`` and ``CE_idx``/``CI_idx`` are
    positions in the level list (0-based), which is how RH numbers them.
    """

    element: T_STR  #: element symbol, e.g. "H"
    # --- levels
    level_E_cm: T_ARRAY  #: (nLevel,) [cm^-1]
    level_g: T_ARRAY  #: (nLevel,) float, as written in the file
    level_label: T_TUPLE[T_STR, ...]  #: label without the quotes, stripped
    level_stage: T_ARRAY  #: (nLevel,) RH stage, 0 = neutral
    level_no: T_ARRAY  #: (nLevel,) the levelNo column
    # --- lines
    line_j: T_ARRAY  #: (nLine,) upper level
    line_i: T_ARRAY  #: (nLine,) lower level
    line_f: T_ARRAY  #: (nLine,) oscillator strength
    line_Nlambda: T_ARRAY  #: (nLine,) raw: half profile for SYMM, full for ASYMM
    line_symm: T_TUPLE[T_STR, ...]  #: "SYMM" or "ASYMM"
    line_qcore: T_ARRAY
    line_qwing: T_ARRAY
    line_vdW_type: T_TUPLE[T_STR, ...]
    line_cvdW: T_ARRAY  #: (nLine, 4)
    line_Grad: T_ARRAY  #: [s^-1]
    line_cStark: T_ARRAY
    # --- continua
    cont_j: T_ARRAY  #: (nCont,) upper level
    cont_i: T_ARRAY  #: (nCont,) lower level
    cont_alpha0_m2: T_ARRAY  #: [m^2]
    cont_Nlambda: T_ARRAY
    cont_wavedep: T_TUPLE[T_STR, ...]
    cont_lamb_min_nm: T_ARRAY  #: [nm]
    # --- collisions
    Te_table: T_ARRAY  #: (nTemp,) [K]
    CE_idx: T_ARRAY  #: (nCE, 2) rows "CE j i": (upper, lower)
    CE_coe: T_ARRAY  #: (nCE, nTemp), SI as in the file
    CI_idx: T_ARRAY  #: (nCI, 2) rows "CI i j": (lower, upper)
    CI_coe: T_ARRAY  #: (nCI, nTemp), SI as in the file


def _data_lines_(path: T_STR | Path) -> T_LIST[T_STR]:
    """Lines that are neither blank nor comments."""
    out = []
    for ln in Path(path).read_text().splitlines():
        if ln.strip() == "" or ln.lstrip().startswith("#"):
            continue
        out.append(ln)
    return out


def _not_implemented_(what: T_STR) -> NotImplementedError:
    return NotImplementedError(f"RH atom file: {what} is not supported")


def read_RH_atom_(path: T_STR | Path) -> RH_Atom_File:  # noqa: C901
    """Parse an RH atom file.

    Parameters
    ----------
    path : T_STR | Path
        path to the ``*.atom`` file

    Returns
    -------
    RH_Atom_File
        raw arrays, see the class for units and index conventions

    Raises
    ------
    NotImplementedError
        a keyword, line type, symmetry, van der Waals approximation,
        continuum wavelength dependence or collision keyword outside the
        supported subset, or fixed transitions
    ValueError
        a malformed or truncated file
    """
    rows = iter(_data_lines_(path))

    def _next(what: T_STR) -> T_STR:
        try:
            return next(rows)
        except StopIteration:
            raise ValueError(f"RH atom file ends before {what}") from None

    # --- header
    element = _next("the atom ID").split()[0]
    head = _next("the Nlevel/Nline/Ncont/Nfixed row").split()
    if len(head) != 4:
        raise ValueError(f"RH atom file: expected 'Nlevel Nline Ncont Nfixed', got {head}")
    nLevel, nLine, nCont, nFixed = (int(x) for x in head)
    if nFixed != 0:
        raise _not_implemented_("fixed transitions (Nfixed > 0)")

    # --- levels:  E  g  'label'  stage  levelNo
    E_cm = _numpy.empty(nLevel, dtype=DT_NB_FLOAT)
    g = _numpy.empty(nLevel, dtype=DT_NB_FLOAT)
    stage = _numpy.empty(nLevel, dtype=DT_NB_INT)
    level_no = _numpy.empty(nLevel, dtype=DT_NB_INT)
    labels: T_LIST[T_STR] = []
    for k in range(nLevel):
        ln = _next("the level list")
        pieces = ln.split("'")
        if len(pieces) != 3:
            raise ValueError(f"RH atom file: level row without a quoted label: {ln!r}")
        num, tail = pieces[0].split(), pieces[2].split()
        if len(num) != 2 or len(tail) != 2:
            raise ValueError(f"RH atom file: level row must be 'E g label stage levelNo': {ln!r}")
        E_cm[k], g[k] = float(num[0]), float(num[1])
        labels.append(pieces[1].strip())
        stage[k], level_no[k] = int(tail[0]), int(tail[1])

    # --- lines:  j i f type Nlambda symm qcore qwing vdWapprx cvdW(4) Grad cStark
    line_j = _numpy.empty(nLine, dtype=DT_NB_INT)
    line_i = _numpy.empty(nLine, dtype=DT_NB_INT)
    line_f = _numpy.empty(nLine, dtype=DT_NB_FLOAT)
    line_N = _numpy.empty(nLine, dtype=DT_NB_INT)
    line_qcore = _numpy.empty(nLine, dtype=DT_NB_FLOAT)
    line_qwing = _numpy.empty(nLine, dtype=DT_NB_FLOAT)
    line_cvdW = _numpy.empty((nLine, 4), dtype=DT_NB_FLOAT)
    line_Grad = _numpy.empty(nLine, dtype=DT_NB_FLOAT)
    line_cStark = _numpy.empty(nLine, dtype=DT_NB_FLOAT)
    symms: T_LIST[T_STR] = []
    vdws: T_LIST[T_STR] = []
    for k in range(nLine):
        p = _next("the line list").split()
        if len(p) != _N_LINE_COLUMNS:
            raise ValueError(f"RH atom file: line row needs {_N_LINE_COLUMNS} columns, got {len(p)}: {p}")
        if p[3].upper() != _SUPPORTED_LINE_TYPE:
            raise _not_implemented_(f"line type {p[3]!r}")
        if p[5].upper() not in _SUPPORTED_SYMM:
            raise _not_implemented_(f"line symmetry {p[5]!r}")
        if p[8].upper() != _SUPPORTED_VDW:
            raise _not_implemented_(f"van der Waals approximation {p[8]!r}")
        line_j[k], line_i[k], line_f[k] = int(p[0]), int(p[1]), float(p[2])
        line_N[k] = int(p[4])
        symms.append(p[5].upper())
        line_qcore[k], line_qwing[k] = float(p[6]), float(p[7])
        vdws.append(p[8].upper())
        line_cvdW[k, :] = [float(x) for x in p[9:13]]
        line_Grad[k], line_cStark[k] = float(p[13]), float(p[14])

    # --- continua:  j i alpha0 Nlambda wavedep lamb_min
    # Lines come first in the file and the builder appends continua after
    # them, so the transition order "lines before continua" holds by
    # construction here.
    cont_j = _numpy.empty(nCont, dtype=DT_NB_INT)
    cont_i = _numpy.empty(nCont, dtype=DT_NB_INT)
    cont_alpha0 = _numpy.empty(nCont, dtype=DT_NB_FLOAT)
    cont_N = _numpy.empty(nCont, dtype=DT_NB_INT)
    cont_lamb_min = _numpy.empty(nCont, dtype=DT_NB_FLOAT)
    wavedeps: T_LIST[T_STR] = []
    for k in range(nCont):
        p = _next("the continuum list").split()
        if len(p) != _N_CONT_COLUMNS:
            raise ValueError(f"RH atom file: continuum row needs {_N_CONT_COLUMNS} columns, got {len(p)}: {p}")
        if p[4].upper() != _SUPPORTED_WAVEDEP:
            raise _not_implemented_(f"continuum wavelength dependence {p[4]!r}")
        cont_j[k], cont_i[k], cont_alpha0[k] = int(p[0]), int(p[1]), float(p[2])
        cont_N[k] = int(p[3])
        wavedeps.append(p[4].upper())
        cont_lamb_min[k] = float(p[5])
        if cont_lamb_min[k] <= 0.0:
            raise ValueError(f"RH atom file: lamb_min must be positive, got {p[5]} in continuum row {p}")

    # --- collisions: TEMP, then CE / CI rows (a row may end with a label), END
    # The file repeats TEMP before the CE and the CI block. One table is
    # stored, so repeated tables must be identical.
    Te_table = _numpy.empty(0, dtype=DT_NB_FLOAT)
    CE_idx: T_LIST[T_LIST[int]] = []
    CI_idx: T_LIST[T_LIST[int]] = []
    CE_vals: T_LIST[T_LIST[float]] = []
    CI_vals: T_LIST[T_LIST[float]] = []
    seen_end = False
    for ln in rows:
        p = ln.split()
        key = p[0].upper()
        if key == "END":
            seen_end = True
            break
        if key == "TEMP":
            this = _numpy.array([float(x) for x in p[2 : 2 + int(p[1])]], dtype=DT_NB_FLOAT)
            if this.size != int(p[1]):
                raise ValueError(f"RH atom file: TEMP announces {p[1]} values, row has {this.size}")
            if Te_table.size > 0 and not _numpy.array_equal(this, Te_table):
                raise _not_implemented_("a second, different TEMP table")
            Te_table = this
        elif key in ("CE", "CI"):
            if Te_table.size == 0:
                raise ValueError(f"RH atom file: {key} row before any TEMP row")
            if len(p) < 3 + Te_table.size:
                raise ValueError(f"RH atom file: {key} row needs {Te_table.size} values: {ln!r}")
            idx = [int(p[1]), int(p[2])]
            vals = [float(x) for x in p[3 : 3 + Te_table.size]]
            if key == "CE":
                CE_idx.append(idx)
                CE_vals.append(vals)
            else:
                CI_idx.append(idx)
                CI_vals.append(vals)
        else:
            raise _not_implemented_(f"keyword {p[0]!r}")
    if not seen_end:
        raise ValueError("RH atom file has no END row")

    return RH_Atom_File(
        element=element,
        level_E_cm=E_cm,
        level_g=g,
        level_label=tuple(labels),
        level_stage=stage,
        level_no=level_no,
        line_j=line_j,
        line_i=line_i,
        line_f=line_f,
        line_Nlambda=line_N,
        line_symm=tuple(symms),
        line_qcore=line_qcore,
        line_qwing=line_qwing,
        line_vdW_type=tuple(vdws),
        line_cvdW=line_cvdW,
        line_Grad=line_Grad,
        line_cStark=line_cStark,
        cont_j=cont_j,
        cont_i=cont_i,
        cont_alpha0_m2=cont_alpha0,
        cont_Nlambda=cont_N,
        cont_wavedep=tuple(wavedeps),
        cont_lamb_min_nm=cont_lamb_min,
        Te_table=Te_table,
        CE_idx=_numpy.array(CE_idx, dtype=DT_NB_INT).reshape(-1, 2),
        CE_coe=_numpy.array(CE_vals, dtype=DT_NB_FLOAT).reshape(-1, Te_table.size),
        CI_idx=_numpy.array(CI_idx, dtype=DT_NB_INT).reshape(-1, 2),
        CI_coe=_numpy.array(CI_vals, dtype=DT_NB_FLOAT).reshape(-1, Te_table.size),
    )
