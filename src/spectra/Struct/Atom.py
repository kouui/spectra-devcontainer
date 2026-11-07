# -------------------------------------------------------------------------------
# definition of struct for Atom
# -------------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
import re as _re

import numpy as _numpy

from ..ImportAll import *
from ..Util import ElementUtil as _ElementUtil
from ..Util.AtomUtils import AtomIO as _AtomIO
from ..Util.AtomUtils import RHAtomIO as _RHAtomIO
from . import WavelengthMesh as _WavelengthMesh

# -------------------------------------------------------------------------------
# struct
# -------------------------------------------------------------------------------


@_dataclass(**STRUCT_KWGS)
class Radiative_Line:
    nRadiativeLine: T_INT
    Coe: T_ARRAY  # struct array


@_dataclass(**STRUCT_KWGS)
class Collisional_Transition:
    _transition_type: T_E_COLLISIONAL_TRANSITION
    _transition_source: T_E_COLLISIONAL_TRANSITION_SOURCE
    _transition_formula: T_E_COLLISIONAL_TRANSITION_FORMULA

    Te_table: T_ARRAY  # 1d, [K]
    # 2d (nTransition, nTe), row k belongs to the k-th line (CE) / continuum (CI).
    # The content and unit depend on `_transition_formula`: OMEGA holds collision
    # strengths (dimensionless); RH_CE / RH_CI hold RH coefficients already in CGS,
    # [cm^3 s^-1 K^-1/2] (see `E_COLLISIONAL_TRANSITION_FORMULA`; an evaluator must
    # not multiply by 1e6 again).
    Omega_table: T_ARRAY
    Coe: T_ARRAY  # struct array, idxI = lower level, idxJ = upper level


@_dataclass(**STRUCT_KWGS)
class Photo_Ionization:
    alpha_table: T_ARRAY  # 2d,  (2, ?)
    alpha_table_idxs: T_ARRAY  # 2d,  (nCont, 2)
    Coe: T_ARRAY  # struct

    # intensity   : T_ARRAY # 2d   (nCont, nContMesh)

    alpha_interp: T_ARRAY  # 2d   (nCont, nContMesh)
    # interpolated photoionization cross section
    # no matter there is doppler shift, alpha_interp always
    # starts from the cross section at ionization limit


@_dataclass(**STRUCT_KWGS)
class ATOMIC_DATA_SOURCE:
    AJI: T_E_ATOMIC_DATA_SOURCE
    CE: T_E_ATOMIC_DATA_SOURCE
    CI: T_E_ATOMIC_DATA_SOURCE
    PI: T_E_ATOMIC_DATA_SOURCE


@_dataclass(**STRUCT_KWGS)
class CTJ_Table:
    Level: T_CTJ_TABLE
    Line: T_CTJ_PAIR_TABLE
    Cont: T_CTJ_PAIR_TABLE


@_dataclass(**STRUCT_KWGS)
class Index_Table:
    Line: T_IDX_PAIR_TABLE
    Cont: T_IDX_PAIR_TABLE


@_dataclass(**STRUCT_KWGS)
class Atom:
    # Title : T_STR
    # Element : T_STR

    Z: T_INT
    Mass: T_FLOAT
    Abun: T_FLOAT

    nLevel: T_INT  # total levels, bound + continuum
    nBoundLevel: T_INT  # bound levels only (excludes the continuum level)
    nLine: T_INT
    nCont: T_INT
    nTran: T_INT
    nRL: T_INT

    Level: T_ARRAY  # struct array
    Line: T_ARRAY  # struct array
    Cont: T_ARRAY  # struct array

    _has_continuum: T_BOOL

    _atomic_data_source: ATOMIC_DATA_SOURCE
    _atom_type: T_E_ATOM

    _ctj_table: CTJ_Table
    _idx_table: Index_Table

    CE: Collisional_Transition
    CI: Collisional_Transition
    PI: Photo_Ionization
    RL: Radiative_Line

    _wave_mesh: _WavelengthMesh.Wavelength_Mesh


# -------------------------------------------------------------------------------
# init function
# -------------------------------------------------------------------------------


def init_Atom_(conf_path: T_STR, is_hydrogen: T_BOOL = False) -> Atom:
    """given the *.conf file, create the Atom struct

    Parameters
    ----------
    conf_path : T_STR
        path to the *.conf file
    is_hydrogen : T_BOOL, optional
        whether is Hydrogen atom, by default False

    Returns
    -------
    atom : Atom
        the Atom struct (the wavelength mesh is stored on ``atom._wave_mesh``)

    Notes
    -----
    If the data-file paths are needed, obtain them separately via
    ``_AtomIO.read_conf_(conf_path)``.
    """

    # path dict
    # --------------------
    path_dict = _AtomIO.read_conf_(conf_path)
    _atom_type: T_E_ATOM
    if is_hydrogen:
        _atom_type = E_ATOM.HYDROGEN
    else:
        _atom_type = E_ATOM.NORMAL

    # read Level
    # --------------------
    if path_dict["Level"] is None:
        raise ValueError("Lack of .Level file")

    Z, Mass, Abun, nLevel, Level, Level_info_table = _AtomIO.make_Atom_Level_(path_dict["Level"])
    # nTran nLine nCont
    # --------------------
    nLine, nCont, nTran, _has_continuum = _AtomIO.nLine_nCont_nTran_(Level["stage"])
    nBoundLevel = nLevel - 1 if _has_continuum else nLevel
    # if not _has_continuum:
    #    raise ValueError("Currently we don't support Atomic Model without comtinuum.")

    # ctj and idx table
    # --------------------
    Line_idx_table, Line_ctj_table, Cont_idx_table, Cont_ctj_table = _AtomIO.prepare_idx_ctj_mapping_(
        Level_info_table, Level["stage"], Level["isGround"], nLine, nCont
    )

    _ctj_table = CTJ_Table(Level=Level_info_table, Line=Line_ctj_table, Cont=Cont_ctj_table)
    _idx_table = Index_Table(Line=Line_idx_table, Cont=Cont_idx_table)

    # make Cont
    # --------------------
    Cont = _AtomIO.make_Atom_Cont_(nCont, Cont_idx_table, Level)

    # read Aji
    # --------------------
    Line, data_source_Aji = _AtomIO.make_Atom_Line_(path_dict["Aji"], Level, Line_idx_table, Line_ctj_table, _atom_type)
    # read CE
    # --------------------
    Te_table, Omega_table, Coe, _transition_type, _transition_source, _transition_formula = _AtomIO.make_Atom_CECI_(
        path_dict["CEe"], "CE", nLine, Line, Level, Level_info_table, Line_ctj_table
    )
    CE = Collisional_Transition(
        _transition_type=_transition_type,
        _transition_source=_transition_source,
        _transition_formula=_transition_formula,
        Te_table=Te_table,
        Omega_table=Omega_table,
        Coe=Coe,
    )
    data_source_CE: T_E_ATOMIC_DATA_SOURCE
    if Te_table.size == 0:
        data_source_CE = E_ATOMIC_DATA_SOURCE.CALCULATE
    else:
        data_source_CE = E_ATOMIC_DATA_SOURCE.EXPERIMENT

    del Te_table, Omega_table, Coe, _transition_type, _transition_source, _transition_formula
    # read CI
    # --------------------
    Te_table, Omega_table, Coe, _transition_type, _transition_source, _transition_formula = _AtomIO.make_Atom_CECI_(
        path_dict["CIe"], "CI", nCont, Cont, Level, Level_info_table, Cont_ctj_table
    )
    CI = Collisional_Transition(
        _transition_type=_transition_type,
        _transition_source=_transition_source,
        _transition_formula=_transition_formula,
        Te_table=Te_table,
        Omega_table=Omega_table,
        Coe=Coe,
    )
    data_source_CI: T_E_ATOMIC_DATA_SOURCE
    if Te_table.size == 0:
        data_source_CI = E_ATOMIC_DATA_SOURCE.CALCULATE
    else:
        data_source_CI = E_ATOMIC_DATA_SOURCE.EXPERIMENT

    del Te_table, Omega_table, Coe, _transition_type, _transition_source, _transition_formula

    # read radiative line
    # --------------------
    Coe, nRadiativeLine = _AtomIO.make_Atom_RL_(path_dict["RadiativeLine"], Level_info_table, Line_ctj_table)
    RL = Radiative_Line(nRadiativeLine=nRadiativeLine, Coe=Coe)
    nRL = nRadiativeLine
    del Coe, nRadiativeLine

    # make mesh
    # --------------------
    waveMesh = _WavelengthMesh.init_Wave_Mesh_(Cont, Line, RL.Coe)

    # read PI
    # --------------------
    Cont_mesh: T_ARRAY = waveMesh.Cont_mesh
    alpha_table, alpha_table_idxs, Coe, alpha_interp, data_source_PI = _AtomIO.make_Atom_PI_(
        path_dict["PI"], Level, Cont, Cont_mesh, _atom_type, Level_info_table, Cont_ctj_table
    )
    PI = Photo_Ionization(
        alpha_table=alpha_table, alpha_table_idxs=alpha_table_idxs, Coe=Coe, alpha_interp=alpha_interp
    )
    del alpha_table, alpha_table_idxs, Coe, alpha_interp

    # make ATOMIC_DATA_SOURCE
    _atomic_data_source = ATOMIC_DATA_SOURCE(
        AJI=data_source_Aji, CE=data_source_CE, CI=data_source_CI, PI=data_source_PI
    )

    atom = Atom(
        Z=Z,
        Mass=Mass,
        Abun=Abun,
        nLevel=nLevel,
        nBoundLevel=nBoundLevel,
        nLine=nLine,
        nCont=nCont,
        nTran=nTran,
        nRL=nRL,
        Level=Level,
        Line=Line,
        Cont=Cont,
        _has_continuum=_has_continuum,
        _atomic_data_source=_atomic_data_source,
        _atom_type=_atom_type,
        _ctj_table=_ctj_table,
        _idx_table=_idx_table,
        CE=CE,
        CI=CI,
        PI=PI,
        RL=RL,
        _wave_mesh=waveMesh,
    )
    return atom


# RH atom files are SI (energies in cm^-1); an Atom is CGS. Every value is
# converted once, here or in `AtomIO.make_Atom_CECI_RH_`, and stored in the unit
# the field comment names.
# f -> Aji factor [m^2 s^-1] of the frozen MALI v2 FALC case. RH's own expression
# (readatom.c: 2 pi (e / eps0) (e / m_e) / c with the values of its constant.h)
# gives 6.670253e-5, which is 9.2e-6 larger in relative terms. The value below is
# kept so that Aji matches that case; switching to RH's value belongs to the
# RH-parity work, which measures its effect on the residual against RH.
_RH_F_TO_AJI_SI = 6.670192e-5


def _principal_n_from_label_(label: T_STR) -> T_INT:
    """Principal quantum number from an RH level label, e.g. ``'H I 1S 2SE'`` -> 1.

    The configuration is the third word and starts with n. RH's ``levelNo`` is
    only a running index.
    """
    words = label.split()
    m = _re.match(r"\d+", words[2]) if len(words) > 2 else None
    if m is None:
        raise ValueError(f"cannot read the principal quantum number from RH level label {label!r}")
    return int(m.group())


def _full_profile_points_(Nlambda: T_INT, symm: T_STR) -> T_INT:
    """Points of the full (two-sided) line profile mesh for an RH ``Nlambda``.

    For SYMM RH counts one half of the profile; ASYMM counts the whole. The
    mesh builder needs an odd count, so an even one gets a point added.
    """
    n = Nlambda if symm == "ASYMM" else 2 * Nlambda - 1
    return n + 1 if n % 2 == 0 else n


def _rh_make_Level_(rh: _RHAtomIO.RH_Atom_File, Z: T_INT) -> T_TUPLE[T_ARRAY, T_CTJ_TABLE]:
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
        0 if st == Z + 1 else _principal_n_from_label_(lab) for st, lab in zip(stage, rh.level_label, strict=True)
    ]
    for s in _numpy.unique(stage):
        members = _numpy.flatnonzero(stage == s)
        Level["isGround"][members[_numpy.argmin(Level["erg"][members])]] = True

    # one tuple per level, as the (configuration, term, J) of the other readers;
    # an RH label is not split, so the whole label is the configuration
    Level_info_table: T_CTJ_TABLE = tuple((lab, "-", "-") for lab in rh.level_label)
    return Level, Level_info_table


def _rh_make_Line_(rh: _RHAtomIO.RH_Atom_File, Level: T_ARRAY) -> T_TUPLE[T_ARRAY, T_IDX_PAIR_TABLE]:
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


def _rh_make_Cont_(rh: _RHAtomIO.RH_Atom_File, Level: T_ARRAY) -> T_TUPLE[T_ARRAY, T_IDX_PAIR_TABLE]:
    """Cont struct array of the continua RH lists, in file order, and their (lower, upper) pairs."""
    idx_table: T_IDX_PAIR_TABLE = tuple(zip(rh.cont_i.tolist(), rh.cont_j.tolist(), strict=True))
    for i, j in idx_table:
        if Level["stage"][j] != Level["stage"][i] + 1 or not Level["isGround"][j]:
            raise ValueError(f"RH atom file: continuum ({j}, {i}) must end on the ground level of the next stage")
    return _AtomIO.make_Atom_Cont_(len(idx_table), idx_table, Level), idx_table


def _rh_make_RL_Coe_(rh: _RHAtomIO.RH_Atom_File, Line: T_ARRAY) -> T_ARRAY:
    """One RL row per line, in the order of ``Line``, with the RH broadening in CGS."""
    nLine = Line.shape[0]
    Coe = _numpy.zeros(nLine, dtype=_AtomIO.RL_DTYPE)
    Coe["idxI"][:] = Line["idxI"]
    Coe["idxJ"][:] = Line["idxJ"]
    Coe["lineIndex"][:] = _numpy.arange(nLine)
    Coe["ProfileType"][:] = int(E_ABSORPTION_PROFILE_TYPE.VOIGT)  # the reader accepts VOIGT only
    Coe["qcore"][:] = rh.line_qcore
    Coe["qwing"][:] = rh.line_qwing
    Coe["nLambda"][:] = [_full_profile_points_(int(n), s) for n, s in zip(rh.line_Nlambda, rh.line_symm, strict=True)]
    Coe["Grad"][:] = rh.line_Grad  # s^-1 in both
    Coe["vdW_type"][:] = [int(E_VDW_TYPE[t]) for t in rh.line_vdW_type]  # the reader accepts UNSOLD only
    Coe["cvdW"][:, :] = rh.line_cvdW  # dimensionless multipliers
    # A negative cStark carries Gamma / Ne in m^3 s^-1 and is converted to cm^3 s^-1
    # (the sign stays the flag); a positive one is a dimensionless multiplier.
    Coe["cStark"][:] = _numpy.where(rh.line_cStark < 0.0, rh.line_cStark * CST.m2cm_**3, rh.line_cStark)
    return Coe


def init_Atom_from_RH_(path: T_STR | _Path) -> Atom:
    """Create the Atom struct from an RH ``*.atom`` file.

    Parameters
    ----------
    path : T_STR | Path
        path to the RH atom file; see `RHAtomIO` for the supported subset

    Returns
    -------
    atom : Atom
        every value of the file is stored in an existing field, in CGS.

    Notes
    -----
    - Levels: stage = RH stage + 1; ``isGround`` is the lowest level of each
      stage (stages must not decrease along the file, so the levels of one stage
      are contiguous); ``Level["erg"]`` is ``E[cm^-1] * h_ * c_`` with project
      constants, and the file's cm^-1 is recovered as ``erg / (h_ * c_)``;
      ``Level["n"]`` is read from the label, 0 for a bare nucleus (stage Z + 1).
    - Only the lines and continua the file lists exist, in file order (lines
      first in the transition order).
    - ``RL.Coe`` has one row per line: the profile mesh (``nLambda`` is the
      number of points of the full profile, ``qcore``, ``qwing``) and the
      broadening (``Grad``, ``vdW_type``, ``cvdW``, ``cStark``; see the dtype).
      ``vdW_type`` is set (UNSOLD), so every broadening field is literal here:
      ``Grad = 0`` means no radiative damping, not "use ``Line.Gamma``".
    - The continuum mesh ends at the file's ``lamb_min``
      (``_wave_mesh.Cont_Coe["lamb_min"]``, [cm]).
    - ``PI`` holds the project's hydrogenic cross section on that mesh, as for
      a "Hydrogenic" entry of a PI file; the file's ``alpha0`` (cm^2) is
      recorded in ``PI.Coe["alpha0"]`` and does not scale the values. The edge
      value differs from ``alpha0`` by up to 3 % (n = 1) and the Gaunt factor
      differs from RH's, so a solver that must reproduce RH computes its cross
      sections from ``alpha0`` itself.
    - ``CE``/``CI`` hold the file's tables with formula ``RH_CE``/``RH_CI``
      (see `E_COLLISIONAL_TRANSITION_FORMULA`): CGS, already multiplied by 1e6,
      rows in the order of ``Line``/``Cont``, ``idxI`` the lower level. The
      statistical equilibrium solver `SELib` refuses them.
    """
    rh = _RHAtomIO.read_RH_atom_(path)

    sym = _ElementUtil.format_sym_(rh.element)
    Z: T_INT = _ElementUtil.sym_to_z_(sym)

    Level, Level_info_table = _rh_make_Level_(rh, Z)
    nLevel = Level.shape[0]
    Line, Line_idx_table = _rh_make_Line_(rh, Level)
    Cont, Cont_idx_table = _rh_make_Cont_(rh, Level)
    nLine, nCont = Line.shape[0], Cont.shape[0]
    _has_continuum = nCont > 0

    Line_ctj_table: T_CTJ_PAIR_TABLE = tuple((Level_info_table[i], Level_info_table[j]) for i, j in Line_idx_table)
    Cont_ctj_table: T_CTJ_PAIR_TABLE = tuple((Level_info_table[i], Level_info_table[j]) for i, j in Cont_idx_table)
    _ctj_table = CTJ_Table(Level=Level_info_table, Line=Line_ctj_table, Cont=Cont_ctj_table)
    _idx_table = Index_Table(Line=Line_idx_table, Cont=Cont_idx_table)

    def _collisions(tran_type: T_STR, Tran: T_ARRAY, ctj: T_CTJ_PAIR_TABLE, idx: T_ARRAY, coe: T_ARRAY):
        Te_table, Omega_table, Coe, _tt, _ts, _tf = _AtomIO.make_Atom_CECI_RH_(
            tran_type, Tran, Level, Level_info_table, ctj, rh.Te_table, idx, coe
        )
        return Collisional_Transition(
            _transition_type=_tt,
            _transition_source=_ts,
            _transition_formula=_tf,
            Te_table=Te_table,
            Omega_table=Omega_table,
            Coe=Coe,
        )

    CE = _collisions("CE", Line, Line_ctj_table, rh.CE_idx, rh.CE_coe)
    CI = _collisions("CI", Cont, Cont_ctj_table, rh.CI_idx, rh.CI_coe)

    RL = Radiative_Line(nRadiativeLine=nLine, Coe=_rh_make_RL_Coe_(rh, Line))
    waveMesh = _WavelengthMesh.init_Wave_Mesh_(Cont, Line, RL.Coe, rh.cont_lamb_min_nm * CST.nm2cm_)

    alpha_table, alpha_table_idxs, PI_Coe, alpha_interp = _AtomIO.make_Atom_PI_hydrogenic_(
        Level, Cont, waveMesh.Cont_mesh, rh.cont_alpha0_m2 * CST.m2cm_**2
    )
    PI = Photo_Ionization(
        alpha_table=alpha_table, alpha_table_idxs=alpha_table_idxs, Coe=PI_Coe, alpha_interp=alpha_interp
    )

    _atomic_data_source = ATOMIC_DATA_SOURCE(
        AJI=E_ATOMIC_DATA_SOURCE.EXPERIMENT,
        CE=E_ATOMIC_DATA_SOURCE.EXPERIMENT,
        CI=E_ATOMIC_DATA_SOURCE.EXPERIMENT,
        PI=E_ATOMIC_DATA_SOURCE.EXPERIMENT,
    )

    return Atom(
        Z=Z,
        Mass=_ElementUtil.sym_to_mass_(sym),
        Abun=_ElementUtil.sym_to_abun_(sym),
        nLevel=nLevel,
        nBoundLevel=nLevel - 1 if _has_continuum else nLevel,
        nLine=nLine,
        nCont=nCont,
        nTran=nLine + nCont,
        nRL=nLine,
        Level=Level,
        Line=Line,
        Cont=Cont,
        _has_continuum=_has_continuum,
        _atomic_data_source=_atomic_data_source,
        _atom_type=E_ATOM.HYDROGEN if Z == 1 else E_ATOM.NORMAL,
        _ctj_table=_ctj_table,
        _idx_table=_idx_table,
        CE=CE,
        CI=CI,
        PI=PI,
        RL=RL,
        _wave_mesh=waveMesh,
    )


def init_theoretical_hydrogen_atom_(
    nLevel: T_INT = 8,
) -> Atom:
    """Create an `Atom` struct for hydrogen from purely theoretical atomic
    data. No file I/O, no conf parsing, no delegation to `init_Atom_`.

    The Level numpy array and CTJ table are constructed directly from the
    hydrogenic energy formula, then the remaining atomic data (Aji, CE,
    CI, PI, RL) are obtained by calling the existing `AtomIO.make_Atom_*_`
    helpers with `path=None`, which already route hydrogen to the analytic
    formulas in `spectra.Atomic.Hydrogen` and
    `spectra.Function.Hydrogen.DegenerateN`.

    Parameters
    ----------
    nLevel : T_INT, optional
        Total number of levels including the continuum. Must be >= 3.
        Default is 8 (n = 1..7 bound + continuum).

    Returns
    -------
    atom : Atom
        the Atom struct (the wavelength mesh is stored on ``atom._wave_mesh``)
    """
    import numpy as _numpy

    from .. import Constants as _CST
    from ..Util import ElementUtil as _ElementUtil

    if nLevel < 3:
        raise ValueError("`nLevel` must be >= 3 (n=1 bound + >=1 excited + continuum)")

    Z: T_INT = 1
    element: T_STR = "H"
    Mass: T_FLOAT = _ElementUtil.sym_to_mass_(element)
    Abun: T_FLOAT = _ElementUtil.sym_to_abun_(element)

    # build Level numpy array directly (mirrors the Level struct built by
    # AtomIO.make_Atom_Level_, but from the analytic hydrogenic formula)
    _level_dtype = _numpy.dtype(
        [
            ("erg", T_FLOAT),
            ("g", T_INT),
            ("stage", T_INT),
            ("gamma", T_FLOAT),
            ("isGround", T_BOOL),
            ("n", T_INT),
        ]
    )
    Level = _numpy.zeros(nLevel, dtype=_level_dtype)
    _Level_info_list: T_LIST[T_TUPLE[T_STR, T_STR, T_STR]] = []

    Ry = _CST.E_Rydberg_H_  # Rydberg energy unit [erg], proton-mass corrected

    # n=1 ground state (1s 2S 1/2), stage=1
    Level[0]["erg"] = 0.0
    Level[0]["g"] = 2
    Level[0]["stage"] = 1
    Level[0]["n"] = 1
    Level[0]["isGround"] = 1
    _Level_info_list.append(("1s", "2S", "1/2"))

    # n=2..nLevel-1 bound states, stage=1
    for k in range(1, nLevel - 1):
        n = k + 1
        Level[k]["erg"] = Ry - Ry * (1.0 / n**2)
        Level[k]["g"] = 2 * n * n
        Level[k]["stage"] = 1
        Level[k]["n"] = n
        Level[k]["isGround"] = 0
        _Level_info_list.append((f"{n}", "-", "-"))

    # continuum, stage=2
    Level[nLevel - 1]["erg"] = Ry
    Level[nLevel - 1]["g"] = 1
    Level[nLevel - 1]["stage"] = 2
    Level[nLevel - 1]["n"] = 0
    Level[nLevel - 1]["isGround"] = 1
    _Level_info_list.append(("-", "-", "-"))

    Level_info_table: T_TUPLE[T_TUPLE[T_STR, T_STR, T_STR], ...] = tuple(_Level_info_list)

    _atom_type = E_ATOM.HYDROGEN

    # counts and CTJ/index mapping tables
    nLine, nCont, nTran, _has_continuum = _AtomIO.nLine_nCont_nTran_(Level["stage"])
    nBoundLevel = nLevel - 1 if _has_continuum else nLevel
    Line_idx_table, Line_ctj_table, Cont_idx_table, Cont_ctj_table = _AtomIO.prepare_idx_ctj_mapping_(
        Level_info_table, Level["stage"], Level["isGround"], nLine, nCont
    )
    _ctj_table = CTJ_Table(Level=Level_info_table, Line=Line_ctj_table, Cont=Cont_ctj_table)
    _idx_table = Index_Table(Line=Line_idx_table, Cont=Cont_idx_table)

    # Cont
    Cont = _AtomIO.make_Atom_Cont_(nCont, Cont_idx_table, Level)

    # Line (path=None -> Aji computed from Hydrogen.einstein_A_coefficient_)
    Line, data_source_Aji = _AtomIO.make_Atom_Line_(None, Level, Line_idx_table, Line_ctj_table, _atom_type)

    # CE (path=None -> empty tables, CALCULATE source; rates computed in SELib)
    Te_table_CE, Omega_table_CE, Coe_CE, _tt, _ts, _tf = _AtomIO.make_Atom_CECI_(
        None, "CE", nLine, Line, Level, Level_info_table, Line_ctj_table
    )
    CE = Collisional_Transition(
        _transition_type=_tt,
        _transition_source=_ts,
        _transition_formula=_tf,
        Te_table=Te_table_CE,
        Omega_table=Omega_table_CE,
        Coe=Coe_CE,
    )
    data_source_CE = E_ATOMIC_DATA_SOURCE.CALCULATE

    # CI (path=None -> empty tables, CALCULATE source)
    Te_table_CI, Omega_table_CI, Coe_CI, _tt, _ts, _tf = _AtomIO.make_Atom_CECI_(
        None, "CI", nCont, Cont, Level, Level_info_table, Cont_ctj_table
    )
    CI = Collisional_Transition(
        _transition_type=_tt,
        _transition_source=_ts,
        _transition_formula=_tf,
        Te_table=Te_table_CI,
        Omega_table=Omega_table_CI,
        Coe=Coe_CI,
    )
    data_source_CI = E_ATOMIC_DATA_SOURCE.CALCULATE

    # RL (path=None -> empty)
    Coe_RL, nRadiativeLine = _AtomIO.make_Atom_RL_(None, Level_info_table, Line_ctj_table)
    RL = Radiative_Line(nRadiativeLine=nRadiativeLine, Coe=Coe_RL)

    # wavelength mesh
    waveMesh = _WavelengthMesh.init_Wave_Mesh_(Cont, Line, RL.Coe)

    # PI (path=None + HYDROGEN -> cross section from DegenerateN)
    alpha_table, alpha_table_idxs, Coe_PI, alpha_interp, data_source_PI = _AtomIO.make_Atom_PI_(
        None, Level, Cont, waveMesh.Cont_mesh, _atom_type, Level_info_table, Cont_ctj_table
    )
    PI = Photo_Ionization(
        alpha_table=alpha_table,
        alpha_table_idxs=alpha_table_idxs,
        Coe=Coe_PI,
        alpha_interp=alpha_interp,
    )

    _atomic_data_source = ATOMIC_DATA_SOURCE(
        AJI=data_source_Aji,
        CE=data_source_CE,
        CI=data_source_CI,
        PI=data_source_PI,
    )

    atom = Atom(
        Z=Z,
        Mass=Mass,
        Abun=Abun,
        nLevel=nLevel,
        nBoundLevel=nBoundLevel,
        nLine=nLine,
        nCont=nCont,
        nTran=nTran,
        nRL=nRadiativeLine,
        Level=Level,
        Line=Line,
        Cont=Cont,
        _has_continuum=_has_continuum,
        _atomic_data_source=_atomic_data_source,
        _atom_type=_atom_type,
        _ctj_table=_ctj_table,
        _idx_table=_idx_table,
        CE=CE,
        CI=CI,
        PI=PI,
        RL=RL,
        _wave_mesh=waveMesh,
    )
    return atom
