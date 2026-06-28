# -------------------------------------------------------------------------------
# definition of struct for Atom
# -------------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass

from ..ImportAll import *
from ..Util.AtomUtils import AtomIO as _AtomIO
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

    Te_table: T_ARRAY  # 1d
    Omega_table: T_ARRAY  # 2d
    Coe: T_ARRAY  # struct array


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
