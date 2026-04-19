# -------------------------------------------------------------------------------
# type definition in spectra
# -------------------------------------------------------------------------------
# VERSION
# 0.1.0
#    2021/05/18   u.k.   spectra-re
# -------------------------------------------------------------------------------

from typing import Any as T_ANY
from typing import Literal as T_LITERAL
from typing import NoReturn as T_NORETURN
from typing import overload as OVERLOAD

# reference : https://numpy.org/devdocs/reference/typing.html
from numpy import ndarray as _ndarray

from .Enums import *

# -------------------------------------------------------------------------------
# fundamental types
# -------------------------------------------------------------------------------

T_FLOAT = float
T_INT = int
T_STR = str
T_BOOL = bool
T_LIST = list
T_DICT = dict
T_TUPLE = tuple
T_TYPE = type

T_SLICE = slice

T_ARRAY = _ndarray
del _ndarray


# -------------------------------------------------------------------------------
# advanced types
# -------------------------------------------------------------------------------

type T_VEC_IFA = T_INT | T_FLOAT | T_ARRAY
type T_VEC_FA = T_FLOAT | T_ARRAY
type T_VEC_IA = T_INT | T_ARRAY

# T_UNION_IS  = T_INT | T_SLICE

# types used in struct
type T_CTJ_TABLE = tuple[tuple[T_STR, T_STR, T_STR], ...]
type T_CTJ_PAIR = tuple[tuple[T_STR, T_STR, T_STR], tuple[T_STR, T_STR, T_STR]]
type T_CTJ_PAIR_TABLE = tuple[T_CTJ_PAIR, ...]
type T_IDX_PAIR_TABLE = tuple[tuple[T_INT, T_INT], ...]

# -------------------------------------------------------------------------------
# Enum types
# -------------------------------------------------------------------------------

type T_E_ATOMIC_DATA_SOURCE = T_LITERAL[E_ATOMIC_DATA_SOURCE.EXPERIMENT, E_ATOMIC_DATA_SOURCE.CALCULATE]
type T_E_ATOM = T_LITERAL[E_ATOM.HYDROGEN, E_ATOM.HYDROGEN_LIKE, E_ATOM.NORMAL]
type T_E_COLLISIONAL_TRANSITION = T_LITERAL[E_COLLISIONAL_TRANSITION.EXCITATION, E_COLLISIONAL_TRANSITION.IONIZATION]
type T_E_COLLISIONAL_TRANSITION_SOURCE = T_LITERAL[
    E_COLLISIONAL_TRANSITION_SOURCE.ELECTRON,
    E_COLLISIONAL_TRANSITION_SOURCE.PROTON,
    E_COLLISIONAL_TRANSITION_SOURCE.CHARGE_TRANSFER,
]
type T_E_COLLISIONAL_TRANSITION_FORMULA = T_LITERAL[E_COLLISIONAL_TRANSITION_FORMULA.OMEGA]
type T_E_ABSORPTION_PROFILE_TYPE = T_LITERAL[E_ABSORPTION_PROFILE_TYPE.VOIGT, E_ABSORPTION_PROFILE_TYPE.GAUSSIAN]
type T_E_ATMOSPHERE_COORDINATE_TYPE = T_LITERAL[
    E_ATMOSPHERE_COORDINATE_TYPE.POINT, E_ATMOSPHERE_COORDINATE_TYPE.CARTESIAN
]
# -------------------------------------------------------------------------------
# numba types
# -------------------------------------------------------------------------------

DT_NB_FLOAT = "float64"
DT_NB_INT = "int64"
DT_NB_COMPLEX = "complex128"

__all__ = [
    "DT_NB_COMPLEX",
    "DT_NB_FLOAT",
    "DT_NB_INT",
    "OVERLOAD",
    "T_ANY",
    "T_ARRAY",
    "T_BOOL",
    "T_CTJ_PAIR",
    "T_CTJ_PAIR_TABLE",
    "T_CTJ_TABLE",
    "T_DICT",
    "T_E_ABSORPTION_PROFILE_TYPE",
    "T_E_ATMOSPHERE_COORDINATE_TYPE",
    "T_E_ATOM",
    "T_E_ATOMIC_DATA_SOURCE",
    "T_E_COLLISIONAL_TRANSITION",
    "T_E_COLLISIONAL_TRANSITION_FORMULA",
    "T_E_COLLISIONAL_TRANSITION_SOURCE",
    "T_FLOAT",
    "T_IDX_PAIR_TABLE",
    "T_INT",
    "T_LIST",
    "T_LITERAL",
    "T_NORETURN",
    "T_SLICE",
    "T_STR",
    "T_TUPLE",
    "T_TYPE",
    "T_VEC_FA",
    "T_VEC_IA",
    "T_VEC_IFA",
]
# import numba.types as nb_types     # type: ignore
#
# T_NB_FLOAT1D = nb_types.float64[:]         # type: ignore
# T_NB_FLOAT2D = nb_types.float64[:,:]       # type: ignore
# T_NB_FLOAT3D = nb_types.float64[:,:,:]     # type: ignore
#
# T_NB_INT1D = nb_types.int64[:]             # type: ignore
# T_NB_INT2D = nb_types.int64[:,:]           # type: ignore
# T_NB_INT3D = nb_types.int64[:,:,:]         # type: ignore
#
# T_NB_STR   = nb_types.unicode_type         # type: ignore


## comment : using numba.typed.Dict will dramatically slow down
##           import time
## comment : incase of List:
##             - preprocessing : list --> tuple, these are homogeneous tuples
##             - njit          : list (--> tuple), list initialized in njit function will
##                                be converted to numba.typed.List automatically
##                                might be heterogeneous tuples, so do not convert the list to tuple

## : example of using numba types without conflicting with mypy
#
# from numba.typed import List as nb_List # type: ignore
# from numba.typed import Dict as nb_Dict # type: ignore
# nb_List = list

# a : T_LIST[T_ARRAY] = nb_List()
# a.append( numpy.arange(6).reshape(2,3) )
#
# b : T_DICT[T_STR,T_ARRAY] = nb_Dict.empty(
#     key_type=T_NB_STR,
#     value_type=T_NB_FLOAT1D,
# )
#
# b['posx'] = numpy.asarray([1, 0.5, 2], dtype=DT_NB_FLOAT)
# b['posy'] = numpy.asarray([1.5, 3.5, 2], dtype=DT_NB_FLOAT)
# b['velx'] = numpy.asarray([0.5, 0, 0.7], dtype=DT_NB_FLOAT)
# b['vely'] = numpy.asarray([0.2, -0.2, 0.1], dtype=DT_NB_FLOAT)
