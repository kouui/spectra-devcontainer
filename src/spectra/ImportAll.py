# -------------------------------------------------------------------------------
# Global imports for the sake of convenience in spectra
# -------------------------------------------------------------------------------

# from .ImportExternalModule import *


from numba.core import config as _nb_config
from numba.np.numpy_support import as_dtype as _nb_as_dtype
import numpy as _numpy

from numba import njit as nb_njit
from numba import vectorize as _nb_vectorize

from . import Configurations as CFG
from . import Constants as CST
from .Elements import ELEMENT_ABUN, ELEMENT_IONIZPOTENTIAL, ELEMENT_MASS, ELEMENT_SYMBOL, ELEMENT_Z
from .Enums import *
from .Types import *
from .Warnings import WARN_

np_vec = _numpy.vectorize


def nb_vec(signatures, **kwargs):
    """numba.vectorize with explicit signatures, or numpy.vectorize when NUMBA_DISABLE_JIT is set.

    The signatures are required. Without them numba compiles lazily per input
    dtype, so the output dtype depends on the caller and a str argument is silently
    typed as a unicode array. With them the loops are compiled at import: the
    output dtype is fixed, a compile error surfaces at import, and an input that
    cannot be cast safely to a loop (str, or float into an integer-only kernel)
    raises TypeError. Use only T_NB_FLOAT / T_NB_INT in the signatures. When one
    kernel lists several signatures, put the integer-input ones first: numpy picks
    the first loop that accepts the inputs under safe casting.

    numba.vectorize stays compiled under NUMBA_DISABLE_JIT, which freezes module
    constants into the ufunc (and into its on-disk cache); the numpy path keeps
    them patchable at runtime. The fallback pins `otypes` to the signatures' return
    type: without otypes numpy infers the output type from the first element, so a
    kernel whose first branch returns an integer literal (e.g. planck_cm_ in its
    cutoff) would yield integer output for some inputs and float for others, and
    fail on empty input. numpy.vectorize takes one output type per output, so
    signatures of one kernel that disagree on the return type are rejected.
    """
    if not signatures:
        raise ValueError("nb_vec needs at least one explicit signature")

    if not _nb_config.DISABLE_JIT:
        return _nb_vectorize(signatures, **kwargs)

    return_types = {sig.return_type for sig in signatures}
    if len(return_types) != 1:
        raise ValueError(f"nb_vec signatures disagree on the return type: {sorted(map(str, return_types))}")
    otype = _nb_as_dtype(return_types.pop())

    def _decorator(f):
        return _numpy.vectorize(f, otypes=[otype])

    return _decorator


## : numba TypedDict much slower than numpy struct array
# comment : currently we will not use dictionary as a data struct in spectra
# https://github.com/numba/numba/issues/4364
# from numba.typed import Dict as nb_Dict # type: ignore


# from numba.experimental import jitclass as nb_jitclass # type: ignore
# comment : currently we will not use jitclass as a data struct in spectra

NB_VEC_KWGS: T_DICT[T_STR, T_BOOL | T_STR] = {
    "cache": CFG._IS_CACHE,
    "target": CFG._VEC_TARGET,
    # "nogil"  : CFG._IS_NOGIL,
    "nopython": True,
    "fastmath": CFG._IS_FASTMATH,
}

NB_NJIT_KWGS: T_DICT[T_STR, T_BOOL] = {
    "cache": CFG._IS_CACHE,
    "nogil": CFG._IS_NOGIL,
    "fastmath": CFG._IS_FASTMATH,
    "parallel": False,
}

NB_NJIT_KWGS_PARALLEL: T_DICT[T_STR, T_BOOL] = {
    "cache": CFG._IS_CACHE,
    "nogil": CFG._IS_NOGIL,
    "fastmath": CFG._IS_FASTMATH,
    "parallel": CFG._IS_PARALLEL,
}


# -------------------------------------------------------------------------------
# numpy related functions/class
# -------------------------------------------------------------------------------


NP_VEC_KWGS: T_DICT[T_STR, T_ANY] = {
    "otypes": [T_FLOAT],
    "cache": True,
}

# -------------------------------------------------------------------------------
# struct related functions/class
# -------------------------------------------------------------------------------
STRUCT_KWGS: T_DICT[T_STR, T_BOOL] = {
    "init": True,
    "repr": True,
    "eq": False,
    "order": False,
    "unsafe_hash": False,
    "frozen": True,
}

STRUCT_KWGS_UNFROZEN: T_DICT[T_STR, T_BOOL] = {
    "init": True,
    "repr": True,
    "eq": False,
    "order": False,
    "unsafe_hash": False,
    "frozen": False,
}

# -------------------------------------------------------------------------------
# logging and warning
# -------------------------------------------------------------------------------

__all__ = [
    # Configurations
    "CFG",
    # Constants
    "CST",
    "DT_NB_COMPLEX",
    # Types — numba dtypes
    "DT_NB_FLOAT",
    "DT_NB_INT",
    "ELEMENT_ABUN",
    "ELEMENT_IONIZPOTENTIAL",
    "ELEMENT_MASS",
    # Elements
    "ELEMENT_SYMBOL",
    "ELEMENT_Z",
    "E_ABSORPTION_PROFILE_TYPE",
    "E_ATMOSPHERE_COORDINATE_TYPE",
    "E_ATOM",
    # Enums
    "E_ATOMIC_DATA_SOURCE",
    "E_COLLISIONAL_TRANSITION",
    "E_COLLISIONAL_TRANSITION_FORMULA",
    "E_COLLISIONAL_TRANSITION_SOURCE",
    "E_FEAUTRIER_ORDER",
    "NB_NJIT_KWGS",
    "NB_NJIT_KWGS_PARALLEL",
    "NB_VEC_KWGS",
    "NP_VEC_KWGS",
    "OVERLOAD",
    # struct
    "STRUCT_KWGS",
    "STRUCT_KWGS_UNFROZEN",
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
    # Types — enum literals
    "T_E_ATOMIC_DATA_SOURCE",
    "T_E_COLLISIONAL_TRANSITION",
    "T_E_COLLISIONAL_TRANSITION_FORMULA",
    "T_E_COLLISIONAL_TRANSITION_SOURCE",
    "T_E_FEAUTRIER_ORDER",
    # Types — fundamental
    "T_FLOAT",
    "T_IDX_PAIR_TABLE",
    "T_INT",
    "T_LIST",
    "T_LITERAL",
    "T_NB_FLOAT",
    "T_NB_INT",
    "T_NORETURN",
    "T_SLICE",
    "T_STR",
    "T_TUPLE",
    "T_TYPE",
    "T_VEC_FA",
    "T_VEC_IA",
    # Types — composite
    "T_VEC_IFA",
    # Warnings
    "WARN_",
    # numba
    "nb_njit",
    "nb_vec",
    # numpy
    "np_vec",
]
