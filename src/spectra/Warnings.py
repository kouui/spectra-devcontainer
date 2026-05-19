# -------------------------------------------------------------------------------
# warning function for normal/numba mode
# -------------------------------------------------------------------------------

import warnings as WARNINGS

from numba import njit as _nb_njit
from numba import objmode

from . import Configurations as CFG
from .Types import *

WARNINGS.simplefilter("once", UserWarning)

_NB_NJIT_KWGS: T_DICT[T_STR, T_BOOL] = {
    "cache": CFG._IS_CACHE,
    "nogil": CFG._IS_NOGIL,
    "fastmath": CFG._IS_FASTMATH,
    "parallel": False,
}


def WARN_(text: T_STR):

    if CFG._IS_JIT:
        with objmode():
            WARNINGS.warn(text, stacklevel=2)
    else:
        WARNINGS.warn(text, stacklevel=2)


if CFG._IS_JIT:
    WARN_ = _nb_njit(**_NB_NJIT_KWGS)(WARN_)
