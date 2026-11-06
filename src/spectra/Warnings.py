# -------------------------------------------------------------------------------
# warning function callable from both python and numba-compiled code
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


@_nb_njit(**_NB_NJIT_KWGS)
def WARN_(text: T_STR):
    # objmode: python's warnings module is not available in nopython mode, and
    # WARN_ must be callable from other compiled functions as well as from python.
    with objmode():
        WARNINGS.warn(text, stacklevel=2)
