# -------------------------------------------------------------------------------
# configurations in spectra
# -------------------------------------------------------------------------------


import os as _os
from pathlib import Path

from numba.core import config as nb_config

from numba import set_num_threads as nb_set_num_threads

from .Types import T_BOOL, T_INT, T_STR

_ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent


# -------------------------------------------------------------------------------
# for numba configuratoin
# -------------------------------------------------------------------------------


_JIT_ENV_VAR: T_STR = "SPECTRA_JIT"


def _jit_from_env_(value: str | None) -> bool:
    r"""Parse the SPECTRA_JIT environment variable.

    Unrecognised values raise instead of falling back to off: a typo such as
    ``SPECTRA_JIT=true`` would otherwise silently run the whole suite without JIT
    and make a JIT-on test run look green.
    """
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{_JIT_ENV_VAR} must be '0' or '1' (or unset), got {value!r}")


_IS_JIT: T_BOOL = _jit_from_env_(_os.environ.get(_JIT_ENV_VAR))

_IS_NOGIL: T_BOOL = True

_IS_PARALLEL: T_BOOL = False

_IS_CACHE: T_BOOL = True

_IS_FASTMATH: T_BOOL = False

_IS_PARALLEL: T_BOOL = False

_VEC_TARGET: T_STR = "cpu"

"""whether to turn on the JIT compilation in all *.py files,
since sphinx does no understand numba thus can not generate documentation for numba jitted functions.

controlled by the environment variable SPECTRA_JIT, read once at import time:

    unset or 0 : JIT off (default); use this before generating documentation using sphinx

    1          : JIT on; use this before simulation and before pushing to github

any other value raises ValueError.
"""


def _SET_NUMBA_THREAD_(threading_layer: T_STR = "threadsafe", n_thread: T_INT = 2) -> None:
    r""" """

    nb_config.THREADING_LAYER = threading_layer
    """ set to thread safe library : tbb, omp, workqueue, default : workqueue. tbb if possible """

    nb_set_num_threads(n_thread)
    """ limiting the number of threads """

    return None
