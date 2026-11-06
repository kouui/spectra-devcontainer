# NUMBA_DISABLE_JIT=1 is the supported way to run the code uncompiled (debugging,
# or patching physical constants at runtime). Compiled code freezes module globals
# at compile time, so these checks must run in a fresh interpreter with a cold cache.

import os
import subprocess
import sys

import pytest

_PATCHED_CE_RATE = 9.024875555695096e-17

_PROBE = """
import sys
from spectra import Constants as CST
from spectra.Atomic import Hydrogen

order = sys.argv[1]
args = (1, 2, 6000.0)
if order == "call_then_patch":
    Hydrogen.CE_rate_coe_(*args)
CST.E_Rydberg_H_ *= 1.01
print(repr(float(Hydrogen.CE_rate_coe_(*args))))
"""


@pytest.mark.parametrize("order", ["call_then_patch", "patch_first", "patch_first_warm_cache"])
def test_constant_patch_reaches_vectorized_kernel(order, tmp_path):
    env = {**os.environ, "NUMBA_DISABLE_JIT": "1", "NUMBA_CACHE_DIR": str(tmp_path)}
    if order == "patch_first_warm_cache":
        # A process with JIT on is the only one that writes the on-disk cache; the
        # patched pure-python process must not pick up the constant frozen in it.
        seed_env = {k: v for k, v in env.items() if k != "NUMBA_DISABLE_JIT"}
        subprocess.run([sys.executable, "-c", _PROBE, "call_then_patch"], env=seed_env, capture_output=True, check=True)
        assert list(tmp_path.rglob("*.nbi")), "seed process wrote no numba cache index"
        assert list(tmp_path.rglob("*.nbc")), "seed process wrote no numba cache data"
        order = "patch_first"
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, order],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    # pytest.approx keeps a default absolute tolerance of 1e-12, which would swallow
    # a value near 1e-16; compare relatively by hand.
    value = float(out.stdout.strip().splitlines()[-1])
    assert abs(value / _PATCHED_CE_RATE - 1.0) < 1e-9


def test_warn_is_callable_without_jit(tmp_path):
    code = (
        "import warnings\n"
        "from spectra.Warnings import WARN_\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    WARN_('probe')\n"
        "print(len(w), w[0].message)\n"
    )
    env = {**os.environ, "NUMBA_DISABLE_JIT": "1", "NUMBA_CACHE_DIR": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "1 probe"


_PLANCK_PROBE = """
import json
import numpy as np
from spectra.Atomic import LTELib

# the first wavelength is inside planck_cm_'s cutoff branch, which returns an
# integer 0 -- numpy.vectorize would infer an integer output type from it
W = np.array([1e-8, 1.0])
cutoff_first = LTELib.planck_cm_(W, 6000.0)
reversed_ = LTELib.planck_cm_(W[::-1], 6000.0)
empty = LTELib.planck_cm_(np.array([]), 6000.0)
print(json.dumps({
    "cutoff_first": [cutoff_first.dtype.name, cutoff_first.tolist()],
    "reversed": [reversed_.dtype.name, reversed_.tolist()],
    "empty": [empty.dtype.name, empty.shape[0]],
}))
"""


def test_nb_vec_fallback_output_is_float(tmp_path):
    import json

    env = {**os.environ, "NUMBA_DISABLE_JIT": "1", "NUMBA_CACHE_DIR": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", _PLANCK_PROBE], env=env, capture_output=True, text=True, check=True)
    res = json.loads(out.stdout.strip().splitlines()[-1])

    assert res["cutoff_first"][0] == "float64"
    assert res["reversed"][0] == "float64"
    assert res["empty"] == ["float64", 0]
    # same values regardless of which element decides the output type
    assert res["cutoff_first"][1][0] == 0.0
    assert res["reversed"][1][1] == 0.0
    value = res["cutoff_first"][1][1]
    assert value > 0.0
    assert abs(res["reversed"][1][0] / value - 1.0) < 1e-12


_IS_ODD_PROBE = """
import json
import numpy as np
from spectra.Math import BasicM

scalar = BasicM.is_odd_(3)
arr = BasicM.is_odd_(np.arange(5))
empty = BasicM.is_odd_(np.array([], dtype=np.int64))
print(json.dumps({
    "scalar": [np.asarray(scalar).dtype.name, int(scalar)],
    "array": [arr.dtype.name, arr.tolist()],
    "empty": [empty.dtype.name, empty.shape[0]],
}))
"""


def _check_is_odd_(res):
    # an integer result in both modes: the old fallback widened it to float
    assert res["scalar"] == ["int64", 1]
    assert res["array"] == ["int64", [0, 1, 0, 1, 0]]
    assert res["empty"] == ["int64", 0]


def test_is_odd_returns_integer_when_compiled(tmp_path):
    import json

    env = {k: v for k, v in os.environ.items() if k != "NUMBA_DISABLE_JIT"}
    env["NUMBA_CACHE_DIR"] = str(tmp_path)
    out = subprocess.run([sys.executable, "-c", _IS_ODD_PROBE], env=env, capture_output=True, text=True, check=True)
    _check_is_odd_(json.loads(out.stdout.strip().splitlines()[-1]))


def test_is_odd_returns_integer_without_jit(tmp_path):
    import json

    env = {**os.environ, "NUMBA_DISABLE_JIT": "1", "NUMBA_CACHE_DIR": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", _IS_ODD_PROBE], env=env, capture_output=True, text=True, check=True)
    _check_is_odd_(json.loads(out.stdout.strip().splitlines()[-1]))


_DISAGREEING_SIGNATURES_PROBE = """
from spectra.ImportAll import *

try:
    @nb_vec([T_NB_FLOAT(T_NB_FLOAT), T_NB_INT(T_NB_INT)], **NB_VEC_KWGS)
    def f_(x):
        return x
except ValueError as e:
    print("ValueError", e)
"""


def test_nb_vec_fallback_rejects_signatures_with_different_return_types(tmp_path):
    # numpy.vectorize takes one output type per output; picking either would
    # silently diverge from the compiled loops
    env = {**os.environ, "NUMBA_DISABLE_JIT": "1", "NUMBA_CACHE_DIR": str(tmp_path)}
    out = subprocess.run(
        [sys.executable, "-c", _DISAGREEING_SIGNATURES_PROBE], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.startswith("ValueError")
