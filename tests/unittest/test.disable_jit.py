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
        # a cache written by an unpatched process must not be reused by a patched one
        subprocess.run([sys.executable, "-c", _PROBE, "call_then_patch"], env=env, capture_output=True, check=True)
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
