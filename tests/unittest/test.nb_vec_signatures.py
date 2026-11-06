# Every nb_vec kernel must be compiled eagerly from explicit signatures. A lazy
# numba.vectorize compiles one loop per input dtype on first call, so the output
# dtype depends on the caller and a str argument is typed as a unicode array.

import re

from numba.core import config as _nb_config
from numba.np.ufunc.dufunc import DUFunc
import numpy as np
import pytest

from numba import vectorize
from spectra import Configurations as CFG
from spectra.Atomic import BasicP, Collision, ContinuumOpacity, Hydrogen, LTELib, emisivity, extinction
from spectra.ImportAll import *
from spectra.Math import BasicM, Special
from spectra.RadiativeTransfer import CloudModel, Profile, Thin

pytestmark = pytest.mark.skipif(
    _nb_config.DISABLE_JIT, reason="NUMBA_DISABLE_JIT replaces the kernels with numpy.vectorize objects"
)

_MODULES = (
    BasicP,
    Collision,
    ContinuumOpacity,
    Hydrogen,
    LTELib,
    emisivity,
    extinction,
    BasicM,
    Special,
    CloudModel,
    Profile,
    Thin,
)


def _kernels_():
    return {f"{m.__name__}.{name}": obj for m in _MODULES for name, obj in vars(m).items() if isinstance(obj, DUFunc)}


def _is_eager_(ufunc) -> bool:
    # `_frozen` is what separates a signature-built ufunc from a lazy one: `types`
    # alone is non-empty for a lazy ufunc once it has been called.
    return bool(ufunc._frozen) and len(ufunc.types) > 0


def test_guard_distinguishes_eager_from_lazy():
    lazy = vectorize(nopython=True)(lambda x: x + 1.0)
    assert not _is_eager_(lazy)
    lazy(1.0)
    assert len(lazy.types) > 0  # type: ignore[attr-defined]
    assert not _is_eager_(lazy)

    eager = vectorize([T_NB_FLOAT(T_NB_FLOAT)], nopython=True)(lambda x: x + 1.0)
    assert _is_eager_(eager)


def test_every_nb_vec_kernel_has_explicit_signatures():
    kernels = _kernels_()
    lazy = sorted(name for name, k in kernels.items() if not _is_eager_(k))
    assert not lazy, f"nb_vec kernels without explicit signatures: {lazy}"


def test_no_nb_vec_kernel_lives_outside_the_checked_modules():
    root = CFG._ROOT_DIR / "src" / "spectra"
    decorated = 0
    for path in root.rglob("*.py"):
        if "Experimental" in path.parts:
            continue
        decorated += len(re.findall(r"^@nb_vec\(", path.read_text(), flags=re.MULTILINE))
    assert decorated == len(_kernels_())


def test_is_odd_has_a_single_integer_loop():
    assert len(BasicM.is_odd_.types) == 1  # type: ignore[attr-defined]
    assert BasicM.is_odd_(np.arange(3)).dtype == np.int64


@pytest.mark.parametrize(
    ("kernel", "args"),
    [
        (Special.E1_, ("a",)),
        (LTELib.planck_cm_, (5000.0e-8, "a")),
        (Hydrogen.einstein_A_coefficient_, (1, "a")),
    ],
)
def test_str_argument_raises_type_error(kernel, args):
    with pytest.raises(TypeError):
        kernel(*args)


def test_float_into_integer_loop_raises_type_error():
    with pytest.raises(TypeError):
        BasicM.is_odd_(3.0)
    with pytest.raises(TypeError):
        Hydrogen.einstein_A_coefficient_(1.0, 2.0)


def test_integer_input_into_float_loop_is_accepted():
    assert Special.E1_(np.array([1, 2])).dtype == np.float64
    assert Special.E1_(1).dtype == np.float64


def test_output_dtype_does_not_depend_on_input_dtype():
    first_cutoff = LTELib.planck_cm_(np.array([1e-8, 1.0]), 6000.0)
    assert first_cutoff.dtype == np.float64
    assert LTELib.planck_cm_(np.array([1, 2]), 6000).dtype == np.float64
    assert LTELib.planck_cm_(np.array([]), 6000.0).dtype == np.float64


def test_statistical_weights_accept_integer_and_float_arrays():
    Eji = np.array([1.63e-11, 1.94e-11])
    gi_int, gj_int = np.array([2, 2]), np.array([8, 18])
    for kernel, rest in [
        (LTELib.boltzmann_distribution_, (Eji, 7000.0)),
        (LTELib.saha_distribution_, (Eji, 1e11, 7000.0)),
    ]:
        as_int = kernel(gi_int, gj_int, *rest)
        as_float = kernel(gi_int.astype(float), gj_int.astype(float), *rest)
        assert as_int.dtype == as_float.dtype == np.float64
        np.testing.assert_array_equal(as_int, as_float)
    assert len(LTELib.boltzmann_distribution_.types) == 2  # type: ignore[attr-defined]
    assert len(LTELib.saha_distribution_.types) == 2  # type: ignore[attr-defined]
