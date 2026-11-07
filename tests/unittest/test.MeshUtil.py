"""Unit tests of `Util.MeshUtil.make_continuum_mesh_in_range_`.

The pinned values of the other `MeshUtil` functions are in
`tests/regression/test_reg_MeshUtil*.py`; how `init_Wave_Mesh_` uses this one is
tested with the RH atom in `test.RH_Atom.py`.
"""

import numpy as np
import pytest

from spectra.Util import MeshUtil

_W0 = 9.11763e-6  # [cm]
_LAMB_MIN = 2.2794e-6  # [cm]


@pytest.mark.parametrize("nLambda", [5, 41])
def test_mesh_spans_lamb_min_to_the_edge_descending(nLambda):
    mesh = MeshUtil.make_continuum_mesh_in_range_(nLambda, _W0, _LAMB_MIN)
    assert mesh.shape == (nLambda,)
    assert mesh[0] == _W0  # exactly the edge
    assert mesh[-1] == pytest.approx(_LAMB_MIN, rel=1e-14)
    assert np.all(np.diff(mesh) < 0.0)


def test_edge_is_exact_for_any_lamb_min():
    # lm_frac + (1 - lm_frac) is not always exactly 1 in floating point
    for lamb_min in np.linspace(0.05, 0.95, 91) * _W0:
        assert MeshUtil.make_continuum_mesh_in_range_(41, _W0, lamb_min)[0] == _W0


def test_spacing_is_the_default_one_remapped():
    # the remap is linear, so the relative positions inside [lamb_min, w0] are those of the default mesh
    s = MeshUtil.make_continuum_mesh_(41)
    mesh = MeshUtil.make_continuum_mesh_in_range_(41, _W0, _LAMB_MIN)
    np.testing.assert_allclose(
        (mesh - mesh[-1]) / (mesh[0] - mesh[-1]), (s - s[-1]) / (s[0] - s[-1]), rtol=1e-13, atol=1e-15
    )


@pytest.mark.parametrize("lamb_min", [_W0, 2.0 * _W0])
def test_lamb_min_not_shorter_than_the_edge_raises(lamb_min):
    with pytest.raises(ValueError, match="lamb_min"):
        MeshUtil.make_continuum_mesh_in_range_(41, _W0, lamb_min)


@pytest.mark.parametrize("lamb_min", [0.0, -1.0e-5])
def test_in_range_mesh_rejects_nonpositive_lamb_min(lamb_min):
    with pytest.raises(ValueError, match="positive"):
        MeshUtil.make_continuum_mesh_in_range_(41, 9.1e-6, lamb_min)
