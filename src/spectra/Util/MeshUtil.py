# -------------------------------------------------------------------------------
# function definition of wavelength mesh utility
# -------------------------------------------------------------------------------

import numpy as _numpy

from ..ImportAll import *
from ..Math import BasicM

# -------------------------------------------------------------------------------
# construct wavelength mesh
# -------------------------------------------------------------------------------


@nb_njit(**NB_NJIT_KWGS)
def make_half_line_mesh_(nLambda: T_INT, qcore: T_FLOAT, qwing: T_FLOAT, q: T_ARRAY):
    """
    Construct half line mesh. Following RH's `getlambda.c`.
    called by it's wrapper function `makeLineMesh_Full`

    Parameters
    ----------

    nLambda : T_INT
        Number of points in full line mesh. Must be an odd number.

    qcore : T_FLOAT
        a parameter to control mesh density.
        there are always half of all points inside [-qcore,qcore].

    qwing : T_FLOAT
        a parameter to control how far away your mesh points reach.

    q : T_ARRAY
        a array to store output half line mesh distribution.

    Notes
    -------
    with q[0] = 0 and q[-1] = qwing, nLambda//2 points belonging to -qcore< q < +qcore. So,

    - change qwing, you change how far your Doppler width mesh reach
    - change qcore, you change how dense in line core.
    """
    if not BasicM.is_odd_(nLambda):
        raise ValueError("`nLambda` should be an odd number.")

    nLhalf = nLambda // 2 + 1

    if qwing <= 2 * qcore:
        beta = 1.0
    else:
        beta = 0.5 * qwing / qcore

    y = beta + (beta * beta + (beta - 1.0) * nLhalf + 2.0 - 3.0 * beta) ** (0.5)
    b = 2.0 * _numpy.log(y) / (nLhalf - 1)
    a = qwing / (nLhalf - 2.0 + y * y)

    for i in range(nLhalf):
        q[i] = a * (i + (_numpy.exp(b * i) - 1.0))


@nb_njit(**NB_NJIT_KWGS)
def make_full_line_mesh_(nLambda: T_INT, qcore: T_FLOAT = 2.5, qwing: T_FLOAT = 10.0) -> T_ARRAY:
    """
    Construct Full line mesh.
    Calls inner function `makeLineMesh_Half` to construct half part
    and then make the use of symmetry.

    Parameters
    ----------

    nLambda : T_INT
        Number of points in full line mesh. Must be an odd number.

    qcore : T_FLOAT
        a parameter to control mesh density.
        there are always half of all points inside [-qcore,qcore].

    qwing : T_FLOAT
        a parameter to control how far away your mesh points reach.

    Returns
    --------
    mesh : T_ARRAY
        Full line mesh distribution.

    Notes
    -------
    with q[0] = 0 and q[-1] = qwing, nLambda//2 points belonging to -qcore< q < +qcore. So,

    - change qwing, you change how far your Doppler width mesh reach
    - change qcore, you change how dense in line core.
    """

    mesh = _numpy.empty(nLambda, dtype=DT_NB_FLOAT)
    nLmid = nLambda // 2
    make_half_line_mesh_(nLambda, qcore, qwing, mesh[nLmid:])
    mesh[:nLmid] = -1.0 * mesh[nLmid + 1 :][::-1]

    return mesh


def make_continuum_mesh_(nLambda: T_INT) -> T_ARRAY:
    """
    Given number of mesh points,
    compute a mesh distribution sampling most aroung wavelength edge.

    Parameters
    ----------

    nLambda : T_INT
        Number of points in Continuum mesh.

    Returns
    --------

    Mesh : T_ARRAY
        Output mesh distribution.

    """
    mesh = _numpy.empty(nLambda, dtype=DT_NB_FLOAT)
    for j in range(1, nLambda + 1):
        qj = (nLambda + 1.0 - j) / nLambda
        mesh[j - 1] = qj ** (0.5)

    return mesh


def make_continuum_mesh_in_range_(nLambda: T_INT, w0: T_FLOAT, lamb_min: T_FLOAT) -> T_ARRAY:
    """
    Continuum mesh confined to ``[lamb_min, w0]``, edge first.

    The spacing of `make_continuum_mesh_` (dense at the edge) is remapped
    linearly so that it spans exactly ``[lamb_min, w0]``. The unscaled
    `make_continuum_mesh_` only reaches ``w0 / sqrt(nLambda)``; use this one when
    a data source states how far to the short-wavelength side the continuum is
    needed (an RH atom file does: it integrates the bound-free rates over
    ``[lamb_min, w0]`` only), so that no mesh point is spent beyond that limit.

    Parameters
    ----------
    nLambda : T_INT
        Number of points.

    w0 : T_FLOAT
        Edge (threshold) wavelength, [:math:`cm`].

    lamb_min : T_FLOAT
        Shortest wavelength of the mesh, [:math:`cm`]; must be shorter than ``w0``.

    Returns
    --------
    mesh : T_ARRAY
        Wavelengths [:math:`cm`], descending from ``w0`` (first) to ``lamb_min``
        (last).
    """
    if not lamb_min > 0.0:
        raise ValueError(f"lamb_min ({lamb_min} cm) must be positive")
    s = make_continuum_mesh_(nLambda)
    lm_frac = lamb_min / w0
    if lm_frac >= 1.0:
        raise ValueError(f"lamb_min ({lamb_min} cm) must be shorter than the edge wavelength ({w0} cm)")
    mesh = w0 * (lm_frac + (1.0 - lm_frac) * (s - s[-1]) / (1.0 - s[-1]))
    # lm_frac + (1 - lm_frac) can round to one ulp above 1; an edge point beyond the
    # threshold would get a zero cross section, so the edge is set exactly
    mesh[0] = w0
    return mesh


def half_to_full_(arr_half: T_ARRAY, isMinus: T_BOOL = False) -> T_ARRAY:
    r"""
    create full (anti-)symmetric full array according to half array

    Parameters
    ----------

    arr_half : T_ARRAY, 1d
        half array.

    isMinus : T_BOOL
        True : anti-symmetric full array; False : symmetric full array

    Returns
    --------

    arr_full : T_ARRAY, 1d,
        full array.

    """
    _nLmid = arr_half.shape[0]
    _nLfull = (_nLmid - 1) * 2 + 1
    _arr_full = _numpy.zeros(_nLfull, dtype=arr_half.dtype)

    fac = -1 if isMinus else 1

    _arr_full[_nLmid:] = arr_half[1:]
    _arr_full[:_nLmid] = fac * arr_half[::-1]

    return _arr_full


def array_from_1D_(arr_1D: T_ARRAY, mesh_idxs: T_ARRAY, k: T_INT) -> T_ARRAY:
    """extract target sub-array from a 1D array containing sub-arrays with different size

    Parameters
    ----------
    arr_1D : T_ARRAY
        the 1D array containing sub-arrays with different size
    mesh_idxs : T_ARRAY, 2d, (nLine, 2)
        start and end index of each sub-array
    k : T_INT
        the index of line you want to extract

    Returns
    -------
    T_ARRAY
        target sub-array
    """
    i1: T_INT = mesh_idxs[k, 0]
    i2: T_INT = mesh_idxs[k, 1]

    return arr_1D[i1:i2].copy()
