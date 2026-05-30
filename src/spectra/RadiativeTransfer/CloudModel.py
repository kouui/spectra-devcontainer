# -------------------------------------------------------------------------------
# emergent intensity of a homogeneous slab / cloud
# -------------------------------------------------------------------------------


import numpy as _numpy

from ..ImportAll import *


# NOTE : replace with numba vectorize if necessary
@_numpy.vectorize
def emergent_intensity_(
    Src: T_FLOAT,
    tau: T_FLOAT,
    I0: T_FLOAT = 0.0,
) -> T_FLOAT:
    r"""emergent intensity of a homogeneous slab illuminated from behind.

    .. math:: I = S (1 - e^{-\tau}) + I_0 e^{-\tau}

    Exact solution for a uniform source function `S` over a finite optical
    depth `tau`; not restricted to the optically-thin regime.

    `@_numpy.vectorize` makes this a ufunc: the body operates on scalars while
    callers may pass arrays for `tau` / `I0` (broadcast elementwise).

    Parameters
    ----------
    Src : T_FLOAT
        source function of one line, constant along the slab (scalar per line).
        [erg/cm^2/Sr/cm/s]
    tau : T_FLOAT
        optical depth across the slab at one wavelength. [-]
    I0 : T_FLOAT
        background intensity entering the slab from behind, aligned with `tau`.
        0 ⇒ no background. [erg/cm^2/Sr/cm/s]

    Returns
    -------
    T_FLOAT
        emergent intensity. [erg/cm^2/Sr/cm/s]
    """
    # NOTE: for extreme population inversion (tau < ~-709) exp(-tau) overflows
    # to +inf and the result is nan. Such tau is unphysical (maser runaway) and
    # not a realistic slab input, so it is left unguarded.
    transmission = _numpy.exp(-tau)
    return Src * (1.0 - transmission) + I0 * transmission
