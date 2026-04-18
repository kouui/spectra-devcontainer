# -------------------------------------------------------------------------------
# function to generate collisional transition rate cmat[*,*] for hydrogen atom
# -------------------------------------------------------------------------------
# VERSION
# 0.1.0
#    2024/04/06   k.i.   from H_spectra.
# -------------------------------------------------------------------------------

import numpy as np
from mpmath import hyp2f1

from spectra import Constants as Cst

# -----------------------------------------------------------------------
#  Gaunt factor,  Menzel and Pakerie 1935
#  2020.5.26  k.i. (hydrogen.jpynb)


def _Dnn1(nl, nu):
    F1 = hyp2f1(-nu + 1, -nl, 1, -4 * nl * nu / (nu - nl) ** 2)
    F2 = hyp2f1(-nl + 1, -nu, 1, -4 * nl * nu / (nu - nl) ** 2)
    FF = F1**2 - F2**2
    return FF


def _Dnn(nl, nu):
    nnu = np.size(np.array(nu))
    if nnu > 1:
        DD = np.repeat(0j, nnu)
        for i in range(0, nnu):
            DD[i] = _Dnn1(nl, nu[i])
        return DD
    return _Dnn1(nl, nu)


def _gbb(nl, nu):
    nume = np.pi * np.sqrt(3) * ((nu - nl) / (nu + nl)) ** (2 * (nu + nl)) * nu * nl * _Dnn(nl, nu)
    deno = nu - nl
    return (nume / deno).real


def _gbf(nl, k):
    nume = np.pi * np.sqrt(3) * nl * k * np.exp(-4 * k * np.arctan(nl / k)) * np.abs(_Dnn(nl, 1j * k))
    deno = np.sqrt(k**2 + nl**2) * (1 - np.exp(-2 * np.pi * k))
    return (nume / deno).real


# -----------------------------------------------------------------------
def _flu(nl, nu):
    f = 32.0 / (3 * np.pi * np.sqrt(3)) / nl**5 / nu**3 * (1 / nl**2 - 1 / nu**2) ** (-3) * _gbb(nl, nu)
    return (f).real


# -----------------------------------------------------------------------
#  calculate collisional transition rate cmat[*,*] of hydrogen
#  based on the formula in Fujimoto 2004
def Cmat_HI_Theoretical(nLevel, Te, Ne):

    G = 2**4 * np.pi * Cst.a0_**2 * Cst.E_Rydberg_ * np.sqrt(1.0 / 2 / np.pi / Cst.me_ / Cst.k_ / Te)
    Cmat = np.zeros([nLevel, nLevel])
    nn = np.linspace(1, nLevel - 1, nLevel - 1, dtype=int)

    for nl in nn[0:-1]:
        nus = np.linspace(nl + 1, nLevel - 1, nLevel - 1 - nl, dtype=int)
        gl = 2 * nl**2
        for nu in nus:
            f1 = _flu(nl, nu)
            Elu = Cst.E_Rydberg_ * (1 / nl**2 - 1 / nu**2)
            gu = 2 * nu**2
            Cmat[nu - 1, nl - 1] = G * Cst.E_Rydberg_ / Elu * np.exp(-Elu / Cst.k_ / Te) * f1 * Ne  # l->u
            Cmat[nl - 1, nu - 1] = G * Cst.E_Rydberg_ / Elu * gl / gu * f1 * Ne  # u->l
    En = Cst.E_Rydberg_ / nn**2
    fn = 2**3 / 3 / np.sqrt(3) / nn  # approx. <g_bf> = 1
    Cmat[nLevel - 1, 0:-1] = G * Cst.E_Rydberg_ / En * fn * np.exp(-En / Cst.k_ / Te) * Ne
    Zn = nn**2 * (Cst.h_**2 / 2 / np.pi / Cst.me_ / Cst.k_ / Te) ** 1.5 * np.exp(En / Cst.k_ / Te)
    Cmat[0:-1, nLevel - 1] = Zn * Cmat[nLevel - 1, 0:-1] * Ne
    return Cmat
