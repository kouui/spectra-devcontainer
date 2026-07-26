# -------------------------------------------------------------------------------
# function/struct for feautrier method
# -------------------------------------------------------------------------------

from collections import namedtuple as _namedtuple

import numpy as _numpy

from ..ImportAll import *
from ..Math import GaussLeg as _GaussLeg

# Hermite (4th order) correction weight. Kept as an exact ratio; the reference
# implementation hardcodes a 12-digit decimal literal.
_A_SIXTH: T_FLOAT = 1.0 / 6.0

# NamedTuple return type for formal_improved_RH_. Module-scope so numba can
# capture the type at @njit-compile time (numba treats a namedtuple as a tuple in
# nopython mode); a locally defined type would fail to compile.
Feautrier_Result = _namedtuple("Feautrier_Result", ["j", "I_emergent", "Psi"])


def formal_improved_RH_(
    tau: T_ARRAY,
    S: T_ARRAY,
    mu: T_FLOAT,
    r0: T_FLOAT,
    h0: T_FLOAT,
    rn: T_FLOAT,
    hn: T_FLOAT,
    order: T_E_FEAUTRIER_ORDER = E_FEAUTRIER_ORDER.SECOND,
    with_psi: T_BOOL = False,
):
    """
    Purpose :
        Evaluate monochromatic intensities j = 0.5*(I(+)+I(-))
        along a ray with given optical depth scale tau and source function.
        [G.B. Rybicki & D.G Hummer, A&A 245, 171-181]
        with:
            finite slab :
                I0(-) = 0             --> r0 = h0 = 0
                In(+) = I_incident(+) --> rn = 0, hn = I_incident(+)
            semi-finite :
                In(+) = In(-)         --> hn = 0, rn = 1
                or
                In(+)=B+mu*dB/dtau    --> rn = 0, hn = B+mu*dB/dtau (diffusion limit)
            illuminated medium :
                I0(-) = I_ill         --> r0 = 0, h0 = I_ill
            symmetrical slab :
                I(+) = I(-)           --> hn = 0, rn = 1, last grid locates at the center of the slab

    Input :
        tau: (ND,), optical depth scale
        S: (ND,), monochromatic source function
        mu: (,), mu=cos(theta)
        r0: (,), 0 or 1
        h0: (,), incident intensity at upper boundary
        <upper boundary condition: I_upper(-)= r0*I_upper(+)+ h0,   at tau[0]>
        rn: (,), 0 or 1
        hn: (,), incident intensity at lower boundary
        <lower boundary condition: I_lower(+)= rn*I_lower(-)+ hn,   at tau[nd-1]>
        order: (,), E_FEAUTRIER_ORDER.SECOND or .HERMITE (Auer 1976)
        with_psi: (,), whether to evaluate the diagonal operator Psi

    Output : Feautrier_Result(j, I_emergent, Psi)
        j: (ND,), 0.5*(I(+) + I(-)) at each optical depth, mean intensity like.
            SECOND converges at 2nd order. HERMITE raises the *interior* rows to
            4th order but its two boundary rows gain only one order (3rd), and the
            boundary is what limits the global rate: measured convergence is 3rd
            order, not 4th, on a uniform grid. This matches the reference
            implementation, whose "fourth order" refers to the interior scheme.
            HERMITE is still worth its cost -- roughly 100x lower error than
            SECOND at a few hundred depth points.
        I_emergent: (,), I(+) at the upper boundary.
        Psi: (ND,) if with_psi else (0,).
            diag(T^-1) = dj/dS, the local (Jacobi) approximate operator for ALI;
            angle-average as Lambda_star[k] = sum_mu w_mu * Psi_mu[k].
            SECOND : the exact diagonal of Lambda.
            HERMITE: approximate. Hermite folds S[k-1], S[k+1] into the right hand
                side (Stmp = M.S), so Lambda = T^-1.M and the exact diagonal would
                need the near-diagonal band of T^-1, not just its diagonal. It is
                returned anyway rather than refused because the ALI fixed point is
                independent of Lambda_star (those terms cancel once S_new = S_old),
                so an imperfect operator changes the convergence rate but not the
                converged solution.
            The bound Psi <= 1 relies on A1, C1 > 0. Under HERMITE either boundary
            term, C1[0] = 2/dtau_m[0]**2 - 1/3 or A1[-1] = 2/dtau_m[-1]**2 - 1/3,
            turns negative once that boundary interval exceeds sqrt(6) ~ 2.45, and
            both the bound and Hermite's accuracy break down there. A log tau grid
            with a coarse deep end trips this at the lower boundary.
    """

    ND = tau.shape[0]
    # -- dtau
    dtau_m = (tau[1:] - tau[:-1]) / mu  # 0 -> ND-2 : 1/2 -> ND-3/2
    dtau = 0.5 * (dtau_m[:-1] + dtau_m[1:])  # 0 -> ND-3 : 1 -> ND-2

    # Coefficients are held as arrays rather than consumed as loop scalars: the
    # Hermite corrections rewrite them once the second-order values are known, and
    # the Psi backward recursion needs every row's coefficients a second time.
    abc = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    A1 = _numpy.zeros(ND, dtype=DT_NB_FLOAT)
    C1 = _numpy.zeros(ND, dtype=DT_NB_FLOAT)
    Stmp = _numpy.empty(ND, dtype=DT_NB_FLOAT)

    # -- upper boundary row
    f0 = (1.0 - r0) / (1.0 + r0)
    abc[0] = 1.0 + (2.0 / dtau_m[0]) * f0
    C1[0] = 2.0 / dtau_m[0] / dtau_m[0]
    Stmp[0] = S[0] + 2.0 * h0 / ((1.0 + r0) * dtau_m[0])

    # -- lower boundary row
    fn = (1.0 - rn) / (1.0 + rn)
    abc[ND - 1] = 1.0 + (2.0 / dtau_m[ND - 2]) * fn
    A1[ND - 1] = 2.0 / dtau_m[ND - 2] / dtau_m[ND - 2]
    Stmp[ND - 1] = S[ND - 1] + 2.0 * hn / ((1.0 + rn) * dtau_m[ND - 2])

    # -- interior rows
    for d in range(1, ND - 1):
        A1[d] = 1.0 / dtau_m[d - 1] / dtau[d - 1]
        C1[d] = 1.0 / dtau_m[d] / dtau[d - 1]
        abc[d] = 1.0
        Stmp[d] = S[d]

    if order == E_FEAUTRIER_ORDER.HERMITE:
        C1[0] -= 2.0 * _A_SIXTH
        Stmp[0] += 2.0 * _A_SIXTH * (S[1] - S[0])
        A1[ND - 1] -= 2.0 * _A_SIXTH
        Stmp[ND - 1] += 2.0 * _A_SIXTH * (S[ND - 2] - S[ND - 1])
        for d in range(1, ND - 1):
            # dtau_m[d] pairs with A1[d] and dtau_m[d-1] with C1[d], i.e. each
            # correction takes the interval on the far side of the coefficient it
            # scales. Verbatim from the reference implementation. The pairing is
            # invisible on a uniform grid (the two intervals are equal), so the
            # convergence tests do not cover it -- do not "correct" it on symmetry
            # grounds without a non-uniform-grid test to justify the change.
            Ak = _A_SIXTH * (1.0 - 0.5 * dtau_m[d] * dtau_m[d] * A1[d])
            Ck = _A_SIXTH * (1.0 - 0.5 * dtau_m[d - 1] * dtau_m[d - 1] * C1[d])
            A1[d] -= Ak
            C1[d] -= Ck
            Stmp[d] += Ak * (S[d - 1] - S[d]) + Ck * (S[d + 1] - S[d])

    # -- forward-elimination
    E = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    F = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    F[0] = abc[0] / C1[0]
    E[0] = Stmp[0] / (abc[0] + C1[0])
    for d in range(1, ND - 1):
        F[d] = (abc[d] + A1[d] * F[d - 1] / (1.0 + F[d - 1])) / C1[d]
        E[d] = (Stmp[d] + A1[d] * E[d - 1]) / (C1[d] * (1.0 + F[d]))
    E[ND - 1] = (Stmp[ND - 1] + A1[ND - 1] * E[ND - 2]) / (abc[ND - 1] + A1[ND - 1] * (F[ND - 2] / (1.0 + F[ND - 2])))

    # -- backward-substitution
    j = _numpy.empty(ND, dtype=DT_NB_FLOAT)
    j[ND - 1] = E[ND - 1]
    for d in range(ND - 2, -1, -1):
        j[d] = (1.0 + F[d]) ** (-1) * j[d + 1] + E[d]

    # I(+) at the upper boundary, from inverting j[0] = 0.5*((1+r0)*I(+) + h0).
    I_emergent = (1.0 + f0) * j[0] - h0 / (1.0 + r0)

    # -- diagonal operator: G mirrors F upwards from the lower boundary, and the
    #    two continued fractions together give diag(T^-1) row by row.
    if with_psi:
        G = _numpy.empty(ND, dtype=DT_NB_FLOAT)
        Psi = _numpy.empty(ND, dtype=DT_NB_FLOAT)
        G[ND - 1] = abc[ND - 1] / A1[ND - 1]
        for d in range(ND - 2, 0, -1):
            G[d] = (abc[d] + C1[d] * G[d + 1] / (1.0 + G[d + 1])) / A1[d]
        Psi[0] = 1.0 / (abc[0] + C1[0] * G[1] / (1.0 + G[1]))
        for d in range(1, ND - 1):
            Psi[d] = 1.0 / (abc[d] + A1[d] * F[d - 1] / (1.0 + F[d - 1]) + C1[d] * G[d + 1] / (1.0 + G[d + 1]))
        Psi[ND - 1] = 1.0 / (abc[ND - 1] + A1[ND - 1] * F[ND - 2] / (1.0 + F[ND - 2]))
    else:
        Psi = _numpy.empty(0, dtype=DT_NB_FLOAT)

    return Feautrier_Result(j, I_emergent, Psi)


# -------------------------------------------------------------------------------
# direct solution
# -------------------------------------------------------------------------------


def direct_feautrier_(
    tau: T_ARRAY,
    I_upper: T_ARRAY,
    I_lower: T_ARRAY,
    planckB: T_ARRAY,
    eps: T_ARRAY,
    n_angle: T_INT = 4,
) -> T_ARRAY:
    """
    Direct (non-iterative) Feautrier solution of the monochromatic two-level-atom
    scattering problem. The angle coupling of the scattering integral is carried
    inside the block-tridiagonal elimination, so the coupled problem is solved in a
    single pass and the returned source function needs no iteration.

    Intended as a *verification reference* for the formal solver and for ALI, not as
    a production solver: it is monochromatic with coherent scattering, i.e. it has
    no line profile and no coupling between frequencies.

    Unlike formal_improved_RH_, mu is kept inside the coefficients instead of
    scaling dtau; both express the same transfer equation.

    Input:
        tau: (ND,), optical depth, [0,ND]~[upper-lower]
        I_upper: (n_angle,), incident intensity at upper surface
        I_lower: (n_angle,), incident intensity at lower surface
        planckB: (ND,), local planck function, [0,ND]~[upper-lower]
        eps: (ND,), destruction coefficient, [0,ND]~[upper-lower]
        n_angle: (,), number of Gauss-Legendre angle quadrature points on [0,1]

    Output:
        S: (ND,), source function, [0,ND]~[upper-lower]
    """
    assert I_upper.size == n_angle, "I_upper must hold one intensity per angle."
    assert I_lower.size == n_angle, "I_lower must hold one intensity per angle."
    ND = tau.shape[0]

    # -- angle quadrature mus and weights
    mus, ws = _GaussLeg.gauss_quad_coe_(0.0, 1.0, n_angle)

    # -- array initialization
    D = _numpy.zeros((ND, n_angle, n_angle), dtype=DT_NB_FLOAT)
    E = _numpy.zeros((ND, n_angle), dtype=DT_NB_FLOAT)
    j = _numpy.zeros((ND, n_angle), dtype=DT_NB_FLOAT)

    # -- dtau
    dtau_m = tau[1:] - tau[:-1]  # 0 -> ND-2 : 1/2 -> ND-3/2
    dtau = 0.5 * (dtau_m[:-1] + dtau_m[1:])  # 0 -> ND-3 : 1 -> ND-2

    A = _numpy.zeros((n_angle, n_angle), dtype=DT_NB_FLOAT)
    B = _numpy.zeros((n_angle, n_angle), dtype=DT_NB_FLOAT)
    C = _numpy.zeros((n_angle, n_angle), dtype=DT_NB_FLOAT)
    R = _numpy.zeros(n_angle, dtype=DT_NB_FLOAT)

    # -- forward-elimination
    for d in range(ND):
        if d == 0:
            for i in range(n_angle):
                C[i, i] = 2 * mus[i] * mus[i] / dtau_m[d] / dtau_m[d]
                B[i, :] = -(1 - eps[d]) * ws[:]
                B[i, i] += 1 + 2 * mus[i] / dtau_m[d] + 2 * mus[i] * mus[i] / dtau_m[d] / dtau_m[d]
                R[i] = eps[d] * planckB[d] + 2 * mus[i] / dtau_m[d] * I_upper[i]
            D[d, :, :] = _numpy.linalg.solve(B, C)
            E[d, :] = _numpy.linalg.solve(B, R)

        elif d == (ND - 1):
            for i in range(n_angle):
                A[i, i] = 2 * mus[i] * mus[i] / dtau_m[d - 1] / dtau_m[d - 1]
                B[i, :] = -(1 - eps[d]) * ws[:]
                B[i, i] += 1 + 2 * mus[i] / dtau_m[d - 1] + 2 * mus[i] * mus[i] / dtau_m[d - 1] / dtau_m[d - 1]
                R[i] = eps[d] * planckB[d] + 2 * mus[i] / dtau_m[d - 1] * I_lower[i]
            M = B - A @ D[d - 1, :, :]
            D[d, :, :] = 0
            E[d, :] = _numpy.linalg.solve(M, R + A @ E[d - 1, :])

        else:
            for i in range(n_angle):
                A[i, i] = mus[i] * mus[i] / dtau_m[d - 1] / dtau[d - 1]
                C[i, i] = mus[i] * mus[i] / dtau_m[d] / dtau[d - 1]
                B[i, :] = -(1 - eps[d]) * ws[:]
                B[i, i] += 1 + A[i, i] + C[i, i]
                R[i] = eps[d] * planckB[d]
            M = B - A @ D[d - 1, :, :]
            D[d, :, :] = _numpy.linalg.solve(M, C)
            E[d, :] = _numpy.linalg.solve(M, R + A @ E[d - 1, :])

    # -- backward-substitution
    d = ND - 1
    j[d, :] = E[d, :]
    for d in range(ND - 2, -1, -1):
        j[d, :] = D[d, :, :] @ j[d + 1, :] + E[d, :]

    # -- compute source function
    S = _numpy.zeros(ND, dtype=DT_NB_FLOAT)
    for d in range(ND):
        S[d] = (1 - eps[d]) * (j[d, :] * ws[:]).sum() + eps[d] * planckB[d]

    return S


# -----------------------------------------------------------------------------
# numba optimization
# -----------------------------------------------------------------------------

if CFG._IS_JIT:
    formal_improved_RH_ = nb_njit(**NB_NJIT_KWGS)(formal_improved_RH_)
    # direct_feautrier_ is left interpreted on purpose: it is a verification-only
    # reference whose runtime is irrelevant, and its per-depth np.linalg.inv would
    # impose array-contiguity constraints for no benefit.

else:
    pass
