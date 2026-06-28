# -------------------------------------------------------------------------------
# definition of functions to perform statistical equilibrium
# -------------------------------------------------------------------------------

import numpy as _numpy

from ...Atomic import emisivity as _emisivity
from ...Atomic import extinction as _extinction
from ...ImportAll import *
from ...Math import Integrate as _Integrate
from ...RadiativeTransfer import CloudModel as _RTCloud
from ...Struct import Atmosphere as _Atmosphere
from ...Struct import Atom as _Atom
from ...Struct import Container as _Container


def _SE_to_slab_0D_bb_(
    atom: _Atom.Atom,
    atmos: _Atmosphere.Atmosphere0D,
    SE_con: _Container.SE_Container,
    depth: T_FLOAT,
    I0: T_ARRAY | None = None,
) -> _Container.CloudModel_BB_Container:
    r"""Bound-bound (line) spectrum emerging from a 0D slab.

    For every line transition this forms, on the per-line Doppler-broadened
    wavelength mesh:

    - the integrated opacity ``absorption = (h nu / 4pi)(N_i B_ij - N_j B_ji)``
      and emissivity ``emissivity = (h nu / 4pi) A_ji N_j``, from which the line
      source function ``Src = emissivity / absorption`` (reduces to the standard
      two-level value ``A_ji n_j / (N_i B_ij - N_j B_ji)``; ``Src = 0`` where
      ``A_ji <= 0``),
    - the optical depth ``tau = depth * absorption * phi(lambda)`` (``phi`` the
      normalized absorption profile, so ``tau`` is dimensionless even though
      ``absorption`` is the wavelength-integrated coefficient),
    - the emergent intensity ``I = Src(1 - e^{-tau}) + I0 e^{-tau}`` and its
      wavelength integral ``Ibar``.

    Wavelength frame: the output ``wl_1D`` is the observer-frame mesh, the
    sun-frame ``SE_con.wm_cm_1d`` (atom-rest-frame line centers in cm) shifted by
    ``+w0 Vd_obs / c``. The cloud model reads only what it needs from ``SE_con``
    (no ``wMesh`` dependency); SE already computed both ``dopWidth_cm`` (baked
    into ``wm_cm_1d``) and the unshifted ``absorb_prof_1d`` (sampled at those
    same wavelengths).

    The b-f continuum contribution to the line opacity/emissivity is ignored
    here: it is ~1e-4 of the line term in the line cores, negligible for the
    line spectrum. The continuum is computed separately in ``_SE_to_slab_0D_bf_``.

    Parameters
    ----------
    atom : _Atom.Atom
        atomic model; uses ``atom.Line``, ``atom.nLine`` and ``atom.Abun``.
    atmos : _Atmosphere.Atmosphere0D
        slab parameters; uses ``Nh`` (with ``Abun`` for the number density) and
        ``Vd_obs`` (observer-frame Doppler velocity).
    SE_con : _Container.SE_Container
        statistical-equilibrium result; uses level populations ``n_SE``, the
        line wavelength mesh ``wm_cm_1d``, the absorption profile
        ``absorb_prof_1d`` and the partition ``Line_mesh_idxs``.
    depth : T_FLOAT
        geometrical thickness of the slab along the line of sight, [cm]
    I0 : T_ARRAY | None
        background intensity entering from behind, a 2d table ``(2, n)`` (row 0
        wavelength [cm], row 1 intensity [erg/cm^2/Sr/cm/s]) interpolated once
        onto ``wl_1D``; ``None`` ⇒ zero background.

    Returns
    -------
    _Container.CloudModel_BB_Container
        per-line line spectrum (see the container for field shapes/units).
    """

    nLine = atom.nLine
    Line = atom.Line

    N_ele = atmos.Nh * atom.Abun

    Line_mesh_idxs = SE_con.Line_mesh_idxs
    wm_cm_1d = SE_con.wm_cm_1d
    absorb_prof_1d = SE_con.absorb_prof_1d

    Vd_obs = atmos.Vd_obs

    ## 1. obtain the upper/lower level population for line transitions
    nj: T_ARRAY = SE_con.n_SE[Line["idxJ"][:]]
    ni: T_ARRAY = SE_con.n_SE[Line["idxI"][:]]

    ## 2. compute integrated opacity via Atomic.extinction.bb_extinction_
    Bij: T_ARRAY = Line["BIJ"][:]
    Bji: T_ARRAY = Line["BJI"][:]
    Ni: T_ARRAY = ni * N_ele
    Nj: T_ARRAY = nj * N_ele
    absorption: T_ARRAY = _extinction.bb_extinction_(Line["w0"][:], Bji, Bij, Nj, Ni)

    ## 3. compute line emissivity j and source function S = j / alpha
    # clip Aji<=0 to 0: no spontaneous emission -> zero emissivity -> Src = 0.
    Aji_clip: T_ARRAY = _numpy.where(Line["AJI"][:] > 0.0, Line["AJI"][:], 0.0)
    emissivity: T_ARRAY = _emisivity.bb_emissivity_(Line["w0"][:], Aji_clip, Nj)
    # S = j / alpha (== Aji*nj / (Bij*ni - Bji*nj); hv/4pi and N_ele cancel).
    # alpha == 0 (Bij*Ni == Bji*Nj) has two cases:
    #   1. Ni == Nj == 0 (both levels unpopulated): j == 0 too, the line is
    #      absent, so Src = 0 -> I = I0 is correct. This is the only case
    #      reachable in floating point (e.g. high levels underflowing to 0.0).
    #   2. ni/nj == gi/gj with j != 0 (infinite-temperature boundary between
    #      absorption and inversion): the true S diverges, so Src = 0 is wrong
    #      (it drops the optically-thin emission j*depth*phi). This is a
    #      measure-zero knife-edge, not reachable bitwise from SE populations.
    # So Src = 0 is a numerical guard, NOT the physical alpha->0 limit of S.
    Src: T_ARRAY = _numpy.divide(emissivity, absorption, out=_numpy.zeros_like(emissivity), where=(absorption != 0.0))

    ## 4. compute optical depth given the thichness of the slab
    ## 5. compute the line profile
    arr_w0 = _numpy.empty(nLine, dtype=DT_NB_FLOAT)
    arr_tau_max = _numpy.empty(nLine, dtype=DT_NB_FLOAT)
    arr_Ibar = _numpy.empty(nLine, dtype=DT_NB_FLOAT)
    arr_prof_1D = _numpy.empty_like(absorb_prof_1d)
    arr_tau_1D = _numpy.empty_like(absorb_prof_1d)
    arr_wl_1D = _numpy.empty_like(absorb_prof_1d)

    # fill the full observer-frame wavelength mesh first, so the background
    # intensity can be interpolated onto it in a single pass below.
    for k in range(nLine):
        i1 = Line_mesh_idxs[k, 0]
        i2 = Line_mesh_idxs[k, 1]
        # observer-frame wavelength mesh: sun-frame atom-rest-frame mesh
        # shifted by +w0*Vd_obs/c. Astronomy radial-velocity convention:
        # +Vd_obs = atom velocity AWAY from observer (source receding) →
        # observer sees line center red-shifted to w0 + w0*Vd_obs/c.
        arr_wl_1D[i1:i2] = wm_cm_1d[i1:i2] + (Line["w0"][k] * Vd_obs / CST.c_)

    # one interpolation of the background onto the full observer-frame mesh;
    # sliced per line below so it stays aligned with each line's tau.
    if I0 is None:
        bg_1D = _numpy.zeros_like(arr_wl_1D)
    else:
        if I0.shape[0] != 2:
            raise ValueError(f"background intensity I0 must have shape (2, n_wavelength), but got {I0.shape}")
        bg_1D = _numpy.interp(arr_wl_1D, I0[0, :], I0[1, :])

    for k in range(nLine):
        i1 = Line_mesh_idxs[k, 0]
        i2 = Line_mesh_idxs[k, 1]

        w0 = Line["w0"][k]
        wl = arr_wl_1D[i1:i2]

        tau = depth * absorption[k] * absorb_prof_1d[i1:i2]

        prof = _RTCloud.emergent_intensity_(Src[k], tau[:], bg_1D[i1:i2])

        # l.tau0[i] = np.max(tau)
        # l.prof[i][:] = S[i] * (1. - np.exp(-tau[:]))
        Ibar = _Integrate.trapze_(prof[:], wl[:])

        # store value
        arr_w0[k] = w0
        # abs() handles population inversion: absorption < 0 when Bji*nj > Bij*ni,
        # which makes tau negative; .max() on a negative array returns the
        # least-negative value, masking the strongest |tau|.
        arr_tau_max[k] = _numpy.abs(tau[:]).max()
        arr_Ibar[k] = Ibar
        arr_prof_1D[i1:i2] = prof[:]
        arr_tau_1D[i1:i2] = tau[:]

    cloud_con = _Container.CloudModel_BB_Container(
        w0=arr_w0,
        tau_max=arr_tau_max,
        Ibar=arr_Ibar,
        Src=Src,
        tau_1D=arr_tau_1D,
        prof_1D=arr_prof_1D,
        wl_1D=arr_wl_1D,
        Line_mesh_idxs=Line_mesh_idxs.copy(),
        emissivity=emissivity,
        absorption=absorption,
        # zero-copy aliases: SAME ndarray object, not a copy
        line_emissivity=emissivity,
        line_absorption=absorption,
    )

    return cloud_con


def _SE_to_slab_0D_bf_(
    atom: _Atom.Atom,
    atmos: _Atmosphere.Atmosphere0D,
    SE_con: _Container.SE_Container,
    depth: T_FLOAT,
    I0: T_ARRAY | None = None,
) -> _Container.CloudModel_BF_Container:
    r"""Bound-free (continuum) spectrum emerging from a 0D slab.

    For every continuum transition this evaluates, on the SE continuum wavelength
    mesh ``SE_con.cont_wm_cm`` (edge-first, aligned column-by-column with
    ``atom.PI.alpha_interp``):

    - the spectral emissivity ``j_lambda`` (``emisivity.bf_emissivity_``, the Milne
      recombination form) and extinction ``alpha_lambda``
      (``extinction.bf_extinction_``), both wavelength-resolved,
    - the source function ``Src = j_lambda / alpha_lambda`` (guarded where the
      extinction is zero), the optical depth ``tau = depth * alpha_lambda``, and
      the emergent intensity ``I = Src(1 - e^{-tau}) + I0 e^{-tau}``.

    The photoionization cross section is taken from ``atom.PI.alpha_interp`` (not
    recomputed). The per-continuum inputs come from the ``Cont`` struct: the
    ionization energy ``chi = h f0`` (continuum edge), the statistical weights
    ``gi``/``gj``, and the level indices mapping to populations — the lower bound
    level ``N_i = n_SE[idxI] * N_ele`` and the next-higher-ion ground level
    ``N_{i+1} = n_SE[idxJ] * N_ele`` (the proton density for hydrogen), with
    ``N_ele = atmos.Nh * atom.Abun`` (the same density the b-b path uses).

    No Doppler shift is applied to the continuum, so the output ``wl`` equals
    ``cont_wm_cm``.

    Parameters
    ----------
    atom : _Atom.Atom
        atomic model; uses ``atom.Cont``, ``atom.nCont``, ``atom.Abun`` and the
        photoionization cross section ``atom.PI.alpha_interp``.
    atmos : _Atmosphere.Atmosphere0D
        slab parameters; uses ``Nh`` (with ``Abun``) and ``Te``.
    SE_con : _Container.SE_Container
        statistical-equilibrium result; uses level populations ``n_SE``, the
        electron density ``Ne`` and the continuum mesh ``cont_wm_cm``.
    depth : T_FLOAT
        geometrical thickness of the slab along the line of sight, [cm]
    I0 : T_ARRAY | None
        background intensity entering from behind, a 2d table ``(2, n)`` (row 0
        wavelength [cm], row 1 intensity [erg/cm^2/Sr/cm/s]) interpolated onto
        each continuum row; ``None`` ⇒ zero background.

    Returns
    -------
    _Container.CloudModel_BF_Container
        per-continuum spectrum, fully wavelength-resolved (see the container for
        field shapes/units).
    """

    nCont = atom.nCont
    Cont = atom.Cont

    N_ele = atmos.Nh * atom.Abun
    Te = atmos.Te
    Ne = SE_con.Ne

    alpha_interp = atom.PI.alpha_interp  # (nCont, nContMesh), edge-first
    cont_wm_cm = SE_con.cont_wm_cm  # (nCont, nContMesh), aligned with alpha_interp
    n_SE = SE_con.n_SE

    if I0 is not None and I0.shape[0] != 2:
        raise ValueError(f"background intensity I0 must have shape (2, n_wavelength), but got {I0.shape}")

    arr_w0 = Cont["w0"][:].copy()
    arr_wl = cont_wm_cm.copy()
    arr_emi = _numpy.empty_like(cont_wm_cm)
    arr_abs = _numpy.empty_like(cont_wm_cm)
    arr_Src = _numpy.zeros_like(cont_wm_cm)
    arr_tau = _numpy.empty_like(cont_wm_cm)
    arr_prof = _numpy.empty_like(cont_wm_cm)

    for k in range(nCont):
        wl = cont_wm_cm[k, :]
        alpha = alpha_interp[k, :]
        # ionization energy from the lower level: chi = h*f0 (continuum edge).
        chi = CST.h_ * Cont["f0"][k]
        gi = Cont["gi"][k]
        gk = Cont["gj"][k]
        Ni: T_ARRAY = n_SE[Cont["idxI"][k]] * N_ele
        # population of the next-higher-ion ground level (proton for hydrogen).
        Ni1: T_ARRAY = n_SE[Cont["idxJ"][k]] * N_ele

        emi = _emisivity.bf_emissivity_(wl, alpha, Te, Ne, Ni1, gi, gk, chi, Cont["w0"][k])
        ext = _extinction.bf_extinction_(wl, alpha, Te, Ni)

        tau = depth * ext
        # source function S = j_lambda / alpha_lambda; guard zero extinction.
        Src = _numpy.divide(emi, ext, out=_numpy.zeros_like(emi), where=(ext != 0.0))

        if I0 is None:
            bg = _numpy.zeros_like(wl)
        else:
            bg = _numpy.interp(wl, I0[0, :], I0[1, :])

        prof = _RTCloud.emergent_intensity_(Src, tau, bg)

        arr_emi[k, :] = emi
        arr_abs[k, :] = ext
        arr_Src[k, :] = Src
        arr_tau[k, :] = tau
        arr_prof[k, :] = prof

    return _Container.CloudModel_BF_Container(
        w0=arr_w0,
        Src=arr_Src,
        tau=arr_tau,
        prof=arr_prof,
        wl=arr_wl,
        emissivity=arr_emi,
        absorption=arr_abs,
    )


def SE_to_slab_0D_(
    atom: _Atom.Atom,
    atmos: _Atmosphere.Atmosphere0D,
    SE_con: _Container.SE_Container,
    depth: T_FLOAT,
    I0: T_ARRAY | None = None,
) -> T_TUPLE[_Container.CloudModel_BB_Container, _Container.CloudModel_BF_Container]:
    """calculate both the bound-bound (line) and bound-free (continuum) spectra
    emerging from a 0D slab model.

    Returns the `(bb, bf)` result containers. See `_SE_to_slab_0D_bb_` and
    `_SE_to_slab_0D_bf_` for the per-component details.
    """
    bb_con = _SE_to_slab_0D_bb_(atom, atmos, SE_con, depth, I0)
    bf_con = _SE_to_slab_0D_bf_(atom, atmos, SE_con, depth, I0)
    return bb_con, bf_con
