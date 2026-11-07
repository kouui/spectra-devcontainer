"""FALC + RH 6-level hydrogen setup for the MALI end-to-end test.

Everything is built from the files in ``tests/data/rh_falc`` (the RH atom file
and the FALC arrays), so the test needs nothing outside the repository. The
conventions mirror RH on purpose, because the pinned result is the frozen
solver's answer to RH's problem: RH's collision-table interpolation, its
level-energy constants for the damping, and the n(1s) RH's profile code sees.
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline  # pyright: ignore[reportAttributeAccessIssue]

from spectra import Constants as CST
from spectra.Atomic import Background as BG
from spectra.Atomic import Broadening as BR
from spectra.Enums import E_ABSORPTION_PROFILE_TYPE
from spectra.Function.MALI import Setup
from spectra.RadiativeTransfer.MALI import GlobalMesh
from spectra.Util import MeshUtil

from . import _toy as Toy

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "rh_falc"
RH_ATOM = DATA_DIR / "H_6_CRD.atom"
FALC_NPZ = DATA_DIR / "falc.npz"

XI_REF = 3.0e5  # [cm/s], the VMICRO_CHAR of the RH run
AM_H = 1.008
WGHT_PER_H = 1.4271  # RH mean molecular weight per H nucleus (its abundance.input)
AMU = 1.6605e-24  # [g]
N_CONT_MESH = 41
E_VOIGT = E_ABSORPTION_PROFILE_TYPE.VOIGT

# RH's own constants: using them for the damping isolates the broadening
# physics from the (slightly different) CODATA values in spectra.Constants
h_RH_, c_RH_ = 6.6260755e-34, 2.99792458e8
HE_ABUND = 10.0 ** (10.99 - 12.0)  # RH abundance.input
N_PRINCIPAL = [1, 2, 3, 4, 5]  # H_6_CRD level labels 1S, 2P, 3D, 4F, 5G


def parse_h6_atom_():
    lines_raw = [ln for ln in RH_ATOM.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    it = iter(lines_raw)
    next(it)  # atom ID
    nlevel, nline, ncont, _nfixed = (int(x) for x in next(it).split())
    levels = []
    for _ in range(nlevel):
        parts = next(it).split("'")[0].split()
        levels.append((float(parts[0]), float(parts[1])))
    lines = []
    for _ in range(nline):
        p = next(it).split()
        lines.append(
            {
                "j": int(p[0]),
                "i": int(p[1]),
                "f": float(p[2]),
                "type": p[3],
                "Nlambda": int(p[4]),
                "symm": p[5],
                "qcore": float(p[6]),
                "qwing": float(p[7]),
                "vdWaals": p[8],
                "cvdW": [float(x) for x in p[9:13]],
                "Grad": float(p[13]),
                "cStark": float(p[14]),
            }
        )
    conts = []
    for _ in range(ncont):
        p = next(it).split()
        conts.append({"j": int(p[0]), "i": int(p[1]), "alpha0_cm2": float(p[2]) * 1.0e4, "lamb_min_nm": float(p[5])})
    # collision blocks: TEMP then CE/CI rows (values may carry a trailing label)
    Ttab = np.empty(0)
    CE, CI = [], []
    for ln in it:
        p = ln.split()
        if p[0] == "TEMP":
            Ttab = np.array([float(x) for x in p[2 : 2 + int(p[1])]])
        elif p[0] in ("CE", "CI"):
            first, second = int(p[1]), int(p[2])  # CE: upper, lower; CI: lower, upper
            vals = np.array([float(x) for x in p[3 : 3 + len(Ttab)]])
            (CE if p[0] == "CE" else CI).append((first, second, vals))
        elif p[0] == "END":
            break
    return levels, lines, conts, Ttab, CE, CI


def build_atom_(levels, lines_in, conts_in):
    level_g = np.array([g for _, g in levels])
    level_erg = np.array([e * CST.h_ * CST.c_ for e, _ in levels])  # E in cm^-1
    pairs, Ajis = [], []
    for ln in lines_in:
        i, j = ln["i"], ln["j"]
        wl_cm = 1.0 / (levels[j][0] - levels[i][0])
        # SI f -> A conversion, the same constant as the RH atom files assume
        Aji = ln["f"] * (6.670192e-5 / (wl_cm * 1.0e-2) ** 2) * (level_g[i] / level_g[j])
        pairs.append((i, j))
        Ajis.append(Aji)
    atom = Toy.make_toy_atom_(
        level_g=level_g,
        level_erg=level_erg,
        line_pairs=pairs,
        line_Aji=np.array(Ajis),
        line_Cij_coe=np.zeros(len(pairs)),  # superseded by the per-depth table
        am=AM_H,
        proftype=E_VOIGT,
    )
    atom.Level["isGround"][5] = True  # the proton is the ion-stage ground

    nCont = len(conts_in)
    Cont = np.zeros(nCont, dtype=Toy._CONT_DTYPE)
    Cont_mesh = np.empty((nCont, N_CONT_MESH))
    alpha = np.empty((nCont, N_CONT_MESH))
    for kC, c in enumerate(conts_in):
        i = c["i"]
        chi_ion = level_erg[5] - level_erg[i]
        wl_edge = CST.h_ * CST.c_ / chi_ion
        Cont["idxI"][kC] = i
        Cont["idxJ"][kC] = 5
        Cont["gi"][kC] = level_g[i]
        Cont["gj"][kC] = level_g[5]
        Cont["f0"][kC] = chi_ion / CST.h_
        # RH integrates the b-f rates over [lamb_min, edge] only; keep the
        # edge-concentrated sqrt spacing but remap it onto that range
        s_unit = MeshUtil.make_continuum_mesh_(N_CONT_MESH)
        lm_frac = c["lamb_min_nm"] * 1.0e-7 / wl_edge
        mesh = wl_edge * (lm_frac + (1.0 - lm_frac) * (s_unit - s_unit[-1]) / (1.0 - s_unit[-1]))
        Cont_mesh[kC, :] = mesh
        n_eff = np.sqrt(CST.E_Rydberg_ / chi_ion)
        g_edge = BG.gaunt_bf_(float(wl_edge), float(n_eff), 1)
        for kW in range(N_CONT_MESH):
            g = BG.gaunt_bf_(float(mesh[kW]), float(n_eff), 1)
            alpha[kC, kW] = c["alpha0_cm2"] * (g / g_edge) * (mesh[kW] / wl_edge) ** 3
    atom.Cont = Cont
    atom.Cont_mesh = Cont_mesh
    atom.alpha = alpha
    atom.Cij_coe = np.zeros(atom.nLine + nCont)
    atom.nCont = nCont
    return atom


def cij_table_(atom, levels, Ttab, CE, CI, Te):
    """(ND, nTran) upward collisional coefficients [cm^3 s^-1] per unit ne.

    RH collision.c: CE gives Cdown = CE(T)*ne*(gi/gj)*sqrt(T) (SI ne), CI gives
    Cup = CI(T)*ne*sqrt(T)*exp(-dE/kT). natural-spline interpolation in T with
    endpoint clamping outside the table mirrors RH's splineCoef/splineEval
    (FALC tops out at 1e5 K, far beyond the 3e4 K table edge -- an
    extrapolating spline goes negative there).
    """
    Tlook = np.clip(Te, Ttab[0], Ttab[-1])  # clamp the TABLE LOOKUP only
    ND = Te.shape[0]
    nTran = atom.nLine + atom.nCont
    Cij = np.zeros((ND, nTran))
    tran_index = {}
    for kL in range(atom.nLine):
        tran_index[(int(atom.Line["idxI"][kL]), int(atom.Line["idxJ"][kL]))] = kL
    for kC in range(atom.nCont):
        tran_index[(int(atom.Cont["idxI"][kC]), int(atom.Cont["idxJ"][kC]))] = atom.nLine + kC

    for j, i, vals in CE:  # "CE j i": upper level first
        kT = tran_index[(i, j)]
        coe = CubicSpline(Ttab, vals, bc_type="natural")(Tlook) * 1.0e6  # SI m^3 -> cm^3
        gij = levels[i][1] / levels[j][1]
        dE = (levels[j][0] - levels[i][0]) * CST.h_ * CST.c_
        Cji = coe * gij * np.sqrt(Te)
        Cij[:, kT] = Cji * (levels[j][1] / levels[i][1]) * np.exp(-dE / (CST.k_ * Te))
    for i, j, vals in CI:  # unlike CE ("CE j i"), CI rows read "CI i j": lower level first
        kT = tran_index[(i, j)]
        coe = CubicSpline(Ttab, vals, bc_type="natural")(Tlook) * 1.0e6
        dE = (levels[j][0] - levels[i][0]) * CST.h_ * CST.c_
        Cij[:, kT] = coe * np.sqrt(Te) * np.exp(-dE / (CST.k_ * Te))
    return Cij


def load_falc_():
    with np.load(FALC_NPZ) as z:
        return {k: z[k] for k in z.files}


def build_atmos_(falc):
    cmass, T, ne = falc["cmass"], falc["T"], falc["ne"]
    nHtot = falc["nH_file"].sum(axis=1) + falc["np_file"]
    rho = WGHT_PER_H * AMU * nHtot
    ND = cmass.shape[0]
    Z = np.empty(ND)
    Z[0] = 0.0
    for k in range(1, ND):
        Z[k] = Z[k - 1] + 2.0 * (cmass[k] - cmass[k - 1]) / (rho[k - 1] + rho[k])
    return Setup.Atmos1D(Z=Z, Te=T.copy(), Ne=ne.copy(), Vt=falc["vturb"].copy(), Nt=nHtot, ND=ND)


def build_mesh_(atom, lines_in):
    """The unified axis: per-line RH templates (SYMM Nlambda counts the half
    profile) followed by every continuum mesh, ascending."""
    meshes = []
    for kL, ln in enumerate(lines_in):
        n_full = ln["Nlambda"] if ln["symm"] == "ASYMM" else 2 * ln["Nlambda"] - 1
        if n_full % 2 == 0:
            n_full += 1
        q = MeshUtil.make_full_line_mesh_(n_full, ln["qcore"], ln["qwing"])
        meshes.append(GlobalMesh.anchor_line_mesh_(q, float(atom.Line["w0"][kL]), XI_REF))
    meshes += [np.ascontiguousarray(atom.Cont_mesh[kC, ::-1]) for kC in range(atom.nCont)]
    return GlobalMesh.merge_meshes_(meshes)


def make_bg_fn_(atmos, falc, atom, mesh):
    """Thermal background = what the active 6-level H atom does not provide:
    H-minus b-f/f-f and H f-f from the FALC-file populations (RH's
    Background() likewise runs once with the atmosphere file's hydrogen
    populations and never revisits them). The atom's own b-f comes from the
    populations inside the sweep. The sigma slot carries Thomson plus
    Rayleigh off H(1s) -- the FALC-file n(1s), as in RH -- with each Lyman
    line's tail starting at its window's red edge."""
    nH_neutral = falc["nH_file"].sum(axis=1)
    thomson = BG.thomson_(atmos.Ne)
    ground = atom.Line["idxI"][:] == 0
    wl_red = mesh.wl[mesh.Nblue[: atom.nLine] + mesh.span[: atom.nLine] - 1]

    def bg_fn(wl_cm):
        chi = BG.hminus_bf_(wl_cm, atmos.Te, atmos.Ne, nH_neutral)
        chi += BG.hminus_ff_(wl_cm, atmos.Te, atmos.Ne, nH_neutral)
        chi += BG.hydrogen_ff_(wl_cm, atmos.Te, atmos.Ne, falc["np_file"])
        ray = BG.rayleigh_(
            np.array([wl_cm]),
            atom.Line["w0"][ground],
            atom.Line["AJI"][ground],
            atom.Line["gi"][ground],
            atom.Line["gj"][ground],
            wl_red[ground],
        )[0]
        return chi, thomson + ray * falc["nH_file"][:, 0]

    return bg_fn


def adamp_all_(levels, lines_in, falc):
    """(ND, nLine) Voigt damping parameter with every RH broadening term:
    radiative (Grad), van der Waals (UNSOLD), quadratic and linear Stark.

    van der Waals uses the n(1s) RH's profile code sees: the file's n(1s)
    rescaled by RH's chemical equilibrium (``n1s_rescale``), fixed during
    the iteration like the profiles themselves.
    """
    Te, Ne, vturb = falc["T"], falc["ne"], falc["vturb"]
    n1s = falc["nH_file"][:, 0] * falc["n1s_rescale"]
    E = np.array([lv[0] for lv in levels]) * 100.0 * h_RH_ * c_RH_ * 1.0e7  # [erg]
    vbroad = BR.broadening_velocity_(Te, vturb, AM_H)
    adamp = np.empty((Te.shape[0], len(lines_in)))
    for kL, ln in enumerate(lines_in):
        i, j = ln["i"], ln["j"]
        assert ln["vdWaals"] == "UNSOLD"
        g_vdw = BR.van_der_waals_unsold_(
            Te, n1s, AM_H, float(E[i]), float(E[j]), float(E[-1]), ln["cvdW"][0], ln["cvdW"][2], HE_ABUND
        )
        g_sq = BR.stark_quadratic_(Te, Ne, AM_H, float(E[i]), float(E[j]), float(E[-1]), ln["cStark"])
        g_sl = BR.stark_linear_hydrogen_(Ne, N_PRINCIPAL[i], N_PRINCIPAL[j])
        w0_cm = float(h_RH_ * c_RH_ / ((E[j] - E[i]) * 1.0e-7) * 100.0)
        adamp[:, kL] = BR.damping_parameter_(ln["Grad"] + g_vdw + g_sq + g_sl, w0_cm, vbroad)
    return adamp
