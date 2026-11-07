# -------------------------------------------------------------------------------
# Enum definition in spectra
# -------------------------------------------------------------------------------


from enum import IntEnum as _IntEnum


class E_ATOMIC_DATA_SOURCE(_IntEnum):
    EXPERIMENT = 1
    CALCULATE = 2


class E_ATOM(_IntEnum):
    HYDROGEN = 1
    HYDROGEN_LIKE = 2
    NORMAL = 3


class E_COLLISIONAL_TRANSITION(_IntEnum):
    EXCITATION = 1
    IONIZATION = 2


class E_COLLISIONAL_TRANSITION_SOURCE(_IntEnum):
    ELECTRON = 1
    PROTON = 2
    CHARGE_TRANSFER = 3


class E_COLLISIONAL_TRANSITION_FORMULA(_IntEnum):
    """What `Collisional_Transition.Omega_table` holds.

    OMEGA: the effective collision strength Omega (dimensionless); the rate
    follows from `Atomic.Collision`. This is the only formula the statistical
    equilibrium solver of `Function.SEquil.SELib` understands.

    RH_CE / RH_CI: the coefficient tabulated in an RH atom file, already in CGS
    [cm^3 s^-1 K^-1/2] (the file's SI value times 1e6, converted once when the
    file is read: an evaluator must NOT multiply by 1e6 again). Per unit electron
    density, with Te the electron temperature and dE = E_j - E_i > 0:

    - RH_CE (excitation table): the downward rate is C_ji = coe * (g_i / g_j) * sqrt(Te);
      the upward rate follows by detailed balance, C_ij = C_ji * (g_j / g_i) * exp(-dE / kTe).
    - RH_CI (ionization table): the upward rate is C_ij = coe * sqrt(Te) * exp(-dE / kTe).

    In both tables the table is interpolated in Te on `Te_table`, and rows
    follow the order of the transitions (lines for CE, continua for CI) with
    `Coe["idxI"]` the lower and `Coe["idxJ"]` the upper level, whatever the
    order in the file.
    """

    OMEGA = 1
    RH_CE = 2
    RH_CI = 3


class E_VDW_TYPE(_IntEnum):
    """Van der Waals broadening prescription of a line (`RL.Coe["vdW_type"]`)."""

    NONE = 0  # not given: the file format has no van der Waals data
    UNSOLD = 1  # Unsold approximation; `cvdW[0]` (H) and `cvdW[2]` (He) multiply it


class E_ABSORPTION_PROFILE_TYPE(_IntEnum):
    VOIGT = 0
    GAUSSIAN = 1


class E_ATMOSPHERE_COORDINATE_TYPE(_IntEnum):
    POINT = 0
    CARTESIAN = 1


class E_FEAUTRIER_ORDER(_IntEnum):
    # Values are the formal accuracy order of the scheme, so the member name and
    # the numeric value cannot drift apart.
    SECOND = 2
    HERMITE = 4


# import numba as _numba
# T_DATA_MEMTYPE = _numba.typeof( T_DATA.CALCULATE )
# T_ATOM_MEMTYPE = _numba.typeof( T_ATOM.HYDROGEN )
