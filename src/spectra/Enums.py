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
    OMEGA = 1


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
