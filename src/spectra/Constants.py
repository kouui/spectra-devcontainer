# -------------------------------------------------------------------------------
# This module defines physics constants and
# some other constants for the convenience of computation.
#
# -------------------------------------------------------------------------------

from .Types import T_DICT, T_FLOAT, T_INT, T_STR

# -------------------------------------------------------------------------------
# mathematical constants
# -------------------------------------------------------------------------------

pi_: T_FLOAT = 3.141592653589793  #: mathematical constant pi, [-]

# -------------------------------------------------------------------------------
# constants for unit conversion
# -------------------------------------------------------------------------------

eV2erg_: T_FLOAT = 1.60217662 * 1.0e-12  #: unit conversion from eV to erg, [:math:`erg/eV`]
J2erg_: T_FLOAT = 1.0e7  #: unit conversion from J to erg, [:math:`erg/J`]
m2cm_: T_FLOAT = 1.0e2  #: unit conversion from m to cm, [:math:`cm/m`]
micro2AA_: T_FLOAT = 1.0e4  #: unit conversion from micro to angstrom, [:math:`A / \mu m`]
K2eV_: T_FLOAT = 8.6173324 * 1.0e-5  #: compute electron temperature in unit eV, [:math:`eV / K`]
eV2AA_div_: T_FLOAT = 1.23984176 * 1.0e4  #: unit conversion from eV to angstrom, [:math:`eV \cdot A`]
cm2K_div_: T_FLOAT = 1.43877713  #: unit conversion from cm to K [:math:`K \cdot cm`]
ni2cm_: T_FLOAT = 9.11266151 * 1.0e-6  #: quantum number ni**2 to its ionizatoin limit wavelength  [:math:`cm`]

# -------------------------------------------------------------------------------
# physcis constants
# -------------------------------------------------------------------------------

c_: T_FLOAT = 2.9979246 * 1.0e10  #: speed of light, [:math:`cm \cdot s^{-1}`]
h_: T_FLOAT = 6.62606885 * 1.0e-27  #: Planck constant, [:math:`erg \cdot s`]
k_: T_FLOAT = 1.3806485 * 1.0e-16  #: Boltzmann constant, [:math:`erg \cdot k^{-1}`]
e_: T_FLOAT = 4.80320425 * 1.0e-10  #: eleceton charge, [:math:`esu`]
mH_: T_FLOAT = 1.660 * 1.0e-24  #: mass of hydrogen atom, [:math:`g`]
me_: T_FLOAT = 9.1093836 * 1.0e-28  #: mass of electron, [:math:`g`]
E_Rydberg_: T_FLOAT = 2.1798741 * 1.0e-11  #: Rydberg constant of hydrogen atom, [:math:`erg`]
a0_: T_FLOAT = 5.2917720859 * 1.0e-9  #: Bohr radius, [:math:`cm`]
alp_: T_FLOAT = 1 / 137.036  #: fine structure constant, [:math:`-`]
AU_: T_FLOAT = 1.49597871 * 1.0e13  #: astronomical unit, distance from the sun to our earth, [:math:`cm`]
R_sun_: T_FLOAT = 6.957 * 1.0e10  #: Solar radius, [:math:`cm`]
R_: T_FLOAT = 8.314 * 1.0e7  #: ideal gas universal gas constant, [:math:`erg \cdot mol^{-1} \cdot K^{-1}`]

# -------------------------------------------------------------------------------
# constants for the convenience of computation
# -------------------------------------------------------------------------------

sqrtPi_: T_FLOAT = 1.7724538509055159  #: square root of pi, [-]
sqrt3_: T_FLOAT = 1.73205080757  #: sqrt(3), [-]

saha_: T_FLOAT = 2 * (2 * pi_ * me_ * k_ / h_ / h_) ** (1.5)
r"""a constant factor in Saha's equation, :math:`2(2 \pi m_e k /h^2)^{3/2}`, [:math:`g^{3/2} \cdot erg^{-3/2} \cdot k^{-3/2} \cdot s^{-3}`]
"""

C0_: T_FLOAT = 5.465366 * 1.0e-11
r"""a constant factor in collisional excitation/ionization rate coefficient, :math:`\pi^a_{0}^{2}(8k/m\pi)^{1/2}`, [:math:`cm^{3}s^{-1}K^{-1/2}`]
"""

# -------------------------------------------------------------------------------
# constants for notations  # need modification
# -------------------------------------------------------------------------------

L_s2i_: T_DICT[T_STR, T_INT] = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6}
"""a hash dictionary mapping symbolic quantum number L to its integer value
"""
L_i2s_: T_DICT[T_INT, T_STR] = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H", 6: "I"}
"""a hash dictionary mapping integer value to its symbolic quantum number L
"""
