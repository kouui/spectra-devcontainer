
#-------------------------------------------------------------------------------
# definition of struct for Atmosphere
#-------------------------------------------------------------------------------
# VERSION
# 0.1.0 
#    2021/05/18   u.k.   spectra-re
# 0.1.1
#    2022/01/07   u.k.   
#        - in `init_VAL_`, added zero array of `column_mass` to initialize struct
# 0.1.2
#    2022/07/29   u.k
#        - added Pg to Atmosphere0D
#-------------------------------------------------------------------------------


from ..ImportAll import *

from dataclasses import dataclass as _dataclass

#-------------------------------------------------------------------------------
# struct
#-------------------------------------------------------------------------------

@_dataclass(**STRUCT_KWGS_UNFROZEN)
class Atmosphere0D:

    Nh : T_FLOAT
    Ne : T_FLOAT
    Te : T_FLOAT
    Vd : T_FLOAT
    Vt : T_FLOAT

    Ti : T_FLOAT = -1.0

    ndim : T_INT     = 0
    is_uniform : T_BOOL = True

    Tr : T_FLOAT     = 6.E3
    use_Tr : T_BOOL  = False

    Pg : T_FLOAT     = 0.05   ##: [Ba] = 0.1 [Pa]

    doppler_shift_continuum : T_BOOL = False

    _coord_type : T_E_ATMOSPHERE_COORDINATE_TYPE = E_ATMOSPHERE_COORDINATE_TYPE.POINT


@_dataclass(**STRUCT_KWGS_UNFROZEN)
class AtmosphereC1D:

    model : T_STR

    Nh : T_ARRAY 
    Ne : T_ARRAY
    Te : T_ARRAY
    Vd : T_ARRAY
    Vt : T_ARRAY
    Z  : T_ARRAY            # height of the atmosphere

    tau5 : T_ARRAY          # (integrated) optical depth at 500nm at each depth point
    column_mass : T_ARRAY   # column mass

    is_uniform : T_BOOL
    ndim : T_INT     = 1
    
    Tr : T_FLOAT     = 6.E3
    use_Tr : T_BOOL  = False

    doppler_shift_continuum : T_BOOL = False

    _coord_type : T_E_ATMOSPHERE_COORDINATE_TYPE = E_ATMOSPHERE_COORDINATE_TYPE.CARTESIAN


#-------------------------------------------------------------------------------
# Create 1D VAL Atmosphere Model
#
# /*   val.pro : VAL model atmosphere
#	Vernazza,J.E., Avrett,E.H., and Loeser,R., 1981,
#	Ap.J.Suppl.,45, 635-725.
#   val_type:
#	A -- a dark point within a cell
#	B -- the average cell center
#	C -- the average quiet sun
#	D -- the average network
#	E -- a bright network element
#	F -- a very bright network element
#		k.i.	'94/07/03
#		k.i.	'04/10/27	calc a.tauc
#       2021.5.13      k.i. from val.pro
#-------------------------------------------------------------------------------

import numpy as _numpy
from ..Atomic import ContinuumOpacity as _ContinuumOpacity

from ..Experimental import ExTau as _Tau

def init_VAL_( val_type : T_STR = "C" ) -> AtmosphereC1D:

    val_type = val_type.upper()

    if val_type == 'A':
        Z = _numpy.array( #* h (km)	*/
            [ -75., -50., -25.,   0.,  50., 100., 150., 250., 350., 450.,
              510., 560., 605., 655., 705., 755., 855., 905., 980.,1080.,
             1205.,1280.,1380.,1505.,1605.,1785.,1915.,1990.,2040.,2074.,
             2096.,2114.,2126.,2141.,2143.,2146.,2150.,2154.,2164.,2190.,
             2240.,2300.,2345.,2375.,2395.,2400.,2405.,2409.,2416.,2430.,
             2439.,2735.], dtype=DT_NB_FLOAT)

        Te = _numpy.array(  #* T (K)	*/
           [8320.,  7610.,  6910.,  6420.,  5840.,  5450.,  5165.,  4720.,  4330.,  4000.,
            3900.,  3930.,  4110.,  4430.,  4740.,  5000.,  5370.,  5530.,  5650.,  5790.,
            5880.,  5945.,  5990.,  6030.,  6080.,  6180.,  6340.,  6460.,  6620.,  6800.,
            7150.,  7600.,  8150.,  9500., 10700., 12300., 18200., 20700., 21700., 22100.,
            22400., 22600., 22800., 23100., 24000., 26700., 30000., 36500., 50000., 89100.,
            141000.,447000.], dtype=DT_NB_FLOAT)

        Nh = _numpy.array(   #* nh (cm-3)	*/
           [ 1.363e17, 1.315e17, 1.259e17, 1.164e17, 9.188e16, 6.860e16, 4.919e16, 2.324e16, 1.000e16, 3.923e15,
            2.118e15, 1.226e15, 7.305e14, 4.146e14, 2.456e14, 1.516e14, 6.326e13, 4.199e13, 2.346e13, 1.117e13,
            4.676e12, 2.838e12, 1.505e12, 7.128e11, 4.073e11, 1.606e11, 8.761e10, 6.313e10, 5.102e10, 4.377e10,
            3.888e10, 3.476e10, 3.146e10, 2.613e10, 2.342e10, 2.044e10, 1.380e10, 1.205e10, 1.137e10, 1.089e10,
            1.022e10, 9.560e09, 9.087e09, 8.718e09, 8.263e09, 7.443e09, 6.652e09, 5.532e09, 4.112e09, 2.290e09,
            1.456e09, 4.544e08	], dtype=DT_NB_FLOAT)

        Ne = _numpy.array(   #* ne (cm-3)	*/
           [ 1.201e15, 4.631e14, 1.542e14, 6.414e13, 2.115e13, 1.059e13, 6.406e12, 2.593e12, 1.035e12, 4.005e11,
            2.260e11, 1.413e11, 9.288e10, 6.036e10, 4.772e10, 4.869e10, 5.515e10, 5.759e10, 5.704e10, 5.572e10,
            5.115e10, 4.585e10, 4.707e10, 4.247e10, 3.527e10, 2.837e10, 2.367e10, 2.195e10, 2.082e10, 2.006e10,
            1.965e10, 1.942e10, 1.912e10, 1.829e10, 1.738e10, 1.630e10, 1.319e10, 1.211e10, 1.162e10, 1.120e10,
            1.058e10, 9.934e09, 9.472e09, 9.118e09, 8.710e09, 7.971e09, 7.210e09, 6.065e09, 4.539e09, 2.705e09,
            1.742e09, 5.443e08], dtype=DT_NB_FLOAT)

        Vt = _numpy.array(   #* vt (km/s)	*/
           [ 1.80, 1.76, 1.70, 1.60, 1.40, 1.20, 1.00, 0.63, 0.52, 0.53,
            0.60, 0.70, 0.83, 0.96, 1.09, 1.23, 1.53, 1.70, 2.14, 2.73,
            3.48, 3.92, 4.51, 5.26, 5.85, 6.92, 7.63, 8.01, 8.22, 8.42,
            8.50, 8.55, 8.60, 8.71, 8.73, 8.74, 8.77, 8.78, 8.81, 8.87,
            9.08, 9.33, 9.49, 9.64, 9.68, 9.70, 9.71, 9.73, 9.76, 9.82,
            9.87, 11.28], dtype=DT_NB_FLOAT)

    elif val_type == 'C':
        Z = _numpy.array( #* h (km)	*/
            [ -75., -50., -25.,   0.,  50., 100., 150., 250., 350., 450.,
              515., 555., 605., 655., 705., 755., 855., 905., 980.,1065., 
             1180.,1280.,1380.,1515.,1605.,1785.,1925.,1990.,2016.,2050.,
             2070.,2080.,2090.,2104.,2107.,2109.,2113.,2115.,2120.,2129.,
             2160.,2200.,2230.,2255.,2263.,2267.,2271.,2274.,2280.,2290.,
             2298.,2543.], dtype=DT_NB_FLOAT)

        Te = _numpy.array(  #* T (K)	*/
            [ 8320.,  7610.,  6910.,  6420.,  5840.,  5455.,  5180.,  4780.,  4465.,  4220.,
              4170.,  4230.,  4420.,  4730.,  5030.,  5280.,  5650.,  5755.,  5925.,  6040.,
              6150.,  6220.,  6280.,  6370.,  6440.,  6630.,  6940.,  7160.,  7360.,  7660.,
              7940.,  8180.,  8440.,  9500., 10700., 12300., 18500., 21000., 22500., 23000.,
             23500., 24000., 24200., 24500., 25500., 28000., 32000., 37000., 50000., 89100.,
             141000.,447000.], dtype=DT_NB_FLOAT)

        Nh = _numpy.array(   #* nh (cm-3)	*/
           [ 1.365e17, 1.317e17, 1.261e17, 1.166e17, 9.203e16, 6.866e16, 4.917e16, 2.315e16, 9.979e15, 3.989e15,
            2.096e15, 1.382e15, 8.119e14, 4.794e14, 2.935e14, 1.864e14, 8.135e13, 5.546e13, 3.147e13, 1.711e13,
            7.865e12, 4.200e12, 2.273e12, 1.048e12, 6.386e11, 2.601e11, 1.380e11, 1.033e11, 9.075e10, 7.705e10,
            6.960e10, 6.541e10, 6.127e10, 5.239e10, 4.673e10, 4.092e10, 2.732e10, 2.403e10, 2.231e10, 2.163e10,
            2.051e10, 1.932e10, 1.862e10, 1.797e10, 1.718e10, 1.567e10, 1.378e10, 1.201e10, 9.038e09, 5.041e09,
            3.205e09, 1.005e09], dtype=DT_NB_FLOAT)

        Ne = _numpy.array(   #* ne (cm-3)	*/
           [ 1.204e15, 4.645e14, 1.547e14, 6.433e13, 2.122e13, 1.066e13, 6.476e12, 2.674e12, 1.110e12, 4.516e11,
            2.495e11, 1.733e11, 1.112e11, 8.085e10, 7.664e10, 8.838e10, 1.064e11, 1.049e11, 1.041e11, 9.349e10,
            8.108e10, 7.486e10, 7.600e10, 6.456e10, 6.005e10, 4.771e10, 4.028e10, 3.858e10, 3.811e10, 3.792e10,
            3.783e10, 3.780e10, 3.799e10, 3.705e10, 3.535e10, 3.306e10, 2.620e10, 2.402e10, 2.276e10, 2.219e10,
            2.120e10, 2.009e10, 1.943e10, 1.881e10, 1.812e10, 1.677e10, 1.498e10, 1.318e10, 9.993e09, 5.961e09,
            3.839e09, 1.205e09], dtype=DT_NB_FLOAT)

        Vt = _numpy.array(   #* vt (km/s)	*/
           [ 1.80, 1.76, 1.70, 1.60, 1.40, 1.20, 1.00, 0.63, 0.52, 0.53,
            0.60, 0.70, 0.83, 0.96, 1.09, 1.23, 1.53, 1.70, 2.14, 2.73,
            3.48, 3.92, 4.51, 5.26, 5.85, 6.92, 7.63, 8.01, 8.22, 8.42,
            8.50, 8.55, 8.60, 8.71, 8.72, 8.74, 8.77, 8.78, 8.81, 8.87,
            9.08, 9.33, 9.49, 9.64, 9.68, 9.70, 9.71, 9.73, 9.76, 9.82,
            9.87, 11.28], dtype=DT_NB_FLOAT)

    elif val_type == 'F':
        Z = _numpy.array( #* h (km)	*/
            [   -75., -50., -25.,   0.,  50., 100., 150., 250., 350., 450.,
              500., 550., 600., 650., 700., 750., 855., 905., 980.,1065., 
             1210.,1280.,1380.,1515.,1605.,1785.,1890.,1973.,1968.,1992.,
             2010.,2023.,2044.,2063.,2065.,2067.,2070.,2074.,2082.,2094.,
             2110.,2122.,2130.,2136.,2144.,2150.,2156.,2160.,2166.,2176.,
             2184.,2429.], dtype=DT_NB_FLOAT)

        Te = _numpy.array(  #* T (K)	*/
            [ 8320.,  7610.,  6910.,  6420.,  5840.,  5450.,  5185.,  4810.,  4570.,  4440.,
              4440.,  4500.,  4690.,  4975.,  5280.,  5540.,  5890.,  6020.,  6190.,  6315.,
              6450.,  6510.,  6600.,  6740.,  6840.,  7150.,  7550.,  7780.,  7950.,  8110.,
              8250.,  8400.,  8660.,  9555., 10800., 13500., 20000., 23000., 23500., 24500.,
             25000., 25500., 26000., 27000., 28000., 30000., 33000., 37000., 50000., 89100.,
             141000.,447000.], dtype=DT_NB_FLOAT)

        Nh = _numpy.array(   #* nh (cm-3)	*/
           [ 1.367e17, 1.319e17, 1.263e17, 1.168e17, 9.215e16, 6.880e16, 4.917e16, 2.308e16, 9.907e15, 3.993e15,
            2.483e15, 1.528e15, 9.258e14, 5.645e14, 3.533e14, 2.287e14, 9.996e13, 6.895e13, 4.005e13, 2.229e13,
            8.828e12, 5.723e12, 3.166e12, 1.494e12, 9.244e11, 3.863e11, 2.386e11, 1.938e11, 1.696e11, 1.521e11,
            1.405e11, 1.320e11, 1.195e11, 1.024e11, 9.136e10, 7.415e10, 5.046e10, 4.393e10, 4.295e10, 4.045e10,
            3.893e10, 3.760e10, 3.654e10, 3.497e10, 3.356e10, 3.130e10, 2.850e10, 2.555e10, 1.923e10, 1.073e10,
            6.819e09, 2.137e09], dtype=DT_NB_FLOAT)

        Ne = _numpy.array(   #* ne (cm-3)	*/
           [ 1.207e15, 4.653e14, 1.550e14, 6.445e13, 2.126e13, 1.065e13, 6.518e12, 2.716e12, 1.160e12, 4.910e11,
            3.179e11, 2.076e11, 1.429e11, 1.236e11, 1.411e11, 1.743e11, 2.091e11, 2.138e11, 2.104e11, 1.889e11,
            1.421e11, 1.346e11, 1.269e11, 1.181e11, 1.140e11, 9.733e10, 8.533e10, 7.976e10, 7.771e10, 7.678e10,
            7.693e10, 7.740e10, 7.823e10, 7.661e10, 7.259e10, 6.438e10, 5.013e10, 4.498e10, 4.408e10, 4.219e10,
            4.086e10, 3.963e10, 3.865e10, 3.722e10, 3.592e10, 3.382e10, 3.111e10, 2.808e10, 2.129e10, 1.270e10,
            8.181e09, 2.567e09], dtype=DT_NB_FLOAT)

        Vt = _numpy.array(   #* vt (km/s)	*/
           [ 1.80, 1.76, 1.70, 1.60, 1.40, 1.20, 1.00, 0.63, 0.52, 0.53,
            0.60, 0.70, 0.83, 0.96, 1.09, 1.23, 1.53, 1.70, 2.14, 2.73,
            3.48, 3.92, 4.51, 5.26, 5.85, 6.92, 7.63, 8.01, 8.22, 8.42,
            8.50, 8.55, 8.60, 8.71, 8.72, 8.74, 8.77, 8.78, 8.81, 8.87,
            9.08, 9.33, 9.49, 9.64, 9.68, 9.70, 9.71, 9.73, 9.76, 9.82,
            9.87, 11.28], dtype=DT_NB_FLOAT)

    else:
        raise NotImplementedError(f"VAL of type {val_type} has not yet been implemented.")

    #-- keep direction : [interior, surface]
    Z[:]  *= 1.E5      # [km] --> [cm]
    Vt[:] *= 1.E5      # [km] --> [cm]
    
    Vd = _numpy.zeros_like( Vt )
    #-- continuum optical depth @500nm (LTE)
    wl0 = 5.E-5 #[cm]
    xc : T_ARRAY = _ContinuumOpacity.H_LTE_continuum_opacity_(Te, Ne, Nh, wl0)
    tau5 = _Tau.tau_( Z[:], xc[:] )

    atmos = AtmosphereC1D(
        model = f"VAL-{val_type}",
        Nh = Nh,
        Ne = Ne,
        Te = Te,
        Vd = Vd,
        Vt = Vt,
        Z  = Z,
        tau5 = tau5,
        column_mass = _numpy.zeros_like(tau5),
        is_uniform = False,
        ndim = 1,
#        column_mass = _numpy.zeros_like( Vt ),
    )

    return atmos




