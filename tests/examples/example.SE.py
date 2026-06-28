from spectra.Function.SEquil import SELib
from spectra.ImportAll import *
from spectra.Struct import Atmosphere, Atom, Radiation

conf_path = CFG._ROOT_DIR / "data/conf/H.conf"
atom = Atom.init_Atom_(str(conf_path), is_hydrogen=True)

atmos = Atmosphere.Atmosphere0D(Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vt=5.0e5)
radiation = Radiation.init_Radiation_()
SE_con, Rate_con = SELib.cal_SE_with_Nh_Te_(atom, atmos, radiation, None)

print(">>> calculate SE given Nh and Te <<<\n")
print(f"Electron temperature  = {atmos.Te:.1E}")
print(f"Electron density      = {atmos.Ne:.1E}")
print("-" * 35)
print("SE  :")
for v in SE_con.n_SE[:]:
    print(f"{v:.4E}", end="  ")
print("\nLTE :")
for v in SE_con.n_LTE[:]:
    print(f"{v:.4E}", end="  ")
print(" ")


"""Correct answer
Electron temperature  = 7.0E+03
Electron density      = 1.3E+11
-----------------------------------
SE  :
8.6628E-01  3.3842E-07  3.0978E-09  1.1299E-09  8.7587E-10  8.3796E-10  8.9827E-10  1.0387E-09  1.3372E-01
LTE :
3.7019E-01  6.7241E-08  6.6075E-09  3.9264E-09  3.6943E-09  4.0386E-09  4.6556E-09  5.4592E-09  6.2981E-01
"""

atmos = Atmosphere.Atmosphere0D(Nh=1.0e11, Ne=5.0e10, Te=7.0e3, Vt=5.0e5)
radiation = Radiation.init_Radiation_()
SE_con, Rate_con = SELib.cal_SE_with_Ne_Te_(atom, atmos, radiation, None)

print("\n>>> calculate SE given Ne and Te <<<\n")
print(f"Electron temperature  = {atmos.Te:.1E}")
print(f"Hydrogen density      = {atmos.Nh:.1E}")
print("-" * 35)
print("SE  :")
for v in SE_con.n_SE[:]:
    print(f"{v:.4E}", end="  ")
print("\nLTE :")
for v in SE_con.n_LTE[:]:
    print(f"{v:.4E}", end="  ")
print(" ")
