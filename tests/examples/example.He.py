from spectra.Function.SEquil import SELib
from spectra.ImportAll import *
from spectra.Struct import Atmosphere, Atom, Radiation

conf_path = CFG._ROOT_DIR / "data" / "conf" / "He.conf"
atom, wMesh, path_dict = Atom.init_Atom_(str(conf_path), is_hydrogen=False)

atmos = Atmosphere.Atmosphere0D(Nh=1.0e11, Ne=5.0e10, Te=7.0e3, Vt=5.0e5)
radiation = Radiation.init_Radiation_()
SE_con, Rate_con = SELib.cal_SE_with_Ne_Te_(atom, atmos, wMesh, radiation, None)

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
