from spectra.Function import SlabModel
from spectra.Function.SEquil import SELib
from spectra.ImportAll import *
from spectra.Struct import Atmosphere, Atom, Radiation

conf_path = CFG._ROOT_DIR / "data/conf/H.conf"
atom, wMesh, path_dict = Atom.init_Atom_(str(conf_path), is_hydrogen=True)

atmos = Atmosphere.Atmosphere0D(Nh=1.0e12, Ne=1.0e11, Te=7.0e3, Vd=0.0, Vt=5.0e5)
radiation = Radiation.init_Radiation_(atmos, wMesh)
SE_con, Rate_con = SELib.cal_SE_with_Nh_Te_(atom, atmos, wMesh, radiation, Nh_SE=None)

Cloud_con = SlabModel.SE_to_slab_0D_(atom, atmos, SE_con, depth=1.0e3 * 1.0e5)  # 1_000 [km]

# Cloud_con.w0[:]             central wavelength in [cm]
# Cloud_con.tau_max[:]        maximum optical depth
# Cloud_con.Ibar[:]           integration of (intensity_profile * wavelength_mesh)

print(f"{'WAVELENGTH [AA]'}  {'MAX TAU'}      {'INTEGRATED INTENSITY [erg/cm^2/Sr/s]'}")
for k in (7, 8, 9):
    print(f"{Cloud_con.w0[k] * 1.0e8:.2f}{' ':10s}{Cloud_con.tau_max[k]:.2E}{' ':5s}{Cloud_con.Ibar[k]:.2E}")
