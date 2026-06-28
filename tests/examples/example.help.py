from spectra.ImportAll import *
from spectra.Struct import Atom
from spectra.Util import HelpUtil

conf_path = CFG._ROOT_DIR / "data/conf/H.conf"
atom = Atom.init_Atom_(str(conf_path), is_hydrogen=True)

HelpUtil.help_(atom)
