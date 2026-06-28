"""Smoke-test Grotrian plotting end-to-end for the Hydrogen atom.

Based on the notebook pattern (`notebooks/Hydrogen_atom.ipynb`):
    gro = Grotrian.Grotrian(atom, path_dict["Grotrian"], _scaleFunc, _scaleFunc_inv)
    gro.make_fig(_figsize=(10, 6), _dpi=150, _f=50)
    gro.save_fig(<path>)

Verifies the Phase 3 Path E refactor (`d566a99`) which split `_ensure_fig_`
out of `make_fig` so that `self.fig` is non-Optional after construction
without forcing a render at construction time. Key behaviors exercised:
    - Construct Grotrian (creates an empty figure via `_ensure_fig_`, no render)
    - Call `make_fig` with custom params (closes the old figure, creates a new
      one, then renders — should leave exactly one open figure)
    - Save the rendered figure to disk

Run non-interactively (Agg backend) so it works in CI / headless sessions.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectra.ImportAll import *
from spectra.Struct import Atom
from spectra.Util.AtomUtils import AtomIO as _AtomIO
from spectra.Visual.Grotrian import Grotrian


def scaleFunc(x):
    return x ** (7)


def scaleFunc_inv(x):
    return x ** (1 / 7)


def main():
    conf_path = CFG._ROOT_DIR / "data" / "conf" / "H.conf"
    atom = Atom.init_Atom_(str(conf_path), is_hydrogen=True)
    path_dict = _AtomIO.read_conf_(str(conf_path))

    print(f"[before construct] fignums = {plt.get_fignums()}")
    gro = Grotrian(
        atom,
        path_dict["Grotrian"],
        _scaleFunc=scaleFunc,
        _scaleFunc_inv=scaleFunc_inv,
    )
    # After __init__ → set_atom → _ensure_fig_(), a default figure exists
    # but no levels/axes have been drawn.
    print(f"[after construct]  fignums = {plt.get_fignums()} (expect 1)")
    assert gro.fig is not None, "self.fig must be non-None post-construction"

    gro.make_fig(_figsize=(10, 6), _dpi=150, _f=50)
    # _ensure_fig_ inside make_fig closes the default figure before creating
    # a new one with user-supplied params, so there should still be exactly
    # one open figure — not two (no orphan leak).
    print(f"[after make_fig]   fignums = {plt.get_fignums()} (expect 1, no leak)")

    out_path = CFG._ROOT_DIR / "tmp" / "example.H_Grotrian.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gro.save_fig(str(out_path), _dpi=120)
    print(f"[saved]            {out_path}")

    plt.close("all")


if __name__ == "__main__":
    main()
