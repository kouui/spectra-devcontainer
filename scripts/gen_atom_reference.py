"""Generate tests/regression/atom_reference_values.json.

One-shot helper that loads 8 canonical atom configurations through
``Atom.init_Atom_()`` and writes a flat ``{key: value}`` JSON file used by
``tests/regression/test_reg_e2e_AtomLoad.py`` for round-trip assertion.

Regenerate whenever ``AtomIO`` business logic changes intentionally.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent
# Let the one-shot script import the test-only helper without making
# `tests/regression/` a package on sys.path.
sys.path.insert(0, str(_REPO_ROOT / "tests" / "regression"))

from _atom_serde import dump_atom  # noqa: E402  # pyright: ignore[reportMissingImports]

from spectra import Configurations as CFG  # noqa: E402
from spectra.Struct import Atom  # noqa: E402
from spectra.Util.AtomUtils import AtomIO as _AtomIO  # noqa: E402

# (relative conf path from repo root, readable name, is_hydrogen)
CONFIGS: tuple[tuple[str, str, bool], ...] = (
    ("data/conf/H.conf", "H", True),
    ("data/conf/H6.conf", "H6", True),
    ("data/conf/H_theory.conf", "H_theory", True),
    ("data/conf/He.conf", "He", False),
    ("data/conf/He_I.conf", "He_I", False),
    ("data/conf/He_I_II.conf", "He_I_II", False),
    ("data/conf/Ca_II.conf", "Ca_II", False),
    ("data/conf/Ca_I-II-III.conf", "Ca_I-II-III", False),
)

DEFAULT_OUT = CFG._ROOT_DIR / "tests" / "regression" / "atom_reference_values.json"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT

    merged: dict[str, Any] = {}
    for rel, name, is_H in CONFIGS:
        conf_path = str(CFG._ROOT_DIR / rel)
        print(f"loading {name}  <- {rel}", flush=True)
        atom = Atom.init_Atom_(conf_path, is_hydrogen=is_H)
        path_dict = _AtomIO.read_conf_(conf_path)
        merged.update(dump_atom(atom, path_dict, name))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
    print(f"wrote {out}  ({len(merged)} keys, {out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
