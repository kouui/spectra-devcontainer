"""Generate the 18 new Stage A SE e2e reference keys.

Runs the three public SELib entries (``cal_SE_with_Nh_Te_``,
``cal_SE_with_Ne_Te_``, ``cal_SE_with_Pg_Te_``) across H, He, and Ca_II,
extracts the asserted fields per ``tests/regression/test_reg_e2e_SE.py``, and
merges them into ``tests/regression/reference_values.json`` in place.

The existing file is not sort-ordered, so ``sort_keys=False`` is used to
preserve the historic ad-hoc key order; new keys are appended at the tail.
This keeps the diff surgical (only the 18 added keys appear).

Re-running the script overwrites the 18 keys with identical values given
deterministic inputs. Keys introduced by earlier runs but no longer in the
CASES table are NOT pruned — the script only inserts/overwrites.

To regenerate on the behavior-locking commit, run this script inside a
throwaway worktree checked out at that commit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from spectra import Configurations as CFG  # noqa: E402
from spectra.Function.SEquil import SELib  # noqa: E402
from spectra.Struct import Atmosphere, Atom, Radiation  # noqa: E402

DEFAULT_OUT = CFG._ROOT_DIR / "tests" / "regression" / "reference_values.json"

# n_SE over all levels must sum to 1 by conservation. Allow ±5% slack: tighter
# than the Pg_Te inner-loop tolerance (~1%) so any NaN / divergence trips this,
# loose enough that legitimate normalisation noise from LU solves does not.
_N_SE_SUM_TOL = 0.05

# One record per new test case. ``fields`` lists the fields asserted by the
# corresponding test method — these are the ONLY keys written per case.
# Keep in sync with tests/regression/test_reg_e2e_SE.py.
CASES: tuple[dict[str, Any], ...] = (
    {
        "key": "H_SE_Pg_Te",
        "conf": "data/conf/H.conf",
        "is_hydrogen": True,
        "entry": "cal_SE_with_Pg_Te_",
        "atmos_kwargs": {"Pg": 1.8, "Nh": 1.0e12, "Ne": 1.0e11, "Te": 7.0e3, "Vt": 5.0e5},
        "fields": ("n_SE", "n_LTE", "Ne", "Ntotal"),
    },
    {
        "key": "He_SE_Nh_Te",
        "conf": "data/conf/He.conf",
        "is_hydrogen": False,
        "entry": "cal_SE_with_Nh_Te_",
        "atmos_kwargs": {"Nh": 1.0e12, "Ne": 1.0e11, "Te": 7.0e3, "Vt": 5.0e5},
        "fields": ("n_SE", "n_LTE", "Ntotal"),
    },
    {
        "key": "He_SE_Pg_Te",
        "conf": "data/conf/He.conf",
        "is_hydrogen": False,
        "entry": "cal_SE_with_Pg_Te_",
        "atmos_kwargs": {"Pg": 1.8, "Nh": 1.0e12, "Ne": 1.0e11, "Te": 7.0e3, "Vt": 5.0e5},
        "fields": ("n_SE", "n_LTE", "Ntotal"),
    },
    {
        "key": "Ca_II_SE_Nh_Te",
        "conf": "data/conf/Ca_II.conf",
        "is_hydrogen": False,
        "entry": "cal_SE_with_Nh_Te_",
        "atmos_kwargs": {"Nh": 1.0e12, "Ne": 1.0e11, "Te": 7.0e3, "Vt": 5.0e5},
        "fields": ("n_SE", "n_LTE", "Ntotal"),
    },
    {
        "key": "Ca_II_SE_Ne_Te",
        "conf": "data/conf/Ca_II.conf",
        "is_hydrogen": False,
        "entry": "cal_SE_with_Ne_Te_",
        "atmos_kwargs": {"Nh": 1.0e11, "Ne": 5.0e10, "Te": 7.0e3, "Vt": 5.0e5},
        "fields": ("n_SE", "n_LTE"),
    },
    {
        "key": "Ca_II_SE_Pg_Te",
        "conf": "data/conf/Ca_II.conf",
        "is_hydrogen": False,
        "entry": "cal_SE_with_Pg_Te_",
        "atmos_kwargs": {"Pg": 1.8, "Nh": 1.0e12, "Ne": 1.0e11, "Te": 7.0e3, "Vt": 5.0e5},
        "fields": ("n_SE", "n_LTE", "Ntotal"),
    },
)


def _run_case_(case: dict[str, Any]) -> dict[str, Any]:
    """Run one SE case and return its reference-key dict.

    Only H_SE_Pg_Te sources `Ne` from `atmos` (the self-consistent loop
    overwrites `atmos.Ne`); every other (case, field) pair reads from `SE_con`
    so that the reference reflects exactly what the matching test method
    asserts. A hard sanity gate on n_SE.sum() prevents a silent NaN or
    divergent run from being written to disk.
    """
    conf_path = str(CFG._ROOT_DIR / case["conf"])
    atom, _ = Atom.init_Atom_(conf_path, is_hydrogen=case["is_hydrogen"])
    atmos = Atmosphere.Atmosphere0D(**case["atmos_kwargs"])
    radiation = Radiation.init_Radiation_()

    entry = getattr(SELib, case["entry"])
    SE_con, _ = entry(atom, atmos, radiation, None)

    n_sum = float(SE_con.n_SE.sum())
    if not math.isfinite(n_sum) or abs(n_sum - 1.0) > _N_SE_SUM_TOL:
        raise RuntimeError(
            f"{case['key']}: n_SE.sum()={n_sum:.6f} fails sanity check "
            f"(expected finite value within {_N_SE_SUM_TOL:.0%} of 1.0)"
        )

    key_prefix = f"E2E.{case['key']}"
    new_keys: dict[str, Any] = {}
    for field in case["fields"]:
        if field == "Ne" and case["key"] == "H_SE_Pg_Te":
            new_keys[f"{key_prefix}.{field}"] = float(atmos.Ne)
            continue
        value = getattr(SE_con, field)
        new_keys[f"{key_prefix}.{field}"] = value.tolist() if hasattr(value, "tolist") else float(value)

    print(
        f"generated E2E.{case['key']:20s}  n_SE.sum()={n_sum:.6f}  (+{len(new_keys)} keys)",
        flush=True,
    )
    return new_keys


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT

    with out.open() as f:
        merged: dict[str, Any] = json.load(f)

    before = len(merged)
    for case in CASES:
        merged.update(_run_case_(case))

    with out.open("w") as f:
        json.dump(merged, f, indent=2, sort_keys=False)

    after = len(merged)
    print(f"wrote {out}  ({after} keys total, +{after - before} new)")


if __name__ == "__main__":
    main()
