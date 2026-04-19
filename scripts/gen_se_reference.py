"""Generate the 18 new Stage A SE e2e reference keys.

Runs the three public SELib entries (``cal_SE_with_Nh_Te_``,
``cal_SE_with_Ne_Te_``, ``cal_SE_with_Pg_Te_``) across H, He, and Ca_II,
extracts the asserted fields per ``tests/regression/test_reg_e2e_SE.py``, and
merges them into ``tests/regression/reference_values.json`` in place.

The existing file is not sort-ordered, so ``sort_keys=False`` is used to
preserve the historic ad-hoc key order; new keys are appended at the tail.
This keeps the diff surgical (only the 18 added keys appear).

Idempotent: re-running overwrites the 18 keys with identical values (assuming
deterministic inputs). Other keys are preserved.

To regenerate on the behavior-locking commit, run this script inside a
throwaway worktree checked out at that commit.
"""

from __future__ import annotations

import json
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

# One record per new test case. ``fields`` lists the fields asserted by the
# corresponding test method — these are the ONLY keys written per case.
# Keep in sync with tests/regression/test_reg_e2e_SE.py.
CASES: tuple[dict[str, Any], ...] = (
    {
        "key": "H_SE_Pg_Te",
        "conf": "data/conf/H.conf",
        "is_hydrogen": True,
        "entry": "cal_SE_with_Pg_Te_",
        "atmos_kwargs": {"Pg": 1.8, "Nh": 1.0e12, "Ne": 1.0e11, "Te": 7.0e3, "Vd": 0.0, "Vt": 5.0e5},
        "fields": ("n_SE", "n_LTE", "Ne", "Ntotal"),
    },
    {
        "key": "He_SE_Nh_Te",
        "conf": "data/conf/He.conf",
        "is_hydrogen": False,
        "entry": "cal_SE_with_Nh_Te_",
        "atmos_kwargs": {"Nh": 1.0e12, "Ne": 1.0e11, "Te": 7.0e3, "Vd": 0.0, "Vt": 5.0e5},
        "fields": ("n_SE", "n_LTE", "Ntotal"),
    },
    {
        "key": "He_SE_Pg_Te",
        "conf": "data/conf/He.conf",
        "is_hydrogen": False,
        "entry": "cal_SE_with_Pg_Te_",
        "atmos_kwargs": {"Pg": 1.8, "Nh": 1.0e12, "Ne": 1.0e11, "Te": 7.0e3, "Vd": 0.0, "Vt": 5.0e5},
        "fields": ("n_SE", "n_LTE", "Ntotal"),
    },
    {
        "key": "Ca_II_SE_Nh_Te",
        "conf": "data/conf/Ca_II.conf",
        "is_hydrogen": False,
        "entry": "cal_SE_with_Nh_Te_",
        "atmos_kwargs": {"Nh": 1.0e12, "Ne": 1.0e11, "Te": 7.0e3, "Vd": 0.0, "Vt": 5.0e5},
        "fields": ("n_SE", "n_LTE", "Ntotal"),
    },
    {
        "key": "Ca_II_SE_Ne_Te",
        "conf": "data/conf/Ca_II.conf",
        "is_hydrogen": False,
        "entry": "cal_SE_with_Ne_Te_",
        "atmos_kwargs": {"Nh": 1.0e11, "Ne": 5.0e10, "Te": 7.0e3, "Vd": 0.0, "Vt": 5.0e5},
        "fields": ("n_SE", "n_LTE"),
    },
    {
        "key": "Ca_II_SE_Pg_Te",
        "conf": "data/conf/Ca_II.conf",
        "is_hydrogen": False,
        "entry": "cal_SE_with_Pg_Te_",
        "atmos_kwargs": {"Pg": 1.8, "Nh": 1.0e12, "Ne": 1.0e11, "Te": 7.0e3, "Vd": 0.0, "Vt": 5.0e5},
        "fields": ("n_SE", "n_LTE", "Ntotal"),
    },
)


def _extract_field(case_key: str, field: str, SE_con: Any, atmos: Any) -> Any:
    """Mirror the assertion's source: `atmos.Ne` for H x Pg_Te, SE_con.* otherwise."""
    if field == "Ne":
        # Only H x Pg_Te asserts atmos.Ne (it is iterated by the self-consistent
        # loop). Stored as a scalar float.
        assert case_key == "H_SE_Pg_Te", f"unexpected Ne assertion for {case_key}"
        return float(atmos.Ne)
    value = getattr(SE_con, field)
    if hasattr(value, "tolist"):
        return value.tolist()
    return float(value)


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    conf_path = str(CFG._ROOT_DIR / case["conf"])
    atom, wMesh, _ = Atom.init_Atom_(conf_path, is_hydrogen=case["is_hydrogen"])
    atmos = Atmosphere.Atmosphere0D(**case["atmos_kwargs"])
    radiation = Radiation.init_Radiation_(atmos, wMesh)

    entry = getattr(SELib, case["entry"])
    SE_con, _ = entry(atom, atmos, wMesh, radiation, None)

    key_prefix = f"E2E.{case['key']}"
    return {f"{key_prefix}.{field}": _extract_field(case["key"], field, SE_con, atmos) for field in case["fields"]}


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT

    with out.open() as f:
        merged: dict[str, Any] = json.load(f)

    before = len(merged)
    for case in CASES:
        new_keys = _run_case(case)
        n_sum = sum(new_keys[f"E2E.{case['key']}.n_SE"])
        print(
            f"generated E2E.{case['key']:20s}  n_SE.sum()={n_sum:.6f}  (+{len(new_keys)} keys)",
            flush=True,
        )
        merged.update(new_keys)

    with out.open("w") as f:
        json.dump(merged, f, indent=2, sort_keys=False)

    after = len(merged)
    print(f"wrote {out}  ({after} keys total, +{after - before} new)")


if __name__ == "__main__":
    main()
