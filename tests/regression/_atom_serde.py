"""Serialization and assertion helpers for Atom regression tests.

Flattens the `(atom, waveMesh, path_dict)` tuple returned by
`Atom.init_Atom_()` into a flat ``{key: json_value}`` dict suitable for
JSON storage, and provides the reverse comparison used by
``test_reg_e2e_AtomLoad.py``.

Design notes
------------
- ``path_dict`` absolute paths (returned by ``read_conf_`` via
  ``Path.resolve()``) are relativized against ``CFG._ROOT_DIR`` so the
  reference JSON is portable across machines.
- A few numpy arrays inside the returned structs are intentionally
  allocated with ``numpy.empty`` and never populated by the load path
  (``waveMesh.Line_absorb_prof``, ``waveMesh.Line_mesh_share_idxs``).
  Comparing their contents is non-deterministic, so only their ``shape``
  is stored/compared (sentinel form: ``{"_shape_only": [d1, d2, ...]}``).
- ``PI.alpha_table_idxs`` and four ``PI.Coe`` fields (``alpha0``, ``gi``,
  ``gj``, ``dEij``) are uninitialized when the conf file has no PI entry
  (``data_source_PI == CALCULATE``); they are also tracked shape-only in
  that branch, and full-value in the EXPERIMENT branch.
- Struct-array dtypes are not preserved: each named field is stored as a
  plain list keyed by ``<prefix>.<field>``. The comparator restores numpy
  arrays per-field for bit-level assertion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as _np

from spectra import Configurations as CFG

# ---------------------------------------------------------------------------
# dtype field listings — mirror the dtypes declared in
# src/spectra/Util/AtomUtils/AtomIO.py and src/spectra/Struct/WavelengthMesh.py
# ---------------------------------------------------------------------------

_LEVEL_FIELDS: tuple[str, ...] = ("erg", "g", "stage", "gamma", "isGround", "n")
_LINE_FIELDS: tuple[str, ...] = (
    "idxI",
    "idxJ",
    "AJI",
    "f0",
    "w0",
    "w0_AA",
    "Gamma",
    "gi",
    "gj",
    "ni",
    "nj",
    "BJI",
    "BIJ",
)
_CONT_FIELDS: tuple[str, ...] = (
    "idxI",
    "idxJ",
    "f0",
    "w0",
    "w0_AA",
    "gi",
    "gj",
    "ni",
    "nj",
)
_CE_COE_FIELDS: tuple[str, ...] = ("idxI", "idxJ", "f1", "f2", "gi", "gj", "dEij")
_CI_COE_FIELDS: tuple[str, ...] = _CE_COE_FIELDS  # identical dtype
_PI_COE_FIELDS: tuple[str, ...] = (
    "idxI",
    "idxJ",
    "nLambda",
    "alpha0",
    "gi",
    "gj",
    "dEij",
)
_RL_COE_FIELDS: tuple[str, ...] = (
    "idxI",
    "idxJ",
    "lineIndex",
    "ProfileType",
    "qcore",
    "qwing",
    "nLambda",
)
_WAVE_CONT_COE_FIELDS: tuple[str, ...] = ("idxI", "idxJ", "w0", "nLambda")
_WAVE_LINE_COE_FIELDS: tuple[str, ...] = (
    "idxI",
    "idxJ",
    "w0",
    "ProfileType",
    "qcore",
    "qwing",
    "nLambda",
)

# PI.Coe fields that are left uninitialised when data_source_PI == CALCULATE
_PI_COE_UNDEF_ON_CALCULATE: frozenset[str] = frozenset(("alpha0", "gi", "gj", "dEij"))

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _path_to_rel(p: str | None) -> str | None:
    """Absolute path (POSIX or OS-native) → POSIX relative to ``_ROOT_DIR``.

    Returns ``None`` unchanged. Paths outside ``_ROOT_DIR`` fall back to
    ``str(absolute_path)`` — this is a safety net; our 8 canonical configs
    are all inside the repo.
    """
    if p is None:
        return None
    abs_path = Path(p).resolve()
    root = CFG._ROOT_DIR.resolve()
    try:
        return abs_path.relative_to(root).as_posix()
    except ValueError:
        return str(abs_path)


def _shape_only(arr: _np.ndarray) -> dict[str, list[int]]:
    return {"_shape_only": list(arr.shape)}


def _struct_array_as_flat(arr: _np.ndarray, fields: tuple[str, ...], prefix: str) -> dict[str, list]:
    """Return {f"{prefix}.{field}": arr[field].tolist()} for each field."""
    return {f"{prefix}.{f}": arr[f].tolist() for f in fields}


def _enum_name(e: Any) -> str:
    return e.name


# ---------------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------------


def dump_atom(atom: Any, wave_mesh: Any, path_dict: dict[str, str | None], name: str) -> dict[str, Any]:
    """Flatten ``(atom, wave_mesh, path_dict)`` into a JSON-ready dict.

    All keys are prefixed with ``<name>.``.
    """
    out: dict[str, Any] = {}

    # --- scalars / enums ---
    out[f"{name}.Z"] = int(atom.Z)
    out[f"{name}.Mass"] = float(atom.Mass)
    out[f"{name}.Abun"] = float(atom.Abun)
    out[f"{name}.nLevel"] = int(atom.nLevel)
    out[f"{name}.nLine"] = int(atom.nLine)
    out[f"{name}.nCont"] = int(atom.nCont)
    out[f"{name}.nTran"] = int(atom.nTran)
    out[f"{name}.nRL"] = int(atom.nRL)
    out[f"{name}._has_continuum"] = bool(atom._has_continuum)
    out[f"{name}._atom_type"] = _enum_name(atom._atom_type)
    out[f"{name}._atomic_data_source.AJI"] = _enum_name(atom._atomic_data_source.AJI)
    out[f"{name}._atomic_data_source.CE"] = _enum_name(atom._atomic_data_source.CE)
    out[f"{name}._atomic_data_source.CI"] = _enum_name(atom._atomic_data_source.CI)
    out[f"{name}._atomic_data_source.PI"] = _enum_name(atom._atomic_data_source.PI)

    # --- path_dict (relativized) ---
    for k, v in path_dict.items():
        out[f"{name}.path_dict.{k}"] = _path_to_rel(v)

    # --- Level / Line / Cont (struct arrays) ---
    out.update(_struct_array_as_flat(atom.Level, _LEVEL_FIELDS, f"{name}.Level"))
    out.update(_struct_array_as_flat(atom.Line, _LINE_FIELDS, f"{name}.Line"))
    out.update(_struct_array_as_flat(atom.Cont, _CONT_FIELDS, f"{name}.Cont"))

    # --- CE ---
    out[f"{name}.CE._transition_type"] = _enum_name(atom.CE._transition_type)
    out[f"{name}.CE._transition_source"] = _enum_name(atom.CE._transition_source)
    out[f"{name}.CE._transition_formula"] = _enum_name(atom.CE._transition_formula)
    out[f"{name}.CE.Te_table"] = atom.CE.Te_table.tolist()
    out[f"{name}.CE.Omega_table"] = atom.CE.Omega_table.tolist()
    out.update(_struct_array_as_flat(atom.CE.Coe, _CE_COE_FIELDS, f"{name}.CE.Coe"))

    # --- CI ---
    out[f"{name}.CI._transition_type"] = _enum_name(atom.CI._transition_type)
    out[f"{name}.CI._transition_source"] = _enum_name(atom.CI._transition_source)
    out[f"{name}.CI._transition_formula"] = _enum_name(atom.CI._transition_formula)
    out[f"{name}.CI.Te_table"] = atom.CI.Te_table.tolist()
    out[f"{name}.CI.Omega_table"] = atom.CI.Omega_table.tolist()
    out.update(_struct_array_as_flat(atom.CI.Coe, _CI_COE_FIELDS, f"{name}.CI.Coe"))

    # --- PI ---
    # When data_source_PI == CALCULATE and nCont > 0, several arrays are
    # allocated via numpy.empty and never assigned in the hydrogenic path.
    pi_is_calc = atom._atomic_data_source.PI.name == "CALCULATE"
    pi_has_uninit = pi_is_calc and int(atom.nCont) > 0

    out[f"{name}.PI.alpha_table"] = atom.PI.alpha_table.tolist()
    out[f"{name}.PI.alpha_interp"] = atom.PI.alpha_interp.tolist()
    out[f"{name}.PI.alpha_table_idxs"] = (
        _shape_only(atom.PI.alpha_table_idxs) if pi_has_uninit else atom.PI.alpha_table_idxs.tolist()
    )
    for f in _PI_COE_FIELDS:
        key = f"{name}.PI.Coe.{f}"
        if pi_has_uninit and f in _PI_COE_UNDEF_ON_CALCULATE:
            out[key] = _shape_only(atom.PI.Coe[f])
        else:
            out[key] = atom.PI.Coe[f].tolist()

    # --- RL ---
    out[f"{name}.RL.nRadiativeLine"] = int(atom.RL.nRadiativeLine)
    out.update(_struct_array_as_flat(atom.RL.Coe, _RL_COE_FIELDS, f"{name}.RL.Coe"))

    # --- ctj / idx tables (nested tuples → nested lists) ---
    out[f"{name}._ctj_table.Level"] = [list(row) for row in atom._ctj_table.Level]
    out[f"{name}._ctj_table.Line"] = [[list(a), list(b)] for (a, b) in atom._ctj_table.Line]
    out[f"{name}._ctj_table.Cont"] = [[list(a), list(b)] for (a, b) in atom._ctj_table.Cont]
    out[f"{name}._idx_table.Line"] = [list(p) for p in atom._idx_table.Line]
    out[f"{name}._idx_table.Cont"] = [list(p) for p in atom._idx_table.Cont]

    # --- Wavelength_Mesh ---
    out[f"{name}.waveMesh.Cont_mesh"] = wave_mesh.Cont_mesh.tolist()
    out.update(_struct_array_as_flat(wave_mesh.Cont_Coe, _WAVE_CONT_COE_FIELDS, f"{name}.waveMesh.Cont_Coe"))
    out[f"{name}.waveMesh.Line_mesh"] = wave_mesh.Line_mesh.tolist()
    out[f"{name}.waveMesh.Line_mesh_idxs"] = wave_mesh.Line_mesh_idxs.tolist()
    out.update(_struct_array_as_flat(wave_mesh.Line_Coe, _WAVE_LINE_COE_FIELDS, f"{name}.waveMesh.Line_Coe"))
    out[f"{name}.waveMesh.Line_mesh_share"] = wave_mesh.Line_mesh_share.tolist()
    # Uninitialised memory — shape-only comparison.
    out[f"{name}.waveMesh.Line_absorb_prof"] = _shape_only(wave_mesh.Line_absorb_prof)
    out[f"{name}.waveMesh.Line_mesh_share_idxs"] = _shape_only(wave_mesh.Line_mesh_share_idxs)

    return out


def _compare_value(key: str, actual: Any, expected: Any) -> None:
    """Per-key type-dispatched assertion with informative message on failure.

    ``actual`` is always produced by ``dump_atom`` (same shape as ``expected``),
    so shape-only sentinels appear on both sides in that branch.
    """
    # shape-only sentinel: both sides are sentinels, compare stored shapes
    if isinstance(expected, dict) and "_shape_only" in expected:
        assert isinstance(actual, dict), f"{key}: expected shape-only sentinel, got {type(actual).__name__}"
        assert "_shape_only" in actual, f"{key}: expected shape-only sentinel, got dict without key"
        assert actual["_shape_only"] == expected["_shape_only"], (
            f"{key}: shape mismatch, got {actual['_shape_only']}, expected {expected['_shape_only']}"
        )
        return

    # None
    if expected is None:
        assert actual is None, f"{key}: expected None, got {actual!r}"
        return

    # bool must be checked before int (isinstance(True, int) is True in Python)
    if isinstance(expected, bool):
        assert isinstance(actual, bool), f"{key}: expected bool, got {type(actual).__name__}"
        assert actual == expected, f"{key}: {actual!r} != {expected!r}"
        return

    if isinstance(expected, int):
        assert actual == expected, f"{key}: {actual!r} != {expected!r}"
        return

    if isinstance(expected, float):
        assert _np.isclose(actual, expected, rtol=1e-12, atol=0.0, equal_nan=True), f"{key}: {actual!r} != {expected!r}"
        return

    if isinstance(expected, str):
        assert actual == expected, f"{key}: {actual!r} != {expected!r}"
        return

    if isinstance(expected, list):
        _compare_list(key, actual, expected)
        return

    raise TypeError(f"{key}: unsupported expected type {type(expected).__name__}")


def _leaf_is_str(obj: Any) -> bool:
    """Recursively peek into nested lists; True iff the leaf element is a str."""
    while isinstance(obj, list):
        if not obj:
            return False
        obj = obj[0]
    return isinstance(obj, str)


def _compare_list(key: str, actual: Any, expected: list) -> None:
    """Dispatch list comparison.

    ``expected`` came from ``.tolist()`` or from serialized ctj/idx tables, so
    it's one of: (a) empty, (b) (nested) list with string leaves
    (ctj tables), (c) numeric (nested) list.
    """
    if not expected:
        actual_np = _np.asarray(actual)
        assert actual_np.size == 0, f"{key}: expected empty, got {actual!r}"
        return

    if _leaf_is_str(expected):
        assert actual == expected, f"{key}: value mismatch (string leaves)"
        return

    actual_np = _np.asarray(actual)
    expected_np = _np.asarray(expected)
    assert actual_np.shape == expected_np.shape, f"{key}: shape {actual_np.shape} != {expected_np.shape}"
    if expected_np.dtype.kind in ("i", "u", "b"):
        assert _np.array_equal(actual_np, expected_np), f"{key}: int/bool array differs"
    else:
        _np.testing.assert_allclose(actual_np, expected_np, rtol=1e-12, atol=0.0, err_msg=key)


def assert_atom_matches(
    atom: Any,
    wave_mesh: Any,
    path_dict: dict[str, str | None],
    name: str,
    ref: dict[str, Any],
) -> None:
    """Assert that the loaded atom matches the reference snapshot for ``name``."""
    actual = dump_atom(atom, wave_mesh, path_dict, name)

    name_prefix = f"{name}."
    expected_keys = {k for k in ref if k.startswith(name_prefix)}
    actual_keys = set(actual)

    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    assert not missing, f"{name}: keys missing in actual: {sorted(missing)}"
    assert not extra, f"{name}: keys present in actual but not in ref: {sorted(extra)}"

    for key in sorted(expected_keys):
        _compare_value(key, actual[key], ref[key])


def load_reference(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)
