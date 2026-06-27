import json
import os
from pathlib import Path

import numpy as np
import pytest

_REF_PATH = Path(__file__).parent / "reference_values.json"
_ATOM_REF_PATH = Path(__file__).parent / "atom_reference_values.json"


# --- golden-file regeneration -------------------------------------------------
# When REGEN_REFS=1 (driven by scripts/gen_reference.py) an ``assert_close``
# whose ``expected`` is the exact object returned by ``ref[key]`` records the
# freshly computed ``actual`` under ``key`` and skips the comparison;
# reference_values.json is rewritten at the end of the session. Recording is
# keyed on object identity, not a global flag, so a locally computed
# ``expected`` (e.g. a sum-to-one invariant) is never misattributed to a stale
# key -- it falls through and is still asserted, even during the record pass.
# Only keys compared via ``assert_close(actual, ref[key])`` are rewritten; keys
# compared with a direct ``assert x == ref[key]`` (all of RomanUtil, the
# string/int keys of ElementUtil) are verified but not regenerated. Completely
# inert during normal test runs.
def _regen_on() -> bool:
    return os.environ.get("REGEN_REFS") == "1"


_rec: dict = {"key": None, "obj": None, "updates": {}}


class _RecordingRef(dict):
    def __getitem__(self, key):
        # Stash the exact object handed out so assert_close records only when
        # that very object is the ``expected`` it receives. A leaky global flag
        # would let a direct ``assert x == ref[k]`` (which never reaches
        # assert_close) misattribute the next assert_close's value to key ``k``.
        obj = super().__getitem__(key)
        _rec["key"] = key
        _rec["obj"] = obj
        return obj


@pytest.fixture(scope="session")
def ref():
    with _REF_PATH.open() as f:
        data = json.load(f)
    return _RecordingRef(data) if _regen_on() else data


@pytest.fixture(scope="session")
def ref_atom():
    """Load atom_reference_values.json for AtomIO load regression."""
    with _ATOM_REF_PATH.open() as f:
        return json.load(f)


def assert_close(actual, expected, rtol=1e-10, atol=0.0):
    if _regen_on() and expected is _rec["obj"]:
        # ``expected`` is the exact object ref[key] just returned -> record the
        # fresh value and skip; a locally computed ``expected`` is not ``obj``,
        # so it falls through to the real assertion below.
        _rec["updates"][_rec["key"]] = np.asarray(actual).tolist()
        _rec["obj"] = None
        return
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol
    )


def pytest_sessionfinish():
    if not _regen_on() or not _rec["updates"]:
        return
    with _REF_PATH.open() as f:
        data = json.load(f)
    data.update(_rec["updates"])
    _REF_PATH.write_text(json.dumps(data, indent=2))
