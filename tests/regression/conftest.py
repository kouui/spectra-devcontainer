import json
import os
from pathlib import Path

import numpy as np
import pytest

_REF_PATH = Path(__file__).parent / "reference_values.json"
_ATOM_REF_PATH = Path(__file__).parent / "atom_reference_values.json"


# --- golden-file regeneration -------------------------------------------------
# When REGEN_REFS=1 (driven by scripts/gen_reference.py) every ``assert_close``
# records the freshly computed value under the ref key it is compared against,
# the assertion is skipped, and reference_values.json is rewritten at the end of
# the session. This lets the golden file be rebuilt from the exact inputs the
# tests already define, with no separate input table to keep in sync. Completely
# inert during normal test runs.
def _regen_on() -> bool:
    return os.environ.get("REGEN_REFS") == "1"


_rec: dict = {"key": None, "fresh": False, "updates": {}}


class _RecordingRef(dict):
    def __getitem__(self, key):
        _rec["key"] = key
        _rec["fresh"] = True
        return super().__getitem__(key)


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
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if _regen_on():
        # Record only when ``expected`` came straight from ref[key] for THIS
        # call; asserts that compare against a locally computed value (no ref
        # access) leave ``fresh`` False and must not overwrite a stored key.
        if _rec["fresh"]:
            _rec["updates"][_rec["key"]] = actual.tolist()
            _rec["fresh"] = False
        return
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def pytest_sessionfinish():
    if not _regen_on() or not _rec["updates"]:
        return
    with _REF_PATH.open() as f:
        data = json.load(f)
    data.update(_rec["updates"])
    _REF_PATH.write_text(json.dumps(data, indent=2))
