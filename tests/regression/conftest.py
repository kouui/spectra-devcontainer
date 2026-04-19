import json
from pathlib import Path

import numpy as np
import pytest

_REF_PATH = Path(__file__).parent / "reference_values.json"
_ATOM_REF_PATH = Path(__file__).parent / "atom_reference_values.json"


@pytest.fixture(scope="session")
def ref():
    with _REF_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="session")
def ref_atom():
    """Load atom_reference_values.json for AtomIO load regression."""
    with _ATOM_REF_PATH.open() as f:
        return json.load(f)


def assert_close(actual, expected, rtol=1e-10, atol=0.0):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
