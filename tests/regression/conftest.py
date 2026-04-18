import json
from pathlib import Path

import numpy as np
import pytest

_REF_PATH = Path(__file__).parent / "reference_values.json"


@pytest.fixture(scope="session")
def ref():
    with open(_REF_PATH) as f:
        return json.load(f)


def assert_close(actual, expected, rtol=1e-10, atol=0.0):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
