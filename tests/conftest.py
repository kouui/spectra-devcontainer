from pathlib import Path

from numba.core import config as _nb_config
import pytest

# Directories whose tests run iterative solvers (many formal solutions per
# iteration). Under NUMBA_DISABLE_JIT=1 every depth loop runs as python, and
# these directories take ~98% of a ~7 min run while everything else stays
# under 1 s per test. A directory listed here is marked, not edited per file;
# add new solver test directories here.
_SLOW_WITHOUT_JIT_DIRS = (
    Path(__file__).parent / "unittest" / "MALI_v1",
    Path(__file__).parent / "unittest" / "MALI_v2",
)


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow-without-jit",
        action="store_true",
        default=False,
        help="run tests marked slow_without_jit even when NUMBA_DISABLE_JIT=1",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow_without_jit: minutes instead of seconds when NUMBA_DISABLE_JIT=1; skipped in that mode "
        "unless --run-slow-without-jit",
    )


def pytest_collection_modifyitems(config, items):
    run_anyway = config.getoption("--run-slow-without-jit")
    skip = pytest.mark.skip(reason="slow_without_jit under NUMBA_DISABLE_JIT=1 (use --run-slow-without-jit)")
    for item in items:
        if any(item.path.is_relative_to(d) for d in _SLOW_WITHOUT_JIT_DIRS):
            item.add_marker(pytest.mark.slow_without_jit)
        if _nb_config.DISABLE_JIT and not run_anyway and item.get_closest_marker("slow_without_jit"):
            item.add_marker(skip)
