"""Package-level contract of spectra.Experimental.MALI.v2: versions never
import each other -- what v2 reuses from v1 is copied, so v1 stays frozen for
the experiments that depend on it."""

import ast
from pathlib import Path

import spectra.Experimental.MALI.v2 as v2

PKG = Path(v2.__file__).parent


def _imported_modules(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            # relative imports: reconstruct the dotted target for the check
            yield "." * node.level + (node.module or "")


def test_no_cross_version_import():
    for path in PKG.glob("*.py"):
        for mod in _imported_modules(path):
            parts = mod.lstrip(".").split(".")
            assert "v1" not in parts, f"{path.name}: {mod}"
