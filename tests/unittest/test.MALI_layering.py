"""Layering contract of the MALI package.

The checks read import statements only (``ast``); they do not follow imports
transitively. The test is pure ast and runs without numba, so it sits outside
tests/unittest/MALI, whose tests are skipped under NUMBA_DISABLE_JIT=1.

- RadiativeTransfer/MALI takes plain arrays: no module there imports an input
  struct (Struct) or a driver (Function), so the kernels can be reused by any
  caller and tested without building an atom.
- the frozen Experimental.MALI copies are reference material for regression
  tests only; library code outside Experimental must not depend on them.
"""

import ast
from pathlib import Path
import textwrap

import pytest

import spectra

SRC = Path(spectra.__file__).parent
KERNEL_PKG = SRC / "RadiativeTransfer" / "MALI"
KERNEL_MODULES = sorted(KERNEL_PKG.glob("*.py"))
FROZEN = "spectra.Experimental.MALI"


def _module_name(path):
    return ".".join(("spectra", *path.relative_to(SRC).with_suffix("").parts))


def _imported_names(source, module_name):
    """Absolute dotted names of every module (and imported submodule) in `source`.

    `module_name` is the dotted name of the module the source belongs to; a
    package's own module name ends in `__init__`, so dropping the last part
    gives the package that relative imports start from in both cases.
    """
    package = module_name.split(".")[:-1]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                anchor = package[: len(package) - (node.level - 1)]
                base = ".".join([*anchor, *([node.module] if node.module else [])])
            yield base
            for alias in node.names:
                yield f"{base}.{alias.name}"


def _imported_modules(path):
    return set(_imported_names(path.read_text(), _module_name(path)))


def _is_under(name, prefix):
    return name == prefix or name.startswith(prefix + ".")


def test_import_resolver_on_synthetic_source():
    # guards the checks below against a resolver that silently yields nothing or resolves wrongly
    source = textwrap.dedent(
        """
        import os.path
        from pkg.top import thing
        from . import sibling
        from .. import cousin
        from ..uncle import niece
        from ...root import leaf
        """
    )
    assert set(_imported_names(source, "spectra.A.B.mod")) == {
        "os.path",
        "pkg.top",
        "pkg.top.thing",
        "spectra.A.B",
        "spectra.A.B.sibling",
        "spectra.A",
        "spectra.A.cousin",
        "spectra.A.uncle",
        "spectra.A.uncle.niece",
        "spectra.root",
        "spectra.root.leaf",
    }
    assert set(_imported_names("from . import x", "spectra.A.__init__")) == {"spectra.A", "spectra.A.x"}


def test_kernel_package_is_not_empty():
    assert KERNEL_MODULES


@pytest.mark.parametrize("path", KERNEL_MODULES, ids=lambda p: p.name)
def test_kernel_package_imports_no_struct_or_function(path):
    bad = sorted(
        m for m in _imported_modules(path) if _is_under(m, "spectra.Struct") or _is_under(m, "spectra.Function")
    )
    assert bad == []


def test_nothing_outside_experimental_imports_frozen_mali():
    offenders = []
    for path in SRC.rglob("*.py"):
        if "Experimental" in path.relative_to(SRC).parts:
            continue
        offenders += [f"{path.relative_to(SRC)} -> {m}" for m in _imported_modules(path) if _is_under(m, FROZEN)]
    assert offenders == []
