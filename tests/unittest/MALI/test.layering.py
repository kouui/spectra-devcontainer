"""Layering contract of the MALI package, checked on the import statements.

- RadiativeTransfer/MALI/Kernels.py takes plain arrays: it imports no input
  struct (Struct) and no driver (Function), so it can be reused by any caller
  and tested without building an atom.
- the frozen Experimental.MALI copies are reference material for regression
  tests only; library code outside Experimental must not depend on them.
"""

import ast
from pathlib import Path

import spectra

SRC = Path(spectra.__file__).parent
KERNELS = SRC / "RadiativeTransfer" / "MALI" / "Kernels.py"
FROZEN = "spectra.Experimental.MALI"


def _module_name(path):
    return ".".join(("spectra", *path.relative_to(SRC).with_suffix("").parts))


def _imported_modules(path):
    """Absolute dotted names of every module (and imported submodule) in `path`."""
    parts = _module_name(path).split(".")
    package = parts[:-1]  # `__init__` is the last part of a package's own module name
    for node in ast.walk(ast.parse(path.read_text())):
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


def _is_under(name, prefix):
    return name == prefix or name.startswith(prefix + ".")


def test_kernels_import_no_struct_or_function():
    imported = set(_imported_modules(KERNELS))
    assert imported, "the import resolver found nothing in Kernels.py"
    bad = sorted(m for m in imported if _is_under(m, "spectra.Struct") or _is_under(m, "spectra.Function"))
    assert bad == []


def test_nothing_outside_experimental_imports_frozen_mali():
    offenders = []
    for path in SRC.rglob("*.py"):
        if "Experimental" in path.relative_to(SRC).parts:
            continue
        offenders += [f"{path.relative_to(SRC)} -> {m}" for m in _imported_modules(path) if _is_under(m, FROZEN)]
    assert offenders == []


def test_import_resolver_handles_relative_imports():
    # guards the two tests above against a resolver that silently yields nothing
    imported = set(_imported_modules(KERNELS))
    assert "spectra.Atomic.SEsolver" in imported
    assert "spectra.RadiativeTransfer.Feautrier" in imported
