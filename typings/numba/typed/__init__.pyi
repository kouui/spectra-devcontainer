# Minimal hand-written stub: only the numba.typed.List API used in this project.
# Extend here when new operations are used; do not aim for parity with numba.typed.
from typing import Generic, TypeVar

_T = TypeVar("_T")

class List(Generic[_T]):
    def append(self, item: _T) -> None: ...
