from typing import Any

import numpy as np

class interp1d:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        kind: str = ...,
        axis: int = ...,
        copy: bool = ...,
        bounds_error: bool = ...,
        fill_value: float | tuple[float, float] | str = ...,
        assume_sorted: bool = ...,
    ) -> None: ...
    def __call__(self, x: np.ndarray) -> np.ndarray: ...
