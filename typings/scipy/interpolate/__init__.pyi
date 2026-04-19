# Minimal hand-written stub: only the scipy.interpolate.interp1d params this project passes.
# Extend fill_value's union or add kwargs here when new usage patterns appear.
import numpy as np

class interp1d:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        kind: str = ...,
        bounds_error: bool = ...,
        fill_value: float | tuple[float, float] = ...,
    ) -> None: ...
    def __call__(self, x: np.ndarray) -> np.ndarray: ...
