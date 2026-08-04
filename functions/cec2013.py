import numpy as np

from learning_ddpg.functions.functions import CEC_functions


CEC2013_OPTIMA = {
    1: -1400.0,
    2: -1300.0,
    3: -1200.0,
    4: -1100.0,
    5: -1000.0,
    6: -900.0,
    7: -800.0,
    8: -700.0,
    9: -600.0,
    10: -500.0,
    11: -400.0,
    12: -300.0,
    13: -200.0,
    14: -100.0,
    15: 100.0,
    16: 200.0,
    17: 300.0,
    18: 400.0,
    19: 500.0,
    20: 600.0,
    21: 700.0,
    22: 800.0,
    23: 900.0,
    24: 1000.0,
    25: 1100.0,
    26: 1200.0,
    27: 1300.0,
    28: 1400.0,
}


class CEC2013Objective:
    """Batch adapter around the original one-vector CEC2013 implementation."""

    def __init__(self, dimensions, function_id):
        self.dimensions = int(dimensions)
        self.function_id = int(function_id)
        if self.function_id not in CEC2013_OPTIMA:
            raise ValueError(f"Unknown CEC2013 function: F{self.function_id}")

        self.optimum = CEC2013_OPTIMA[self.function_id]
        self._cec = CEC_functions(self.dimensions)

    @property
    def shift(self):
        return self._cec.O.copy()

    def __call__(self, positions):
        positions = np.asarray(positions, dtype=np.float64)

        if positions.ndim == 1:
            if positions.shape != (self.dimensions,):
                raise ValueError(
                    f"Expected shape ({self.dimensions},), got {positions.shape}"
                )
            return float(self._cec.Y(positions, self.function_id))

        if positions.ndim != 2 or positions.shape[1] != self.dimensions:
            raise ValueError(
                f"Expected shape (batch, {self.dimensions}), got {positions.shape}"
            )

        return np.asarray(
            [self._cec.Y(row, self.function_id) for row in positions],
            dtype=np.float64,
        )
