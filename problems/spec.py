from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProblemSpec:
    suite: str
    problem_id: object
    name: str
    dimensions: int
    lower_bound: object
    upper_bound: object
    optimum: float
    objective: object

    def __post_init__(self):
        if not isinstance(self.suite, str) or not self.suite.strip():
            raise ValueError(
                f"suite must be a non-empty string, got {self.suite!r}"
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                f"name must be a non-empty string, got {self.name!r}"
            )
        if (
            isinstance(self.dimensions, (bool, np.bool_))
            or not isinstance(self.dimensions, (int, np.integer))
            or self.dimensions <= 0
        ):
            raise ValueError(
                "dimensions must be a positive integer, "
                f"got {self.dimensions!r}"
            )
        dimensions = int(self.dimensions)

        lower_bound = self._normalize_bound(
            "lower_bound",
            self.lower_bound,
            dimensions,
        )
        upper_bound = self._normalize_bound(
            "upper_bound",
            self.upper_bound,
            dimensions,
        )
        if np.any(lower_bound >= upper_bound):
            raise ValueError(
                "lower_bound must be smaller than upper_bound in every "
                f"dimension, got lower_bound={lower_bound!r}, "
                f"upper_bound={upper_bound!r}"
            )

        optimum_array = np.asarray(self.optimum, dtype=np.float64)
        if optimum_array.shape != () or not np.isfinite(optimum_array):
            raise ValueError(
                f"optimum must be a finite scalar, got {self.optimum!r}"
            )
        if not callable(self.objective):
            raise ValueError(
                f"objective must be callable, got {self.objective!r}"
            )

        object.__setattr__(self, "suite", self.suite.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "lower_bound", lower_bound)
        object.__setattr__(self, "upper_bound", upper_bound)
        object.__setattr__(self, "optimum", float(optimum_array))

    @staticmethod
    def _normalize_bound(name, value, dimensions):
        try:
            bound = np.asarray(value, dtype=np.float64)
            bound = np.broadcast_to(bound, (dimensions,)).copy()
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name}={value!r} cannot be broadcast to ({dimensions},)"
            ) from error
        if not np.all(np.isfinite(bound)):
            raise ValueError(
                f"{name} must contain only finite values, got {value!r}"
            )
        return bound

    def evaluate(self, positions):
        positions_array = np.asarray(positions, dtype=np.float64)
        expected_format = f"(n, {self.dimensions})"
        if (
            positions_array.ndim != 2
            or positions_array.shape[1] != self.dimensions
        ):
            raise ValueError(
                f"positions has shape {positions_array.shape}, "
                f"expected {expected_format}"
            )
        if not np.all(np.isfinite(positions_array)):
            raise FloatingPointError(
                "positions must contain only finite values"
            )

        values = np.asarray(
            self.objective(positions_array),
            dtype=np.float64,
        )
        expected_shape = (positions_array.shape[0],)
        if values.shape != expected_shape:
            raise ValueError(
                f"objective returned shape {values.shape}, "
                f"expected {expected_shape}"
            )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(
                "objective returned non-finite values"
            )
        return values
