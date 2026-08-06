"""CEC2017 problem definitions under the project's 29-function protocol."""

from dataclasses import dataclass

import numpy as np
from cec2017 import transforms as _cec2017_transforms
from cec2017.functions import all_functions as _cec2017_functions

from problems.spec import ProblemSpec


CEC2017_FUNCTION_IDS = tuple(range(1, 30))
CEC2017_SOURCE_FUNCTION_IDS = (1, *range(3, 31))
CEC2017_SUPPORTED_DIMENSIONS = (10, 30, 50, 100)
CEC2017_REPRESENTATIVE_IDS = (1, 5, 12, 23)

_COMPOSITION_COMPONENT_COUNTS = {
    21: 3,
    22: 3,
    23: 4,
    24: 4,
    25: 5,
    26: 5,
    27: 6,
    28: 6,
    29: 3,
    30: 3,
}


def _validate_integer(name, value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def public_to_source_function_id(function_id):
    """Map a public consecutive function ID (1..29) to its CEC source ID."""
    public_id = _validate_integer("function_id", function_id)
    if public_id not in CEC2017_FUNCTION_IDS:
        raise ValueError(
            "function_id must be a public CEC2017 ID in [1, 29], "
            f"got {function_id!r}"
        )
    return public_id if public_id == 1 else public_id + 1


def cec2017_category(function_id):
    """Return the category assigned to a public consecutive function ID."""
    public_id = _validate_integer("function_id", function_id)
    if public_id not in CEC2017_FUNCTION_IDS:
        raise ValueError(
            "function_id must be a public CEC2017 ID in [1, 29], "
            f"got {function_id!r}"
        )
    if public_id <= 2:
        return "unimodal"
    if public_id <= 9:
        return "multimodal"
    if public_id <= 19:
        return "hybrid"
    return "composition"


def _validate_dimensions(dimensions):
    dimensions_value = _validate_integer("dimensions", dimensions)
    if dimensions_value not in CEC2017_SUPPORTED_DIMENSIONS:
        supported = ", ".join(
            str(value) for value in CEC2017_SUPPORTED_DIMENSIONS
        )
        raise ValueError(
            f"dimensions must be one of ({supported}), "
            f"got {dimensions!r}"
        )
    return dimensions_value


def cec2017_max_fe(dimensions):
    """Return the formal CEC2017 evaluation budget, MaxFES = 10000D."""
    dimensions_value = _validate_dimensions(dimensions)
    return 10_000 * dimensions_value


class _CEC2017Objective:
    """Bind a source function and handle exact composition shift points."""

    def __init__(self, source_function_id, dimensions):
        self.source_function_id = source_function_id
        self.source_function = _cec2017_functions[source_function_id - 1]
        self.optimum = 100.0 * source_function_id
        self.composition_shifts = None

        if source_function_id in _COMPOSITION_COMPONENT_COUNTS:
            component_count = _COMPOSITION_COMPONENT_COUNTS[
                source_function_id
            ]
            source_shifts = _cec2017_transforms.shifts_cf[
                source_function_id - 21
            ]
            self.composition_shifts = np.asarray(
                source_shifts[:component_count, :dimensions],
                dtype=np.float64,
            )

    def __call__(self, positions):
        if self.composition_shifts is None:
            return self.source_function(positions)

        exact_matches = np.all(
            positions[:, None, :] == self.composition_shifts[None, :, :],
            axis=2,
        )
        matched_rows = np.any(exact_matches, axis=1)
        values = np.empty(positions.shape[0], dtype=np.float64)

        ordinary_rows = ~matched_rows
        if np.any(ordinary_rows):
            values[ordinary_rows] = self.source_function(
                positions[ordinary_rows]
            )
        if np.any(matched_rows):
            component_indices = np.argmax(
                exact_matches[matched_rows],
                axis=1,
            )
            values[matched_rows] = (
                self.optimum + 100.0 * component_indices
            )
        return values


@dataclass(frozen=True)
class CEC2017ProblemSpec(ProblemSpec):
    """ProblemSpec with both public and original CEC2017 identifiers."""

    source_function_id: int
    category: str

    def __post_init__(self):
        super().__post_init__()
        source_function_id = _validate_integer(
            "source_function_id",
            self.source_function_id,
        )
        if source_function_id not in CEC2017_SOURCE_FUNCTION_IDS:
            raise ValueError(
                "source_function_id must be one of the supported CEC2017 "
                f"source IDs, got {self.source_function_id!r}"
            )
        if self.category not in {
            "unimodal",
            "multimodal",
            "hybrid",
            "composition",
        }:
            raise ValueError(
                f"category is not a CEC2017 category, got {self.category!r}"
            )
        object.__setattr__(self, "source_function_id", source_function_id)


def make_cec2017_problem(function_id, dimensions):
    """Create one problem using the project's consecutive public numbering."""
    public_id = _validate_integer("function_id", function_id)
    source_id = public_to_source_function_id(public_id)
    dimensions_value = _validate_dimensions(dimensions)
    category = cec2017_category(public_id)

    return CEC2017ProblemSpec(
        suite="cec2017",
        problem_id=public_id,
        name=f"CEC2017 F{public_id} (source F{source_id})",
        dimensions=dimensions_value,
        lower_bound=-100.0,
        upper_bound=100.0,
        optimum=100.0 * source_id,
        objective=_CEC2017Objective(source_id, dimensions_value),
        source_function_id=source_id,
        category=category,
    )


__all__ = [
    "CEC2017_FUNCTION_IDS",
    "CEC2017_SOURCE_FUNCTION_IDS",
    "CEC2017_SUPPORTED_DIMENSIONS",
    "CEC2017_REPRESENTATIVE_IDS",
    "CEC2017ProblemSpec",
    "public_to_source_function_id",
    "cec2017_category",
    "cec2017_max_fe",
    "make_cec2017_problem",
]
