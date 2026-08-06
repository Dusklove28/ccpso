"""JSON-safe metadata serialization for optimization problems."""

import json

import numpy as np

from problems.cec2017 import (
    CEC2017ProblemSpec,
    cec2017_category,
    public_to_source_function_id,
)
from problems.spec import ProblemSpec


def _json_problem_id(problem_id):
    if isinstance(problem_id, np.generic):
        return problem_id.item()
    return problem_id


def _validate_cec2017_metadata(problem):
    if problem.suite != "cec2017":
        raise ValueError(
            "CEC2017ProblemSpec suite must be 'cec2017', "
            f"got {problem.suite!r}"
        )

    expected_source_id = public_to_source_function_id(problem.problem_id)
    source_function_id = problem.source_function_id
    if (
        isinstance(source_function_id, (bool, np.bool_))
        or not isinstance(source_function_id, (int, np.integer))
    ):
        raise ValueError(
            "CEC2017 source_function_id must be an integer, "
            f"got {source_function_id!r}"
        )
    source_function_id = int(source_function_id)
    if source_function_id != expected_source_id:
        raise ValueError(
            "CEC2017 numbering mismatch: "
            f"problem_id={problem.problem_id!r} maps to source "
            f"F{expected_source_id}, got source_function_id="
            f"{source_function_id!r}"
        )

    expected_category = cec2017_category(problem.problem_id)
    if problem.category != expected_category:
        raise ValueError(
            "CEC2017 category mismatch: "
            f"problem_id={problem.problem_id!r} requires "
            f"{expected_category!r}, got {problem.category!r}"
        )

    expected_optimum = 100.0 * source_function_id
    if problem.optimum != expected_optimum:
        raise ValueError(
            "CEC2017 optimum mismatch: "
            f"source_function_id={source_function_id} requires "
            f"{expected_optimum!r}, got {problem.optimum!r}"
        )
    return source_function_id, expected_category


def serialize_problem(problem):
    """Serialize a ProblemSpec without inferring suite-specific fields."""
    if not isinstance(problem, ProblemSpec):
        raise TypeError("problem must be an instance of ProblemSpec")

    metadata = {
        "suite": problem.suite,
        "problem_id": _json_problem_id(problem.problem_id),
        "name": problem.name,
        "dimensions": int(problem.dimensions),
        "lower_bound": problem.lower_bound.tolist(),
        "upper_bound": problem.upper_bound.tolist(),
        "optimum": float(problem.optimum),
    }

    if isinstance(problem, CEC2017ProblemSpec):
        source_function_id, category = _validate_cec2017_metadata(problem)
        metadata["source_function_id"] = source_function_id
        metadata["category"] = category

    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "problem metadata must be strictly JSON serializable"
        ) from error
    return metadata


__all__ = ["serialize_problem"]
