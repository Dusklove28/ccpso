import numpy as np

from problems.spec import ProblemSpec


def sphere(positions):
    return np.sum(positions**2, axis=1)


def rastrigin(positions):
    dimensions = positions.shape[1]
    return 10.0 * dimensions + np.sum(
        positions**2 - 10.0 * np.cos(2.0 * np.pi * positions),
        axis=1,
    )


def rosenbrock(positions):
    return np.sum(
        100.0 * (
            positions[:, 1:] - positions[:, :-1] ** 2
        ) ** 2
        + (1.0 - positions[:, :-1]) ** 2,
        axis=1,
    )


_CLASSIC_PROBLEMS = {
    "sphere": {
        "name": "Sphere",
        "lower_bound": -100.0,
        "upper_bound": 100.0,
        "objective": sphere,
    },
    "rastrigin": {
        "name": "Rastrigin",
        "lower_bound": -5.12,
        "upper_bound": 5.12,
        "objective": rastrigin,
    },
    "rosenbrock": {
        "name": "Rosenbrock",
        "lower_bound": -30.0,
        "upper_bound": 30.0,
        "objective": rosenbrock,
    },
}


def make_classic_problem(name, dimensions):
    if not isinstance(name, str):
        raise ValueError(
            f"classic problem name must be a string, got {name!r}"
        )
    problem_id = name.strip().lower()
    if problem_id not in _CLASSIC_PROBLEMS:
        available = ", ".join(sorted(_CLASSIC_PROBLEMS))
        raise ValueError(
            f"unknown classic problem {name!r}; available: {available}"
        )

    definition = _CLASSIC_PROBLEMS[problem_id]
    return ProblemSpec(
        suite="classic",
        problem_id=problem_id,
        name=definition["name"],
        dimensions=dimensions,
        lower_bound=definition["lower_bound"],
        upper_bound=definition["upper_bound"],
        optimum=0.0,
        objective=definition["objective"],
    )
