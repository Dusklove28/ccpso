from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problems import ProblemSpec, make_classic_problem


class TestClassicProblems(unittest.TestCase):
    def test_optimum_points_and_known_non_optimum_points(self):
        cases = (
            (
                "sphere",
                np.zeros(3),
                np.array([1.0, 2.0, 3.0]),
                14.0,
            ),
            (
                "rastrigin",
                np.zeros(3),
                np.array([1.0, 0.0, 0.0]),
                1.0,
            ),
            (
                "rosenbrock",
                np.ones(3),
                np.zeros(3),
                2.0,
            ),
        )

        for name, optimum_point, non_optimum_point, expected in cases:
            with self.subTest(name=name):
                problem = make_classic_problem(name, dimensions=3)
                values = problem.evaluate(
                    np.stack([optimum_point, non_optimum_point])
                )

                self.assertEqual(values.shape, (2,))
                self.assertEqual(values.dtype, np.float64)
                self.assertEqual(values[0], 0.0)
                self.assertAlmostEqual(values[1], expected)

    def test_batch_output_and_metadata(self):
        expected_metadata = {
            "sphere": ("Sphere", -100.0, 100.0),
            "rastrigin": ("Rastrigin", -5.12, 5.12),
            "rosenbrock": ("Rosenbrock", -30.0, 30.0),
        }
        positions = np.array(
            [[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]],
            dtype=np.float32,
        )

        for problem_id, metadata in expected_metadata.items():
            with self.subTest(problem_id=problem_id):
                expected_name, lower, upper = metadata
                problem = make_classic_problem(
                    problem_id.upper(),
                    dimensions=2,
                )
                values = problem.evaluate(positions)

                self.assertEqual(problem.suite, "classic")
                self.assertEqual(problem.problem_id, problem_id)
                self.assertEqual(problem.name, expected_name)
                self.assertEqual(problem.dimensions, 2)
                self.assertEqual(problem.optimum, 0.0)
                np.testing.assert_array_equal(
                    problem.lower_bound,
                    np.full(2, lower, dtype=np.float64),
                )
                np.testing.assert_array_equal(
                    problem.upper_bound,
                    np.full(2, upper, dtype=np.float64),
                )
                self.assertEqual(values.shape, (3,))
                self.assertEqual(values.dtype, np.float64)
                self.assertTrue(np.all(np.isfinite(values)))

    def test_evaluate_rejects_invalid_input_and_output(self):
        problem = make_classic_problem("sphere", dimensions=3)
        for positions in (
            np.zeros(3),
            np.zeros((2, 2)),
            np.zeros((1, 2, 3)),
        ):
            with self.subTest(shape=positions.shape):
                with self.assertRaises(ValueError):
                    problem.evaluate(positions)

        with self.assertRaises(FloatingPointError):
            problem.evaluate(np.array([[0.0, np.nan, 0.0]]))

        wrong_shape = ProblemSpec(
            suite="test",
            problem_id="wrong-shape",
            name="Wrong shape",
            dimensions=2,
            lower_bound=-1.0,
            upper_bound=1.0,
            optimum=0.0,
            objective=lambda positions: np.zeros((positions.shape[0], 1)),
        )
        with self.assertRaises(ValueError):
            wrong_shape.evaluate(np.zeros((2, 2)))

        non_finite = ProblemSpec(
            suite="test",
            problem_id="non-finite",
            name="Non-finite",
            dimensions=2,
            lower_bound=-1.0,
            upper_bound=1.0,
            optimum=0.0,
            objective=lambda positions: np.full(positions.shape[0], np.inf),
        )
        with self.assertRaises(FloatingPointError):
            non_finite.evaluate(np.zeros((2, 2)))

    def test_rejects_invalid_configuration_and_names(self):
        valid = {
            "suite": "test",
            "problem_id": "valid",
            "name": "Valid",
            "dimensions": 2,
            "lower_bound": -1.0,
            "upper_bound": 1.0,
            "optimum": 0.0,
            "objective": lambda positions: np.sum(positions**2, axis=1),
        }
        cases = (
            ("dimensions", 0),
            ("dimensions", 2.0),
            ("dimensions", True),
            ("lower_bound", np.nan),
            ("upper_bound", np.inf),
            ("lower_bound", np.array([-1.0, -2.0, -3.0])),
            ("lower_bound", np.array([-1.0, 1.0])),
            ("upper_bound", np.array([-2.0, 1.0])),
            ("optimum", np.nan),
            ("objective", None),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                parameters = valid.copy()
                parameters[name] = value
                with self.assertRaises(ValueError):
                    ProblemSpec(**parameters)

        for invalid_name in ("ackley", "", 1):
            with self.subTest(invalid_name=invalid_name):
                with self.assertRaises(ValueError) as context:
                    make_classic_problem(invalid_name, dimensions=2)
                self.assertIn(repr(invalid_name), str(context.exception))

    def test_optimum_is_metadata_only(self):
        problem = ProblemSpec(
            suite="test",
            problem_id="raw-objective",
            name="Raw objective",
            dimensions=2,
            lower_bound=np.array([-2.0, -3.0]),
            upper_bound=np.array([2.0, 3.0]),
            optimum=100.0,
            objective=lambda positions: np.sum(positions, axis=1),
        )

        values = problem.evaluate(
            np.array([[1.0, 2.0], [-1.0, 0.5]])
        )

        np.testing.assert_array_equal(
            values,
            np.array([3.0, -0.5], dtype=np.float64),
        )
        self.assertEqual(problem.optimum, 100.0)


if __name__ == "__main__":
    unittest.main()
