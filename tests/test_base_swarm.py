from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swarm.base_swarm import BaseSwarm


class MinimalSwarm(BaseSwarm):
    def _reset_algorithm_state(self):
        return None

    def step(self, conv):
        return None


class TestBaseSwarmFitnessValidation(unittest.TestCase):
    def make_swarm(self, objective, **overrides):
        parameters = {
            "particles": 3,
            "dimensions": 2,
            "fun": objective,
            "lower_bound": -5.0,
            "upper_bound": 5.0,
            "max_fe": 10,
            "seed": 0,
        }
        parameters.update(overrides)
        return MinimalSwarm(**parameters)

    def assert_invalid_parameter(self, name, value, **overrides):
        def objective(positions):
            return np.sum(positions ** 2, axis=1)

        parameters = {name: value, **overrides}
        with self.assertRaises(ValueError) as context:
            self.make_swarm(objective, **parameters)

        message = str(context.exception)
        self.assertIn(name, message)
        self.assertIn(repr(value), message)

    def assert_reset_rejects_non_finite(self, invalid_value):
        def objective(positions):
            fitness = np.sum(positions ** 2, axis=1)
            fitness[1] = invalid_value
            return fitness

        swarm = self.make_swarm(objective)

        with self.assertRaises(FloatingPointError) as context:
            swarm.reset()

        message = str(context.exception)
        self.assertIn("1=", message)
        self.assertIn(repr(float(invalid_value)), message)
        self.assertEqual(swarm.fe_count, 0)
        self.assertIsNone(swarm.pbest_positions)
        self.assertIsNone(swarm.pbest_fitness)
        self.assertIsNone(swarm.gbest_position)
        self.assertIsNone(swarm.gbest_fitness)

    def test_reset_rejects_nan_fitness(self):
        self.assert_reset_rejects_non_finite(np.nan)

    def test_reset_rejects_positive_infinite_fitness(self):
        self.assert_reset_rejects_non_finite(np.inf)

    def test_reset_rejects_negative_infinite_fitness(self):
        self.assert_reset_rejects_non_finite(-np.inf)

    def test_rejects_invalid_particle_counts(self):
        for value in (3.5, 3.0, True, 0, -1):
            with self.subTest(value=value):
                self.assert_invalid_parameter("particles", value)

    def test_rejects_invalid_dimension_counts(self):
        for value in (2.5, 2.0, False, 0, -1):
            with self.subTest(value=value):
                self.assert_invalid_parameter("dimensions", value)

    def test_rejects_invalid_max_fe(self):
        for value in (10.5, 10.0, True, 2):
            with self.subTest(value=value):
                self.assert_invalid_parameter("max_fe", value)

    def test_rejects_non_finite_bounds(self):
        cases = (
            ("lower_bound", np.nan),
            ("lower_bound", -np.inf),
            ("upper_bound", np.inf),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                self.assert_invalid_parameter(name, value)

    def test_rejects_lower_bound_not_less_than_upper_bound(self):
        def objective(positions):
            return np.sum(positions ** 2, axis=1)

        for lower_bound, upper_bound in ((1.0, 1.0), (2.0, 1.0)):
            with self.subTest(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ):
                with self.assertRaises(ValueError) as context:
                    self.make_swarm(
                        objective,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )

                message = str(context.exception)
                self.assertIn("lower_bound", message)
                self.assertIn("upper_bound", message)
                self.assertIn(repr(lower_bound), message)
                self.assertIn(repr(upper_bound), message)

    def test_accepts_scalar_bounds_and_exact_initial_fe_budget(self):
        def objective(positions):
            return np.sum(positions ** 2, axis=1)

        swarm = self.make_swarm(objective, max_fe=3)
        swarm.reset()

        np.testing.assert_array_equal(
            swarm.lower_bound,
            np.full(2, -5.0),
        )
        np.testing.assert_array_equal(
            swarm.upper_bound,
            np.full(2, 5.0),
        )
        self.assertEqual(swarm.fe_count, swarm.particles)
        self.assertTrue(swarm.done)

    def test_accepts_numpy_integers_and_vector_bounds(self):
        def objective(positions):
            return np.sum(positions ** 2, axis=1)

        lower_bound = np.array([-5.0, -2.0])
        upper_bound = np.array([3.0, 7.0])
        swarm = self.make_swarm(
            objective,
            particles=np.int64(3),
            dimensions=np.int32(2),
            max_fe=np.int64(3),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        swarm.reset()

        np.testing.assert_array_equal(swarm.lower_bound, lower_bound)
        np.testing.assert_array_equal(swarm.upper_bound, upper_bound)
        self.assertEqual(swarm.fe_count, 3)

    def test_invalid_position_shapes_do_not_call_objective(self):
        objective_calls = 0

        def objective(positions):
            nonlocal objective_calls
            objective_calls += 1
            return np.sum(positions ** 2, axis=1)

        swarm = self.make_swarm(objective)
        invalid_positions = (
            np.zeros(2),
            np.zeros((3, 3)),
            np.zeros((2, 1, 2)),
        )

        for positions in invalid_positions:
            with self.subTest(shape=positions.shape):
                fe_count = swarm.fe_count

                with self.assertRaises(ValueError) as context:
                    swarm.evaluate(positions)

                message = str(context.exception)
                self.assertIn(str(positions.shape), message)
                self.assertIn("(n, dimensions)", message)
                self.assertEqual(objective_calls, 0)
                self.assertEqual(swarm.fe_count, fe_count)

    def test_finite_fitness_is_returned_and_counted(self):
        def objective(positions):
            return np.sum(positions ** 2, axis=1)

        swarm = self.make_swarm(objective)
        swarm.reset()

        self.assertEqual(swarm.fe_count, swarm.particles)
        self.assertIsNotNone(swarm.pbest_positions)
        self.assertIsNotNone(swarm.gbest_position)

        positions = np.zeros((2, swarm.dimensions))
        fitness = swarm.evaluate(positions)

        np.testing.assert_array_equal(fitness, np.zeros(2))
        self.assertEqual(swarm.fe_count, swarm.particles + 2)


if __name__ == "__main__":
    unittest.main()
