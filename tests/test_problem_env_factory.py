from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.factory import make_ccpso_env
from problems import ProblemSpec, make_classic_problem


class TestProblemEnvFactory(unittest.TestCase):
    def test_classic_problems_create_working_environments(self):
        for problem_name in ("sphere", "rastrigin", "rosenbrock"):
            with self.subTest(problem_name=problem_name):
                problem = make_classic_problem(
                    problem_name,
                    dimensions=3,
                )
                env = make_ccpso_env(
                    problem,
                    particles=4,
                    max_fe=16,
                    seed=123,
                )

                observation, info = env.reset(seed=123)

                self.assertEqual(observation.shape, (6,))
                self.assertEqual(observation.dtype, np.float32)
                self.assertTrue(
                    env.observation_space.contains(observation)
                )
                self.assertEqual(
                    info["function_id"],
                    problem.problem_id,
                )
                self.assertIs(env.problem, problem)
                self.assertEqual(
                    env.swarm.dimensions,
                    problem.dimensions,
                )
                np.testing.assert_array_equal(
                    env.swarm.lower_bound,
                    problem.lower_bound,
                )
                np.testing.assert_array_equal(
                    env.swarm.upper_bound,
                    problem.upper_bound,
                )
                self.assertIs(
                    env.swarm.fun.__self__,
                    problem,
                )
                self.assertIs(
                    env.swarm.fun.__func__,
                    ProblemSpec.evaluate,
                )

                sample_positions = np.array(
                    [[0.0, 1.0, -1.0], [1.0, 1.0, 1.0]],
                    dtype=np.float64,
                )
                np.testing.assert_array_equal(
                    env.swarm.fun(sample_positions),
                    problem.evaluate(sample_positions),
                )

                step_result = env.step(
                    np.array([0.0], dtype=np.float32)
                )
                self.assertEqual(len(step_result), 5)
                self.assertTrue(
                    env.observation_space.contains(step_result[0])
                )
                self.assertIs(step_result[2], False)
                self.assertIs(step_result[3], False)

    def test_same_seed_produces_identical_initial_state(self):
        for problem_name in ("sphere", "rastrigin", "rosenbrock"):
            with self.subTest(problem_name=problem_name):
                problem = make_classic_problem(
                    problem_name,
                    dimensions=3,
                )
                first_env = make_ccpso_env(
                    problem,
                    particles=4,
                    max_fe=16,
                    seed=456,
                )
                second_env = make_ccpso_env(
                    problem,
                    particles=4,
                    max_fe=16,
                    seed=456,
                )

                first_observation, first_info = first_env.reset(seed=456)
                second_observation, second_info = second_env.reset(seed=456)

                np.testing.assert_array_equal(
                    first_observation,
                    second_observation,
                )
                np.testing.assert_array_equal(
                    first_env.swarm.positions,
                    second_env.swarm.positions,
                )
                np.testing.assert_array_equal(
                    first_env.swarm.q_positions,
                    second_env.swarm.q_positions,
                )
                self.assertEqual(first_info, second_info)

    def test_integer_problem_id_is_preserved(self):
        problem = ProblemSpec(
            suite="test",
            problem_id=7,
            name="Integer ID",
            dimensions=2,
            lower_bound=-1.0,
            upper_bound=1.0,
            optimum=0.0,
            objective=lambda positions: np.sum(positions**2, axis=1),
        )
        env = make_ccpso_env(
            problem,
            particles=4,
            max_fe=8,
            seed=789,
        )

        _, info = env.reset(seed=789)

        self.assertIs(type(env.function_id), int)
        self.assertEqual(env.function_id, 7)
        self.assertEqual(info["function_id"], 7)


if __name__ == "__main__":
    unittest.main()
