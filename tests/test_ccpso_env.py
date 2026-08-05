from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.ccpso_env import CCPSOEnv
from swarm.ccpso import CCPSOSwarm


def sphere(positions):
    return np.sum(positions ** 2, axis=1)


class TestCCPSOEnvLifecycle(unittest.TestCase):
    def setUp(self):
        self.swarm = CCPSOSwarm(
            particles=4,
            dimensions=3,
            fun=sphere,
            lower_bound=-100.0,
            upper_bound=100.0,
            max_fe=16,
            seed=0,
        )
        self.env = CCPSOEnv(
            swarm=self.swarm,
            c_min=0.0,
            c_max=1.5,
            optimum=0.0,
        )

    def assert_valid_observation(self, observation):
        self.assertEqual(observation.shape, (6,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(observation))

    def test_non_finite_actions_do_not_mutate_episode(self):
        self.env.reset(seed=0)

        for action_value in (np.nan, np.inf, -np.inf):
            with self.subTest(action_value=action_value):
                fe_count = self.swarm.fe_count
                positions = self.swarm.positions.copy()
                q_positions = self.swarm.q_positions.copy()
                generation_prepared = self.swarm.generation_prepared

                with self.assertRaises(ValueError) as context:
                    self.env.step(
                        np.array([action_value], dtype=np.float32)
                    )

                self.assertIn(
                    repr(float(action_value)),
                    str(context.exception),
                )
                self.assertEqual(self.swarm.fe_count, fe_count)
                np.testing.assert_array_equal(
                    self.swarm.positions,
                    positions,
                )
                np.testing.assert_array_equal(
                    self.swarm.q_positions,
                    q_positions,
                )
                self.assertIs(
                    self.swarm.generation_prepared,
                    generation_prepared,
                )

    def test_finite_out_of_range_actions_are_clipped(self):
        for action_value, expected_raw, expected_conv in (
            (2.0, 1.0, 1.5),
            (-2.0, -1.0, 0.0),
        ):
            with self.subTest(action_value=action_value):
                self.env.reset(seed=0)
                _, _, _, _, info = self.env.step(
                    np.array([action_value], dtype=np.float32)
                )

                self.assertAlmostEqual(
                    info["raw_action"],
                    expected_raw,
                )
                self.assertAlmostEqual(
                    info["conv"],
                    expected_conv,
                )

    def test_reward_rejects_non_finite_inputs(self):
        cases = (
            ("old_best", np.nan, 0.0, 1.0, np.nan),
            ("new_best", 1.0, np.inf, 1.0, np.inf),
            (
                "initial_fitness_scale",
                1.0,
                0.0,
                np.inf,
                np.inf,
            ),
        )

        for name, old_best, new_best, scale, invalid_value in cases:
            with self.subTest(name=name):
                self.env.initial_fitness_scale = scale

                with self.assertRaises(FloatingPointError) as context:
                    self.env._calculate_reward(old_best, new_best)

                message = str(context.exception)
                self.assertIn(name, message)
                self.assertIn(repr(float(invalid_value)), message)

    def test_reward_rejects_extreme_finite_overflow(self):
        max_float = np.finfo(np.float64).max

        self.env.initial_fitness_scale = 1.0
        with self.assertRaises(FloatingPointError) as subtraction_error:
            self.env._calculate_reward(max_float, -max_float)
        self.assertIn("improvement", str(subtraction_error.exception))

        self.env.initial_fitness_scale = np.finfo(np.float64).tiny
        with self.assertRaises(FloatingPointError) as division_error:
            self.env._calculate_reward(max_float, 0.0)
        self.assertIn(
            "scaled_improvement",
            str(division_error.exception),
        )

    def test_reward_normal_input_is_unchanged(self):
        self.env.initial_fitness_scale = 2.0

        reward, progress = self.env._calculate_reward(10.0, 8.0)
        expected = float(np.log1p(1.0))

        self.assertEqual(progress, expected)
        self.assertEqual(reward, expected)
        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 5.0)

    def test_recent_progress_requires_complete_window(self):
        self.env.recent_window = 3
        self.env.initial_fitness_scale = 2.0

        for best_history in (
            [10.0],
            [10.0, 9.0],
            [10.0, 9.0, 8.0],
        ):
            with self.subTest(history_length=len(best_history)):
                self.env.best_history = best_history
                self.assertEqual(
                    self.env._calculate_recent_progress(),
                    0.0,
                )

        self.env.best_history = [10.0, 9.0, 8.0, 7.0]
        expected = float(
            np.clip(
                np.tanh(np.log1p(3.0 / 2.0)),
                0.0,
                1.0,
            )
        )
        self.assertEqual(
            self.env._calculate_recent_progress(),
            expected,
        )

        self.env.best_history = [10.0, 10.0, 10.0, 10.0]
        self.assertEqual(
            self.env._calculate_recent_progress(),
            0.0,
        )

    def test_reset_step_terminal_lifecycle(self):
        reset_result = self.env.reset(seed=0)

        self.assertIsInstance(reset_result, tuple)
        self.assertEqual(len(reset_result), 2)

        observation, info = reset_result
        self.assert_valid_observation(observation)
        self.assertIsInstance(info, dict)

        action = np.array([0.0], dtype=np.float32)
        step_count = 0
        normal_step_seen = False

        while True:
            q_used_by_step = self.swarm.q_positions.copy()
            step_result = self.env.step(action)

            self.assertIsInstance(step_result, tuple)
            self.assertEqual(len(step_result), 5)

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = step_result

            step_count += 1
            self.assert_valid_observation(observation)
            self.assertIsInstance(info, dict)
            self.assertTrue(np.isfinite(reward))
            self.assertAlmostEqual(info["conv"], 0.75)

            if terminated:
                self.assertIs(truncated, False)
                self.assertIs(info["terminal"], True)
                self.assertIs(
                    self.swarm.generation_prepared,
                    False,
                )
                np.testing.assert_array_equal(
                    self.swarm.q_positions,
                    q_used_by_step,
                )
                break

            normal_step_seen = True
            self.assertIs(terminated, False)
            self.assertIs(truncated, False)

        self.assertTrue(normal_step_seen)
        self.assertEqual(step_count, 3)
        self.assertEqual(
            self.swarm.fe_count,
            self.swarm.max_fe,
        )

        with self.assertRaises(RuntimeError):
            self.env.step(action)


if __name__ == "__main__":
    unittest.main()
