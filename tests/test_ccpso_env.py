from pathlib import Path
import sys
import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env


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
        self.env = self.make_env()

    def make_env(self, **overrides):
        parameters = {
            "swarm": self.swarm,
            "c_min": 0.0,
            "c_max": 1.5,
            "optimum": 0.0,
        }
        parameters.update(overrides)
        return CCPSOEnv(**parameters)

    def test_passes_gymnasium_env_checker(self):
        check_env(self.env, skip_render_check=True)

    def assert_invalid_env_parameter(self, name, value, **overrides):
        parameters = {name: value, **overrides}
        with self.assertRaises(ValueError) as context:
            self.make_env(**parameters)

        message = str(context.exception)
        self.assertIn(name, message)
        self.assertIn(repr(value), message)

    def assert_valid_observation(self, observation):
        self.assertEqual(observation.shape, (6,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(observation))

    def test_rejects_non_finite_c_bounds(self):
        cases = (
            ("c_min", np.nan),
            ("c_min", -np.inf),
            ("c_max", np.inf),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                self.assert_invalid_env_parameter(name, value)

    def test_rejects_reversed_c_bounds(self):
        with self.assertRaises(ValueError) as context:
            self.make_env(c_min=1.0, c_max=0.5)

        message = str(context.exception)
        self.assertIn("c_min", message)
        self.assertIn("1.0", message)
        self.assertIn("c_max", message)
        self.assertIn("0.5", message)

    def test_rejects_invalid_window_parameters(self):
        for name in ("recent_window", "stagnation_horizon"):
            for value in (3.0, True, 0, -1):
                with self.subTest(name=name, value=value):
                    self.assert_invalid_env_parameter(name, value)

    def test_rejects_invalid_movement_log_floor(self):
        for value in (np.nan, np.inf, 0.0, 1.0, -0.1, 1.1):
            with self.subTest(value=value):
                self.assert_invalid_env_parameter(
                    "movement_log_floor",
                    value,
                )

    def test_accepts_equal_c_bounds_and_valid_parameters(self):
        self.env = self.make_env(
            c_min=0.75,
            c_max=0.75,
            recent_window=np.int64(3),
            stagnation_horizon=np.int32(4),
            movement_log_floor=1e-4,
        )

        observation, info = self.env.reset(seed=0)

        self.assert_valid_observation(observation)
        self.assertIsInstance(info, dict)
        self.assertEqual(self.env.c_min, self.env.c_max)

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

    def test_invalid_action_shapes_report_original_shape_without_mutation(self):
        self.env.reset(seed=0)

        for action, expected_shape, flattened_shape in (
            (np.zeros((2, 2), dtype=np.float32), "(2, 2)", "(4,)"),
            (np.zeros((2,), dtype=np.float32), "(2,)", None),
        ):
            with self.subTest(shape=action.shape):
                fe_count = self.swarm.fe_count
                positions = self.swarm.positions.copy()
                q_positions = self.swarm.q_positions.copy()
                generation_prepared = self.swarm.generation_prepared

                with self.assertRaises(ValueError) as context:
                    self.env.step(action)

                message = str(context.exception)
                self.assertIn(expected_shape, message)
                if flattened_shape is not None:
                    self.assertNotIn(flattened_shape, message)
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

    def test_single_element_vector_action_still_executes(self):
        self.env.reset(seed=0)
        fe_count = self.swarm.fe_count

        result = self.env.step(np.array([0.0], dtype=np.float32))

        self.assertEqual(len(result), 5)
        self.assertEqual(
            self.swarm.fe_count,
            fe_count + self.swarm.particles,
        )
        self.assertAlmostEqual(result[4]["conv"], 0.75)

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

    def test_default_reward_clipping_and_nonnegative_behavior_are_unchanged(self):
        self.env.initial_fitness_scale = 1.0

        clipped_reward, unclipped_progress = self.env._calculate_reward(
            1e6,
            0.0,
        )
        degraded_reward, degraded_progress = self.env._calculate_reward(
            1.0,
            2.0,
        )

        self.assertEqual(clipped_reward, 5.0)
        self.assertGreater(unclipped_progress, 5.0)
        self.assertEqual(degraded_reward, 0.0)
        self.assertEqual(degraded_progress, 0.0)

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

    def test_observation_rejects_unexpected_swarm_shapes(self):
        expected_shape = (
            self.swarm.particles,
            self.swarm.dimensions,
        )
        wrong_shape = (2, 2)
        cases = (
            ("positions", wrong_shape, expected_shape),
            ("q_positions", expected_shape, wrong_shape),
            ("positions", wrong_shape, wrong_shape),
        )

        for array_name, positions_shape, q_shape in cases:
            with self.subTest(
                positions_shape=positions_shape,
                q_shape=q_shape,
            ):
                self.env.reset(seed=0)
                self.swarm.positions = np.zeros(positions_shape)
                self.swarm.q_positions = np.zeros(q_shape)

                with self.assertRaises(ValueError) as context:
                    self.env._get_observation()

                actual_shape = (
                    positions_shape
                    if array_name == "positions"
                    else q_shape
                )
                message = str(context.exception)
                self.assertIn(array_name, message)
                self.assertIn(str(actual_shape), message)
                self.assertIn(str(expected_shape), message)

    def test_observation_accepts_expected_swarm_shapes(self):
        observation, _ = self.env.reset(seed=0)
        expected_shape = (
            self.swarm.particles,
            self.swarm.dimensions,
        )

        self.assertEqual(self.swarm.positions.shape, expected_shape)
        self.assertEqual(self.swarm.q_positions.shape, expected_shape)
        self.assert_valid_observation(observation)

    def test_reset_step_terminal_lifecycle(self):
        reset_result = self.env.reset(seed=0)

        self.assertIsInstance(reset_result, tuple)
        self.assertEqual(len(reset_result), 2)

        observation, info = reset_result
        self.assert_valid_observation(observation)
        self.assertIsInstance(info, dict)
        self.assertEqual(info["boundary_clip_ratio"], 0.0)

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
            self.assertIn("boundary_clip_ratio", info)
            self.assertGreaterEqual(info["boundary_clip_ratio"], 0.0)
            self.assertLessEqual(info["boundary_clip_ratio"], 1.0)

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
