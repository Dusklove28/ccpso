from pathlib import Path
import sys
import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.ccpso_env import CCPSOEnv
from environments.factory import make_ccpso_env
from problems import ProblemSpec, make_classic_problem
from swarm.ccpso import CCPSOSwarm


def sphere(positions):
    return np.sum(positions**2, axis=1)


def make_env(
    *,
    state_mode="legacy_v1",
    max_fe=16,
    objective=sphere,
    bounds=100.0,
    optimum=0.0,
    function_id=None,
):
    swarm = CCPSOSwarm(
        particles=4,
        dimensions=3,
        fun=objective,
        lower_bound=-bounds,
        upper_bound=bounds,
        max_fe=max_fe,
        seed=0,
    )
    return CCPSOEnv(
        swarm=swarm,
        c_min=0.0,
        c_max=1.5,
        optimum=optimum,
        function_id=function_id,
        recent_window=2,
        state_mode=state_mode,
    )


class CCPSOStateModeTests(unittest.TestCase):
    def test_state_mode_validation_is_strict(self):
        for value in (True, None, "LEGACY_V1", "relative_log", ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as context:
                    make_env(state_mode=value)
                self.assertIn("state_mode", str(context.exception))
                self.assertIn(repr(value), str(context.exception))

    def test_relative_v2_reset_scales_and_initial_diversities(self):
        env = make_env(state_mode="relative_log_v2")
        observation, info = env.reset(seed=11)

        self.assertEqual(observation.shape, (6,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertEqual(observation[2], np.float32(0.5))
        self.assertEqual(observation[3], np.float32(0.5))
        self.assertEqual(observation[4], np.float32(0.0))
        self.assertEqual(observation[5], np.float32(0.0))
        self.assertGreater(env.initial_position_scale, 0.0)
        self.assertGreater(env.initial_q_scale, 0.0)
        self.assertTrue(np.isfinite(env.initial_position_scale))
        self.assertTrue(np.isfinite(env.initial_q_scale))
        self.assertEqual(env.max_episode_updates, 3)
        self.assertEqual(info["state_mode"], "relative_log_v2")
        self.assertEqual(
            info["initial_position_scale"],
            env.initial_position_scale,
        )
        self.assertEqual(info["initial_q_scale"], env.initial_q_scale)

    def test_gymnasium_checker_accepts_both_modes(self):
        for state_mode in ("legacy_v1", "relative_log_v2"):
            with self.subTest(state_mode=state_mode):
                env = make_env(state_mode=state_mode)
                check_env(env, skip_render_check=True)

    def test_stagnation_does_not_saturate_after_ten_nonterminal_steps(self):
        env = make_env(state_mode="relative_log_v2", max_fe=404)
        env.reset(seed=3)
        values = []
        for steps in (0, 1, 10, 99):
            env.stagnation_steps = steps
            values.append(float(env._get_observation()[5]))
        self.assertTrue(all(left < right for left, right in zip(values, values[1:])))
        self.assertLess(values[2], 1.0)
        self.assertLess(values[3], 1.0)

    def test_state_mode_changes_only_observation_and_state_diagnostics(self):
        legacy = make_env(state_mode="legacy_v1")
        relative = make_env(state_mode="relative_log_v2")
        legacy_observation, legacy_info = legacy.reset(seed=27)
        relative_observation, relative_info = relative.reset(seed=27)
        self.assertNotIn("state_mode", legacy_info)
        self.assertNotIn("initial_position_scale", legacy_info)
        self.assertNotIn("initial_q_scale", legacy_info)
        self.assertNotIn("max_episode_updates", legacy_info)
        np.testing.assert_array_equal(
            legacy.swarm.positions,
            relative.swarm.positions,
        )
        np.testing.assert_array_equal(
            legacy.swarm.fitness,
            relative.swarm.fitness,
        )
        np.testing.assert_array_equal(
            legacy.swarm.q_positions,
            relative.swarm.q_positions,
        )
        self.assertFalse(np.array_equal(legacy_observation, relative_observation))

        diagnostic_fields = {
            "state_mode",
            "initial_position_scale",
            "initial_q_scale",
            "max_episode_updates",
        }
        for key in set(legacy_info).difference(diagnostic_fields):
            self.assertEqual(legacy_info[key], relative_info[key])

        actions = (
            np.array([0.0], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
            np.array([-0.5], dtype=np.float32),
        )
        for action in actions:
            legacy_step = legacy.step(action)
            relative_step = relative.step(action)
            self.assertEqual(legacy_step[1:4], relative_step[1:4])
            self.assertFalse(np.array_equal(legacy_step[0], relative_step[0]))
            for name in (
                "positions",
                "previous_positions",
                "fitness",
                "pbest_positions",
                "pbest_fitness",
                "q_positions",
                "c1_r1",
                "c2_r2",
                "c_sum",
            ):
                np.testing.assert_array_equal(
                    getattr(legacy.swarm, name),
                    getattr(relative.swarm, name),
                    err_msg=name,
                )
            self.assertEqual(
                legacy.swarm.gbest_fitness,
                relative.swarm.gbest_fitness,
            )
            self.assertEqual(legacy.swarm.fe_count, relative.swarm.fe_count)
            legacy_info = legacy_step[4]
            relative_info = relative_step[4]
            for key in set(legacy_info).difference(diagnostic_fields):
                self.assertEqual(legacy_info[key], relative_info[key])

    def test_objective_translation_and_scaling_preserve_v2_state(self):
        def transformed(positions):
            return 13.0 * sphere(positions) + 12345.0

        base = make_env(state_mode="relative_log_v2", max_fe=24)
        changed = make_env(
            state_mode="relative_log_v2",
            max_fe=24,
            objective=transformed,
        )
        base_state, _ = base.reset(seed=44)
        changed_state, _ = changed.reset(seed=44)
        np.testing.assert_array_equal(base_state, changed_state)
        for action in (
            np.array([0.2], dtype=np.float32),
            np.array([-0.4], dtype=np.float32),
            np.array([0.6], dtype=np.float32),
        ):
            base_state = base.step(action)[0]
            changed_state = changed.step(action)[0]
            np.testing.assert_array_equal(base_state, changed_state)

    def test_v2_state_does_not_read_optimum_or_function_id(self):
        first = make_env(
            state_mode="relative_log_v2",
            optimum=0.0,
            function_id="first",
        )
        second = make_env(
            state_mode="relative_log_v2",
            optimum=-1e12,
            function_id=999,
        )
        first_state, _ = first.reset(seed=71)
        second_state, _ = second.reset(seed=71)
        np.testing.assert_array_equal(first_state, second_state)
        first_state = first.step(np.array([0.25], dtype=np.float32))[0]
        second_state = second.step(np.array([0.25], dtype=np.float32))[0]
        np.testing.assert_array_equal(first_state, second_state)

    def test_coordinate_and_bound_scaling_preserves_spatial_states(self):
        factor = 10.0

        def scaled_objective(positions):
            return sphere(positions / factor)

        base = make_env(state_mode="relative_log_v2", max_fe=24, bounds=10.0)
        scaled = make_env(
            state_mode="relative_log_v2",
            max_fe=24,
            bounds=10.0 * factor,
            objective=scaled_objective,
        )
        base_state, _ = base.reset(seed=55)
        scaled_state, _ = scaled.reset(seed=55)
        np.testing.assert_array_equal(base_state[2:5], scaled_state[2:5])
        for action in (
            np.array([0.1], dtype=np.float32),
            np.array([-0.2], dtype=np.float32),
            np.array([0.3], dtype=np.float32),
        ):
            base_state = base.step(action)[0]
            scaled_state = scaled.step(action)[0]
            np.testing.assert_allclose(
                base_state[2:5],
                scaled_state[2:5],
                rtol=0.0,
                atol=1e-7,
            )

    def test_factory_forwards_state_mode(self):
        problem = make_classic_problem("sphere", dimensions=3)
        env = make_ccpso_env(
            problem,
            particles=4,
            max_fe=16,
            seed=9,
            state_mode="relative_log_v2",
        )
        observation, info = env.reset(seed=9)
        self.assertEqual(env.state_mode, "relative_log_v2")
        self.assertEqual(info["state_mode"], "relative_log_v2")
        self.assertTrue(env.observation_space.contains(observation))


if __name__ == "__main__":
    unittest.main()
