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
