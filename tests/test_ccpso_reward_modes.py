import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.ccpso_env import CCPSOEnv
from environments.reward_functions import (
    calculate_initial_improvement_scale,
    linear_improvement_reward,
    log_gap_reduction_reward,
)
from swarm.ccpso import CCPSOSwarm


REWARD_MODES = (
    "step_log_improvement",
    "linear_improvement",
    "oracle_log_gap_reduction",
)


def sphere(positions):
    return np.sum(positions ** 2, axis=1)


class TestCCPSORewardModes(unittest.TestCase):
    actions = (0.0, 0.4, -0.6)

    def make_env(
        self,
        *,
        reward_mode="step_log_improvement",
        reward_epsilon=1e-12,
        optimum=0.0,
        objective=sphere,
        explicit_mode=True,
    ):
        swarm = CCPSOSwarm(
            particles=4,
            dimensions=3,
            fun=objective,
            lower_bound=-100.0,
            upper_bound=100.0,
            max_fe=16,
            seed=0,
        )
        parameters = {
            "swarm": swarm,
            "c_min": 0.0,
            "c_max": 1.5,
            "optimum": optimum,
            "reward_epsilon": reward_epsilon,
        }
        if explicit_mode:
            parameters["reward_mode"] = reward_mode
        return CCPSOEnv(**parameters)

    @staticmethod
    def _search_snapshot(env, observation, terminated, truncated, info):
        return {
            "observation": observation.tolist(),
            "positions": env.swarm.positions.tolist(),
            "fitness": env.swarm.fitness.tolist(),
            "pbest_positions": env.swarm.pbest_positions.tolist(),
            "pbest_fitness": env.swarm.pbest_fitness.tolist(),
            "gbest_position": env.swarm.gbest_position.tolist(),
            "gbest_fitness": float(env.swarm.gbest_fitness),
            "q_positions": env.swarm.q_positions.tolist(),
            "fe_count": int(env.swarm.fe_count),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "boundary_clip_ratio": float(info["boundary_clip_ratio"]),
        }

    def run_episode(self, env, seed=0):
        observation, reset_info = env.reset(seed=seed)
        initial_best = float(env.swarm.gbest_fitness)
        search = [
            self._search_snapshot(
                env,
                observation,
                False,
                False,
                reset_info,
            )
        ]
        transitions = []

        for action_value in self.actions:
            old_best = float(env.swarm.gbest_fitness)
            observation, reward, terminated, truncated, info = env.step(
                np.array([action_value], dtype=np.float32)
            )
            new_best = float(env.swarm.gbest_fitness)
            transitions.append({
                "old_best": old_best,
                "new_best": new_best,
                "reward": float(reward),
                "reward_progress": float(info["reward_progress"]),
                "info": info,
            })
            search.append(
                self._search_snapshot(
                    env,
                    observation,
                    terminated,
                    truncated,
                    info,
                )
            )

        self.assertTrue(search[-1]["terminated"])
        self.assertFalse(search[-1]["truncated"])
        return {
            "reset_info": reset_info,
            "initial_best": initial_best,
            "final_best": float(env.swarm.gbest_fitness),
            "initial_improvement_scale": float(
                env.initial_improvement_scale
            ),
            "initial_gap_scale": float(env.initial_gap_scale),
            "transitions": transitions,
            "search": search,
        }

    def test_default_and_explicit_step_mode_are_identical(self):
        default_run = self.run_episode(
            self.make_env(explicit_mode=False)
        )
        explicit_run = self.run_episode(
            self.make_env(reward_mode="step_log_improvement")
        )

        self.assertEqual(default_run, explicit_run)

    def test_all_modes_have_identical_search_trajectories(self):
        runs = {
            mode: self.run_episode(self.make_env(reward_mode=mode))
            for mode in REWARD_MODES
        }
        reference = runs["step_log_improvement"]["search"]

        for mode in REWARD_MODES[1:]:
            with self.subTest(mode=mode):
                self.assertEqual(runs[mode]["search"], reference)

    def test_linear_rewards_match_pure_function_and_telescope(self):
        run = self.run_episode(
            self.make_env(reward_mode="linear_improvement")
        )
        scale = run["initial_improvement_scale"]
        rewards = []
        for transition in run["transitions"]:
            expected = linear_improvement_reward(
                transition["old_best"],
                transition["new_best"],
                scale,
            )
            self.assertEqual(transition["reward"], expected)
            self.assertEqual(transition["reward_progress"], expected)
            rewards.append(transition["reward"])

        self.assertTrue(all(reward >= 0.0 for reward in rewards))
        expected_return = (
            run["initial_best"] - run["final_best"]
        ) / scale
        self.assertAlmostEqual(sum(rewards), expected_return, places=15)

    def test_linear_reward_is_shift_and_positive_scale_invariant(self):
        offset = 1024.0
        factor = 8.0

        def shifted_sphere(positions):
            return sphere(positions) + offset

        def scaled_sphere(positions):
            return factor * sphere(positions)

        base = self.run_episode(
            self.make_env(reward_mode="linear_improvement")
        )
        shifted = self.run_episode(
            self.make_env(
                reward_mode="linear_improvement",
                objective=shifted_sphere,
            )
        )
        scaled = self.run_episode(
            self.make_env(
                reward_mode="linear_improvement",
                objective=scaled_sphere,
            )
        )

        base_rewards = [item["reward"] for item in base["transitions"]]
        shifted_rewards = [
            item["reward"] for item in shifted["transitions"]
        ]
        scaled_rewards = [
            item["reward"] for item in scaled["transitions"]
        ]
        np.testing.assert_allclose(
            shifted_rewards,
            base_rewards,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_array_equal(scaled_rewards, base_rewards)
        self.assertAlmostEqual(
            shifted["initial_improvement_scale"],
            base["initial_improvement_scale"],
            places=12,
        )
        self.assertEqual(
            scaled["initial_improvement_scale"],
            factor * base["initial_improvement_scale"],
        )

    def test_optimum_only_changes_oracle_reward(self):
        linear_zero = self.run_episode(
            self.make_env(
                reward_mode="linear_improvement",
                optimum=0.0,
            )
        )
        linear_shifted = self.run_episode(
            self.make_env(
                reward_mode="linear_improvement",
                optimum=-1000.0,
            )
        )
        self.assertEqual(
            [item["reward"] for item in linear_zero["transitions"]],
            [item["reward"] for item in linear_shifted["transitions"]],
        )
        self.assertEqual(linear_zero["search"], linear_shifted["search"])

        oracle_zero = self.run_episode(
            self.make_env(
                reward_mode="oracle_log_gap_reduction",
                optimum=0.0,
            )
        )
        oracle_shifted = self.run_episode(
            self.make_env(
                reward_mode="oracle_log_gap_reduction",
                optimum=-1000.0,
            )
        )
        self.assertEqual(oracle_zero["search"], oracle_shifted["search"])
        self.assertNotEqual(
            [item["reward"] for item in oracle_zero["transitions"]],
            [item["reward"] for item in oracle_shifted["transitions"]],
        )

    def test_reward_epsilon_only_changes_oracle_rewards(self):
        for mode in ("step_log_improvement", "linear_improvement"):
            with self.subTest(mode=mode):
                default_epsilon = self.run_episode(
                    self.make_env(reward_mode=mode, reward_epsilon=1e-12)
                )
                changed_epsilon = self.run_episode(
                    self.make_env(reward_mode=mode, reward_epsilon=0.25)
                )
                self.assertEqual(
                    [
                        item["reward"]
                        for item in default_epsilon["transitions"]
                    ],
                    [
                        item["reward"]
                        for item in changed_epsilon["transitions"]
                    ],
                )

        oracle_default = self.run_episode(
            self.make_env(
                reward_mode="oracle_log_gap_reduction",
                reward_epsilon=1e-12,
            )
        )
        oracle_changed = self.run_episode(
            self.make_env(
                reward_mode="oracle_log_gap_reduction",
                reward_epsilon=0.25,
            )
        )
        self.assertNotEqual(
            [item["reward"] for item in oracle_default["transitions"]],
            [item["reward"] for item in oracle_changed["transitions"]],
        )

    def test_oracle_rewards_match_pure_function_and_telescope(self):
        optimum = -10.0
        epsilon = 1e-9
        run = self.run_episode(
            self.make_env(
                reward_mode="oracle_log_gap_reduction",
                reward_epsilon=epsilon,
                optimum=optimum,
            )
        )
        scale = run["initial_gap_scale"]
        rewards = []
        for transition in run["transitions"]:
            expected = log_gap_reduction_reward(
                transition["old_best"],
                transition["new_best"],
                optimum,
                scale,
                epsilon=epsilon,
            )
            self.assertEqual(transition["reward"], expected)
            self.assertEqual(transition["reward_progress"], expected)
            rewards.append(transition["reward"])

        direct = log_gap_reduction_reward(
            run["initial_best"],
            run["final_best"],
            optimum,
            scale,
            epsilon=epsilon,
        )
        self.assertAlmostEqual(sum(rewards), direct, places=14)

    def test_reset_recomputes_both_scales_without_leakage(self):
        env = self.make_env(reward_mode="linear_improvement")
        env.reset(seed=0)
        first_scales = (
            env.initial_improvement_scale,
            env.initial_gap_scale,
        )
        env.initial_improvement_scale = 999.0
        env.initial_gap_scale = 999.0

        _, info = env.reset(seed=1)
        expected_improvement = calculate_initial_improvement_scale(
            np.asarray(env.swarm.fitness, dtype=np.float64)
        )
        expected_gap = max(
            float(env.swarm.gbest_fitness) - env.optimum,
            0.0,
            1e-12,
        )

        self.assertEqual(
            env.initial_improvement_scale,
            expected_improvement,
        )
        self.assertEqual(env.initial_gap_scale, expected_gap)
        self.assertEqual(
            info["initial_improvement_scale"],
            expected_improvement,
        )
        self.assertEqual(info["initial_gap_scale"], expected_gap)
        self.assertNotEqual(
            (env.initial_improvement_scale, env.initial_gap_scale),
            (999.0, 999.0),
        )
        self.assertNotEqual(
            first_scales,
            (env.initial_improvement_scale, env.initial_gap_scale),
        )

    def test_terminal_rewards_and_diagnostics_are_finite_json(self):
        for mode in REWARD_MODES:
            with self.subTest(mode=mode):
                run = self.run_episode(self.make_env(reward_mode=mode))
                terminal = run["transitions"][-1]
                self.assertTrue(math.isfinite(terminal["reward"]))
                self.assertEqual(terminal["info"]["reward_mode"], mode)
                self.assertTrue(math.isfinite(
                    terminal["info"]["initial_improvement_scale"]
                ))
                self.assertTrue(math.isfinite(
                    terminal["info"]["initial_gap_scale"]
                ))
                json.dumps(run, allow_nan=False)

    def test_rejects_invalid_mode_epsilon_and_optimum(self):
        for invalid in (
            True,
            None,
            "STEP_LOG_IMPROVEMENT",
            "linear",
            "oracle_log_gap_reduction ",
        ):
            with self.subTest(reward_mode=invalid):
                with self.assertRaisesRegex(ValueError, "reward_mode"):
                    self.make_env(reward_mode=invalid)

        for invalid in (
            True,
            np.bool_(False),
            "1e-12",
            1e-12 + 0.0j,
            0.0,
            -1.0,
            np.nan,
            np.inf,
        ):
            with self.subTest(reward_epsilon=invalid):
                with self.assertRaisesRegex(ValueError, "reward_epsilon"):
                    self.make_env(reward_epsilon=invalid)

        for invalid in (
            True,
            np.bool_(False),
            "0.0",
            0.0 + 0.0j,
            np.nan,
            np.inf,
            -np.inf,
        ):
            with self.subTest(optimum=invalid):
                with self.assertRaisesRegex(ValueError, "optimum"):
                    self.make_env(optimum=invalid)

    def test_all_modes_pass_gymnasium_checker(self):
        for mode in REWARD_MODES:
            with self.subTest(mode=mode):
                env = self.make_env(reward_mode=mode)
                check_env(env, skip_render_check=True)


if __name__ == "__main__":
    unittest.main()
