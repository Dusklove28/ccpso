import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import STATE_FIELDS
from evaluation.evaluate_c_baselines import (
    evaluate_fixed_c,
    evaluate_linear_c,
)
from evaluation.evaluate_td3 import evaluate_td3_policy
from problems import make_classic_problem


class ZeroActionPolicy:
    def select_action(self, state):
        return np.array([0.0], dtype=np.float32)


class TruncatingEnv:
    def __init__(self):
        self.swarm = SimpleNamespace(
            max_fe=8,
            fe_count=4,
            particles=4,
            gbest_fitness=1.0,
        )
        self.closed = False

    def reset(self, seed=None):
        return np.zeros(6, dtype=np.float32), {}

    def step(self, action):
        info = {
            "fe_count": 4,
            "gbest_fitness": 1.0,
            "gap": 1.0,
            "raw_action": float(action[0]),
            "conv": 0.75,
        }
        return (
            np.zeros(6, dtype=np.float32),
            0.0,
            False,
            True,
            info,
        )

    def close(self):
        self.closed = True


class TestCBaselineEvaluation(unittest.TestCase):
    def setUp(self):
        self.problem = make_classic_problem("sphere", dimensions=3)

    def assert_finite_tree(self, value):
        if isinstance(value, float):
            self.assertTrue(np.isfinite(value))
        elif isinstance(value, dict):
            for child in value.values():
                self.assert_finite_tree(child)
        elif isinstance(value, list):
            for child in value:
                self.assert_finite_tree(child)

    def test_fixed_c_multi_seed_is_complete_and_repeatable(self):
        arguments = {
            "problem": self.problem,
            "particles": 4,
            "max_fe": 16,
            "seeds": [11, 12],
            "c_value": 0.75,
        }

        first = evaluate_fixed_c(**arguments)
        second = evaluate_fixed_c(**arguments)

        self.assertEqual(first, second)
        self.assertEqual(
            first["baseline"],
            {"name": "fixed_c", "c_value": 0.75},
        )
        self.assertEqual(len(first["episodes"]), 2)
        self.assertEqual(len(first["steps"]), 6)
        for episode in first["episodes"]:
            self.assertEqual(episode["steps"], 3)
            self.assertEqual(episode["final_fe"], 16)
            self.assertEqual(episode["c_mean"], 0.75)
            self.assertEqual(episode["c_min"], 0.75)
            self.assertEqual(episode["c_max"], 0.75)
        for step in first["steps"]:
            self.assertEqual(step["raw_action"], 0.0)
            self.assertEqual(step["c_value"], 0.75)
            for field in STATE_FIELDS:
                self.assertTrue(np.isfinite(step[field]))

        self.assert_finite_tree(first)
        self.assertIsInstance(json.dumps(first, allow_nan=False), str)

    def test_linear_c_uses_exact_three_step_schedule(self):
        result = evaluate_linear_c(
            self.problem,
            particles=4,
            max_fe=16,
            seeds=[21],
            c_start=1.5,
            c_end=0.0,
        )

        self.assertEqual(
            result["baseline"],
            {"name": "linear_c", "c_start": 1.5, "c_end": 0.0},
        )
        self.assertEqual(
            [step["c_value"] for step in result["steps"]],
            [1.5, 0.75, 0.0],
        )
        self.assertEqual(
            [step["raw_action"] for step in result["steps"]],
            [1.0, 0.0, -1.0],
        )
        self.assertEqual(
            [step["fe_count"] for step in result["steps"]],
            [8, 12, 16],
        )

    def test_non_divisible_fe_uses_only_complete_swarm_steps(self):
        result = evaluate_fixed_c(
            self.problem,
            particles=4,
            max_fe=18,
            seeds=[31],
            c_value=0.75,
        )

        self.assertEqual(result["episodes"][0]["steps"], 3)
        self.assertEqual(result["episodes"][0]["final_fe"], 16)
        self.assertEqual(
            [step["fe_count"] for step in result["steps"]],
            [8, 12, 16],
        )

    def test_fixed_c_matches_zero_action_td3_trajectory_exactly(self):
        common = {
            "problem": self.problem,
            "particles": 4,
            "max_fe": 16,
            "seeds": [41, 42],
        }

        td3_result = evaluate_td3_policy(
            ZeroActionPolicy(),
            **common,
        )
        baseline_result = evaluate_fixed_c(
            c_value=0.75,
            **common,
        )

        self.assertEqual(
            baseline_result["problem"],
            td3_result["problem"],
        )
        self.assertEqual(
            len(baseline_result["steps"]),
            len(td3_result["steps"]),
        )
        for baseline_step, td3_step in zip(
            baseline_result["steps"],
            td3_result["steps"],
        ):
            self.assertEqual(
                baseline_step,
                {
                    field: td3_step[field]
                    for field in baseline_step
                },
            )
            self.assertIn("reward_progress", td3_step)

        self.assertEqual(
            len(baseline_result["episodes"]),
            len(td3_result["episodes"]),
        )
        for baseline_episode, td3_episode in zip(
            baseline_result["episodes"],
            td3_result["episodes"],
        ):
            self.assertEqual(
                baseline_episode,
                {
                    field: td3_episode[field]
                    for field in baseline_episode
                },
            )
            self.assertIn("reward_mode", td3_episode)
            self.assertIn("initial_improvement_scale", td3_episode)
            self.assertIn("initial_gap_scale", td3_episode)
        self.assertEqual(
            baseline_result["final_gap_statistics"],
            td3_result["final_gap_statistics"],
        )

    def test_truncation_is_rejected_and_environment_is_closed(self):
        fake_env = TruncatingEnv()
        with patch(
            "evaluation.evaluate_c_baselines.make_ccpso_env",
            return_value=fake_env,
        ):
            with self.assertRaisesRegex(RuntimeError, "did not terminate by FE"):
                evaluate_fixed_c(
                    self.problem,
                    particles=4,
                    max_fe=8,
                    seeds=[51],
                    c_value=0.75,
                )

        self.assertTrue(fake_env.closed)

    def test_invalid_seeds_and_c_ranges_are_rejected(self):
        invalid_seeds = ([], [True], [1.0], "1,2")
        for seeds in invalid_seeds:
            with self.subTest(seeds=seeds):
                with self.assertRaises(ValueError):
                    evaluate_fixed_c(
                        self.problem,
                        particles=4,
                        max_fe=16,
                        seeds=seeds,
                        c_value=0.75,
                    )

        invalid_fixed = (
            {"c_value": -0.1},
            {"c_value": 1.6},
            {"c_value": np.nan},
            {"c_value": 0.75, "c_min": 1.5, "c_max": 0.0},
        )
        for parameters in invalid_fixed:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    evaluate_fixed_c(
                        self.problem,
                        particles=4,
                        max_fe=16,
                        seeds=[61],
                        **parameters,
                    )

        for parameters in (
            {"c_start": 1.6},
            {"c_end": -0.1},
            {"c_start": np.inf},
        ):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    evaluate_linear_c(
                        self.problem,
                        particles=4,
                        max_fe=16,
                        seeds=[62],
                        **parameters,
                    )


if __name__ == "__main__":
    unittest.main()
