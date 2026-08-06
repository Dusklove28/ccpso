import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.reward_functions import (
    calculate_initial_improvement_scale,
    linear_improvement_reward,
    log_gap_reduction_reward,
)


class TestInitialImprovementScale(unittest.TestCase):
    def test_uses_maximum_of_median_distance_iqr_and_floor(self):
        fitness = np.array([1.0, 2.0, 3.0, 100.0], dtype=np.float64)
        expected = max(
            float(np.median(fitness) - np.min(fitness)),
            float(np.percentile(fitness, 75) - np.percentile(fitness, 25)),
            1e-12,
        )

        scale = calculate_initial_improvement_scale(fitness)

        self.assertIs(type(scale), float)
        self.assertTrue(math.isfinite(scale))
        self.assertGreater(scale, 0.0)
        self.assertEqual(scale, expected)

    def test_floor_handles_identical_fitness(self):
        fitness = np.full(4, 7.0, dtype=np.float64)

        self.assertEqual(
            calculate_initial_improvement_scale(fitness, floor=0.25),
            0.25,
        )

    def test_rejects_invalid_initial_fitness(self):
        cases = (
            [1.0, 2.0],
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([], dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            np.array([1.0, np.nan], dtype=np.float64),
            np.array([1.0, np.inf], dtype=np.float64),
        )

        for invalid in cases:
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "initial_fitness"):
                    calculate_initial_improvement_scale(invalid)

    def test_rejects_invalid_floor(self):
        fitness = np.array([1.0, 2.0], dtype=np.float64)
        for invalid in (True, np.bool_(False), 0.0, -1.0, np.nan, np.inf):
            with self.subTest(floor=invalid):
                with self.assertRaisesRegex(ValueError, "floor"):
                    calculate_initial_improvement_scale(
                        fitness,
                        floor=invalid,
                    )

    def test_rejects_scale_statistic_overflow(self):
        limit = np.finfo(np.float64).max
        fitness = np.array([-limit, limit], dtype=np.float64)

        with self.assertRaises(FloatingPointError):
            calculate_initial_improvement_scale(fitness)


class TestLinearImprovementReward(unittest.TestCase):
    def test_improvement_unchanged_and_degradation(self):
        self.assertEqual(
            linear_improvement_reward(10.0, 8.0, 4.0),
            0.5,
        )
        self.assertEqual(
            linear_improvement_reward(8.0, 8.0, 4.0),
            0.0,
        )
        self.assertEqual(
            linear_improvement_reward(8.0, 10.0, 4.0),
            -0.5,
        )

    def test_returns_finite_python_float(self):
        reward = linear_improvement_reward(3.0, 2.0, 3.0)

        self.assertIs(type(reward), float)
        self.assertTrue(math.isfinite(reward))

    def test_rejects_invalid_inputs_and_scale(self):
        defaults = {
            "old_best": 2.0,
            "new_best": 1.0,
            "initial_improvement_scale": 2.0,
        }
        for name in defaults:
            for invalid in (
                True,
                np.bool_(False),
                np.nan,
                np.inf,
                -np.inf,
            ):
                arguments = defaults.copy()
                arguments[name] = invalid
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, name):
                        linear_improvement_reward(**arguments)

        for invalid in (0.0, -1.0):
            with self.subTest(scale=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "initial_improvement_scale",
                ):
                    linear_improvement_reward(2.0, 1.0, invalid)

    def test_rejects_subtraction_and_division_overflow(self):
        limit = np.finfo(np.float64).max
        with self.assertRaisesRegex(FloatingPointError, "improvement"):
            linear_improvement_reward(limit, -limit, 1.0)
        with self.assertRaisesRegex(FloatingPointError, "reward"):
            linear_improvement_reward(
                limit,
                0.0,
                np.finfo(np.float64).tiny,
            )


class TestLogGapReductionReward(unittest.TestCase):
    def assert_finite_float(self, value):
        self.assertIs(type(value), float)
        self.assertTrue(math.isfinite(value))

    def test_improvement_is_positive_and_matches_formula(self):
        reward = log_gap_reduction_reward(10.0, 5.0, 0.0, 10.0)
        expected = math.log(1.0 + 1e-12) - math.log(0.5 + 1e-12)

        self.assert_finite_float(reward)
        self.assertGreater(reward, 0.0)
        self.assertAlmostEqual(reward, expected, places=15)

    def test_unchanged_gap_has_zero_reward(self):
        reward = log_gap_reduction_reward(4.0, 4.0, 1.0, 3.0)

        self.assertEqual(reward, 0.0)

    def test_degradation_has_negative_reward(self):
        reward = log_gap_reduction_reward(2.0, 5.0, 0.0, 5.0)

        self.assertLess(reward, 0.0)

    def test_reaching_optimum_is_finite_and_positive(self):
        reward = log_gap_reduction_reward(1.0, 0.0, 0.0, 1.0)

        self.assert_finite_float(reward)
        self.assertGreater(reward, 0.0)

    def test_best_slightly_below_optimum_uses_zero_gap(self):
        at_optimum = log_gap_reduction_reward(1.0, 0.0, 0.0, 1.0)
        below_optimum = log_gap_reduction_reward(
            1.0,
            -np.finfo(np.float64).eps,
            0.0,
            1.0,
        )

        self.assertEqual(below_optimum, at_optimum)

    def test_extreme_finite_gaps_remain_finite(self):
        cases = (
            (1e-300, 5e-301, 0.0, 1e-300),
            (1e308, 5e307, -1e308, 1e308),
            (1e308, 1e307, 0.0, 1e-308),
        )

        for old_best, new_best, optimum, scale in cases:
            with self.subTest(
                old_best=old_best,
                new_best=new_best,
                optimum=optimum,
                scale=scale,
            ):
                reward = log_gap_reduction_reward(
                    old_best,
                    new_best,
                    optimum,
                    scale,
                )
                self.assert_finite_float(reward)
                self.assertGreater(reward, 0.0)

    def test_non_finite_inputs_are_rejected(self):
        defaults = {
            "old_best": 2.0,
            "new_best": 1.0,
            "optimum": 0.0,
            "initial_gap_scale": 2.0,
            "epsilon": 1e-12,
        }
        for name in defaults:
            for invalid in (float("nan"), float("inf"), -float("inf")):
                arguments = defaults.copy()
                arguments[name] = invalid
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, name):
                        log_gap_reduction_reward(**arguments)

    def test_scale_and_epsilon_must_be_positive(self):
        for scale in (0.0, -1.0):
            with self.subTest(initial_gap_scale=scale):
                with self.assertRaisesRegex(
                    ValueError,
                    "initial_gap_scale",
                ):
                    log_gap_reduction_reward(2.0, 1.0, 0.0, scale)

        for epsilon in (0.0, -1e-12):
            with self.subTest(epsilon=epsilon):
                with self.assertRaisesRegex(ValueError, "epsilon"):
                    log_gap_reduction_reward(
                        2.0,
                        1.0,
                        0.0,
                        2.0,
                        epsilon,
                    )

    def test_bool_and_non_real_values_are_rejected(self):
        defaults = {
            "old_best": 2.0,
            "new_best": 1.0,
            "optimum": 0.0,
            "initial_gap_scale": 2.0,
            "epsilon": 1e-12,
        }
        for name in defaults:
            for invalid in (True, np.bool_(False)):
                arguments = defaults.copy()
                arguments[name] = invalid
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, name):
                        log_gap_reduction_reward(**arguments)

        with self.assertRaisesRegex(ValueError, "old_best"):
            log_gap_reduction_reward("2.0", 1.0, 0.0, 2.0)
        with self.assertRaisesRegex(ValueError, "new_best"):
            log_gap_reduction_reward(2.0, 1.0 + 0.0j, 0.0, 2.0)

    def test_direct_and_segmented_improvements_have_equal_return(self):
        direct = log_gap_reduction_reward(100.0, 4.0, 0.0, 100.0)
        segmented = sum(
            log_gap_reduction_reward(old, new, 0.0, 100.0)
            for old, new in zip(
                (100.0, 60.0, 20.0),
                (60.0, 20.0, 4.0),
            )
        )

        self.assertAlmostEqual(segmented, direct, places=14)

    def test_different_intermediate_paths_have_equal_return(self):
        def path_return(path):
            return sum(
                log_gap_reduction_reward(old, new, 0.0, 100.0)
                for old, new in zip(path, path[1:])
            )

        monotonic_path = path_return([100.0, 80.0, 20.0, 1.0])
        nonmonotonic_path = path_return([100.0, 10.0, 30.0, 1.0])

        self.assertAlmostEqual(
            nonmonotonic_path,
            monotonic_path,
            places=14,
        )

    def test_smaller_final_gap_has_larger_cumulative_reward(self):
        larger_final_gap = log_gap_reduction_reward(
            100.0,
            10.0,
            0.0,
            100.0,
        )
        smaller_final_gap = log_gap_reduction_reward(
            100.0,
            1.0,
            0.0,
            100.0,
        )

        self.assertGreater(smaller_final_gap, larger_final_gap)

    def test_positive_objective_rescaling_preserves_reward(self):
        original = log_gap_reduction_reward(
            110.0,
            20.0,
            10.0,
            100.0,
        )
        factor = 1e6
        rescaled = log_gap_reduction_reward(
            110.0 * factor,
            20.0 * factor,
            10.0 * factor,
            100.0 * factor,
        )

        self.assertAlmostEqual(rescaled, original, places=14)

    def test_result_can_be_strictly_json_serialized(self):
        reward = log_gap_reduction_reward(3.0, 2.0, 0.0, 3.0)

        self.assert_finite_float(reward)
        json.dumps({"reward": reward}, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
