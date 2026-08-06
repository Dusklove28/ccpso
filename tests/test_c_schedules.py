from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.c_schedules import (
    c_value_to_action,
    make_fixed_c_schedule,
    make_linear_c_schedule,
)


class TestCSchedules(unittest.TestCase):
    def assert_valid_schedule(self, schedule, expected_shape):
        self.assertIsInstance(schedule, np.ndarray)
        self.assertEqual(schedule.dtype, np.float64)
        self.assertEqual(schedule.shape, expected_shape)
        self.assertTrue(np.all(np.isfinite(schedule)))

    def test_fixed_c_schedule(self):
        schedule = make_fixed_c_schedule(
            np.float32(0.75),
            np.int64(4),
        )

        self.assert_valid_schedule(schedule, (4,))
        np.testing.assert_array_equal(
            schedule,
            np.full(4, 0.75, dtype=np.float64),
        )

    def test_decreasing_linear_c_schedule(self):
        schedule = make_linear_c_schedule(1.5, 0.0, 4)

        self.assert_valid_schedule(schedule, (4,))
        np.testing.assert_array_equal(
            schedule,
            np.array([1.5, 1.0, 0.5, 0.0], dtype=np.float64),
        )
        self.assertEqual(schedule[0], 1.5)
        self.assertEqual(schedule[-1], 0.0)

    def test_increasing_linear_c_schedule(self):
        schedule = make_linear_c_schedule(-1.0, 1.0, 5)

        self.assert_valid_schedule(schedule, (5,))
        np.testing.assert_array_equal(
            schedule,
            np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
        )

    def test_single_step_uses_c_start(self):
        schedule = make_linear_c_schedule(1.25, -4.0, 1)

        self.assert_valid_schedule(schedule, (1,))
        np.testing.assert_array_equal(
            schedule,
            np.array([1.25], dtype=np.float64),
        )

    def test_c_value_to_action_endpoints_and_midpoint(self):
        cases = (
            (0.0, -1.0),
            (0.75, 0.0),
            (1.5, 1.0),
        )
        for c_value, expected_action in cases:
            with self.subTest(c_value=c_value):
                action = c_value_to_action(c_value, 0.0, 1.5)
                self.assertIs(type(action), float)
                self.assertTrue(np.isfinite(action))
                self.assertEqual(action, expected_action)
                self.assertGreaterEqual(action, -1.0)
                self.assertLessEqual(action, 1.0)

    def test_equal_boundaries_have_one_canonical_action(self):
        self.assertEqual(c_value_to_action(0.5, 0.5, 0.5), 0.0)

        for c_value in (0.4, 0.6):
            with self.subTest(c_value=c_value):
                with self.assertRaises(ValueError):
                    c_value_to_action(c_value, 0.5, 0.5)

    def test_invalid_num_steps_are_rejected(self):
        invalid_values = (
            0,
            -1,
            True,
            np.bool_(False),
            1.0,
            np.float64(2.0),
            np.nan,
            np.inf,
            "2",
            None,
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    make_fixed_c_schedule(0.5, value)
                with self.assertRaises(ValueError):
                    make_linear_c_schedule(1.0, 0.0, value)

    def test_invalid_numeric_inputs_are_rejected(self):
        invalid_values = (
            True,
            np.bool_(False),
            np.nan,
            np.inf,
            -np.inf,
            "0.5",
            None,
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    make_fixed_c_schedule(value, 3)
                with self.assertRaises(ValueError):
                    make_linear_c_schedule(value, 0.0, 3)
                with self.assertRaises(ValueError):
                    make_linear_c_schedule(1.0, value, 3)
                with self.assertRaises(ValueError):
                    c_value_to_action(value, 0.0, 1.5)
                with self.assertRaises(ValueError):
                    c_value_to_action(0.5, value, 1.5)
                with self.assertRaises(ValueError):
                    c_value_to_action(0.5, 0.0, value)

    def test_out_of_range_and_reversed_bounds_are_rejected(self):
        for c_value in (-0.1, 1.6):
            with self.subTest(c_value=c_value):
                with self.assertRaises(ValueError):
                    c_value_to_action(c_value, 0.0, 1.5)

        with self.assertRaises(ValueError):
            c_value_to_action(0.5, 1.0, 0.0)

    def test_extreme_finite_linear_inputs_cannot_return_infinity(self):
        with self.assertRaises(ValueError):
            make_linear_c_schedule(1e308, -1e308, 3)


if __name__ == "__main__":
    unittest.main()
