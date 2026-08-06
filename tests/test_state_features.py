from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.state_features import (
    build_relative_log_state,
    log_stagnation_feature,
    mean_radial_diversity,
    relative_log_transform,
    safe_initial_spatial_scale,
)


class RelativeLogStateFeatureTests(unittest.TestCase):
    def test_relative_log_reference_values_are_ordered(self):
        ratios = (1.0, 1e-2, 1e-4, 1e-6, 1e-8, 0.0)
        actual = [relative_log_transform(value, k=8) for value in ratios]
        np.testing.assert_allclose(
            actual,
            [0.5, 0.375, 0.25, 0.125, 0.0, 0.0],
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(all(left >= right for left, right in zip(actual, actual[1:])))
        self.assertEqual(len(set(actual[:5])), 5)

    def test_initial_ratio_is_half_and_contraction_expansion_straddle_half(self):
        self.assertEqual(relative_log_transform(1.0), 0.5)
        self.assertLess(relative_log_transform(0.1), 0.5)
        self.assertGreater(relative_log_transform(10.0), 0.5)

    def test_zero_spatial_scale_uses_search_relative_positive_floor(self):
        first = safe_initial_spatial_scale(0.0, 200.0)
        second = safe_initial_spatial_scale(0.0, 2000.0)
        self.assertGreater(first, 0.0)
        self.assertTrue(np.isfinite(first))
        self.assertEqual(second / first, 10.0)

    def test_stagnation_uses_episode_budget_not_fixed_ten(self):
        values = [
            log_stagnation_feature(step, 100)
            for step in (0, 1, 10, 99, 100)
        ]
        self.assertEqual(values[0], 0.0)
        self.assertTrue(all(left < right for left, right in zip(values, values[1:])))
        self.assertLess(values[2], 1.0)
        self.assertEqual(values[-1], 1.0)

    def test_improvement_is_translation_and_positive_scale_invariant(self):
        base = build_relative_log_state(
            fe_progress=0.2,
            recent_improvement=20.0,
            initial_improvement_scale=100.0,
            position_diversity=2.0,
            initial_position_scale=4.0,
            q_diversity=3.0,
            initial_q_scale=6.0,
            movement=1.0,
            stagnation_steps=3,
            max_episode_updates=100,
        )
        scaled = build_relative_log_state(
            fe_progress=0.2,
            recent_improvement=20.0 * 17.0,
            initial_improvement_scale=100.0 * 17.0,
            position_diversity=2.0,
            initial_position_scale=4.0,
            q_diversity=3.0,
            initial_q_scale=6.0,
            movement=1.0,
            stagnation_steps=3,
            max_episode_updates=100,
        )
        self.assertEqual(base[1], scaled[1])
        # Adding a constant to old/new fitness changes neither improvement nor S0.
        translated = build_relative_log_state(
            fe_progress=0.2,
            recent_improvement=(1020.0 - 1000.0),
            initial_improvement_scale=100.0,
            position_diversity=2.0,
            initial_position_scale=4.0,
            q_diversity=3.0,
            initial_q_scale=6.0,
            movement=1.0,
            stagnation_steps=3,
            max_episode_updates=100,
        )
        self.assertEqual(base[1], translated[1])

    def test_coordinate_scaling_preserves_three_spatial_features(self):
        positions = np.array(
            [[-2.0, 1.0], [0.0, -1.0], [3.0, 2.0]],
            dtype=np.float64,
        )
        q_positions = positions * 0.6 + 0.25
        position_scale = mean_radial_diversity(positions)
        q_scale = mean_radial_diversity(q_positions)
        base = build_relative_log_state(
            fe_progress=0.3,
            recent_improvement=1.0,
            initial_improvement_scale=2.0,
            position_diversity=position_scale * 0.2,
            initial_position_scale=position_scale,
            q_diversity=q_scale * 1.5,
            initial_q_scale=q_scale,
            movement=position_scale * 0.01,
            stagnation_steps=2,
            max_episode_updates=50,
        )
        factor = 37.0
        scaled = build_relative_log_state(
            fe_progress=0.3,
            recent_improvement=1.0,
            initial_improvement_scale=2.0,
            position_diversity=mean_radial_diversity(positions * factor) * 0.2,
            initial_position_scale=mean_radial_diversity(positions * factor),
            q_diversity=mean_radial_diversity(q_positions * factor) * 1.5,
            initial_q_scale=mean_radial_diversity(q_positions * factor),
            movement=position_scale * 0.01 * factor,
            stagnation_steps=2,
            max_episode_updates=50,
        )
        np.testing.assert_array_equal(base[2:5], scaled[2:5])


if __name__ == "__main__":
    unittest.main()
