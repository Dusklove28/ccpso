from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swarm.ccpso import CCPSOSwarm, W


def sphere(positions):
    return np.sum(positions**2, axis=1)


class FakeRNG:
    def __init__(self, *uniform_results):
        self.uniform_results = [
            np.asarray(result, dtype=np.float64)
            for result in uniform_results
        ]

    def uniform(self, low, high, size):
        if not self.uniform_results:
            raise AssertionError("Unexpected call to FakeRNG.uniform().")
        result = self.uniform_results.pop(0)
        if result.shape != size:
            raise AssertionError(
                f"Fake RNG result has shape {result.shape}, expected {size}."
            )
        return result.copy()


class TestCalculateQ(unittest.TestCase):
    def setUp(self):
        self.swarm = CCPSOSwarm(
            particles=2,
            dimensions=3,
            fun=sphere,
            lower_bound=-10.0,
            upper_bound=10.0,
            max_fe=2,
            seed=0,
        )
        self.swarm.c1 = 1.0
        self.swarm.c2 = 1.0
        self.swarm.pbest_positions = np.array(
            [[2.0, -4.0, 6.0], [8.0, 10.0, -12.0]],
            dtype=np.float64,
        )
        self.swarm.gbest_position = np.array(
            [-2.0, 4.0, 14.0],
            dtype=np.float64,
        )

    def _calculate_with_weights(self, pbest_weights, gbest_weights):
        pbest_weights = np.asarray(pbest_weights, dtype=np.float64)
        gbest_weights = np.asarray(gbest_weights, dtype=np.float64)
        self.swarm.rng = FakeRNG(pbest_weights, gbest_weights)

        self.swarm._calculate_q()

        np.testing.assert_array_equal(self.swarm.c1_r1, pbest_weights)
        np.testing.assert_array_equal(self.swarm.c2_r2, gbest_weights)
        np.testing.assert_array_equal(
            self.swarm.c_sum,
            pbest_weights + gbest_weights,
        )
        self.assertTrue(self.swarm.generation_prepared)
        self.assertEqual(
            self.swarm.q_positions.shape,
            (self.swarm.particles, self.swarm.dimensions),
        )
        self.assertTrue(np.all(np.isfinite(self.swarm.q_positions)))
        return self.swarm.q_positions

    def test_zero_weights_use_pbest_gbest_mean(self):
        zeros = np.zeros((2, 3), dtype=np.float64)

        q_positions = self._calculate_with_weights(zeros, zeros)

        expected = (
            self.swarm.pbest_positions + self.swarm.gbest_position
        ) / 2.0
        np.testing.assert_array_equal(q_positions, expected)

    def test_only_pbest_weight_returns_pbest(self):
        pbest_weights = np.array(
            [[0.25, 0.5, 1.0], [0.125, 0.25, 0.5]],
            dtype=np.float64,
        )
        zeros = np.zeros((2, 3), dtype=np.float64)

        q_positions = self._calculate_with_weights(pbest_weights, zeros)

        np.testing.assert_array_equal(q_positions, self.swarm.pbest_positions)

    def test_only_gbest_weight_returns_gbest(self):
        zeros = np.zeros((2, 3), dtype=np.float64)
        gbest_weights = np.array(
            [[0.25, 0.5, 1.0], [0.125, 0.25, 0.5]],
            dtype=np.float64,
        )

        q_positions = self._calculate_with_weights(zeros, gbest_weights)

        expected = np.broadcast_to(
            self.swarm.gbest_position,
            q_positions.shape,
        )
        np.testing.assert_array_equal(q_positions, expected)

    def test_nonzero_weights_use_exact_weighted_center(self):
        pbest_weights = np.array(
            [[0.2, 0.7, 0.4], [0.9, 0.3, 0.8]],
            dtype=np.float64,
        )
        gbest_weights = np.array(
            [[0.8, 0.3, 0.6], [0.1, 0.7, 0.2]],
            dtype=np.float64,
        )

        q_positions = self._calculate_with_weights(
            pbest_weights,
            gbest_weights,
        )

        expected = (
            pbest_weights * self.swarm.pbest_positions
            + gbest_weights * self.swarm.gbest_position
        ) / (pbest_weights + gbest_weights)
        np.testing.assert_array_equal(q_positions, expected)


class TestStepUsesInstanceWeight(unittest.TestCase):
    def test_step_uses_instance_w_for_position_update(self):
        swarm = CCPSOSwarm(
            particles=2,
            dimensions=3,
            fun=sphere,
            lower_bound=-3.0,
            upper_bound=3.0,
            max_fe=4,
            seed=0,
        )
        swarm.reset()
        swarm.w = 0.25
        self.assertNotEqual(swarm.w, W)

        swarm.positions = np.array(
            [[2.0, -3.0, 4.0], [1.0, 5.0, -2.0]],
            dtype=np.float64,
        )
        swarm.previous_positions = np.array(
            [[-1.0, 2.0, 0.0], [3.0, -4.0, 1.0]],
            dtype=np.float64,
        )
        swarm.q_positions = np.array(
            [[0.5, -0.5, 1.0], [-1.0, 2.0, 0.25]],
            dtype=np.float64,
        )
        swarm.c_sum = np.array(
            [[0.2, 1.1, 0.7], [1.3, 0.4, 0.9]],
            dtype=np.float64,
        )
        swarm.generation_prepared = True
        conv = 0.8

        positions_before = swarm.positions.copy()
        previous_positions_before = swarm.previous_positions.copy()
        q_before = swarm.q_positions.copy()
        c_sum_before = swarm.c_sum.copy()

        expected_p = (
            (1 + swarm.w - c_sum_before) * (positions_before - q_before)
            - swarm.w * (previous_positions_before - q_before)
        )
        expected_positions = np.clip(
            q_before + conv * expected_p,
            swarm.lower_bound,
            swarm.upper_bound,
        )

        wrong_p = (
            (1 + W - c_sum_before) * (positions_before - q_before)
            - W * (previous_positions_before - q_before)
        )
        wrong_positions = np.clip(
            q_before + conv * wrong_p,
            swarm.lower_bound,
            swarm.upper_bound,
        )
        self.assertFalse(np.array_equal(expected_positions, wrong_positions))

        swarm.step(conv)

        np.testing.assert_array_equal(swarm.positions, expected_positions)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(swarm.positions, wrong_positions)


class TestConvergenceControlledPositionDefinition(unittest.TestCase):
    def _make_prepared_swarm(self):
        swarm = CCPSOSwarm(
            particles=2,
            dimensions=3,
            fun=sphere,
            lower_bound=-1_000_000.0,
            upper_bound=1_000_000.0,
            max_fe=4,
            seed=0,
        )
        swarm.reset()
        swarm.w = 0.25
        swarm.positions = np.array(
            [[2.0, -1.0, 3.0], [-2.0, 4.0, 1.0]],
            dtype=np.float64,
        )
        swarm.previous_positions = np.array(
            [[1.0, -2.0, 2.0], [-3.0, 2.0, 0.0]],
            dtype=np.float64,
        )
        swarm.pbest_positions = np.array(
            [[0.5, -0.5, 1.0], [-1.0, 1.5, 0.25]],
            dtype=np.float64,
        )
        swarm.gbest_position = np.array(
            [0.25, -0.25, 0.5],
            dtype=np.float64,
        )
        swarm.c1_r1 = np.array(
            [[0.25, 0.5, 0.75], [0.125, 0.625, 0.375]],
            dtype=np.float64,
        )
        swarm.c2_r2 = 1.0 - swarm.c1_r1
        swarm.c_sum = swarm.c1_r1 + swarm.c2_r2
        swarm.q_positions = (
            swarm.c1_r1 * swarm.pbest_positions
            + swarm.c2_r2 * swarm.gbest_position
        ) / swarm.c_sum
        swarm.generation_prepared = True
        return swarm

    @staticmethod
    def _snapshot_and_calculate_p(swarm):
        positions = swarm.positions.copy()
        previous_positions = swarm.previous_positions.copy()
        q_positions = swarm.q_positions.copy()
        pbest_positions = swarm.pbest_positions.copy()
        gbest_position = swarm.gbest_position.copy()
        c1_r1 = swarm.c1_r1.copy()
        c2_r2 = swarm.c2_r2.copy()
        phi = c1_r1 + c2_r2
        p = (
            (1.0 + swarm.w - phi) * (positions - q_positions)
            - swarm.w * (previous_positions - q_positions)
        )
        return {
            "positions": positions,
            "previous_positions": previous_positions,
            "q_positions": q_positions,
            "pbest_positions": pbest_positions,
            "gbest_position": gbest_position,
            "c1_r1": c1_r1,
            "c2_r2": c2_r2,
            "phi": phi,
            "p": p,
        }

    def test_c_one_matches_expanded_classic_pso_second_order_formula(self):
        swarm = self._make_prepared_swarm()
        before = self._snapshot_and_calculate_p(swarm)
        self.assertTrue(np.any(before["p"] != 0.0))

        expected_positions = (
            (1.0 + swarm.w - before["phi"]) * before["positions"]
            - swarm.w * before["previous_positions"]
            + before["c1_r1"] * before["pbest_positions"]
            + before["c2_r2"] * before["gbest_position"]
        )
        self.assertTrue(
            np.all(expected_positions > swarm.lower_bound)
            and np.all(expected_positions < swarm.upper_bound)
        )

        swarm.step(conv=1.0)

        np.testing.assert_array_equal(swarm.positions, expected_positions)
        self.assertEqual(swarm.boundary_clip_ratio, 0.0)

    def test_c_zero_places_particles_exactly_at_q(self):
        swarm = self._make_prepared_swarm()
        before = self._snapshot_and_calculate_p(swarm)
        self.assertTrue(np.any(before["p"] != 0.0))
        self.assertTrue(
            np.all(before["q_positions"] > swarm.lower_bound)
            and np.all(before["q_positions"] < swarm.upper_bound)
        )

        swarm.step(conv=0.0)

        np.testing.assert_array_equal(
            swarm.positions,
            before["q_positions"],
        )
        self.assertEqual(swarm.boundary_clip_ratio, 0.0)


class TestBoundaryClipRatio(unittest.TestCase):
    def _run_candidate_positions(self, candidate_positions):
        swarm = CCPSOSwarm(
            particles=2,
            dimensions=3,
            fun=sphere,
            lower_bound=-1.0,
            upper_bound=1.0,
            max_fe=4,
            seed=0,
        )
        swarm.reset()
        self.assertEqual(swarm.boundary_clip_ratio, 0.0)

        candidate_positions = np.asarray(
            candidate_positions,
            dtype=np.float64,
        )
        swarm.positions = candidate_positions.copy()
        swarm.previous_positions = candidate_positions.copy()
        swarm.q_positions = candidate_positions.copy()
        swarm.c_sum = np.ones_like(candidate_positions)
        swarm.generation_prepared = True

        swarm.step(conv=0.75)

        np.testing.assert_array_equal(
            swarm.positions,
            np.clip(candidate_positions, -1.0, 1.0),
        )
        self.assertGreaterEqual(swarm.boundary_clip_ratio, 0.0)
        self.assertLessEqual(swarm.boundary_clip_ratio, 1.0)
        return swarm.boundary_clip_ratio

    def test_no_coordinates_clipped_has_zero_ratio(self):
        ratio = self._run_candidate_positions(
            [[0.0, -0.5, 0.75], [1.0, -1.0, 0.25]]
        )

        self.assertEqual(ratio, 0.0)

    def test_partially_clipped_coordinates_have_exact_ratio(self):
        ratio = self._run_candidate_positions(
            [[2.0, -0.5, -3.0], [0.0, 1.2, 0.25]]
        )

        self.assertEqual(ratio, 3 / 6)

    def test_all_coordinates_clipped_has_one_ratio(self):
        ratio = self._run_candidate_positions(
            [[2.0, -2.0, 3.0], [-4.0, 5.0, -6.0]]
        )

        self.assertEqual(ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
