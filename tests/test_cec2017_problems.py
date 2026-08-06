from collections import Counter
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np
from cec2017 import transforms
from cec2017.functions import all_functions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problems import (
    CEC2017_FUNCTION_IDS,
    CEC2017_REPRESENTATIVE_IDS,
    CEC2017_SOURCE_FUNCTION_IDS,
    CEC2017_SUPPORTED_DIMENSIONS,
    CEC2017ProblemSpec,
    ProblemSpec,
    cec2017_category,
    cec2017_max_fe,
    make_cec2017_problem,
    public_to_source_function_id,
)


COMPOSITION_COMPONENT_COUNTS = {
    21: 3,
    22: 3,
    23: 4,
    24: 4,
    25: 5,
    26: 5,
    27: 6,
    28: 6,
    29: 3,
    30: 3,
}


class TestCEC2017Numbering(unittest.TestCase):
    def test_public_and_source_ids_are_complete_and_one_to_one(self):
        self.assertEqual(CEC2017_FUNCTION_IDS, tuple(range(1, 30)))
        self.assertEqual(
            CEC2017_SOURCE_FUNCTION_IDS,
            (1, *range(3, 31)),
        )
        mapped = tuple(
            public_to_source_function_id(function_id)
            for function_id in CEC2017_FUNCTION_IDS
        )
        self.assertEqual(mapped, CEC2017_SOURCE_FUNCTION_IDS)
        self.assertEqual(len(set(mapped)), 29)
        self.assertNotIn(2, mapped)
        self.assertEqual(public_to_source_function_id(29), 30)
        self.assertIs(type(public_to_source_function_id(np.int64(5))), int)

    def test_mapping_rejects_non_public_ids_and_invalid_types(self):
        invalid_values = (0, 30, -1, True, np.bool_(False), 1.0, "1")
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    public_to_source_function_id(value)
                with self.assertRaises(ValueError):
                    make_cec2017_problem(value, 10)

    def test_categories_have_the_protocol_counts(self):
        categories = [
            cec2017_category(function_id)
            for function_id in CEC2017_FUNCTION_IDS
        ]
        self.assertEqual(
            Counter(categories),
            {
                "unimodal": 2,
                "multimodal": 7,
                "hybrid": 10,
                "composition": 10,
            },
        )
        self.assertEqual(cec2017_category(1), "unimodal")
        self.assertEqual(cec2017_category(5), "multimodal")
        self.assertEqual(cec2017_category(12), "hybrid")
        self.assertEqual(cec2017_category(23), "composition")

    def test_formal_dimensions_and_fe_budget_are_strict(self):
        self.assertEqual(
            CEC2017_SUPPORTED_DIMENSIONS,
            (10, 30, 50, 100),
        )
        for dimensions in CEC2017_SUPPORTED_DIMENSIONS:
            with self.subTest(dimensions=dimensions):
                problem = make_cec2017_problem(1, np.int64(dimensions))
                self.assertEqual(problem.dimensions, dimensions)
                self.assertEqual(
                    cec2017_max_fe(np.int64(dimensions)),
                    10_000 * dimensions,
                )

        for dimensions in (2, 20, 0, -10, 11, True, 10.0, "10"):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    make_cec2017_problem(1, dimensions)
                with self.assertRaises(ValueError):
                    cec2017_max_fe(dimensions)


class TestCEC2017ProblemMetadata(unittest.TestCase):
    def test_all_29_problem_specs_and_lightweight_values(self):
        positions = np.vstack(
            [
                np.zeros(10, dtype=np.float64),
                np.linspace(-1.0, 1.0, 10, dtype=np.float64),
            ]
        )
        for public_id in CEC2017_FUNCTION_IDS:
            with self.subTest(public_id=public_id):
                source_id = public_to_source_function_id(public_id)
                problem = make_cec2017_problem(public_id, 10)
                values = problem.evaluate(positions)

                self.assertIsInstance(problem, ProblemSpec)
                self.assertIsInstance(problem, CEC2017ProblemSpec)
                self.assertEqual(problem.suite, "cec2017")
                self.assertEqual(problem.problem_id, public_id)
                self.assertEqual(problem.source_function_id, source_id)
                self.assertEqual(
                    problem.category,
                    cec2017_category(public_id),
                )
                self.assertEqual(problem.dimensions, 10)
                np.testing.assert_array_equal(
                    problem.lower_bound,
                    np.full(10, -100.0, dtype=np.float64),
                )
                np.testing.assert_array_equal(
                    problem.upper_bound,
                    np.full(10, 100.0, dtype=np.float64),
                )
                self.assertEqual(problem.optimum, 100.0 * source_id)
                self.assertEqual(values.shape, (2,))
                self.assertEqual(values.dtype, np.float64)
                self.assertTrue(np.all(np.isfinite(values)))

    def test_representative_mapping_and_source_equivalence(self):
        self.assertEqual(CEC2017_REPRESENTATIVE_IDS, (1, 5, 12, 23))
        representatives = {
            1: (1, "unimodal"),
            5: (6, "multimodal"),
            12: (13, "hybrid"),
            23: (24, "composition"),
        }
        positions = np.array(
            [
                np.linspace(-3.0, 3.0, 10),
                np.linspace(2.5, -2.5, 10),
            ],
            dtype=np.float64,
        )
        for public_id, (source_id, category) in representatives.items():
            with self.subTest(public_id=public_id):
                problem = make_cec2017_problem(public_id, 10)
                expected = all_functions[source_id - 1](positions)
                actual = problem.evaluate(positions)
                self.assertEqual(problem.source_function_id, source_id)
                self.assertEqual(problem.category, category)
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(np.all(np.isfinite(actual)))

    def test_representatives_evaluate_at_every_formal_dimension(self):
        for dimensions in CEC2017_SUPPORTED_DIMENSIONS:
            positions = np.zeros((1, dimensions), dtype=np.float64)
            for public_id in CEC2017_REPRESENTATIVE_IDS:
                with self.subTest(
                    public_id=public_id,
                    dimensions=dimensions,
                ):
                    values = make_cec2017_problem(
                        public_id,
                        dimensions,
                    ).evaluate(positions)
                    self.assertEqual(values.shape, (1,))
                    self.assertEqual(values.dtype, np.float64)
                    self.assertTrue(np.all(np.isfinite(values)))

    def test_invalid_positions_are_still_rejected_by_problem_spec(self):
        problem = make_cec2017_problem(1, 10)
        for positions in (
            np.zeros(10),
            np.zeros((2, 9)),
            np.zeros((1, 2, 10)),
        ):
            with self.subTest(shape=positions.shape):
                with self.assertRaises(ValueError):
                    problem.evaluate(positions)

        for value in (np.nan, np.inf, -np.inf):
            positions = np.zeros((1, 10), dtype=np.float64)
            positions[0, 0] = value
            with self.subTest(value=value):
                with self.assertRaises(FloatingPointError):
                    problem.evaluate(positions)


class TestCEC2017CompositionCompatibility(unittest.TestCase):
    def test_raw_source_exposes_exact_shift_point_problem(self):
        source_id = 24
        shift = transforms.shifts_cf[source_id - 21][0, :10]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw_value = all_functions[source_id - 1](shift[None, :])[0]

        self.assertTrue(np.isnan(raw_value))
        self.assertTrue(
            any(issubclass(item.category, RuntimeWarning) for item in caught)
        )

    def test_every_actual_composition_shift_has_defined_value(self):
        for source_id, component_count in (
            COMPOSITION_COMPONENT_COUNTS.items()
        ):
            public_id = source_id - 1
            shifts = transforms.shifts_cf[source_id - 21][
                :component_count,
                :10,
            ]
            problem = make_cec2017_problem(public_id, 10)
            with self.subTest(source_id=source_id):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    values = problem.evaluate(shifts)

                expected = problem.optimum + 100.0 * np.arange(
                    component_count,
                    dtype=np.float64,
                )
                np.testing.assert_array_equal(values, expected)
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertFalse(
                    any(
                        issubclass(item.category, RuntimeWarning)
                        for item in caught
                    )
                )

    def test_representative_first_shift_uses_source_optimum(self):
        problem = make_cec2017_problem(23, 10)
        self.assertEqual(problem.source_function_id, 24)
        first_shift = transforms.shifts_cf[24 - 21][0, :10]
        value = problem.evaluate(first_shift[None, :])[0]
        self.assertEqual(value, 2400.0)
        self.assertNotEqual(value, 2300.0)

    def test_ordinary_points_match_each_raw_composition_function(self):
        rng = np.random.default_rng(20260806)
        positions = rng.uniform(-75.0, 75.0, size=(3, 10))
        for source_id in COMPOSITION_COMPONENT_COUNTS:
            public_id = source_id - 1
            with self.subTest(source_id=source_id):
                actual = make_cec2017_problem(
                    public_id,
                    10,
                ).evaluate(positions)
                expected = all_functions[source_id - 1](positions)
                np.testing.assert_array_equal(actual, expected)

    def test_one_float_spacing_does_not_trigger_replacement(self):
        source_id = 30
        public_id = 29
        point = transforms.shifts_cf[source_id - 21][0, :10].copy()
        point[0] = np.nextafter(point[0], np.inf)

        actual = make_cec2017_problem(public_id, 10).evaluate(
            point[None, :]
        )[0]
        expected = all_functions[source_id - 1](point[None, :])[0]

        self.assertEqual(actual, expected)
        self.assertNotEqual(actual, 3000.0)


if __name__ == "__main__":
    unittest.main()
