from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.factory import make_ccpso_env
from evaluation.common import (
    STATE_FIELDS,
    serialize_problem as legacy_serialize_problem,
)
from evaluation.evaluate_c_baselines import (
    evaluate_fixed_c,
    evaluate_linear_c,
)
from problems import (
    ProblemSpec,
    cec2017_category,
    make_cec2017_problem,
    make_classic_problem,
    public_to_source_function_id,
    serialize_problem,
)


REPRESENTATIVES = {
    1: (1, "unimodal"),
    5: (6, "multimodal"),
    12: (13, "hybrid"),
    23: (24, "composition"),
}
SEEDS = [20260806, 20260807]


class TestProblemMetadataSerialization(unittest.TestCase):
    def test_legacy_import_reexports_the_unique_serializer(self):
        self.assertIs(legacy_serialize_problem, serialize_problem)

    def test_classic_metadata_is_exactly_unchanged(self):
        cases = {
            "sphere": ("Sphere", -100.0, 100.0),
            "rastrigin": ("Rastrigin", -5.12, 5.12),
            "rosenbrock": ("Rosenbrock", -30.0, 30.0),
        }
        for problem_id, (name, lower, upper) in cases.items():
            with self.subTest(problem_id=problem_id):
                problem = make_classic_problem(problem_id, 3)
                metadata = serialize_problem(problem)
                self.assertEqual(
                    metadata,
                    {
                        "suite": "classic",
                        "problem_id": problem_id,
                        "name": name,
                        "dimensions": 3,
                        "lower_bound": [lower] * 3,
                        "upper_bound": [upper] * 3,
                        "optimum": 0.0,
                    },
                )
                self.assertNotIn("source_function_id", metadata)
                self.assertNotIn("category", metadata)
                json.dumps(metadata, allow_nan=False)

    def test_suite_name_alone_does_not_add_cec_fields(self):
        generic_problem = ProblemSpec(
            suite="cec2017",
            problem_id="generic",
            name="Generic problem",
            dimensions=2,
            lower_bound=-1.0,
            upper_bound=1.0,
            optimum=0.0,
            objective=lambda positions: np.sum(positions**2, axis=1),
        )

        metadata = serialize_problem(generic_problem)

        self.assertNotIn("source_function_id", metadata)
        self.assertNotIn("category", metadata)

    def test_all_cec2017_metadata_keeps_both_identifiers(self):
        source_ids = []
        for public_id in range(1, 30):
            with self.subTest(public_id=public_id):
                problem = make_cec2017_problem(public_id, 10)
                metadata = serialize_problem(problem)
                source_id = public_to_source_function_id(public_id)
                source_ids.append(metadata["source_function_id"])

                self.assertEqual(metadata["suite"], "cec2017")
                self.assertEqual(metadata["problem_id"], public_id)
                self.assertEqual(
                    metadata["source_function_id"],
                    source_id,
                )
                self.assertEqual(
                    metadata["category"],
                    cec2017_category(public_id),
                )
                self.assertEqual(metadata["optimum"], 100.0 * source_id)
                json.dumps(metadata, allow_nan=False)

        self.assertEqual(source_ids, [1, *range(3, 31)])
        self.assertNotIn(2, source_ids)

    def test_representative_metadata_is_exact(self):
        for public_id, (source_id, category) in REPRESENTATIVES.items():
            with self.subTest(public_id=public_id):
                metadata = serialize_problem(
                    make_cec2017_problem(public_id, 10)
                )
                self.assertEqual(metadata["problem_id"], public_id)
                self.assertEqual(metadata["source_function_id"], source_id)
                self.assertEqual(metadata["category"], category)

    def test_invalid_type_and_inconsistent_cec_metadata_are_rejected(self):
        with self.assertRaises(TypeError):
            serialize_problem(object())

        problem = make_cec2017_problem(1, 10)
        inconsistent_cases = (
            replace(problem, source_function_id=3),
            replace(problem, category="hybrid"),
            replace(problem, optimum=999.0),
            replace(problem, suite="classic"),
        )
        for inconsistent in inconsistent_cases:
            with self.subTest(problem=inconsistent):
                with self.assertRaises(ValueError):
                    serialize_problem(inconsistent)


class TestCEC2017EnvironmentLifecycle(unittest.TestCase):
    def assert_finite_environment(self, env, observation, reward, info):
        if observation.shape != (6,) or observation.dtype != np.float32:
            raise AssertionError(
                f"invalid observation shape/dtype: "
                f"{observation.shape}/{observation.dtype}"
            )
        if not np.all(np.isfinite(observation)):
            raise AssertionError("observation contains non-finite values")
        if not np.isfinite(reward):
            raise AssertionError("reward is non-finite")
        for array in (
            env.swarm.positions,
            env.swarm.fitness,
            env.swarm.q_positions,
            env.swarm.gbest_position,
        ):
            if not np.all(np.isfinite(array)):
                raise AssertionError("swarm state contains non-finite values")
        for value in (
            env.swarm.gbest_fitness,
            info["gbest_fitness"],
            info["gap"],
        ):
            if not np.isfinite(value):
                raise AssertionError("fitness diagnostic is non-finite")

    def test_representatives_run_reset_step_terminal_lifecycle(self):
        action = np.array([0.0], dtype=np.float32)
        for public_id, (source_id, category) in REPRESENTATIVES.items():
            with self.subTest(public_id=public_id):
                problem = make_cec2017_problem(public_id, 10)
                metadata = serialize_problem(problem)
                env = make_ccpso_env(
                    problem,
                    particles=4,
                    max_fe=16,
                    seed=20260806 + public_id,
                )
                try:
                    observation, info = env.reset(
                        seed=20260806 + public_id
                    )
                    self.assertEqual(observation.shape, (6,))
                    self.assertEqual(observation.dtype, np.float32)
                    self.assertTrue(
                        env.observation_space.contains(observation)
                    )
                    self.assertTrue(np.all(np.isfinite(observation)))
                    self.assertEqual(env.swarm.fe_count, 4)
                    self.assertEqual(info["function_id"], public_id)

                    self.assertEqual(metadata["problem_id"], public_id)
                    self.assertEqual(
                        metadata["source_function_id"],
                        source_id,
                    )
                    self.assertEqual(metadata["category"], category)
                    if public_id == 23:
                        self.assertEqual(problem.optimum, 2400.0)

                    for step_index in range(1, 4):
                        (
                            observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = env.step(action)
                        self.assertTrue(
                            env.observation_space.contains(observation)
                        )
                        self.assertIs(truncated, False)
                        self.assertEqual(info["function_id"], public_id)
                        self.assert_finite_environment(
                            env,
                            observation,
                            reward,
                            info,
                        )
                        self.assertIs(terminated, step_index == 3)

                    self.assertEqual(env.swarm.fe_count, 16)
                    self.assertIs(info["terminal"], True)
                    with self.assertRaises(RuntimeError):
                        env.step(action)
                finally:
                    env.close()

    def test_representatives_pass_gymnasium_checker(self):
        for public_id in REPRESENTATIVES:
            with self.subTest(public_id=public_id):
                env = make_ccpso_env(
                    make_cec2017_problem(public_id, 10),
                    particles=4,
                    max_fe=16,
                    seed=20260900 + public_id,
                )
                try:
                    check_env(env, skip_render_check=True)
                finally:
                    env.close()


class TestCEC2017CBaselineSmokeEvaluation(unittest.TestCase):
    def assert_finite_result(self, result, public_id, source_id, category):
        self.assertEqual(result["problem"]["problem_id"], public_id)
        self.assertEqual(
            result["problem"]["source_function_id"],
            source_id,
        )
        self.assertEqual(result["problem"]["category"], category)
        self.assertEqual(
            [episode["seed"] for episode in result["episodes"]],
            SEEDS,
        )
        self.assertEqual(len(result["steps"]), 6)
        for episode in result["episodes"]:
            self.assertEqual(episode["steps"], 3)
            self.assertEqual(episode["final_fe"], 16)
            for field in (
                "final_best",
                "gap",
                "return",
                "c_mean",
                "c_min",
                "c_max",
            ):
                self.assertTrue(np.isfinite(episode[field]))
        for step in result["steps"]:
            for field in (
                "gbest_fitness",
                "gap",
                "raw_action",
                "c_value",
                "reward",
                *STATE_FIELDS,
            ):
                self.assertTrue(np.isfinite(step[field]))
        self.assertIsInstance(
            json.dumps(result, allow_nan=False),
            str,
        )

    def test_fixed_and_linear_c_complete_for_all_representatives(self):
        fixed_action = float(np.float32(2.0 * (1.0 / 1.5) - 1.0))
        expected_fixed_c = (fixed_action + 1.0) * 0.5 * 1.5

        for public_id, (source_id, category) in REPRESENTATIVES.items():
            with self.subTest(public_id=public_id):
                problem = make_cec2017_problem(public_id, 10)
                fixed_result = evaluate_fixed_c(
                    problem,
                    particles=4,
                    max_fe=16,
                    seeds=SEEDS,
                    c_value=1.0,
                )
                linear_result = evaluate_linear_c(
                    problem,
                    particles=4,
                    max_fe=16,
                    seeds=SEEDS,
                )

                self.assert_finite_result(
                    fixed_result,
                    public_id,
                    source_id,
                    category,
                )
                self.assert_finite_result(
                    linear_result,
                    public_id,
                    source_id,
                    category,
                )
                self.assertTrue(all(
                    step["c_value"] == expected_fixed_c
                    for step in fixed_result["steps"]
                ))
                for seed in SEEDS:
                    seed_steps = [
                        step
                        for step in linear_result["steps"]
                        if step["seed"] == seed
                    ]
                    self.assertEqual(
                        [step["c_value"] for step in seed_steps],
                        [1.5, 0.75, 0.0],
                    )


if __name__ == "__main__":
    unittest.main()
