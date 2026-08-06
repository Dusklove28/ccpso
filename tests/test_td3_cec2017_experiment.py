import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problems import make_cec2017_problem
from training.td3_experiment import (
    TD3ProblemExperimentConfig,
    TD3ProblemExperimentResult,
    run_td3_problem,
)
from training.td3_online import TD3OnlineConfig


REPRESENTATIVES = {
    1: (1, "unimodal"),
    5: (6, "multimodal"),
    12: (13, "hybrid"),
    23: (24, "composition"),
}
STATE_FIELDS = (
    "state_fe_progress",
    "state_recent_progress",
    "state_position_diversity",
    "state_q_diversity",
    "state_movement",
    "state_stagnation",
)


class TestTD3CEC2017Experiment(unittest.TestCase):
    @staticmethod
    def make_config():
        return TD3ProblemExperimentConfig(
            particles=4,
            max_fe=16,
            buffer_capacity=16,
            device="cpu",
            discount=1.0,
            reward_mode="linear_improvement",
            online=TD3OnlineConfig(
                episodes=1,
                learning_starts=1,
                batch_size=1,
                exploration_noise=0.05,
                updates_per_step=1,
                seed=20260806,
            ),
        )

    def assert_complete_finite_result(
        self,
        result,
        public_id,
        source_id,
        category,
    ):
        self.assertIsInstance(result, TD3ProblemExperimentResult)
        self.assertEqual(result.problem.problem_id, public_id)
        self.assertEqual(result.problem.source_function_id, source_id)
        self.assertEqual(result.problem.category, category)
        self.assertEqual(result.problem.dimensions, 10)

        metadata = result.problem_metadata
        self.assertEqual(metadata["suite"], "cec2017")
        self.assertEqual(metadata["problem_id"], public_id)
        self.assertEqual(metadata["source_function_id"], source_id)
        self.assertEqual(metadata["category"], category)
        self.assertEqual(metadata["optimum"], 100.0 * source_id)
        if public_id == 23:
            self.assertEqual(metadata["optimum"], 2400.0)

        self.assertEqual(result.config["dimensions"], 10)
        self.assertNotIn("problem_name", result.config)
        self.assertEqual(result.config["device"], "cpu")
        self.assertEqual(
            result.config["environment"]["reward_mode"],
            "linear_improvement",
        )
        self.assertEqual(result.config["td3"]["discount"], 1.0)

        records = result.training_records
        self.assertEqual(records["total_steps"], 3)
        self.assertEqual(records["total_updates"], 3)
        self.assertEqual(len(records["episodes"]), 1)
        self.assertEqual(len(records["steps"]), 3)
        self.assertEqual(len(records["updates"]), 3)

        episode = records["episodes"][0]
        self.assertEqual(episode["steps"], 3)
        self.assertEqual(episode["final_fe"], 16)
        self.assertIs(episode["terminated"], True)
        self.assertIs(episode["truncated"], False)
        self.assertEqual(episode["reward_mode"], "linear_improvement")
        for field in (
            "return",
            "final_best",
            "c_mean",
            "c_min",
            "c_max",
            "initial_improvement_scale",
            "initial_gap_scale",
        ):
            self.assertTrue(np.isfinite(episode[field]), msg=field)

        for step_index, step in enumerate(records["steps"], start=1):
            self.assertEqual(step["episode_step"], step_index)
            self.assertEqual(step["fe_count"], 4 * (step_index + 1))
            for field in (
                "reward",
                "reward_progress",
                "gbest_fitness",
                "gap",
                "raw_action",
                "c_value",
                *STATE_FIELDS,
            ):
                self.assertTrue(np.isfinite(step[field]), msg=field)

        for update in records["updates"]:
            for field in (
                "critic_loss",
                "target_q_mean",
                "q1_mean",
                "q2_mean",
            ):
                self.assertTrue(np.isfinite(update[field]), msg=field)
            if update["actor_loss"] is not None:
                self.assertTrue(np.isfinite(update["actor_loss"]))
        self.assertTrue(any(
            update["actor_loss"] is not None
            for update in records["updates"]
        ))

        for network_name in (
            "actor",
            "actor_target",
            "critic",
            "critic_target",
        ):
            for parameter in getattr(
                result.policy,
                network_name,
            ).parameters():
                self.assertEqual(parameter.device, torch.device("cpu"))
                self.assertTrue(torch.isfinite(parameter).all().item())

        terminal_masks = result.replay_buffer.bootstrap_mask[
            :len(result.replay_buffer),
            0,
        ]
        self.assertEqual(len(result.replay_buffer), 3)
        self.assertEqual(
            int(np.count_nonzero(terminal_masks == 0.0)),
            1,
        )
        self.assertIsInstance(
            json.dumps(
                {
                    "config": result.config,
                    "problem": result.problem_metadata,
                    "training": result.training_records,
                },
                allow_nan=False,
            ),
            str,
        )

    def test_four_representatives_complete_lightweight_training(self):
        config = self.make_config()
        f23_result = None
        f23_actor_parameters = None

        for public_id, (source_id, category) in REPRESENTATIVES.items():
            with self.subTest(public_id=public_id):
                result = run_td3_problem(
                    make_cec2017_problem(public_id, 10),
                    config,
                )
                self.assert_complete_finite_result(
                    result,
                    public_id,
                    source_id,
                    category,
                )
                if public_id == 23:
                    f23_result = result
                    f23_actor_parameters = [
                        parameter.detach().clone()
                        for parameter in result.policy.actor.parameters()
                    ]

        repeated = run_td3_problem(
            make_cec2017_problem(23, 10),
            config,
        )
        self.assertEqual(
            f23_result.training_records,
            repeated.training_records,
        )
        for expected, actual in zip(
            f23_actor_parameters,
            repeated.policy.actor.parameters(),
        ):
            self.assertTrue(torch.equal(expected, actual.detach()))

    def test_common_entry_rejects_invalid_problem_and_config_types(self):
        config = self.make_config()
        problem = make_cec2017_problem(1, 10)

        with self.assertRaises(TypeError):
            run_td3_problem(object(), config)
        with self.assertRaises(TypeError):
            run_td3_problem(problem, object())


if __name__ == "__main__":
    unittest.main()
