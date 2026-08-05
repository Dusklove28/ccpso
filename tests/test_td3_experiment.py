from pathlib import Path
import json
import sys
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.td3_experiment import (
    ClassicTD3ExperimentConfig,
    run_classic_td3,
)
from training.td3_online import TD3OnlineConfig


class TestClassicTD3Experiment(unittest.TestCase):
    @staticmethod
    def make_config():
        return ClassicTD3ExperimentConfig(
            problem_name="sphere",
            dimensions=3,
            particles=4,
            max_fe=16,
            buffer_capacity=16,
            device="cpu",
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            online=TD3OnlineConfig(
                episodes=2,
                learning_starts=5,
                batch_size=2,
                exploration_noise=0.05,
                updates_per_step=1,
                seed=123,
            ),
        )

    def assert_complete_result(self, result):
        self.assertEqual(result.problem.suite, "classic")
        self.assertEqual(result.problem.problem_id, "sphere")
        self.assertEqual(result.problem.name, "Sphere")
        self.assertEqual(result.problem.dimensions, 3)
        self.assertEqual(result.problem.optimum, 0.0)

        self.assertEqual(result.config["device"], "cpu")
        self.assertEqual(result.config["dimensions"], 3)
        self.assertEqual(result.config["particles"], 4)
        self.assertEqual(result.config["max_fe"], 16)
        self.assertEqual(result.config["buffer_capacity"], 16)
        self.assertEqual(result.config["online"]["episodes"], 2)

        metadata = result.problem_metadata
        self.assertEqual(metadata["suite"], "classic")
        self.assertEqual(metadata["problem_id"], "sphere")
        self.assertEqual(metadata["dimensions"], 3)
        self.assertEqual(metadata["lower_bound"], [-100.0] * 3)
        self.assertEqual(metadata["upper_bound"], [100.0] * 3)
        self.assertEqual(metadata["optimum"], 0.0)

        records = result.training_records
        self.assertEqual(records["total_steps"], 6)
        self.assertEqual(records["total_updates"], 2)
        self.assertEqual(len(records["episodes"]), 2)
        self.assertEqual(len(records["updates"]), 2)
        self.assertEqual(len(result.replay_buffer), 6)
        self.assertEqual(result.policy.total_it, 2)
        for episode in records["episodes"]:
            self.assertEqual(episode["steps"], 3)
            self.assertEqual(episode["final_fe"], 16)
            self.assertIs(episode["terminated"], True)
            self.assertIs(episode["truncated"], False)

        terminal_masks = result.replay_buffer.bootstrap_mask[
            :len(result.replay_buffer),
            0,
        ]
        self.assertEqual(
            int(np.count_nonzero(terminal_masks == 0.0)),
            2,
        )

        expected_device = torch.device("cpu")
        self.assertEqual(result.policy.device, expected_device)
        self.assertEqual(result.replay_buffer.device, expected_device)
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
                self.assertEqual(parameter.device, expected_device)
                self.assertTrue(torch.isfinite(parameter).all().item())

        serialized = json.dumps(
            {
                "config": result.config,
                "problem": result.problem_metadata,
                "training": result.training_records,
            }
        )
        self.assertIsInstance(serialized, str)

    def test_complete_training_and_cpu_determinism(self):
        config = self.make_config()

        first_result = run_classic_td3(config)
        first_actor_parameters = [
            parameter.detach().clone()
            for parameter in first_result.policy.actor.parameters()
        ]
        second_result = run_classic_td3(config)

        self.assert_complete_result(first_result)
        self.assert_complete_result(second_result)
        self.assertEqual(
            first_result.training_records,
            second_result.training_records,
        )
        for expected, actual in zip(
            first_actor_parameters,
            second_result.policy.actor.parameters(),
        ):
            self.assertTrue(torch.equal(expected, actual.detach()))


if __name__ == "__main__":
    unittest.main()
