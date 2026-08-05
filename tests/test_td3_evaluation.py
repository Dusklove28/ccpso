import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.td3.replay_buffer import ReplayBuffer
from agents.td3.td3 import TD3
from evaluation.evaluate_td3 import STATE_FIELDS, evaluate_td3_policy
from problems import ProblemSpec, make_classic_problem


class TestTD3Evaluation(unittest.TestCase):
    @staticmethod
    def make_trained_policy():
        torch.manual_seed(123)
        policy = TD3(
            state_dim=6,
            action_dim=1,
            max_action=1.0,
            policy_freq=2,
            device="cpu",
        )
        replay_buffer = ReplayBuffer(
            state_dim=6,
            action_dim=1,
            max_size=16,
            seed=456,
            device="cpu",
        )
        for index in range(8):
            value = float(index + 1) / 10.0
            replay_buffer.add(
                state=np.full(6, value, dtype=np.float32),
                action=np.array([value - 0.5], dtype=np.float32),
                next_state=np.full(6, value + 0.01, dtype=np.float32),
                reward=value,
                terminated=index in (3, 7),
            )
        for _ in range(4):
            policy.train(replay_buffer, batch_size=4)
        return policy

    def load_frozen_policy(self, directory):
        source = self.make_trained_policy()
        checkpoint_path = Path(directory) / "policy.pt"
        source.save_checkpoint(
            checkpoint_path,
            metadata={"purpose": "frozen evaluation"},
        )
        policy = TD3(
            state_dim=6,
            action_dim=1,
            max_action=1.0,
            device="cpu",
        )
        metadata = policy.load_checkpoint(
            checkpoint_path,
            load_optimizers=False,
        )
        self.assertEqual(metadata, {"purpose": "frozen evaluation"})
        return policy

    @staticmethod
    def snapshot_networks(policy):
        return {
            network_name: {
                name: value.detach().cpu().clone()
                for name, value in getattr(
                    policy,
                    network_name,
                ).state_dict().items()
            }
            for network_name in (
                "actor",
                "actor_target",
                "critic",
                "critic_target",
            )
        }

    def assert_networks_unchanged(self, expected, policy):
        actual = self.snapshot_networks(policy)
        self.assertEqual(set(expected), set(actual))
        for network_name in expected:
            self.assertEqual(
                set(expected[network_name]),
                set(actual[network_name]),
            )
            for name in expected[network_name]:
                self.assertTrue(
                    torch.equal(
                        expected[network_name][name],
                        actual[network_name][name],
                    ),
                    msg=f"{network_name}.{name} changed",
                )

    def test_select_action_disables_gradient_tracking(self):
        policy = TD3(6, 1, 1.0, device="cpu")
        gradient_modes = []
        hook = policy.actor.register_forward_pre_hook(
            lambda _module, _inputs: gradient_modes.append(
                torch.is_grad_enabled()
            )
        )
        try:
            action = policy.select_action(
                np.zeros(6, dtype=np.float32)
            )
        finally:
            hook.remove()

        self.assertEqual(gradient_modes, [False])
        self.assertEqual(action.shape, (1,))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_frozen_multi_seed_evaluation_is_complete_and_repeatable(self):
        problem = make_classic_problem("sphere", dimensions=3)
        seeds = [11, 12, 13]

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            policy = self.load_frozen_policy(temp_dir)
            networks_before = self.snapshot_networks(policy)
            actor_optimizer_before = copy.deepcopy(
                policy.actor_optimizer.state_dict()
            )
            critic_optimizer_before = copy.deepcopy(
                policy.critic_optimizer.state_dict()
            )
            total_it_before = policy.total_it
            original_select_action = policy.select_action
            policy.select_action = Mock(wraps=original_select_action)
            policy.train = Mock(
                side_effect=AssertionError(
                    "evaluation must not call policy.train()"
                )
            )

            first = evaluate_td3_policy(
                policy,
                problem,
                particles=4,
                max_fe=16,
                seeds=seeds,
            )
            first_call_count = policy.select_action.call_count
            second = evaluate_td3_policy(
                policy,
                problem,
                particles=4,
                max_fe=16,
                seeds=seeds,
            )

        self.assertEqual(first, second)
        self.assertEqual(len(first["episodes"]), 3)
        self.assertEqual(len(first["steps"]), 9)
        self.assertEqual(first_call_count, 9)
        self.assertEqual(policy.select_action.call_count, 18)
        policy.train.assert_not_called()

        for episode in first["episodes"]:
            self.assertIn(episode["seed"], seeds)
            self.assertEqual(episode["steps"], 3)
            self.assertEqual(episode["final_fe"], 16)
            self.assertTrue(np.isfinite(episode["final_best"]))
            self.assertGreaterEqual(episode["gap"], 0.0)
            self.assertTrue(np.isfinite(episode["return"]))
            self.assertLessEqual(episode["c_min"], episode["c_mean"])
            self.assertLessEqual(episode["c_mean"], episode["c_max"])

        for seed in seeds:
            seed_steps = [
                step for step in first["steps"] if step["seed"] == seed
            ]
            self.assertEqual(
                [step["episode_step"] for step in seed_steps],
                [1, 2, 3],
            )
            self.assertEqual(
                [step["fe_count"] for step in seed_steps],
                [8, 12, 16],
            )

        for step in first["steps"]:
            action_state = np.array(
                [step[field] for field in STATE_FIELDS],
                dtype=np.float32,
            )
            expected_action = original_select_action(action_state)
            self.assertAlmostEqual(
                step["raw_action"],
                float(expected_action[0]),
                places=7,
            )
            self.assertAlmostEqual(
                step["c_value"],
                0.75 * (float(expected_action[0]) + 1.0),
                places=7,
            )
            self.assertTrue(np.isfinite(step["reward"]))
            self.assertTrue(np.isfinite(step["gbest_fitness"]))
            self.assertGreaterEqual(step["gap"], 0.0)

        final_gaps = np.array(
            [episode["gap"] for episode in first["episodes"]]
        )
        statistics = first["final_gap_statistics"]
        self.assertEqual(statistics["mean"], float(np.mean(final_gaps)))
        self.assertEqual(statistics["median"], float(np.median(final_gaps)))
        self.assertEqual(statistics["std"], float(np.std(final_gaps)))
        self.assertEqual(statistics["min"], float(np.min(final_gaps)))
        self.assertEqual(statistics["max"], float(np.max(final_gaps)))
        self.assertIsInstance(json.dumps(first, allow_nan=False), str)

        self.assert_networks_unchanged(networks_before, policy)
        self.assertEqual(
            policy.actor_optimizer.state_dict(),
            actor_optimizer_before,
        )
        self.assertEqual(
            policy.critic_optimizer.state_dict(),
            critic_optimizer_before,
        )
        self.assertEqual(policy.total_it, total_it_before)

    def test_optimum_changes_only_gap_metadata_and_logs(self):
        def sphere(positions):
            return np.sum(positions**2, axis=1)

        common = {
            "suite": "test",
            "dimensions": 3,
            "lower_bound": -100.0,
            "upper_bound": 100.0,
            "objective": sphere,
        }
        zero_optimum = ProblemSpec(
            problem_id="sphere-zero",
            name="Sphere zero optimum",
            optimum=0.0,
            **common,
        )
        shifted_optimum = ProblemSpec(
            problem_id="sphere-shifted-log",
            name="Sphere shifted log optimum",
            optimum=-100.0,
            **common,
        )

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            policy = self.load_frozen_policy(temp_dir)
            zero_result = evaluate_td3_policy(
                policy,
                zero_optimum,
                particles=4,
                max_fe=16,
                seeds=[21],
            )
            shifted_result = evaluate_td3_policy(
                policy,
                shifted_optimum,
                particles=4,
                max_fe=16,
                seeds=[21],
            )

        for zero_step, shifted_step in zip(
            zero_result["steps"],
            shifted_result["steps"],
        ):
            for field in (
                "seed",
                "episode_step",
                "fe_count",
                "gbest_fitness",
                "raw_action",
                "c_value",
                "reward",
                *STATE_FIELDS,
            ):
                self.assertEqual(zero_step[field], shifted_step[field])
            self.assertEqual(
                shifted_step["gap"],
                zero_step["gap"] + 100.0,
            )

        zero_episode = zero_result["episodes"][0]
        shifted_episode = shifted_result["episodes"][0]
        for field in (
            "seed",
            "final_best",
            "steps",
            "final_fe",
            "return",
            "c_mean",
            "c_min",
            "c_max",
        ):
            self.assertEqual(zero_episode[field], shifted_episode[field])
        self.assertEqual(
            shifted_episode["gap"],
            zero_episode["gap"] + 100.0,
        )

    def test_seeds_must_be_a_non_empty_integer_sequence(self):
        problem = make_classic_problem("sphere", dimensions=3)
        policy = TD3(6, 1, 1.0, device="cpu")
        invalid_values = (
            [],
            (),
            "1,2",
            1,
            [True],
            [1.0],
            [1, np.nan],
            np.array([[1, 2]]),
        )
        for seeds in invalid_values:
            with self.subTest(seeds=seeds):
                with self.assertRaises(ValueError):
                    evaluate_td3_policy(
                        policy,
                        problem,
                        particles=4,
                        max_fe=16,
                        seeds=seeds,
                    )


if __name__ == "__main__":
    unittest.main()
