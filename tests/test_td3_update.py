from pathlib import Path
import sys
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.td3.replay_buffer import ReplayBuffer
from agents.td3.td3 import TD3


class TestTD3UpdateMetrics(unittest.TestCase):
    @staticmethod
    def snapshot_parameters(network):
        return [
            parameter.detach().clone()
            for parameter in network.parameters()
        ]

    def assert_parameters_equal(self, expected, network):
        actual = list(network.parameters())
        self.assertEqual(len(expected), len(actual))
        for expected_parameter, actual_parameter in zip(expected, actual):
            self.assertTrue(
                torch.equal(expected_parameter, actual_parameter.detach())
            )

    def assert_parameters_changed(self, expected, network):
        actual = list(network.parameters())
        self.assertEqual(len(expected), len(actual))
        self.assertTrue(
            any(
                not torch.equal(
                    expected_parameter,
                    actual_parameter.detach(),
                )
                for expected_parameter, actual_parameter in zip(
                    expected,
                    actual,
                )
            )
        )

    def assert_valid_metrics(self, metrics, actor_updated):
        self.assertEqual(
            set(metrics),
            {
                "total_it",
                "critic_loss",
                "actor_updated",
                "actor_loss",
                "target_q_mean",
                "q1_mean",
                "q2_mean",
            },
        )
        self.assertIs(type(metrics["total_it"]), int)
        self.assertIs(type(metrics["actor_updated"]), bool)
        self.assertIs(metrics["actor_updated"], actor_updated)

        for name in (
            "critic_loss",
            "target_q_mean",
            "q1_mean",
            "q2_mean",
        ):
            self.assertIs(type(metrics[name]), float)
            self.assertTrue(np.isfinite(metrics[name]), msg=name)

        if actor_updated:
            self.assertIs(type(metrics["actor_loss"]), float)
            self.assertTrue(np.isfinite(metrics["actor_loss"]))
        else:
            self.assertIsNone(metrics["actor_loss"])

    def test_policy_delay_and_structured_metrics(self):
        np.random.seed(123)
        torch.manual_seed(456)
        device = torch.device("cpu")
        replay_buffer = ReplayBuffer(
            state_dim=3,
            action_dim=1,
            max_size=16,
            seed=789,
            device=device,
        )
        for index in range(8):
            value = float(index + 1)
            replay_buffer.add(
                state=np.array(
                    [value, value * 0.5, -value],
                    dtype=np.float32,
                ),
                action=np.array(
                    [(-1.0) ** index * 0.25],
                    dtype=np.float32,
                ),
                next_state=np.array(
                    [value + 0.1, value * 0.5 - 0.2, -value + 0.3],
                    dtype=np.float32,
                ),
                reward=value / 10.0,
                terminated=index in (3, 7),
            )

        policy = TD3(
            state_dim=3,
            action_dim=1,
            max_action=1.0,
            policy_freq=2,
            device=device,
        )

        actor_before_first = self.snapshot_parameters(policy.actor)
        actor_target_before_first = self.snapshot_parameters(
            policy.actor_target
        )
        critic_before_first = self.snapshot_parameters(policy.critic)
        critic_target_before_first = self.snapshot_parameters(
            policy.critic_target
        )

        first_metrics = policy.train(replay_buffer, batch_size=4)

        self.assert_valid_metrics(first_metrics, actor_updated=False)
        self.assertEqual(first_metrics["total_it"], 1)
        self.assert_parameters_equal(actor_before_first, policy.actor)
        self.assert_parameters_equal(
            actor_target_before_first,
            policy.actor_target,
        )
        self.assert_parameters_changed(critic_before_first, policy.critic)
        self.assert_parameters_equal(
            critic_target_before_first,
            policy.critic_target,
        )

        actor_before_second = self.snapshot_parameters(policy.actor)
        actor_target_before_second = self.snapshot_parameters(
            policy.actor_target
        )
        critic_target_before_second = self.snapshot_parameters(
            policy.critic_target
        )

        second_metrics = policy.train(replay_buffer, batch_size=4)

        self.assert_valid_metrics(second_metrics, actor_updated=True)
        self.assertEqual(second_metrics["total_it"], 2)
        self.assertEqual(
            second_metrics["total_it"],
            first_metrics["total_it"] + 1,
        )
        self.assert_parameters_changed(actor_before_second, policy.actor)
        self.assert_parameters_changed(
            actor_target_before_second,
            policy.actor_target,
        )
        self.assert_parameters_changed(
            critic_target_before_second,
            policy.critic_target,
        )

        for network_name in (
            "actor",
            "actor_target",
            "critic",
            "critic_target",
        ):
            for parameter in getattr(policy, network_name).parameters():
                self.assertTrue(
                    torch.isfinite(parameter).all().item(),
                    msg=f"{network_name} contains non-finite parameters",
                )


if __name__ == "__main__":
    unittest.main()
