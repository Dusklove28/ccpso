from pathlib import Path
import json
import random
import sys
import unittest

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.td3.replay_buffer import ReplayBuffer
from agents.td3.td3 import TD3
from environments.ccpso_env import CCPSOEnv
from swarm.ccpso import CCPSOSwarm
from training.td3_online import TD3OnlineConfig, train_online


def sphere(positions):
    return np.sum(positions**2, axis=1)


class TestTD3OnlineConfig(unittest.TestCase):
    def test_rejects_invalid_values(self):
        valid = {
            "episodes": 3,
            "learning_starts": 7,
            "batch_size": 2,
            "exploration_noise": 0.05,
            "updates_per_step": 1,
            "seed": 123,
        }
        cases = (
            ("episodes", 0),
            ("episodes", 3.0),
            ("learning_starts", -1),
            ("learning_starts", True),
            ("batch_size", 0),
            ("batch_size", 2.0),
            ("updates_per_step", 0),
            ("updates_per_step", False),
            ("seed", -1),
            ("seed", 1.5),
            ("exploration_noise", -0.1),
            ("exploration_noise", np.inf),
            ("exploration_noise", True),
        )

        for name, value in cases:
            with self.subTest(name=name, value=value):
                parameters = valid.copy()
                parameters[name] = value
                with self.assertRaises(ValueError):
                    TD3OnlineConfig(**parameters)


class TestTD3OnlineTraining(unittest.TestCase):
    def test_real_td3_ccpso_online_training(self):
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device("cpu")

        swarm = CCPSOSwarm(
            particles=4,
            dimensions=3,
            fun=sphere,
            lower_bound=-100.0,
            upper_bound=100.0,
            max_fe=16,
            seed=seed,
        )
        env = CCPSOEnv(
            swarm=swarm,
            c_min=0.0,
            c_max=1.5,
            optimum=0.0,
        )
        policy = TD3(
            state_dim=6,
            action_dim=1,
            max_action=1.0,
            policy_freq=2,
            device=device,
        )
        replay_buffer = ReplayBuffer(
            state_dim=6,
            action_dim=1,
            max_size=32,
            seed=seed,
            device=device,
        )
        config = TD3OnlineConfig(
            episodes=3,
            learning_starts=7,
            batch_size=2,
            exploration_noise=0.05,
            updates_per_step=1,
            seed=seed,
        )

        result = train_online(
            env=env,
            policy=policy,
            replay_buffer=replay_buffer,
            config=config,
        )

        self.assertEqual(len(result["episodes"]), 3)
        self.assertEqual(
            [episode["seed"] for episode in result["episodes"]],
            [123, 124, 125],
        )
        for episode in result["episodes"]:
            self.assertEqual(episode["steps"], 3)
            self.assertEqual(episode["final_fe"], 16)
            self.assertIs(episode["terminated"], True)
            self.assertIs(episode["truncated"], False)
            self.assertTrue(np.isfinite(episode["return"]))
            self.assertTrue(np.isfinite(episode["final_best"]))
            self.assertLessEqual(episode["c_min"], episode["c_mean"])
            self.assertLessEqual(episode["c_mean"], episode["c_max"])
            self.assertGreaterEqual(episode["c_min"], 0.0)
            self.assertLessEqual(episode["c_max"], 1.5)

        self.assertEqual(result["total_steps"], 9)
        self.assertEqual(len(replay_buffer), result["total_steps"])
        self.assertEqual(result["warmup_steps"], 7)
        self.assertEqual(result["actor_steps"], 2)
        self.assertGreater(result["warmup_steps"], 0)
        self.assertGreater(result["actor_steps"], 0)

        expected_updates = 3
        self.assertEqual(result["total_updates"], expected_updates)
        self.assertEqual(len(result["updates"]), expected_updates)
        self.assertEqual(policy.total_it, expected_updates)

        terminal_masks = replay_buffer.bootstrap_mask[
            :len(replay_buffer),
            0,
        ]
        self.assertEqual(
            int(np.count_nonzero(terminal_masks == 0.0)),
            config.episodes,
        )

        serialized = json.dumps(result)
        self.assertIsInstance(serialized, str)

        for network_name in (
            "actor",
            "actor_target",
            "critic",
            "critic_target",
        ):
            for parameter in getattr(policy, network_name).parameters():
                self.assertEqual(parameter.device, device)
                self.assertTrue(
                    torch.isfinite(parameter).all().item(),
                    msg=f"{network_name} contains non-finite parameters",
                )


if __name__ == "__main__":
    unittest.main()
