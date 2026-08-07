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
    def test_relative_v2_episode_records_reset_scales(self):
        seed = 77
        env = CCPSOEnv(
            swarm=CCPSOSwarm(
                particles=4,
                dimensions=3,
                fun=sphere,
                lower_bound=-100.0,
                upper_bound=100.0,
                max_fe=16,
                seed=seed,
            ),
            c_min=0.0,
            c_max=1.5,
            optimum=0.0,
            state_mode="relative_log_v2",
        )
        device = torch.device("cpu")
        policy = TD3(6, 1, 1.0, device=device)
        replay_buffer = ReplayBuffer(
            6,
            1,
            max_size=8,
            seed=seed,
            device=device,
        )
        result = train_online(
            env,
            policy,
            replay_buffer,
            TD3OnlineConfig(
                episodes=1,
                learning_starts=100,
                batch_size=2,
                exploration_noise=0.0,
                updates_per_step=1,
                seed=seed,
            ),
        )

        episode = result["episodes"][0]
        self.assertEqual(episode["state_mode"], "relative_log_v2")
        self.assertGreater(episode["initial_position_scale"], 0.0)
        self.assertGreater(episode["initial_q_scale"], 0.0)
        self.assertEqual(episode["max_episode_updates"], 3)
        self.assertTrue(all(
            "state_mode" not in step for step in result["steps"]
        ))

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
            self.assertEqual(
                episode["reward_mode"],
                "step_log_improvement",
            )
            self.assertGreater(episode["initial_improvement_scale"], 0.0)
            self.assertGreater(episode["initial_gap_scale"], 0.0)
            self.assertEqual(episode["state_mode"], "legacy_v1")
            self.assertIsNone(episode["initial_position_scale"])
            self.assertIsNone(episode["initial_q_scale"])
            self.assertIsNone(episode["max_episode_updates"])
            self.assertLessEqual(episode["c_min"], episode["c_mean"])
            self.assertLessEqual(episode["c_mean"], episode["c_max"])
            self.assertGreaterEqual(episode["c_min"], 0.0)
            self.assertLessEqual(episode["c_max"], 1.5)

        self.assertEqual(result["total_steps"], 9)
        self.assertEqual(len(result["steps"]), result["total_steps"])
        self.assertEqual(len(replay_buffer), result["total_steps"])
        self.assertEqual(result["warmup_steps"], 7)
        self.assertEqual(result["actor_steps"], 2)
        self.assertGreater(result["warmup_steps"], 0)
        self.assertGreater(result["actor_steps"], 0)

        expected_updates = 3
        self.assertEqual(result["total_updates"], expected_updates)
        self.assertEqual(len(result["updates"]), expected_updates)
        self.assertEqual(policy.total_it, expected_updates)

        steps = result["steps"]
        self.assertEqual(
            [step["global_step"] for step in steps],
            list(range(1, 10)),
        )
        self.assertEqual(
            [step["episode_index"] for step in steps],
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
        )
        self.assertEqual(
            [step["episode_seed"] for step in steps],
            [123, 123, 123, 124, 124, 124, 125, 125, 125],
        )
        self.assertEqual(
            [step["episode_step"] for step in steps],
            [1, 2, 3] * 3,
        )
        self.assertEqual(
            sum(step["action_source"] == "warmup" for step in steps),
            result["warmup_steps"],
        )
        self.assertEqual(
            sum(step["action_source"] == "actor" for step in steps),
            result["actor_steps"],
        )
        self.assertEqual(
            [step["action_source"] for step in steps],
            ["warmup"] * 7 + ["actor"] * 2,
        )
        self.assertEqual(steps[6]["global_step"], 7)
        self.assertEqual(steps[6]["action_source"], "warmup")
        self.assertEqual(steps[7]["global_step"], 8)
        self.assertEqual(steps[7]["action_source"], "actor")

        state_fields = (
            "state_fe_progress",
            "state_recent_progress",
            "state_position_diversity",
            "state_q_diversity",
            "state_movement",
            "state_stagnation",
        )
        for episode_index in range(3):
            episode_steps = [
                step
                for step in steps
                if step["episode_index"] == episode_index
            ]
            self.assertEqual(
                [step["terminated"] for step in episode_steps],
                [False, False, True],
            )
            self.assertTrue(
                all(not step["truncated"] for step in episode_steps)
            )
            np.testing.assert_allclose(
                [step["state_fe_progress"] for step in episode_steps],
                [0.25, 0.5, 0.75],
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(
                [step["fe_count"] for step in episode_steps],
                [8, 12, 16],
            )
            self.assertEqual(
                [step["decision_fe"] for step in episode_steps],
                [4, 8, 12],
            )
            cumulative_return = 0.0
            for step in episode_steps:
                cumulative_return += step["reward"]
                self.assertAlmostEqual(
                    step["episode_return"],
                    cumulative_return,
                )

        for step in steps:
            self.assertNotIn("state_mode", step)
            self.assertIn(step["action_source"], ("warmup", "actor"))
            self.assertTrue(np.isfinite(step["reward"]))
            self.assertTrue(np.isfinite(step["reward_progress"]))
            self.assertTrue(np.isfinite(step["episode_return"]))
            self.assertTrue(np.isfinite(step["gbest_fitness"]))
            self.assertGreaterEqual(step["gap"], 0.0)
            self.assertGreaterEqual(step["raw_action"], -1.0)
            self.assertLessEqual(step["raw_action"], 1.0)
            self.assertGreaterEqual(step["c_value"], 0.0)
            self.assertLessEqual(step["c_value"], 1.5)
            self.assertTrue(np.isfinite(step["mean_movement"]))
            self.assertGreaterEqual(step["recent_progress"], 0.0)
            self.assertLessEqual(step["recent_progress"], 1.0)
            self.assertGreaterEqual(step["stagnation_steps"], 0)
            self.assertGreaterEqual(step["boundary_clip_ratio"], 0.0)
            self.assertLessEqual(step["boundary_clip_ratio"], 1.0)
            for field in state_fields:
                self.assertTrue(np.isfinite(step[field]), msg=field)
                self.assertGreaterEqual(step[field], 0.0, msg=field)
                self.assertLessEqual(step[field], 1.0, msg=field)

        self.assertEqual(
            [step["cumulative_training_fe"] for step in steps],
            [8, 12, 16, 24, 28, 32, 40, 44, 48],
        )
        self.assertEqual(
            [
                episode["cumulative_training_fe"]
                for episode in result["episodes"]
            ],
            [16, 32, 48],
        )
        self.assertEqual(
            [update["cumulative_training_fe"] for update in result["updates"]],
            [40, 44, 48],
        )

        update_contexts = [
            (
                update["global_step"],
                update["episode_index"],
                update["episode_step"],
            )
            for update in result["updates"]
        ]
        self.assertEqual(
            update_contexts,
            [(7, 2, 1), (8, 2, 2), (9, 2, 3)],
        )
        self.assertEqual(
            [update["total_it"] for update in result["updates"]],
            [1, 2, 3],
        )
        for update in result["updates"]:
            self.assertNotIn("total_step", update)
            self.assertIn("critic_loss", update)
            self.assertIn("actor_updated", update)
            self.assertIn("target_q_mean", update)

        terminal_masks = replay_buffer.bootstrap_mask[
            :len(replay_buffer),
            0,
        ]
        self.assertEqual(
            int(np.count_nonzero(terminal_masks == 0.0)),
            config.episodes,
        )

        serialized = json.dumps(result, allow_nan=False)
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
