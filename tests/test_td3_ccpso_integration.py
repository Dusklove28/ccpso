from pathlib import Path
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


def sphere(positions):
    return np.sum(positions**2, axis=1)


class TestTD3CCPSOIntegration(unittest.TestCase):

    def test_three_episode_single_step_td3_loop_on_cpu(self):
        python_seed = 101
        numpy_seed = 202
        torch_seed = 303
        environment_seed = 404
        buffer_seed = 505

        random.seed(python_seed)
        np.random.seed(numpy_seed)
        torch.manual_seed(torch_seed)

        cpu_device = torch.device("cpu")
        # 初始化群体智能算法
        swarm = CCPSOSwarm(
            particles=4,
            dimensions=3,
            fun=sphere,
            lower_bound=-100.0,
            upper_bound=100.0,
            max_fe=16,
            seed=environment_seed,
        )
        # 初始化Env
        env = CCPSOEnv(
            swarm=swarm,
            c_min=0.0,
            c_max=1.5,
            optimum=0.0,
        )
        # 初始化TD3算法
        policy = TD3(
            state_dim=6,
            action_dim=1,
            max_action=1.0,
            device=cpu_device,
        )
        # 初始化经验回池子
        replay_buffer = ReplayBuffer(
            state_dim=6,
            action_dim=1,
            max_size=32,
            seed=buffer_seed,
            device=cpu_device,
        )

        self.assertEqual(policy.device, cpu_device)
        self.assertEqual(replay_buffer.device, cpu_device)
        update_count = 0

        for episode_index in range(3):
            state, _ = env.reset(
                seed=environment_seed + episode_index
            )
            self.assertIsInstance(state, np.ndarray)
            episode_start = len(replay_buffer)
            episode_steps = 0
            terminated = False

            while not terminated:
                actor_action = policy.select_action(state)
                self.assertEqual(actor_action.shape, (1,))
                self.assertTrue(np.all(np.isfinite(actor_action)))

                exploration_noise = np.random.normal(
                    loc=0.0,
                    scale=0.05,
                    size=actor_action.shape,
                )
                action = np.clip(
                    actor_action + exploration_noise,
                    -1.0,
                    1.0,
                ).astype(np.float32)

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)
                self.assertIs(truncated, False)

                replay_buffer.add(
                    state=state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                    terminated=terminated,
                )
                state = next_state
                episode_steps += 1

                if len(replay_buffer) >= 2 and update_count < 2:
                    policy.train(replay_buffer, batch_size=2)
                    update_count += 1

            self.assertIs(terminated, True)
            self.assertEqual(episode_steps, 3)
            self.assertEqual(swarm.fe_count, swarm.max_fe)

            episode_end = len(replay_buffer)
            episode_masks = replay_buffer.bootstrap_mask[
                episode_start:episode_end,
                0,
            ]
            self.assertEqual(
                int(np.count_nonzero(episode_masks == 0.0)),
                1,
            )

        self.assertGreaterEqual(update_count, 1)
        self.assertEqual(policy.total_it, update_count)
        self.assertEqual(len(replay_buffer), 9)

        for network_name in (
            "actor",
            "actor_target",
            "critic",
            "critic_target",
        ):
            network = getattr(policy, network_name)
            for parameter in network.parameters():
                self.assertEqual(parameter.device, cpu_device)
                self.assertTrue(
                    torch.isfinite(parameter).all().item(),
                    msg=f"{network_name} contains non-finite parameters",
                )


if __name__ == "__main__":
    unittest.main()
