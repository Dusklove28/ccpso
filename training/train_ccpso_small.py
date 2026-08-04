import random
from pathlib import Path

import numpy as np
import torch

from learning_ddpg.agents.agent import DDPGAgent
from learning_ddpg.buffers.replay_buffer import ReplayBuffer
from learning_ddpg.configs.ccpso_config import (
    ACTOR_LR,
    BATCH_SIZE,
    CHECKPOINT_EXPERIMENT_NAME,
    CHECKPOINT_INTERVAL,
    CONTROL_LEVEL,
    CRITIC_LR,
    DEVICE,
    FUNCTION_IDS,
    GAMMA,
    LEARNING_STARTS,
    MAX_FE,
    NOISE_STD,
    PARTICLES,
    REPLAY_CAPACITY,
    TAU,
    TRAIN_EPISODES,
    TRAIN_SEEDS,
)
from learning_ddpg.environments.factory import make_cec2013_env


def set_training_seed(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env.action_space.seed(seed)


def save_actor_checkpoint(agent, checkpoint_dir, episode, env, training_seed):
    checkpoint_path = checkpoint_dir / f"actor_episode_{episode:03d}.pt"
    checkpoint = {
        "episode": int(episode),
        "training_seed": int(training_seed),
        "objective_name": "CEC2013",
        "function_id": int(env.function_id),
        "function_optimum": float(env.optimum),
        "dimensions": int(env.swarm.dimensions),
        "actor_state_dict": agent.actor.state_dict(),
        "state_dim": int(env.observation_space.shape[0]),
        "action_dim": int(env.action_space.shape[0]),
        "action_low": env.action_space.low.tolist(),
        "action_high": env.action_space.high.tolist(),
        "control_level": CONTROL_LEVEL,
        "stage_fe": int(env.stage_fe),
        "stage_action_mode": env.stage_action_mode,
        "stage_smoothing_alpha": float(env.stage_smoothing_alpha),
        "stage_max_delta_c": float(env.stage_max_delta_c),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def train_one_seed(function_id, training_seed, checkpoint_root, device):
    env = make_cec2013_env(function_id)
    set_training_seed(training_seed, env)

    checkpoint_dir = (
        checkpoint_root
        / f"F{int(function_id):02d}"
        / f"train_seed_{training_seed}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    agent = DDPGAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        device=device,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        gamma=GAMMA,
        tau=TAU,
        noise_std=NOISE_STD,
    )
    buffer = ReplayBuffer(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        capacity=REPLAY_CAPACITY,
        device=device,
    )

    last_result = None
    saved_paths = []
    try:
        for episode in range(TRAIN_EPISODES):
            state, info = env.reset(seed=training_seed + episode)
            expected_steps = env.remaining_stage_steps
            episode_reward = 0.0
            episode_steps = 0

            while True:
                if len(buffer) < LEARNING_STARTS:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state, add_noise=True)

                next_state, reward, terminated, truncated, info = env.step(action)
                episode_done = terminated or truncated
                buffer.add(state, action, reward, next_state, episode_done)

                state = next_state
                episode_reward += reward
                episode_steps += 1

                if len(buffer) >= LEARNING_STARTS:
                    last_result = agent.update(buffer, batch_size=BATCH_SIZE)

                if episode_done:
                    break

            assert episode_steps == expected_steps
            assert env.swarm.fe_count == env.swarm.max_fe
            assert np.isfinite(episode_reward)

            episode_number = episode + 1
            if episode_number % CHECKPOINT_INTERVAL == 0:
                checkpoint_path = save_actor_checkpoint(
                    agent=agent,
                    checkpoint_dir=checkpoint_dir,
                    episode=episode_number,
                    env=env,
                    training_seed=training_seed,
                )
                saved_paths.append(checkpoint_path)

                print(
                    f"train_seed={training_seed} "
                    f"function=F{int(function_id):02d} "
                    f"episode={episode_number:03d} "
                    f"reward={episode_reward:.6f} "
                    f"best={env.swarm.gbest_fitness:.6e} "
                    f"buffer={len(buffer)} "
                    f"actor_loss={last_result['actor_loss']:.6f} "
                    f"critic_loss={last_result['critic_loss']:.6f}"
                )
                print(f"saved candidate Actor: {checkpoint_path}")
    finally:
        env.close()

    expected_candidates = TRAIN_EPISODES // CHECKPOINT_INTERVAL
    assert len(saved_paths) == expected_candidates
    assert last_result is not None
    assert np.isfinite(last_result["actor_loss"])
    assert np.isfinite(last_result["critic_loss"])

    print(
        f"training seed {training_seed} complete | "
        f"function=F{int(function_id):02d} | "
        f"saved_candidates={len(saved_paths)}"
    )
    return saved_paths


def main():
    device = torch.device(DEVICE)
    checkpoint_root = (
        Path(__file__).resolve().parents[1]
        / "checkpoints"
        / CHECKPOINT_EXPERIMENT_NAME
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    all_saved_paths = []
    for function_id in FUNCTION_IDS:
        for training_seed in TRAIN_SEEDS:
            all_saved_paths.extend(
                train_one_seed(
                    function_id,
                    training_seed,
                    checkpoint_root,
                    device,
                )
            )

    expected_total = (
        len(FUNCTION_IDS)
        * len(TRAIN_SEEDS)
        * TRAIN_EPISODES
        // CHECKPOINT_INTERVAL
    )
    assert len(all_saved_paths) == expected_total
    print(
        f"multi-seed training complete | "
        f"functions=F1-F28 | "
        f"training_seeds={list(TRAIN_SEEDS)} | "
        f"total_candidates={len(all_saved_paths)} | "
        f"checkpoint_root={checkpoint_root}"
    )


if __name__ == "__main__":
    main()
