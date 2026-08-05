from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TD3OnlineConfig:
    episodes: int
    learning_starts: int
    batch_size: int
    exploration_noise: float
    updates_per_step: int
    seed: int

    def __post_init__(self):
        integer_rules = {
            "episodes": (1, None),
            "learning_starts": (0, None),
            "batch_size": (1, None),
            "updates_per_step": (1, None),
            "seed": (0, None),
        }
        for name, (minimum, _) in integer_rules.items():
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(
                    f"{name} must be an integer >= {minimum}, got {value!r}"
                )
            object.__setattr__(self, name, int(value))

        noise = self.exploration_noise
        if (
            isinstance(noise, (bool, np.bool_))
            or not isinstance(noise, (int, float, np.integer, np.floating))
            or not np.isfinite(noise)
            or noise < 0.0
        ):
            raise ValueError(
                "exploration_noise must be a finite number >= 0.0, "
                f"got {noise!r}"
            )
        object.__setattr__(self, "exploration_noise", float(noise))


def train_online(env, policy, replay_buffer, config):
    if not isinstance(config, TD3OnlineConfig):
        raise TypeError(
            "config must be an instance of TD3OnlineConfig"
        )

    action_rng = np.random.default_rng(config.seed)
    episode_results = []
    update_results = []
    total_steps = 0
    warmup_steps = 0
    actor_steps = 0

    for episode_index in range(config.episodes):
        episode_seed = config.seed + episode_index
        state, _ = env.reset(seed=episode_seed)
        episode_return = 0.0
        episode_steps = 0
        c_values = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if total_steps < config.learning_starts:
                action = action_rng.uniform(
                    low=-1.0,
                    high=1.0,
                    size=env.action_space.shape,
                ).astype(np.float32)
                warmup_steps += 1
            else:
                actor_action = policy.select_action(
                    np.asarray(state, dtype=np.float32)
                )
                exploration = action_rng.normal(
                    loc=0.0,
                    scale=config.exploration_noise,
                    size=actor_action.shape,
                )
                action = np.clip(
                    actor_action + exploration,
                    -1.0,
                    1.0,
                ).astype(np.float32)
                actor_steps += 1

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            replay_buffer.add(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                terminated=terminated,
            )

            state = next_state
            episode_return += float(reward)
            episode_steps += 1
            total_steps += 1
            c_values.append(float(info["conv"]))

            if (
                total_steps >= config.learning_starts
                and len(replay_buffer) >= config.batch_size
            ):
                for _ in range(config.updates_per_step):
                    metrics = policy.train(
                        replay_buffer,
                        batch_size=config.batch_size,
                    )
                    update_results.append(
                        {
                            "episode_index": int(episode_index),
                            "episode_step": int(episode_steps),
                            "total_step": int(total_steps),
                            **metrics,
                        }
                    )

        c_array = np.asarray(c_values, dtype=np.float64)
        episode_results.append(
            {
                "episode_index": int(episode_index),
                "seed": int(episode_seed),
                "steps": int(episode_steps),
                "return": float(episode_return),
                "final_fe": int(env.swarm.fe_count),
                "final_best": float(env.swarm.gbest_fitness),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "c_mean": float(np.mean(c_array)),
                "c_min": float(np.min(c_array)),
                "c_max": float(np.max(c_array)),
            }
        )

    return {
        "episodes": episode_results,
        "updates": update_results,
        "total_steps": int(total_steps),
        "total_updates": int(len(update_results)),
        "warmup_steps": int(warmup_steps),
        "actor_steps": int(actor_steps),
    }
