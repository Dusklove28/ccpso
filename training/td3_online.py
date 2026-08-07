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
    """Run online TD3 control over independent CCPSO optimizations.

    Counter semantics are strict:

    - An ``episode`` is one complete, independent CCPSO optimization. The
      swarm is reset at the start of each episode, while the TD3 policy and
      replay buffer persist across episodes.
    - ``episode_step`` is the number of CCPSO generations already executed
      in the current episode and restarts from one after each reset.
    - ``global_step`` is the cumulative number of ``env.step()`` calls made
      within this invocation of ``train_online()``.
    - ``decision_fe`` is the number of function evaluations already consumed
      in the current episode immediately before the recorded action.
    - ``fe_count`` is the current episode's function-evaluation count after
      that action has been executed.
    - ``cumulative_training_fe`` is the total training evaluation cost across
      episodes, including every reset's initial-population evaluation.
    - ``total_it`` is TD3's cumulative number of gradient updates.

    One ``global_step`` equals one ``env.step()``, one complete swarm update,
    and one RL transition. The initial population evaluation performed by
    ``reset()`` is not counted as a ``global_step``.

    The warmup boundary is inclusive. For ``learning_starts=7``, global steps
    1 through 7 use warmup actions; updates may begin after the transition at
    global step 7 is stored; Actor actions begin at global step 8.
    """
    if not isinstance(config, TD3OnlineConfig):
        raise TypeError(
            "config must be an instance of TD3OnlineConfig"
        )

    action_rng = np.random.default_rng(config.seed)
    episode_results = []
    step_results = []
    update_results = []
    total_steps = 0
    cumulative_training_fe = 0
    warmup_steps = 0
    actor_steps = 0

    for episode_index in range(config.episodes):
        episode_seed = config.seed + episode_index
        state, reset_info = env.reset(seed=episode_seed)
        state_mode = reset_info.get("state_mode", "legacy_v1")
        if state_mode == "relative_log_v2":
            initial_position_scale = float(
                reset_info["initial_position_scale"]
            )
            initial_q_scale = float(reset_info["initial_q_scale"])
            max_episode_updates = int(
                reset_info["max_episode_updates"]
            )
        else:
            initial_position_scale = None
            initial_q_scale = None
            max_episode_updates = None
        reset_fe = int(env.swarm.fe_count)
        cumulative_training_fe += reset_fe
        episode_return = 0.0
        episode_steps = 0
        c_values = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            decision_fe = int(env.swarm.fe_count)
            action_state = np.asarray(
                state,
                dtype=np.float32,
            ).copy()
            if action_state.shape != (6,):
                raise ValueError(
                    "train_online expected a six-dimensional state, "
                    f"got shape {action_state.shape}"
                )

            if total_steps < config.learning_starts:
                action_source = "warmup"
                action = action_rng.uniform(
                    low=-1.0,
                    high=1.0,
                    size=env.action_space.shape,
                ).astype(np.float32)
                warmup_steps += 1
            else:
                action_source = "actor"
                actor_action = policy.select_action(
                    action_state
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
            post_step_fe = int(info["fe_count"])
            cumulative_training_fe += post_step_fe - decision_fe

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
            step_results.append(
                {
                    "episode_index": int(episode_index),
                    "episode_seed": int(episode_seed),
                    "global_step": int(total_steps),
                    "episode_step": int(episode_steps),
                    "decision_fe": int(decision_fe),
                    "cumulative_training_fe": int(
                        cumulative_training_fe
                    ),
                    "action_source": action_source,
                    "reward": float(reward),
                    "reward_progress": float(info["reward_progress"]),
                    "episode_return": float(episode_return),
                    "fe_count": post_step_fe,
                    "gbest_fitness": float(info["gbest_fitness"]),
                    "gap": float(info["gap"]),
                    "raw_action": float(info["raw_action"]),
                    "c_value": float(info["conv"]),
                    "mean_movement": float(info["mean_movement"]),
                    "recent_progress": float(info["recent_progress"]),
                    "stagnation_steps": int(info["stagnation_steps"]),
                    "boundary_clip_ratio": float(
                        info["boundary_clip_ratio"]
                    ),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "state_fe_progress": float(action_state[0]),
                    "state_recent_progress": float(action_state[1]),
                    "state_position_diversity": float(action_state[2]),
                    "state_q_diversity": float(action_state[3]),
                    "state_movement": float(action_state[4]),
                    "state_stagnation": float(action_state[5]),
                }
            )

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
                            "global_step": int(total_steps),
                            "episode_index": int(episode_index),
                            "episode_step": int(episode_steps),
                            "cumulative_training_fe": int(
                                cumulative_training_fe
                            ),
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
                "reward_mode": reset_info["reward_mode"],
                "initial_improvement_scale": float(
                    reset_info["initial_improvement_scale"]
                ),
                "initial_gap_scale": float(
                    reset_info["initial_gap_scale"]
                ),
                "final_fe": int(env.swarm.fe_count),
                "cumulative_training_fe": int(
                    cumulative_training_fe
                ),
                "final_best": float(env.swarm.gbest_fitness),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "c_mean": float(np.mean(c_array)),
                "c_min": float(np.min(c_array)),
                "c_max": float(np.max(c_array)),
                "state_mode": state_mode,
                "initial_position_scale": initial_position_scale,
                "initial_q_scale": initial_q_scale,
                "max_episode_updates": max_episode_updates,
            }
        )

    return {
        "episodes": episode_results,
        "steps": step_results,
        "updates": update_results,
        "total_steps": int(total_steps),
        "total_updates": int(len(update_results)),
        "warmup_steps": int(warmup_steps),
        "actor_steps": int(actor_steps),
    }
