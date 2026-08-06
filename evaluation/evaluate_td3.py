"""Deterministic evaluation of a frozen TD3 policy on CCPSO."""

import numpy as np

from environments.factory import make_ccpso_env
from evaluation.common import (
    STATE_FIELDS,
    serialize_problem,
    validate_evaluation_seeds,
)
from problems.spec import ProblemSpec


def evaluate_td3_policy(
    policy,
    problem: ProblemSpec,
    *,
    particles: int,
    max_fe: int,
    seeds,
    c_min: float = 0.0,
    c_max: float = 1.5,
    recent_window: int = 5,
    stagnation_horizon: int = 10,
    reward_mode: str = "step_log_improvement",
    reward_epsilon: float = 1e-12,
):
    """Evaluate one deterministic TD3 policy without training or replay."""
    if not isinstance(problem, ProblemSpec):
        raise TypeError("problem must be an instance of ProblemSpec")
    evaluation_seeds = validate_evaluation_seeds(seeds)

    episode_records = []
    step_records = []
    environment_metadata = None

    for seed in evaluation_seeds:
        env = make_ccpso_env(
            problem,
            particles=particles,
            max_fe=max_fe,
            seed=seed,
            c_min=c_min,
            c_max=c_max,
            recent_window=recent_window,
            stagnation_horizon=stagnation_horizon,
            reward_mode=reward_mode,
            reward_epsilon=reward_epsilon,
        )
        current_environment = {
            "c_min": float(env.c_min),
            "c_max": float(env.c_max),
            "recent_window": int(env.recent_window),
            "stagnation_horizon": int(env.stagnation_horizon),
            "reward_mode": env.reward_mode,
            "reward_epsilon": float(env.reward_epsilon),
        }
        if environment_metadata is None:
            environment_metadata = current_environment
        elif current_environment != environment_metadata:
            raise RuntimeError(
                "evaluation environment configuration changed across seeds"
            )
        episode_return = 0.0
        episode_step = 0
        c_values = []

        try:
            state, reset_info = env.reset(seed=seed)
            initial_fe = int(env.swarm.fe_count)
            initial_best = float(env.swarm.gbest_fitness)
            initial_gap = float(
                max(initial_best - problem.optimum, 0.0)
            )
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
                        "evaluate_td3_policy expected a six-dimensional "
                        f"state, got shape {action_state.shape}"
                    )

                action = np.asarray(
                    policy.select_action(action_state),
                    dtype=np.float32,
                )
                if action.shape != env.action_space.shape:
                    raise ValueError(
                        "policy action has shape "
                        f"{action.shape}, expected {env.action_space.shape}"
                    )
                if not np.all(np.isfinite(action)):
                    raise FloatingPointError(
                        f"policy action must be finite, got {action!r}"
                    )

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                episode_step += 1
                episode_return += float(reward)
                c_values.append(float(info["conv"]))
                step_records.append(
                    {
                        "seed": int(seed),
                        "episode_step": int(episode_step),
                        "decision_fe": int(decision_fe),
                        "fe_count": int(info["fe_count"]),
                        "gbest_fitness": float(
                            info["gbest_fitness"]
                        ),
                        "gap": float(info["gap"]),
                        "raw_action": float(info["raw_action"]),
                        "c_value": float(info["conv"]),
                        "reward": float(reward),
                        "reward_progress": float(
                            info["reward_progress"]
                        ),
                        "state_fe_progress": float(action_state[0]),
                        "state_recent_progress": float(action_state[1]),
                        "state_position_diversity": float(
                            action_state[2]
                        ),
                        "state_q_diversity": float(action_state[3]),
                        "state_movement": float(action_state[4]),
                        "state_stagnation": float(action_state[5]),
                    }
                )
                state = next_state

            if truncated or not terminated:
                raise RuntimeError(
                    f"evaluation for seed {seed} did not terminate by FE"
                )

            c_array = np.asarray(c_values, dtype=np.float64)
            episode_records.append(
                {
                    "seed": int(seed),
                    "final_best": float(env.swarm.gbest_fitness),
                    "gap": float(
                        max(
                            env.swarm.gbest_fitness - problem.optimum,
                            0.0,
                        )
                    ),
                    "steps": int(episode_step),
                    "initial_fe": initial_fe,
                    "initial_best": initial_best,
                    "initial_gap": initial_gap,
                    "final_fe": int(env.swarm.fe_count),
                    "return": float(episode_return),
                    "reward_mode": reset_info["reward_mode"],
                    "initial_improvement_scale": float(
                        reset_info["initial_improvement_scale"]
                    ),
                    "initial_gap_scale": float(
                        reset_info["initial_gap_scale"]
                    ),
                    "c_mean": float(np.mean(c_array)),
                    "c_min": float(np.min(c_array)),
                    "c_max": float(np.max(c_array)),
                }
            )
        finally:
            env.close()

    final_gaps = np.asarray(
        [episode["gap"] for episode in episode_records],
        dtype=np.float64,
    )
    return {
        "problem": serialize_problem(problem),
        "environment": environment_metadata,
        "episodes": episode_records,
        "steps": step_records,
        "final_gap_statistics": {
            "mean": float(np.mean(final_gaps)),
            "median": float(np.median(final_gaps)),
            "std": float(np.std(final_gaps)),
            "min": float(np.min(final_gaps)),
            "max": float(np.max(final_gaps)),
        },
    }
