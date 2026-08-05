"""Deterministic evaluation of a frozen TD3 policy on CCPSO."""

from collections.abc import Sequence

import numpy as np

from environments.factory import make_ccpso_env
from problems.spec import ProblemSpec


STATE_FIELDS = (
    "state_fe_progress",
    "state_recent_progress",
    "state_position_diversity",
    "state_q_diversity",
    "state_movement",
    "state_stagnation",
)


def _validate_seeds(seeds):
    if isinstance(seeds, np.ndarray):
        if seeds.ndim != 1:
            raise ValueError(
                "seeds must be a non-empty one-dimensional integer sequence"
            )
        seed_values = seeds.tolist()
    elif isinstance(seeds, Sequence) and not isinstance(
        seeds,
        (str, bytes),
    ):
        seed_values = list(seeds)
    else:
        raise ValueError("seeds must be a non-empty integer sequence")

    if not seed_values:
        raise ValueError("seeds must be a non-empty integer sequence")

    validated = []
    for index, seed in enumerate(seed_values):
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
        ):
            raise ValueError(
                "seeds must contain only integers; "
                f"seeds[{index}]={seed!r}"
            )
        validated.append(int(seed))
    return validated


def _serialize_problem(problem):
    problem_id = problem.problem_id
    if isinstance(problem_id, np.generic):
        problem_id = problem_id.item()
    return {
        "suite": problem.suite,
        "problem_id": problem_id,
        "name": problem.name,
        "dimensions": int(problem.dimensions),
        "lower_bound": problem.lower_bound.tolist(),
        "upper_bound": problem.upper_bound.tolist(),
        "optimum": float(problem.optimum),
    }


def evaluate_td3_policy(
    policy,
    problem: ProblemSpec,
    *,
    particles: int,
    max_fe: int,
    seeds,
    c_min: float = 0.0,
    c_max: float = 1.5,
):
    """Evaluate one deterministic TD3 policy without training or replay."""
    if not isinstance(problem, ProblemSpec):
        raise TypeError("problem must be an instance of ProblemSpec")
    evaluation_seeds = _validate_seeds(seeds)

    episode_records = []
    step_records = []

    for seed in evaluation_seeds:
        env = make_ccpso_env(
            problem,
            particles=particles,
            max_fe=max_fe,
            seed=seed,
            c_min=c_min,
            c_max=c_max,
        )
        episode_return = 0.0
        episode_step = 0
        c_values = []

        try:
            state, _ = env.reset(seed=seed)
            terminated = False
            truncated = False

            while not (terminated or truncated):
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
                        "fe_count": int(info["fe_count"]),
                        "gbest_fitness": float(
                            info["gbest_fitness"]
                        ),
                        "gap": float(info["gap"]),
                        "raw_action": float(info["raw_action"]),
                        "c_value": float(info["conv"]),
                        "reward": float(reward),
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
                    "final_fe": int(env.swarm.fe_count),
                    "return": float(episode_return),
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
        "problem": _serialize_problem(problem),
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
