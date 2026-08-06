"""Deterministic CCPSO evaluation for non-learning C schedules."""

import numpy as np

from baselines.c_schedules import (
    c_value_to_action,
    make_fixed_c_schedule,
    make_linear_c_schedule,
)
from environments.factory import make_ccpso_env
from evaluation.common import (
    serialize_problem,
    validate_evaluation_seeds,
)
from problems.spec import ProblemSpec


def _evaluate_c_schedule(
    problem,
    *,
    particles,
    max_fe,
    seeds,
    c_min,
    c_max,
    baseline,
    schedule_factory,
):
    if not isinstance(problem, ProblemSpec):
        raise TypeError("problem must be an instance of ProblemSpec")
    evaluation_seeds = validate_evaluation_seeds(seeds)

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
            remaining_fe = env.swarm.max_fe - env.swarm.fe_count
            num_steps = remaining_fe // env.swarm.particles
            if num_steps < 1:
                raise ValueError(
                    "baseline evaluation requires at least one complete "
                    "CCPSO step after reset"
                )
            schedule = schedule_factory(int(num_steps))

            terminated = False
            truncated = False
            for planned_c in schedule:
                action_state = np.asarray(
                    state,
                    dtype=np.float32,
                ).copy()
                if action_state.shape != (6,):
                    raise ValueError(
                        "C baseline evaluation expected a "
                        "six-dimensional state, "
                        f"got shape {action_state.shape}"
                    )

                action_value = c_value_to_action(
                    planned_c,
                    c_min,
                    c_max,
                )
                action = np.array([action_value], dtype=np.float32)
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

                if terminated or truncated:
                    break

            if truncated or not terminated:
                raise RuntimeError(
                    f"evaluation for seed {seed} did not terminate by FE"
                )
            if episode_step != int(num_steps):
                raise RuntimeError(
                    f"evaluation for seed {seed} terminated after "
                    f"{episode_step} steps; expected {num_steps}"
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
        "baseline": baseline,
        "problem": serialize_problem(problem),
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


def evaluate_fixed_c(
    problem: ProblemSpec,
    *,
    particles,
    max_fe,
    seeds,
    c_value,
    c_min=0.0,
    c_max=1.5,
):
    """Evaluate one fixed C value through the CCPSO environment."""
    validated_c = float(make_fixed_c_schedule(c_value, 1)[0])
    c_value_to_action(validated_c, c_min, c_max)
    return _evaluate_c_schedule(
        problem,
        particles=particles,
        max_fe=max_fe,
        seeds=seeds,
        c_min=c_min,
        c_max=c_max,
        baseline={"name": "fixed_c", "c_value": validated_c},
        schedule_factory=lambda num_steps: make_fixed_c_schedule(
            validated_c,
            num_steps,
        ),
    )


def evaluate_linear_c(
    problem: ProblemSpec,
    *,
    particles,
    max_fe,
    seeds,
    c_start=1.5,
    c_end=0.0,
    c_min=0.0,
    c_max=1.5,
):
    """Evaluate the deterministic schedule named ``linear_c``."""
    validated_start = float(make_fixed_c_schedule(c_start, 1)[0])
    validated_end = float(make_fixed_c_schedule(c_end, 1)[0])
    c_value_to_action(validated_start, c_min, c_max)
    c_value_to_action(validated_end, c_min, c_max)
    return _evaluate_c_schedule(
        problem,
        particles=particles,
        max_fe=max_fe,
        seeds=seeds,
        c_min=c_min,
        c_max=c_max,
        baseline={
            "name": "linear_c",
            "c_start": validated_start,
            "c_end": validated_end,
        },
        schedule_factory=lambda num_steps: make_linear_c_schedule(
            validated_start,
            validated_end,
            num_steps,
        ),
    )
