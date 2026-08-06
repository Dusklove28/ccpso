"""Shared disk pipeline for one TD3-controlled optimization problem."""

from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np

from evaluation.evaluate_td3 import evaluate_td3_policy
from evaluation.plot_run import plot_run
from problems.spec import ProblemSpec
from training.run_artifacts import save_td3_run
from training.td3_experiment import (
    ClassicTD3ExperimentConfig,
    ClassicTD3ExperimentResult,
    TD3ProblemExperimentConfig,
    run_td3_problem,
)


def _validate_run_name(run_name):
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError(
            f"run_name must be a non-empty string, got {run_name!r}"
        )
    normalized = run_name.strip()
    if Path(normalized).name != normalized or normalized in (".", ".."):
        raise ValueError(
            "run_name must be a single directory name, "
            f"got {run_name!r}"
        )
    return normalized


def _validate_evaluation_seeds(seeds):
    if isinstance(seeds, np.ndarray):
        if seeds.ndim != 1:
            raise ValueError(
                "evaluation_seeds must be a non-empty integer sequence"
            )
        values = seeds.tolist()
    elif isinstance(seeds, Sequence) and not isinstance(
        seeds,
        (str, bytes),
    ):
        values = list(seeds)
    else:
        raise ValueError(
            "evaluation_seeds must be a non-empty integer sequence"
        )

    if not values:
        raise ValueError(
            "evaluation_seeds must be a non-empty integer sequence"
        )

    validated = []
    for index, seed in enumerate(values):
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
        ):
            raise ValueError(
                "evaluation_seeds must contain only integers; "
                f"evaluation_seeds[{index}]={seed!r}"
            )
        validated.append(int(seed))
    return validated


def _prepare_run_path(output_root, run_name):
    output_path = Path(output_root).expanduser().resolve()
    run_path = output_path / _validate_run_name(run_name)
    if run_path.exists():
        if not run_path.is_dir() or any(run_path.iterdir()):
            raise FileExistsError(
                f"run directory already exists and is not empty: {run_path}"
            )
    return run_path


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")


def run_td3_pipeline(
    problem: ProblemSpec,
    config: TD3ProblemExperimentConfig,
    *,
    output_root,
    run_name,
    evaluation_seeds,
):
    """Train, persist, freeze-evaluate, and plot one problem run."""
    if not isinstance(problem, ProblemSpec):
        raise TypeError("problem must be an instance of ProblemSpec")
    if not isinstance(
        config,
        (TD3ProblemExperimentConfig, ClassicTD3ExperimentConfig),
    ):
        raise TypeError(
            "config must be a TD3ProblemExperimentConfig or "
            "ClassicTD3ExperimentConfig"
        )

    run_path = _prepare_run_path(output_root, run_name)
    evaluation_seed_values = _validate_evaluation_seeds(
        evaluation_seeds
    )
    training_seeds = [
        config.online.seed + episode_index
        for episode_index in range(config.online.episodes)
    ]
    overlapping_seeds = sorted(
        set(training_seeds).intersection(evaluation_seed_values)
    )
    if overlapping_seeds:
        raise ValueError(
            "evaluation_seeds must not overlap training episode seeds; "
            f"overlap={overlapping_seeds}"
        )

    result = run_td3_problem(problem, config)
    artifact_paths = save_td3_run(result, run_path)

    probe_state = np.zeros(result.policy.state_dim, dtype=np.float32)
    probe_action = result.policy.select_action(probe_state)
    checkpoint_kind = (
        "classic_td3_policy"
        if isinstance(result, ClassicTD3ExperimentResult)
        else "td3_problem_policy"
    )
    checkpoint_metadata = {
        "kind": checkpoint_kind,
        "problem": result.problem_metadata,
        "training_seeds": training_seeds,
        "evaluation_seeds": evaluation_seed_values,
        "experiment_config": result.config,
        "actor_probe": {
            "state": probe_state.tolist(),
            "action": probe_action.tolist(),
        },
    }
    checkpoint_path = result.policy.save_checkpoint(
        run_path / "checkpoints" / "policy.pt",
        metadata=checkpoint_metadata,
    )

    evaluation = evaluate_td3_policy(
        result.policy,
        result.problem,
        particles=config.particles,
        max_fe=config.max_fe,
        seeds=evaluation_seed_values,
        c_min=config.c_min,
        c_max=config.c_max,
        recent_window=config.recent_window,
        stagnation_horizon=config.stagnation_horizon,
        reward_mode=config.reward_mode,
        reward_epsilon=config.reward_epsilon,
    )
    evaluation_path = (run_path / "evaluation.json").resolve()
    _write_json(evaluation_path, evaluation)

    figure_paths = plot_run(run_path)
    paths = {
        "run_dir": str(run_path.resolve()),
        **artifact_paths,
        "evaluation": str(evaluation_path),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "figures": figure_paths,
    }
    summary = {
        "run_name": run_path.name,
        "problem": result.problem_metadata,
        "training_seeds": training_seeds,
        "evaluation_seeds": evaluation_seed_values,
        "total_steps": int(result.training_records["total_steps"]),
        "total_updates": int(result.training_records["total_updates"]),
        "reward_mode": result.config["environment"]["reward_mode"],
        "discount": float(result.config["td3"]["discount"]),
        "evaluation_final_gap_statistics": evaluation[
            "final_gap_statistics"
        ],
        "actor_probe": checkpoint_metadata["actor_probe"],
    }
    json.dumps(summary, ensure_ascii=False, allow_nan=False)
    return {
        "paths": paths,
        "summary": summary,
    }
