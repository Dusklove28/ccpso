"""One-command classic-function TD3 experiment pipeline."""

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np

from environments.ccpso_env import REWARD_MODES
from evaluation.evaluate_td3 import evaluate_td3_policy
from evaluation.plot_run import plot_run
from training.run_artifacts import save_classic_td3_run
from training.td3_experiment import (
    ClassicTD3ExperimentConfig,
    run_classic_td3,
)
from training.td3_online import TD3OnlineConfig


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


def run_classic_td3_pipeline(
    config,
    *,
    output_root,
    run_name,
    evaluation_seeds,
):
    """Train, persist, evaluate, and plot one classic TD3 run."""
    if not isinstance(config, ClassicTD3ExperimentConfig):
        raise TypeError(
            "config must be an instance of ClassicTD3ExperimentConfig"
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

    result = run_classic_td3(config)
    artifact_paths = save_classic_td3_run(result, run_path)

    probe_state = np.zeros(result.policy.state_dim, dtype=np.float32)
    probe_action = result.policy.select_action(probe_state)
    checkpoint_metadata = {
        "kind": "classic_td3_policy",
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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one classic-function TD3-CCPSO experiment."
    )
    parser.add_argument(
        "--problem",
        choices=("sphere", "rastrigin", "rosenbrock"),
        default="sphere",
    )
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--max-fe", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--c-min", type=float, default=0.0)
    parser.add_argument("--c-max", type=float, default=1.5)
    parser.add_argument("--recent-window", type=int, default=5)
    parser.add_argument("--stagnation-horizon", type=int, default=10)
    parser.add_argument(
        "--reward-mode",
        choices=REWARD_MODES,
        default="step_log_improvement",
    )
    parser.add_argument("--reward-epsilon", type=float, default=1e-12)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        required=True,
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def config_from_args(args):
    online = TD3OnlineConfig(
        episodes=args.episodes,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        exploration_noise=args.exploration_noise,
        updates_per_step=args.updates_per_step,
        seed=args.seed,
    )
    return ClassicTD3ExperimentConfig(
        problem_name=args.problem,
        dimensions=args.dimensions,
        particles=args.particles,
        max_fe=args.max_fe,
        buffer_capacity=args.buffer_capacity,
        device=args.device,
        c_min=args.c_min,
        c_max=args.c_max,
        recent_window=args.recent_window,
        stagnation_horizon=args.stagnation_horizon,
        reward_mode=args.reward_mode,
        reward_epsilon=args.reward_epsilon,
        discount=args.discount,
        online=online,
    )


def main(argv=None):
    args = parse_args(argv)
    config = config_from_args(args)
    pipeline_result = run_classic_td3_pipeline(
        config,
        output_root=args.output_root,
        run_name=args.run_name,
        evaluation_seeds=args.evaluation_seeds,
    )
    print(
        json.dumps(
            pipeline_result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return pipeline_result


if __name__ == "__main__":
    main()
