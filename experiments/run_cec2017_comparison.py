"""Sequential CEC2017 comparison for two C baselines and one-step TD3."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from environments.ccpso_env import REWARD_MODES
from evaluation.common import validate_evaluation_seeds
from evaluation.evaluate_c_baselines import (
    evaluate_fixed_c,
    evaluate_linear_c,
)
from experiments.comparison_artifacts import (
    save_comparison_artifacts,
    write_strict_json,
)
from experiments.td3_pipeline import run_td3_pipeline
from problems.cec2017 import (
    CEC2017_FUNCTION_IDS,
    CEC2017_REPRESENTATIVE_IDS,
    CEC2017_SUPPORTED_DIMENSIONS,
    make_cec2017_problem,
)
from training.td3_experiment import TD3ProblemExperimentConfig
from training.td3_online import TD3OnlineConfig


FIXED_C_METHOD = "fixed_c_100"
LINEAR_C_METHOD = "linear_c"
TD3_METHOD = "td3_n1"


def _validate_function_ids(function_ids):
    values = validate_evaluation_seeds(function_ids)
    invalid = [value for value in values if value not in CEC2017_FUNCTION_IDS]
    if invalid:
        raise ValueError(
            "function_ids must use public CEC2017 IDs in [1, 29], "
            f"got invalid IDs {invalid}"
        )
    if len(set(values)) != len(values):
        raise ValueError(
            f"function_ids must not contain duplicates, got {values}"
        )
    return tuple(values)


def _validate_seed_group(name, seeds):
    try:
        values = validate_evaluation_seeds(seeds)
    except ValueError as error:
        raise ValueError(f"invalid {name}: {error}") from error
    if any(seed < 0 for seed in values):
        raise ValueError(f"{name} must contain non-negative seeds, got {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates, got {values}")
    return tuple(values)


@dataclass(frozen=True)
class CEC2017ComparisonConfig:
    """Configuration shared by all functions and comparison methods."""

    function_ids: object = CEC2017_REPRESENTATIVE_IDS
    dimensions: int = 10
    particles: int = 20
    max_fe: int = 100_000
    training_seeds: object = (0,)
    evaluation_seeds: object = (10_000,)
    episodes: int = 1
    learning_starts: int = 1_000
    batch_size: int = 256
    buffer_capacity: int = 100_000
    exploration_noise: float = 0.1
    updates_per_step: int = 1
    device: object = "auto"
    max_action: float = 1.0
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    c_min: float = 0.0
    c_max: float = 1.5
    recent_window: int = 5
    stagnation_horizon: int = 10
    reward_mode: str = "step_log_improvement"
    reward_epsilon: float = 1e-12

    def __post_init__(self):
        function_ids = _validate_function_ids(self.function_ids)
        training_seeds = _validate_seed_group(
            "training_seeds",
            self.training_seeds,
        )
        evaluation_seeds = _validate_seed_group(
            "evaluation_seeds",
            self.evaluation_seeds,
        )
        if (
            isinstance(self.dimensions, (bool, np.bool_))
            or not isinstance(self.dimensions, (int, np.integer))
            or int(self.dimensions) not in CEC2017_SUPPORTED_DIMENSIONS
        ):
            raise ValueError(
                "dimensions must be one of "
                f"{CEC2017_SUPPORTED_DIMENSIONS}, got {self.dimensions!r}"
            )

        probe = self._make_td3_config(training_seeds[0])
        if probe.c_min > 0.0 or probe.c_max < 1.5:
            raise ValueError(
                "comparison C range must contain fixed C=1.0 and the "
                "linear_c endpoints [0.0, 1.5], got "
                f"[{probe.c_min}, {probe.c_max}]"
            )

        training_episode_seeds = {
            seed + episode_index
            for seed in training_seeds
            for episode_index in range(probe.online.episodes)
        }
        overlap = sorted(
            training_episode_seeds.intersection(evaluation_seeds)
        )
        if overlap:
            raise ValueError(
                "TD3 training episode seeds and evaluation_seeds must not "
                f"overlap; overlap={overlap}"
            )

        object.__setattr__(self, "function_ids", function_ids)
        object.__setattr__(self, "training_seeds", training_seeds)
        object.__setattr__(self, "evaluation_seeds", evaluation_seeds)
        object.__setattr__(self, "dimensions", int(self.dimensions))

        normalized_fields = (
            "particles",
            "max_fe",
            "buffer_capacity",
            "device",
            "max_action",
            "discount",
            "tau",
            "policy_noise",
            "noise_clip",
            "policy_freq",
            "c_min",
            "c_max",
            "recent_window",
            "stagnation_horizon",
            "reward_mode",
            "reward_epsilon",
        )
        for name in normalized_fields:
            object.__setattr__(self, name, getattr(probe, name))
        for name in (
            "episodes",
            "learning_starts",
            "batch_size",
            "exploration_noise",
            "updates_per_step",
        ):
            object.__setattr__(self, name, getattr(probe.online, name))

    def _make_td3_config(self, training_seed):
        online = TD3OnlineConfig(
            episodes=self.episodes,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            exploration_noise=self.exploration_noise,
            updates_per_step=self.updates_per_step,
            seed=training_seed,
        )
        return TD3ProblemExperimentConfig(
            particles=self.particles,
            max_fe=self.max_fe,
            buffer_capacity=self.buffer_capacity,
            online=online,
            device=self.device,
            max_action=self.max_action,
            discount=self.discount,
            tau=self.tau,
            policy_noise=self.policy_noise,
            noise_clip=self.noise_clip,
            policy_freq=self.policy_freq,
            c_min=self.c_min,
            c_max=self.c_max,
            recent_window=self.recent_window,
            stagnation_horizon=self.stagnation_horizon,
            reward_mode=self.reward_mode,
            reward_epsilon=self.reward_epsilon,
        )

    def td3_config_for_seed(self, training_seed):
        if training_seed not in self.training_seeds:
            raise ValueError(
                f"training_seed {training_seed!r} is not configured"
            )
        return self._make_td3_config(training_seed)

    def to_manifest_dict(self):
        device = None if self.device is None else str(self.device)
        return {
            "function_ids": list(self.function_ids),
            "dimensions": self.dimensions,
            "particles": self.particles,
            "max_fe": self.max_fe,
            "training_seeds": list(self.training_seeds),
            "evaluation_seeds": list(self.evaluation_seeds),
            "td3": {
                "episodes": self.episodes,
                "learning_starts": self.learning_starts,
                "batch_size": self.batch_size,
                "buffer_capacity": self.buffer_capacity,
                "exploration_noise": self.exploration_noise,
                "updates_per_step": self.updates_per_step,
                "device": device,
                "max_action": self.max_action,
                "discount": self.discount,
                "tau": self.tau,
                "policy_noise": self.policy_noise,
                "noise_clip": self.noise_clip,
                "policy_freq": self.policy_freq,
            },
            "environment": {
                "c_min": self.c_min,
                "c_max": self.c_max,
                "recent_window": self.recent_window,
                "stagnation_horizon": self.stagnation_horizon,
                "reward_mode": self.reward_mode,
                "reward_epsilon": self.reward_epsilon,
            },
        }


def _validate_run_name(run_name):
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError(
            f"run_name must be a non-empty string, got {run_name!r}"
        )
    normalized = run_name.strip()
    if Path(normalized).name != normalized or normalized in {".", ".."}:
        raise ValueError(
            f"run_name must be a single directory name, got {run_name!r}"
        )
    return normalized


def _prepare_comparison_directory(output_root, run_name):
    run_path = (
        Path(output_root).expanduser().resolve()
        / _validate_run_name(run_name)
    )
    if run_path.exists():
        if not run_path.is_dir() or any(run_path.iterdir()):
            raise FileExistsError(
                "comparison directory already exists and is not empty: "
                f"{run_path}"
            )
    else:
        run_path.mkdir(parents=True, exist_ok=False)
    return run_path


def _load_strict_json(path):
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value}")

    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file, parse_constant=reject_constant)


def _append_episode_rows(
    records,
    *,
    problem,
    method,
    training_seed,
    evaluation,
):
    metadata = evaluation["problem"]
    if metadata["problem_id"] != problem.problem_id:
        raise RuntimeError("evaluation problem_id does not match task")
    if metadata["source_function_id"] != problem.source_function_id:
        raise RuntimeError(
            "evaluation source_function_id does not match task"
        )
    expected_seeds = [episode["seed"] for episode in evaluation["episodes"]]
    for episode in evaluation["episodes"]:
        records.append(
            {
                "problem_id": int(problem.problem_id),
                "source_function_id": int(problem.source_function_id),
                "category": problem.category,
                "method": method,
                "training_seed": training_seed,
                "evaluation_seed": int(episode["seed"]),
                "final_best": float(episode["final_best"]),
                "gap": float(episode["gap"]),
                "steps": int(episode["steps"]),
                "final_fe": int(episode["final_fe"]),
            }
        )
    return expected_seeds


def _summarize(per_run_records):
    grouped = {}
    for record in per_run_records:
        key = (
            record["problem_id"],
            record["source_function_id"],
            record["category"],
            record["method"],
            record["training_seed"],
        )
        grouped.setdefault(key, []).append(record["gap"])

    summaries = []
    for key, gaps in grouped.items():
        values = np.asarray(gaps, dtype=np.float64)
        summaries.append(
            {
                "problem_id": key[0],
                "source_function_id": key[1],
                "category": key[2],
                "method": key[3],
                "training_seed": key[4],
                "runs": int(values.size),
                "mean_gap": float(np.mean(values)),
                "median_gap": float(np.median(values)),
                "std_gap": float(np.std(values)),
                "min_gap": float(np.min(values)),
                "max_gap": float(np.max(values)),
            }
        )
    return summaries


def run_cec2017_comparison(config, *, output_root, run_name):
    """Sequentially run fixed C, linear C, and one-step TD3 tasks."""
    if not isinstance(config, CEC2017ComparisonConfig):
        raise TypeError(
            "config must be an instance of CEC2017ComparisonConfig"
        )

    run_path = _prepare_comparison_directory(output_root, run_name)
    per_run_records = []
    task_records = []
    task_index = 0

    for function_id in config.function_ids:
        problem = make_cec2017_problem(function_id, config.dimensions)
        function_dir = f"f{function_id:02d}"

        baseline_tasks = (
            (
                FIXED_C_METHOD,
                lambda: evaluate_fixed_c(
                    problem,
                    particles=config.particles,
                    max_fe=config.max_fe,
                    seeds=config.evaluation_seeds,
                    c_value=1.0,
                    c_min=config.c_min,
                    c_max=config.c_max,
                ),
            ),
            (
                LINEAR_C_METHOD,
                lambda: evaluate_linear_c(
                    problem,
                    particles=config.particles,
                    max_fe=config.max_fe,
                    seeds=config.evaluation_seeds,
                    c_start=1.5,
                    c_end=0.0,
                    c_min=config.c_min,
                    c_max=config.c_max,
                ),
            ),
        )
        for method, evaluate in baseline_tasks:
            evaluation = evaluate()
            evaluation_path = (
                run_path
                / "baselines"
                / function_dir
                / method
                / "evaluation.json"
            )
            write_strict_json(evaluation_path, evaluation)
            episode_seeds = _append_episode_rows(
                per_run_records,
                problem=problem,
                method=method,
                training_seed=None,
                evaluation=evaluation,
            )
            if episode_seeds != list(config.evaluation_seeds):
                raise RuntimeError(
                    f"{method} evaluation seeds do not match the "
                    "comparison configuration"
                )
            task_index += 1
            task_records.append(
                {
                    "task_index": task_index,
                    "problem_id": problem.problem_id,
                    "source_function_id": problem.source_function_id,
                    "category": problem.category,
                    "method": method,
                    "training_seed": None,
                    "evaluation_seeds": episode_seeds,
                    "evaluation": evaluation_path.relative_to(
                        run_path
                    ).as_posix(),
                    "checkpoint": None,
                }
            )

        for training_seed in config.training_seeds:
            td3_config = config.td3_config_for_seed(training_seed)
            pipeline = run_td3_pipeline(
                problem,
                td3_config,
                output_root=run_path / "td3" / function_dir,
                run_name=f"seed_{training_seed}",
                evaluation_seeds=config.evaluation_seeds,
            )
            evaluation_path = Path(pipeline["paths"]["evaluation"])
            evaluation = _load_strict_json(evaluation_path)
            episode_seeds = _append_episode_rows(
                per_run_records,
                problem=problem,
                method=TD3_METHOD,
                training_seed=training_seed,
                evaluation=evaluation,
            )
            if episode_seeds != list(config.evaluation_seeds):
                raise RuntimeError(
                    "TD3 evaluation seeds do not match the comparison "
                    "configuration"
                )
            task_index += 1
            task_records.append(
                {
                    "task_index": task_index,
                    "problem_id": problem.problem_id,
                    "source_function_id": problem.source_function_id,
                    "category": problem.category,
                    "method": TD3_METHOD,
                    "training_seed": training_seed,
                    "training_episode_seeds": [
                        training_seed + episode_index
                        for episode_index in range(config.episodes)
                    ],
                    "evaluation_seeds": episode_seeds,
                    "evaluation": evaluation_path.relative_to(
                        run_path
                    ).as_posix(),
                    "checkpoint": Path(
                        pipeline["paths"]["checkpoint"]
                    ).relative_to(run_path).as_posix(),
                }
            )

    summary_records = _summarize(per_run_records)
    manifest = {
        "format_version": 1,
        "suite": "cec2017",
        "run_name": run_path.name,
        "execution": "sequential",
        "methods": [
            {
                "name": FIXED_C_METHOD,
                "kind": "baseline",
                "c_value": 1.0,
            },
            {
                "name": LINEAR_C_METHOD,
                "kind": "baseline",
                "c_start": 1.5,
                "c_end": 0.0,
            },
            {
                "name": TD3_METHOD,
                "kind": "td3",
                "n_step": 1,
            },
        ],
        "config": config.to_manifest_dict(),
        "task_count": len(task_records),
        "tasks": task_records,
    }
    artifact_paths = save_comparison_artifacts(
        run_path,
        manifest=manifest,
        per_run_records=per_run_records,
        summary_records=summary_records,
    )
    result = {
        "paths": {
            "run_dir": str(run_path),
            **artifact_paths,
        },
        "task_count": len(task_records),
        "per_run_count": len(per_run_records),
        "summary_count": len(summary_records),
    }
    json.dumps(result, allow_nan=False)
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially compare fixed C=1.0, linear_c, and one-step "
            "TD3 on public CEC2017 function IDs."
        )
    )
    parser.add_argument(
        "--function-ids",
        type=int,
        choices=CEC2017_FUNCTION_IDS,
        nargs="+",
        default=list(CEC2017_REPRESENTATIVE_IDS),
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        choices=CEC2017_SUPPORTED_DIMENSIONS,
        default=10,
    )
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--max-fe", type=int, required=True)
    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-action", type=float, default=1.0)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-freq", type=int, default=2)
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def config_from_args(args):
    return CEC2017ComparisonConfig(
        function_ids=args.function_ids,
        dimensions=args.dimensions,
        particles=args.particles,
        max_fe=args.max_fe,
        training_seeds=args.training_seeds,
        evaluation_seeds=args.evaluation_seeds,
        episodes=args.episodes,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        exploration_noise=args.exploration_noise,
        updates_per_step=args.updates_per_step,
        device=args.device,
        max_action=args.max_action,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        c_min=args.c_min,
        c_max=args.c_max,
        recent_window=args.recent_window,
        stagnation_horizon=args.stagnation_horizon,
        reward_mode=args.reward_mode,
        reward_epsilon=args.reward_epsilon,
    )


def main(argv=None):
    args = parse_args(argv)
    config = config_from_args(args)
    result = run_cec2017_comparison(
        config,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()


__all__ = [
    "CEC2017ComparisonConfig",
    "FIXED_C_METHOD",
    "LINEAR_C_METHOD",
    "TD3_METHOD",
    "build_parser",
    "config_from_args",
    "main",
    "parse_args",
    "run_cec2017_comparison",
]
