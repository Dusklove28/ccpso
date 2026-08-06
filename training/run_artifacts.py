import csv
import json
from pathlib import Path

from training.td3_experiment import (
    ClassicTD3ExperimentResult,
    TD3ProblemExperimentResult,
)


STEP_COLUMNS = (
    "episode_index",
    "episode_seed",
    "global_step",
    "episode_step",
    "decision_fe",
    "cumulative_training_fe",
    "action_source",
    "reward",
    "reward_progress",
    "episode_return",
    "fe_count",
    "gbest_fitness",
    "gap",
    "raw_action",
    "c_value",
    "mean_movement",
    "recent_progress",
    "stagnation_steps",
    "boundary_clip_ratio",
    "terminated",
    "truncated",
    "state_fe_progress",
    "state_recent_progress",
    "state_position_diversity",
    "state_q_diversity",
    "state_movement",
    "state_stagnation",
)

EPISODE_COLUMNS = (
    "episode_index",
    "seed",
    "steps",
    "return",
    "reward_mode",
    "initial_improvement_scale",
    "initial_gap_scale",
    "final_fe",
    "cumulative_training_fe",
    "final_best",
    "terminated",
    "truncated",
    "c_mean",
    "c_min",
    "c_max",
)

UPDATE_COLUMNS = (
    "global_step",
    "episode_index",
    "episode_step",
    "cumulative_training_fe",
    "total_it",
    "critic_loss",
    "actor_updated",
    "actor_loss",
    "target_q_mean",
    "q1_mean",
    "q2_mean",
)


def _prepare_run_directory(run_dir):
    run_path = Path(run_dir).expanduser()
    if run_path.exists():
        if not run_path.is_dir() or any(run_path.iterdir()):
            raise FileExistsError(
                f"run_dir already exists and is not empty: {run_path}"
            )
    else:
        run_path.mkdir(parents=True, exist_ok=False)
    return run_path.resolve()


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


def _write_csv(path, columns, records):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def _make_summary(result):
    records = result.training_records
    episodes = records["episodes"]
    if not episodes:
        raise ValueError("training records contain no episodes")

    summary = {
        "suite": result.problem_metadata["suite"],
        "problem_id": result.problem_metadata["problem_id"],
        "problem_name": result.problem_metadata["name"],
        "dimensions": int(result.problem_metadata["dimensions"]),
        "reward_mode": result.config["environment"]["reward_mode"],
        "discount": float(result.config["td3"]["discount"]),
        "episode_count": int(len(episodes)),
        "total_steps": int(records["total_steps"]),
        "total_updates": int(records["total_updates"]),
        "warmup_steps": int(records["warmup_steps"]),
        "actor_steps": int(records["actor_steps"]),
        "last_episode_final_best": float(episodes[-1]["final_best"]),
        "best_episode_final_best": float(
            min(episode["final_best"] for episode in episodes)
        ),
    }
    if "source_function_id" in result.problem_metadata:
        summary["source_function_id"] = int(
            result.problem_metadata["source_function_id"]
        )
        summary["category"] = result.problem_metadata["category"]
    return summary


def save_td3_run(result, run_dir):
    if not isinstance(result, TD3ProblemExperimentResult):
        raise TypeError(
            "result must be an instance of TD3ProblemExperimentResult"
        )

    run_path = _prepare_run_directory(run_dir)
    paths = {
        "config": run_path / "config.json",
        "problem": run_path / "problem.json",
        "summary": run_path / "summary.json",
        "steps": run_path / "steps.csv",
        "episodes": run_path / "episodes.csv",
        "updates": run_path / "updates.csv",
    }

    _write_json(paths["config"], result.config)
    _write_json(paths["problem"], result.problem_metadata)
    _write_json(paths["summary"], _make_summary(result))
    _write_csv(
        paths["steps"],
        STEP_COLUMNS,
        result.training_records["steps"],
    )
    _write_csv(
        paths["episodes"],
        EPISODE_COLUMNS,
        result.training_records["episodes"],
    )
    _write_csv(
        paths["updates"],
        UPDATE_COLUMNS,
        result.training_records["updates"],
    )

    return {
        name: str(path)
        for name, path in paths.items()
    }


def save_classic_td3_run(result, run_dir):
    if not isinstance(result, ClassicTD3ExperimentResult):
        raise TypeError(
            "result must be an instance of ClassicTD3ExperimentResult"
        )
    return save_td3_run(result, run_dir)
