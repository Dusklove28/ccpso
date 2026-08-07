"""Plot frozen evaluation and training diagnostics for one TD3 run."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np


FIGURE_FILENAMES = {
    "convergence": "convergence.png",
    "c_trajectory": "c_trajectory.png",
    "rewards": "rewards.png",
    "states": "states.png",
    "td3_losses": "td3_losses.png",
    "training_summary": "training_summary.png",
}

STATE_COLUMNS = (
    ("state_fe_progress", "FE progress"),
    ("state_recent_progress", "Recent progress"),
    ("state_position_diversity", "Position diversity"),
    ("state_q_diversity", "Q diversity"),
    ("state_movement", "Movement"),
    ("state_stagnation", "Stagnation"),
)


def _read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _run_label(config, problem):
    seed = config["online"]["seed"]
    state_mode = config.get("environment", {}).get(
        "state_mode",
        "legacy_v1",
    )
    return (
        f'{problem["name"]} (D={int(problem["dimensions"])}, '
        f'seed={seed}, state={state_mode})'
    )


def _aggregate_curves(curves):
    """Align seed curves on common FE values and calculate mean and IQR."""
    if not curves:
        return {"fe": [], "mean": [], "q25": [], "q75": [], "runs": 0}

    common_fe = set(curves[0])
    for curve in curves[1:]:
        common_fe.intersection_update(curve)
    fe_values = sorted(common_fe)
    if not fe_values:
        raise ValueError("evaluation seed curves have no common FE values")

    matrix = np.asarray(
        [[curve[fe] for fe in fe_values] for curve in curves],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError("evaluation curves contain non-finite values")
    return {
        "fe": [int(value) for value in fe_values],
        "mean": np.mean(matrix, axis=0).tolist(),
        "q25": np.percentile(matrix, 25.0, axis=0).tolist(),
        "q75": np.percentile(matrix, 75.0, axis=0).tolist(),
        "runs": int(matrix.shape[0]),
    }


def _group_evaluation(evaluation, particles):
    episodes = {int(row["seed"]): row for row in evaluation["episodes"]}
    grouped = {seed: [] for seed in episodes}
    for row in evaluation["steps"]:
        grouped.setdefault(int(row["seed"]), []).append(row)
    for records in grouped.values():
        records.sort(key=lambda row: int(row["episode_step"]))

    normalized = {}
    for seed, records in grouped.items():
        previous_fe = None
        rows = []
        for row in records:
            post_fe = int(row["fe_count"])
            if "decision_fe" in row:
                decision_fe = int(row["decision_fe"])
            elif previous_fe is not None:
                decision_fe = previous_fe
            else:
                decision_fe = post_fe - particles
            if decision_fe <= 0 or post_fe <= decision_fe:
                raise ValueError(
                    "evaluation FE fields must satisfy "
                    f"0 < decision_fe < fe_count, got {decision_fe}, {post_fe}"
                )
            copy = dict(row)
            copy["decision_fe"] = decision_fe
            copy["fe_count"] = post_fe
            rows.append(copy)
            previous_fe = post_fe
        normalized[seed] = rows
    return episodes, normalized


def _build_evaluation_plot_data(evaluation, particles):
    episodes, grouped = _group_evaluation(evaluation, particles)
    convergence_curves = []
    c_curves = []
    reward_curves = []
    state_curves = {column: [] for column, _ in STATE_COLUMNS}

    for seed, records in grouped.items():
        episode = episodes[seed]
        convergence = {}
        if all(key in episode for key in ("initial_fe", "initial_gap")):
            initial_fe = int(episode["initial_fe"])
            if initial_fe <= 0:
                raise ValueError(f"initial_fe must be positive, got {initial_fe}")
            convergence[initial_fe] = float(episode["initial_gap"])
        convergence.update(
            {int(row["fe_count"]): float(row["gap"]) for row in records}
        )
        convergence_curves.append(convergence)

        c_curves.append(
            {int(row["decision_fe"]): float(row["c_value"]) for row in records}
        )
        for column, _ in STATE_COLUMNS:
            state_curves[column].append(
                {
                    int(row["decision_fe"]): float(row[column])
                    for row in records
                }
            )

        cumulative = 0.0
        reward_curve = {}
        if records:
            reward_curve[int(records[0]["decision_fe"])] = 0.0
        for row in records:
            cumulative += float(row["reward"])
            reward_curve[int(row["fe_count"])] = cumulative
        reward_curves.append(reward_curve)

    return {
        "convergence": _aggregate_curves(convergence_curves),
        "c_trajectory": _aggregate_curves(c_curves),
        "rewards": _aggregate_curves(reward_curves),
        "states": {
            column: _aggregate_curves(curves)
            for column, curves in state_curves.items()
        },
    }


def _normalize_training_fe(steps, episodes, updates, particles):
    episode_offsets = {}
    cumulative = 0
    for episode in sorted(episodes, key=lambda row: int(row["episode_index"])):
        index = int(episode["episode_index"])
        episode_offsets[index] = cumulative
        cumulative += int(episode["final_fe"])

    normalized_steps = []
    by_global_step = {}
    previous_by_episode = {}
    for row in steps:
        copy = dict(row)
        episode_index = int(row["episode_index"])
        post_fe = int(row["fe_count"])
        if row.get("decision_fe", "") != "":
            decision_fe = int(row["decision_fe"])
        else:
            decision_fe = previous_by_episode.get(
                episode_index,
                post_fe - particles,
            )
        if row.get("cumulative_training_fe", "") != "":
            cumulative_fe = int(row["cumulative_training_fe"])
        else:
            cumulative_fe = episode_offsets[episode_index] + post_fe
        copy["decision_fe"] = decision_fe
        copy["cumulative_training_fe"] = cumulative_fe
        normalized_steps.append(copy)
        by_global_step[int(row["global_step"])] = cumulative_fe
        previous_by_episode[episode_index] = post_fe

    normalized_updates = []
    for row in updates:
        copy = dict(row)
        if row.get("cumulative_training_fe", "") != "":
            cumulative_fe = int(row["cumulative_training_fe"])
        else:
            cumulative_fe = by_global_step[int(row["global_step"])]
        copy["cumulative_training_fe"] = cumulative_fe
        normalized_updates.append(copy)
    return normalized_steps, normalized_updates


def build_plot_data(run_dir):
    """Load and numerically aggregate all data used by :func:`plot_run`."""
    run_path = Path(run_dir).expanduser().resolve()
    config = _read_json(run_path / "config.json")
    problem = _read_json(run_path / "problem.json")
    evaluation = _read_json(run_path / "evaluation.json")
    steps = _read_csv(run_path / "steps.csv")
    episodes = _read_csv(run_path / "episodes.csv")
    updates = _read_csv(run_path / "updates.csv")
    particles = int(config["particles"])
    training_steps, training_updates = _normalize_training_fe(
        steps,
        episodes,
        updates,
        particles,
    )
    return {
        "config": config,
        "problem": problem,
        "evaluation": _build_evaluation_plot_data(evaluation, particles),
        "training_steps": training_steps,
        "training_episodes": episodes,
        "training_updates": training_updates,
    }


def _save_and_close(fig, path, *, rect=None):
    try:
        if rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=rect)
        fig.savefig(path, dpi=150, format="png")
    finally:
        plt.close(fig)


def _plot_band(axis, aggregate, *, label, color=None):
    fe = aggregate["fe"]
    axis.plot(fe, aggregate["mean"], label=label, color=color, linewidth=1.8)
    axis.fill_between(
        fe,
        aggregate["q25"],
        aggregate["q75"],
        color=color,
        alpha=0.22,
        label="IQR",
    )


def _plot_convergence(data, run_label, path):
    fig, axis = plt.subplots(figsize=(8, 5))
    aggregate = data["convergence"]
    _plot_band(
        axis,
        aggregate,
        label=f'Frozen policy mean ({aggregate["runs"]} seeds)',
        color="tab:blue",
    )
    values = np.asarray(aggregate["mean"] + aggregate["q25"])
    axis.set_yscale("log" if values.size and np.all(values > 0.0) else "symlog")
    axis.set_title(f"Frozen-policy convergence — {run_label}")
    axis.set_xlabel("Episode function evaluations (FE)")
    axis.set_ylabel("Optimality gap")
    axis.grid(alpha=0.3)
    axis.legend()
    _save_and_close(fig, path)


def _plot_c_trajectory(data, run_label, path):
    fig, axis = plt.subplots(figsize=(8, 5))
    aggregate = data["c_trajectory"]
    _plot_band(axis, aggregate, label="Frozen policy mean C", color="tab:blue")
    axis.set_title(f"Frozen-policy C trajectory — {run_label}")
    axis.set_xlabel("Decision FE (before action)")
    axis.set_ylabel("C value")
    axis.grid(alpha=0.3)
    axis.legend()
    _save_and_close(fig, path)


def _plot_rewards(data, run_label, path):
    fig, axis = plt.subplots(figsize=(8, 5))
    aggregate = data["rewards"]
    _plot_band(
        axis,
        aggregate,
        label="Frozen policy mean cumulative return",
        color="tab:green",
    )
    axis.set_title(f"Frozen-policy cumulative return — {run_label}")
    axis.set_xlabel("Episode function evaluations (FE)")
    axis.set_ylabel("Cumulative return")
    axis.grid(alpha=0.3)
    axis.legend()
    _save_and_close(fig, path)


def _plot_states(data, run_label, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, (column, label) in zip(axes.flat, STATE_COLUMNS):
        _plot_band(axis, data["states"][column], label="Mean", color="tab:blue")
        axis.set_title(label)
        axis.set_xlabel("Decision FE")
        axis.set_ylabel("State value")
        axis.grid(alpha=0.3)
    fig.suptitle(f"Frozen-policy pre-action states — {run_label}")
    _save_and_close(fig, path, rect=(0.0, 0.0, 1.0, 0.95))


def _rolling_mean(values, window):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    window = max(1, min(int(window), values.size))
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result = values.copy()
    result[window - 1:] = (cumulative[window:] - cumulative[:-window]) / window
    if window > 1:
        for index in range(window - 1):
            result[index] = np.mean(values[:index + 1])
    return result


def _plot_td3_losses(updates, run_label, path):
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.set_title(f"Training TD3 losses — {run_label}")
    axis.set_xlabel("Cumulative training FE (including resets)")
    axis.set_ylabel("Loss")
    if not updates:
        axis.text(0.5, 0.5, "No TD3 updates", ha="center", va="center", transform=axis.transAxes)
    else:
        window = max(1, min(100, len(updates) // 20))
        x = [int(row["cumulative_training_fe"]) for row in updates]
        critic = _rolling_mean([float(row["critic_loss"]) for row in updates], window)
        axis.plot(x, critic, label=f"Critic loss (rolling mean, w={window})")
        actor = [row for row in updates if row["actor_loss"].strip()]
        if actor:
            actor_window = max(1, min(100, len(actor) // 20))
            actor_x = [int(row["cumulative_training_fe"]) for row in actor]
            actor_y = _rolling_mean([float(row["actor_loss"]) for row in actor], actor_window)
            axis.plot(actor_x, actor_y, label=f"Actor loss (rolling mean, w={actor_window})")
        axis.legend()
    axis.grid(alpha=0.3)
    _save_and_close(fig, path)


def _plot_training_summary(episodes, problem, run_label, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = [int(row["episode_index"]) + 1 for row in episodes]
    gap = [max(float(row["final_best"]) - float(problem["optimum"]), 0.0) for row in episodes]
    series = (
        (gap, "Final gap"),
        ([float(row["return"]) for row in episodes], "Episode return"),
        ([float(row["c_mean"]) for row in episodes], "Mean C"),
    )
    for axis, (values, label) in zip(axes, series):
        axis.plot(x, values, linewidth=1.5)
        axis.set_xlabel("Training episode")
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.grid(alpha=0.3)
    if gap and all(value > 0.0 for value in gap):
        axes[0].set_yscale("log")
    fig.suptitle(f"Training episode summary — {run_label}")
    _save_and_close(fig, path, rect=(0.0, 0.0, 1.0, 0.93))


def plot_run(run_dir, output_dir=None):
    """Generate frozen-evaluation and training diagnostic PNGs."""
    run_path = Path(run_dir).expanduser().resolve()
    output_path = run_path / "figures" if output_dir is None else Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    data = build_plot_data(run_path)
    run_label = _run_label(data["config"], data["problem"])
    paths = {name: (output_path / filename).resolve() for name, filename in FIGURE_FILENAMES.items()}
    try:
        _plot_convergence(data["evaluation"], run_label, paths["convergence"])
        _plot_c_trajectory(data["evaluation"], run_label, paths["c_trajectory"])
        _plot_rewards(data["evaluation"], run_label, paths["rewards"])
        _plot_states(data["evaluation"], run_label, paths["states"])
        _plot_td3_losses(data["training_updates"], run_label, paths["td3_losses"])
        _plot_training_summary(data["training_episodes"], data["problem"], run_label, paths["training_summary"])
    finally:
        plt.close("all")
    return {name: str(path) for name, path in paths.items()}
