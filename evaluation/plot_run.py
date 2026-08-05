"""Plot the records of one saved classic TD3-CCPSO run."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt


FIGURE_FILENAMES = {
    "convergence": "convergence.png",
    "c_trajectory": "c_trajectory.png",
    "rewards": "rewards.png",
    "states": "states.png",
    "td3_losses": "td3_losses.png",
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
    return (
        f'{problem["name"]} '
        f'(D={int(problem["dimensions"])}, seed={seed})'
    )


def _save_and_close(fig, path, *, rect=None):
    try:
        if rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=rect)
        fig.savefig(path, dpi=150, format="png")
    finally:
        plt.close(fig)


def _plot_convergence(steps, run_label, path):
    fig, axis = plt.subplots(figsize=(8, 5))
    episode_ids = sorted(
        {int(row["episode_index"]) for row in steps}
    )
    for episode_id in episode_ids:
        records = [
            row
            for row in steps
            if int(row["episode_index"]) == episode_id
        ]
        axis.plot(
            [int(row["fe_count"]) for row in records],
            [float(row["gap"]) for row in records],
            marker="o",
            markersize=3,
            label=(
                f"Episode {episode_id} "
                f'(seed={records[0]["episode_seed"]})'
            ),
        )
    axis.set_title(f"Convergence — {run_label}")
    axis.set_xlabel("Function evaluations (FE)")
    axis.set_ylabel("Optimality gap")
    axis.grid(alpha=0.3)
    if episode_ids:
        axis.legend()
    _save_and_close(fig, path)


def _plot_c_trajectory(steps, run_label, path):
    fig, axis = plt.subplots(figsize=(8, 5))
    styles = {
        "warmup": {"color": "tab:orange", "marker": "o"},
        "actor": {"color": "tab:blue", "marker": "x"},
    }
    for action_source, style in styles.items():
        records = [
            row
            for row in steps
            if row["action_source"] == action_source
        ]
        if records:
            axis.scatter(
                [int(row["global_step"]) for row in records],
                [float(row["c_value"]) for row in records],
                label=action_source,
                s=25,
                **style,
            )
    axis.set_title(f"C trajectory — {run_label}")
    axis.set_xlabel("Global step")
    axis.set_ylabel("C value")
    axis.grid(alpha=0.3)
    if steps:
        axis.legend(title="Action source")
    _save_and_close(fig, path)


def _plot_rewards(steps, run_label, path):
    fig, axis = plt.subplots(figsize=(8, 5))
    global_steps = [int(row["global_step"]) for row in steps]
    axis.plot(
        global_steps,
        [float(row["reward"]) for row in steps],
        label="Step reward",
        linewidth=1.5,
    )
    axis.plot(
        global_steps,
        [float(row["episode_return"]) for row in steps],
        label="Episode cumulative return",
        linewidth=1.5,
    )
    axis.set_title(f"Rewards — {run_label}")
    axis.set_xlabel("Global step")
    axis.set_ylabel("Reward / return")
    axis.grid(alpha=0.3)
    if steps:
        axis.legend()
    _save_and_close(fig, path)


def _plot_states(steps, run_label, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    global_steps = [int(row["global_step"]) for row in steps]
    for axis, (column, label) in zip(axes.flat, STATE_COLUMNS):
        axis.plot(
            global_steps,
            [float(row[column]) for row in steps],
            linewidth=1.5,
        )
        axis.set_title(label)
        axis.set_xlabel("Global step")
        axis.set_ylabel("State value")
        axis.grid(alpha=0.3)
    fig.suptitle(f"Pre-action states — {run_label}")
    _save_and_close(fig, path, rect=(0.0, 0.0, 1.0, 0.95))


def _add_total_it_axis(axis, updates):
    by_global_step = {}
    for row in updates:
        global_step = int(row["global_step"])
        by_global_step.setdefault(global_step, []).append(
            int(row["total_it"])
        )

    tick_steps = sorted(by_global_step)
    if len(tick_steps) > 8:
        last_index = len(tick_steps) - 1
        indices = {
            round(index * last_index / 7)
            for index in range(8)
        }
        tick_steps = [tick_steps[index] for index in sorted(indices)]
    top_axis = axis.twiny()
    top_axis.set_xlim(axis.get_xlim())
    top_axis.set_xticks(tick_steps)
    top_axis.set_xticklabels(
        [
            ",".join(str(value) for value in by_global_step[step])
            for step in tick_steps
        ]
    )
    top_axis.set_xlabel("TD3 total_it (aligned to global step)")


def _plot_td3_losses(updates, run_label, path):
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.set_title(f"TD3 losses — {run_label}")
    axis.set_xlabel("Global step")
    axis.set_ylabel("Loss")

    if not updates:
        axis.text(
            0.5,
            0.5,
            "No TD3 updates",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=12,
        )
    else:
        global_steps = [int(row["global_step"]) for row in updates]
        axis.plot(
            global_steps,
            [float(row["critic_loss"]) for row in updates],
            marker="o",
            markersize=3,
            label="Critic loss",
        )
        actor_records = [
            row
            for row in updates
            if row["actor_loss"].strip()
        ]
        if actor_records:
            axis.plot(
                [int(row["global_step"]) for row in actor_records],
                [float(row["actor_loss"]) for row in actor_records],
                marker="x",
                markersize=4,
                label="Actor loss",
            )
        axis.legend()
        _add_total_it_axis(axis, updates)

    axis.grid(alpha=0.3)
    _save_and_close(fig, path)


def plot_run(run_dir, output_dir=None):
    """Generate five diagnostic PNGs from one saved experiment run."""
    run_path = Path(run_dir).expanduser().resolve()
    output_path = (
        run_path / "figures"
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    output_path.mkdir(parents=True, exist_ok=True)

    config = _read_json(run_path / "config.json")
    problem = _read_json(run_path / "problem.json")
    _read_json(run_path / "summary.json")
    steps = _read_csv(run_path / "steps.csv")
    _read_csv(run_path / "episodes.csv")
    updates = _read_csv(run_path / "updates.csv")
    run_label = _run_label(config, problem)

    paths = {
        name: (output_path / filename).resolve()
        for name, filename in FIGURE_FILENAMES.items()
    }

    try:
        _plot_convergence(steps, run_label, paths["convergence"])
        _plot_c_trajectory(steps, run_label, paths["c_trajectory"])
        _plot_rewards(steps, run_label, paths["rewards"])
        _plot_states(steps, run_label, paths["states"])
        _plot_td3_losses(updates, run_label, paths["td3_losses"])
    finally:
        plt.close("all")

    return {name: str(path) for name, path in paths.items()}
