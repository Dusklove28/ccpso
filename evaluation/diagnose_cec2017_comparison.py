"""Read-only diagnostics for a saved CEC2017 TD3 comparison run."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np


PHASES = ("early", "middle", "late")
CREDIT_HORIZONS = (1, 3, 5, 10)
STATE_FIELDS = (
    "state_fe_progress",
    "state_recent_progress",
    "state_position_diversity",
    "state_q_diversity",
    "state_movement",
    "state_stagnation",
)
CORRELATION_FIELDS = ("c_value", "reward", *STATE_FIELDS)

POLICY_COLUMNS = (
    "problem_id",
    "source_function_id",
    "category",
    "training_seed",
    "evaluation_seeds",
    "evaluation_runs",
    "final_gap_mean",
    "final_gap_median",
    "final_gap_q25",
    "final_gap_q75",
    "final_gap_iqr",
    "return_final_gap_spearman",
    "c_mean",
    "c_q01",
    "c_q25",
    "c_median",
    "c_q75",
    "c_q99",
    "c_min",
    "c_max",
    "c_lower_saturation_rate",
    "c_upper_saturation_rate",
    "action_abs_saturation_rate",
    "boundary_clip_mean",
    "boundary_clip_nonzero_rate",
    "reward_zero_rate",
    "reward_q01",
    "reward_q25",
    "reward_median",
    "reward_q75",
    "reward_q99",
    "reward_abs_median",
    "reward_abs_q95",
)

STATE_STAT_SUFFIXES = (
    "mean",
    "q25",
    "median",
    "q75",
    "lower_saturation_rate",
    "upper_saturation_rate",
)
STATE_PHASE_COLUMNS = (
    "problem_id",
    "source_function_id",
    "category",
    "training_seed",
    "phase",
    "samples",
    *(
        f"{field}_{suffix}"
        for field in STATE_FIELDS
        for suffix in STATE_STAT_SUFFIXES
    ),
)

CREDIT_COLUMNS = (
    "problem_id",
    "source_function_id",
    "category",
    "training_seed",
    "horizon",
    "samples",
    "future_improvement_mean",
    "future_improvement_q25",
    "future_improvement_median",
    "future_improvement_q75",
    *(f"spearman_{field}" for field in CORRELATION_FIELDS),
)

FIGURE_FILENAMES = (
    "state_saturation_heatmap.png",
    "c_action_phase_distribution.png",
    "reward_diagnostics.png",
    "future_improvement_correlations.png",
)


def classify_fe_phase(decision_fe, max_fe):
    """Classify a decision by its true pre-action FE progress."""
    decision_fe = int(decision_fe)
    max_fe = int(max_fe)
    if max_fe <= 0 or decision_fe < 0 or decision_fe > max_fe:
        raise ValueError(
            f"invalid FE values: decision_fe={decision_fe}, max_fe={max_fe}"
        )
    progress = decision_fe / max_fe
    if progress < 0.2:
        return "early"
    if progress < 0.8:
        return "middle"
    return "late"


def summarize_distribution(values):
    """Return finite distribution statistics as Python scalars."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("values contain NaN or Infinity")
    return {
        "mean": float(np.mean(array)),
        "q01": float(np.percentile(array, 1.0)),
        "q25": float(np.percentile(array, 25.0)),
        "median": float(np.median(array)),
        "q75": float(np.percentile(array, 75.0)),
        "q99": float(np.percentile(array, 99.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _average_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_correlation(left, right):
    """Calculate Spearman correlation; return None for a constant sequence."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if (
        left_array.ndim != 1
        or right_array.ndim != 1
        or left_array.size != right_array.size
        or left_array.size < 2
    ):
        raise ValueError("correlation inputs must be equal-length 1-D arrays")
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        raise FloatingPointError("correlation inputs contain NaN or Infinity")
    left_ranks = _average_ranks(left_array)
    right_ranks = _average_ranks(right_array)
    left_centered = left_ranks - np.mean(left_ranks)
    right_centered = right_ranks - np.mean(right_ranks)
    denominator = float(
        np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    )
    if denominator == 0.0:
        return None
    result = float(np.sum(left_centered * right_centered) / denominator)
    if not np.isfinite(result):
        raise FloatingPointError("Spearman correlation is non-finite")
    return result


def calculate_future_improvement_samples(steps, horizon, initial_gap=None):
    """Calculate h-step improvement relative to each action's pre-action gap.

    For action index ``i``, the future gap is the post-action gap at
    ``i + horizon - 1``. If the reset gap is unavailable, the first action is
    excluded because its pre-action gap cannot be reconstructed safely.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    records = list(steps)
    samples = []
    for index, row in enumerate(records):
        if index == 0:
            if initial_gap is None:
                continue
            pre_gap = float(initial_gap)
        else:
            pre_gap = float(records[index - 1]["gap"])
        future_index = index + horizon - 1
        if future_index >= len(records):
            continue
        future_gap = float(records[future_index]["gap"])
        if not np.isfinite(pre_gap) or not np.isfinite(future_gap):
            raise FloatingPointError("gap trajectory contains NaN or Infinity")
        denominator = max(pre_gap, 1e-12)
        improvement = float((pre_gap - future_gap) / denominator)
        if not np.isfinite(improvement):
            raise FloatingPointError("normalized future improvement is non-finite")
        sample = {field: float(row[field]) for field in CORRELATION_FIELDS}
        sample["future_improvement"] = improvement
        samples.append(sample)
    return samples


def _read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value} in {path}")
        ))


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def _write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _decision_fe_rows(steps, particles):
    grouped = {}
    for row in steps:
        grouped.setdefault(int(row["seed"]), []).append(dict(row))
    for records in grouped.values():
        records.sort(key=lambda row: int(row["episode_step"]))
        previous_fe = None
        for row in records:
            post_fe = int(row["fe_count"])
            decision_fe = row.get("decision_fe")
            if decision_fe is None:
                decision_fe = previous_fe if previous_fe is not None else post_fe - particles
            decision_fe = int(decision_fe)
            if decision_fe <= 0 or post_fe <= decision_fe:
                raise ValueError(
                    f"invalid evaluation FE transition {decision_fe} -> {post_fe}"
                )
            row["decision_fe"] = decision_fe
            previous_fe = post_fe
    return grouped


def _read_training_boundary(path):
    values = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            value = float(row["boundary_clip_ratio"])
            if not np.isfinite(value):
                raise FloatingPointError(f"non-finite boundary clip value in {path}")
            values.append(value)
    if not values:
        raise ValueError(f"training steps contain no records: {path}")
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.mean(array > 0.0))


def _state_phase_rows(problem, training_seed, grouped, max_fe):
    phase_values = {
        phase: {field: [] for field in STATE_FIELDS}
        for phase in PHASES
    }
    for records in grouped.values():
        for row in records:
            phase = classify_fe_phase(row["decision_fe"], max_fe)
            for field in STATE_FIELDS:
                phase_values[phase][field].append(float(row[field]))

    rows = []
    for phase in PHASES:
        samples = len(phase_values[phase][STATE_FIELDS[0]])
        if samples == 0:
            raise ValueError(f"phase {phase} contains no evaluation samples")
        output = {
            "problem_id": int(problem["problem_id"]),
            "source_function_id": int(problem["source_function_id"]),
            "category": problem["category"],
            "training_seed": int(training_seed),
            "phase": phase,
            "samples": int(samples),
        }
        for field in STATE_FIELDS:
            values = np.asarray(phase_values[phase][field], dtype=np.float64)
            stats = summarize_distribution(values)
            output.update(
                {
                    f"{field}_mean": stats["mean"],
                    f"{field}_q25": stats["q25"],
                    f"{field}_median": stats["median"],
                    f"{field}_q75": stats["q75"],
                    f"{field}_lower_saturation_rate": float(np.mean(values <= 0.01)),
                    f"{field}_upper_saturation_rate": float(np.mean(values >= 0.99)),
                }
            )
        rows.append(output)
    return rows, phase_values


def _credit_rows(problem, training_seed, grouped, episodes):
    episode_by_seed = {int(row["seed"]): row for row in episodes}
    outputs = []
    for horizon in CREDIT_HORIZONS:
        samples = []
        for seed, records in grouped.items():
            episode = episode_by_seed[seed]
            initial_gap = episode.get("initial_gap")
            samples.extend(
                calculate_future_improvement_samples(
                    records,
                    horizon,
                    initial_gap=initial_gap,
                )
            )
        improvements = [row["future_improvement"] for row in samples]
        stats = summarize_distribution(improvements)
        output = {
            "problem_id": int(problem["problem_id"]),
            "source_function_id": int(problem["source_function_id"]),
            "category": problem["category"],
            "training_seed": int(training_seed),
            "horizon": int(horizon),
            "samples": int(len(samples)),
            "future_improvement_mean": stats["mean"],
            "future_improvement_q25": stats["q25"],
            "future_improvement_median": stats["median"],
            "future_improvement_q75": stats["q75"],
        }
        for field in CORRELATION_FIELDS:
            output[f"spearman_{field}"] = spearman_correlation(
                [row[field] for row in samples],
                improvements,
            )
        outputs.append(output)
    return outputs


def _policy_row(task, evaluation, grouped, environment, boundary_stats):
    episodes = evaluation["episodes"]
    evaluation_seeds = [int(row["seed"]) for row in episodes]
    gaps = np.asarray([float(row["gap"]) for row in episodes], dtype=np.float64)
    returns = np.asarray([float(row["return"]) for row in episodes], dtype=np.float64)
    all_steps = [row for records in grouped.values() for row in records]
    c_values = np.asarray([float(row["c_value"]) for row in all_steps], dtype=np.float64)
    actions = np.asarray([float(row["raw_action"]) for row in all_steps], dtype=np.float64)
    rewards = np.asarray([float(row["reward"]) for row in all_steps], dtype=np.float64)
    for name, values in (("gaps", gaps), ("returns", returns), ("C", c_values), ("actions", actions), ("rewards", rewards)):
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{name} contain NaN or Infinity")

    gap_stats = summarize_distribution(gaps)
    c_stats = summarize_distribution(c_values)
    reward_stats = summarize_distribution(rewards)
    c_min = float(environment["c_min"])
    c_max = float(environment["c_max"])
    if c_max == c_min:
        normalized_c = np.full(c_values.shape, 0.5)
    else:
        normalized_c = (c_values - c_min) / (c_max - c_min)
    reward_abs = np.abs(rewards)
    return {
        "problem_id": int(task["problem_id"]),
        "source_function_id": int(task["source_function_id"]),
        "category": task["category"],
        "training_seed": int(task["training_seed"]),
        "evaluation_seeds": ";".join(str(seed) for seed in evaluation_seeds),
        "evaluation_runs": int(len(episodes)),
        "final_gap_mean": gap_stats["mean"],
        "final_gap_median": gap_stats["median"],
        "final_gap_q25": gap_stats["q25"],
        "final_gap_q75": gap_stats["q75"],
        "final_gap_iqr": float(gap_stats["q75"] - gap_stats["q25"]),
        "return_final_gap_spearman": spearman_correlation(returns, gaps),
        "c_mean": c_stats["mean"],
        "c_q01": c_stats["q01"],
        "c_q25": c_stats["q25"],
        "c_median": c_stats["median"],
        "c_q75": c_stats["q75"],
        "c_q99": c_stats["q99"],
        "c_min": c_stats["min"],
        "c_max": c_stats["max"],
        "c_lower_saturation_rate": float(np.mean(normalized_c <= 0.01)),
        "c_upper_saturation_rate": float(np.mean(normalized_c >= 0.99)),
        "action_abs_saturation_rate": float(np.mean(np.abs(actions) >= 0.99)),
        "boundary_clip_mean": boundary_stats[0],
        "boundary_clip_nonzero_rate": boundary_stats[1],
        "reward_zero_rate": float(np.mean(rewards == 0.0)),
        "reward_q01": reward_stats["q01"],
        "reward_q25": reward_stats["q25"],
        "reward_median": reward_stats["median"],
        "reward_q75": reward_stats["q75"],
        "reward_q99": reward_stats["q99"],
        "reward_abs_median": float(np.median(reward_abs)),
        "reward_abs_q95": float(np.percentile(reward_abs, 95.0)),
    }


def _policy_label(problem_id, training_seed):
    return f"F{int(problem_id)} / {int(training_seed)}"


def _save_figure(fig, path, *, rect=None):
    try:
        fig.tight_layout(rect=rect)
        fig.savefig(path, dpi=150, format="png")
    finally:
        plt.close(fig)


def _plot_state_saturation(rows, path):
    ordered = sorted(
        rows,
        key=lambda row: (int(row["problem_id"]), float(row["final_gap_order"]), PHASES.index(row["phase"])),
    )
    labels = [f'{_policy_label(row["problem_id"], row["training_seed"])} {row["phase"][0].upper()}' for row in ordered]
    columns = []
    values = []
    for field in STATE_FIELDS:
        short = field.removeprefix("state_")
        columns.extend((f"{short} <=.01", f"{short} >=.99"))
    for row in ordered:
        values.append(
            [
                value
                for field in STATE_FIELDS
                for value in (
                    row[f"{field}_lower_saturation_rate"],
                    row[f"{field}_upper_saturation_rate"],
                )
            ]
        )
    fig, axis = plt.subplots(figsize=(15, max(8, len(ordered) * 0.28)))
    image = axis.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap="magma")
    axis.set_xticks(range(len(columns)), labels=columns, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_title("Frozen-policy state saturation by true-FE phase")
    fig.colorbar(image, ax=axis, label="Saturation rate")
    _save_figure(fig, path)


def _plot_c_action_distributions(payloads, policy_order, path):
    function_ids = sorted({key[0] for key in payloads})
    fig, axes = plt.subplots(len(function_ids), 2, figsize=(15, 3.8 * len(function_ids)), squeeze=False)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for row_index, function_id in enumerate(function_ids):
        policies = sorted(
            [key for key in payloads if key[0] == function_id],
            key=lambda key: policy_order[key],
        )
        for column_index, field in enumerate(("c_value", "raw_action")):
            axis = axes[row_index, column_index]
            datasets = []
            positions = []
            labels = []
            box_colors = []
            position = 1
            for policy_index, key in enumerate(policies):
                for phase in PHASES:
                    datasets.append(payloads[key][phase][field])
                    positions.append(position)
                    labels.append(f"{key[1]}\n{phase[0].upper()}")
                    box_colors.append(colors[policy_index % len(colors)])
                    position += 1
                position += 0.6
            boxes = axis.boxplot(
                datasets,
                positions=positions,
                widths=0.65,
                showfliers=False,
                patch_artist=True,
            )
            for box, color in zip(boxes["boxes"], box_colors):
                box.set_facecolor(color)
                box.set_alpha(0.55)
            axis.set_xticks(positions, labels=labels, rotation=45, ha="right", fontsize=8)
            axis.set_ylabel("C" if field == "c_value" else "Raw action")
            axis.set_title(f"F{function_id}: {'C' if field == 'c_value' else 'action'} by FE phase")
            axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def _plot_reward_diagnostics(policy_rows, path):
    ordered = sorted(policy_rows, key=lambda row: (row["problem_id"], row["final_gap_mean"]))
    labels = [_policy_label(row["problem_id"], row["training_seed"]) for row in ordered]
    x = np.arange(len(ordered))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].bar(x, [row["reward_zero_rate"] for row in ordered], color="tab:gray")
    axes[0].set_ylabel("Exact-zero reward rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot(x, [row["reward_abs_median"] for row in ordered], label="median |reward|")
    axes[1].plot(x, [row["reward_abs_q95"] for row in ordered], label="Q95 |reward|")
    axes[1].set_yscale("symlog", linthresh=1e-12)
    axes[1].set_ylabel("Reward magnitude")
    axes[1].set_xticks(x, labels=labels, rotation=45, ha="right")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("Frozen-policy reward sparsity and scale")
    _save_figure(fig, path, rect=(0.0, 0.0, 1.0, 0.96))


def _plot_credit_correlations(rows, path):
    ordered = sorted(
        rows,
        key=lambda row: (row["problem_id"], row["final_gap_order"], row["horizon"]),
    )
    labels = [f'{_policy_label(row["problem_id"], row["training_seed"])} h={row["horizon"]}' for row in ordered]
    matrix = np.asarray(
        [
            [np.nan if row[f"spearman_{field}"] is None else row[f"spearman_{field}"] for field in CORRELATION_FIELDS]
            for row in ordered
        ],
        dtype=np.float64,
    )
    fig, axis = plt.subplots(figsize=(13, max(9, len(ordered) * 0.25)))
    image = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(range(len(CORRELATION_FIELDS)), labels=[field.removeprefix("state_") for field in CORRELATION_FIELDS], rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels, fontsize=7)
    axis.set_title("Spearman correlation with normalized future improvement")
    fig.colorbar(image, ax=axis, label="Spearman rho")
    _save_figure(fig, path)


def _plot_final_gap_return(function_id, policies, path):
    fig, axis = plt.subplots(figsize=(8, 5.5))
    for policy in sorted(policies, key=lambda item: item["final_gap_mean"]):
        axis.scatter(
            policy["episode_gaps"],
            policy["episode_returns"],
            s=34,
            alpha=0.8,
            label=f'seed {policy["training_seed"]}',
        )
    if all(gap > 0.0 for policy in policies for gap in policy["episode_gaps"]):
        axis.set_xscale("log")
    axis.set_xlabel("Frozen-evaluation final gap")
    axis.set_ylabel("Frozen-evaluation episode return")
    axis.set_title(f"CEC2017 F{function_id}: final gap vs return")
    axis.grid(alpha=0.25)
    axis.legend(title="Training seed (ordered by mean gap)")
    _save_figure(fig, path)


def diagnose_comparison(comparison_dir, output_dir=None):
    """Generate read-only offline diagnostics from a saved comparison run."""
    comparison_path = Path(comparison_dir).expanduser().resolve()
    manifest = _read_json(comparison_path / "manifest.json")
    if manifest.get("suite") != "cec2017":
        raise ValueError("comparison manifest suite must be 'cec2017'")
    config = manifest["config"]
    max_fe = int(config["max_fe"])
    particles = int(config["particles"])
    td3_tasks = [
        task
        for task in manifest["tasks"]
        if task.get("method") == "td3_n1" and task.get("training_seed") is not None
    ]
    if not td3_tasks:
        raise ValueError("comparison manifest contains no TD3 policy tasks")

    output_path = (
        comparison_path / "diagnostics_v1"
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError(f"diagnostics output is not empty: {output_path}")

    policy_rows = []
    state_rows = []
    credit_rows = []
    distribution_payloads = {}
    relationship_payloads = {}

    for task in td3_tasks:
        evaluation_path = comparison_path / task["evaluation"]
        evaluation = _read_json(evaluation_path)
        problem = evaluation["problem"]
        expected_seeds = [int(seed) for seed in task["evaluation_seeds"]]
        actual_seeds = [int(row["seed"]) for row in evaluation["episodes"]]
        if actual_seeds != expected_seeds:
            raise ValueError(
                f"evaluation seed order mismatch for {evaluation_path}: "
                f"{actual_seeds} != {expected_seeds}"
            )
        grouped = _decision_fe_rows(evaluation["steps"], particles)
        boundary_stats = _read_training_boundary(evaluation_path.parent / "steps.csv")
        policy = _policy_row(
            task,
            evaluation,
            grouped,
            evaluation["environment"],
            boundary_stats,
        )
        policy_rows.append(policy)

        per_phase_rows, phase_values = _state_phase_rows(
            problem,
            task["training_seed"],
            grouped,
            max_fe,
        )
        state_rows.extend(per_phase_rows)
        credit_rows.extend(
            _credit_rows(
                problem,
                task["training_seed"],
                grouped,
                evaluation["episodes"],
            )
        )

        key = (int(task["problem_id"]), int(task["training_seed"]))
        distribution_payloads[key] = {
            phase: {
                "c_value": [
                    float(row["c_value"])
                    for records in grouped.values()
                    for row in records
                    if classify_fe_phase(row["decision_fe"], max_fe) == phase
                ],
                "raw_action": [
                    float(row["raw_action"])
                    for records in grouped.values()
                    for row in records
                    if classify_fe_phase(row["decision_fe"], max_fe) == phase
                ],
            }
            for phase in PHASES
        }
        relationship_payloads.setdefault(int(task["problem_id"]), []).append(
            {
                "training_seed": int(task["training_seed"]),
                "final_gap_mean": policy["final_gap_mean"],
                "episode_gaps": [float(row["gap"]) for row in evaluation["episodes"]],
                "episode_returns": [float(row["return"]) for row in evaluation["episodes"]],
            }
        )

    policy_rows.sort(key=lambda row: (row["problem_id"], row["final_gap_mean"]))
    gap_order = {
        (row["problem_id"], row["training_seed"]): index
        for index, row in enumerate(policy_rows)
    }
    for row in state_rows:
        row["final_gap_order"] = gap_order[(row["problem_id"], row["training_seed"])]
    for row in credit_rows:
        row["final_gap_order"] = gap_order[(row["problem_id"], row["training_seed"])]

    output_path.mkdir(parents=True, exist_ok=True)
    policy_path = output_path / "policy_summary.csv"
    state_path = output_path / "state_phase_summary.csv"
    credit_path = output_path / "credit_horizon.csv"
    _write_csv(policy_path, POLICY_COLUMNS, policy_rows)
    _write_csv(
        state_path,
        STATE_PHASE_COLUMNS,
        [{key: row[key] for key in STATE_PHASE_COLUMNS} for row in state_rows],
    )
    _write_csv(
        credit_path,
        CREDIT_COLUMNS,
        [{key: row[key] for key in CREDIT_COLUMNS} for row in credit_rows],
    )

    figure_paths = {
        "state_saturation": output_path / FIGURE_FILENAMES[0],
        "c_action_phase": output_path / FIGURE_FILENAMES[1],
        "reward_diagnostics": output_path / FIGURE_FILENAMES[2],
        "credit_correlations": output_path / FIGURE_FILENAMES[3],
    }
    _plot_state_saturation(state_rows, figure_paths["state_saturation"])
    _plot_c_action_distributions(
        distribution_payloads,
        gap_order,
        figure_paths["c_action_phase"],
    )
    _plot_reward_diagnostics(policy_rows, figure_paths["reward_diagnostics"])
    _plot_credit_correlations(credit_rows, figure_paths["credit_correlations"])
    relationship_paths = {}
    for function_id, policies in sorted(relationship_payloads.items()):
        path = output_path / f"final_gap_return_f{function_id:02d}.png"
        _plot_final_gap_return(function_id, policies, path)
        relationship_paths[str(function_id)] = path

    summary = {
        "format_version": 1,
        "source_comparison_dir": str(comparison_path),
        "policy_count": int(len(policy_rows)),
        "function_ids": sorted({row["problem_id"] for row in policy_rows}),
        "evaluation_seeds": [int(seed) for seed in config["evaluation_seeds"]],
        "fe_phase_definition": {
            "early": "decision_fe / max_fe < 0.2",
            "middle": "0.2 <= decision_fe / max_fe < 0.8",
            "late": "decision_fe / max_fe >= 0.8",
        },
        "credit_horizons": list(CREDIT_HORIZONS),
        "credit_definition": "(pre_action_gap - gap_after_h_actions) / max(pre_action_gap, 1e-12)",
        "legacy_reset_gap_handling": "exclude first action when initial_gap is unavailable",
        "boundary_clip_source": "training steps.csv; unavailable in legacy frozen evaluation.json",
        "policy_summary": policy_rows,
        "state_phase_summary": [
            {key: row[key] for key in STATE_PHASE_COLUMNS}
            for row in state_rows
        ],
        "credit_horizon": [
            {key: row[key] for key in CREDIT_COLUMNS}
            for row in credit_rows
        ],
    }
    summary_path = output_path / "summary.json"
    _write_json(summary_path, summary)
    return {
        "output_dir": str(output_path),
        "policy_summary": str(policy_path.resolve()),
        "state_phase_summary": str(state_path.resolve()),
        "credit_horizon": str(credit_path.resolve()),
        "summary": str(summary_path.resolve()),
        "figures": {
            **{key: str(path.resolve()) for key, path in figure_paths.items()},
            "final_gap_return": {
                key: str(path.resolve()) for key, path in relationship_paths.items()
            },
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Offline diagnostics for a saved CEC2017 TD3 comparison",
    )
    parser.add_argument("comparison_dir")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = diagnose_comparison(args.comparison_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
