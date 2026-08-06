import tempfile
import unittest
from pathlib import Path
import sys
import json
import csv

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

from evaluation.evaluate_td3 import evaluate_td3_policy
from evaluation.plot_run import FIGURE_FILENAMES, build_plot_data, plot_run
from matplotlib import pyplot as plt
from training.run_artifacts import save_classic_td3_run
from training.td3_experiment import (
    ClassicTD3ExperimentConfig,
    run_classic_td3,
)
from training.td3_online import TD3OnlineConfig


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _run_experiment(*, learning_starts, episodes):
    config = ClassicTD3ExperimentConfig(
        problem_name="sphere",
        dimensions=3,
        particles=4,
        max_fe=16,
        buffer_capacity=32,
        device="cpu",
        online=TD3OnlineConfig(
            episodes=episodes,
            learning_starts=learning_starts,
            batch_size=2,
            exploration_noise=0.05,
            updates_per_step=1,
            seed=17,
        ),
    )
    return run_classic_td3(config)


def _save_complete_run(result, run_dir):
    log_paths = save_classic_td3_run(result, run_dir)
    evaluation = evaluate_td3_policy(
        result.policy,
        result.problem,
        particles=4,
        max_fe=16,
        seeds=[1001, 1002],
    )
    (Path(run_dir) / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return log_paths, evaluation


class PlotRunTests(unittest.TestCase):
    def assert_valid_pngs(self, paths):
        self.assertEqual(set(paths), set(FIGURE_FILENAMES))
        for name, filename in FIGURE_FILENAMES.items():
            path = Path(paths[name])
            self.assertTrue(path.is_absolute())
            self.assertEqual(path.name, filename)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, len(PNG_SIGNATURE))
            self.assertEqual(
                path.read_bytes()[:len(PNG_SIGNATURE)],
                PNG_SIGNATURE,
            )

    def test_plot_saved_run_headlessly_without_modifying_logs(self):
        self.assertEqual(matplotlib.get_backend().lower(), "agg")
        result = _run_experiment(learning_starts=2, episodes=2)
        self.assertGreater(result.training_records["total_updates"], 0)

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            run_dir = Path(temp_dir) / "run"
            log_paths, evaluation = _save_complete_run(result, run_dir)
            original_logs = {
                name: Path(path).read_bytes()
                for name, path in log_paths.items()
            }

            figure_paths = plot_run(run_dir)

            self.assert_valid_pngs(figure_paths)
            self.assertEqual(
                original_logs,
                {
                    name: Path(path).read_bytes()
                    for name, path in log_paths.items()
                },
            )
            self.assertEqual(plt.get_fignums(), [])

            plot_data = build_plot_data(run_dir)
            convergence = plot_data["evaluation"]["convergence"]
            self.assertEqual(convergence["fe"], [4, 8, 12, 16])
            expected = []
            for seed in (1001, 1002):
                episode = next(
                    row for row in evaluation["episodes"]
                    if row["seed"] == seed
                )
                records = [
                    row for row in evaluation["steps"]
                    if row["seed"] == seed
                ]
                expected.append(
                    [episode["initial_gap"]]
                    + [row["gap"] for row in records]
                )
            expected = np.asarray(expected, dtype=np.float64)
            np.testing.assert_allclose(
                convergence["mean"],
                np.mean(expected, axis=0),
            )
            np.testing.assert_allclose(
                convergence["q25"],
                np.percentile(expected, 25, axis=0),
            )
            np.testing.assert_allclose(
                convergence["q75"],
                np.percentile(expected, 75, axis=0),
            )
            self.assertNotIn(0, convergence["fe"])
            self.assertEqual(
                plot_data["evaluation"]["c_trajectory"]["fe"],
                [4, 8, 12],
            )
            self.assertEqual(
                plot_data["evaluation"]["rewards"]["fe"],
                [4, 8, 12, 16],
            )
            self.assertEqual(
                [
                    int(row["cumulative_training_fe"])
                    for row in plot_data["training_steps"]
                ],
                [8, 12, 16, 24, 28, 32],
            )

    def test_empty_updates_still_produce_loss_figure(self):
        result = _run_experiment(learning_starts=100, episodes=1)
        self.assertEqual(result.training_records["total_updates"], 0)

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            run_dir = Path(temp_dir) / "run"
            output_dir = Path(temp_dir) / "custom-figures"
            _save_complete_run(result, run_dir)

            figure_paths = plot_run(run_dir, output_dir=output_dir)

            self.assert_valid_pngs(figure_paths)
            self.assertEqual(
                Path(figure_paths["td3_losses"]).parent,
                output_dir.resolve(),
            )
            self.assertEqual(plt.get_fignums(), [])

    def test_old_logs_are_supported_without_fabricating_fe_zero(self):
        result = _run_experiment(learning_starts=2, episodes=2)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            run_dir = Path(temp_dir) / "legacy-run"
            _, evaluation = _save_complete_run(result, run_dir)

            for episode in evaluation["episodes"]:
                episode.pop("initial_fe")
                episode.pop("initial_best")
                episode.pop("initial_gap")
            for step in evaluation["steps"]:
                step.pop("decision_fe")
            (run_dir / "evaluation.json").write_text(
                json.dumps(evaluation, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

            for filename, removed in (
                ("steps.csv", {"decision_fe", "cumulative_training_fe"}),
                ("episodes.csv", {"cumulative_training_fe"}),
                ("updates.csv", {"cumulative_training_fe"}),
            ):
                path = run_dir / filename
                with path.open("r", encoding="utf-8", newline="") as file:
                    rows = list(csv.DictReader(file))
                columns = [key for key in rows[0] if key not in removed] if rows else []
                if not rows:
                    with path.open("r", encoding="utf-8", newline="") as file:
                        columns = [key for key in next(csv.reader(file)) if key not in removed]
                with path.open("w", encoding="utf-8", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(
                        {key: value for key, value in row.items() if key in columns}
                        for row in rows
                    )

            data = build_plot_data(run_dir)
            self.assertEqual(data["evaluation"]["convergence"]["fe"], [8, 12, 16])
            self.assertEqual(data["evaluation"]["c_trajectory"]["fe"], [4, 8, 12])
            self.assertNotIn(0, data["evaluation"]["convergence"]["fe"])
            paths = plot_run(run_dir, output_dir=run_dir / "figures_v2")
            self.assert_valid_pngs(paths)


if __name__ == "__main__":
    unittest.main()
