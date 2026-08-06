from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.run_artifacts import (
    EPISODE_COLUMNS,
    STEP_COLUMNS,
    UPDATE_COLUMNS,
    save_classic_td3_run,
    save_td3_run,
)
from training.td3_experiment import (
    ClassicTD3ExperimentConfig,
    run_classic_td3,
)
from training.td3_online import TD3OnlineConfig


class TestRunArtifacts(unittest.TestCase):
    @staticmethod
    def make_result(learning_starts=5):
        config = ClassicTD3ExperimentConfig(
            problem_name="sphere",
            dimensions=3,
            particles=4,
            max_fe=16,
            buffer_capacity=16,
            device="cpu",
            online=TD3OnlineConfig(
                episodes=2,
                learning_starts=learning_starts,
                batch_size=2,
                exploration_noise=0.05,
                updates_per_step=1,
                seed=123,
            ),
        )
        return run_classic_td3(config)

    def assert_csv_matches_records(self, path, columns, records):
        with Path(path).open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        self.assertEqual(reader.fieldnames, list(columns))
        self.assertEqual(len(rows), len(records))
        for row, record in zip(rows, records):
            for column in columns:
                expected = record[column]
                expected_text = "" if expected is None else str(expected)
                self.assertEqual(row[column], expected_text)

    def test_saves_six_structured_artifacts(self):
        result = self.make_result()

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        ) as temporary_directory:
            run_dir = Path(temporary_directory) / "run"

            with patch(
                "training.run_artifacts.save_td3_run",
                wraps=save_td3_run,
            ) as common_saver:
                paths = save_classic_td3_run(result, run_dir)
            common_saver.assert_called_once_with(result, run_dir)

            self.assertEqual(
                set(paths),
                {
                    "config",
                    "problem",
                    "summary",
                    "steps",
                    "episodes",
                    "updates",
                },
            )
            self.assertEqual(
                {Path(path).name for path in paths.values()},
                {
                    "config.json",
                    "problem.json",
                    "summary.json",
                    "steps.csv",
                    "episodes.csv",
                    "updates.csv",
                },
            )
            for path in paths.values():
                self.assertTrue(Path(path).is_absolute())
                self.assertTrue(Path(path).is_file())

            config = json.loads(
                Path(paths["config"]).read_text(encoding="utf-8")
            )
            problem = json.loads(
                Path(paths["problem"]).read_text(encoding="utf-8")
            )
            summary = json.loads(
                Path(paths["summary"]).read_text(encoding="utf-8")
            )
            self.assertEqual(config, result.config)
            self.assertEqual(problem, result.problem_metadata)
            for name in ("config", "problem", "summary"):
                text = Path(paths[name]).read_text(encoding="utf-8")
                self.assertNotIn("NaN", text)
                self.assertNotIn("Infinity", text)

            records = result.training_records
            self.assertEqual(summary["suite"], "classic")
            self.assertEqual(summary["problem_id"], "sphere")
            self.assertEqual(summary["problem_name"], "Sphere")
            self.assertEqual(summary["dimensions"], 3)
            self.assertEqual(
                summary["reward_mode"],
                "step_log_improvement",
            )
            self.assertEqual(summary["discount"], 0.99)
            self.assertEqual(summary["episode_count"], 2)
            self.assertEqual(summary["total_steps"], 6)
            self.assertEqual(summary["total_updates"], 2)
            self.assertEqual(summary["warmup_steps"], 5)
            self.assertEqual(summary["actor_steps"], 1)
            self.assertEqual(
                summary["last_episode_final_best"],
                records["episodes"][-1]["final_best"],
            )
            self.assertEqual(
                summary["best_episode_final_best"],
                min(
                    episode["final_best"]
                    for episode in records["episodes"]
                ),
            )

            self.assert_csv_matches_records(
                paths["steps"],
                STEP_COLUMNS,
                records["steps"],
            )
            self.assert_csv_matches_records(
                paths["episodes"],
                EPISODE_COLUMNS,
                records["episodes"],
            )
            self.assert_csv_matches_records(
                paths["updates"],
                UPDATE_COLUMNS,
                records["updates"],
            )
            self.assertIn("reward_progress", STEP_COLUMNS)
            self.assertIn("decision_fe", STEP_COLUMNS)
            self.assertIn("cumulative_training_fe", STEP_COLUMNS)
            self.assertIn("cumulative_training_fe", UPDATE_COLUMNS)
            self.assertIn("reward_mode", EPISODE_COLUMNS)
            self.assertIn("initial_improvement_scale", EPISODE_COLUMNS)
            self.assertIn("initial_gap_scale", EPISODE_COLUMNS)

    def test_zero_updates_keeps_complete_header(self):
        result = self.make_result(learning_starts=100)
        self.assertEqual(result.training_records["total_updates"], 0)

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        ) as temporary_directory:
            paths = save_classic_td3_run(
                result,
                Path(temporary_directory) / "zero-updates",
            )

            with Path(paths["updates"]).open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                reader = csv.DictReader(file)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, list(UPDATE_COLUMNS))
            self.assertEqual(rows, [])

    def test_non_empty_directory_is_not_overwritten(self):
        result = self.make_result()

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        ) as temporary_directory:
            run_dir = Path(temporary_directory) / "existing"
            run_dir.mkdir()
            sentinel = run_dir / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                save_classic_td3_run(result, run_dir)

            self.assertEqual(
                sentinel.read_text(encoding="utf-8"),
                "keep",
            )
            self.assertEqual(
                [path.name for path in run_dir.iterdir()],
                ["keep.txt"],
            )


if __name__ == "__main__":
    unittest.main()
