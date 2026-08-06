import csv
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.td3.td3 import TD3
from evaluation.evaluate_c_baselines import (
    evaluate_fixed_c,
    evaluate_linear_c,
)
from experiments.comparison_artifacts import (
    PER_RUN_COLUMNS,
    SUMMARY_COLUMNS,
)
from experiments.run_cec2017_comparison import (
    CEC2017ComparisonConfig,
    FIXED_C_METHOD,
    LINEAR_C_METHOD,
    TD3_METHOD,
    config_from_args,
    parse_args,
    run_cec2017_comparison,
)
from experiments.td3_pipeline import run_td3_pipeline


REPRESENTATIVES = {
    1: (1, "unimodal"),
    5: (6, "multimodal"),
    12: (13, "hybrid"),
    23: (24, "composition"),
}


def strict_json_load(path):
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


class TestCEC2017ComparisonCLI(unittest.TestCase):
    @staticmethod
    def minimal_args():
        return [
            "--particles",
            "4",
            "--max-fe",
            "16",
            "--training-seeds",
            "101",
            "--evaluation-seeds",
            "201",
            "202",
            "--output-root",
            "output",
            "--run-name",
            "comparison",
        ]

    def test_default_representatives_and_td3_options_are_preserved(self):
        args = parse_args(
            self.minimal_args()
            + [
                "--dimensions",
                "10",
                "--particles",
                "4",
                "--episodes",
                "2",
                "--learning-starts",
                "3",
                "--batch-size",
                "2",
                "--buffer-capacity",
                "32",
                "--exploration-noise",
                "0.07",
                "--updates-per-step",
                "2",
                "--device",
                "cpu",
                "--reward-mode",
                "linear_improvement",
                "--discount",
                "1.0",
            ]
        )
        config = config_from_args(args)

        self.assertEqual(config.function_ids, (1, 5, 12, 23))
        self.assertEqual(config.training_seeds, (101,))
        self.assertEqual(config.evaluation_seeds, (201, 202))
        self.assertEqual(config.particles, 4)
        self.assertEqual(config.episodes, 2)
        self.assertEqual(config.learning_starts, 3)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.buffer_capacity, 32)
        self.assertEqual(config.exploration_noise, 0.07)
        self.assertEqual(config.updates_per_step, 2)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.reward_mode, "linear_improvement")
        self.assertEqual(config.discount, 1.0)

    def test_custom_public_ids_and_invalid_ids(self):
        args = parse_args(
            self.minimal_args()
            + ["--function-ids", "23", "1", "5"]
        )
        self.assertEqual(config_from_args(args).function_ids, (23, 1, 5))

        for function_id in (0, 30):
            with self.subTest(function_id=function_id):
                with self.assertRaises(SystemExit):
                    with redirect_stderr(io.StringIO()):
                        parse_args(
                            self.minimal_args()
                            + ["--function-ids", str(function_id)]
                        )

    def test_training_and_evaluation_seed_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            CEC2017ComparisonConfig(
                function_ids=[1],
                particles=4,
                max_fe=16,
                training_seeds=[100],
                evaluation_seeds=[101],
                episodes=2,
                learning_starts=1,
                batch_size=1,
                buffer_capacity=16,
            )


class TestCEC2017ComparisonPipeline(unittest.TestCase):
    def make_config(self):
        return CEC2017ComparisonConfig(
            dimensions=10,
            particles=4,
            max_fe=16,
            training_seeds=[101],
            evaluation_seeds=[201, 202],
            episodes=1,
            learning_starts=1,
            batch_size=1,
            buffer_capacity=16,
            exploration_noise=0.05,
            updates_per_step=1,
            device="cpu",
            reward_mode="linear_improvement",
            discount=1.0,
        )

    def assert_finite_tree(self, value):
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, (int, float)):
            self.assertTrue(np.isfinite(value))
            return
        if isinstance(value, list):
            for child in value:
                self.assert_finite_tree(child)
            return
        if isinstance(value, dict):
            for child in value.values():
                self.assert_finite_tree(child)
            return
        self.fail(f"unexpected JSON value type: {type(value).__name__}")

    def test_lightweight_four_function_end_to_end_comparison(self):
        config = self.make_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "experiments.run_cec2017_comparison.evaluate_fixed_c",
                wraps=evaluate_fixed_c,
            ) as fixed_evaluator, patch(
                "experiments.run_cec2017_comparison.evaluate_linear_c",
                wraps=evaluate_linear_c,
            ) as linear_evaluator, patch(
                "experiments.run_cec2017_comparison.run_td3_pipeline",
                wraps=run_td3_pipeline,
            ) as td3_pipeline:
                result = run_cec2017_comparison(
                    config,
                    output_root=temp_dir,
                    run_name="cec2017-comparison-smoke",
                )

            self.assertEqual(fixed_evaluator.call_count, 4)
            self.assertEqual(linear_evaluator.call_count, 4)
            self.assertEqual(td3_pipeline.call_count, 4)
            self.assertEqual(result["task_count"], 12)
            self.assertEqual(result["per_run_count"], 24)
            self.assertEqual(result["summary_count"], 12)

            run_dir = Path(result["paths"]["run_dir"])
            self.assertTrue(run_dir.is_absolute())
            for name in (
                "manifest",
                "per_run",
                "summary_csv",
                "summary_json",
            ):
                path = Path(result["paths"][name])
                self.assertTrue(path.is_file(), msg=name)
                self.assertGreater(path.stat().st_size, 0, msg=name)

            manifest = strict_json_load(result["paths"]["manifest"])
            summary_json = strict_json_load(
                result["paths"]["summary_json"]
            )
            self.assertEqual(manifest["execution"], "sequential")
            self.assertEqual(manifest["task_count"], 12)
            self.assertEqual(
                [method["name"] for method in manifest["methods"]],
                [FIXED_C_METHOD, LINEAR_C_METHOD, TD3_METHOD],
            )
            self.assertEqual(manifest["methods"][2]["n_step"], 1)
            self.assertEqual(
                manifest["config"]["function_ids"],
                [1, 5, 12, 23],
            )
            self.assertEqual(
                manifest["config"]["training_seeds"],
                [101],
            )
            self.assertEqual(
                manifest["config"]["evaluation_seeds"],
                [201, 202],
            )

            with Path(result["paths"]["per_run"]).open(
                "r", encoding="utf-8", newline=""
            ) as file:
                reader = csv.DictReader(file)
                self.assertEqual(tuple(reader.fieldnames), PER_RUN_COLUMNS)
                per_run_rows = list(reader)
            with Path(result["paths"]["summary_csv"]).open(
                "r", encoding="utf-8", newline=""
            ) as file:
                reader = csv.DictReader(file)
                self.assertEqual(tuple(reader.fieldnames), SUMMARY_COLUMNS)
                summary_rows = list(reader)

            self.assertEqual(len(per_run_rows), 24)
            self.assertEqual(len(summary_rows), 12)
            self.assertEqual(len(summary_json["records"]), 12)
            self.assertEqual(
                {int(row["problem_id"]) for row in per_run_rows},
                set(REPRESENTATIVES),
            )
            for row in per_run_rows:
                public_id = int(row["problem_id"])
                source_id, category = REPRESENTATIVES[public_id]
                self.assertEqual(int(row["source_function_id"]), source_id)
                self.assertEqual(row["category"], category)
                self.assertIn(
                    row["method"],
                    {FIXED_C_METHOD, LINEAR_C_METHOD, TD3_METHOD},
                )
                if row["method"] == TD3_METHOD:
                    self.assertEqual(row["training_seed"], "101")
                else:
                    self.assertEqual(row["training_seed"], "")
                self.assertIn(int(row["evaluation_seed"]), {201, 202})
                self.assertEqual(int(row["steps"]), 3)
                self.assertEqual(int(row["final_fe"]), 16)
                self.assertTrue(np.isfinite(float(row["final_best"])))
                self.assertTrue(np.isfinite(float(row["gap"])))

            for row in summary_rows:
                self.assertEqual(int(row["runs"]), 2)
                for field in (
                    "mean_gap",
                    "median_gap",
                    "std_gap",
                    "min_gap",
                    "max_gap",
                ):
                    self.assertTrue(np.isfinite(float(row[field])))

            tasks = manifest["tasks"]
            self.assertEqual(
                [task["task_index"] for task in tasks],
                list(range(1, 13)),
            )
            self.assertEqual(
                sum(task["method"] == TD3_METHOD for task in tasks),
                4,
            )
            self.assertEqual(
                sum(task["method"] != TD3_METHOD for task in tasks),
                8,
            )
            checkpoint_paths = []
            for task in tasks:
                self.assertEqual(task["evaluation_seeds"], [201, 202])
                evaluation_path = run_dir / task["evaluation"]
                evaluation = strict_json_load(evaluation_path)
                self.assertEqual(
                    [episode["seed"] for episode in evaluation["episodes"]],
                    [201, 202],
                )
                self.assertEqual(len(evaluation["episodes"]), 2)
                self.assert_finite_tree(evaluation)

                if task["method"] == TD3_METHOD:
                    self.assertEqual(task["training_seed"], 101)
                    self.assertEqual(task["training_episode_seeds"], [101])
                    checkpoint_path = run_dir / task["checkpoint"]
                    self.assertTrue(checkpoint_path.is_file())
                    checkpoint_paths.append(checkpoint_path)
                    restored = TD3(6, 1, 1.0, device="cpu")
                    metadata = restored.load_checkpoint(
                        checkpoint_path,
                        load_optimizers=False,
                    )
                    self.assertEqual(metadata["training_seeds"], [101])
                    self.assertEqual(metadata["evaluation_seeds"], [201, 202])
                else:
                    self.assertIsNone(task["training_seed"])
                    self.assertIsNone(task["checkpoint"])
                    self.assertEqual(
                        list(evaluation["baseline"])[0],
                        "name",
                    )
                    self.assertFalse(
                        (evaluation_path.parent / "checkpoints").exists()
                    )

            self.assertEqual(len(checkpoint_paths), 4)
            self.assertEqual(len(set(checkpoint_paths)), 4)
            self.assert_finite_tree(manifest)
            self.assert_finite_tree(summary_json)
            self.assertIsInstance(
                json.dumps(result, allow_nan=False),
                str,
            )

    def test_nonempty_directory_is_rejected_before_any_task(self):
        config = self.make_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "existing"
            run_dir.mkdir()
            marker = run_dir / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")

            with patch(
                "experiments.run_cec2017_comparison.evaluate_fixed_c"
            ) as fixed_evaluator, patch(
                "experiments.run_cec2017_comparison.run_td3_pipeline"
            ) as td3_pipeline:
                with self.assertRaises(FileExistsError):
                    run_cec2017_comparison(
                        config,
                        output_root=temp_dir,
                        run_name="existing",
                    )

            fixed_evaluator.assert_not_called()
            td3_pipeline.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
