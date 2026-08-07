import csv
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.image as mpimg
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.td3.td3 import TD3
from experiments.run_cec2017_td3 import (
    config_from_args,
    main,
    parse_args,
    problem_from_args,
)
from experiments.td3_pipeline import run_td3_pipeline


REPRESENTATIVES = {
    1: (1, "unimodal"),
    5: (6, "multimodal"),
    12: (13, "hybrid"),
    23: (24, "composition"),
}
PNG_NAMES = {
    "convergence.png",
    "c_trajectory.png",
    "rewards.png",
    "states.png",
    "td3_losses.png",
    "training_summary.png",
}


def strict_json_load(path):
    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


class TestCEC2017TD3CLI(unittest.TestCase):
    @staticmethod
    def minimal_args(function_id):
        return [
            "--function-id",
            str(function_id),
            "--max-fe",
            "16",
            "--output-root",
            "output",
            "--run-name",
            f"f{function_id}",
            "--evaluation-seeds",
            "9001",
        ]

    def test_representative_public_ids_map_to_expected_sources(self):
        for public_id, (source_id, category) in REPRESENTATIVES.items():
            with self.subTest(public_id=public_id):
                args = parse_args(self.minimal_args(public_id))
                problem = problem_from_args(args)
                self.assertEqual(problem.problem_id, public_id)
                self.assertEqual(problem.source_function_id, source_id)
                self.assertEqual(problem.category, category)
                self.assertEqual(problem.dimensions, 10)

        public_f2 = problem_from_args(parse_args(self.minimal_args(2)))
        self.assertEqual(public_f2.problem_id, 2)
        self.assertEqual(public_f2.source_function_id, 3)
        self.assertNotEqual(public_f2.source_function_id, 2)

    def test_cli_builds_requested_generic_config(self):
        args = parse_args(
            [
                "--function-id",
                "23",
                "--dimensions",
                "30",
                "--particles",
                "9",
                "--max-fe",
                "90",
                "--episodes",
                "4",
                "--learning-starts",
                "11",
                "--batch-size",
                "5",
                "--seed",
                "321",
                "--evaluation-seeds",
                "9001",
                "9002",
                "--device",
                "cpu",
                "--buffer-capacity",
                "200",
                "--exploration-noise",
                "0.07",
                "--updates-per-step",
                "3",
                "--c-min",
                "0.2",
                "--c-max",
                "1.2",
                "--recent-window",
                "3",
                "--stagnation-horizon",
                "4",
                "--reward-mode",
                "linear_improvement",
                "--reward-epsilon",
                "1e-9",
                "--state-mode",
                "relative_log_v2",
                "--discount",
                "1.0",
                "--output-root",
                "output",
                "--run-name",
                "cec-f23",
            ]
        )
        config = config_from_args(args)

        self.assertEqual(problem_from_args(args).source_function_id, 24)
        self.assertEqual(config.particles, 9)
        self.assertEqual(config.max_fe, 90)
        self.assertEqual(config.buffer_capacity, 200)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.reward_mode, "linear_improvement")
        self.assertEqual(config.reward_epsilon, 1e-9)
        self.assertEqual(config.state_mode, "relative_log_v2")
        self.assertEqual(config.discount, 1.0)
        self.assertEqual(config.online.episodes, 4)
        self.assertEqual(config.online.learning_starts, 11)
        self.assertEqual(config.online.batch_size, 5)
        self.assertEqual(config.online.seed, 321)

    def test_invalid_public_ids_fail_during_parsing(self):
        for function_id in (0, 30):
            with self.subTest(function_id=function_id):
                with self.assertRaises(SystemExit):
                    with redirect_stderr(io.StringIO()):
                        parse_args(self.minimal_args(function_id))


class TestCEC2017TD3Pipeline(unittest.TestCase):
    def assert_finite_tree(self, value):
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, (int, float)):
            self.assertTrue(np.isfinite(value))
            return
        if isinstance(value, dict):
            for child in value.values():
                self.assert_finite_tree(child)
            return
        if isinstance(value, list):
            for child in value:
                self.assert_finite_tree(child)

    def test_f12_relative_v2_lightweight_complete_cli_pipeline(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            argv = [
                "--function-id",
                "12",
                "--dimensions",
                "10",
                "--particles",
                "4",
                "--max-fe",
                "16",
                "--episodes",
                "1",
                "--learning-starts",
                "1",
                "--batch-size",
                "1",
                "--seed",
                "20260806",
                "--evaluation-seeds",
                "20261806",
                "20261807",
                "--device",
                "cpu",
                "--buffer-capacity",
                "16",
                "--exploration-noise",
                "0.05",
                "--updates-per-step",
                "1",
                "--reward-mode",
                "linear_improvement",
                "--discount",
                "1.0",
                "--state-mode",
                "relative_log_v2",
                "--output-root",
                temp_dir,
                "--run-name",
                "cec2017-f12-v2-smoke",
            ]
            with patch(
                "experiments.run_cec2017_td3.run_td3_pipeline",
                wraps=run_td3_pipeline,
            ) as common_pipeline, patch("builtins.print"):
                pipeline = main(argv)
            common_pipeline.assert_called_once()

            run_dir = Path(temp_dir) / "cec2017-f12-v2-smoke"
            paths = pipeline["paths"]
            required = {
                "config": run_dir / "config.json",
                "problem": run_dir / "problem.json",
                "summary": run_dir / "summary.json",
                "steps": run_dir / "steps.csv",
                "episodes": run_dir / "episodes.csv",
                "updates": run_dir / "updates.csv",
                "evaluation": run_dir / "evaluation.json",
                "checkpoint": run_dir / "checkpoints" / "policy.pt",
            }
            for name, expected in required.items():
                actual = Path(paths[name])
                self.assertTrue(actual.is_absolute(), msg=name)
                self.assertEqual(actual, expected.resolve(), msg=name)
                self.assertTrue(actual.is_file(), msg=name)
                self.assertGreater(actual.stat().st_size, 0, msg=name)

            config = strict_json_load(paths["config"])
            problem = strict_json_load(paths["problem"])
            summary = strict_json_load(paths["summary"])
            evaluation = strict_json_load(paths["evaluation"])
            expected_problem_fields = {
                "problem_id": 12,
                "source_function_id": 13,
                "category": "hybrid",
                "optimum": 1300.0,
            }
            for field, expected in expected_problem_fields.items():
                self.assertEqual(problem[field], expected)
                self.assertEqual(evaluation["problem"][field], expected)
            self.assertEqual(summary["source_function_id"], 13)
            self.assertEqual(summary["category"], "hybrid")
            self.assertEqual(config["dimensions"], 10)
            self.assertEqual(
                config["environment"]["reward_mode"],
                "linear_improvement",
            )
            self.assertEqual(config["td3"]["discount"], 1.0)
            self.assertEqual(
                config["environment"]["state_mode"],
                "relative_log_v2",
            )
            self.assertEqual(summary["state_mode"], "relative_log_v2")
            self.assertEqual(
                pipeline["summary"]["state_mode"],
                "relative_log_v2",
            )
            self.assertEqual(
                evaluation["environment"]["state_mode"],
                "relative_log_v2",
            )

            with Path(paths["steps"]).open(
                "r", encoding="utf-8", newline=""
            ) as file:
                step_rows = list(csv.DictReader(file))
            with Path(paths["episodes"]).open(
                "r", encoding="utf-8", newline=""
            ) as file:
                episode_rows = list(csv.DictReader(file))
            with Path(paths["updates"]).open(
                "r", encoding="utf-8", newline=""
            ) as file:
                update_rows = list(csv.DictReader(file))
            self.assertEqual(len(step_rows), 3)
            self.assertEqual(len(episode_rows), 1)
            self.assertEqual(len(update_rows), 3)
            self.assertEqual(
                episode_rows[0]["state_mode"],
                "relative_log_v2",
            )
            self.assertGreater(
                float(episode_rows[0]["initial_position_scale"]),
                0.0,
            )
            self.assertGreater(
                float(episode_rows[0]["initial_q_scale"]),
                0.0,
            )
            self.assertEqual(
                int(episode_rows[0]["max_episode_updates"]),
                3,
            )

            self.assertEqual(
                [episode["seed"] for episode in evaluation["episodes"]],
                [20261806, 20261807],
            )
            for episode in evaluation["episodes"]:
                self.assertEqual(episode["steps"], 3)
                self.assertEqual(episode["final_fe"], 16)
                self.assertEqual(
                    episode["state_mode"],
                    "relative_log_v2",
                )
                self.assertGreater(episode["initial_position_scale"], 0.0)
                self.assertGreater(episode["initial_q_scale"], 0.0)
                self.assertEqual(episode["max_episode_updates"], 3)
            self.assertEqual(len(evaluation["steps"]), 6)
            self.assert_finite_tree(evaluation)

            restored = TD3(6, 1, 1.0, device="cpu")
            checkpoint_metadata = restored.load_checkpoint(
                paths["checkpoint"],
                load_optimizers=False,
            )
            self.assertEqual(
                checkpoint_metadata["kind"],
                "td3_problem_policy",
            )
            self.assertEqual(checkpoint_metadata["problem"], problem)
            self.assertEqual(
                checkpoint_metadata["experiment_config"],
                config,
            )
            self.assertEqual(
                checkpoint_metadata["experiment_config"]["environment"][
                    "state_mode"
                ],
                "relative_log_v2",
            )
            probe = checkpoint_metadata["actor_probe"]
            np.testing.assert_array_equal(
                restored.select_action(
                    np.asarray(probe["state"], dtype=np.float32)
                ),
                np.asarray(probe["action"]),
            )

            figure_paths = {
                Path(path).name: Path(path)
                for path in paths["figures"].values()
            }
            self.assertEqual(set(figure_paths), PNG_NAMES)
            for name, path in figure_paths.items():
                image = mpimg.imread(path)
                self.assertGreater(image.shape[0], 0, msg=name)
                self.assertGreater(image.shape[1], 0, msg=name)
                self.assertTrue(np.all(np.isfinite(image)), msg=name)

            self.assertIsInstance(
                json.dumps(pipeline, allow_nan=False),
                str,
            )


if __name__ == "__main__":
    unittest.main()
