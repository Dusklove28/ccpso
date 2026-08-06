import json
import csv
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.td3.td3 import TD3
from experiments.run_classic_td3 import (
    config_from_args,
    parse_args,
    run_classic_td3_pipeline,
)
from training.td3_experiment import ClassicTD3ExperimentConfig
from training.td3_online import TD3OnlineConfig


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class TestClassicTD3Pipeline(unittest.TestCase):
    @staticmethod
    def make_config():
        return ClassicTD3ExperimentConfig(
            problem_name="sphere",
            dimensions=3,
            particles=4,
            max_fe=16,
            buffer_capacity=16,
            device="cpu",
            online=TD3OnlineConfig(
                episodes=2,
                learning_starts=5,
                batch_size=2,
                exploration_noise=0.05,
                updates_per_step=1,
                seed=123,
            ),
        )

    def test_complete_pipeline_and_checkpoint_reload(self):
        config = self.make_config()
        evaluation_seeds = [1_001, 1_002]

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            pipeline = run_classic_td3_pipeline(
                config,
                output_root=output_root,
                run_name="sphere-smoke",
                evaluation_seeds=evaluation_seeds,
            )

            self.assertIsInstance(
                json.dumps(pipeline, allow_nan=False),
                str,
            )
            paths = pipeline["paths"]
            run_dir = (output_root / "sphere-smoke").resolve()
            self.assertEqual(Path(paths["run_dir"]), run_dir)

            required_files = {
                "config": run_dir / "config.json",
                "problem": run_dir / "problem.json",
                "summary": run_dir / "summary.json",
                "steps": run_dir / "steps.csv",
                "episodes": run_dir / "episodes.csv",
                "updates": run_dir / "updates.csv",
                "evaluation": run_dir / "evaluation.json",
                "checkpoint": run_dir / "checkpoints" / "policy.pt",
            }
            for name, expected_path in required_files.items():
                actual_path = Path(paths[name])
                self.assertTrue(actual_path.is_absolute())
                self.assertEqual(actual_path, expected_path)
                self.assertTrue(actual_path.is_file(), msg=name)
                self.assertGreater(actual_path.stat().st_size, 0, msg=name)

            expected_figures = {
                "convergence": "convergence.png",
                "c_trajectory": "c_trajectory.png",
                "rewards": "rewards.png",
                "states": "states.png",
                "td3_losses": "td3_losses.png",
            }
            self.assertEqual(set(paths["figures"]), set(expected_figures))
            for name, filename in expected_figures.items():
                figure_path = Path(paths["figures"][name])
                self.assertEqual(
                    figure_path,
                    run_dir / "figures" / filename,
                )
                self.assertEqual(
                    figure_path.read_bytes()[:len(PNG_SIGNATURE)],
                    PNG_SIGNATURE,
                )

            evaluation = json.loads(
                Path(paths["evaluation"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                [episode["seed"] for episode in evaluation["episodes"]],
                evaluation_seeds,
            )
            for episode in evaluation["episodes"]:
                self.assertEqual(episode["steps"], 3)
                self.assertEqual(episode["final_fe"], 16)

            summary = pipeline["summary"]
            self.assertEqual(summary["training_seeds"], [123, 124])
            self.assertEqual(summary["evaluation_seeds"], evaluation_seeds)
            self.assertTrue(
                set(summary["training_seeds"]).isdisjoint(
                    summary["evaluation_seeds"]
                )
            )
            self.assertEqual(summary["total_steps"], 6)
            self.assertEqual(summary["total_updates"], 2)
            self.assertEqual(
                summary["reward_mode"],
                "step_log_improvement",
            )
            self.assertEqual(summary["discount"], 0.99)

            restored_policy = TD3(
                state_dim=6,
                action_dim=1,
                max_action=1.0,
                device="cpu",
            )
            checkpoint_metadata = restored_policy.load_checkpoint(
                paths["checkpoint"],
                load_optimizers=False,
            )
            stored_problem = json.loads(
                Path(paths["problem"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                stored_problem,
                {
                    "suite": "classic",
                    "problem_id": "sphere",
                    "name": "Sphere",
                    "dimensions": 3,
                    "lower_bound": [-100.0] * 3,
                    "upper_bound": [100.0] * 3,
                    "optimum": 0.0,
                },
            )
            self.assertEqual(
                checkpoint_metadata["problem"],
                stored_problem,
            )
            self.assertEqual(evaluation["problem"], stored_problem)
            probe = summary["actor_probe"]
            np.testing.assert_array_equal(
                restored_policy.select_action(
                    np.asarray(probe["state"], dtype=np.float32)
                ),
                np.asarray(probe["action"]),
            )
            self.assertEqual(
                checkpoint_metadata["evaluation_seeds"],
                evaluation_seeds,
            )
            stored_config = json.loads(
                Path(paths["config"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                checkpoint_metadata["experiment_config"],
                stored_config,
            )
            self.assertEqual(
                evaluation["environment"],
                stored_config["environment"],
            )

            original_bytes = {
                name: path.read_bytes()
                for name, path in required_files.items()
            }
            with self.assertRaises(FileExistsError):
                run_classic_td3_pipeline(
                    config,
                    output_root=output_root,
                    run_name="sphere-smoke",
                    evaluation_seeds=evaluation_seeds,
                )
            self.assertEqual(
                original_bytes,
                {
                    name: path.read_bytes()
                    for name, path in required_files.items()
                },
            )

    def test_evaluation_seeds_must_be_explicit_and_independent(self):
        config = self.make_config()
        invalid_seed_sets = ([], [123], [124, 1_000], [1.5], [True])

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            for index, evaluation_seeds in enumerate(invalid_seed_sets):
                with self.subTest(evaluation_seeds=evaluation_seeds):
                    run_name = f"invalid-{index}"
                    with self.assertRaises(ValueError):
                        run_classic_td3_pipeline(
                            config,
                            output_root=temp_dir,
                            run_name=run_name,
                            evaluation_seeds=evaluation_seeds,
                        )
                    self.assertFalse(
                        (Path(temp_dir) / run_name).exists()
                    )

    def test_cli_arguments_build_expected_config(self):
        args = parse_args(
            [
                "--problem",
                "rastrigin",
                "--dimensions",
                "7",
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
                "--discount",
                "1.0",
                "--output-root",
                "pipeline-output",
                "--run-name",
                "cli-run",
                "--evaluation-seeds",
                "9001",
                "9002",
            ]
        )
        config = config_from_args(args)

        self.assertEqual(config.problem_name, "rastrigin")
        self.assertEqual(config.dimensions, 7)
        self.assertEqual(config.particles, 9)
        self.assertEqual(config.max_fe, 90)
        self.assertEqual(config.buffer_capacity, 200)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.online.episodes, 4)
        self.assertEqual(config.online.learning_starts, 11)
        self.assertEqual(config.online.batch_size, 5)
        self.assertEqual(config.online.seed, 321)
        self.assertEqual(config.online.exploration_noise, 0.07)
        self.assertEqual(config.online.updates_per_step, 3)
        self.assertEqual(config.c_min, 0.2)
        self.assertEqual(config.c_max, 1.2)
        self.assertEqual(config.recent_window, 3)
        self.assertEqual(config.stagnation_horizon, 4)
        self.assertEqual(config.reward_mode, "linear_improvement")
        self.assertEqual(config.reward_epsilon, 1e-9)
        self.assertEqual(config.discount, 1.0)
        self.assertEqual(args.output_root, Path("pipeline-output"))
        self.assertEqual(args.run_name, "cli-run")
        self.assertEqual(args.evaluation_seeds, [9001, 9002])

    def test_cli_defaults_preserve_previous_experiment_behavior(self):
        args = parse_args([
            "--output-root",
            "pipeline-output",
            "--run-name",
            "default-run",
            "--evaluation-seeds",
            "9001",
        ])
        config = config_from_args(args)

        self.assertEqual(config.c_min, 0.0)
        self.assertEqual(config.c_max, 1.5)
        self.assertEqual(config.recent_window, 5)
        self.assertEqual(config.stagnation_horizon, 10)
        self.assertEqual(config.reward_mode, "step_log_improvement")
        self.assertEqual(config.reward_epsilon, 1e-12)
        self.assertEqual(config.discount, 0.99)

    def test_linear_discount_one_propagates_to_every_artifact(self):
        config = replace(
            self.make_config(),
            c_min=0.2,
            c_max=1.2,
            recent_window=3,
            stagnation_horizon=4,
            reward_mode="linear_improvement",
            reward_epsilon=1e-9,
            discount=1.0,
        )
        expected_environment = {
            "c_min": 0.2,
            "c_max": 1.2,
            "recent_window": 3,
            "stagnation_horizon": 4,
            "reward_mode": "linear_improvement",
            "reward_epsilon": 1e-9,
        }

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            pipeline = run_classic_td3_pipeline(
                config,
                output_root=temp_dir,
                run_name="linear-discount-one",
                evaluation_seeds=[2_001],
            )
            paths = pipeline["paths"]
            stored_config = json.loads(
                Path(paths["config"]).read_text(encoding="utf-8")
            )
            stored_summary = json.loads(
                Path(paths["summary"]).read_text(encoding="utf-8")
            )
            evaluation = json.loads(
                Path(paths["evaluation"]).read_text(encoding="utf-8")
            )
            restored_policy = TD3(6, 1, 1.0, device="cpu")
            metadata = restored_policy.load_checkpoint(
                paths["checkpoint"],
                load_optimizers=False,
            )
            with Path(paths["steps"]).open(
                "r", encoding="utf-8", newline=""
            ) as file:
                step_reader = csv.DictReader(file)
                step_rows = list(step_reader)
            with Path(paths["episodes"]).open(
                "r", encoding="utf-8", newline=""
            ) as file:
                episode_reader = csv.DictReader(file)
                episode_rows = list(episode_reader)

        self.assertEqual(stored_config["environment"], expected_environment)
        self.assertEqual(stored_config["td3"]["discount"], 1.0)
        for duplicate in expected_environment:
            self.assertNotIn(duplicate, stored_config)
        self.assertEqual(stored_summary["reward_mode"], "linear_improvement")
        self.assertEqual(stored_summary["discount"], 1.0)
        self.assertEqual(
            metadata["experiment_config"],
            stored_config,
        )
        self.assertEqual(evaluation["environment"], expected_environment)
        self.assertEqual(
            pipeline["summary"]["reward_mode"],
            "linear_improvement",
        )
        self.assertEqual(pipeline["summary"]["discount"], 1.0)
        self.assertIn("reward_progress", step_reader.fieldnames)
        self.assertIn("reward_mode", episode_reader.fieldnames)
        self.assertIn("initial_improvement_scale", episode_reader.fieldnames)
        self.assertIn("initial_gap_scale", episode_reader.fieldnames)
        self.assertTrue(step_rows)
        self.assertTrue(episode_rows)
        self.assertTrue(all(
            row["reward_mode"] == "linear_improvement"
            for row in episode_rows
        ))
        self.assertTrue(all(
            episode["reward_mode"] == "linear_improvement"
            for episode in evaluation["episodes"]
        ))
        self.assertTrue(all(
            "reward_progress" in step
            for step in evaluation["steps"]
        ))
        self.assertIsInstance(
            json.dumps(
                {
                    "config": stored_config,
                    "summary": stored_summary,
                    "evaluation": evaluation,
                    "pipeline": pipeline,
                    "metadata": metadata,
                },
                allow_nan=False,
            ),
            str,
        )


if __name__ == "__main__":
    unittest.main()
