import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.diagnose_cec2017_comparison import (
    CREDIT_COLUMNS,
    POLICY_COLUMNS,
    STATE_PHASE_COLUMNS,
    calculate_future_improvement_samples,
    classify_fe_phase,
    diagnose_comparison,
    spearman_correlation,
    summarize_distribution,
)


class DiagnosticMathTests(unittest.TestCase):
    def test_fe_phase_boundaries_are_exact(self):
        self.assertEqual(classify_fe_phase(0, 100), "early")
        self.assertEqual(classify_fe_phase(19, 100), "early")
        self.assertEqual(classify_fe_phase(20, 100), "middle")
        self.assertEqual(classify_fe_phase(79, 100), "middle")
        self.assertEqual(classify_fe_phase(80, 100), "late")
        self.assertEqual(classify_fe_phase(100, 100), "late")

    def test_distribution_quantiles_and_spearman(self):
        stats = summarize_distribution([0.0, 1.0, 2.0, 3.0])
        self.assertEqual(stats["mean"], 1.5)
        self.assertEqual(stats["q25"], 0.75)
        self.assertEqual(stats["median"], 1.5)
        self.assertEqual(stats["q75"], 2.25)
        self.assertEqual(
            spearman_correlation([1, 2, 3, 4], [8, 6, 4, 2]),
            -1.0,
        )
        self.assertIsNone(spearman_correlation([1, 1, 1], [1, 2, 3]))

    def test_future_improvement_uses_pre_action_gap_and_full_windows(self):
        steps = []
        for index, gap in enumerate((80.0, 60.0, 50.0, 40.0)):
            row = {
                "gap": gap,
                "c_value": float(index),
                "reward": float(index + 1),
            }
            row.update({field: float(index) / 10.0 for field in (
                "state_fe_progress",
                "state_recent_progress",
                "state_position_diversity",
                "state_q_diversity",
                "state_movement",
                "state_stagnation",
            )})
            steps.append(row)

        one_step = calculate_future_improvement_samples(
            steps,
            1,
            initial_gap=100.0,
        )
        np.testing.assert_allclose(
            [row["future_improvement"] for row in one_step],
            [0.2, 0.25, 1.0 / 6.0, 0.2],
        )
        three_step = calculate_future_improvement_samples(
            steps,
            3,
            initial_gap=100.0,
        )
        np.testing.assert_allclose(
            [row["future_improvement"] for row in three_step],
            [0.5, 0.5],
        )
        legacy = calculate_future_improvement_samples(steps, 3)
        self.assertEqual(len(legacy), 1)
        self.assertAlmostEqual(legacy[0]["future_improvement"], 0.5)


class DiagnosticArtifactTests(unittest.TestCase):
    @staticmethod
    def _write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _make_evaluation(training_seed):
        decisions = (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95)
        episodes = []
        steps = []
        for seed_index, seed in enumerate((9001, 9002)):
            start_gap = 120.0 + seed_index * 20.0 + training_seed % 10
            gaps = [start_gap - (index + 1) * (2.0 + seed_index) for index in range(12)]
            episode_return = 0.0
            for index, (decision_fe, gap) in enumerate(zip(decisions, gaps)):
                reward = 0.0 if index % 2 == 0 else float(index + 1) / 100.0
                episode_return += reward
                phase_value = 0.0 if decision_fe < 20 else (1.0 if decision_fe >= 80 else 0.5)
                steps.append(
                    {
                        "seed": seed,
                        "episode_step": index + 1,
                        "decision_fe": decision_fe,
                        "fe_count": decision_fe + 1,
                        "gbest_fitness": 1300.0 + gap,
                        "gap": gap,
                        "raw_action": (-1.0, 0.0, 1.0)[index % 3],
                        "c_value": (0.0, 0.75, 1.5)[index % 3],
                        "reward": reward,
                        "reward_progress": reward,
                        "state_fe_progress": decision_fe / 100.0,
                        "state_recent_progress": phase_value,
                        "state_position_diversity": index / 11.0,
                        "state_q_diversity": 1.0 - index / 11.0,
                        "state_movement": 0.5,
                        "state_stagnation": 0.0,
                    }
                )
            episodes.append(
                {
                    "seed": seed,
                    "initial_fe": 1,
                    "initial_best": 1300.0 + start_gap,
                    "initial_gap": start_gap,
                    "final_best": 1300.0 + gaps[-1],
                    "gap": gaps[-1],
                    "steps": 12,
                    "final_fe": 100,
                    "return": episode_return,
                    "reward_mode": "linear_improvement",
                    "initial_improvement_scale": 10.0,
                    "initial_gap_scale": start_gap,
                    "c_mean": 0.75,
                    "c_min": 0.0,
                    "c_max": 1.5,
                }
            )
        return {
            "problem": {
                "suite": "cec2017",
                "problem_id": 12,
                "name": "CEC2017 F12 (source F13)",
                "dimensions": 10,
                "lower_bound": [-100.0] * 10,
                "upper_bound": [100.0] * 10,
                "optimum": 1300.0,
                "source_function_id": 13,
                "category": "hybrid",
            },
            "environment": {
                "c_min": 0.0,
                "c_max": 1.5,
                "recent_window": 5,
                "stagnation_horizon": 10,
                "reward_mode": "linear_improvement",
                "reward_epsilon": 1e-12,
            },
            "episodes": episodes,
            "steps": steps,
            "final_gap_statistics": {},
        }

    def _make_comparison(self, root):
        tasks = []
        for task_index, training_seed in enumerate((101, 102, 103), start=1):
            relative = Path("td3") / "f12" / f"seed_{training_seed}" / "evaluation.json"
            evaluation = self._make_evaluation(training_seed)
            self._write_json(root / relative, evaluation)
            steps_path = (root / relative).parent / "steps.csv"
            with steps_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=("boundary_clip_ratio",),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(
                    {"boundary_clip_ratio": value}
                    for value in (0.0, 0.25, 0.0, 0.75)
                )
            tasks.append(
                {
                    "task_index": task_index,
                    "problem_id": 12,
                    "source_function_id": 13,
                    "category": "hybrid",
                    "method": "td3_n1",
                    "training_seed": training_seed,
                    "evaluation_seeds": [9001, 9002],
                    "evaluation": relative.as_posix(),
                }
            )
        manifest = {
            "format_version": 1,
            "suite": "cec2017",
            "config": {
                "function_ids": [12],
                "dimensions": 10,
                "particles": 1,
                "max_fe": 100,
                "training_seeds": [101, 102, 103],
                "evaluation_seeds": [9001, 9002],
            },
            "tasks": tasks,
        }
        self._write_json(root / "manifest.json", manifest)

    def test_diagnostics_are_numerically_correct_and_strict(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            comparison = Path(temp_dir) / "comparison"
            comparison.mkdir()
            self._make_comparison(comparison)

            result = diagnose_comparison(comparison)
            output = comparison / "diagnostics_v1"
            self.assertEqual(Path(result["output_dir"]), output.resolve())

            with Path(result["policy_summary"]).open(
                "r", encoding="utf-8", newline=""
            ) as file:
                policy_reader = csv.DictReader(file)
                policies = list(policy_reader)
            self.assertEqual(policy_reader.fieldnames, list(POLICY_COLUMNS))
            self.assertEqual(len(policies), 3)
            self.assertTrue(all(row["evaluation_seeds"] == "9001;9002" for row in policies))
            self.assertTrue(all(float(row["c_lower_saturation_rate"]) == 1.0 / 3.0 for row in policies))
            self.assertTrue(all(float(row["c_upper_saturation_rate"]) == 1.0 / 3.0 for row in policies))
            self.assertTrue(all(float(row["action_abs_saturation_rate"]) == 2.0 / 3.0 for row in policies))
            self.assertTrue(all(float(row["boundary_clip_mean"]) == 0.25 for row in policies))
            self.assertTrue(all(float(row["boundary_clip_nonzero_rate"]) == 0.5 for row in policies))
            self.assertTrue(all(float(row["reward_zero_rate"]) == 0.5 for row in policies))

            with Path(result["state_phase_summary"]).open(
                "r", encoding="utf-8", newline=""
            ) as file:
                state_reader = csv.DictReader(file)
                states = list(state_reader)
            self.assertEqual(state_reader.fieldnames, list(STATE_PHASE_COLUMNS))
            self.assertEqual(len(states), 9)
            early = next(row for row in states if row["training_seed"] == "101" and row["phase"] == "early")
            self.assertEqual(int(early["samples"]), 6)
            self.assertAlmostEqual(float(early["state_fe_progress_mean"]), 0.1)
            self.assertEqual(float(early["state_recent_progress_lower_saturation_rate"]), 1.0)

            with Path(result["credit_horizon"]).open(
                "r", encoding="utf-8", newline=""
            ) as file:
                credit_reader = csv.DictReader(file)
                credits = list(credit_reader)
            self.assertEqual(credit_reader.fieldnames, list(CREDIT_COLUMNS))
            self.assertEqual(len(credits), 12)
            horizon_ten = next(row for row in credits if row["training_seed"] == "101" and row["horizon"] == "10")
            self.assertEqual(int(horizon_ten["samples"]), 6)
            self.assertEqual(horizon_ten["spearman_state_movement"], "")

            summary_text = Path(result["summary"]).read_text(encoding="utf-8")
            self.assertNotIn("NaN", summary_text)
            self.assertNotIn("Infinity", summary_text)
            summary = json.loads(summary_text)
            self.assertEqual(summary["policy_count"], 3)
            self.assertIsNone(
                summary["credit_horizon"][0]["spearman_state_movement"]
            )

            figure_paths = list(result["figures"].values())[:-1]
            figure_paths.extend(result["figures"]["final_gap_return"].values())
            self.assertEqual(len(figure_paths), 5)
            for path in figure_paths:
                data = Path(path).read_bytes()
                self.assertGreater(len(data), 8)
                self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")

            with self.assertRaises(FileExistsError):
                diagnose_comparison(comparison)


if __name__ == "__main__":
    unittest.main()
