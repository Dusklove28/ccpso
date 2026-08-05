import tempfile
import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

from evaluation.plot_run import FIGURE_FILENAMES, plot_run
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
            log_paths = save_classic_td3_run(result, run_dir)
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

    def test_empty_updates_still_produce_loss_figure(self):
        result = _run_experiment(learning_starts=100, episodes=1)
        self.assertEqual(result.training_records["total_updates"], 0)

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            run_dir = Path(temp_dir) / "run"
            output_dir = Path(temp_dir) / "custom-figures"
            save_classic_td3_run(result, run_dir)

            figure_paths = plot_run(run_dir, output_dir=output_dir)

            self.assert_valid_pngs(figure_paths)
            self.assertEqual(
                Path(figure_paths["td3_losses"]).parent,
                output_dir.resolve(),
            )
            self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
