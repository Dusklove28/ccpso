from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from learning_ddpg.evaluation.model_selection import select_best_models
from learning_ddpg.training import train_ccpso_small as training


FUNCTION_IDS = (1, 28)
TRAINING_SEED = 7
VALIDATION_SEEDS = (10_007,)


def main():
    package_root = Path(__file__).resolve().parents[1]

    # Keep this regression test short. The production values remain in config.
    training.TRAIN_EPISODES = 1
    training.CHECKPOINT_INTERVAL = 1
    training.LEARNING_STARTS = 8
    training.BATCH_SIZE = 8

    with TemporaryDirectory(
        prefix="_multifunction_chain_",
        dir=package_root,
    ) as temporary_directory:
        checkpoint_root = Path(temporary_directory) / "checkpoints"

        for function_id in FUNCTION_IDS:
            saved_paths = training.train_one_seed(
                function_id=function_id,
                training_seed=TRAINING_SEED,
                checkpoint_root=checkpoint_root,
                device=torch.device("cpu"),
            )
            assert len(saved_paths) == 1

        selection = select_best_models(
            checkpoint_dir=checkpoint_root,
            seeds=VALIDATION_SEEDS,
            device="cpu",
        )

        assert tuple(selection["grouped_results"]) == FUNCTION_IDS
        assert len(selection["selected_models"]) == len(FUNCTION_IDS)

        for function_id in FUNCTION_IDS:
            function_dir = checkpoint_root / f"F{function_id:02d}"
            assert (function_dir / "best_actor.pt").is_file()
            assert (function_dir / "selected_models.csv").is_file()
            assert (function_dir / "model_selection.json").is_file()

        assert (checkpoint_root / "selected_models.csv").is_file()
        assert (checkpoint_root / "model_selection.json").is_file()

    print("CEC2013 multi-function train/evaluate/select chain: passed")


if __name__ == "__main__":
    main()
