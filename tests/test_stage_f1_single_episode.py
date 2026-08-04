from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from learning_ddpg.configs.ccpso_config import (
    CONTROL_LEVEL,
    STAGE_ACTION_MODE,
    STAGE_FE,
    STAGE_MAX_DELTA_C,
    STAGE_SMOOTHING_ALPHA,
    TRAIN_SEEDS,
)
from learning_ddpg.evaluation.evaluate_actor import load_actor
from learning_ddpg.training import train_ccpso_small as training


def main():
    package_root = Path(__file__).resolve().parents[1]
    training_seed = TRAIN_SEEDS[0]

    # One short training episode, while still exercising backward updates.
    training.TRAIN_EPISODES = 1
    training.CHECKPOINT_INTERVAL = 1
    training.LEARNING_STARTS = 4
    training.BATCH_SIZE = 4

    with TemporaryDirectory(
        prefix="_stage_f1_single_episode_",
        dir=package_root,
    ) as temporary_directory:
        saved_paths = training.train_one_seed(
            function_id=1,
            training_seed=training_seed,
            checkpoint_root=Path(temporary_directory) / "checkpoints",
            device=torch.device("cpu"),
        )

        assert len(saved_paths) == 1
        actor, checkpoint = load_actor(saved_paths[0], torch.device("cpu"))
        assert actor.training is False
        assert checkpoint["training_seed"] == training_seed
        assert checkpoint["control_level"] == CONTROL_LEVEL
        assert checkpoint["stage_fe"] == STAGE_FE
        assert checkpoint["stage_action_mode"] == STAGE_ACTION_MODE
        assert checkpoint["stage_smoothing_alpha"] == STAGE_SMOOTHING_ALPHA
        assert checkpoint["stage_max_delta_c"] == STAGE_MAX_DELTA_C

    print(
        "F1 single Stage training episode: passed | "
        f"training_seed={training_seed}"
    )


if __name__ == "__main__":
    main()
