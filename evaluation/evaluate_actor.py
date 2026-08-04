import argparse
from pathlib import Path

import numpy as np
import torch

from learning_ddpg.configs.ccpso_config import (
    CONTROL_LEVEL,
    DIMENSIONS,
    FUNCTION_IDS,
    STAGE_ACTION_MODE,
    STAGE_FE,
    STAGE_MAX_DELTA_C,
    STAGE_SMOOTHING_ALPHA,
    TEST_SEEDS,
    VALIDATION_SEEDS,
)
from learning_ddpg.environments.factory import make_cec2013_env
from learning_ddpg.functions.cec2013 import CEC2013_OPTIMA
from learning_ddpg.networks.networks import Actor


DEFAULT_VALIDATION_SEEDS = VALIDATION_SEEDS
DEFAULT_TEST_SEEDS = TEST_SEEDS


def load_actor(model_path, device):
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Actor checkpoint not found: {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    required_keys = {
        "episode",
        "actor_state_dict",
        "state_dim",
        "action_dim",
        "action_low",
        "action_high",
        "objective_name",
        "function_id",
        "function_optimum",
        "dimensions",
        "control_level",
        "stage_fe",
        "stage_action_mode",
        "stage_smoothing_alpha",
        "stage_max_delta_c",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing required keys: {sorted(missing_keys)}"
        )

    function_id = int(checkpoint["function_id"])
    if function_id not in FUNCTION_IDS:
        raise ValueError(
            f"Checkpoint is for disabled/unknown function F{function_id}"
        )
    if int(checkpoint["dimensions"]) != DIMENSIONS:
        raise ValueError(
            f"Checkpoint dimension is {checkpoint['dimensions']}, configured for {DIMENSIONS}"
        )
    expected_optimum = CEC2013_OPTIMA[function_id]
    if not np.isclose(
        float(checkpoint["function_optimum"]),
        expected_optimum,
    ):
        raise ValueError(
            "Checkpoint optimum does not match the configured CEC2013 function"
        )
    if checkpoint["control_level"] != CONTROL_LEVEL:
        raise ValueError(
            "Checkpoint control level does not match the configured Stage environment"
        )
    if int(checkpoint["stage_fe"]) != STAGE_FE:
        raise ValueError(
            "Checkpoint stage_fe does not match the configured Stage environment"
        )
    if checkpoint["stage_action_mode"] != STAGE_ACTION_MODE:
        raise ValueError(
            "Checkpoint stage_action_mode does not match the configured Stage environment"
        )
    if not np.isclose(
        float(checkpoint["stage_smoothing_alpha"]),
        STAGE_SMOOTHING_ALPHA,
    ):
        raise ValueError(
            "Checkpoint stage_smoothing_alpha does not match the configured Stage environment"
        )
    if not np.isclose(
        float(checkpoint["stage_max_delta_c"]),
        STAGE_MAX_DELTA_C,
    ):
        raise ValueError(
            "Checkpoint stage_max_delta_c does not match the configured Stage environment"
        )

    actor = Actor(
        state_dim=int(checkpoint["state_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        action_low=checkpoint["action_low"],
        action_high=checkpoint["action_high"],
    ).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    return actor, checkpoint


@torch.no_grad()
def select_action(actor, state, device):
    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    action = actor(state_tensor)
    return action.cpu().numpy()[0].astype(np.float32)


def evaluate_one_seed(actor, env, seed, device):
    state, info = env.reset(seed=seed)
    episode_reward = 0.0
    steps = 0

    while True:
        action = select_action(actor, state, device)
        next_state, reward, terminated, truncated, info = env.step(action)

        state = next_state
        episode_reward += reward
        steps += 1

        if terminated or truncated:
            break

    return {
        "seed": int(seed),
        "steps": int(steps),
        "episode_reward": float(episode_reward),
        "final_gbest": float(info["gbest_fitness"]),
        "final_gap": float(info["gap"]),
    }


def evaluate_checkpoint(
    model_path,
    seeds=DEFAULT_TEST_SEEDS,
    device=None,
):
    device = torch.device(device or "cpu")
    actor, checkpoint = load_actor(model_path, device)
    function_id = int(checkpoint["function_id"])
    env = make_cec2013_env(function_id)

    try:
        runs = [
            evaluate_one_seed(actor, env, seed, device)
            for seed in seeds
        ]
    finally:
        env.close()

    final_gbests = np.asarray(
        [run["final_gbest"] for run in runs],
        dtype=np.float64,
    )
    final_gaps = np.asarray(
        [run["final_gap"] for run in runs],
        dtype=np.float64,
    )
    rewards = np.asarray(
        [run["episode_reward"] for run in runs],
        dtype=np.float64,
    )

    return {
        "model_path": str(Path(model_path).resolve()),
        "training_episode": int(checkpoint["episode"]),
        "training_seed": checkpoint.get("training_seed"),
        "function_id": int(checkpoint["function_id"]),
        "function_optimum": float(checkpoint["function_optimum"]),
        "dimensions": int(checkpoint["dimensions"]),
        "control_level": checkpoint["control_level"],
        "stage_fe": int(checkpoint["stage_fe"]),
        "stage_action_mode": checkpoint["stage_action_mode"],
        "stage_smoothing_alpha": float(
            checkpoint["stage_smoothing_alpha"]
        ),
        "stage_max_delta_c": float(checkpoint["stage_max_delta_c"]),
        "seeds": [int(seed) for seed in seeds],
        "runs": runs,
        "gbest_mean": float(np.mean(final_gbests)),
        "gbest_std": float(np.std(final_gbests)),
        "gap_mean": float(np.mean(final_gaps)),
        "gap_std": float(np.std(final_gaps)),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
    }


def print_evaluation(result):
    print(
        f"model={result['model_path']} | "
        f"F{result['function_id']} D={result['dimensions']} | "
        f"control={result['control_level']} stage_fe={result['stage_fe']} | "
        f"training_seed={result['training_seed']} | "
        f"training_episode={result['training_episode']}"
    )

    for run in result["runs"]:
        print(
            f"seed={run['seed']} "
            f"steps={run['steps']} "
            f"reward={run['episode_reward']:.6f} "
            f"gbest={run['final_gbest']:.6e} "
            f"gap={run['final_gap']:.6e}"
        )

    print(
        f"summary "
        f"gbest={result['gbest_mean']:.6e} +/- {result['gbest_std']:.6e} "
        f"gap={result['gap_mean']:.6e} +/- {result['gap_std']:.6e} "
        f"reward={result['reward_mean']:.6f} +/- {result['reward_std']:.6f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one selected CCPSO Actor on fixed final-test seeds."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to an actor_episode_XXX.pt checkpoint.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_TEST_SEEDS),
        help="Independent final-test seeds; model selection passes validation seeds explicitly.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, for example cpu or cuda.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate_checkpoint(
        model_path=args.model,
        seeds=args.seeds,
        device=args.device,
    )
    print_evaluation(result)


if __name__ == "__main__":
    main()
