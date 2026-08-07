"""One-command TD3 pipeline for one public CEC2017 function."""

import argparse
import json
from pathlib import Path

from environments.ccpso_env import REWARD_MODES, STATE_MODES
from experiments.td3_pipeline import run_td3_pipeline
from problems.cec2017 import (
    CEC2017_FUNCTION_IDS,
    CEC2017_SUPPORTED_DIMENSIONS,
    make_cec2017_problem,
)
from training.td3_experiment import TD3ProblemExperimentConfig
from training.td3_online import TD3OnlineConfig


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run one TD3-CCPSO experiment using the project's public "
            "CEC2017 numbering F1..F29 (source F2 is excluded)."
        )
    )
    parser.add_argument(
        "--function-id",
        type=int,
        choices=CEC2017_FUNCTION_IDS,
        required=True,
        help="Project public function ID in 1..29, not a source ID.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        choices=CEC2017_SUPPORTED_DIMENSIONS,
        default=10,
    )
    parser.add_argument("--particles", type=int, default=20)
    parser.add_argument("--max-fe", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--c-min", type=float, default=0.0)
    parser.add_argument("--c-max", type=float, default=1.5)
    parser.add_argument("--recent-window", type=int, default=5)
    parser.add_argument("--stagnation-horizon", type=int, default=10)
    parser.add_argument(
        "--reward-mode",
        choices=REWARD_MODES,
        default="step_log_improvement",
    )
    parser.add_argument("--reward-epsilon", type=float, default=1e-12)
    parser.add_argument(
        "--state-mode",
        choices=STATE_MODES,
        default="legacy_v1",
    )
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        required=True,
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def problem_from_args(args):
    return make_cec2017_problem(
        args.function_id,
        args.dimensions,
    )


def config_from_args(args):
    online = TD3OnlineConfig(
        episodes=args.episodes,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        exploration_noise=args.exploration_noise,
        updates_per_step=args.updates_per_step,
        seed=args.seed,
    )
    return TD3ProblemExperimentConfig(
        particles=args.particles,
        max_fe=args.max_fe,
        buffer_capacity=args.buffer_capacity,
        device=args.device,
        c_min=args.c_min,
        c_max=args.c_max,
        recent_window=args.recent_window,
        stagnation_horizon=args.stagnation_horizon,
        reward_mode=args.reward_mode,
        reward_epsilon=args.reward_epsilon,
        state_mode=args.state_mode,
        discount=args.discount,
        online=online,
    )


def main(argv=None):
    args = parse_args(argv)
    problem = problem_from_args(args)
    config = config_from_args(args)
    pipeline_result = run_td3_pipeline(
        problem,
        config,
        output_root=args.output_root,
        run_name=args.run_name,
        evaluation_seeds=args.evaluation_seeds,
    )
    print(
        json.dumps(
            pipeline_result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return pipeline_result


if __name__ == "__main__":
    main()
