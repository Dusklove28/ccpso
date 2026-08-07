from dataclasses import asdict, dataclass
import random

import numpy as np
import torch

from agents.td3.replay_buffer import ReplayBuffer
from agents.td3.td3 import TD3
from environments.ccpso_env import REWARD_MODES, STATE_MODES
from environments.factory import make_ccpso_env
from problems.classic import make_classic_problem
from problems.metadata import serialize_problem
from problems.spec import ProblemSpec
from training.td3_online import TD3OnlineConfig, train_online


def _validate_common_config(config, integer_fields):
    for name in integer_fields:
        value = getattr(config, name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(
                f"{name} must be a positive integer, got {value!r}"
            )
        object.__setattr__(config, name, int(value))

    if config.max_fe < config.particles:
        raise ValueError(
            "max_fe must be at least particles, "
            f"got max_fe={config.max_fe}, particles={config.particles}"
        )
    if not isinstance(config.online, TD3OnlineConfig):
        raise ValueError("online must be an instance of TD3OnlineConfig")
    if not (
        config.device is None
        or isinstance(config.device, (str, torch.device))
    ):
        raise ValueError(
            "device must be None, a string, or torch.device, "
            f"got {config.device!r}"
        )

    float_rules = {
        "max_action": lambda value: value > 0.0,
        "discount": lambda value: 0.0 <= value <= 1.0,
        "tau": lambda value: 0.0 < value <= 1.0,
        "policy_noise": lambda value: value >= 0.0,
        "noise_clip": lambda value: value >= 0.0,
    }
    for name, valid_range in float_rules.items():
        value = getattr(config, name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(
                value,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(value)
            or not valid_range(float(value))
        ):
            raise ValueError(f"invalid {name} value {value!r}")
        object.__setattr__(config, name, float(value))

    for name in ("c_min", "c_max"):
        value = getattr(config, name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(
                value,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"{name} must be a finite real number, got {value!r}"
            )
        object.__setattr__(config, name, float(value))
    if config.c_min > config.c_max:
        raise ValueError(
            "c_min must be <= c_max, got "
            f"c_min={config.c_min!r}, c_max={config.c_max!r}"
        )

    if not isinstance(config.reward_mode, str):
        raise ValueError(
            "reward_mode must be a string, got "
            f"{config.reward_mode!r}"
        )
    if config.reward_mode not in REWARD_MODES:
        raise ValueError(
            f"reward_mode must be one of {REWARD_MODES}, "
            f"got {config.reward_mode!r}"
        )

    if not isinstance(config.state_mode, str):
        raise ValueError(
            "state_mode must be a string, got "
            f"{config.state_mode!r}"
        )
    if config.state_mode not in STATE_MODES:
        raise ValueError(
            f"state_mode must be one of {STATE_MODES}, "
            f"got {config.state_mode!r}"
        )

    reward_epsilon = config.reward_epsilon
    if (
        isinstance(reward_epsilon, (bool, np.bool_))
        or not isinstance(
            reward_epsilon,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(reward_epsilon)
        or float(reward_epsilon) <= 0.0
    ):
        raise ValueError(
            "reward_epsilon must be a finite real number > 0, got "
            f"{reward_epsilon!r}"
        )
    object.__setattr__(config, "reward_epsilon", float(reward_epsilon))


@dataclass(frozen=True)
class TD3ProblemExperimentConfig:
    particles: int
    max_fe: int
    buffer_capacity: int
    online: TD3OnlineConfig
    device: object = "auto"
    max_action: float = 1.0
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    c_min: float = 0.0
    c_max: float = 1.5
    recent_window: int = 5
    stagnation_horizon: int = 10
    reward_mode: str = "step_log_improvement"
    reward_epsilon: float = 1e-12
    state_mode: str = "legacy_v1"

    def __post_init__(self):
        _validate_common_config(
            self,
            (
                "particles",
                "max_fe",
                "buffer_capacity",
                "policy_freq",
                "recent_window",
                "stagnation_horizon",
            ),
        )


@dataclass(frozen=True)
class ClassicTD3ExperimentConfig:
    problem_name: str
    dimensions: int
    particles: int
    max_fe: int
    buffer_capacity: int
    online: TD3OnlineConfig
    device: object = "auto"
    max_action: float = 1.0
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    c_min: float = 0.0
    c_max: float = 1.5
    recent_window: int = 5
    stagnation_horizon: int = 10
    reward_mode: str = "step_log_improvement"
    reward_epsilon: float = 1e-12
    state_mode: str = "legacy_v1"

    def __post_init__(self):
        if (
            not isinstance(self.problem_name, str)
            or not self.problem_name.strip()
        ):
            raise ValueError(
                "problem_name must be a non-empty string, "
                f"got {self.problem_name!r}"
            )
        _validate_common_config(
            self,
            (
                "dimensions",
                "particles",
                "max_fe",
                "buffer_capacity",
                "policy_freq",
                "recent_window",
                "stagnation_horizon",
            ),
        )


@dataclass(frozen=True)
class TD3ProblemExperimentResult:
    policy: TD3
    replay_buffer: ReplayBuffer
    problem: ProblemSpec
    config: dict
    problem_metadata: dict
    training_records: dict


@dataclass(frozen=True)
class ClassicTD3ExperimentResult(TD3ProblemExperimentResult):
    pass


def _resolve_device(device):
    if device is None or device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(device)


def _serialize_config(config, resolved_device, problem):
    common = {
        "particles": int(config.particles),
        "max_fe": int(config.max_fe),
        "buffer_capacity": int(config.buffer_capacity),
        "device": str(resolved_device),
        "environment": {
            "c_min": float(config.c_min),
            "c_max": float(config.c_max),
            "recent_window": int(config.recent_window),
            "stagnation_horizon": int(config.stagnation_horizon),
            "reward_mode": config.reward_mode,
            "reward_epsilon": float(config.reward_epsilon),
            "state_mode": config.state_mode,
        },
        "td3": {
            "max_action": float(config.max_action),
            "discount": float(config.discount),
            "tau": float(config.tau),
            "policy_noise": float(config.policy_noise),
            "noise_clip": float(config.noise_clip),
            "policy_freq": int(config.policy_freq),
        },
        "online": asdict(config.online),
    }

    if isinstance(config, ClassicTD3ExperimentConfig):
        return {
            "problem_name": config.problem_name,
            "dimensions": int(config.dimensions),
            **common,
        }
    return {
        "dimensions": int(problem.dimensions),
        **common,
    }


def run_td3_problem(problem: ProblemSpec, config):
    if not isinstance(problem, ProblemSpec):
        raise TypeError("problem must be an instance of ProblemSpec")
    if not isinstance(
        config,
        (TD3ProblemExperimentConfig, ClassicTD3ExperimentConfig),
    ):
        raise TypeError(
            "config must be a TD3ProblemExperimentConfig or "
            "ClassicTD3ExperimentConfig"
        )
    if isinstance(config, ClassicTD3ExperimentConfig):
        if problem.suite != "classic":
            raise ValueError(
                "ClassicTD3ExperimentConfig requires a classic problem"
            )
        if problem.dimensions != config.dimensions:
            raise ValueError(
                "problem dimensions must match config.dimensions, got "
                f"{problem.dimensions} and {config.dimensions}"
            )
        expected_problem_id = config.problem_name.strip().lower()
        if problem.problem_id != expected_problem_id:
            raise ValueError(
                "classic problem does not match config.problem_name, got "
                f"problem_id={problem.problem_id!r}, "
                f"problem_name={config.problem_name!r}"
            )

    seed = config.online.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    env = make_ccpso_env(
        problem,
        particles=config.particles,
        max_fe=config.max_fe,
        seed=seed,
        c_min=config.c_min,
        c_max=config.c_max,
        recent_window=config.recent_window,
        stagnation_horizon=config.stagnation_horizon,
        reward_mode=config.reward_mode,
        reward_epsilon=config.reward_epsilon,
        state_mode=config.state_mode,
    )

    resolved_device = _resolve_device(config.device)
    policy = TD3(
        state_dim=int(np.prod(env.observation_space.shape)),
        action_dim=int(np.prod(env.action_space.shape)),
        max_action=config.max_action,
        discount=config.discount,
        tau=config.tau,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        policy_freq=config.policy_freq,
        device=resolved_device,
    )
    replay_buffer = ReplayBuffer(
        state_dim=int(np.prod(env.observation_space.shape)),
        action_dim=int(np.prod(env.action_space.shape)),
        max_size=config.buffer_capacity,
        seed=seed,
        device=resolved_device,
    )
    training_records = train_online(
        env=env,
        policy=policy,
        replay_buffer=replay_buffer,
        config=config.online,
    )

    result_type = (
        ClassicTD3ExperimentResult
        if isinstance(config, ClassicTD3ExperimentConfig)
        else TD3ProblemExperimentResult
    )
    return result_type(
        policy=policy,
        replay_buffer=replay_buffer,
        problem=problem,
        config=_serialize_config(config, resolved_device, problem),
        problem_metadata=serialize_problem(problem),
        training_records=training_records,
    )


def run_classic_td3(config):
    if not isinstance(config, ClassicTD3ExperimentConfig):
        raise TypeError(
            "config must be an instance of ClassicTD3ExperimentConfig"
        )
    problem = make_classic_problem(
        config.problem_name,
        dimensions=config.dimensions,
    )
    return run_td3_problem(problem, config)
