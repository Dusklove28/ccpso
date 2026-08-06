from dataclasses import asdict, dataclass
import random

import numpy as np
import torch

from agents.td3.replay_buffer import ReplayBuffer
from agents.td3.td3 import TD3
from environments.ccpso_env import REWARD_MODES
from environments.factory import make_ccpso_env
from problems.classic import make_classic_problem
from problems.metadata import serialize_problem
from training.td3_online import TD3OnlineConfig, train_online


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

    def __post_init__(self):
        if (
            not isinstance(self.problem_name, str)
            or not self.problem_name.strip()
        ):
            raise ValueError(
                "problem_name must be a non-empty string, "
                f"got {self.problem_name!r}"
            )

        for name in (
            "dimensions",
            "particles",
            "max_fe",
            "buffer_capacity",
            "policy_freq",
            "recent_window",
            "stagnation_horizon",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )
            object.__setattr__(self, name, int(value))

        if self.max_fe < self.particles:
            raise ValueError(
                "max_fe must be at least particles, "
                f"got max_fe={self.max_fe}, particles={self.particles}"
            )
        if not isinstance(self.online, TD3OnlineConfig):
            raise ValueError(
                "online must be an instance of TD3OnlineConfig"
            )
        if not (
            self.device is None
            or isinstance(self.device, (str, torch.device))
        ):
            raise ValueError(
                "device must be None, a string, or torch.device, "
                f"got {self.device!r}"
            )

        float_rules = {
            "max_action": lambda value: value > 0.0,
            "discount": lambda value: 0.0 <= value <= 1.0,
            "tau": lambda value: 0.0 < value <= 1.0,
            "policy_noise": lambda value: value >= 0.0,
            "noise_clip": lambda value: value >= 0.0,
        }
        for name, valid_range in float_rules.items():
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(
                    value,
                    (int, float, np.integer, np.floating),
                )
                or not np.isfinite(value)
                or not valid_range(float(value))
            ):
                raise ValueError(
                    f"invalid {name} value {value!r}"
                )
            object.__setattr__(self, name, float(value))

        for name in ("c_min", "c_max"):
            value = getattr(self, name)
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
            object.__setattr__(self, name, float(value))
        if self.c_min > self.c_max:
            raise ValueError(
                "c_min must be <= c_max, got "
                f"c_min={self.c_min!r}, c_max={self.c_max!r}"
            )

        if not isinstance(self.reward_mode, str):
            raise ValueError(
                "reward_mode must be a string, got "
                f"{self.reward_mode!r}"
            )
        if self.reward_mode not in REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {REWARD_MODES}, "
                f"got {self.reward_mode!r}"
            )

        reward_epsilon = self.reward_epsilon
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
        object.__setattr__(
            self,
            "reward_epsilon",
            float(reward_epsilon),
        )


@dataclass(frozen=True)
class ClassicTD3ExperimentResult:
    policy: TD3
    replay_buffer: ReplayBuffer
    problem: object
    config: dict
    problem_metadata: dict
    training_records: dict


def _resolve_device(device):
    if device is None or device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(device)


def _serialize_config(config, resolved_device):
    return {
        "problem_name": config.problem_name,
        "dimensions": int(config.dimensions),
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


def run_classic_td3(config):
    if not isinstance(config, ClassicTD3ExperimentConfig):
        raise TypeError(
            "config must be an instance of ClassicTD3ExperimentConfig"
        )

    seed = config.online.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    problem = make_classic_problem(
        config.problem_name,
        dimensions=config.dimensions,
    )
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

    return ClassicTD3ExperimentResult(
        policy=policy,
        replay_buffer=replay_buffer,
        problem=problem,
        config=_serialize_config(config, resolved_device),
        problem_metadata=serialize_problem(problem),
        training_records=training_records,
    )
