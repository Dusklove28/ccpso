import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_size=int(1e6),
            seed=None,
            device=None,
    ):
        self.state_dim = self._validate_positive_integer(
            "state_dim",
            state_dim,
        )
        self.action_dim = self._validate_positive_integer(
            "action_dim",
            action_dim,
        )
        self.max_size = self._validate_positive_integer(
            "max_size",
            max_size,
        )
        self.device = self._resolve_device(device)
        self.rng = np.random.default_rng(seed)

        self.ptr = 0
        self.size = 0

        self.state = np.zeros(
            (self.max_size, self.state_dim),
            dtype=np.float32,
        )
        self.action = np.zeros(
            (self.max_size, self.action_dim),
            dtype=np.float32,
        )
        self.next_state = np.zeros(
            (self.max_size, self.state_dim),
            dtype=np.float32,
        )
        self.reward = np.zeros(
            (self.max_size, 1),
            dtype=np.float32,
        )
        self.bootstrap_mask = np.zeros(
            (self.max_size, 1),
            dtype=np.float32,
        )

    @staticmethod
    def _validate_positive_integer(name, value):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(
                f"{name} must be a positive integer, got {value!r}"
            )
        return int(value)

    @staticmethod
    def _resolve_device(device):
        if device is None or device == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return torch.device(device)

    @staticmethod
    def _validate_vector(name, value, expected_shape):
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {vector.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain only finite values")
        return vector

    @staticmethod
    def _validate_reward(reward):
        reward_array = np.asarray(reward, dtype=np.float32)
        if reward_array.shape != ():
            raise ValueError(
                f"reward has shape {reward_array.shape}, expected scalar shape ()"
            )
        reward_value = float(reward_array)
        if not np.isfinite(reward_value):
            raise ValueError(
                f"reward must be finite, got {reward_value!r}"
            )
        return reward_value

    def __len__(self):
        return self.size

    def add(self, state, action, next_state, reward, terminated):
        state_array = self._validate_vector(
            "state",
            state,
            (self.state_dim,),
        )
        action_array = self._validate_vector(
            "action",
            action,
            (self.action_dim,),
        )
        next_state_array = self._validate_vector(
            "next_state",
            next_state,
            (self.state_dim,),
        )
        reward_value = self._validate_reward(reward)
        if not isinstance(terminated, (bool, np.bool_)):
            raise ValueError(
                "terminated must be a bool or numpy.bool_, "
                f"got {terminated!r}"
            )
        bootstrap_mask = 0.0 if bool(terminated) else 1.0

        self.state[self.ptr] = state_array
        self.action[self.ptr] = action_array
        self.next_state[self.ptr] = next_state_array
        self.reward[self.ptr, 0] = reward_value
        self.bootstrap_mask[self.ptr, 0] = bootstrap_mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        batch_size = self._validate_positive_integer(
            "batch_size",
            batch_size,
        )
        if self.size == 0:
            raise ValueError("cannot sample from an empty ReplayBuffer")

        indices = self.rng.integers(
            0,
            self.size,
            size=batch_size,
        )

        return (
            torch.as_tensor(
                self.state[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                self.action[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                self.next_state[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                self.reward[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                self.bootstrap_mask[indices],
                dtype=torch.float32,
                device=self.device,
            ),
        )
