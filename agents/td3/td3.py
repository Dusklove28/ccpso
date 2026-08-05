import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    CHECKPOINT_FORMAT_VERSION = 1

    @staticmethod
    def _resolve_device(device):
        if device is None or device == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return torch.device(device)

    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            device=None,
    ):
        self.device = self._resolve_device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, bootstrap_mask = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + bootstrap_mask * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        actor_updated = False
        actor_loss_value = None
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            actor_updated = True
            actor_loss_value = float(actor_loss.detach().item())

        return {
            "total_it": int(self.total_it),
            "critic_loss": float(critic_loss.detach().item()),
            "actor_updated": actor_updated,
            "actor_loss": actor_loss_value,
            "target_q_mean": float(target_Q.detach().mean().item()),
            "q1_mean": float(current_Q1.detach().mean().item()),
            "q2_mean": float(current_Q2.detach().mean().item()),
        }

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(
            torch.load(
                filename + "_critic",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.critic_optimizer.load_state_dict(
            torch.load(
                filename + "_critic_optimizer",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor.load_state_dict(
            torch.load(
                filename + "_actor",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.actor_optimizer.load_state_dict(
            torch.load(
                filename + "_actor_optimizer",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

    def save_checkpoint(self, path, metadata=None, overwrite=False):
        """Save model state for transfer and evaluation.

        ReplayBuffer contents and random-number-generator states are not
        included, so this checkpoint does not provide exact mid-training
        resumption.
        """
        checkpoint_path = Path(path).expanduser()
        if checkpoint_path.exists() and not overwrite:
            raise FileExistsError(
                f"checkpoint already exists: {checkpoint_path}"
            )
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            raise IsADirectoryError(
                f"checkpoint path is a directory: {checkpoint_path}"
            )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "format_version": self.CHECKPOINT_FORMAT_VERSION,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_action": self.max_action,
            "hyperparameters": {
                "discount": self.discount,
                "tau": self.tau,
                "policy_noise": self.policy_noise,
                "noise_clip": self.noise_clip,
                "policy_freq": self.policy_freq,
            },
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
            "metadata": copy.deepcopy(metadata),
        }
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path.resolve())

    def _validate_checkpoint(self, checkpoint, checkpoint_path):
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"invalid TD3 checkpoint structure in {checkpoint_path}"
            )

        required_keys = {
            "format_version",
            "state_dim",
            "action_dim",
            "max_action",
            "hyperparameters",
            "actor",
            "actor_target",
            "critic",
            "critic_target",
            "actor_optimizer",
            "critic_optimizer",
            "total_it",
            "metadata",
        }
        missing_keys = required_keys.difference(checkpoint)
        if missing_keys:
            raise ValueError(
                "invalid TD3 checkpoint: missing keys "
                f"{sorted(missing_keys)}"
            )

        format_version = checkpoint["format_version"]
        if format_version != self.CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "incompatible TD3 checkpoint format_version "
                f"{format_version!r}; expected "
                f"{self.CHECKPOINT_FORMAT_VERSION}"
            )

        for name in ("state_dim", "action_dim"):
            checkpoint_value = checkpoint[name]
            current_value = getattr(self, name)
            if checkpoint_value != current_value:
                raise ValueError(
                    f"incompatible TD3 checkpoint {name}: "
                    f"checkpoint has {checkpoint_value!r}, "
                    f"current model has {current_value!r}"
                )

        hyperparameters = checkpoint["hyperparameters"]
        required_hyperparameters = {
            "discount",
            "tau",
            "policy_noise",
            "noise_clip",
            "policy_freq",
        }
        if not isinstance(hyperparameters, dict):
            raise ValueError(
                "invalid TD3 checkpoint hyperparameters: expected a dict"
            )
        missing_hyperparameters = required_hyperparameters.difference(
            hyperparameters
        )
        if missing_hyperparameters:
            raise ValueError(
                "invalid TD3 checkpoint hyperparameters: missing keys "
                f"{sorted(missing_hyperparameters)}"
            )

    @staticmethod
    def _move_optimizer_to_device(optimizer, device):
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    state[name] = value.to(device)

    def load_checkpoint(self, path, load_optimizers=False):
        """Load a single-file checkpoint on ``self.device``.

        With ``load_optimizers=False`` only model state, TD3
        hyperparameters, and the update counter are restored, which is the
        intended mode for frozen evaluation. Optimizer restoration supports
        continued training but does not restore ReplayBuffer or RNG state.
        """
        checkpoint_path = Path(path).expanduser()
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self._validate_checkpoint(checkpoint, checkpoint_path)

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        for network in (
            self.actor,
            self.actor_target,
            self.critic,
            self.critic_target,
        ):
            network.to(self.device)

        self.max_action = checkpoint["max_action"]
        self.actor.max_action = self.max_action
        self.actor_target.max_action = self.max_action
        hyperparameters = checkpoint["hyperparameters"]
        self.discount = hyperparameters["discount"]
        self.tau = hyperparameters["tau"]
        self.policy_noise = hyperparameters["policy_noise"]
        self.noise_clip = hyperparameters["noise_clip"]
        self.policy_freq = hyperparameters["policy_freq"]
        self.total_it = checkpoint["total_it"]

        if load_optimizers:
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer"]
            )
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer"]
            )
            self._move_optimizer_to_device(
                self.actor_optimizer,
                self.device,
            )
            self._move_optimizer_to_device(
                self.critic_optimizer,
                self.device,
            )

        for network in (
            self.actor,
            self.actor_target,
            self.critic,
            self.critic_target,
        ):
            network.train(mode=bool(load_optimizers))
            network.requires_grad_(bool(load_optimizers))

        return copy.deepcopy(checkpoint["metadata"])
