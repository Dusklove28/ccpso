# 动作选择、一次 update、软更新
import copy

import  numpy as np
import torch
from torch import nn
from torch.optim import Adam

from learning_ddpg.networks.networks import Actor, Critic

# 创建在线网络和目标网络
class DDPGAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            action_low,
            action_high,
            device, # GPU/CPU
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            noise_std=0.1,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std

        self.action_low = np.asarray(
            action_low, dtype=np.float32
        )
        self.action_high = np.asarray(
            action_high, dtype=np.float32
        )

        self.actor = Actor(
            state_dim,
            action_dim,
            action_low,
            action_high,
        ).to(device)

        self.critic = Critic(
            state_dim,
            action_dim,
        ).to(device)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # 目标网络只由软更新修改。
        for parameter in self.target_actor.parameters():
            parameter.requires_grad = False

        for parameter in self.target_critic.parameters():
            parameter.requires_grad = False

        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

        self.critic_loss_fn = nn.MSELoss()

    @torch.no_grad()
    def parameter_distance(self, network_a, network_b):
        """Return the L1 distance between two networks with the same structure."""
        return sum(
            torch.sum(torch.abs(parameter_a - parameter_b)).item()
            for parameter_a, parameter_b in zip(
                network_a.parameters(),
                network_b.parameters(),
            )
        )

    @torch.no_grad()
    def soft_update(self, source_network, target_network):
        for source_parameter, target_parameter in zip(
            source_network.parameters(),
            target_network.parameters(),
        ):
            target_parameter.data.mul_(1.0 - self.tau)
            target_parameter.data.add_(
                self.tau * source_parameter.data
            )

    # 动作选择
    def select_action(self, state, add_noise=True):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.actor.eval()

        with torch.no_grad():
            action = self.actor(state_tensor)

        self.actor.train()

        action = action.cpu().numpy()[0]

        if add_noise:
            noise = np.random.normal(
                loc=0.0,
                scale=self.noise_std,
                size=action.shape,
            ).astype(np.float32)

            action = action + noise

        action = np.clip(
            action,
            self.action_low,
            self.action_high,
        )

        return action.astype(np.float32)

    # 计算targetQ
    # 更新Critic
    # 更新Actor
    # 软更新目标网络
    # 实现一次 update()
    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = (
            replay_buffer.sample(batch_size)
        )

        # 1. 计算 Critic 的监督目标。
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(
                next_states,
                next_actions,
            )

            target_q_values = (
                rewards
                + self.gamma
                * (1.0 - dones)
                * next_q_values
            )# done存进buffer，取出来组成 batch 时就变成了 dones（浮点张量）；True对应1.0，False对应0.0

        # 2. 更新 Critic。
        current_q_values = self.critic(states, actions)

        critic_loss = self.critic_loss_fn(
            current_q_values,
            target_q_values,
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3. 更新 Actor。
        predicted_actions = self.actor(states)

        actor_loss = -self.critic(
            states,
            predicted_actions,
        ).mean()

        self.actor_optimizer.zero_grad()

        # 检查参数是否真的获得梯度
        actor_loss.backward()

        # 检查梯度不是全零
        actor_gradient_sum = sum(
            parameter.grad.abs().sum().item()
            for parameter in self.actor.parameters()
            if parameter.grad is not None
        )

        self.actor_optimizer.step()

        actor_distance_before_soft_update = self.parameter_distance(
            self.actor, self.target_actor
        )
        critic_distance_before_soft_update = self.parameter_distance(
            self.critic, self.target_critic
        )

        # 4. 软更新目标网络。
        self.soft_update(
            self.actor,
            self.target_actor,
        )

        self.soft_update(
            self.critic,
            self.target_critic,
        )

        actor_distance_after_soft_update = self.parameter_distance(
            self.actor, self.target_actor
        )
        critic_distance_after_soft_update = self.parameter_distance(
            self.critic, self.target_critic
        )

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "q_shape": tuple(current_q_values.shape),
            "target_q_shape": tuple(target_q_values.shape),
            "actor_gradient_sum": actor_gradient_sum,
            "actor_distance_before_soft_update": actor_distance_before_soft_update,
            "actor_distance_after_soft_update": actor_distance_after_soft_update,
            "critic_distance_before_soft_update": critic_distance_before_soft_update,
            "critic_distance_after_soft_update": critic_distance_after_soft_update,
        }
