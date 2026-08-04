# Actor、Critic 网络实现，共四个

# 钟摆的动作网络 Actor
import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)

        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0

        self.register_buffer("action_scale",action_scale)
        self.register_buffer("action_bias",action_bias)

    def forward(self,state):
        normalized_action = self.net(state)
        return normalized_action * self.action_scale + self.action_bias

# 钟摆的评论网络 Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        return self.net(state_action)


if __name__ == "__main__":
    state_dim = 4
    action_dim = 3
    action_low = -2
    action_high = 2

    actor = Actor(state_dim, action_dim, action_low, action_high)
    critic = Critic(state_dim, action_dim)

    batch_size = 8
    test_state = torch.randn(batch_size,state_dim)

    pred_action = actor(test_state)
    print("=== Actor 维度校验 ===")
    print(f"输入state shape: {test_state.shape}")
    print(f"输出action shape: {pred_action.shape}")
    print(f"预期输出shape: ({batch_size}, {action_dim})")
    print("Actor维度正常！\n" if pred_action.shape == (batch_size, action_dim) else "Actor维度出错！\n")
    # critic.forward()
