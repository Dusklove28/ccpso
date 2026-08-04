import gymnasium as gym
import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from collections import deque

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

# ========= 1. Environment Wrapper =========
class EpisodicRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cumulative_reward = 0.0

    def step(self, action):
        result = self.env.step(action)
        obs, reward, terminated, truncated, info = result
        self.cumulative_reward += reward
        
        done = terminated or truncated
        if done:
            reward = self.cumulative_reward
            self.cumulative_reward = 0.0
        
        else:
            reward = 0.0
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.cumulative_reward = 0.0
        result = self.env.reset(**kwargs)
        return result


# ========= 2. Trajectory Replay Buffer =========
class TrajectoryReplay:
    def __init__(self, max_len=200):
        self.trajectories = deque(maxlen=max_len)

    def add_trajectory(self, traj):
        self.trajectories.append(traj)

    def sample(self, batch_size):
        return random.sample(self.trajectories, min(batch_size, len(self)))

    def __len__(self):
        return len(self.trajectories)


# ========= 3. GP-based Reward Model =========
class GPRewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        
        # Neural network for mean function μ_θ
        self.mean_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Learnable kernel hyperparameters
        self.log_sigma_f = nn.Parameter(torch.log(torch.tensor(1.0)))       # Signal variance
        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(1.0)))  # Length scale
        self.log_sigma_tau = nn.Parameter(torch.log(torch.tensor(0.1)))     # Observation noise
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(1.0)))         # RQ kernel alpha parameter
        
    def compute_mean(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.mean_network(x).squeeze(-1)
    
    def rbf_kernel(self, X1, X2):
        """Rational Quadratic kernel"""
        sigma_f = torch.exp(self.log_sigma_f)
        length_scale = torch.exp(self.log_length_scale)
        alpha = torch.exp(self.log_alpha)
        
        # Compute pairwise squared distances
        X1_sqnorms = torch.sum(X1**2, dim=-1, keepdim=True)
        X2_sqnorms = torch.sum(X2**2, dim=-1, keepdim=True)
        sq_dist = X1_sqnorms + X2_sqnorms.t() - 2 * torch.mm(X1, X2.t())
        
        # Rational Quadratic kernel
        K = sigma_f**2 * (1 + sq_dist / (2 * alpha * length_scale**2))**(-alpha)
        
        return K
    
    def forward(self, states, actions):
        """For compatibility with SAC - just return mean"""
        return self.compute_mean(states, actions)


# ========= 4. GP Training Function =========
def train_gp_reward_model(reward_model, optimizer, trajectories, device='cuda'):
    """
    Train GP reward model using marginal likelihood maximization
    """
    reward_model.train()
    total_loss = torch.tensor(0.0, device=device)
    
    for traj in trajectories:
        # Extract trajectory information
        R_ep = torch.tensor(sum([t[2] for t in traj]), dtype=torch.float32, device=device)
        T = len(traj)
        
        # Prepare states and actions
        states_list, actions_list = [], []
        for (s, a, r, s_next, done) in traj:
            states_list.append(s)
            actions_list.append(a)
        
        states = torch.tensor(np.array(states_list), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions_list), dtype=torch.float32, device=device)
        
        # Concatenate state-action pairs for kernel computation
        X = torch.cat([states, actions], dim=-1)  # (T, state_dim + action_dim)
        
        # Compute mean vector
        mu = reward_model.compute_mean(states, actions)  # (T,)
        
        # Compute kernel matrix
        K = reward_model.rbf_kernel(X, X)  # (T, T)
        
        # Add observation noise
        sigma_tau_sq = torch.exp(2 * reward_model.log_sigma_tau)
        K_noise = K + sigma_tau_sq * torch.eye(T, device=device)
        
        # Ensure numerical stability
        K_noise = K_noise + 1e-4 * torch.eye(T, device=device)
        
        # Compute leave-one-out targets
        mu_sum = torch.sum(mu)
        loo_targets = R_ep - (mu_sum - mu)  # (T,)
        
        try:
            L = torch.linalg.cholesky(K_noise)
            
            # Solve K_noise * alpha = (loo_targets - mu)
            residual = loo_targets - mu
            alpha = torch.cholesky_solve(residual.unsqueeze(-1), L).squeeze(-1)
            
            # Compute negative log marginal likelihood
            # 0.5 * (loo_targets - mu)^T * K_noise^{-1} * (loo_targets - mu)
            data_fit = 0.5 * torch.dot(residual, alpha)
            
            # 0.5 * log|K_noise|
            complexity = torch.sum(torch.log(torch.diag(L)))
            
            # 0.5 * T * log(2π)
            const = 0.5 * T * math.log(2 * math.pi)
            
            loss = data_fit + complexity + const
            total_loss += loss
            
        except RuntimeError as e:
            # If Cholesky fails, add large penalty
            print(f"Cholesky decomposition failed: {e}")
            total_loss += 1e6
    
    # Average loss over trajectories
    loss_mean = total_loss / len(trajectories)
    
    # Backpropagation
    optimizer.zero_grad()
    loss_mean.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss_mean.item()


# ========= 5. Helper Functions =========
def collect_episodes(env, model, n_episodes, device='cuda'):
    """Collect trajectories using current policy"""
    trajectories = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        traj = []
        
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action, _ = model.predict(obs_tensor.cpu().numpy(), deterministic=False)
            
            next_obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            traj.append((obs, action[0], reward, next_obs, done))
            obs = next_obs
        
        trajectories.append(traj)
    
    return trajectories


def add_shaped_transitions_to_replay(model, reward_model, trajectories, device='cuda'):
    """Add transitions to SAC replay buffer with GP-predicted rewards"""
    for traj in trajectories:
        states_list = []
        actions_list = []
        next_states_list = []
        dones_list = []
        
        for (s, a, r, s_next, d) in traj:
            states_list.append(s)
            actions_list.append(a)
            next_states_list.append(s_next)
            dones_list.append(d)
        
        states = torch.tensor(np.array(states_list), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions_list), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            # Use mean function as shaped reward
            shaped_rewards = reward_model.compute_mean(states, actions).cpu().numpy()
        
        for i in range(len(traj)):
            s, a, _, s_next, d = traj[i]
            r_shaped = shaped_rewards[i]
            model.replay_buffer.add(
                s, s_next, a, r_shaped, d, infos=[{}],
            )


def evaluate_performance(model, env, n_eval_episodes=10, deterministic=True):
    """Evaluate policy performance"""
    returns = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
        
        returns.append(ep_return)
    
    return float(np.mean(returns))


# ========= 6. Main Training Loop =========
def main():
    class Args:
        env = "HalfCheetah-v4"
        episodes = 1000
        
        # Replay Buffer
        traj_replay_buffer_size = 200
        sac_replay_buffer_size = 1e6
        num_traj_collect_per_ep = 1
        
        # Training Parameters - SAC
        sac_learning_rate = 3e-4
        sac_gradient_steps = 1000
        sac_gradient_batch = 64
        gamma = 0.99
        
        # Training Parameters - GP-LRR
        gp_learning_rate = 1e-3
        gp_gradient_steps = 10
        gp_gradient_batch = 4
        
        # Architecture
        hidden_size = 256
        
        # Other
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seed = 42
        
        # Evaluation
        eval_freq = 5
        n_eval_episodes = 20
    
    args = Args()
    
    # Create environment
    base_env = gym.make(args.env)
    env = EpisodicRewardWrapper(base_env)
    vec_env = DummyVecEnv([lambda: env])
    
    eval_env = gym.make(args.env)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create SAC model
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.sac_learning_rate,
        buffer_size=int(args.sac_replay_buffer_size),
        gamma=args.gamma,
        device=args.device,
        verbose=1,
        seed=args.seed
    )
    
    if not hasattr(model, '_logger'):
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(folder=None, format_strings=["stdout"]))
    
    # Get dimensions
    state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]
    
    # Create GP reward model
    reward_model = GPRewardModel(state_dim, action_dim, hidden_size=args.hidden_size).to(args.device)
    reward_optimizer = optim.Adam(reward_model.parameters(), lr=args.gp_learning_rate)
    
    # Create trajectory replay buffer
    traj_replay = TrajectoryReplay(max_len=args.traj_replay_buffer_size)
    
    # Training metrics
    ep_returns = []
    eval_returns = []
    gp_losses = []
    
    print(f"Training GP-LRR on {args.env}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {args.device}")
    print("-" * 50)
    
    for ep in tqdm(range(args.episodes), desc="Training"):
        # Collect trajectories
        new_trajectories = collect_episodes(env, model, n_episodes=args.num_traj_collect_per_ep, device=args.device)
        
        # Add to trajectory buffer
        for traj in new_trajectories:
            traj_replay.add_trajectory(traj)
        
        # Train GP reward model if we have enough trajectories
        if len(traj_replay) >= args.gp_gradient_batch:
            gp_loss_vals = []
            for _ in range(args.gp_gradient_steps):
                sampled_trajs = traj_replay.sample(args.gp_gradient_batch)
                loss_val = train_gp_reward_model(
                    reward_model, reward_optimizer, sampled_trajs, device=args.device
                )
                gp_loss_vals.append(loss_val)
            gp_losses.append(np.mean(gp_loss_vals))
        
        # Add shaped transitions to SAC buffer
        add_shaped_transitions_to_replay(model, reward_model, new_trajectories, device=args.device)
        
        # Train SAC
        if model.replay_buffer.size() > args.sac_gradient_batch:
            try:
                model.train(gradient_steps=args.sac_gradient_steps, batch_size=args.sac_gradient_batch)
            except AttributeError:
                for _ in range(args.sac_gradient_steps):
                    model._train_step(batch_size=args.sac_gradient_batch, gradient_steps=1)
        
        # Record episodic return
        ep_return = sum([t[2] for t in new_trajectories[-1]])
        ep_returns.append(ep_return)
        
        # Evaluation
        if (ep + 1) % args.eval_freq == 0:
            eval_return = evaluate_performance(model, eval_env, n_eval_episodes=args.n_eval_episodes)
            eval_returns.append(eval_return)
            
            # Print kernel parameters
            with torch.no_grad():
                sigma_f = torch.exp(reward_model.log_sigma_f).item()
                length_scale = torch.exp(reward_model.log_length_scale).item()
                sigma_tau = torch.exp(reward_model.log_sigma_tau).item()
            
            print(f"\nEpisode {ep+1}/{args.episodes}")
            print(f"  Episodic return: {ep_return:.2f}")
            print(f"  Eval return: {eval_return:.2f}")
            print(f"  GP Loss: {gp_losses[-1] if gp_losses else 'N/A':.4f}")
            print(f"  Kernel params - σ_f: {sigma_f:.3f}, ℓ: {length_scale:.3f}, σ_τ: {sigma_tau:.3f}")
    
    print("\nTraining completed!")
    return {
        'ep_returns': ep_returns,
        'eval_returns': eval_returns,
        'gp_losses': gp_losses
    }