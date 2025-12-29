import gymnasium as gym 
from gymnasium.wrappers import NormalizeObservation
from gymnasium.wrappers.vector import NormalizeObservation as VecNormalizeObservation
import multiprocessing as mp


import numpy as np
import pandas as pd
import typing as tt

import torch  
import torch.nn as nn 
import torch.nn.functional as F
import wandb
import os
HIDDEN_LAYER1  = 128

GAMMA = 0.99
LR_policy = 1e-4
LR_value = 1e-4

N_STEPS = 20
ENTROPY_BETA = 0.1
# ENV_ID = 'InvertedPendulum-v5'
N_ENV = 1
BATCH_SIZE = 64
ENV_ID = "InvertedPendulum-v5"
N_ENV = 8  # <- parallel envs

def make_env(env_id, seed, idx):
    def thunk():
        env = gym.make(env_id)
        env.reset(seed=seed + idx)
        return env
    return thunk



if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu' 
print(f'Using device : {device}')

# run = wandb.init(
#     entity=None,
#     project='RL_diary', 
#     config={
#         'env':ENV_ID,
#         "algorithm": "a2c",
#         "hidden_layer": HIDDEN_LAYER1,
#         "batch_size":  N_STEPS,
#         "gamma": GAMMA,
#         "lr_policy": LR_policy,
#         "lr_value":LR_value,
#         "entropy_beta":ENTROPY_BETA,
#         "N_STEPS": N_STEPS
#     }
    
# )

# eval_env = NormalizeReward(eval_env, gamma=GAMMA)  # Normalizes rewards

# class PolicyNet(nn.Module):
#     def __init__(self, input_size, fc, action_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, fc), nn.ReLU(),
#             nn.Linear(fc, fc), nn.ReLU(),   # optional but helps
#         )
#         self.mu = nn.Linear(fc, action_dim)
#         self.log_std = nn.Linear(fc, action_dim)
#         self.critic = nn.Linear(fc, 1)

#     def forward(self, x):
#         h = self.net(x)
#         mu = self.mu(h)

#         log_std = self.log_std(h).clamp(-5, 1)   # key: cap exploration
#         std = torch.exp(log_std)

#         v = self.critic(h)
#         return mu, std, v


def record_video(env, policy, device, low, high, max_steps=500, ):
    """Record a single episode and return frames + reward"""
    frames = []
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
   
    while not done and steps < max_steps:
        frame = env.render()
        frames.append(frame)        
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            mu, std, val = policy(state_tensor)
        dist = torch.distributions.Normal(mu, std)
        u = dist.sample()
        # a = torch.tanh(u)
        action = torch.clamp(u, low, high)
        # action = low + (a+1) * (high-low)*0.5
            
        action_env = action.squeeze(0).detach().cpu().numpy()
        state, reward, terminated, truncated, _ = env.step(action_env)
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
    return frames, total_reward, steps

def smooth(old: tt.Optional[float], val: float, alpha: float = 0.95,) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val

class PolicyNet(nn.Module):
    def __init__(self, input_size, fc, action_dim, action_low=-3,action_high=3, log_std_min=-20, log_std_max=-0.5, ):
        super().__init__()
        self.input_size = input_size
        self.fc = fc 
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        
        # # Store action space bounds
        action_low = torch.as_tensor(action_low, dtype=torch.float32).expand(action_dim)
        action_high = torch.as_tensor(action_high, dtype=torch.float32).expand(action_dim)
        self.register_buffer('action_low', action_low)
        self.register_buffer('action_high', action_high)
        self.register_buffer('action_scale', (action_high - action_low) / 2)
        self.register_buffer('action_bias', (action_high + action_low) / 2)
        
        
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.fc), 
            nn.ReLU(), 
            nn.Linear(self.fc, self.fc),
            nn.ReLU(), 
            
        )
        
        self.mu = nn.Sequential(
            nn.Linear(self.fc, self.action_dim), 
            # nn.Tanh()

        )
        self.log_std = nn.Sequential(
            nn.Linear(self.fc, self.action_dim), 
            # nn.Softplus()
        )
        # self.log_std = nn.Parameter(torch.zeros(action_dim))

        
        self.critic_head = nn.Linear(self.fc, 1)
        
        # Initialize with smaller weights to prevent saturation
    #     self.apply(self._init_weights)
    
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         # Use smaller gain for the mean head specifically
    #         if m is self.mu[-1]:  # If mu is Sequential
    #             nn.init.orthogonal_(m.weight, gain=0.001)  # Very small initial actions
    #         else:
    #             nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
    #         nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.net(x)
        # mu = self.mu(x)
        mu_normalized = torch.tanh(self.mu(x))  # Output in [-1, 1]
        mu = mu_normalized * self.action_scale + self.action_bias  # Scale to action space

        
        log_std = self.log_std(x)
        std = torch.exp(torch.clamp(log_std, self.log_std_min, self.log_std_max))
        
        
        

        
        # std = (self.std_net(x)+1e-4).clamp(1e-3, 2.0)
        
         # Use learned constant std (common in simple continuous control)
        # std = torch.exp(self.log_std.clamp(-2, 0.5))  # exp(-2)=0.135, exp(0.5)=1.65
        # std = std.expand_as(mu)  # Broadcast to batch size
        
        
        v = self.critic_head(x)
        return mu, std, v
    
    
@torch.no_grad()
def collect_rollout(venv, policy, device, n_steps, action_low_t, action_high_t, gamma, ep_ret_t=None, ep_len_t=None):    # venv obs shape: (N_ENV, obs_dim)
    obs, _ = venv.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

        # running episodic stats per env
    if ep_ret_t is None:
        ep_ret_t = torch.zeros(obs_t.shape[0], dtype=torch.float32, device=device)
    if ep_len_t is None:
        ep_len_t = torch.zeros(obs_t.shape[0], dtype=torch.int32, device=device)

    completed_ep_returns = []
    completed_ep_lengths = []


    obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf, ent_buf = [], [], [], [], [], [], []

    for t in range(n_steps):
        mu, std, v = policy(obs_t)                       # mu/std/v: (N_ENV, act_dim)/(N_ENV,1)
        v = v.squeeze(-1)                                # (N_ENV,)

        dist = torch.distributions.Normal(mu, std)
        u = dist.sample()                                # (N_ENV, act_dim)
        a = torch.clamp(u, action_low_t, action_high_t)   # clamp per-dim

        logp = dist.log_prob(a).sum(dim=-1)              # (N_ENV,)
        ent = dist.entropy().sum(dim=-1)                 # (N_ENV,)

        next_obs, rew, term, trunc, _ = venv.step(a.detach().cpu().numpy())
        done = np.logical_or(term, trunc)                # (N_ENV,)
        
        # update episodic trackers
        rew_t = torch.tensor(rew, dtype=torch.float32, device=device)
        done_t = torch.tensor(done, dtype=torch.bool, device=device)
        ep_ret_t = ep_ret_t + rew_t
        ep_len_t = ep_len_t + 1

        # record finished episodes and reset their counters
        if done_t.any():
            idxs = torch.nonzero(done_t, as_tuple=False).squeeze(-1)
            completed_ep_returns.extend(ep_ret_t[idxs].detach().cpu().tolist())
            completed_ep_lengths.extend(ep_len_t[idxs].detach().cpu().tolist())
            ep_ret_t[idxs] = 0.0
            ep_len_t[idxs] = 0


        obs_buf.append(obs_t)
        act_buf.append(a)
        rew_buf.append(rew_t)
        # rew_buf.append(torch.tensor(rew, dtype=torch.float32, device=device))
        done_buf.append(torch.tensor(done, dtype=torch.float32, device=device))
        val_buf.append(v)
        logp_buf.append(logp)
        ent_buf.append(ent)

        obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

    # bootstrap value for last state
    mu, std, last_v = policy(obs_t)
    last_v = last_v.squeeze(-1)  # (N_ENV,)

    # compute n-step returns backward: R_t = r_t + gamma*(1-done)*R_{t+1}
    returns = []
    R = last_v
    for t in reversed(range(n_steps)):
        R = rew_buf[t] + gamma * (1.0 - done_buf[t]) * R
        returns.append(R)
    returns.reverse()  # list length T, each (N_ENV,)

    # flatten (T, N_ENV, ...) -> (T*N_ENV, ...)
    obs_b   = torch.cat(obs_buf, dim=0)                                # (T*N_ENV, obs_dim)
    act_b   = torch.cat(act_buf, dim=0)                                # (T*N_ENV, act_dim)
    ret_b   = torch.cat(returns, dim=0)                                # (T*N_ENV,)
    val_b   = torch.cat(val_buf, dim=0)                                # (T*N_ENV,)
    logp_b  = torch.cat(logp_buf, dim=0)                               # (T*N_ENV,)
    ent_b   = torch.cat(ent_buf, dim=0).mean()                         # scalar mean entropy

    # return obs_b, act_b, ret_b, val_b, logp_b, ent_b
    return obs_b, act_b, ret_b, val_b, logp_b, ent_b, completed_ep_returns, completed_ep_lengths, ep_ret_t, ep_len_t

def main(): 


    # W&B: only initialize in the main process (important when using AsyncVectorEnv / multiprocessing spawn)
    os.environ.setdefault("WANDB_START_METHOD", "thread")
    run = wandb.init(
        entity=None,
        project='RL_diary',
        config={
            'env': ENV_ID,
            "algorithm": "a2c",
            "hidden_layer": HIDDEN_LAYER1,
            "batch_size": N_STEPS,
            "gamma": GAMMA,
            "lr_policy": LR_policy,
            "lr_value": LR_value,
            "entropy_beta": ENTROPY_BETA,
            "N_STEPS": N_STEPS,
            "N_ENV": N_ENV,
        }
    )
    venv = gym.vector.AsyncVectorEnv([make_env(ENV_ID, seed=42, idx=i) for i in range(N_ENV)])
    venv = VecNormalizeObservation(venv)              # works for vector env


    eval_env = gym.make(ENV_ID, render_mode='rgb_array')
    eval_env = NormalizeObservation(eval_env)  # Normalizes observations to ~N(0,1)

    # NOTE: VectorEnv spaces are batched (shape starts with N_ENV). Use single_* spaces for model dims.
    obs_dim = venv.single_observation_space.shape[0]
    act_dim = venv.single_action_space.shape[0]
    policy = PolicyNet(
        obs_dim,
        HIDDEN_LAYER1,
        act_dim,
        action_low=venv.single_action_space.low,
        action_high=venv.single_action_space.high,
    ).to(device)
    
    # optimizer = torch.optim.Adam(policy.parameters(),lr=LR,)
    policy_params = list(policy.net.parameters()) + list(policy.mu.parameters()) + list(policy.log_std.parameters())
    value_params = list(policy.critic_head.parameters())

    optimizer_policy = torch.optim.Adam(policy_params, lr=LR_policy)
    optimizer_value = torch.optim.Adam(value_params, lr=LR_value)  # 2x learning rate for value




    action_low  = torch.tensor(venv.single_action_space.low,  dtype=torch.float32, device=device)
    action_high = torch.tensor(venv.single_action_space.high, dtype=torch.float32, device=device)

    ep_returns_hist = []
    ep_lens_hist = []
    running_ep_ret_t = None
    running_ep_len_t = None


    for update_idx in range(1_000_000):
        # obs_b, act_b, ret_b, val_b, logp_b, entropy = collect_rollout(
        #     venv, policy, device, N_STEPS, action_low, action_high, GAMMA
        # )
        (obs_b, act_b, ret_b, val_b, logp_b, entropy,
         completed_rets, completed_lens, running_ep_ret_t, running_ep_len_t) = collect_rollout(
            venv, policy, device, N_STEPS, action_low, action_high, GAMMA,
            ep_ret_t=running_ep_ret_t, ep_len_t=running_ep_len_t
        )
         
        if len(completed_rets) > 0:
            ep_returns_hist.extend(completed_rets)
            ep_lens_hist.extend(completed_lens)
            mean_ret_100 = float(np.mean(ep_returns_hist[-100:]))
            mean_len_100 = float(np.mean(ep_lens_hist[-100:]))
            wandb.log({
                "ep_return": float(np.mean(completed_rets)),
                "ep_len": float(np.mean(completed_lens)),
                "mean_ep_return_100": mean_ret_100,
                "mean_ep_len_100": mean_len_100,
                "episodes_total": len(ep_returns_hist),
            }, step=update_idx)


        # forward again for value (so value grads flow)
        mu, std, value_t = policy(obs_b)
        value_t = value_t.squeeze(-1)
        dist_t = torch.distributions.Normal(mu, std)
        logp = dist_t.log_prob(act_b).sum(dim=-1)

        adv = (ret_b - value_t).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        loss_policy = -(logp * adv).mean()
        loss_value  = F.mse_loss(value_t, ret_b.detach())
        # loss_total  = loss_value + loss_policy - ENTROPY_BETA * entropy

        # value update
        optimizer_value.zero_grad()
        loss_value.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(value_params, 0.5)
        optimizer_value.step()

        # policy update
        optimizer_policy.zero_grad()
        (loss_policy - ENTROPY_BETA * entropy).backward()
        torch.nn.utils.clip_grad_norm_(policy_params, 0.5)
        optimizer_policy.step()

        wandb.log({
            "loss_policy": loss_policy.item(),
            "loss_value": loss_value.item(),
            "entropy": entropy.item(),
            "batch_adv_mean": adv.mean().item(),
            "batch_ret_mean": ret_b.mean().item(),
            "rollout_return_mean": ret_b.mean().item(),
        }, step=update_idx)
        
if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    main()