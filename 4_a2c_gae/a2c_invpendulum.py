import gymnasium as gym 
import numpy as np
import pandas as pd
import typing as tt
import torch  
import torch.nn as nn 
import torch.nn.functional as F
import wandb
from collections import deque
import os

from gymnasium.wrappers import NormalizeObservation, NormalizeReward

HIDDEN_LAYER1  = 256
# ALPHA = 0.95
GAMMA = 0.95 # DISCOUNT FACTOR
LAMBDA = 0.95 # FOR GAE
LR = 3e-4
# N_STEPS = 20
ENV_ID = 'InvertedPendulum-v5'
N_ENV = 1
BATCH_SIZE = 64

ENTROPY_BETA = 0.01
ENTROPY_BETA_MIN = 1e-5
entropy_smoothing_factor = 0.05
total_updates = 500000


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu' 
print(f'Using device : {device}')

run = wandb.init(
    entity=None,
    project='RL_diary', 
    config={
        'env':ENV_ID,
        "algorithm": "a2c",
        "hidden_layer": HIDDEN_LAYER1,
        "batch_size":  BATCH_SIZE,
        "gamma": GAMMA,
        "entropy_beta_min": ENTROPY_BETA_MIN,
        # "entropy_smoothing_factor":entropy_smoothing_factor,
        'lr':LR,
        "entropy_beta":ENTROPY_BETA,
        # "N_STEPS": N_STEPS
    }
    
)

env = gym.make(ENV_ID)
eval_env = gym.make(ENV_ID, render_mode='rgb_array')



    
def smooth(old: tt.Optional[float], val: float, alpha: float = 0.95,) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val    

def record_video(env, policy, device, low, high, max_steps=500, ):
    """Record a single episode and return frames + reward"""
    frames = []
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    frame = env.render()
    if frame is not None:
        frames.append(np.array(frame, copy=True))
    while not done and steps < max_steps:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu, std, _ = policy(state_tensor)
        dist = torch.distributions.Normal(mu, std)
        action = torch.clamp(dist.sample(), low, high)

        state, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        total_reward += reward
        done = terminated or truncated
        steps += 1

        frame = env.render()
        if frame is not None:
            frames.append(np.array(frame, copy=True))
        
    return frames, total_reward, steps


def compute_gae(deltas, dones, gamma, lam):
    deltas_t = torch.tensor(deltas, dtype=torch.float32)
    dones_t = torch.tensor(dones, dtype=torch.float32)

    mask = 1.0 - dones_t

    T = deltas_t.shape[0]
    adv = torch.zeros_like(deltas_t)
    gae = 0.0

    for t in reversed(range(T)):
        gae = deltas_t[t] + gamma * lam * mask[t] * gae
        adv[t] = gae

    return adv


class PolicyNet(nn.Module):
    def __init__(self, input_size, fc, action_dim, log_std_min, log_std_max):
        super().__init__()
        self.input_size = input_size
        self.fc = fc
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.fc), 
            nn.ReLU(), 
            nn.Linear(self.fc, self.fc), 
            nn.ReLU()
        )
        
        self.mu = nn.Linear(self.fc, self.action_dim)
        
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        
        self.critic_head = nn.Linear(self.fc, 1)
        
    def forward(self, x):
        x = self.net(x)
        
        mu = self.mu(x)
        std = torch.exp(torch.clamp(self.log_std, self.log_std_min, self.log_std_max))
        std = std.expand_as(mu)
        
        val = self.critic_head(x)
        return mu, std, val
    
class LinearBetaScheduler:
    def __init__(self, beta_start, beta_end, total_steps):
        self.start = beta_start
        self.end = beta_end
        self.total_steps = total_steps

    def update(self, current_step):
        # Linearly decay beta based on step count
        frac = min(1.0, current_step / self.total_steps)
        return self.start + frac * (self.end - self.start)
    
class BetaScheduler:
    def __init__(self, target_reward, beta_start, beta_min=1e-4, smoothing_factor=0.01):
        self.target = target_reward
        self.start = beta_start
        self.min = beta_min
        self.alpha = smoothing_factor
        self.ema_reward = None  # Exponential Moving Average of Reward
        self.current_beta = beta_start

    def update(self, reward):
        # 1. Update EMA of Reward
        if self.ema_reward is None:
            self.ema_reward = reward
        else:
            self.ema_reward = (self.ema_reward * (1 - self.alpha)) + (reward * self.alpha)
        
        # 2. Calculate Progress (0.0 to 1.0) based on EMA
        # If ema_reward is negative, treat progress as 0
        progress = max(0.0, min(1.0, self.ema_reward / self.target))
        
        # 3. Decay Beta linearly with progress
        self.current_beta = self.start * (1.0 - progress)
        
        # 4. Clamp to minimum
        self.current_beta = max(self.current_beta, self.min)
        
        return self.current_beta

class NStepCollector:
    def __init__(self, env, policy, gamma, lam, batch_size,action_low, action_high,  device):
        # super().__init__(self,)
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size
        self.device = device
        
        self.ep_reward = 0
        
        self.state, _ = env.reset()
        self.action_bias = (action_high + action_low) / 2
        self.action_scale = (action_high - action_low) / 2
                
        self.states = []
        self.raw_actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.deltas = []
        self.values = []         
          
    def rollout(self):
        finished_episode_rewards = []
        while True:
            # print(f"state: {self.state}")
            
            state_t = torch.tensor(self.state, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                mu, std, _ = self.policy(state_t)
            
            # print('mu', mu)
            # print('std', std)
            dist = torch.distributions.Normal(mu,std)
            u = dist.sample()
            a = torch.tanh(u)
            action = a*self.action_scale + self.action_bias
            action_env = action.squeeze(0).detach().cpu().numpy()
            
            next_state, rew, term, trunc, info = self.env.step(action_env)
            # self.next_state_t = torch.Tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            self.ep_reward += rew
            done = term or trunc
            
            if done:
                finished_episode_rewards.append(self.ep_reward)
                self.state, _ = self.env.reset()
                self.ep_reward = 0
            else:
                self.state = next_state
            
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            if not term:
                with torch.no_grad():
                    _, _ , v_t = self.policy(state_t)
                    _, _, v_t1 = self.policy(next_state_t)
                v_t = v_t.item()
                v_t1 = v_t1.item()
            else:
                with torch.no_grad():
                    _, _ , v_t = self.policy(state_t)
                v_t = v_t.item()
                v_t1 = 0
            
            
            delta = rew + self.gamma*v_t1 - v_t
            
            self.states.append(state_t)
            self.rewards.append(rew)
            self.dones.append(done)
            self.next_states.append(next_state)
            self.deltas.append(delta)
            self.raw_actions.append(u)
            self.values.append(v_t)
            if len(self.states)>=self.batch_size:
                yield {
                        'states':list(self.states), 
                        'actions':list(self.raw_actions), 
                        'done':list(self.dones), 
                        'deltas':list(self.deltas),
                        'ep_rewards': finished_episode_rewards,
                        'values':list(self.values), 
                }
                
                self.states.clear()
                self.rewards.clear()
                self.dones.clear()
                self.next_states.clear()
                self.deltas.clear()
                self.raw_actions.clear()
                self.values.clear()
                finished_episode_rewards = []
            
            else: 
                yield None
                
            self.state = next_state
            if term or trunc:
                # print("reset")
                self.state, _ = self.env.reset()
                self.ep_reward = 0
            
        
policy = PolicyNet(
    env.observation_space.shape[0], 
    HIDDEN_LAYER1, 
    env.action_space.shape[0], 
    log_std_min=-20, 
    log_std_max=1,
).to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr = LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5, 
    patience=500,  # If reward doesn't go up for 500 steps, lower LR
)

current_beta = ENTROPY_BETA
beta_scheduler = LinearBetaScheduler(
    beta_start=ENTROPY_BETA, 
    beta_end=ENTROPY_BETA_MIN, 
    total_steps=total_updates   # Decay fully in the first 33% of training
)
# beta_scheduler = BetaScheduler(
#     target_reward=950, 
#     beta_start=ENTROPY_BETA, 
#     beta_min=ENTROPY_BETA_MIN, 
#     smoothing_factor=entropy_smoothing_factor
# )

action_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
exp_collector = NStepCollector(env, policy, GAMMA, LAMBDA, BATCH_SIZE,action_low, action_high, device)
total_rewards = []
episode_idx = 0
mu_old = 0
adv_smoothed = None
l_entropy = None
l_policy = None
l_value = None
l_total = None
mean_reward = 0.0
solved = False


# print("Recording initial video (before training)...")
# initial_frames, initial_reward, initial_steps = record_video(eval_env, policy, device, low = action_low, high = action_high)
# wandb.log({
#     "video": wandb.Video(
#         np.array(initial_frames).transpose(0, 3, 1, 2), 
#         fps=30, 
#         format="mp4",
#         caption=f"Initial (untrained) - Reward: {initial_reward}, Steps: {initial_steps}"
#     ),
#     "initial_reward": initial_reward
# }, step=0)
# print(f"Initial reward: {initial_reward}, steps: {initial_steps}")


for step_idx, exp in enumerate(exp_collector.rollout()):
    # exp = exp_collector.rollout()
    # print(exp)
    if exp is None:
        continue
    
    if exp['ep_rewards'] is not None:
        for ep_rew in exp['ep_rewards']:
            # Update Beta / Logger for EACH episode found
            current_beta = beta_scheduler.update(ep_rew)
            total_rewards.append(ep_rew)
            mean_reward = float(np.mean(total_rewards[-100:]))
            
            print(f"Episode: {episode_idx} |Steps: {step_idx} | Reward: {ep_rew} | Mean: {mean_reward:.1f}")
            
            wandb.log({
                "episode_reward": ep_rew, 
                "mean_reward_100": mean_reward,
                "episode_number": episode_idx
            }, step=step_idx)
            
            episode_idx += 1
            
            
        
            if mean_reward>950:
                save_path = os.path.join(wandb.run.dir, "policy_best.pt")
                torch.save(policy.state_dict(), save_path)
                wandb.log({"best_policy_path": save_path}, step=step_idx)
                print(f"Solved! Mean reward > 950 at episode {episode_idx}")
                solved = True
                break
        if solved: 
            break
    
    states_list = exp['states']
    raw_actions_list = exp['actions']
    done_list = exp['done']
    deltas_list = exp['deltas']
    values_list = exp['values']
    
    batch_adv_t = compute_gae(deltas_list, done_list, GAMMA, LAMBDA).to(device)
    
    
    batch_states_t = torch.cat(states_list, dim =0)
    batch_actions_t = torch.cat(raw_actions_list, dim=0)
    batch_value = torch.tensor(values_list,dtype = torch.float32, device=device)

    # batch_adv_t = torch.tensor(adv_list, dtype = torch.float32, device=device)
    
    
    mu_new, std, value = policy(batch_states_t)
    value_t = value.squeeze(dim=1)
    returns = batch_adv_t + batch_value
    # loss_value = F.mse_loss(value_t, returns.detach())
    #huberloss
    delta = 1.0
    loss_value = F.smooth_l1_loss(value_t,returns.detach(), beta=delta)
      
    
    dist_t = torch.distributions.Normal(mu_new, std)
    logp_u = dist_t.log_prob(batch_actions_t).sum(dim=-1)
    a_t = torch.tanh(batch_actions_t)
    logp_correction = torch.log(( 1 - a_t.pow(2))+1e-6).sum(dim=-1)
    logp = logp_u - logp_correction
    
    
    batch_adv_t = (batch_adv_t - batch_adv_t.mean())/(batch_adv_t.std() + 1e-8) # normalize adv_t after returns

    loss_policy = -(logp * batch_adv_t.detach()).mean()
    
    
    
    entropy = dist_t.entropy().sum(dim=-1).mean()
    
    loss_total = loss_value + loss_policy - current_beta*entropy
    
    optimizer.zero_grad()
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()
    scheduler.step(mean_reward)
    
    
    
    with torch.no_grad():
        
        mu_t, std_t, v_t = policy(batch_states_t)
        new_dist_t = torch.distributions.Normal(mu_t, std_t)
        
        kl_div = torch.distributions.kl_divergence(dist_t, new_dist_t).mean()
        
    grad_max = 0.0
    grad_means = 0.0
    grad_count = 0
    for p in policy.parameters():

        grad_max = max(grad_max, p.grad.abs().max().item())
        grad_means += (p.grad ** 2).mean().sqrt().item()
        grad_count += 1
        
        
    adv_smoothed = smooth(
                    adv_smoothed,
                    float(np.mean(batch_adv_t.abs().mean().item()))
                )
    l_entropy = smooth(l_entropy, entropy.item())
    l_policy = smooth(l_policy, loss_policy.item())
    l_value = smooth(l_value, loss_value.item())
    l_total = smooth(l_total, loss_total.item())
    
    
    
    # break
    # print(f"Episode: {episode_idx} |Steps: {step_idx} | Reward: {ep_rew} | Mean: {mean_reward:.1f}")
    wandb.log({
        # 'baseline':baseline,
        'entropy_beta':current_beta,
        'advantage':adv_smoothed,
        'entropy':entropy,
        'loss_policy':l_policy,
        'loss_value':l_value,
        'loss_entropy': l_entropy, 
        'loss_total': l_total,
        'kl div': kl_div.item(),
        "mu_delta": (mu_new - mu_old).abs().mean().item(),
        "std": std.mean().item(),
        "adv_abs": batch_adv_t.abs().mean().item(),
        'grad_l2':grad_means/grad_count if grad_count else 0.0,
        'grad_max':grad_max,
        'batch_returns': returns,
        "current_episode": episode_idx, 
        'saturation_fractions':(a_t.abs() > 0.99).float().mean().item(),
        'action_mean': batch_actions_t.mean().item(),
        'action_std': batch_actions_t.std().item(),
        'action_clamp_rate': (
            ((batch_actions_t <= action_low + 0.01).any(dim=-1) | 
            (batch_actions_t >= action_high - 0.01).any(dim=-1))
            .float().mean().item()
        ),
        'mu_mean': mu_new.mean().item(),
        'mu_std': mu_new.std().item(),
        'policy_std_mean': std.mean().item(),
    }, step = step_idx)
    
    # batch_raw_actions.clear()
    # batch_returns.clear()
    # batch_states.clear()
    mu_old = mu_new
   

    
# NEW: Record final video (after training)
print("\nRecording final video (after training)...")
final_frames, final_reward, final_steps = record_video(eval_env, policy, device, low=action_low, high =action_high)
wandb.log({
    "video": wandb.Video(
        np.array(final_frames).transpose(0, 3, 1, 2), 
        fps=30, 
        format="mp4",
        caption=f"Final (trained) - Reward: {final_reward}, Steps: {final_steps}, Episodes: {episode_idx}"
    ),
    "final_reward": final_reward
}, step=step_idx)
print(f"Final reward: {final_reward}, steps: {final_steps}")

print(f"\nTraining complete!")
print(f"Total episodes: {episode_idx}")
print(f"Total steps: {step_idx}")
print(f"Final mean reward: {mean_reward}")
    
wandb.finish()
env.close()
# eval_env.close()
