import gymnasium as gym 
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
import os
import numpy as np
import pandas as pd
import typing as tt
import torch  
import torch.nn as nn 
import torch.nn.functional as F
import wandb

HIDDEN_LAYER1  = 128
ALPHA = 0.99
GAMMA = 0.9
# LR_policy = 1e-4
# LR_value = 1e-3
LR = 5e-4
N_STEPS = 10
ENTROPY_BETA = 0.01
ENTROPY_BETA_MIN = 0.001
entropy_smoothing_factor = 0.05
ENV_ID = 'InvertedPendulum-v5'
N_ENV = 1
BATCH_SIZE = 64


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
        "batch_size":  N_STEPS,
        "gamma": GAMMA,
        "entropy_beta_miin": ENTROPY_BETA_MIN,
        "entropy_smoothing_factor":entropy_smoothing_factor,
        'lr':LR,
        "entropy_beta":ENTROPY_BETA,
        "N_STEPS": N_STEPS
    }
    
)

env = gym.make(ENV_ID)
# env = NormalizeObservation(env)  # Normalizes observations to ~N(0,1)
# env = NormalizeReward(env, gamma=GAMMA,)  # Normalizes rewards
eval_env = gym.make(ENV_ID, render_mode='rgb_array')
# eval_env = NormalizeObservation(eval_env)  # Normalizes observations to ~N(0,1)
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
        
    def get_beta(self):
        return self.current_beta

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

def smooth(old: tt.Optional[float], val: float, alpha: float = 0.95,) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val

class PolicyNet(nn.Module):
    def __init__(self, input_size, fc, action_dim, action_low=-3,action_high=3, log_std_min=-20, log_std_max=1, ):
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
        # self.log_std = nn.Sequential(
        #     nn.Linear(self.fc, self.action_dim), 
        #     # nn.Softplus()
        # )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        
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
        mu = self.mu(x)
        # mu_normalized = torch.tanh(self.mu(x))  # Output in [-1, 1]
        # mu = mu_normalized * self.action_scale + self.action_bias  # Scale to action space

        
        # log_std = self.log_std()
        # std = torch.exp(torch.clamp(log_std, self.log_std_min, self.log_std_max))
        
        
        

        
        # std = (self.std_net(x)+1e-4).clamp(1e-3, 2.0)
        
         # Use learned constant std (common in simple continuous control)
        std = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))  # exp(-2)=0.135, exp(0.5)=1.65
        std = std.expand_as(mu)  # Broadcast to batch size
        
        
        v = self.critic_head(x)
        return mu, std, v
    
    
def ca_experience_generator(env, policy, gamma, n_steps, action_scale, action_bias):  #continuous actions experience generator
    while True: 
        state_list = []
        raw_action_list = []
        # action_list = []
        reward_list = []
        return_list = []
        done_list = []
        last_state_list = []
        
        done = False
        ep_rew = 0
        state, _ = env.reset()
        ep_steps =0
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device = device).unsqueeze(0)
            with torch.no_grad():
                mu_v, std_v,  value = policy(state_t)
            dist = torch.distributions.Normal(mu_v, std_v)
            # print(dist)
            u = dist.sample()
            # action = torch.clamp(u, low, high)
            a = torch.tanh(u)
            # # print(f'a:{a}')
            action = a * action_scale + action_bias
            # print(f'action:{action}')
            # print(action)
            action_env = action.squeeze(0).detach().cpu().numpy()
            new_state, rew, term, trunc, info = env.step(action_env)
            done = term or trunc
            ep_rew += rew
            state_list.append(state_t)
            # action_list.append(action)
            raw_action_list.append(u)
            reward_list.append(rew)
            done_list.append(done)
            
            last_state_list.append(new_state)
                
            if len(reward_list)>=n_steps:
                ret = sum([reward_list[i]* (gamma**i) for i in range(n_steps)])
                
                yield { 
                    'state':state_list[0], 
                    'raw_action':raw_action_list[0],
                    # 'action':action_list[0],
                    'ret':ret,
                    'done':term,
                    'last_state':last_state_list[n_steps-1] if not term else None, 
                    'ep_reward': ep_rew if done_list[0] else None, 
                    'reward_list':reward_list,
                    'ep_steps':None
                }
                
                state_list.pop(0)
                raw_action_list.pop(0)
                # action_list.pop(0)
                reward_list.pop(0)
                done_list.pop(0)
                last_state_list.pop(0)
                
            state = new_state
            ep_steps += 1
                
        else:
            while len(reward_list)>0:
                ret = sum([reward_list[i]* (gamma**i) for i in range(len(reward_list))])
                
                yield { 
                    'state':state_list[0], 
                    'raw_action':raw_action_list[0],
                    # 'action':action_list[0],
                    'ret':ret,
                    'done':term,
                    'last_state': None, 
                    'ep_reward': ep_rew if done_list[0] else None,
                    'reward_list':reward_list,
                    'ep_steps': ep_steps if done_list[0] else None,
                }
                
                state_list.pop(0)
                raw_action_list.pop(0)
                # action_list.pop(0)
                reward_list.pop(0)
                done_list.pop(0)
                last_state_list.pop(0)
                



policy = PolicyNet(
    env.observation_space.shape[0], 
    HIDDEN_LAYER1, 
    env.action_space.shape[0]
).to(device)
optimizer = torch.optim.Adam(policy.parameters(),lr=LR,)

# --- NEW: Initialize Beta Scheduler ---
# target_reward=950 matches your break condition
beta_scheduler = BetaScheduler(
    target_reward=950, 
    beta_start=ENTROPY_BETA, 
    beta_min=ENTROPY_BETA_MIN, 
    smoothing_factor=entropy_smoothing_factor
)
current_beta = ENTROPY_BETA # Default start value

total_updates = 20000  # number of gradient updates you expect (not env steps)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda upd: 1.0 - min(upd, total_updates) / total_updates
)

update_idx = 0


# policy_params = list(policy.net.parameters()) + list(policy.mu.parameters()) + list(policy.log_std.parameters())
# value_params = list(policy.critic_head.parameters())

# optimizer_policy = torch.optim.Adam(policy_params, lr=LR_policy)
# optimizer_value = torch.optim.Adam(value_params, lr=LR_value)  # 2x learning rate for value


batch_states = []
batch_returns = []
batch_raw_actions = []
# batch_actions=[]
batch_values = []
done_list = []
last_state_list = []
total_rewards = []
adv_smoothed = l_entropy = l_policy = l_value = l_total = None
episode_idx = 0
bootstrap_ema = None
mu_old = 0
# BATCH_SIZE = N_ENV * N_STEPS  # n_env * N_STEPSs

action_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
action_bias = (action_high + action_low) / 2
action_scale = (action_high - action_low) / 2


print("Recording initial video (before training)...")
initial_frames, initial_reward, initial_steps = record_video(eval_env, policy, device, low = action_low, high = action_high)
wandb.log({
    "video": wandb.Video(
        np.array(initial_frames).transpose(0, 3, 1, 2), 
        fps=30, 
        format="mp4",
        caption=f"Initial (untrained) - Reward: {initial_reward}, Steps: {initial_steps}"
    ),
    "initial_reward": initial_reward
}, step=0)
print(f"Initial reward: {initial_reward}, steps: {initial_steps}")




for step_idx, exp in enumerate(ca_experience_generator(env, policy, GAMMA, N_STEPS, action_scale, action_bias)):
    batch_states.append(exp['state']) 
    batch_raw_actions.append(exp['raw_action'])
    # batch_actions.append(exp['action'])
    done_list.append(exp['done'])
    ## bootstrapping if the episode is not completed withing N_STEPS
    if exp['last_state'] is not None:
        last_state = exp['last_state']
        last_state_t = torch.tensor(last_state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, bs_val = policy(last_state_t)
        bs_val = bs_val.item()
        # bootstrap_ema = bs_val if bootstrap_ema is None else ALPHA*bootstrap_ema + (1-ALPHA)*bs_val
        # ret = exp['ret'] + (GAMMA**N_STEPS) * bootstrap_ema
        ret = exp['ret'] +  (bs_val) * (GAMMA**N_STEPS) 
        batch_returns.append(ret)
    else:
        batch_returns.append(exp['ret'])
        
        
    if exp['ep_reward'] is not None:
        # --- NEW: Update Beta when episode finishes ---
        current_beta = beta_scheduler.update(exp['ep_reward'])
        episode_reward = exp['ep_reward']
        total_rewards.append(episode_reward)
        mean_reward = float(np.mean(total_rewards[-100:]))
        print(f"episode : {episode_idx} | step: {step_idx} | episode reward : {episode_reward} | mean reward/100 eps : {mean_reward}")
        wandb.log({
            "episode_reward": episode_reward, 
            "mean_reward_100": mean_reward,  
            "entropy_beta": current_beta,  # Log this to track decay!
            'episode_number': episode_idx,   
            "steps_per_episode": exp['ep_steps']
        }, step=step_idx)
        episode_idx += 1
        
        if mean_reward>950:
            save_path = os.path.join(wandb.run.dir, "policy_best.pt")
            torch.save(policy.state_dict(), save_path)
            wandb.log({"best_policy_path": save_path}, step=step_idx)
            print(f"Solved! Mean reward > 450 at episode {episode_idx}")
            break
        
        
    # eval logging - periodic videos
    if (episode_idx%10000==0 and episode_idx>0):
        print(f"Recording periodic video at episode {episode_idx}...")
        frames, eval_reward, eval_steps = record_video(eval_env, policy, device, low = action_low, high=action_high)
            
        wandb.log({
            "video": wandb.Video(
                np.array(frames).transpose(0, 3, 1, 2), 
                fps=30, 
                format="mp4",
                caption=f"Episode {episode_idx} - Reward: {eval_reward}, Steps: {eval_steps}, Mean100: {mean_reward:.1f}"
            ),
            "eval_reward": eval_reward
        }, step=step_idx)
        print(f"Eval reward: {eval_reward}, steps: {eval_steps}")
        
 


        
    if len(batch_states) < BATCH_SIZE:
        continue
    batch_states_t = torch.cat(batch_states, dim=0)
    batch_actions_t = torch.cat(batch_raw_actions, dim=0).to(device).float()  # each element in batch_raw_actions is [1, act_dim]
    batch_returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=device)
    # batch_u = torch.cat(batch_raw_actions, dim=0).to(device).float()
    # batch_a = torch.cat([a for u, a in batch_raw_actions], dim=0).to(device)
    
    mu_new, std, value_t = policy(batch_states_t)
    value_t = value_t.squeeze(-1)
    
    dist_t = torch.distributions.Normal(mu_new, std)
    
    # u_t = batch_actions_t                         # pre-tanh actions, [B, act_dim]
    # a_t = torch.tanh(u_t)
    
    

    logp_u = dist_t.log_prob(batch_actions_t).sum(dim=-1)     # [B]
    a_t = torch.tanh(batch_actions_t)
    log_prob_correction = torch.log(1.0 - a_t.pow(2) + 1e-6).sum(dim=-1)  # [B]
    logp = logp_u - log_prob_correction                       # [B]

    
    adv_t = (batch_returns_t - value_t).detach()
    adv = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    loss_policy = - (logp * adv).mean()
    
    # returns = adv + value_t.detach()
    # loss_value = F.mse_loss(value_t, batch_returns_t.detach())
    
    #huberloss
    delta = 1.0
    loss_value = F.smooth_l1_loss(value_t, batch_returns_t.detach(), beta=delta)
        
        
    entropy = dist_t.entropy().sum(dim=-1).mean()
    
    loss_total = loss_value + loss_policy - (current_beta*entropy)
    
    optimizer.zero_grad()
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()
    scheduler.step()
    update_idx += 1
    # optimizer_value.zero_grad()
    # loss_value.backward(retain_graph=True)
    # torch.nn.utils.clip_grad_norm_(value_params, max_norm=0.5)
    # optimizer_value.step()

    # optimizer_policy.zero_grad()
    # (loss_policy - ENTROPY_BETA * entropy).backward()
    # torch.nn.utils.clip_grad_norm_(policy_params, max_norm=0.5)
    # optimizer_policy.step()
    
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
                    float(np.mean(adv_t.mean().item()))
                )
    l_entropy = smooth(l_entropy, entropy.item())
    l_policy = smooth(l_policy, loss_policy.item())
    l_value = smooth(l_value, loss_value.item())
    l_total = smooth(l_total, loss_total.item())
    
    
    
    # break

    wandb.log({
        # 'baseline':baseline,
        'advantage':adv_smoothed,
        'entropy':entropy,
        'loss_policy':l_policy,
        'loss_value':l_value,
        'loss_entropy': l_entropy, 
        'loss_total': l_total,
        'kl div': kl_div.item(),
        "mu_delta": (mu_new - mu_old).abs().mean().item(),
        "std": std.mean().item(),
        "adv_abs": adv.abs().mean().item(),
        'grad_l2':grad_means/grad_count,
        'grad_max':grad_max,
        'batch_scales': batch_returns,
        "current_episode": episode_idx, 
        # 'saturation_fractions':(a_t.abs() > 0.99).float().mean().item()
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
    
    batch_raw_actions.clear()
    batch_returns.clear()
    batch_states.clear()
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
eval_env.close()