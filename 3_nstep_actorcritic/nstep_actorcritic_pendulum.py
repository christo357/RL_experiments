import gymnasium as gym 
from gymnasium.wrappers import NormalizeObservation, NormalizeReward

import numpy as np
import pandas as pd
import typing as tt
import torch  
import torch.nn as nn 
import torch.nn.functional as F
import wandb

HIDDEN_LAYER1  = 64

GAMMA = 0.99
LR = 5e-3

N_STEPS = 20
ENTROPY_BETA = 0.0
ENV_ID = 'Pendulum-v1'
N_ENV = 1
BATCH_SIZE = N_STEPS


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
        "lr": LR,
        "entropy_beta":ENTROPY_BETA,
        "N_STEPS": N_STEPS
    }
    
)

env = gym.make(ENV_ID)
env = NormalizeObservation(env)  # Normalizes observations to ~N(0,1)
# env = NormalizeReward(env, gamma=GAMMA,)  # Normalizes rewards
eval_env = gym.make(ENV_ID, render_mode='rgb_array')
eval_env = NormalizeObservation(eval_env)  # Normalizes observations to ~N(0,1)
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

def smooth(old: tt.Optional[float], val: float, alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val

class PolicyNet(nn.Module):
    def __init__(self, input_size, fc, action_dim):
        super().__init__()
        self.input_size = input_size
        self.fc = fc 
        self.action_dim = action_dim
        
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
        # self.std_net = nn.Sequential(
        #     nn.Linear(self.fc, self.action_dim), 
        #     # nn.Softplus()
        # )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        
        self.critic_head = nn.Linear(self.fc, 1)
        
    def forward(self, x):
        x = self.net(x)
        mu = self.mu(x)
        
        # std = (self.std_net(x)+1e-4).clamp(1e-3, 2.0)
         # Use learned constant std (common in simple continuous control)
        std = torch.exp(self.log_std.clamp(-2, 0.5))  # exp(-2)=0.135, exp(0.5)=1.65
        std = std.expand_as(mu)  # Broadcast to batch size
        
        
        v = self.critic_head(x)
        return mu, std, v
    
    
def ca_experience_generator(env, policy, gamma, n_steps, low, high):  #continuous actions experience generator
    while True: 
        state_list = []
        raw_action_list = []
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
            a = torch.tanh(u)
            # print(f'a:{a}')
            action = low + (a+1) * (high-low)*0.5
            # print(f'action:{action}')
            # print(action)
            action_env = action.squeeze(0).detach().cpu().numpy()
            new_state, rew, term, trunc, info = env.step(action_env)
            done = term or trunc
            ep_rew += rew
            state_list.append(state_t)
            raw_action_list.append(u)
            reward_list.append(rew)
            done_list.append(done)
            
            last_state_list.append(new_state)
                
            if len(reward_list)>=n_steps:
                ret = sum([reward_list[i]* (gamma**i) for i in range(n_steps)])
                
                yield { 
                    'state':state_list[0], 
                    'raw_action':raw_action_list[0],
                    'ret':ret,
                    'done':done,
                    'last_state':last_state_list[n_steps-1] if not done else None, 
                    'ep_reward': None, 
                    'reward_list':reward_list,
                    'ep_steps':None
                }
                
                state_list.pop(0)
                raw_action_list.pop(0)
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
                    'ret':ret,
                    'done':done,
                    'last_state': None, 
                    'ep_reward': ep_rew if done_list[0] else None,
                    'reward_list':reward_list,
                    'ep_steps': ep_steps if done_list[0] else None,
                }
                
                state_list.pop(0)
                raw_action_list.pop(0)
                reward_list.pop(0)
                done_list.pop(0)
                last_state_list.pop(0)
                



policy = PolicyNet(
    env.observation_space.shape[0], 
    HIDDEN_LAYER1, 
    env.action_space.shape[0]
).to(device)
optimizer = torch.optim.Adam(policy.parameters(),lr=LR,)

batch_states = []
batch_returns = []
batch_raw_actions = []
batch_values = []
done_list = []
last_state_list = []
total_rewards = []
adv_smoothed = l_entropy = l_policy = l_value = l_total = None
episode_idx = 0
# BATCH_SIZE = N_ENV * N_STEPS  # n_env * N_STEPSs

actions_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
actions_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)


for step_idx, exp in enumerate(ca_experience_generator(env, policy, GAMMA, N_STEPS, actions_low, actions_high)):
    batch_states.append(exp['state']) 
    batch_raw_actions.append(exp['raw_action'])
    ## bootstrapping if the episode is not completed withing N_STEPS
    if exp['last_state'] is not None:
        last_state = exp['last_state']
        last_state_t = torch.tensor(last_state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, bs_val = policy(last_state_t)
        bs_val = bs_val.item()
        ret = exp['ret'] +  (bs_val) * (GAMMA**N_STEPS) 
        batch_returns.append(ret)
    else:
        batch_returns.append(exp['ret'])
        
        
    if exp['ep_reward'] is not None:
        episode_reward = exp['ep_reward']
        total_rewards.append(episode_reward)
        mean_reward = float(np.mean(total_rewards[-100:]))
        print(f"episode : {episode_idx} | step: {step_idx} | episode reward : {episode_reward} | mean reward/100 eps : {mean_reward}")
        wandb.log({
            "episode_reward": episode_reward, 
            "mean_reward_100": mean_reward,  
            'episode_number': episode_idx,   
            "steps_per_episode": exp['ep_steps']
        }, step=step_idx)
        episode_idx += 1
        
        if mean_reward>-200:
            print(f"Solved! Mean reward > -200 at episode {episode_idx}")
            break
        
        
    # eval logging - periodic videos
    # if (episode_idx%1000==0 and episode_idx>0):
    #     print(f"Recording periodic video at episode {episode_idx}...")
    #     frames, eval_reward, eval_steps = record_video(eval_env, policy, device, low = actions_low, high=actions_high)
            
    #     wandb.log({
    #         "video": wandb.Video(
    #             np.array(frames).transpose(0, 3, 1, 2), 
    #             fps=30, 
    #             format="mp4",
    #             caption=f"Episode {episode_idx} - Reward: {eval_reward}, Steps: {eval_steps}, Mean100: {mean_reward:.1f}"
    #         ),
    #         "eval_reward": eval_reward
    #     }, step=step_idx)
    #     print(f"Eval reward: {eval_reward}, steps: {eval_steps}")
        
        
        
    if len(batch_states) < BATCH_SIZE:
        continue
    batch_states_t = torch.cat(batch_states, dim=0)
    batch_actions_t = torch.cat(batch_raw_actions, dim=0).to(device).float()  # each element in batch_raw_actions is [1, act_dim]
    batch_returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=device)
    # batch_u = torch.cat(batch_raw_actions, dim=0).to(device).float()
    # batch_a = torch.cat([a for u, a in batch_raw_actions], dim=0).to(device)
    
    mu, std, value_t = policy(batch_states_t)
    value_t = value_t.squeeze(-1)
    
    dist_t = torch.distributions.Normal(mu, std)
    
    # u_t = batch_actions_t                         # pre-tanh actions, [B, act_dim]
    # a_t = torch.tanh(u_t)
    
    

    logp_u = dist_t.log_prob(batch_actions_t).sum(dim=-1)     # [B]
    a_t = torch.tanh(batch_actions_t)
    log_prob_correction = torch.log(1.0 - a_t.pow(2) + 1e-6).sum(dim=-1)  # [B]
    logp = logp_u - log_prob_correction                       # [B]

    
    adv_t = (batch_returns_t - value_t).detach()
    loss_policy = - (logp * adv_t).mean()
    
    returns = adv_t + value_t.detach()
    loss_value = F.mse_loss(value_t, batch_returns_t.detach())
        
        
    entropy = dist_t.entropy().sum(dim=-1).mean()
    
    loss_total = loss_value + loss_policy - (ENTROPY_BETA*entropy)
    
    optimizer.zero_grad()
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()
    
    
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
        'grad_l2':grad_means/grad_count,
        'grad_max':grad_max,
        'batch_scales': batch_returns,
        "current_episode": episode_idx, 
        'saturation_fractions':(a_t.abs() > 0.99).float().mean().item()
    }, step = step_idx)
    
    batch_raw_actions.clear()
    batch_returns.clear()
    batch_states.clear()
    
