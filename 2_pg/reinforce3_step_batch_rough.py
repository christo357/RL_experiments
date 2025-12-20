""" 
Adding more features to the reinforce_batches:
    1. discounted return is calculated per 10 steps. 
    2. updation is done after steps instead of whole episodes.
    ie. batching is done based on steps rather than episode.
    
    3. added kl divergence in logging 
    4. added grad_l2 and grad_max in logging
"""


import gymnasium as gym 
import numpy as np
import pandas as pd
import torch  
import torch.nn as nn 
import wandb


HIDDEN_LAYER1  = 128
BATCH_SIZE = 8
GAMMA = 0.99
LR = 0.01

REWARD_STEP = 10

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
    
)
env = gym.make('CartPole-v1')
eval_env = gym.make('CartPole-v1', render_mode='rgb_array')



class PGNet(nn.Module):
    def __init__(self, input_size, fc, n_actions):
        super().__init__()
        self.input_size = input_size
        self.fc = fc 
        self.n_actions = n_actions
        
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.fc), 
            nn.ReLU(), 
            nn.Linear(self.fc, self.n_actions)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x
    
def experience_generator(env, policy, gamma, n_steps):
    
    while True:
        
        state, _ = env.reset()
        done = False
        state_list = []
        action_list = []
        reward_list = []
        done_list = []
        episode_reward = 0
        
        while not done: 
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            
            new_state, rew, term, trunc, info = env.step(action)
            done = term or trunc
            episode_reward += rew
            
            state_list.append(state_t)
            action_list.append(action)
            reward_list.append(rew)
            done_list.append(done)
            
            if len(reward_list)>=n_steps:
                ret = sum([reward_list[i]*(gamma**i) for i in range(n_steps)])
                
                yield {
                    'state':state_list[0], 
                    'action':action_list[0], 
                    'ret':ret, 
                    'done': False, 
                    'episode_reward': None, 
                }
                state_list.pop(0)
                action_list.pop(0)
                reward_list.pop(0)
                done_list.pop(0)
                
            state = new_state
            
                
        if done:
            while len(reward_list)>0:
                
                
                ret = sum([reward_list[i] * (gamma ** i) for i in range(len(reward_list))])
                is_last_transition = (len(reward_list) == 1)
                yield {
                    'state':  state_list[0],
                    'action': action_list[0],
                    'ret': ret,
                    'done': is_last_transition,
                    'episode_reward': episode_reward if is_last_transition else None,
                }
                
                state_list.pop(0)
                action_list.pop(0)
                reward_list.pop(0)
                done_list.pop(0)
 
    
    
policy = PGNet(
    env.observation_space.shape[0], 
    fc = HIDDEN_LAYER1, 
    n_actions= env.action_space.n
).to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr = LR, )

batch_states = []
batch_actions = []
batch_returns = []
total_rewards = []
reward_sum = 0
episode_reward = 0
episode_idx = 0
mean_reward = 0

exp_gen = experience_generator(env, policy, GAMMA, n_steps=REWARD_STEP)

for step_idx, exp in enumerate(exp_gen):
    reward_sum += exp['ret']
    baseline = reward_sum/(step_idx+1)
    
    batch_states.append(exp['state'])
    batch_actions.append(exp['action'])
    batch_returns.append(exp['ret'] - baseline)

    
    if exp['done']==True:
        episode_reward = exp['episode_reward']
        total_rewards.append(episode_reward)
        mean_reward = float(np.mean(total_rewards[-100:]))
        print(f"episode : {episode_idx} | step: {step_idx} | episode reward : {episode_reward} | mean reward/100 eps : {mean_reward}")
        wandb.log({
            "episode_reward": episode_reward, 
            "Mean reward/100 eps": mean_reward, 
            'episode': episode_idx, 
            "steps_per_episode": step_idx / max(episode_idx, 1)  # Sample efficiency metric
        }, step=step_idx)
        episode_idx += 1 # count the number of episodes
        
        if mean_reward>450:
            break
        
        
    # eval logging
    if (episode_idx%50==0 and episode_idx>0) :
        frames = []
        eval_state, _ = eval_env.reset()
        eval_done = False
        eval_reward = 0 
        
        while not eval_done:
            frames.append(eval_env.render())
            eval_state_tensor = torch.tensor(eval_state, dtype=torch.float32).unsqueeze(dim=0).to(device)
            with torch.no_grad():
                eval_logits = policy(eval_state_tensor)
                eval_action = torch.distributions.Categorical(logits=eval_logits).sample().item()
                
            eval_state, eval_rew, eval_term, eval_trunc, _ = eval_env.step(eval_action)
            eval_reward += eval_rew
            eval_done = eval_term or eval_trunc
            
        wandb.log({
            "eval_reward": eval_reward, 
            'video': wandb.Video(np.array(frames).transpose(0, 3,1,2), fps=30, format="mp4")
        }, step=step_idx)
        
    
    if len(batch_states)< BATCH_SIZE:
        continue
    
    states_t = torch.cat(batch_states, dim=0)
    actions_t = torch.tensor(batch_actions, dtype=torch.long, device=device)
    returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=device)
    

    
    logits_t = policy(states_t)
    dist_t = torch.distributions.Categorical(logits=logits_t)
    actions_prob_t = dist_t.log_prob(actions_t)
    
    loss_policy_t = -(returns_t * actions_prob_t).mean()
    optimizer.zero_grad()
    loss_policy_t.backward()
    optimizer.step()
    
    with torch.no_grad():
        old_logits = logits_t.detach()
        old_dist = torch.distributions.Categorical(logits=logits_t)
        
        new_logits = policy(states_t)
        new_dist = torch.distributions.Categorical(logits=new_logits)
        kl_div = torch.distributions.kl_divergence(old_dist, new_dist).mean()
    
    grad_max = 0.0
    grad_means = 0.0
    grad_count = 0
    for p in policy.parameters():
        grad_max = max(grad_max, p.grad.abs().max().item())
        grad_means += (p.grad ** 2).mean().sqrt().item()
        grad_count += 1
    
    wandb.log({
        'baseline':baseline,
        'loss_policy':loss_policy_t.item(),
        'kl_div' : kl_div.item(),
        'grad_l2':grad_means/grad_count,
        'grad_max':grad_max,
        'batch_scales': returns_t.mean(),
        "current_episode": episode_idx
    }, step = step_idx)
     
    batch_states.clear()
    batch_returns.clear()
    batch_actions.clear()
    
wandb.finish()
env.close()
eval_env.close()