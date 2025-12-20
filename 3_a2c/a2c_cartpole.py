import gymnasium as gym 
import numpy as np
import pandas as pd
import typing as tt
import torch  
import torch.nn as nn 
import torch.nn.functional as F
import wandb

HIDDEN_LAYER1  = 128
BATCH_SIZE = 8
GAMMA = 0.99
LR = 0.001

REWARD_STEP = 10
ENTROPY_BETA = 0.01
CLIP_GRAD = 0.1


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
        'env':'CartPole-v1',
        "algorithm": "a2c",
        "hidden_layer": HIDDEN_LAYER1,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "lr": LR,
        "reward_step": REWARD_STEP
    }
    
)

env = gym.make('CartPole-v1')
eval_env = gym.make('CartPole-v1', render_mode='rgb_array')

class PolicyNet(nn.Module):
    def __init__(self, input_size, fc, n_actions):
        super().__init__()
        self.input_size = input_size
        self.fc = fc 
        self.n_actions = n_actions
        
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.fc), 
            nn.ReLU(), 
            
        )
        
        self.policy_head = nn.Linear(self.fc, self.n_actions)
        
        self.critic_head = nn.Linear(self.fc, 1)
        
    def forward(self, x):
        x = self.net(x)
        return self.policy_head(x), self.critic_head(x)
    
    
def experience_generator(env, policy, gamma, n_steps):
    while True: 
        state_list = []
        action_list = []
        reward_list = []
        return_list = []
        done_list = []
        last_state_list = []
        
        done = False
        ep_rew = 0
        state, _ = env.reset()
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device = device).unsqueeze(0)
            logits, value = policy(state_t)
            # print(logits)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            # print(action)
            
            new_state, rew, term, trunc, info = env.step(action)
            done = term or trunc
            ep_rew += rew
            state_list.append(state_t)
            action_list.append(action)
            reward_list.append(rew)
            done_list.append(done)
            
            last_state_list.append(new_state)
                
            if len(reward_list)>=n_steps:
                ret = sum([reward_list[i]* (gamma**i) for i in range(n_steps)])
                
                yield { 
                    'state':state_list[0], 
                    'action':int(action_list[0]),
                    'ret':ret,
                    'done':done_list[0],
                    'last_state':last_state_list[n_steps-1] if not done else None, 
                    'ep_reward': None
                }
                
                state_list.pop(0)
                action_list.pop(0)
                reward_list.pop(0)
                done_list.pop(0)
                last_state_list.pop(0)
                
            state = new_state
                
        else:
            while len(reward_list)>0:
                ret = sum([reward_list[i]* (gamma**i) for i in range(len(reward_list))])
                
                yield { 
                    'state':state_list[0], 
                    'action':int(action_list[0]),
                    'ret':ret,
                    'done':done_list[0],
                    'last_state': None, 
                    'ep_reward': ep_rew if done_list[0] else None
                }
                
                state_list.pop(0)
                action_list.pop(0)
                reward_list.pop(0)
                done_list.pop(0)
                last_state_list.pop(0)
                
def record_video(env, policy, device, max_steps=500):
    """Record a single episode and return frames + reward"""
    frames = []
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < max_steps:
        frames.append(env.render())
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = policy(state_tensor)
            action = torch.distributions.Categorical(logits=logits).sample().item()
            
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
    return frames, total_reward, steps

def smooth(old: tt.Optional[float], val: float, alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val



policy = PolicyNet(
    env.observation_space.shape[0], 
    128, 
    env.action_space.n
).to(device)
optimizer = torch.optim.Adam(policy.parameters(),lr=LR, )


print("Recording initial video (before training)...")
initial_frames, initial_reward, initial_steps = record_video(eval_env, policy, device)
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


batch_states = []
batch_returns = []
batch_actions = []
batch_values = []
done_list = []
last_state_list = []
total_rewards = []
adv_smoothed = l_entropy = l_policy = l_value = l_total = None
episode_idx = 0

for step_idx, exp in enumerate(experience_generator(env, policy, GAMMA, REWARD_STEP)):
    batch_states.append(exp['state']) 
    batch_actions.append(exp['action'])
    
    ## bootstrapping if the episode is not completed withing REWARD_STEP
    if exp['last_state'] is not None:
        last_state = exp['last_state']
        last_state_t = torch.tensor(last_state, dtype=torch.float32, device=device).unsqueeze(0)
        _, bs_val = policy(last_state_t)
        ret = exp['ret'] +  (bs_val.item()) * (GAMMA**REWARD_STEP) 
        batch_returns.append(ret)
    else:
        batch_returns.append(exp['ret'])
        
        
    if exp['done'] :
        episode_reward = exp['ep_reward']
        total_rewards.append(episode_reward)
        mean_reward = float(np.mean(total_rewards[-100:]))
        print(f"episode : {episode_idx} | step: {step_idx} | episode reward : {episode_reward} | mean reward/100 eps : {mean_reward}")
        wandb.log({
            "episode_reward": episode_reward, 
            "mean_reward_100": mean_reward,  # FIXED: No spaces in metric name
            'episode_number': episode_idx,   # FIXED: More descriptive name
            "steps_per_episode": step_idx / max(episode_idx, 1)
        }, step=step_idx)
        episode_idx += 1
        
        if mean_reward>450:
            print(f"Solved! Mean reward > 450 at episode {episode_idx}")
            break
        
        
    # eval logging - periodic videos
    # if (episode_idx%1000==0 and episode_idx>0):
    #     print(f"Recording periodic video at episode {episode_idx}...")
    #     frames, eval_reward, eval_steps = record_video(eval_env, policy, device)
            
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
    batch_actions_t = torch.tensor(batch_actions, dtype=torch.long, device=device)
    batch_returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=device)
    
    logits_t, value_t = policy(batch_states_t)
    value_t = value_t.squeeze(-1)
    dist_t = torch.distributions.Categorical(logits=logits_t)
    actions_prob_t = dist_t.log_prob(batch_actions_t)
    
    loss_value = F.mse_loss(value_t, batch_returns_t.detach())
    
    
    adv_t = (batch_returns_t - value_t).detach()
    loss_policy = - (actions_prob_t * adv_t).mean()
    
    entropy = dist_t.entropy().mean()
    loss_entropy = - ENTROPY_BETA*entropy

    
    loss_total = loss_value + loss_policy + loss_entropy
    
    optimizer.zero_grad()
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=CLIP_GRAD)  # CRITICAL
    optimizer.step()
    
    
    with torch.no_grad():
        l_t, v_t = policy(batch_states_t)
        new_dist_t = torch.distributions.Categorical(logits=l_t)
        
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
    l_entropy = smooth(l_entropy, loss_entropy.item())
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
        "current_episode": episode_idx
    }, step = step_idx)
    
    batch_actions.clear()
    batch_returns.clear()
    batch_states.clear()
    
# NEW: Record final video (after training)
print("\nRecording final video (after training)...")
final_frames, final_reward, final_steps = record_video(eval_env, policy, device)
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