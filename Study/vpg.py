import gym
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make('CartPole-v1')
env.reset(seed = 543)
torch.manual_seed(seed = 543)
n_obs = env.observation_space.shape[0]
n_act = env.action_space.n

class reinforce(nn.Module):    
    def __init__(self, n_observations, n_actions):
        super(reinforce, self).__init__()
        self.affine1  = nn.Linear(n_observations, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, n_actions)  
        self.saved_log_probs = []
        self.rewards = []
    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim = 1)  
    
policy = reinforce(n_obs,n_act)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()

def select_policy(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def update():
    R = 0
    gamma = 0.99
    policy_loss = []
    returns = deque()
    
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)
    
    returns = torch.tensor(returns)
    #print(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    
running_reward = 10
for i in count():
    state,_ = env.reset()
    ep_reward = 0
    for t in range(1, 10000):
        action = select_policy(state)
        state, reward, done, _, _ = env.step(action)
        #env.render()
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            break
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    update()
    if i % 10 == 0:
        print("Episode number:", i, "Prev. Reward:", ep_reward, "Ave. Reward:", running_reward)
        torch.save(policy.state_dict(), './vpg_actor.pth')
    if running_reward > env.spec.reward_threshold:
        env.close()
        print("Done!")
        break