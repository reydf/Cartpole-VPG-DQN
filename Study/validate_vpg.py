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

env = gym.make('CartPole-v1', render_mode = 'human')
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
policy.load_state_dict(torch.load('./vpg_actor.pth'))

def select_policy(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

while True:
    state,_ = env.reset()
    for t in range(1, 10000):
        action = select_policy(state)
        state, reward, done, _, _ = env.step(action)
        env.render()
        if done:
            break

# state,_ = env.reset()
# ep_reward = 0
# for t in range(1, 10000):
#     action = select_policy(state)
#     state, reward, done, _, _ = env.step(action)
#     env.render()
#     policy.rewards.append(reward)
#     ep_reward += reward
#     if done:
#         break
