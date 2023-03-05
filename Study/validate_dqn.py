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

n_obs = env.observation_space.shape[0]
n_act = env.action_space.n


class DQN(nn.Module):
    
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    
policy_net = DQN(n_obs,n_act)
policy_net.load_state_dict(torch.load('./dqn_actor.pth'))

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

def select_action(state):
    steps_done = 0
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():

            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]],  dtype=torch.long)
    
while True:
    state,_ = env.reset(seed = 543)
    torch.manual_seed(seed = 543)
    state = torch.from_numpy(state).float().unsqueeze(0)  
    for t in count():
        action = select_action(state)
        next_state, reward, done, _, _ = env.step(action.item())
        env.render()
        if done:
            break

# state,_ = env.reset()
# env.reset(seed = 543)
# torch.manual_seed(seed = 543)
# for t in range(1, 10000):
#     state = torch.from_numpy(state).float().unsqueeze(0)
#     action = select_action(state)
#     print(action)
#     state, reward, done, _, _ = env.step(action.item())
#     env.render()
#     if done:
#         break
