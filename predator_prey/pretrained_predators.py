import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from misc import onehot_from_logits

def after_prey(obs):
    prey_obs = obs[12:14]
    if abs(prey_obs[0]) >  abs(prey_obs[1]):
        if prey_obs[0] >= 0:
            action = 1
        else:
            action = 2
    else:
        if prey_obs[1] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


def after_pred(obs):
    pred = random.choice([8, 10])
    pred_obs = obs[pred:pred+2]
    if abs(pred_obs[0]) >=  abs(pred_obs[1]):
        if pred_obs[0] >= 0:
            action = 1
        else:
            action = 2
    else:
        if pred_obs[1] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


def after_agent_far(obs):
    agent_obs = obs[8:14]
    index = np.argmax(np.abs(agent_obs))
    if index % 2 == 0:
        if agent_obs[index] >= 0:
            action = 1
        else:
            action = 2
    else:
        if agent_obs[index] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


def after_agent_close(obs):
    pred_obs1 = obs[8:10]
    pred_obs2 = obs[10:12]
    prey_obs = obs[12:14]
    s_pred1 = (pred_obs1**2).sum()
    s_pred2 = (pred_obs2**2).sum()
    s_prey = (prey_obs**2).sum()
    dists = np.array([s_pred1, s_pred2, s_prey])
    index = np.argmin(dists)
    if index == 0:
        agent_obs = pred_obs1
    elif index == 1:
        agent_obs = pred_obs2
    else:
        agent_obs = prey_obs

    if abs(agent_obs[0]) >=  abs(agent_obs[1]):
        if agent_obs[0] >= 0:
            action = 1
        else:
            action = 2
    else:
        if agent_obs[1] >= 0:
            action = 3
        else:
            action = 4
    one_hot_action = np.zeros(5)
    one_hot_action[action] = 1
    return one_hot_action


#~ def random_agent(obs):
    #~ action = random.choice(range(5))
    #~ one_hot_action = np.zeros(5)
    #~ one_hot_action[action] = 1
    #~ return one_hot_action


class IA2CPolicyNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IA2CPolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, output_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, net_input):
        out = F.relu(self.fc1(torch.Tensor(net_input)))
        out = F.relu(self.fc2(out))
        pol_out = F.softmax(self.policy(out), dim=-1)
        m = OneHotCategorical(pol_out)
        action = m.sample()
        return action.detach().numpy()


class MADDPGNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super(MADDPGNet, self).__init__()
        self.in_fn = nn.BatchNorm1d(input_dim)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(self.in_fn(torch.Tensor(x).unsqueeze(0))))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        action = onehot_from_logits(out)
        return action[0].detach().numpy()


ia2c_agents = [IA2CPolicyNet(16, 64, 5) for _ in range(3)]
for i in range(3):
    save_dict = torch.load('pretrained/params_ia2c_agent_' + str(i) + '.pt')
    ia2c_agents[i].load_state_dict(save_dict['agent_params']['actor_critic'])
maddpg_agents = [MADDPGNet(16, 5, 64) for _ in range(3)]
for i in range(3):
    save_dict = torch.load('pretrained/params_maddpg_agent_' + str(i) + '.pt')
    maddpg_agents[i].load_state_dict(save_dict['policy_params'])
    maddpg_agents[i].eval()

opp = [after_prey,  after_agent_far, after_agent_close, after_pred]
opp += ia2c_agents
opp += maddpg_agents

index = np.array([[1,  8,  0],
                   [2,  5,  8],
                   [3,  0,  1],
                   [8,  8, 9],
                   [6,  0,  1],
                   [7,  3,  4],
                   [7,  1,  2],
                   [3,  7,  3],
                   [9,  3,  5],
                   [0,  4,  6]])
pretrained_agents = []

for i in range(10):
    pretrained_agents.append([opp[index[i][0]], opp[index[i][1]], opp[index[i][2]]])


def get_opponent_actions(obs, task_id):
    actions = []
    for i, o in enumerate(obs):
        actions.append(pretrained_agents[task_id][i](o))
    return actions
