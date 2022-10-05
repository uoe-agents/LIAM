import torch
import pickle
import lbforaging
import lzma
from lbforaging.agents import H1, H2, H3, H4
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical

import torch.nn as nn
import gym
from os import listdir
from os.path import isfile, join


class PolicyNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, output_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, net_input):
        out = F.relu(self.fc1(net_input))
        out = F.relu(self.fc2(out))
        pol_out = F.softmax(self.policy(out), dim=-1)
        val_out = self.value(out)
        return pol_out, val_out


class FCNetwork(nn.Module):
    def __init__(self, dims, dropout=False):
        """
        Creates a network using ReLUs between layers and no activation at the end
        :param dims: tuple in the form of (100, 100, ..., 5). for dim sizes
        """
        super().__init__()
        input_size = dims[0]
        h_sizes = dims[1:]

        mods = [nn.Linear(input_size, h_sizes[0])]
        for i in range(len(h_sizes) - 1):
            mods.append(nn.ReLU())
            if dropout:
                mods.append(nn.Dropout(p=dropout))
            mods.append(nn.Linear(h_sizes[i], h_sizes[i + 1]))

        self.layers = nn.Sequential(*mods)

    @staticmethod
    def calc_layer_size(size, extra):
        if type(size) is int:
            return size
        return extra["size"]

    def forward(self, x):
        # Feedforward
        return self.layers(x)

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


path = "pretrained/maddpg"
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

agents = []

for filename in onlyfiles:
    with lzma.open(path + "/" + filename, "rb") as f:
        data = f.read()

    data = pickle.loads(data)
    best_timestep = data["rewards"][1:].sum(axis=1).argmax()
    print(filename, data["config"]["gym_env"], best_timestep)

    network_size = data["config"]["network_size"]
    env = gym.make(data["config"]["gym_env"])
    state_sizes = env.observation_space
    action_sizes = env.action_space
    agent_count = len(action_sizes)

    controlled_id = 0
    s, a = state_sizes[controlled_id], action_sizes[controlled_id]
    new_agent = FCNetwork((s.shape[0], *network_size, a.n), dropout=0.1)
    params = data["saved_networks"][best_timestep][controlled_id]
    new_agent.load_state_dict(params)

    agents += [new_agent]

agents = agents[:3] # only keep 6

path = "pretrained/ia2c"
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
for filename in onlyfiles:
    env = gym.make(data["config"]["gym_env"])
    state_sizes = env.observation_space[0].shape[0]
    action_sizes = env.action_space[0].n
    hidden_size = 64
    print(state_sizes, action_sizes)
    new_agent1 = PolicyNet(state_sizes, hidden_size, action_sizes)
    new_agent2 = PolicyNet(state_sizes, hidden_size, action_sizes)
    save_dict = torch.load(path + "/" + filename)['agent_params']
    new_agent1.load_state_dict(save_dict[0]['actor_critic'])
    new_agent2.load_state_dict(save_dict[1]['actor_critic'])

    agents += [new_agent1]
    agents += [new_agent2]

agents = agents[:6]
agents += [h(env.players[0]) for h in (H1, H2, H3, H4)]
AGENTS = agents

def onehot_from_logits(logits, epsilon=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if epsilon == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(
        torch.eye(logits.shape[1])[
            [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        ],
        requires_grad=False,
    )
    # chooses between best and random actions using epsilon greedy
    return torch.stack(
        [
            argmax_acs[i] if r > epsilon else rand_acs[i]
            for i, r in enumerate(torch.rand(logits.shape[0]))
        ]
    )

def pretrained_output(agent_id, input):
    # agent_id = 9
    if type(AGENTS[agent_id-1]) is FCNetwork:
        obs = input[0]
        act = onehot_from_logits(AGENTS[agent_id-1](obs), epsilon=0.1)
        act = act.argmax().detach().numpy()
        return act
    if type(AGENTS[agent_id-1]) is PolicyNet:
        obs = input[0]
        act = Categorical(AGENTS[agent_id-1](obs)[0]).sample()
        return act
    return AGENTS[agent_id-1]._step(input[1])
