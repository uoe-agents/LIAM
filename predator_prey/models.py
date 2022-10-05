import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


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


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        # create network layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.embedding = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        h, hidden = self.lstm(x, hidden)
        h = F.relu(self.fc1(h))
        embedding = self.embedding(h)
        return embedding, hidden


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim1)

        self.fc4 = nn.Linear(input_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim2)
        self.fc7 = nn.Linear(hidden_dim, output_dim2)
        self.fc8 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc2(h1))
        out = self.fc3(h1)

        h2 = F.relu(self.fc4(x))
        h2 = F.relu(self.fc5(h2))
        prob1 = F.softmax(self.fc6(h2), dim=-1)
        prob2 = F.softmax(self.fc7(h2), dim=-1)
        prob3 = F.softmax(self.fc8(h2), dim=-1)
        return out, prob1, prob2, prob3
