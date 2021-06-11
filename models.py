import torch.nn.functional as F
from utils import *


class PolicyNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.policy1 = nn.Linear(hidden_dim, output_dim1)
        self.policy2 = nn.Linear(hidden_dim, output_dim2)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, net_input):
        out = F.relu(self.fc1(net_input))
        out = F.relu(self.fc2(out))
        pol_out1 = F.softmax(self.policy1(out), dim=-1)
        pol_out2 = F.softmax(self.policy2(out), dim=-1)
        val_out = self.value(out)
        return pol_out1, pol_out2, val_out


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        # create network layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.m_z = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        h, hidden = self.lstm(x, hidden)
        h = F.relu(self.fc1(h))
        m_z = self.m_z(h)
        return m_z, hidden


class Decoder(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim1, output_dim2):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim1)

        self.fc4 = nn.Linear(input_dim2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim2)
        self.fc7 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, x1, x2):
        h1 = F.relu(self.fc1(x1))
        h1 = F.relu(self.fc2(h1))
        out1 = self.fc3(h1)

        h2 = F.relu(self.fc4(x2))
        h2 = F.relu(self.fc5(h2))
        probs1 = F.softmax(self.fc6(h2), dim=-1)
        probs2 = F.softmax(self.fc7(h2), dim=-1)

        return out1, probs1, probs2


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, norm_in=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # create network layers
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(self.in_fn(x)))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        return out


