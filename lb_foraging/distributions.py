import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AddBias, init
from torch.distributions import Normal
"""
Modify standard PyTorch distributions so they are compatible with this code.
"""



class DiagGaussian(nn.Module):
    def __init__(self, mean, log_std):
        self.mean = mean
        self.std = log_std.exp()
        self.dist = Normal(self.mean, self.std)

    def sample(self):
        eps = Variable(torch.randn(*self.mean.size()))
        return self.mean + self.std * eps

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def entropy(self):
        return self.dist.entropy()
