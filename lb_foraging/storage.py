import torch


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, hidden_dim, obs_opp_shape, act_opp_shape):
        self.obs = torch.zeros(num_steps, num_processes, obs_shape)
        self.hidden = (torch.zeros((1, num_processes, hidden_dim)),
                       torch.zeros((1, num_processes, hidden_dim)))
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.actions = torch.zeros(num_steps + 1, num_processes, action_space)
        self.dones = torch.ones(num_steps, num_processes, 1)

        self.modelled_agent_obs = torch.zeros(num_steps, num_processes, obs_opp_shape)
        self.modelled_agent_act = torch.zeros(num_steps, num_processes, act_opp_shape)
        self.num_steps = num_steps
        self.step = 0

    def initialize(self, actions, hidden):
        self.actions[0].copy_(actions)
        self.hidden = hidden

    def insert(self, obs, actions, value_preds, rewards, dones, opp_o, opp_a):
        self.obs[self.step].copy_(obs)
        self.actions[self.step + 1].copy_(actions)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones)

        self.modelled_agent_obs[self.step].copy_(opp_o.squeeze(1))
        self.modelled_agent_act[self.step].copy_(opp_a.squeeze(1))

        self.step = (self.step + 1) % self.num_steps

    def compute_returns(self, gamma, gae_lambda, standardise):
        gae = 0
        value_pred = self.value_preds * torch.sqrt(standardise.var) + standardise.mean
        for step in reversed(range(self.num_steps)):
            delta = self.rewards[step] + gamma * value_pred[step + 1] * (1. - self.dones[step]) - \
                    value_pred[step]
            gae = delta + gamma * gae_lambda * (1 - self.dones[step]) * gae
            self.returns[step] = gae + value_pred[step]
        standardise.update(self.returns[:-1])
        self.returns = (self.returns - standardise.mean) / torch.sqrt(standardise.var)

    def sample(self):
        return self.obs, self.actions, self.returns[:-1].detach(), self.dones, self.hidden, \
               self.modelled_agent_obs, self.modelled_agent_act