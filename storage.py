import torch


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_dim, act_dim, hidden_dim, modelled_obs_dim,
                 modelled_act_dim):
        self.obs = torch.zeros(num_steps, num_processes, obs_dim)
        self.hidden = (torch.zeros((1, num_processes, hidden_dim)),
                       torch.zeros((1, num_processes, hidden_dim)))
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.actions1 = torch.zeros(num_steps + 1, num_processes, act_dim)
        self.actions2 = torch.zeros(num_steps + 1, num_processes, act_dim)
        self.dones = torch.ones(num_steps, num_processes, 1)

        self.modelled_obs = torch.zeros(num_steps, num_processes, modelled_obs_dim)
        self.modelled_acts = torch.zeros(num_steps, num_processes, modelled_act_dim)
        self.num_steps = num_steps
        self.step = 0

    def initialize(self, actions1, actions2, hidden):
        self.actions1[0].copy_(actions1)
        self.actions2[0].copy_(actions2)
        self.hidden = hidden

    def insert(self, obs, actions1, actions2, value_pred, reward, dones, modelled_obs, modelled_acts):
        self.obs[self.step].copy_(obs)
        self.actions1[self.step + 1].copy_(actions1)
        self.actions2[self.step + 1].copy_(actions2)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step].copy_(dones)
        self.modelled_obs[self.step].copy_(modelled_obs.squeeze(1))
        self.modelled_acts[self.step].copy_(modelled_acts.squeeze(1))
        self.step = (self.step + 1) % self.num_steps

    def compute_returns(self, gamma, gae_lambda, standardise):

        gae = 0
        value_pred = self.value_preds * torch.sqrt(standardise.var) + standardise.mean
        norm_rewards = self.rewards
        for step in reversed(range(self.num_steps)):
            delta = norm_rewards[step] + gamma * self.value_preds[step + 1] * (1. - self.dones[step]) - \
                    self.value_preds[step]
            gae = delta + gamma * gae_lambda * (1 - self.dones[step]) * gae
            self.returns[step] = gae + self.value_preds[step]

        standardise.update(self.returns[:-1])
        self.returns = (self.returns - standardise.mean) / torch.sqrt(standardise.var)

    def sample(self):
        return self.obs, self.actions1, self.actions2, self.value_preds[:-1].detach(), self.returns[:-1].detach(), \
               self.hidden, self.modelled_obs, self.modelled_acts
