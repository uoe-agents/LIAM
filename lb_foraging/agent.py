from models import *
import torch
from torch.distributions import OneHotCategorical
import torch.optim as optim
import torch.nn as nn
from misc import *


class A2C:
    def __init__(self, input_dim, hidden_dim1, embedding_dim, act_dim, modelled_agent_obs_dim, modelled_agent_act_dim,
                 lr1, lr2, entropy_coef, max_grad_norm):

        self.actor_critic = PolicyNet(input_dim + embedding_dim, hidden_dim1, act_dim)
        self.encoder = Encoder(input_dim + act_dim, hidden_dim1, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim1, modelled_agent_obs_dim, modelled_agent_act_dim)

        self.optimizer1 = optim.Adam(list(self.actor_critic.parameters()), lr=lr1)
        self.optimizer2 = optim.Adam(list(self.encoder.parameters()) +
                                     list(self.decoder.parameters()), lr=lr2)
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.embedding_dim = embedding_dim

    def compute_embedding(self, obs, action, hidden):
        input_tensor = torch.cat((obs, action), dim=-1)
        embedding, hidden = self.encoder(input_tensor, hidden)
        return embedding, hidden

    def act(self, obs, action, hidden):
        embedding, hidden = self.compute_embedding(obs, action, hidden)
        input_tensor = torch.cat((obs.unsqueeze(0), embedding), dim=-1)
        pol_probs, value = self.actor_critic(input_tensor)
        m = OneHotCategorical(pol_probs)
        action = m.sample()
        return action, value, hidden

    def compute_value(self, obs, action, hidden):
        embedding, hidden = self.compute_embedding(obs, action, hidden)
        input_tensor = torch.cat((obs.unsqueeze(0), embedding), dim=-1)
        _, value = self.actor_critic(input_tensor)
        return value

    def evaluate(self, obs, actions, hidden, modelled_agent_obs, modelled_agent_act):

        embeddings, _ = self.compute_embedding(obs, actions[:-1], hidden)
        input_tensor = torch.cat((obs, embeddings.detach()), dim=-1)
        pol_probs, value = self.actor_critic(input_tensor)
        log_prob = torch.sum(torch.log(pol_probs + 1e-20) * actions[1:], dim=-1)
        entropy = -torch.sum(pol_probs * torch.log(pol_probs + 1e-20), dim=-1).unsqueeze(-1)
        rec_loss1, rec_loss2 = self.eval_decoding(embeddings, modelled_agent_obs, modelled_agent_act)
        return log_prob, entropy, value, rec_loss1.unsqueeze(-1), rec_loss2.unsqueeze(-1)

    def eval_decoding(self, embeddings, modelled_agent_obs, modelled_agent_act):
        out, probs = self.decoder(embeddings)
        return 0.5 * ((modelled_agent_obs - out) ** 2).sum(-1), -torch.log(torch.sum(probs * modelled_agent_act, dim=-1))

    def update(self, batches):
        for rollout in batches:

            obs_batch, actions_batch, return_batch, dones_batch, hidden, modelled_agent_obs, \
            modelled_agent_act = rollout.sample()

            log_probs, entropy, values, recon_loss1, recon_loss2 = self.evaluate(obs_batch,
                                                                                 actions_batch,
                                                                                 hidden,
                                                                                 modelled_agent_obs,
                                                                                 modelled_agent_act)
            advantages = rollout.returns[:-1] - values.detach()
            action_loss = -(advantages.detach() * log_probs.unsqueeze(-1))
            value_loss = (return_batch - values).pow(2)

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss1 = torch.mean((1 - dones_batch.float()) * (value_loss + action_loss - entropy * self.entropy_coef))
            loss1.backward()
            loss2 = torch.mean((1 - dones_batch.float()) * (recon_loss1 + recon_loss2))
            loss2.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer1.step()
            self.optimizer2.step()

    def get_params(self):
        return {'actor_critic': self.actor_critic.state_dict(),
                'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict()}

    def load_params(self, params):
        self.actor_critic.load_state_dict(params['actor_critic'])
        self.encoder.load_state_dict(params['encoder'])
        self.decoder.load_state_dict(params['decoder'])

    def save_params(self, name):
        save_dict = {'agent_params': self.get_params()}
        torch.save(save_dict, 'trained_parameters/params_' + str(name) + '.pt')

    def init_from_save(self, filename):
        save_dict = torch.load(filename)
        self.load_params(save_dict['agent_params'])
