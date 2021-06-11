from models import *
from torch.distributions import OneHotCategorical
import torch.optim as optim
import torch.nn as nn

class A2C:
    def __init__(self, input_dim, hidden_dim, embedding_dim, act_dim, modelled_obs_dim, modelled_act_dim,
                 lr1, lr2, entropy_coef, value_loss_coef,  max_grad_norm):

        self.actor_critic = PolicyNet(input_dim + embedding_dim, hidden_dim, act_dim, act_dim)
        self.encoder = Encoder(input_dim + 2 * act_dim, hidden_dim, embedding_dim)
        self.decoder = Decoder(embedding_dim, embedding_dim + modelled_obs_dim, hidden_dim, modelled_obs_dim,
                               modelled_act_dim)

        self.optimizer1 = optim.Adam(list(self.actor_critic.parameters()), lr=lr1)
        self.optimizer2 = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr2)
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.embedding_dim = embedding_dim
        self.value_loss_coef = value_loss_coef

    def compute_embedding(self, obs, action, hidden):
        input_tensor = torch.cat((obs, action), dim=-1)
        embedding, hidden = self.encoder(input_tensor, hidden)
        return embedding, hidden

    def act(self, obs, action, hidden):
        embedding, hidden = self.compute_embedding(obs, action, hidden)
        input_tensor = torch.cat((obs.unsqueeze(0), embedding), dim=-1)
        pol_probs1, pol_probs2, value = self.actor_critic(input_tensor)

        m1 = OneHotCategorical(pol_probs1)
        action1 = m1.sample()
        m2 = OneHotCategorical(pol_probs2)
        action2 = m2.sample()
        action = torch.cat((action1, action2), dim=-1)

        return action, action1, action2, value, hidden

    def compute_value(self, obs, action, hidden):
        embedding, hidden = self.compute_embedding(obs, action, hidden)
        input_tensor = torch.cat((obs.unsqueeze(0), embedding), dim=-1)
        _, _, value = self.actor_critic(input_tensor)
        return value

    def evaluate(self, obs, actions, actions1, actions2, hidden, modelled_obs, modelled_act):

        embeddings, _ = self.compute_embedding(obs, actions, hidden)
        input_tensor = torch.cat((obs, embeddings.detach()), dim=-1)

        pol_probs1, pol_probs2, value = self.actor_critic(input_tensor)
        log_prob1 = torch.sum(torch.log(pol_probs1 + 1e-20) * actions1, dim=-1)
        log_prob2 = torch.sum(torch.log(pol_probs2 + 1e-20) * actions2, dim=-1)
        entropy = -torch.sum(pol_probs1 * torch.log(pol_probs1 + 1e-20), dim=-1).mean() - \
                  torch.sum(pol_probs2 * torch.log(pol_probs2 + 1e-20), dim=-1).mean()

        rec_loss1, rec_loss2 = self.eval_decoding(embeddings, modelled_obs, modelled_act)
        return log_prob1, log_prob2, entropy, value, rec_loss1.mean(), rec_loss2.mean()

    def eval_decoding(self, embeddings, modelled_obs, modelled_act):
        dec_input1 = embeddings
        dec_input2 = torch.cat((dec_input1, modelled_obs), dim=-1)
        out1, probs1, probs2 = self.decoder(embeddings, dec_input2)
        recon_loss1 = 0.5 * ((out1 - modelled_obs) ** 2).sum(-1)
        recon_loss2 = -torch.log(torch.sum(modelled_act[:, :, :5] * probs1, dim=-1))
        recon_loss2 -=torch.log(torch.sum(modelled_act[:, :, 5:] * probs2, dim=-1))
        return recon_loss1, recon_loss2

    def update(self, batches):
        for en, rollout in enumerate(batches):

            advantages = rollout.returns[:-1] - rollout.value_preds[:-1]

            obs_batch, actions_batch1, actions_batch2, value_pred_batch, return_batch, \
            hidden, modelled_obs, modelled_acts = rollout.sample()
            actions = torch.cat((actions_batch1[:-1].detach(), actions_batch2[:-1].detach()), dim=-1)

            log_probs1, log_probs2, entropy, values, recon_loss1, recon_loss2 = self.evaluate(obs_batch,
                                                                                              actions,
                                                                                              actions_batch1[1:],
                                                                                              actions_batch2[1:],
                                                                                              hidden,
                                                                                              modelled_obs,
                                                                                              modelled_acts)

            action_loss = -(advantages.detach() * (log_probs1.unsqueeze(-1) + log_probs2.unsqueeze(-1))).mean()

            value_loss1 = 0.5 * (return_batch - values).pow(2).mean()

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()

            (value_loss1 * self.value_loss_coef + action_loss - entropy * self.entropy_coef).backward()
            (recon_loss1 + recon_loss2).backward()

            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)

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
