from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from models import *
from misc import *

class DDPG():

    def __init__(self, actor_input_dim, actor_output_dim, critic_input_dim, hidden_dim, lr):

        self.policy = MLP(actor_input_dim, actor_output_dim, hidden_dim)
        self.target_policy = MLP(actor_input_dim, actor_output_dim, hidden_dim)

        self.critic = MLP(critic_input_dim, 1, hidden_dim)
        self.target_critic = MLP(critic_input_dim, 1, hidden_dim)

        self.policy_optimizer = Adam(self.policy.parameters(), lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr)

    def step(self, obs, explore=False):
        self.policy.eval()
        action = self.policy(obs)
        if explore:
            action = gumbel_softmax(action, hard=True)
        else:
            action = onehot_from_logits(action)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])