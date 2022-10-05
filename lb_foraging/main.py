import copy
import glob
import os
import time
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent import A2C
from storage import RolloutStorage
from envs import DummyVecEnv
from utils import *
import pickle
import uuid
import lbforaging
from pretrained_output import pretrained_output
from standardise_stream import RunningMeanStd

def make_env(scenario_name):
    env = gym.make(scenario_name)
    return env


def main(args):
    results = []
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    standardise = RunningMeanStd(shape=1)
    torch.set_num_threads(1)

    env = [make_env(args['env']) for _ in range(args['num_processes'])]
    for i in range(args['num_processes']):
        env[i].seed((args['seed'] + 1) * i)

    envs = DummyVecEnv(env)
    obs_dim = envs.observation_space[0].shape[0]
    act_dim = 6
    name = uuid.uuid4()
    try:
        os.mkdir('results')
    except:
        pass

    agent = A2C(obs_dim,
                args['hidden_dim1'],
                args['embedding_dim'],
                args['action_dim'], obs_dim, act_dim,
                args['lr1'],
                args['lr2'],
                args['entropy_coef'],
                args['max_grad_norm'])

    num_batches = args['episode_length'] // args['batch_size']
    num_updates = int(args['num_env_steps'] // num_batches // args['batch_size'] // args['num_processes'])
    episode_passed = -1
    dones = torch.ones(args['num_processes'], 1)
    average_reward = torch.zeros(args['num_processes'], 1)
    print('Number of updates that will be performed ' + str(num_updates))
    for j in range(num_updates):
        batches = [RolloutStorage(args['batch_size'], args['num_processes'], obs_dim,
                                  act_dim, args['hidden_dim1'], obs_dim, act_dim)
                   for _ in range(num_batches)]
        for step in range(num_batches):
            if np.all(dones.detach().numpy()):
                counter = 0

                obs = envs.reset()
                agent_obs = torch.FloatTensor([o[1][2] for o in obs])
                modelled_agent_obs = [torch.FloatTensor(o[0][1]).unsqueeze(0) for o in obs]
                # we need that for the heuristic agents
                true_modelled_agent_obs = [o[0][0] for o in obs]
                dones = torch.zeros((args['num_processes'], 1))
                actions = torch.zeros((args['num_processes'], args['action_dim']))
                episode_passed += 1
                average_reward = torch.zeros(args['num_processes'], 1)

                if episode_passed % args['update_episode'] == 0:
                    tasks = np.random.choice(range(args['num_tasks']), size=args['num_processes'])
                    hidden = (torch.zeros((1, args['num_processes'], args['hidden_dim1'])),
                              torch.zeros((1, args['num_processes'], args['hidden_dim1'])))
            batches[step].initialize(actions, hidden)

            for i in range(args['batch_size']):
                with torch.no_grad():
                    actions, value, hidden = agent.act(agent_obs, actions, hidden)
                counter += 1
                actions = actions[0]
                value = value[0]
                modelled_agent_actions = [pretrained_output(t, (o, to)) for t, o, to in zip(tasks, modelled_agent_obs,
                                                                                            true_modelled_agent_obs)]
                modelled_agent_actions = [F.one_hot(torch.tensor([int(ac)]), 6) for ac in modelled_agent_actions]
                env_actions = [[modelled_agent_actions[id][0].detach().numpy().argmax(),
                                actions[id].detach().numpy().argmax()] for id in
                               range(args['num_processes'])]

                next_obs, rewards, next_dones, _ = envs.step(env_actions)

                next_dones = np.expand_dims(np.array(next_dones)[:, 0], axis=-1)
                dones = np.logical_or(dones, next_dones)
                dones = torch.Tensor(dones)
                next_agent_obs = torch.FloatTensor([o[1][2] for o in next_obs])
                next_modelled_agent_obs = [torch.FloatTensor(o[0][1]).unsqueeze(0) for o in next_obs]
                rewards = torch.FloatTensor([r[1] for r in rewards]).unsqueeze(1)
                average_reward += rewards
                if counter == args['episode_length']:
                    dones = torch.ones(args['num_processes'], 1)

                batches[step].insert(agent_obs, actions, value, rewards, dones,
                                     torch.stack(modelled_agent_obs), torch.stack(modelled_agent_actions))
                agent_obs = next_agent_obs
                modelled_agent_obs = next_modelled_agent_obs
                true_modelled_agent_obs = [o[0][0] for o in next_obs]

            if np.all(dones.detach().numpy()):
                last_value = torch.zeros(args['num_processes'], 1)
            else:
                last_value = agent.compute_value(next_agent_obs, actions, hidden).detach()

            batches[step].value_preds[-1] = last_value
            batches[step].compute_returns(args['gamma'], args['gae_lambda'], standardise)
        agent.update(batches)
        if episode_passed % 100 == 0:
            returns = evaluate(agent, args)
            results.append(returns)
            print(results)
            data = {
                'hyperparameters': args,
                'results': results}
            pickle.dump(data, open('results/' + str(name) + '.p', "wb"))
            agent.save_params(str(name))


def evaluate(agent, args):
    env = [make_env(args['env']) for _ in range(100)]
    envs = DummyVecEnv(env)

    tasks = np.random.choice(range(args['num_tasks']), size=100)

    hidden = (torch.zeros((1, 100, args['hidden_dim1'])),
              torch.zeros((1, 100, args['hidden_dim1'])))
    for t in range(args['update_episode']):
        obs = envs.reset()
        agent_obs = torch.FloatTensor([o[1][2] for o in obs])
        modelled_agent_obs = [torch.FloatTensor(o[0][1]).unsqueeze(0) for o in obs]
        true_modelled_agent_obs = [o[0][0] for o in obs]

        dones = torch.zeros((100, 1))
        actions = torch.zeros((100, args['action_dim']))
        average_reward = torch.zeros(100, 1)
        for i in range(args['episode_length']):
            with torch.no_grad():
                actions, _, hidden = agent.act(agent_obs, actions, hidden)

            actions = actions[0]
            modelled_agent_actions = [pretrained_output(t, (o, to)) for t, o, to in zip(tasks, modelled_agent_obs,
                                                                                        true_modelled_agent_obs)]
            modelled_agent_actions = [F.one_hot(torch.tensor([int(ac)]), 6) for ac in modelled_agent_actions]

            env_actions = [[modelled_agent_actions[id][0].detach().numpy().argmax(),
                            actions[id].detach().numpy().argmax()] for id in range(100)]
            next_obs, rewards, next_dones, _ = envs.step(env_actions)
            dones = torch.Tensor(dones)
            rewards = torch.FloatTensor([r[1] for r in rewards]).unsqueeze(1)
            average_reward += rewards * (1 - dones)

            next_dones = np.expand_dims(np.array(next_dones)[:, 0], axis=-1)
            dones = np.logical_or(dones, next_dones)

            agent_obs = torch.FloatTensor([o[1][2] for o in next_obs])
            modelled_agent_obs = [torch.FloatTensor(o[0][1]).unsqueeze(0) for o in next_obs]
            true_modelled_agent_obs = [o[0][0] for o in next_obs]
            dones = torch.FloatTensor([d for d in dones]).unsqueeze(1)

    return average_reward.mean().item()
