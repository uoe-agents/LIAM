import numpy as np
import torch
from agent import A2C
from storage import RolloutStorage
from envs import DummyVecEnv
from speaker_output import *
import pickle
import uuid
from ddpg import DDPG
import os


def make_env(scenario_name, benchmark=False, discrete_action=True):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                        discrete_action=discrete_action)
    return env


def main(args):
    results = []
    torch.manual_seed(args['seed'])
    torch.set_num_threads(1)
    np.random.seed(args['seed'])
    try:
        os.mkdir('results')
    except:
        pass
    try:
        os.mkdir('trained_parameters')
    except:
        pass
    env = [make_env('simple_reference') for _ in range(args['num_processes'])]
    for i in range(args['num_processes']):
        env[i].seed((args['seed'] + 1) * i)

    envs = DummyVecEnv(env)
    name = uuid.uuid4()
    fixed_agents = [DDPG(args['modelled_obs_dim'], 5, 52, 128, 0.01) for _ in range(10)]
    for i in range(5):
        save_dict = torch.load('simple_ref_params/agent' + str(i + 1) + '/params_30000.pt')
        params = save_dict['agent_params']
        fixed_agents[2 * i].load_params(params[0])
        fixed_agents[2 * i + 1].load_params(params[1])
    agent = A2C(args['obs_dim'],
                args['hidden_dim'],
                args['embedding_dim'],
                args['action_dim'],
                args['modelled_obs_dim'],
                args['modelled_act_dim'],
                args['lr1'],
                args['lr2'],
                args['entropy_coef'],
                args['value_loss_coef'],
                max_grad_norm=args['max_grad_norm'])

    num_batches = args['episode_length'] // args['batch_size']
    num_updates = int(args['num_env_steps'] // num_batches // args['batch_size'] // args['num_processes'])
    episode_passed = -1
    dones = torch.ones(args['num_processes'], 1)
    average_reward = torch.zeros(args['num_processes'], 1)
    print('Number of updates that will be performed ' + str(num_updates))

    for j in range(num_updates):
        batches = [RolloutStorage(args['batch_size'], args['num_processes'], args['obs_dim'], args['action_dim'],
                                  args['hidden_dim'], args['modelled_obs_dim'], 2 * args['modelled_act_dim'])
                   for _ in range(num_batches)]

        for step in range(num_batches):
            if dones[0]:
                counter = 0

                obs = envs.reset()
                agent_obs = torch.FloatTensor([o[1] for o in obs])
                modelled_obs = [torch.FloatTensor(o[0]).unsqueeze(0) for o in obs]
                actions1 = torch.zeros((args['num_processes'], args['action_dim']))
                actions2 = torch.zeros((args['num_processes'], args['action_dim']))
                actions = torch.cat((actions1, actions2), dim=-1)
                episode_passed += 1
                average_reward = torch.zeros(args['num_processes'], 1)

                tasks = np.random.choice(range(1, args['num_tasks'] + 1), size=args['num_processes'])
                task_agents = [fixed_agents[tasks[id] - 1] for id in range(args['num_processes'])]

                hidden = (torch.zeros((1, args['num_processes'], args['hidden_dim'])),
                          torch.zeros((1, args['num_processes'], args['hidden_dim'])))
            batches[step].initialize(actions1, actions2, hidden)

            for i in range(args['batch_size']):
                with torch.no_grad():
                    actions, actions1, actions2, value, hidden = agent.act(agent_obs, actions, hidden)

                counter += 1
                actions1 = actions1[0]
                actions2 = actions2[0]
                actions = actions[0]
                value = value[0]
                modelled_actions1 = torch.stack([task_agents[id].step(modelled_obs[id])
                                                 for id in range(args['num_processes'])])
                modelled_actions2 = torch.stack([speaker_output(tasks[id], modelled_obs[id][:, 8: 11])
                                                 for id in range(args['num_processes'])])

                modelled_actions = torch.cat((modelled_actions1, modelled_actions2), dim=-1)
                env_actions = [[modelled_actions[id][0].detach().numpy(),
                                actions[id].detach().numpy()] for id in range(args['num_processes'])]
                next_obs, rewards, dones, infos = envs.step(env_actions)
                next_agent_obs = torch.FloatTensor([o[1] for o in next_obs])
                next_modelled_obs = [torch.FloatTensor(o[0]).unsqueeze(0) for o in next_obs]

                rewards1 = torch.FloatTensor([r[0] for r in rewards]).unsqueeze(1)
                rewards2 = torch.FloatTensor([r[1] for r in rewards]).unsqueeze(1)
                rewards = (rewards1 + rewards2) / 2
                average_reward += rewards
                dones = torch.FloatTensor([d[0] for d in dones]).unsqueeze(1)
                if counter == args['episode_length']:
                    dones = torch.ones(args['num_processes'], 1)

                batches[step].insert(agent_obs, actions1, actions2, value, rewards, dones,
                                     torch.stack(modelled_obs), modelled_actions)
                agent_obs = next_agent_obs
                modelled_obs = next_modelled_obs
            if dones[0]:
                last_value = torch.zeros(args['num_processes'], 1)
            else:
                last_value = agent.compute_value(next_agent_obs, actions, hidden)
                
            batches[step].value_preds[-1] = last_value.detach()
            batches[step].compute_returns(args['gamma'], args['gae_lambda'])

        agent.update(batches)
        if episode_passed % 100 == 0:
            returns = evaluate(agent, args, fixed_agents)
            results.append(returns)
            print(results)
            data = {
                'hyperparameters': args,
                'results': results}

            pickle.dump(data, open('results/' + str(name) + '.p', "wb"))
            agent.save_params(str(name))


def evaluate(agent, args, fixed_agents):
    env = [make_env('simple_reference') for _ in range(100)]
    envs = DummyVecEnv(env)

    tasks = np.random.choice(range(1, args['num_tasks'] + 1), size=100)
    tasks_agents = [fixed_agents[tasks[id] - 1] for id in range(100)]
    hidden = (torch.zeros((1, 100, args['hidden_dim'])),
              torch.zeros((1, 100, args['hidden_dim'])))
    obs = envs.reset()
    agent_obs = torch.FloatTensor([o[1] for o in obs])
    modelled_obs = [torch.FloatTensor(o[0]).unsqueeze(0) for o in obs]
    actions = torch.zeros((100, 2 * args['action_dim']))
    average_reward = torch.zeros(100, 1)
    for _ in range(args['episode_length']):
        with torch.no_grad():
            actions, _, _, _, hidden = agent.act(agent_obs, actions, hidden)
        actions = actions[0]
        modelled_actions1 = torch.stack([tasks_agents[id].step(modelled_obs[id]) for id in range(100)])
        modelled_actions2 = torch.stack([speaker_output(tasks[id], modelled_obs[id][:, 8: 11]) for id in range(100)])

        modelled_actions = torch.cat((modelled_actions1, modelled_actions2), dim=-1)
        env_actions = [[modelled_actions[id][0].detach().numpy(), actions[id].detach().numpy()] for id in
                       range(100)]
        next_obs, rewards, dones, infos = envs.step(env_actions)
        next_agent_obs = torch.FloatTensor([o[1] for o in next_obs])
        next_modelled_obs = [torch.FloatTensor(o[0]).unsqueeze(0) for o in next_obs]
        rewards1 = torch.FloatTensor([r[0] for r in rewards]).unsqueeze(1)
        rewards2 = torch.FloatTensor([r[1] for r in rewards]).unsqueeze(1)
        average_reward += (rewards1 + rewards2) / 2
        agent_obs = next_agent_obs
        modelled_obs = next_modelled_obs

    return average_reward.mean().item()
