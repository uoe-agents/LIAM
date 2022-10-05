import numpy as np
import torch
from agent import A2C
from storage import RolloutStorage
from envs import DummyVecEnv
import pickle
import uuid
from pretrained_predators import get_opponent_actions
from standardise_stream import RunningMeanStd


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
    torch.cuda.manual_seed_all(args['seed'])

    torch.set_num_threads(1)
    env = [make_env('predator_prey') for _ in range(args['num_processes'])]
    for i in range(args['num_processes']):
        env[i].seed((args['seed'] + 1) * i)
    name = uuid.uuid4()
    envs = DummyVecEnv(env)
    standardise = RunningMeanStd(shape=1)
    agent = A2C(args['obs_dim'],
                args['hidden_dim1'],
                args['embedding_dim'],
                args['action_dim'], args['opp_obs_dim'], args['opp_act_dim'],
                args['lr1'],
                args['lr2'],
                args['entropy_coef'],
                max_grad_norm=args['max_grad_norm'])

    num_batches = args['episode_length'] // args['batch_size']
    num_updates = int(args['num_env_steps'] // num_batches // args['batch_size'] // args['num_processes'])
    episode_passed = -1
    dones = torch.ones(args['num_processes'], 1)
    print('Number of updates that will be performed ' + str(num_updates))

    for j in range(num_updates):
        batches = [RolloutStorage(args['batch_size'], args['num_processes'], args['obs_dim'],
                                  args['action_dim'], args['hidden_dim1'], args['opp_obs_dim'], args['opp_act_dim'])
                   for _ in range(num_batches)]

        for step in range(num_batches):
            if dones[0]:
                counter = 0

                obs = envs.reset()
                agent_obs = torch.FloatTensor([o[3] for o in obs])
                modelled_agent_obs = [o[:3] for o in obs]
                dones = torch.zeros((args['num_processes'], 1))
                actions = torch.zeros((args['num_processes'], args['action_dim']))
                episode_passed += 1

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
                modelled_agent_actions = [get_opponent_actions(modelled_agent_obs[id], tasks[id]) for id in range(args['num_processes'])]

                env_actions = [[modelled_agent_actions[id][0], modelled_agent_actions[id][1], modelled_agent_actions[id][2],
                                actions[id].detach().numpy()] for id in range(args['num_processes'])]
                next_obs, rewards, dones, _ = envs.step(env_actions)

                next_agent_obs = torch.FloatTensor([o[3] for o in next_obs])
                next_modelled_agent_obs = [o[:3] for o in next_obs]
                rewards = torch.FloatTensor([r[3] for r in rewards]).unsqueeze(1)
                dones = torch.FloatTensor([d[0] for d in dones]).unsqueeze(1)
                if counter == args['episode_length']:
                    dones = torch.ones(args['num_processes'], 1)

                modelled_agent_obs_torch = torch.Tensor(modelled_agent_obs)
                modelled_agent_act_torch = torch.Tensor(modelled_agent_actions)
                batches[step].insert(agent_obs, actions, value, rewards, dones,
                                     modelled_agent_obs_torch[:, 0], modelled_agent_obs_torch[:, 1],
                                     modelled_agent_obs_torch[:, 2], modelled_agent_act_torch[:, 0],
                                     modelled_agent_act_torch[:, 1], modelled_agent_act_torch[:, 2])

                agent_obs = next_agent_obs
                modelled_agent_obs = next_modelled_agent_obs
            if dones[0]:
                last_value = torch.zeros(args['num_processes'], 1)
            else:
                last_value = agent.compute_value(next_agent_obs, actions, hidden)
                
            batches[step].value_preds[-1] = last_value.detach()
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
    env = [make_env('predator_prey') for _ in range(100)]
    envs = DummyVecEnv(env)

    tasks = np.random.choice(range(args['num_tasks']), size=100)
    hidden = (torch.zeros((1, 100, args['hidden_dim1'])),
              torch.zeros((1, 100, args['hidden_dim1'])))
    for t in range(args['update_episode']):
        obs = envs.reset()
        agent_obs = torch.FloatTensor([o[3] for o in obs])
        modelled_agent_obs = [o[:3] for o in obs]
        actions = torch.zeros((100, args['action_dim']))
        average_reward = torch.zeros(100, 1)
        for _ in range(args['episode_length']):
            with torch.no_grad():
                actions, _, hidden = agent.act(agent_obs, actions, hidden)
            actions = actions[0]
            modelled_agent_actions = [get_opponent_actions(modelled_agent_obs[id], tasks[id]) for id in range(100)]

            env_actions = [[modelled_agent_actions[id][0], modelled_agent_actions[id][1], modelled_agent_actions[id][2],
                            actions[id].detach().numpy()] for id in range(100)]
            next_obs, rewards, dones, _ = envs.step(env_actions)
            next_agent_obs = torch.FloatTensor([o[3] for o in next_obs])
            next_modelled_agent_obs = [o[:3] for o in next_obs]
            rewards = torch.FloatTensor([r[3] for r in rewards]).unsqueeze(1)
            rewards[rewards<-2] += 10
            average_reward += rewards
            agent_obs = next_agent_obs
            modelled_agent_obs = next_modelled_agent_obs

    return average_reward.mean().item()
