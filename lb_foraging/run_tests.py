from main import main
import argparse
from itertools import product

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)
    args = parser.parse_args()

    params = dict(
        {'env': ['Foraging-20x20-2p-4f-v0', ],
         'cuda': [False, ],
         'num_env_steps': [10e6, ],
         'episode_length': [50,],
         'num_processes': [10],
         'lr1': [3e-4],
         'lr2': [7e-4],
         'num_tasks': [10, ],
         'gamma': [0.99, ],
         'gae_lambda': [0.95, ],
         'use_proper_time_limits': [True],
         'entropy_coef': [0.001, ],
         'eps': [1e-5, ],
         'max_grad_norm': [0.5, ],
         'batch_size': [5,],
         'action_dim': [6, ],
         'hidden_dim1': [128],
         'update_episode': [1, ],
         'embedding_dim': [20],
         'seed': [0, 1, 2, 3, 4],
         'opp_obs_dim': [30, ],
         'opp_act_dim': [6, ],
         'backprop_embeddings': [False,]
         })

    param_keys = list(params.keys())

    combs = list(product(*[params[key] for key in param_keys]))
    print(len(combs))
    if args.index >= len(combs):
        raise ValueError("Unique index exceeds hyperparameter combination ({})".format(len(combs) - 1))

    exec_params = {key: combs[args.index][i] for i, key in enumerate(param_keys)}
    main(exec_params)
