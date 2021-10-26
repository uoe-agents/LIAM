from main import main
import argparse
from itertools import product

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)
    args = parser.parse_args()

    params = dict(
        {'cuda': [False, ],
         'num_env_steps': [40e6],
         'episode_length': [25, ],
         'num_processes': [10],
         'lr1': [3e-4],
         'lr2': [7e-4],
         'num_tasks': [10],
         'gamma': [0.99, ],
         'gae_lambda': [0.95, ],
         'value_loss_coef': [0.5, ],
         'entropy_coef': [0.01,],
         'eps': [1e-5, ],
         'max_grad_norm': [0.5, ],
         'batch_size': [5],
         'obs_dim': [18, ],
         'action_dim': [5, ],
         'hidden_dim': [128],
         'embedding_dim': [20, ],
         'seed': [0, 1, 2, 3, 4],
         'modelled_obs_dim': [16],
         'modelled_act_dim': [5],
         'backprop_embeddings': [False]
         })

    param_keys = list(params.keys())

    combs = list(product(*[params[key] for key in param_keys]))
    print(len(combs))
    if args.index >= len(combs):
        raise ValueError("Unique index exceeds hyperparameter combination ({})".format(len(combs) - 1))

    exec_params = {key: combs[args.index][i] for i, key in enumerate(param_keys)}
    main(exec_params)
