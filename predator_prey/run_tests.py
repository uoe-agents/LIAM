from main import main
import argparse
from itertools import product

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)
    args = parser.parse_args()

    params = dict(
        {'cuda': [False, ],
         'num_env_steps': [10e6],
         'episode_length': [50, ],
         'num_processes': [10],
         'lr1': [3e-4],
         'lr2': [7e-4],
         'num_tasks': [10],
         'use_gae': [True, ],
         'gamma': [0.99, ],
         'gae_lambda': [0.95, ],
         'value_loss_coef': [0.5, ],
         'entropy_coef': [0.01,],
         'eps': [1e-5, ],
         'max_grad_norm': [0.5, ],
         'batch_size': [5],
         'obs_dim': [14, ],
         'action_dim': [5, ],
         'hidden_dim1': [128],
         'update_episode': [1],
         'embedding_dim': [20, ],
         'kl_coeff': [1.0,],
         'recon_coeff1': [1],
         'recon_coeff2': [1],
         'seed': [0, 1, 2, 4, 3],
         'opp_obs_dim': [16],
         'opp_act_dim': [5],
         'backprop_embeddings': [False]
         })

    param_keys = list(params.keys())

    combs = list(product(*[params[key] for key in param_keys]))
    print(len(combs))
    if args.index >= len(combs):
        raise ValueError("Unique index exceeds hyperparameter combination ({})".format(len(combs) - 1))

    exec_params = {key: combs[args.index][i] for i, key in enumerate(param_keys)}
    main(exec_params)
