# Local Information Agent Modelling (LIAM)

Source code of LIAM from Agent Modelling under Partial-Observability for Deep Reinforcement Learning (NeurIPS 2021).

The code is written in python 3, using Pytorch for the implementation of the deep networks. Other important packages are OpenAI Baselines, OpenAI gym, and the Multi-agent Particle Environment.
## Installation
To install the required codebase, it is recommended to create a conda or a virtual environment. Then, run the following command
```
./install.sh
```
## Execution

The file `run_tests.py` contains the hyperparameters that were used in the experiments. Different hyperparameters can be used by modifying this file.
This file will generate all the possible hyperparameter configurations based on the specified values.
To train LIAM run the following command
```
python run_tests.py 0
```
where `0` indicates that the first configuration of hyperparameters will be used for training.

## Citing LIAM

If you use this repository in your work, please consider citing the [LIAM paper](https://arxiv.org/abs/2006.09447)
```tex
@article{papoudakis2021agent,
  title={Agent Modelling under Partial Observability for Deep Reinforcement Learning},
  author={Papoudakis, Georgios and Christianos, Filippos and Albrecht, Stefano V.},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}
```
## Acknowledgements

For LIAM's implementation, we use the MPE environment from this [source](https://github.com/shariqiqbal2810/multiagent-particle-envs).

For the implementation of MADDPG algorithm we use the source code from this [source](https://github.com/shariqiqbal2810/maddpg-pytorch)