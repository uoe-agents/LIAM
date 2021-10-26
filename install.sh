pip install torch
pip install tensorflow
cd ..
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..
git clone https://github.com/shariqiqbal2810/multiagent-particle-envs.git
cd multiagent-particle-envs
pip install -e .
cd ../LIAM
pip install gym==0.9.4
pip install seaborn
cp multi_agent_env/simple_reference.py ../multiagent-particle-envs/multiagent/scenarios/.