## Reinforcement learning Assignment 2: Deep Q learning
To obtain the graphs that are presented in the report the following command line commands should be used:

- "dqn_experiment.py optimization" to get the the results for the optimization study
- "dqn_experiment.py ablation" to get the results for the ablation study
- "dqn_experiment.py exploration" to get the results for the exploration strategy study

The experiment uses the two files called Helper.py and dqn.py. Helper.py consists of the functions to make and smooth the plots and the function for the softmax exploration strategy. The file dqn.py consists of the Deep Q-Network algorithm with the target network and the replay buffer.

Python packages numpy, matplotlib, scipy, gym and tensorflow should be installed. The results presented in the report were obtained using python version 3.6.8 with the following package versions:

- tensorflow 1.12.0
- gym 0.21.0
- numpy 1.19.5
- matplitlib 3.3.4
- scipy 1.5.4
