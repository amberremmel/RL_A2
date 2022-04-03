#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
Altered by Ricardo Michels, Paula Mieras, Amber Remmelzwaal
"""

import numpy as np
import time
import sys

from dqn import q_learning
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(n_repetitions, n_episodes=250,
               learning_rate=0.001, gamma=0.9, n_nodes=[64, 128],
               epsilon_max=0.5, epsilon_min=0.05, epsilon_decay=0.99, temp=1,
               ER_buffer=False, ER_size=1000, ER_batch=50, update_TN=False, n_update_TN = 10, strategy="epsilon", 
               render=False, smoothing_window=51):

    reward_results = np.empty([n_repetitions, n_episodes]) # Result array
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        print("Repetition {}".format(rep))
        rewards = q_learning(n_episodes, learning_rate, gamma, 
               n_nodes, epsilon_max, epsilon_min, epsilon_decay, temp,
               ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render)
        reward_results[rep] = rewards
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve  

def experiment(study):
    ####### Settings
    # Experiment    
    n_repetitions = 20
    smoothing_window = 21

    n_episodes = 1000
    gamma = 1
    learning_rate = 0.05
    
    # Hidden layers
    n_nodes = [32, 16]

    # Exploration
    epsilon_max = 0.8
    epsilon_min = 0.1
    epsilon_decay = 0.995

    temp = 1

    # Experience replay
    ER_buffer = False
    ER_size = 1000
    ER_batch = 256
    
    # After how much episodes the target network will be updated
    update_TN = False
    n_update_TN = 5
    
    # Exploration strategy
    strategy = "softmax"
    
    # Plotting parameters
    render = False
    
    ####### Experiments
    if study == "optimization":
        """Creates figures for the optimization of the standard parameters 
           and for the parameters of the replay buffer and target network"""

        # Varying the learning_rates
        Plot = LearningCurvePlot(title = 'Deep Q-network optimizing learning rate') 
        learning_rates = [0.1,0.09,0.075,0.06,0.05]
        for learning_rate in learning_rates:
            learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                    learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                    epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                    smoothing_window)
            Plot.add_curve(learning_curve,label=r'$\alpha$ = {} '.format(learning_rate))        
        Plot.save('dqn_result_alpha={}.png'.format(learning_rates))
        
        learning_rate = 0.05

        # Varying the discount parameter
        Plot = LearningCurvePlot(title = 'Deep Q-network optimizing discount parameter') 
        gammas = [0.9,0.7,0.5,1]   
        for gamma in gammas:
            learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                    learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                    epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                    smoothing_window)
            Plot.add_curve(learning_curve,label=r'$\gamma$ = {} '.format(gamma))        
        Plot.save('dqn_result_gamma={}.png'.format(gammas))

        gamma = 1

        # Varying the epsilon decay
        Plot = LearningCurvePlot(title = 'Deep Q-network optimizing epsilon decay') 
        epsilon_decays = [0.9,0.92,0.94,0.96,0.98,0.995]
        for epsilon_decay in epsilon_decays:
            learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                    learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                    epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                    smoothing_window)
            Plot.add_curve(learning_curve,label=r'$\epsilon$-decay = {} '.format(epsilon_decay))        
        Plot.save('dqn_result_epsilon_decay={}.png'.format(epsilon_decays))

        epsilon_decay = 0.995

        # Varying the number of nodes for 2 layers
        Plot = LearningCurvePlot(title = 'Deep Q-network optimizing number of nodes') 
        n_nodess = [[32, 16], [32, 32], [16, 32], [32, 64]]   
        for n_nodes in n_nodess:
            learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                    learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                    epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                    smoothing_window)
            Plot.add_curve(learning_curve,label='nodes = {} '.format((n_nodes)))        
        Plot.save('dqn_result_n_nodes={}.png'.format(n_nodess))  
        
        # Varying the number of layers
        Plot = LearningCurvePlot(title = 'Deep Q-network optimizing number of layers') 
        n_nodess = [[32], [32, 32], [32, 64, 32], [32, 64, 32, 16]]   
        for n_nodes in n_nodess:
            learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                    learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                    epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                    smoothing_window)
            Plot.add_curve(learning_curve,label='# layers = {} '.format(len(n_nodes)))        
        Plot.save('dqn_result_n_layers={}.png'.format([len(n_nodes) for n_nodes in n_nodess]))    

        n_nodes = [32, 16]

        # Varying the replay buffer parameters
        Plot = LearningCurvePlot(title = 'Deep Q-network replay buffer')
        update_TN = False
        ER_buffer = False
        learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                smoothing_window)
        Plot.add_curve(learning_curve,label="no replay buffer") 
        ER_buffer = True
        ER_sizes = [1000, 5000, 10000]
        ER_batchs = [64, 128, 256]
        for ER_size in ER_sizes:
            for ER_batch in ER_batchs:
                learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                    learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                    epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                    smoothing_window)
                Plot.add_curve(learning_curve,label='s, b = [{}, {}]'.format(ER_size, ER_batch))        
                Plot.save('dqn_result_replay_buffer_s_b={}_{}.png'.format(ER_sizes, ER_batchs))
               
        # Varying the target network parameters
        Plot = LearningCurvePlot(title = 'Deep Q-network target network')
        ER_buffer = False
        update_TN = False
        learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                smoothing_window)
        Plot.add_curve(learning_curve,label="no target network") 
        update_TN = True
        n_update_TNs = [5, 10, 25]
        for n_update_TN in n_update_TNs:
            learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                smoothing_window)
            Plot.add_curve(learning_curve,label='update frequency = {}'.format(n_update_TN))
            Plot.save('dqn_result_target_network_{}.png'.format(n_update_TNs))
        
        
    elif study == "ablation":
        """Creates multiple figures for the ablation study"""
              
        # DQN with(out) TN and ER
        Plot = LearningCurvePlot(title = 'Deep Q-network ER and TN')
        # without ER and TN
        learning_curve_dqn = average_over_repetitions(n_repetitions, n_episodes,
                learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                smoothing_window)
        Plot.add_curve(learning_curve_dqn,label="DQN")
        Plot.save('dqn_result_TN_ER_[s,b,n]={}_{}_{}_{}.png'.format(ER_size, ER_batch, n_update_TN, strategy))
        # with ER, without TN
        ER_buffer = True
        learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                    learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                    epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                    smoothing_window)
        Plot.add_curve(learning_curve,label='DQN with ER')
        Plot.save('dqn_result_TN_ER_[s,b,n]={}_{}_{}_{}.png'.format(ER_size, ER_batch, n_update_TN, strategy))
        # with TN, without ER        
        ER_buffer = False
        update_TN = True
        learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                    learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                    epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                    smoothing_window)
        Plot.add_curve(learning_curve,label='DQN with TN')
        Plot.save('dqn_result_TN_ER_[s,b,n]={}_{}_{}_{}.png'.format(ER_size, ER_batch, n_update_TN, strategy))
        # with ER and TN
        ER_buffer = True
        update_TN = True
        learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                        learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                        epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                        smoothing_window)
        Plot.add_curve(learning_curve,label='DQN with ER and TN')
        Plot.save('dqn_result_TN_ER_[s,b,n]={}_{}_{}_{}.png'.format(ER_size, ER_batch, n_update_TN, strategy))
        
    
    elif study == "exploration":
        """Creates one figure with exploration strategy plots"""
        
        # DQN with TN and ER annealing epsilon greedy, epsilon greedy and softmax
        Plot = LearningCurvePlot(title = 'Deep Q-network exploration strategy')
        ER_buffer = True
        update_TN = True
        
        # annealing epsilon greedy
        strategy = "epsilon"
        learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                smoothing_window)
        Plot.add_curve(learning_curve,label="Annealing epsilon greedy (epsilon decay={})".format(epsilon_decay))
        Plot.save('dqn_result_TN_ER_exploration_strategy.png')
        
        # epsilon greedy
        epsilon_max = 0.2
        epsilon_decay = 1
        learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                smoothing_window)
        Plot.add_curve(learning_curve,label="Epsilon greedy (epsilon={})".format(epsilon_max))
        Plot.save('dqn_result_TN_ER_exploration_strategy.png')    
        
        # softmax
        strategy = "softmax"
        temps = [0.1, 0.5, 1]
        for temp in temps:
            learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                smoothing_window)
            Plot.add_curve(learning_curve,label="Softmax (temp={})".format(temp))
            Plot.save('dqn_result_TN_ER_exploration_strategy.png')    
    
    
    else:
        print("Run python 'dqn_experiment.py optimization' to get the optimization study.")
        print("Run python 'dqn_experiment.py ablation' to get the ablation study.")
        print("Run python 'dqn_experiment.py exploration' to get the exploration strategy study.")


if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        study = sys.argv[1] # "optimization" or "ablation" or "exploration"

        experiment(study)
    
    else:
        print("Run python 'dqn_experiment.py optimization' to get the optimization study.")
        print("Run python 'dqn_experiment.py ablation' to get the ablation study.")
        print("Run python 'dqn_experiment.py exploration' to get the exploration strategy study.")
