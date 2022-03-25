#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
import time

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
        # rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
        rewards = q_learning(n_episodes, learning_rate, gamma, 
               n_nodes, epsilon_max, epsilon_min, epsilon_decay, temp,
               ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render)
        reward_results[rep] = rewards
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve  

def experiment():
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
    epsilon_min = 0.005
    epsilon_decay = 0.995

    temp = 1

    # Experience replay
    ER_buffer = False
    ER_size = 1000
    ER_batch = 50
    
    # After how much episodes the target network will be updated
    update_TN = False
    n_update_TN = 25
    
    # Exploration strategy
    strategy = "epsilon"
    
    # Plotting parameters
    render = False
    
    ####### Experiments

    # Varying the replay buffer parameters
    Plot = LearningCurvePlot(title = 'Deep Q-network')
    ER_buffer = False
    learning_curve = average_over_repetitions(n_repetitions, n_episodes,
               learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
               epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
               smoothing_window)
    Plot.add_curve(learning_curve,label="no replay buffer") 
    ER_buffer = True
    ER_sizes = [1000, 5000, 10000]
    ER_batchs = [50, 100, 200]
    for ER_size in ER_sizes:
        for ER_batch in ER_batchs:
            learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                   learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                   epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                   smoothing_window)
            Plot.add_curve(learning_curve,label='s, b = [{}, {}]'.format(ER_size, ER_batch))        
    Plot.save('dqn_result_replay_buffer.png')
    
    ER_buffer = False
    ER_size = 1000
    ER_batch = 50
    
    # Varying the target network parameters
    Plot = LearningCurvePlot(title = 'Deep Q-network')
    update_TN = False
    learning_curve = average_over_repetitions(n_repetitions, n_episodes,
               learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
               epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
               smoothing_window)
    Plot.add_curve(learning_curve,label="no target network") 
    update_TN = True
    n_update_TNs = [25, 50, 100]
    for n_update_TN in n_update_TNs:
        learning_curve = average_over_repetitions(n_repetitions, n_episodes,
               learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
               epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
               smoothing_window)
        Plot.add_curve(learning_curve,label='update frequency = {}'.format(n_update_TN))
    Plot.save('dqn_result_target_network.png')
    
    update_TN = False
    n_update_TN = 25
    
    # # Varying the learning_rates
    # Plot = LearningCurvePlot(title = 'Deep Q-network') 
    # # learning_rates = [0.1,0.05,0.01,0.005, 0.001]
    # learning_rates = [0.1,0.09,0.075,0.06,0.05]
    # for learning_rate in learning_rates:
        # learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                   # learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                   # epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                   # smoothing_window)
        # Plot.add_curve(learning_curve,label=r'$\alpha$ = {} '.format(learning_rate))        
    # Plot.save('dqn_result_alpha={}.png'.format(learning_rates))
    
    # learning_rate = 0.05

    # # Varying the discount parameter
    # Plot = LearningCurvePlot(title = 'Deep Q-network') 
    # # gammas = [0.9,0.7,0.5,1]   
    # gammas = [0.8,0.9,0.95,1]   
    # for gamma in gammas:
        # learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                   # learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                   # epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                   # smoothing_window)
        # Plot.add_curve(learning_curve,label=r'$\gamma$ = {} '.format(gamma))        
    # Plot.save('dqn_result_gamma={}.png'.format(gammas))

    # gamma = 0.9

    # # Varying the epsilon decay
    # Plot = LearningCurvePlot(title = 'Deep Q-network') 
    # # epsilon_decays = [0.99,0.995,0.98]   
    # epsilon_decays = [0.9,0.92,0.94,0.96,0.98,0.995]
    # for epsilon_decay in epsilon_decays:
        # learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                   # learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                   # epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                   # smoothing_window)
        # Plot.add_curve(learning_curve,label=r'$\epsilon$-decay = {} '.format(epsilon_decay))        
    # Plot.save('dqn_result_epsilon_decay={}.png'.format(epsilon_decays))

    # epsilon_decay = 0.995

    # # Varying the number of nodes for 2 layers
    # Plot = LearningCurvePlot(title = 'Deep Q-network') 
    # n_nodess = [[32, 16], [32, 32], [16, 32], [32, 64]]   
    # for n_nodes in n_nodess:
        # learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                   # learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                   # epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                   # smoothing_window)
        # Plot.add_curve(learning_curve,label='nodes = {} '.format((n_nodes)))        
    # Plot.save('dqn_result_n_nodes={}.png'.format(n_nodess))  
    
    # Varying the number of layers
    # Plot = LearningCurvePlot(title = 'Deep Q-network') 
    # n_nodess = [[32], [32, 32], [32, 64, 32], [32, 64, 32, 16]]   
    # for n_nodes in n_nodess:
        # learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                   # learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                   # epsilon_decay, temp, ER_buffer, ER_size, ER_batch, update_TN, n_update_TN, strategy, render, 
                   # smoothing_window)
        # Plot.add_curve(learning_curve,label='# layers = {} '.format(len(n_nodes)))        
    # Plot.save('dqn_result_n_layers={}.png'.format([len(n_nodes) for n_nodes in n_nodess]))    
    
if __name__ == '__main__':
    experiment()
