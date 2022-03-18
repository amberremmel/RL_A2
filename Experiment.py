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
               epsilon_max=0.5, epsilon_min=0.05, epsilon_decay=0.99,
               ER_buffer=False, ER_size=100, n_update_TN = 10,
               render=False, smoothing_window=51):

    reward_results = np.empty([n_repetitions, n_episodes]) # Result array
    now = time.time()
    
    for rep in range(n_repetitions): # Loop over repetitions
        # rewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
        rewards = q_learning(n_episodes, learning_rate, gamma, 
               n_nodes, epsilon_max, epsilon_min, epsilon_decay,
               ER_buffer, ER_size, n_update_TN, render)
        reward_results[rep] = rewards
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve  

def experiment():
    ####### Settings
    # Experiment    
    n_repetitions = 2
    smoothing_window = 21#1001

    n_episodes = 500
    gamma = 0.9
    learning_rate = 0.005
    
    # Hidden layers
    n_nodes = [64, 32]

    # Exploration
    epsilon_max = 0.8
    epsilon_min = 0.05
    epsilon_decay = 0.995

    # Experience replay
    ER_buffer = True
    ER_size = 200
    
    # After how much episodes the target network will be updated
    # When value of 1 is chosen > same as if no target network is used
    # Because the target network will then be updated at every episode
    n_update_TN = 25

    # Plotting parameters
    render = False
    
    ####### Experiments
    
    # #### Assignment 2: Effect of exploration
    # backup = 'q'
    # Plot = LearningCurvePlot(title = 'Q-learning: effect of $\epsilon$-greedy versus softmax exploration')    
    # policy = 'egreedy'
    # epsilons = [0.01,0.05,0.2]
    # for epsilon in epsilons:        
        # learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              # gamma, policy, epsilon, temp, smoothing_window, plot, n)
        # Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))    
    # policy = 'softmax'
    # temps = [0.01,0.1,1.0]
    # for temp in temps:
        # learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              # gamma, policy, epsilon, temp, smoothing_window, plot, n)
        # Plot.add_curve(learning_curve,label=r'softmax, $ \tau $ = {}'.format(temp))
    # Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    # Plot.save('exploration.png')
    # policy = 'egreedy'
    # epsilon = 0.05 # set epsilon back to original value 
    # temp = 1.0
    
    
    # ###### Assignment 3: Q-learning versus SARSA
    # backups = ['q','sarsa']
    # learning_rates = [0.05,0.2,0.4]
    # Plot = LearningCurvePlot(title = 'Q-learning versus SARSA')    
    # for backup in backups:
        # for learning_rate in learning_rates:
            # learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                                  # gamma, policy, epsilon, temp, smoothing_window, plot, n)
            # Plot.add_curve(learning_curve,label=r'{}, $\alpha$ = {} '.format(backup_labels[backup],learning_rate))
    # Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    # Plot.save('on_off_policy.png')
    # # Set back to original values
    # learning_rate = 0.25
    # backup = 'q'


    # # ##### Assignment 4: Back-up depth
    # backup = 'nstep'
    # ns = [1,3,5,10,20,100]
    # Plot = LearningCurvePlot(title = 'Effect of target depth')    
    # for n in ns:
        # learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                              # gamma, policy, epsilon, temp, smoothing_window, plot, n)
        # Plot.add_curve(learning_curve,label=r'{}-step Q-learning'.format(n))
    # backup = 'mc'
    # learning_curve = average_over_repetitions(backup, n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                          # gamma, policy, epsilon, temp, smoothing_window, plot, n)
    # Plot.add_curve(learning_curve,label='Monte Carlo')        
    # Plot.add_hline(optimal_average_reward_per_timestep, label="DP optimum")
    # Plot.save('depth.png')

    Plot = LearningCurvePlot(title = 'Deep Q-network') 
    learning_rates = [0.1,0.05,0.01]   
    for learning_rate in learning_rates:
        learning_curve = average_over_repetitions(n_repetitions, n_episodes,
                   learning_rate, gamma, n_nodes,epsilon_max, epsilon_min, 
                   epsilon_decay, ER_buffer, ER_size, n_update_TN, render, 
                   smoothing_window)
        Plot.add_curve(learning_curve,label=r'$\alpha$ = {} '.format(learning_rate))        
    Plot.save('dqn_result.png')
    
if __name__ == '__main__':
    experiment()
