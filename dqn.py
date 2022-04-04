#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Ricardo Michels, Paula Mieras, Amber Remmelzwaal
"""

import gym

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
from Helper import softmax


class DQN_agent:

    def __init__(self, n_actions=2, n_nodes=[32, 16], learning_rate=0.05, gamma=1, ER_size=0, update_TN=False):
        """
        :param n_actions: Number of actions available to agent
        :param n_nodes: Number of nodes in hidden layers
        :param learning_rate: Learning rate of neural network
        :param gamma: discount factor, used to calculate the targets during the update
        :param ER_size: size of the experience replay buffer.
        :param update_TN: boolean which sets the target network update on or off
        The experience replay buffer uses a Deque with an adjustable max size. When an item is added to a Deque
        with the max length, the first item is removed from the deque, and the new item appended at the end,
        keeping the size at the max length.
        """
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.n_nodes = n_nodes
        self.Q_model = self.make_q_model()
        self.update_TN = update_TN
        self.target_network = keras.models.clone_model(self.Q_model)
        self.target_network.set_weights(self.Q_model.get_weights())
        self.ER_buffer = None
        if ER_size > 0:
            self.ER_buffer = deque(maxlen=ER_size)

    def select_action(self, s, strategy, epsilon=None, temp=None):
        """Select the action for a given state and strategy"""
        
        # Epsilon-greedy policy
        if strategy == "epsilon":
            
            # Take random value with probability epsilon
            if np.random.uniform() < epsilon:
                return np.random.randint(low=0, high=self.n_actions)
                
            # Take best Q network value
            else:
                return np.argmax(self.Q_model.predict(s)[0])
                
        # Boltzmann policy
        elif strategy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            
            x = self.Q_model.predict(s)[0]
            y = softmax(x, temp)
            z = random.random()
            for i, val in enumerate(y):
                if z <= val:
                    a = i
                    break
                else:
                    z -= val

            return a

    def update(self, observations):
        """Update Q model"""
        
        states = []
        targets = []
        for s, a, r, s_next, done in observations:
            states.append(s)
            Gt = r
            
            # Update the backup target with target or Q network
            if not done:
                if self.update_TN:
                    Gt += self.gamma * np.amax(self.target_network.predict(s_next)[0])
                else:
                    Gt += self.gamma * np.amax(self.Q_model.predict(s_next)[0])
            target = self.Q_model.predict(s)
            target[0][a] = Gt
            targets.append(target)
        states = np.concatenate(states)
        targets = np.concatenate(targets)
        
        # Shuffle on False gives the episodes in chronological order
        self.Q_model.fit(states, targets, epochs=1, verbose=0, shuffle=False)

    def make_q_model(self):
        """Construct the Q network with input, hidden and output layers"""
        
        # Make input layer
        inputs = layers.Input(shape=(4,))
        layers_list = [inputs]
        
        # Make hidden layers
        for i in range(len(self.n_nodes)):
            layers_list.append(layers.Dense(self.n_nodes[i], activation='sigmoid')(layers_list[i]))
        
        # Make output layer
        output = layers.Dense(self.n_actions, activation="linear")(layers_list[-1])
        
        # Combine input and output layers to construct the network
        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def update_target_network(self):
        """Target network will be updated with current Q network"""
        
        self.target_network.set_weights(self.Q_model.get_weights())


def q_learning(n_episodes=250,
               learning_rate=0.05, gamma=1, n_nodes=[32, 16],
               epsilon_max=0.8, epsilon_min=0.1, epsilon_decay=0.995,
               temp=1, ER_buffer=False, ER_size=1000, ER_batch=128, update_TN=False,
               n_update_TN=5, strategy="epsilon", render=False):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep '''

    epsilon = epsilon_max
    
    # Initialize the right agent with(out) ER and/or TN
    if ER_buffer and update_TN:
        ER_batch = ER_batch
        agent = DQN_agent(gamma=gamma, learning_rate=learning_rate, ER_size=ER_size, update_TN=update_TN)
    elif ER_buffer:
        ER_batch = ER_batch
        agent = DQN_agent(gamma=gamma, learning_rate=learning_rate, ER_size=ER_size)
    elif update_TN:
        agent = DQN_agent(gamma=gamma, learning_rate=learning_rate, update_TN=update_TN)
    else:
        agent = DQN_agent(gamma=gamma, learning_rate=learning_rate)
    print(agent.Q_model)
    
    reward_per_episode = []
    
    # Make the CartPole environment
    env = gym.make("CartPole-v1")
    
    for i in range(n_episodes):
        rewards = []
        state = env.reset()
        state = np.reshape(state, [1, 4])
        episode = []
        done = False
        while not done:
            if render:
                env.render()
                
            # Select action for given state and strategy
            action = agent.select_action(state, strategy, epsilon=epsilon, temp=temp)
            
            # Simulate environment
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            episode.append((state, action, reward, next_state, done))
            rewards.append(reward)
            state = next_state
        
        if update_TN:
            # Update target network when at the right episode
            if i % n_update_TN == 0:
                agent.update_target_network()
        reward_per_episode.append(np.sum(rewards))
        
        # Update Q network with or without ER buffer
        if ER_buffer:
            agent.ER_buffer.extend(episode)
            if len(agent.ER_buffer) > ER_batch:
                agent.update(random.sample(agent.ER_buffer, ER_batch))
        else:
            agent.update(episode)
        
        # Decrease epsilon if exploration strategy is annealing epsilon-greedy    
        if strategy == "epsilon" and epsilon_decay != 1:
            print("episode: ", i, " score: ", reward_per_episode[-1], " Epsilon: ", epsilon)
            
            if epsilon > epsilon_min:
                epsilon = epsilon * epsilon_decay
        else:
            print("episode: ", i, " score: ", reward_per_episode[-1])
            
    env.close()
    return reward_per_episode


def test():
    """Test function which prints the obtained rewards for the parameters below"""
    
    n_episodes = 500
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
    ER_size = 5000
    ER_batch = 128

    # The update frequency of the target network
    update_TN = True
    n_update_TN = 5

    # Exploration strategy
    strategy = "epsilon"  # "softmax"

    # Plotting parameters
    render = False
    rewards = q_learning(n_episodes=n_episodes,
                         learning_rate=learning_rate,
                         gamma=gamma,
                         n_nodes=n_nodes,
                         epsilon_max=epsilon_max,
                         epsilon_min=epsilon_min,
                         epsilon_decay=epsilon_decay,
                         temp=temp,
                         ER_buffer=ER_buffer,
                         ER_size=ER_size,
                         ER_batch=ER_batch,
                         update_TN=update_TN,
                         n_update_TN=n_update_TN,
                         strategy="epsilon",
                         render=render)
    print("Obtained rewards: {}".format(rewards))


if __name__ == '__main__':
    test()
