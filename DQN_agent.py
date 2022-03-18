import gym

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque



class DQN_agent:

    def __init__(self, n_actions=2, learning_rate=0.05, gamma=0.95, ER_size=0):
        """


        :param n_actions: Number of actions available to agent
        :param learning_rate: Learning rate of neural network
        :param gamma: discount factor, used to calculate the targets during the update
        :param ER_size: size of the experience replay buffer.

        The experience replay buffer uses a Deque with an adjustable max size. When an item is added to a Deque
        with the max length, the first item is removed from the deque, and the new item appended at the end,
        keeping the size at the max length.
        """
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.Q_model = self.make_q_model()
        self.ER_buffer = None
        if ER_size > 0:
            self.ER_buffer = deque(maxlen=ER_size)



    def select_action(self, s, epsilon=None):
        if np.random.uniform() < epsilon:
            return np.random.randint(low=0, high=self.n_actions)
        else:
            return np.argmax(self.Q_model.predict(s)[0])


    def update(self, observations):
        states = []
        targets = []
        for s, a, r, s_next, done in observations:
            states.append(s)
            Gt = r
            if not done:
                Gt += self.gamma * np.amax(self.Q_model.predict(s_next)[0])
            target = self.Q_model.predict(s)
            target[0][a] = Gt
            targets.append(target)
        states = np.concatenate(states)
        targets = np.concatenate(targets)
        self.Q_model.fit(states, targets, epochs=1, verbose=0)
        pass

    def make_q_model(self):
        inputs = layers.Input(shape=(4,))

        layer1 = layers.Dense(64, activation='sigmoid')(inputs)
        layer2 = layers.Dense(128, activation='sigmoid')(layer1)

        output = layers.Dense(self.n_actions, activation="linear")(layer2)
        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

def q_learning(n_episodes=250,
               learning_rate=0.001, gamma=0.9,
               epsilon_max=0.5, epsilon_min=0.05, epsilon_decay=0.99,
               ER_buffer=False, ER_size=100,
               render=False):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep '''

    epsilon = epsilon_max
    if ER_buffer:
        ER_batch = 50
        agent = DQN_agent(gamma=gamma, learning_rate=learning_rate, ER_size=ER_size)
    else:
        agent = DQN_agent(gamma=gamma, learning_rate=learning_rate)
    print(agent.Q_model)
    #model = agent.make_q_model()
    reward_per_episode = []
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
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1,4])
            episode.append((state, action, reward, next_state, done))
            rewards.append(reward)
            state = next_state
        reward_per_episode.append(np.sum(rewards))
        print("episode: ", i, " score: ", reward_per_episode[-1], " Epsilon: ", epsilon)
        if ER_buffer:
            agent.ER_buffer.extend(episode)
            if len(agent.ER_buffer) > ER_batch:
                agent.update(random.sample(agent.ER_buffer, ER_batch))
        else:
            agent.update(episode)
        if epsilon > epsilon_min:
            epsilon = epsilon*epsilon_decay
    env.close()
    return reward_per_episode



def test():
    n_episodes = 500
    gamma = 0.9
    learning_rate = 0.001

    # Exploration
    epsilon_max = 0.8
    epsilon_min = 0.05
    epsilon_decay = 0.995

    #Experience replay
    ER_buffer = True
    ER_size = 200

    # Plotting parameters
    render = False
    rewards = q_learning(n_episodes=n_episodes,
                         learning_rate=learning_rate,
                         gamma=gamma,
                         epsilon_max=epsilon_max,
                         epsilon_min=epsilon_min,
                         epsilon_decay=epsilon_decay,
                         ER_buffer=ER_buffer,
                         ER_size=ER_size,
                         render=render)
    print("Obtained rewards: {}".format(rewards))


if __name__ == '__main__':
    test()
