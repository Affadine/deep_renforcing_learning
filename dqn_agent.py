import argparse
import sys

import gym
from gym import wrappers, logger


import matplotlib.pyplot as plt
import numpy as np

'''
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
'''

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import random

class DQNAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env):
        self.env = env
        self.state_size = 4 #self.env.observation_space
        self.Q_size = self.env.action_space.n
        self.memoryBuffer = []
        self.maxSize = 100 * 1000
        self.gamma = 0.6
        self.init_epsilon = 1
        self.final_epsilon = 0.05
        self.step_epsilon = 0
        self.epsilon = 0.1
        self.learning_rate =0.01
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = 40
        self.epochs = 1
        self.verbose = 0
        self.model = self.creer_model()
        self.target_model = self.creer_model()
        self.target_model.set_weights(self.model.get_weights())

    def creer_model(self):
        model = keras.Sequential()
        model.add(layers.Flatten(input_shape=(1,))),
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.Q_size, activation='linear'))
        model.compile(loss='mse', optimizer=self.optimizer)
        model.summary()
        return model

    def train(self):
        print('retain after each ' +str(self.batch_size)+' episode')
        minibatch = self.get_batch()
        for state, action, reward, next_state, terminated in minibatch:

            target = self.model.predict(state)

            if terminated:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)

            self.model.fit(state, target, epochs=self.epochs, verbose=self.verbose)

        
    def get_batch(self):
        return random.sample(self.memoryBuffer, self.batch_size)
              

    def act(self, observation):
        if np.random.rand() <= self.epsilon:
            print(self.env.action_space.sample())
            return self.env.action_space.sample()
        self.Q_valeur = self.model.predict(observation)
        print(self.Q_valeur )
        print(np.argmax(self.Q_valeur[0]))
        print(np.argmax(self.Q_valeur, axis=1)[0])
        return np.argmax(self.Q_valeur[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    #logger.set_level(logger.DEBUG)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    MODEL_PATH = "model_cartpole.h5"
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = DQNAgent(env)
    agent.model.load_weights(MODEL_PATH)
    episode_count = 200
    agent.step_epsilon = 4 / (3 * episode_count)
    done = False
    reward_tab = []
    for i in range(episode_count):
        #if i >= episode_count/4 and agent.epsilon >= agent.final_epsilon:
        #    agent.epsilon = agent.epsilon - agent.step_epsilon
        state = env.reset()
        reward = 0
        sumReward = 0
        print("Episode        " + str(i))
        print("Epsilon        " + str(agent.epsilon))
        print(len(agent.memoryBuffer))
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            # actionobservation
            interaction = (state, action, reward, next_state, done)
            agent.memoryBuffer.append(interaction)
            state = next_state

            # Recuperer la taille de l'action
            sumReward += reward
            if done:
                reward_tab.append(sumReward)
                agent.target_model.set_weights(agent.model.get_weights())
                break

        if len(agent.memoryBuffer) > agent.batch_size :
            pass
            #agent.train()

    #agent.model.save(MODEL_PATH)
    # Close the env and write monitor result info to disk
    print("before env.close")
    env.close()

    x = range(1, episode_count+1)
    plt.scatter(x, reward_tab)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()
