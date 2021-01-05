import argparse
import sys
from skimage.color import rgb2gray

import gym
import vizdoomgym
import matplotlib.pyplot as plt
import numpy as np

import skimage
from skimage.transform import resize
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

from collections import deque

class DQNAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env):
        self.env = env
        self.Q_size = self.env.action_space.n
        self.maxSize = 100000
        self.memoryBuffer = deque(maxlen=self.maxSize)
        self.gamma = 0.1
        self.init_epsilon = 1
        self.final_epsilon = 0.05
        self.step_epsilon = 0
        self.epsilon = 0.1
        self.learning_rate = 0.01
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = 200
        self.epochs = 1
        self.verbose = 0
        self.input_shape = (112, 64, 1)
        self.model = self.creer_model()
        self.target_model = self.creer_model()
        self.target_model.set_weights(self.model.get_weights())

    def preprocess(self, img, resolution=(112, 64)):
        img = resize(img, resolution)
        # passage en noir et blanc
        img = rgb2gray(img)
        img = img.reshape(1, 112, 64, 1)
        return img

    def creer_model(self):
        model = keras.Sequential()
        # 1ere couche de convolution
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=self.input_shape))
        #model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((4, 4), padding='same'))

        # 2ème couche de convolution
        model.add(
            layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((4, 4), padding='same'))

        # 3ème couche de convolution
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))


        # couche dense de 800 unités
        model.add(layers.Flatten())
        model.add(layers.Dense(8, activation='relu'))

        # Couche de sortie avec les Qvaleurs  à predire pour chaque action
        model.add(layers.Dense(self.Q_size, activation='linear'))

        # compile model
        model.compile(optimizer=self.optimizer, loss='mse')
        model.summary()
        return model

    def train(self):
        print('retain modele')
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
            return self.env.action_space.sample()

        self.Q_valeur = self.model.predict(observation)
        print(self.Q_valeur[0])
        print(np.argmax(self.Q_valeur[0]))
        return np.argmax(self.Q_valeur[0])


if __name__ == '__main__':
    '''parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()'''

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    #logger.set_level(logger.DEBUG)

    #env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    MODEL_PATH = "model_vizdoomCorridor.h5"
    #VizdoomBasic-v0
    #VizdoomCorridor-v0
    env = gym.make('VizdoomCorridor-v0', depth=True, labels=True, position=True, health=True)
    agent = DQNAgent(env)
    agent.model.load_weights(MODEL_PATH)
    episode_count = 200
    agent.step_epsilon = 4/(3*episode_count)
    reward = 0
    done = False
    reward_tab = []
    for i in range(episode_count):
        state = agent.preprocess(env.reset()[0])
        reward = 0
        sumReward = 0
        print("Episode        " + str(i))
        print("Epsilon        " + str(agent.epsilon))
        print(len(agent.memoryBuffer))
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = agent.preprocess(next_state[0])

            interaction = (state, action, reward, next_state, done)
            agent.memoryBuffer.append(interaction)
            state = next_state

            # Recuperer la taille de l'action
            sumReward += reward
            if done:
                reward_tab.append(sumReward)
                agent.target_model.set_weights(agent.model.get_weights())
                break

            env.render()

        if len(agent.memoryBuffer) > 0:
            #agent.train()
            pass

    agent.model.save(MODEL_PATH)

    # Close the env and write monitor result info to disk
    print("before env.close")
    env.close()

    x = range(1, episode_count+1)
    plt.scatter(x, reward_tab)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()
