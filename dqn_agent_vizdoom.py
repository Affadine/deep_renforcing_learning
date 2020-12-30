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
        self.alpha = 0.1
        self.gamma = 0.6
        self.init_epsilon = 1
        self.final_epsilon = 0.1
        self.step_epsilon = 0
        self.epsilon = self.init_epsilon
        self.learning_rate = 0.0001
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = 40
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
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=self.input_shape,
                                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        # 2ème couche de convolution
        model.add(
            layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        # 3ème couche de convolution
        model.add(layers.Conv2D(32, (2, 2), activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        layers.Dropout(0.25)
        # couche dense de 64 unités
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        layers.Dropout(0.5)

        # Couche de sortie avec les Qvaleurs  à predire pour chaque action
        model.add(layers.Dense(self.Q_size, activation='softmax'))

        #optimiser = tf.optimizers.Adam(learning_rate=self.learning_rate)
        # compile model
        model.compile(optimizer=self.optimizer,
                      loss='mse',
                      metrics=["accuracy"])
        model.summary()
        return model

    def train(self):
        print('retain after each ' + str(self.batch_size) + ' episode')
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
    MODEL_PATH = "model.h5"
    env = gym.make('VizdoomBasic-v0', depth=True, labels=True, position=True, health=True)
    #env.seed(0)
    agent = DQNAgent(env)
    episode_count = 1000
    agent.step_epsilon = 1/episode_count
    reward = 0
    done = False
    sumReward = 0
    rewardFollowingTab = {}
    for i in range(episode_count):
        #if i >= episode_count/2 and agent.epsilon >= agent.final_epsilon:
        agent.epsilon = agent.init_epsilon - i * agent.step_epsilon
        observation = agent.preprocess(env.reset()[0])
        rewardFollowing = []
        sumReward = 0
        print("Episode        " + str(i))
        while True:
            action = agent.act(observation)

            next_observation, reward, done, info = env.step(action)
            #state = preprocess(state[0], [112, 64])
            next_observation = agent.preprocess(next_observation[0])
            # actionobservation
            interaction = (observation, action, reward, next_observation, done)
            agent.memoryBuffer.append(interaction)

            observation = next_observation

            # Recuperer la taille de l'action
            sumReward += reward
            rewardFollowing.append(sumReward)
            #print("size memory : " + str(len(agent.memoryBuffer)))
            '''if len(agent.memoryBuffer) >= agent.maxSize:
                agent.memoryBuffer.pop()
            '''
            if done:
                agent.target_model.set_weights(agent.model.get_weights())
                rewardFollowingTab[i] = rewardFollowing
                break


            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
            env.render()
        if i > 0 and i % 100 == 0:
            agent.train()
    agent.model.save(MODEL_PATH)

    # Close the env and write monitor result info to disk
    print("before env.close")
    env.close()

    '''for key in rewardFollowingTab:
        rewardFollowing = rewardFollowingTab[key]
        xval = []
        yval = []
        index = 1
        for nextReward in rewardFollowing:
            xval.append(index)
            yval.append(nextReward)
            index = index + 1
        print("Episode : " + str(key))
        print(rewardFollowing)
    plt.scatter(xval, yval)
    plt.ylabel('Learning')
    plt.show()'''
