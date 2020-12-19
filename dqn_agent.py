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



class DQNAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env):
        self.env = env
        self.state_size = 4 #self.env.observation_space
        self.Q_size = self.env.action_space.n
        self.Q_valeur = {}
        self.memoryBuffer = []
        self.maxSize = 100 * 1000
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.learning_rate =0.0001
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = 1000
        self.epochs = 14
        self.model = self.creer_model()

    def creer_model(self):
        model = keras.Sequential()
        model.add(layers.Embedding(self.state_size, self.Q_size, input_length=1))
        model.add(layers.Reshape((self.Q_size,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.Q_size, activation='linear'))
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    '''def train_model(self,data):
        X, Y = data
        history = self.model.fit(X, Y,
                            epochs = self.epochs,
                            batch_size = self.batch_size)
    '''
    def retrain(self):
        minibatch = self.get_batch(self.batch_size)

        for state, action, reward, next_state, terminated in minibatch:

            target = self.predict(state)

            if terminated:
                target[0][action] = reward
            else:
                t = self.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)

            self.model.fit(state, target, epochs=1, verbose=0)

    def predict(self,state):
        return self.model.predict(state)


        
    def get_batch(self):
        return np.random.sample(self.memoryBuffer, self.batch_size)
              

    def act(self, observation):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()

        self.Q_valeur = self.predict(observation)
        return np.argmax(self.Q_valeur)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.DEBUG)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = DQNAgent(env)

    episode_count = 100
    reward = 0
    done = False
    sumReward = 0
    lastObservation = []
    rewardFollowingTab = {}
    memoryBuffer = []

    for i in range(episode_count):
        observation = env.reset()
        rewardFollowing = []
        sumReward = 0
        while True:
            action = agent.act(observation)
            next_observation, reward, done, info = env.step(action)

            # actionobservation
            interaction = (observation, action, next_observation, reward, done)
            memoryBuffer.append(interaction)

            observation = next_observation

            #Recuperer la taille de l'action
            sumReward+=reward
            rewardFollowing.append(sumReward)
            if(reward!=1):
                print("reward = " + str(reward))


            print(" memory size = " + str(sys.getsizeof(memoryBuffer)))
            '''while(sys.getsizeof(memoryBuffer) > maxSize):
                print("remove last item")
                memoryBuffer.pop()
                print("after remove : memory size = " + str(sys.getsizeof(memoryBuffer))) '''
            if done:
                rewardFollowingTab[i] = rewardFollowing
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    print("before env.close")
    env.close()


    for key in rewardFollowingTab:
        rewardFollowing = rewardFollowingTab[key]
        xval=[]
        yval=[]
        index=1
        for nextReward in rewardFollowing :
            xval.append(index)
            yval.append(nextReward)
            index=index+1
        print("Episode : " + str(key))
        print(rewardFollowing)
    plt.scatter(xval, yval)
    plt.ylabel('Learning')
    plt.show()
