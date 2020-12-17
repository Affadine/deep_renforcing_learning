import argparse
import sys

import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt


#from __future__ import absolute_import, division, print_function

import base64
#import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
#import pyvirtualdisplay

import tensorflow as tf

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
maxSize = 100 * 1000

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

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
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False
    sumReward = 0
    lastObservation = []
    rewardFollowingTab = {}
    memeryBuffer = []

    for i in range(episode_count):
        #print("i= " + str(i))
        observation = env.reset()
        rewardFollowing = []
        sumReward = 0
        while True:
            action = agent.act(observation, reward, done)
            lastObservation = observation
            observation, reward, done, info = env.step(action)
            sumReward+=reward
            rewardFollowing.append(sumReward)
            if(reward!=1):
                print("reward = " + str(reward))
            #if(sumReward % 100 == 0):
            #    print("sumReward = " + str(sumReward))
            #  interaction=(état,action,étatsuivant,récompense,finépisode)
            # actionobservation
            interaction=(lastObservation,action,observation,reward,done)
            memeryBuffer.insert(0,interaction)
            #print("interaction = " + str(interaction) + " memory size = " + str(sys.getsizeof(memeryBuffer)))
            print(" memory size = " + str(sys.getsizeof(memeryBuffer)))
            while(sys.getsizeof(memeryBuffer) > maxSize):
                print("remove last item")
                memeryBuffer.pop()
                print("after remove : memory size = " + str(sys.getsizeof(memeryBuffer)))
            if done:
                rewardFollowingTab[i] = rewardFollowing
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    print("before env.close")
    env.close()

    #plt.plot([1, 2, 3, 4])
    for key in rewardFollowingTab:
        #print(rewardFollowingTab[key])
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
