import argparse
import sys

import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt


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

    episode_count = 200
    reward = 0
    done = False
    sumReward = 0
    reward_tab = []

    for i in range(episode_count):
        #print("i= " + str(i))
        ob = env.reset()
        sumReward = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            sumReward+=reward

            if done:
                reward_tab.append(sumReward)
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    print("before env.close")
    env.close()

    x = range(1, episode_count+1)
    plt.scatter(x, reward_tab)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()
