import custom_envs
import gym
import numpy as np

from gym.utils.env_checker import check_env
from gym.utils.play import play, PlayPlot


env = gym.make('custom_envs/GridWorld-v0', size=7, render_mode="human")

observation = env.reset()

for _ in range(1000):
# while True:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, done, info = env.step(action)
    

    if done:
        observation = env.reset()
        print(f"===============> Reward: {reward}")
        break
        
env.close()
