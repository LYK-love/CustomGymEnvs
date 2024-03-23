import custom_envs
import gym
import numpy as np
from gym.wrappers import FlattenObservation

from gym.utils.env_checker import check_env
from gym.utils.play import play, PlayPlot
# from gym.envs.box2d import CarRacing

env = gym.make('custom_envs/BouncingBall-v0', render_mode="human", size=2,  velocity_scale = 1.0, ball_diameter_ratio = 0.1, wall_thickness_ratio = 0.1, apply_action=False, log=True)


# env = gym.make('custom_envs/BouncingBall-v0', size=2, ball_diameter_ratio=0.02, render_mode="rgb_array", apply_action=False, log=True)

# play(env, keys_to_action = {
#             # Direction vectors represented as [x, y]
#             "w": [0, -1],  # Up
#             "a": [-1, 0],  # Left
#             "s": [0, 1],   # Down
#             "d": [1, 0],   # Right
#         })

# env = gym.make('gym_examples/GridWorld-v0', render_mode="rgb_array")

# play(env, callback=plotter.callback)
# print(env.get_keys_to_action)
        
# play(gym.make("CartPole-v1"), callback=plotter.callback)

# play(gym.make("CarRacing-v2", render_mode="rgb_array"), keys_to_action={
#     "w": np.array([0, 0.7, 0]),
#     "a": np.array([-1, 0, 0]),
#     "s": np.array([0, 0, 1]),
#     "d": np.array([1, 0, 0]),
#     "wa": np.array([-1, 0.7, 0]),
#     "dw": np.array([1, 0.7, 0]),
#     "ds": np.array([1, 0, 1]),
#     "as": np.array([-1, 0, 1]),
#     }, noop=np.array([0,0,0]))

# check_env(env)

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
