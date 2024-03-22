from gym.envs.registration import register

register(
     id="custom_envs/GridWorld-v0",
     entry_point="custom_envs.envs:GridWorldEnv",
     max_episode_steps=300,
)

register(
     id="custom_envs/BouncingBall-v0",
     entry_point="custom_envs.envs:BouncingBallEnv",
     max_episode_steps=300,
)