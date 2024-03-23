import numpy as np
import pygame
import time

import gym
from gym import spaces
from gym.utils import seeding

class BouncingBallEnv(gym.Env):
    """
    Custom Gym environment for a bouncing ball.

    The goal of this environment is to control the movement of a ball within a square grid.
    The ball can move in any direction and bounces off the walls of the grid.
    The agent receives a reward when the ball hits a wall and stops when the ball comes to a rest.

    Parameters:
    - render_mode (str): The rendering mode for the environment. Can be "human" or "rgb_array".
    - size (int): The size of the square grid.
    - ball_diameter_ratio (float): The ratio of the ball's diameter to the size of the grid.
    - apply_action (bool): Whether to apply the action to the ball's velocity.
    - log (bool): Whether to log additional information during the environment's execution.

    Attributes:
    - metadata (dict): Metadata about the environment, including available render modes and render FPS.
    - observation_space (gym.Space): The observation space of the environment.
    - action_space (gym.Space): The action space of the environment.
    - state (np.ndarray): The current state of the environment, including the ball's position and velocity.

    Methods:
    - reset(): Resets the environment to its initial state and returns the initial observation.
    - step(action): Takes a step in the environment based on the given action and returns the new observation, reward, done flag, and additional info.
    - render(): Renders the current state of the environment.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, size=2, velocity_scale = 1.0, ball_diameter_ratio = 0.01, wall_thickness_ratio = 0.01, apply_action=True, log=False):
        """
        Initialize the BouncingBallEnv.

        Args:
        - render_mode (str): The rendering mode for the environment. Can be "human" or "rgb_array".
        - size (int): The size of the square grid.
        - ball_diameter_ratio (float): The ratio of the ball's diameter to the size of the grid.
        - apply_action (bool): Whether to apply the action to the ball's velocity.
        - log (bool): Whether to log additional information during the environment's execution.
        """
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.seed()
        
        # Define observation space (ball's position and velocity in 2D)
        self.observation_space = spaces.Box(low=0, high=size, shape=(2,), dtype=np.float32)
        # Define action space (2D continuous: move in any direction)
        self.action_space = spaces.Box(low=-velocity_scale, high=velocity_scale, shape=(2,), dtype=np.float32)
        # Initialize state (position of the ball)
        self.state = None
        
        self.energy_loss_factor = 0.9  # Control how much energy is lost on collision

        if apply_action:
            self.velocity_change_factor = 0.1 # Control how much the action affects the velocity
        else:
            self.velocity_change_factor = 0 
        
        self.min_velocity = 0.01  # Threshold for considering the velocity to be effectively zero
        self.velocity_initial_size = velocity_scale
        # Example of adjusting units to the world size
        self.wall_thickness_ratio = wall_thickness_ratio  # Wall thickness as a proportion of world size
        self.ball_diameter_ratio = ball_diameter_ratio  # Ball diameter as a proportion of world size
        
        self.safe_margin = self.size * self.wall_thickness_ratio  + self.size * ( self.ball_diameter_ratio / 2 ) # The size of the wall + the size of the (radius) if the ball 
            
        # Adjust for ball's radius in collision detection
        self.left_bound = 0 + self.safe_margin 
        self.right_bound = self.size  - self.safe_margin
        self.bottom_bound = 0 + self.safe_margin
        self.top_bound = self.size  - self.safe_margin
        
        
        self.log = log
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        """
        Get the current observation of the environment.

        Returns:
        - obs (np.ndarray): The current observation of the environment as an RGB array.
        """
        # Temporarily set render_mode to 'rgb_array' for this operation, if necessary
        original_render_mode = self.render_mode
        self.render_mode = 'rgb_array'
        
        # Generate the observation as an RGB array
        obs = self._render_frame()

        # Reset the render mode to its original state, if needed
        self.render_mode = original_render_mode
        
        return obs
    
    def _get_info(self):
        """
        Get additional information about the environment.

        Returns:
        - info (dict): Additional information about the environment.
        """
        return {
            
        }
    
    def get_keys_to_action(self):
        """
        Get the mapping of keys to actions.

        Returns:
        - keys_to_action (dict): The mapping of keys to actions.
        """
        return self.keys_to_action
    
    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.

        Returns:
        - observation (np.ndarray): The initial observation of the environment.
        """
        

        # Randomly initialize the ball's position within the safe bounds
        initial_x = self.np_random.uniform(low=self.left_bound, high=self.right_bound)
        initial_y = self.np_random.uniform(low=self.bottom_bound, high=self.top_bound)
        
        # Randomly initialize the ball's velocity
        
        initial_velocity_x = self.np_random.uniform(low=-self.velocity_initial_size, high=self.velocity_initial_size)
        initial_velocity_y = self.np_random.uniform(low=-self.velocity_initial_size, high=self.velocity_initial_size)

        # Set the initial state
        self.state = np.array([initial_x, initial_y, initial_velocity_x, initial_velocity_y], dtype=np.float32)
        
        if self.render_mode == "human":
            self._render_frame()
        
        observation = self._get_obs()
        return observation


    def step(self, action):
        """
        Take a step in the environment based on the given action.
        

        Args:
        - action (np.ndarray): The action to take in the environment.

        Returns:
        - observation (np.ndarray): The new observation of the environment.
        - reward (float): The reward for the current step.
        - done (bool): Whether the episode is done or not.
        - info (dict): Additional information about the environment.
        """
        # Normalize the action to ensure it's a unit vector
        action_direction = action / np.linalg.norm(action)
        
        # Adjust the ball's velocity based on the action
        self.state[2] += action_direction[0] * self.velocity_change_factor
        self.state[3] += action_direction[1] * self.velocity_change_factor
        
        
        # Apply old velocity to calculate the new position
        # Here we assume each step has unit time duration. So s = v * t.
        self.state[:2] += self.state[2:]
        next_position = self.state[:2] # The idea next position (unclipped), used for collision detection
        print(f"next_position: {next_position}")
        # Ensure the ball's position is within the environment bounds
        self.state[:2] = np.clip(self.state[:2], self.left_bound, self.top_bound)
        
        next_velocity = self.state[2:]

        # Initialize reward and done flag
        reward = 0
        done = False

        
        # Check collisions with adjustments for ball radius
        collision = False
        if next_position[0] <= self.left_bound or next_position[0] >= self.right_bound:
            next_velocity[0] = -next_velocity[0] * self.energy_loss_factor  # Reverse X velocity
            collision = True
        if next_position[1] <= self.bottom_bound or next_position[1] >= self.top_bound:
            next_velocity[1] = -next_velocity[1] * self.energy_loss_factor  # Reverse Y velocity
            collision = True

        if collision:
            reward = 10  # Reward for hitting a wall

        # Update the velocity
        self.state[2] = next_velocity[0]
        self.state[3] = next_velocity[1]
        
        
        # Check if the ball's velocity is effectively zero
        if np.linalg.norm(self.state[2:]) < self.min_velocity:
            done = True  # End the episode if the ball has stopped moving
        
        # Update the observation with the current state
        observation = self.state

        info = {
            "message": "Ball has stopped" if done else "In motion",
            "Energy": {np.linalg.norm(self.state[2:])}
            }

        if self.render_mode == "human":
            self._render_frame()
            
        if self.log:
            if collision:
                print(f"Collision detected! =====> reward: {reward}")
            print(f"bottom bound: {self.bottom_bound}, left bound: {self.left_bound}, top bound: {self.top_bound}, right bound: {self.right_bound}")
            print(f"Velocity: {self.state[2:]}, Position: {self.state[:2]}, Energy: {np.linalg.norm(self.state[2:])}")
        return observation, reward, done, info

    
    def render(self, mode="human"):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background

    
        # Convert these ratios to pixel dimensions for rendering
        ball_diameter_pixels = self.window_size * self.ball_diameter_ratio
        wall_thickness_pixels = self.window_size * self.wall_thickness_ratio

        # Draw walls as thick lines around the perimeter using the pixel dimensions
        wall_color = (0, 0, 0)  # Black for walls
        pygame.draw.line(canvas, wall_color, (0, 0), (self.window_size, 0), int(wall_thickness_pixels))  # Top wall
        pygame.draw.line(canvas, wall_color, (0, self.window_size), (self.window_size, self.window_size), int(wall_thickness_pixels))  # Bottom wall
        pygame.draw.line(canvas, wall_color, (0, 0), (0, self.window_size), int(wall_thickness_pixels))  # Left wall
        pygame.draw.line(canvas, wall_color, (self.window_size, 0), (self.window_size, self.window_size), int(wall_thickness_pixels))  # Right wall

        # Draw the agent (ball)
        agent_location = self.state[:2]
        agent_color = (0, 0, 255)  # Blue for the agent
        # Convert agent location to pixel coordinates, adjusting for ball radius to keep it within bounds
        agent_center_x = (agent_location[0] * self.window_size / self.size)
        agent_center_y = (agent_location[1] * self.window_size / self.size)
        agent_center = (int(agent_center_x), int(agent_center_y))
        pygame.draw.circle(canvas, agent_color, agent_center, int(ball_diameter_pixels // 2))
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            returned_img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            return returned_img
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    